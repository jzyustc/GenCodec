"""Compiled BF16 H100 graphs, pinned compact transfers and multicore entropy."""
import os
from pathlib import Path
import torch

from codec.images import pad_image
from deployment.common import load_bundle
from deployment.entropy import EntropyCodec
from codec import bitstream
from .entropy import SplitEntropyCodec
from .graphs import ProductionEncode, ProductionDecode
from . import HOST_CPUS


class Codec:
    def __init__(self, model, *, height, width, qp=3, device="cuda:0", stream_metadata=None):
        from .cpp import pulse_gpu_entropy_ext as entropy_ext
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("H100 deployment requires a CUDA device")
        torch.cuda.set_device(self.device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        net, artifact, fingerprint, config = load_bundle(model)
        self.height, self.width, self.qp = height, width, qp
        self.ph, self.pw = (height + 63) // 64 * 64, (width + 63) // 64 * 64
        # OpenMP may pin its master during Torch initialization. Do not let
        # independent rANS workers inherit a one-core affinity from that master.
        os.sched_setaffinity(0, HOST_CPUS)
        before = {int(p.name) for p in Path("/proc/self/task").iterdir()}
        self.entropy = SplitEntropyCodec(
            entropy_ext, artifact, fingerprint, height=height, width=width, qp=qp,
            cutoff=2 if stream_metadata is None else stream_metadata["skip_index_cutoff"],
            cdf_mode=None if stream_metadata is None else stream_metadata["y_cdf_mode"])
        self._bundle = (entropy_ext, artifact, fingerprint, config)
        self._combined_reader = None
        workers = {int(p.name) for p in Path("/proc/self/task").iterdir()} - before
        for worker in workers:
            os.sched_setaffinity(worker, HOST_CPUS)
        os.sched_setaffinity(0, HOST_CPUS)
        self.host_cpus = sorted(HOST_CPUS)
        net = net.to(device=self.device, dtype=torch.bfloat16)
        meta_cdf, meta_lengths = artifact["cdf_tables"]["meta_prior"][qp]
        self.sender = torch.compile(
            ProductionEncode(net, qp=qp, height=self.ph, width=self.pw,
                             meta_cdf=meta_cdf, meta_lengths=meta_lengths).eval().to(self.device),
            dynamic=False, fullgraph=True, mode="reduce-overhead")
        self.receiver = torch.compile(
            ProductionDecode(net, qp=qp, height=self.ph, width=self.pw).eval().to(self.device),
            dynamic=False, fullgraph=True, mode="reduce-overhead")
        shapes = (self.entropy.z_shape, self.entropy.compact_shape, self.entropy.compact_shape)
        dtypes = (torch.int16, torch.int8, torch.int8)
        self.host_encode = tuple(torch.empty(s, dtype=d, pin_memory=True) for s, d in zip(shapes, dtypes))
        self.host_meta = torch.empty(self.entropy.meta_indexes.shape, dtype=torch.uint8, pin_memory=True)
        self.host_decode = tuple(torch.empty_like(a, pin_memory=True) for a in self.host_encode)
        self.gpu_decode = tuple(torch.empty(s, dtype=d, device=self.device) for s, d in zip(shapes, dtypes))
        self.transfer = torch.cuda.Stream(device=self.device)
        self.image = None

    def set_image(self, image):
        self.entropy.require_encoding_defaults()
        if tuple(image.shape) != (1, 3, self.height, self.width):
            raise ValueError("input shape does not match this runtime")
        if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("expected finite RGB pixels in [0,1]")
        self.image = pad_image(image).to(device=self.device, dtype=torch.bfloat16).contiguous()
        torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def compress(self, image=None):
        if image is not None:
            self.set_image(image)
        if self.image is None:
            raise ValueError("set an input image before compression")
        torch.compiler.cudagraph_mark_step_begin()
        outputs = self.sender(self.image)
        for host, gpu in zip((*self.host_encode, self.host_meta), outputs):
            host.copy_(gpu, non_blocking=True)
        torch.cuda.synchronize(self.device)
        return self.entropy.compress(*(a.numpy() for a in self.host_encode),
                                     meta_indexes=self.host_meta.numpy())

    @torch.inference_mode()
    def decompress(self, stream):
        def upload(part):
            with torch.cuda.stream(self.transfer):
                self.gpu_decode[part].copy_(self.host_decode[part], non_blocking=True)
        metadata, _ = bitstream.unpack(stream)
        if metadata.get("entropy_dense", False):
            reader = self.entropy
        elif "entropy_sections" not in metadata:
            if self._combined_reader is None:
                ext, artifact, fingerprint, config = self._bundle
                self._combined_reader = EntropyCodec(
                    ext, artifact, fingerprint, height=self.height, width=self.width, qp=self.qp,
                    cutoff=metadata["skip_index_cutoff"], lanes=8, threads=16,
                    cdf_mode=metadata["y_cdf_mode"])
            reader = self._combined_reader
        else:
            raise ValueError("use the standard decoder for the mobile retained-symbol profile")
        reader.decompress(stream, *(a.numpy() for a in self.host_decode), on_decoded=upload)
        torch.cuda.current_stream(self.device).wait_stream(self.transfer)
        torch.compiler.cudagraph_mark_step_begin()
        output = self.receiver(*self.gpu_decode)[..., :self.height, :self.width]
        torch.cuda.synchronize(self.device)
        return output
