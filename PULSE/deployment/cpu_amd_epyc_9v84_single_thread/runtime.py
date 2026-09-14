"""Single-core OpenVINO BF16 sender and native AVX-512 BF16 receiver."""
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

from codec.images import pad_image
from deployment.common import load_bundle
from deployment.entropy import EntropyCodec
from .graphs import CompactSymbolEncode, extension_state_dict


class Codec:
    def __init__(self, model, *, height, width, qp=3, export_dir="outputs/amd_ir", stream_metadata=None):
        import openvino as ov
        from .entropy_cpp import pulse_cpu_amd_entropy_ext as entropy_ext
        self.net, artifact, fingerprint, config = load_bundle(model)
        self.height, self.width, self.qp = height, width, qp
        self.ph, self.pw = (height + 63) // 64 * 64, (width + 63) // 64 * 64
        self.entropy = EntropyCodec(
            entropy_ext, artifact, fingerprint, height=height, width=width, qp=qp,
            cutoff=2 if stream_metadata is None else stream_metadata["skip_index_cutoff"],
            cdf_mode=None if stream_metadata is None else stream_metadata["y_cdf_mode"])
        packages = {
            "s": ("cpp", "pulse_cpu_amd_pulse_s_decoder_ext"),
            "base": ("cpp", "pulse_cpu_amd_decoder_ext"),
            "l": ("cpp", "pulse_cpu_amd_pulse_l_decoder_ext"),
        }
        folder, module = packages[config["size"]]
        extension = import_module(f"deployment.cpu_amd_epyc_9v84_single_thread.{folder}.{module}")
        self.receiver = extension.DecodeWrapperProxy()
        self.receiver.set_param(extension_state_dict(self.net), self.ph // 16, self.pw // 16)
        self.z = torch.empty(self.entropy.z_shape, dtype=torch.int16)
        self.s0 = torch.empty(self.entropy.compact_shape, dtype=torch.int8)
        self.s1 = torch.empty_like(self.s0)
        self.core = ov.Core()
        # Artifact/model identity, shape, QP and graph format all key the IR.
        self.ir = Path(export_dir) / f"{fingerprint}-q{qp}-{self.ph}x{self.pw}-compact1" / "encoder.xml"
        self.request = None
        self.image = None

    @torch.inference_mode()
    def prepare_encoder(self):
        import openvino as ov
        if self.request is not None:
            return
        if not self.ir.is_file():
            module = CompactSymbolEncode(
                self.net, qp=self.qp, image_height=self.ph, image_width=self.pw).eval()
            example = torch.zeros(1, 3, self.ph, self.pw)
            graph = ov.convert_model(module, example_input=(example,))
            graph.reshape({graph.input(0): [1, 3, self.ph, self.pw]})
            for port, name in zip(graph.inputs, ["image"]):
                port.get_tensor().set_names({name})
            for port, name in zip(graph.outputs, ["z", "s0", "s1"]):
                port.get_tensor().set_names({name})
            self.ir.parent.mkdir(parents=True, exist_ok=True)
            ov.save_model(graph, self.ir, compress_to_fp16=False)
        self.compiled = self.core.compile_model(str(self.ir), "CPU", {
            "PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": 1, "INFERENCE_NUM_THREADS": 1,
            "INFERENCE_PRECISION_HINT": "bf16", "ENABLE_CPU_PINNING": True,
            "ENABLE_HYPER_THREADING": False, "EXECUTION_MODE_HINT": "PERFORMANCE",
        })
        self.request = self.compiled.create_infer_request()
        self.encoded = (np.empty(self.entropy.z_shape, dtype=np.int16),
                        np.empty(self.entropy.compact_shape, dtype=np.int8),
                        np.empty(self.entropy.compact_shape, dtype=np.int8))
        self.tensors = [ov.Tensor(a, shared_memory=True) for a in self.encoded]
        for name, tensor in zip(("z", "s0", "s1"), self.tensors):
            self.request.set_tensor(name, tensor)

    def set_image(self, image):
        self.entropy.require_encoding_defaults()
        import openvino as ov
        if tuple(image.shape) != (1, 3, self.height, self.width):
            raise ValueError("input shape does not match this runtime")
        if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("expected finite RGB pixels in [0,1]")
        self.prepare_encoder()
        self.image = pad_image(image.detach().cpu().float()).contiguous().numpy()
        self.input_tensor = ov.Tensor(self.image, shared_memory=True)
        self.request.set_tensor("image", self.input_tensor)

    def compress(self, image=None):
        if image is not None:
            self.set_image(image)
        if self.image is None:
            raise ValueError("set an input image before compression")
        self.request.infer()
        return self.entropy.compress(*self.encoded)

    @torch.inference_mode()
    def decompress(self, stream):
        self.entropy.decompress(stream, self.z.numpy(), self.s0.numpy(), self.s1.numpy())
        return self.receiver.forward_compact(self.z, self.s0, self.s1, self.qp)[
            ..., :self.height, :self.width]
