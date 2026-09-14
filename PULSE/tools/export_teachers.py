"""Prepare frozen ROI feature teachers; third-party weights are not redistributed."""
import argparse
from pathlib import Path
import tempfile

import torch
from torch import nn


class GrayOCR(nn.Module):
    """Accept the same BGR [-1,1] contract as three-channel OCR teachers."""
    def __init__(self, cnn):
        super().__init__()
        self.cnn = cnn
        self.register_buffer("gray_weights", torch.tensor([.114, .587, .299]).view(1, 3, 1, 1))

    def forward(self, image):
        return self.cnn((image * self.gray_weights).sum(dim=1, keepdim=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["face", "ocr"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--onnx", help="OCR: downloaded CRNN ONNX model with BGR [-1,1] input")
    parser.add_argument("--feature-node", help="optional 4-D CNN output tensor name")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.kind == "face":
        from facenet_pytorch import InceptionResnetV1
        model = InceptionResnetV1(pretrained="vggface2").eval()
        example = torch.zeros(1, 3, 160, 160)
    else:
        if not args.onnx:
            parser.error("OCR export requires --onnx")
        import onnx
        from onnx2torch import convert
        graph = onnx.shape_inference.infer_shapes(onnx.load(args.onnx))
        channels = graph.graph.input[0].type.tensor_type.shape.dim[1].dim_value
        if channels not in (1, 3):
            parser.error("OCR model must have NCHW input with one or three channels")
        shapes = {v.name: v.type.tensor_type.shape.dim
                  for v in [*graph.graph.value_info, *graph.graph.output, *graph.graph.input]}
        feature = args.feature_node
        if feature is None:
            candidates = []
            for node in graph.graph.node:
                if node.op_type in {"LSTM", "GRU"}:
                    break
                for output in node.output:
                    shape = shapes.get(output)
                    if shape is not None and len(shape) == 4 and shape[2].dim_value == 1:
                        candidates.append(output)
            if not candidates:
                parser.error("cannot identify a [N,C,1,T] CNN feature; provide --feature-node")
            feature = candidates[-1]
        with tempfile.TemporaryDirectory() as temp:
            source, extracted = Path(temp) / "source.onnx", Path(temp) / "cnn.onnx"
            onnx.save(graph, source)
            onnx.utils.extract_model(str(source), str(extracted),
                                     [graph.graph.input[0].name], [feature])
            model = convert(str(extracted)).eval()
        if channels == 1:
            model = GrayOCR(model).eval()
        example = torch.zeros(1, 3, 32, 100)
        print(f"OCR CNN feature: {feature}")
    model.requires_grad_(False)
    with torch.no_grad():
        output = model(example)
    if args.kind == "face" and output.ndim != 2:
        raise ValueError("face teacher must produce [N, embedding_dim]")
    if args.kind == "ocr" and (output.ndim != 4 or output.shape[2] != 1):
        raise ValueError("OCR teacher must produce [N,C,1,T]")
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.trace(model, example, strict=False).save(str(destination))
    # Verify feature alignment remains differentiable with respect to pixels.
    traced = torch.jit.load(str(destination)).eval()
    probe = example.clone().requires_grad_(True)
    traced(probe).square().mean().backward()
    if probe.grad is None or not torch.isfinite(probe.grad).all():
        raise RuntimeError("exported teacher does not support input gradients")
    print(destination)


if __name__ == "__main__":
    main()
