import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import onnx
import torch


class OWLv2ExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, input_ids, attention_mask):
        outputs = self.model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits, outputs.objectness_logits, outputs.pred_boxes


def export_owlv2_to_onnx(
    model_name="google/owlv2-base-patch16-ensemble",
    output_path="owlv2.onnx",
    image_height=960,
    image_width=960,
    max_queries=16,
    sequence_length=16,
    opset_version=17,
    dynamic_batch=True,
):
    try:
        from transformers import AutoProcessor, Owlv2ForObjectDetection
    except ImportError as exc:
        print(f"Transformers not available: {exc}")
        return None

    print(f"Loading OWLv2 model: {model_name}")
    print(f"PyTorch version: {torch.__version__}")

    model = Owlv2ForObjectDetection.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()

    print("Model config:")
    print(f"   - Image size: {image_height}x{image_width}")
    print(f"   - Max queries: {max_queries}")
    print(f"   - Sequence length: {sequence_length}")
    print(f"   - Processor: {processor.__class__.__name__}")

    wrapped = OWLv2ExportWrapper(model)
    pixel_values = torch.randn(1, 3, image_height, image_width, dtype=torch.float32)
    input_ids = torch.zeros((max_queries, sequence_length), dtype=torch.int64)
    attention_mask = torch.ones((max_queries, sequence_length), dtype=torch.int64)

    with torch.no_grad():
        logits, objectness_logits, pred_boxes = wrapped(pixel_values, input_ids, attention_mask)
        print("Forward pass successful:")
        print(f"   - logits: {tuple(logits.shape)}")
        print(f"   - objectness_logits: {tuple(objectness_logits.shape)}")
        print(f"   - pred_boxes: {tuple(pred_boxes.shape)}")

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "pixel_values": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "objectness_logits": {0: "batch_size"},
            "pred_boxes": {0: "batch_size"},
        }

    torch.onnx.export(
        wrapped,
        (pixel_values, input_ids, attention_mask),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["pixel_values", "input_ids", "attention_mask"],
        output_names=["logits", "objectness_logits", "pred_boxes"],
        dynamic_axes=dynamic_axes,
        verbose=False,
    )

    print(f"Model exported to: {output_path}")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification passed")

    for output in onnx_model.graph.output:
        dims = output.type.tensor_type.shape.dim
        shape = [dim.dim_param or dim.dim_value for dim in dims]
        print(f"   - {output.name}: {shape}")

    return output_path


def test_onnx_model(onnx_path, image_height=960, image_width=960, max_queries=16, sequence_length=16):
    try:
        import onnxruntime as ort
    except ImportError:
        print("ONNXRuntime not installed. Skipping ONNX test.")
        return False

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inputs = {
        "pixel_values": np.random.randn(1, 3, image_height, image_width).astype(np.float32),
        "input_ids": np.zeros((max_queries, sequence_length), dtype=np.int64),
        "attention_mask": np.ones((max_queries, sequence_length), dtype=np.int64),
    }
    outputs = session.run(None, inputs)
    print("ONNX inference successful")
    for output_meta, value in zip(session.get_outputs(), outputs):
        print(f"   - {output_meta.name}: {value.shape}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Export OWLv2 to ONNX")
    parser.add_argument("--model", default="google/owlv2-base-patch16-ensemble", help="Hugging Face model id")
    parser.add_argument("--output", default="owlv2.onnx", help="Output ONNX file path")
    parser.add_argument("--image-height", type=int, default=960, help="Input image height")
    parser.add_argument("--image-width", type=int, default=960, help="Input image width")
    parser.add_argument("--max-queries", type=int, default=16, help="Maximum number of text prompts")
    parser.add_argument("--sequence-length", type=int, default=16, help="Sequence length per prompt")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--static-batch", action="store_true", help="Disable dynamic batch axis")
    parser.add_argument("--test", action="store_true", help="Run ONNX Runtime smoke test")
    args = parser.parse_args()

    try:
        exported = export_owlv2_to_onnx(
            model_name=args.model,
            output_path=args.output,
            image_height=args.image_height,
            image_width=args.image_width,
            max_queries=args.max_queries,
            sequence_length=args.sequence_length,
            opset_version=args.opset,
            dynamic_batch=not args.static_batch,
        )
        if exported is None:
            sys.exit(1)

        if args.test:
            if not test_onnx_model(
                exported,
                image_height=args.image_height,
                image_width=args.image_width,
                max_queries=args.max_queries,
                sequence_length=args.sequence_length,
            ):
                sys.exit(1)
    except Exception as exc:
        print(f"Export failed: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
