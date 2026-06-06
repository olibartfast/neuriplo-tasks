import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import onnx
import torch
import torch as _torch


def _onnx_safe_isin(elements, test_elements):
    """ONNX-exportable drop-in for torch.isin via broadcast equality check."""
    test = test_elements.view(-1)
    return (elements.unsqueeze(-1) == test).any(-1)


def _onnx_safe_cummax(x, dim):
    """ONNX-exportable cummax: sequential pairwise maximums unrolled at trace time."""
    slices = torch.unbind(x, dim=dim)
    out = [slices[0]]
    for s in slices[1:]:
        out.append(torch.maximum(out[-1], s))
    values = torch.stack(out, dim=dim)
    return values, torch.zeros_like(values, dtype=torch.long)


def _onnx_safe_cummin(x, dim):
    """ONNX-exportable cummin: sequential pairwise minimums unrolled at trace time."""
    slices = torch.unbind(x, dim=dim)
    out = [slices[0]]
    for s in slices[1:]:
        out.append(torch.minimum(out[-1], s))
    values = torch.stack(out, dim=dim)
    return values, torch.zeros_like(values, dtype=torch.long)


class GroundingDinoExportWrapper(torch.nn.Module):
    """Wraps GroundingDinoForObjectDetection to expose 4 inputs.

    position_ids are derived from input_ids at runtime so callers only need
    to supply pixel_values, input_ids, attention_mask, and token_type_ids —
    the same set neuriplo-tasks's OpenVocabDetectionTask populates.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, input_ids, attention_mask, token_type_ids):
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.int64, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        return outputs.logits, outputs.pred_boxes


def export_groundingdino_to_onnx(
    model_name="IDEA-Research/grounding-dino-base",
    output_path="groundingdino.onnx",
    image_height=800,
    image_width=1333,
    max_text_length=256,
    opset_version=17,
    dynamic_axes_enabled=True,
):
    try:
        import transformers
        from transformers import AutoProcessor, GroundingDinoForObjectDetection
    except ImportError as exc:
        print(f"Transformers not available: {exc}")
        return None

    print(f"Loading Grounding DINO model: {model_name}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")

    model = GroundingDinoForObjectDetection.from_pretrained(model_name)
    model.eval()

    print("Model config:")
    print(f"   - Image size: {image_height}x{image_width}")
    print(f"   - Max text length: {max_text_length}")

    wrapped = GroundingDinoExportWrapper(model)
    wrapped.eval()

    pixel_values = torch.randn(1, 3, image_height, image_width, dtype=torch.float32)
    input_ids = torch.zeros((1, max_text_length), dtype=torch.int64)
    attention_mask = torch.ones((1, max_text_length), dtype=torch.int64)
    token_type_ids = torch.zeros((1, max_text_length), dtype=torch.int64)

    with torch.no_grad():
        pred_logits, pred_boxes = wrapped(pixel_values, input_ids, attention_mask, token_type_ids)
        print("Forward pass successful:")
        print(f"   - logits:     {tuple(pred_logits.shape)}")
        print(f"   - pred_boxes: {tuple(pred_boxes.shape)}")

    dynamic_axes = None
    if dynamic_axes_enabled:
        dynamic_axes = {
            "pixel_values": {0: "batch_size"},
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "token_type_ids": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "pred_boxes": {0: "batch_size"},
        }

    # Grounding DINO uses torch.isin (not in ONNX opsets ≤17) and aten.cummax
    # (not supported in the dynamo exporter). Patch torch.isin for the duration
    # of the export trace and use the legacy TorchScript-based tracer.
    original_isin = torch.isin
    original_cummax = torch.cummax
    original_cummin = torch.cummin
    torch.isin = _onnx_safe_isin
    torch.cummax = _onnx_safe_cummax
    torch.cummin = _onnx_safe_cummin
    try:
      with torch.no_grad():
        torch.onnx.export(
            wrapped,
            (pixel_values, input_ids, attention_mask, token_type_ids),
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["pixel_values", "input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits", "pred_boxes"],
            dynamic_axes=dynamic_axes,
            verbose=False,
            dynamo=False,
        )
    finally:
        torch.isin = original_isin
        torch.cummax = original_cummax
        torch.cummin = original_cummin

    print(f"Model exported to: {output_path}")
    onnx_model = onnx.load(output_path)
    onnx_model = fix_eyelike_bool(onnx_model)
    onnx.save(onnx_model, output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification passed")

    for output in onnx_model.graph.output:
        dims = output.type.tensor_type.shape.dim
        shape = [dim.dim_param or dim.dim_value for dim in dims]
        print(f"   - {output.name}: {shape}")

    return output_path


def test_onnx_model(onnx_path, image_height=800, image_width=1333, max_text_length=256):
    try:
        import onnxruntime as ort
    except ImportError:
        print("ONNXRuntime not installed. Skipping ONNX test.")
        return False

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inputs = {
        "pixel_values": np.random.randn(1, 3, image_height, image_width).astype(np.float32),
        "input_ids": np.zeros((1, max_text_length), dtype=np.int64),
        "attention_mask": np.ones((1, max_text_length), dtype=np.int64),
        "token_type_ids": np.zeros((1, max_text_length), dtype=np.int64),
    }
    outputs = session.run(None, inputs)
    print("ONNX inference successful")
    for output_meta, value in zip(session.get_outputs(), outputs):
        print(f"   - {output_meta.name}: {value.shape}")
    return True


def fix_eyelike_bool(model):
    """Replace bool EyeLike (unsupported in ORT ≤1.25) with Cast→EyeLike(float)→Cast.

    New nodes are inserted at the position of the original EyeLike node so the
    graph remains topologically sorted after the edit.
    """
    from onnx import TensorProto

    cs_dtype = {}
    for node in model.graph.node:
        if node.op_type == "ConstantOfShape":
            for attr in node.attribute:
                if attr.name == "value":
                    for out in node.output:
                        cs_dtype[out] = attr.t.data_type

    nodes = list(model.graph.node)
    del model.graph.node[:]

    for node in nodes:
        if node.op_type == "EyeLike" and cs_dtype.get(node.input[0]) == TensorProto.BOOL:
            in_name, out_name = node.input[0], node.output[0]
            float_in = in_name + "_f32_for_eye"
            float_out = out_name + "_f32_eye"
            model.graph.node.append(
                onnx.helper.make_node("Cast", inputs=[in_name], outputs=[float_in], to=TensorProto.FLOAT)
            )
            model.graph.node.append(
                onnx.helper.make_node("EyeLike", inputs=[float_in], outputs=[float_out])
            )
            model.graph.node.append(
                onnx.helper.make_node("Cast", inputs=[float_out], outputs=[out_name], to=TensorProto.BOOL)
            )
        else:
            model.graph.node.append(node)

    return model


def main():
    parser = argparse.ArgumentParser(description="Export Grounding DINO to ONNX")
    parser.add_argument("--model", default="IDEA-Research/grounding-dino-base", help="Hugging Face model id")
    parser.add_argument("--output", default="groundingdino.onnx", help="Output ONNX file path")
    parser.add_argument("--image-height", type=int, default=800, help="Input image height")
    parser.add_argument("--image-width", type=int, default=1333, help="Input image width")
    parser.add_argument("--max-text-length", type=int, default=256, help="Max tokenized text sequence length")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--static-batch", action="store_true", help="Disable dynamic batch axis")
    parser.add_argument("--test", action="store_true", help="Run ONNX Runtime smoke test after export")
    args = parser.parse_args()

    try:
        exported = export_groundingdino_to_onnx(
            model_name=args.model,
            output_path=args.output,
            image_height=args.image_height,
            image_width=args.image_width,
            max_text_length=args.max_text_length,
            opset_version=args.opset,
            dynamic_axes_enabled=not args.static_batch,
        )
        if exported is None:
            sys.exit(1)

        if args.test:
            if not test_onnx_model(
                exported,
                image_height=args.image_height,
                image_width=args.image_width,
                max_text_length=args.max_text_length,
            ):
                sys.exit(1)
    except Exception as exc:
        print(f"Export failed: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
