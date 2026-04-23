"""
Export LGM / GRM Gaussian Splatting model to ONNX format.

The exported model maps one (or concatenated multi-view) image tensors of
shape [N, 3, H, W] to a Gaussian parameter tensor of shape [N, G, 14].

Each of the 14 Gaussian features, in order:
  0-2   xyz position
  3-5   log-scale (sx, sy, sz)
  6-9   rotation quaternion (w, x, y, z)
  10    opacity logit
  11-13 SH DC colour (r, g, b)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def export_lgm_to_onnx(
    model_name: str = "dylanebert/LGM",
    output_path: str = "lgm.onnx",
    image_size: int = 256,
    opset_version: int = 17,
    dynamic_batch: bool = True,
) -> str | None:
    """Export an LGM-style Gaussian Splatting model to ONNX."""

    print(f"Loading Gaussian Splatting model: {model_name}")
    print(f"Python  : {sys.version}")
    print(f"PyTorch : {torch.__version__}")

    try:
        import diffusers  # noqa: F401
        import transformers  # noqa: F401
        from diffusers import DiffusionPipeline
    except ImportError as exc:
        print(f"Required packages not available: {exc}")
        print("Install with: pip install diffusers transformers")
        return None

    # Load the splat pipeline (trust_remote_code is required for custom pipelines)
    pipeline = DiffusionPipeline.from_pretrained(
        model_name,
        custom_pipeline=model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    pipeline.eval()

    # The underlying UNet / transformer that maps images → Gaussian params
    model = pipeline.unet if hasattr(pipeline, "unet") else pipeline.model
    model.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)

    print(f"Dummy input shape: {dummy_input.shape}")
    print("Running forward pass to verify shapes...")

    with torch.no_grad():
        output = model(dummy_input)
        if isinstance(output, torch.Tensor):
            gaussians = output
        else:
            gaussians = output[0]

    print(f"Output shape: {gaussians.shape}")

    input_names  = ["image"]
    output_names = ["gaussians"]

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "image":     {0: "batch_size"},
            "gaussians": {0: "batch_size"},
        }

    print(f"Exporting to ONNX (opset {opset_version}) → {output_path} ...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    # Validate
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation: OK")

    return output_path


def test_onnx_inference(output_path: str, image_size: int) -> None:
    """Run a quick inference test with ONNX Runtime."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not available — skipping inference test.")
        return

    sess = ort.InferenceSession(output_path)
    dummy = np.random.rand(1, 3, image_size, image_size).astype(np.float32)
    outputs = sess.run(None, {"image": dummy})

    print(f"ONNX Runtime test: output shape = {outputs[0].shape}")
    assert outputs[0].ndim in (2, 3), "Expected [G,14] or [N,G,14] output"
    assert outputs[0].shape[-1] == 14, "Expected 14 features per Gaussian"
    print("Inference test: PASSED")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LGM Gaussian Splatting model to ONNX")
    parser.add_argument("--model",       default="dylanebert/LGM",
                        help="HuggingFace model ID (default: dylanebert/LGM)")
    parser.add_argument("--output",      default="lgm.onnx",
                        help="Output ONNX file path (default: lgm.onnx)")
    parser.add_argument("--image-size",  type=int, default=256,
                        help="Square input resolution (default: 256)")
    parser.add_argument("--opset",       type=int, default=17,
                        help="ONNX opset version (default: 17)")
    parser.add_argument("--static-batch", action="store_true",
                        help="Disable dynamic batch axis")
    parser.add_argument("--test",        action="store_true",
                        help="Run ONNX Runtime inference test after export")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    result = export_lgm_to_onnx(
        model_name    = args.model,
        output_path   = args.output,
        image_size    = args.image_size,
        opset_version = args.opset,
        dynamic_batch = not args.static_batch,
    )

    if result and args.test:
        test_onnx_inference(result, args.image_size)
