import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import onnx
import torch


def export_depth_anything_v2_to_onnx(
    model_name="depth-anything/Depth-Anything-V2-Small-hf",
    output_path="depth_anything_v2.onnx",
    image_height=518,
    image_width=518,
    opset_version=14,
    dynamic_batch=True,
):
    """Export Depth Anything V2 model to ONNX format."""

    try:
        print(f"Loading Depth Anything V2 model: {model_name}")
        print(f"Python version: {sys.version}")
        print(f"PyTorch version: {torch.__version__}")

        try:
            import transformers

            print(f"Transformers version: {transformers.__version__}")
        except ImportError as exc:
            print(f"Transformers not available: {exc}")
            return None

        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        model = AutoModelForDepthEstimation.from_pretrained(model_name)
        processor = AutoImageProcessor.from_pretrained(model_name)
        model.eval()

        print("Model config:")
        print(f"   - Image size: {image_height}x{image_width}")
        print(f"   - Hidden size: {getattr(model.config, 'hidden_size', 'N/A')}")
        print(f"   - Processor: {processor.__class__.__name__}")

        dummy_input = torch.randn(1, 3, image_height, image_width, dtype=torch.float32)

        print(f"Dummy input shape: {dummy_input.shape}")

        print("Testing forward pass...")
        with torch.no_grad():
            output = model(dummy_input)
            predicted_depth = output.predicted_depth
            print(f"Forward pass successful, depth shape: {predicted_depth.shape}")

        input_names = ["pixel_values"]
        output_names = ["predicted_depth"]

        if dynamic_batch:
            dynamic_axes = {
                "pixel_values": {0: "batch_size"},
                "predicted_depth": {0: "batch_size"},
            }
        else:
            dynamic_axes = None

        print("Exporting to ONNX...")

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
            verbose=True,
        )

        print(f"Model exported to: {output_path}")

        print("Verifying ONNX model...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed!")

        print("\nONNX Model Info:")
        print(f"   - Input shape: {onnx_model.graph.input[0].type.tensor_type.shape}")
        print(f"   - Output shape: {onnx_model.graph.output[0].type.tensor_type.shape}")
        print(f"   - Opset version: {opset_version}")
        print(f"   - File size: {Path(output_path).stat().st_size / (1024 * 1024):.2f} MB")

        return output_path

    except Exception as exc:
        print(f"Error during export: {exc}")
        traceback.print_exc()
        return None


def test_onnx_model(onnx_path, image_height=518, image_width=518):
    """Test the exported ONNX model."""
    try:
        import onnxruntime as ort

        print(f"Testing ONNX model: {onnx_path}")

        session = ort.InferenceSession(onnx_path)

        test_input = np.random.randn(1, 3, image_height, image_width).astype(np.float32)

        outputs = session.run(None, {"pixel_values": test_input})
        depth = outputs[0]

        print("ONNX inference successful!")
        print(f"   - Input shape: {test_input.shape}")
        print(f"   - Output shape: {depth.shape}")
        print(f"   - Output min/max: {depth.min():.6f} / {depth.max():.6f}")

        return True

    except ImportError:
        print("ONNXRuntime not installed. Skipping ONNX test.")
        print("   Install with: pip install onnxruntime")
        return False
    except Exception as exc:
        print(f"ONNX test failed: {exc}")
        traceback.print_exc()
        return False


def main():
    print("Starting Depth Anything V2 ONNX export...")

    try:
        parser = argparse.ArgumentParser(description="Export Depth Anything V2 to ONNX")
        parser.add_argument(
            "--model",
            default="depth-anything/Depth-Anything-V2-Small-hf",
            help="HuggingFace model name",
        )
        parser.add_argument("--output", default="depth_anything_v2.onnx", help="Output ONNX file path")
        parser.add_argument("--image-height", type=int, default=518, help="Input image height")
        parser.add_argument("--image-width", type=int, default=518, help="Input image width")
        parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
        parser.add_argument(
            "--static-batch",
            action="store_true",
            help="Use static batch size (default: dynamic)",
        )
        parser.add_argument("--test", action="store_true", help="Test the exported ONNX model")

        args = parser.parse_args()

        print("Arguments:")
        print(f"   - Model: {args.model}")
        print(f"   - Output: {args.output}")
        print(f"   - Image size: {args.image_height}x{args.image_width}")
        print(f"   - Opset: {args.opset}")
        print(f"   - Dynamic batch: {not args.static_batch}")
        print(f"   - Test: {args.test}")

        onnx_path = export_depth_anything_v2_to_onnx(
            model_name=args.model,
            output_path=args.output,
            image_height=args.image_height,
            image_width=args.image_width,
            opset_version=args.opset,
            dynamic_batch=not args.static_batch,
        )

        if onnx_path is None:
            print("Export failed!")
            sys.exit(1)

        if args.test:
            success = test_onnx_model(onnx_path, args.image_height, args.image_width)
            if not success:
                print("Testing failed!")
                sys.exit(1)

        print("Export completed successfully!")

    except Exception as exc:
        print(f"Fatal error: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
