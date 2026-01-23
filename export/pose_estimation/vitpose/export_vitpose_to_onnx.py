import torch
import onnx
import numpy as np
import argparse
from pathlib import Path
import sys
import traceback


def export_vitpose_to_onnx(
    model_name="usyd-community/vitpose-base-simple",
    output_path="vitpose.onnx",
    image_height=256,
    image_width=192,
    opset_version=14,
    dynamic_batch=True
):
    """Export ViTPose model to ONNX format."""

    try:
        print(f"Loading ViTPose model: {model_name}")
        print(f"Python version: {sys.version}")
        print(f"PyTorch version: {torch.__version__}")

        try:
            import transformers
            print(f"Transformers version: {transformers.__version__}")
        except ImportError as e:
            print(f"Transformers not available: {e}")
            return None

        from transformers import VitPoseForPoseEstimation

        model = VitPoseForPoseEstimation.from_pretrained(model_name)
        model.eval()

        # Get num_keypoints from config
        num_keypoints = getattr(model.config, 'num_labels', 17)

        print(f"Model config:")
        print(f"   - Image size: {image_height}x{image_width}")
        print(f"   - Num keypoints: {num_keypoints}")

        # Shape: [batch_size, channels, height, width]
        dummy_input = torch.randn(1, 3, image_height, image_width)

        print(f"Dummy input shape: {dummy_input.shape}")

        print(f"Testing forward pass...")
        with torch.no_grad():
            output = model(dummy_input)
            heatmaps = output.heatmaps
            print(f"Forward pass successful, heatmaps shape: {heatmaps.shape}")

        input_names = ["pixel_values"]
        output_names = ["heatmaps"]

        if dynamic_batch:
            dynamic_axes = {
                "pixel_values": {0: "batch_size"},
                "heatmaps": {0: "batch_size"}
            }
        else:
            dynamic_axes = None

        print(f"Exporting to ONNX...")

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
            verbose=True
        )

        print(f"Model exported to: {output_path}")

        print(f"Verifying ONNX model...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"ONNX model verification passed!")

        print(f"\nONNX Model Info:")
        print(f"   - Input shape: {onnx_model.graph.input[0].type.tensor_type.shape}")
        print(f"   - Output shape: {onnx_model.graph.output[0].type.tensor_type.shape}")
        print(f"   - Opset version: {opset_version}")
        print(f"   - File size: {Path(output_path).stat().st_size / (1024*1024):.2f} MB")

        return output_path

    except Exception as e:
        print(f"Error during export: {e}")
        traceback.print_exc()
        return None


def test_onnx_model(onnx_path, image_height=256, image_width=192):
    """Test the exported ONNX model."""
    try:
        import onnxruntime as ort

        print(f"Testing ONNX model: {onnx_path}")

        session = ort.InferenceSession(onnx_path)

        test_input = np.random.randn(1, 3, image_height, image_width).astype(np.float32)

        outputs = session.run(None, {"pixel_values": test_input})
        heatmaps = outputs[0]

        print(f"ONNX inference successful!")
        print(f"   - Input shape: {test_input.shape}")
        print(f"   - Heatmaps shape: {heatmaps.shape}")

        return True

    except ImportError:
        print("ONNXRuntime not installed. Skipping ONNX test.")
        print("   Install with: pip install onnxruntime")
        return False
    except Exception as e:
        print(f"ONNX test failed: {e}")
        traceback.print_exc()
        return False


def main():
    print("Starting ViTPose ONNX export...")

    try:
        parser = argparse.ArgumentParser(description="Export ViTPose to ONNX")
        parser.add_argument("--model", default="usyd-community/vitpose-base-simple",
                           help="HuggingFace model name")
        parser.add_argument("--output", default="vitpose.onnx",
                           help="Output ONNX file path")
        parser.add_argument("--image-height", type=int, default=256,
                           help="Input image height")
        parser.add_argument("--image-width", type=int, default=192,
                           help="Input image width")
        parser.add_argument("--opset", type=int, default=14,
                           help="ONNX opset version")
        parser.add_argument("--static-batch", action="store_true",
                           help="Use static batch size (default: dynamic)")
        parser.add_argument("--test", action="store_true",
                           help="Test the exported ONNX model")

        args = parser.parse_args()

        print(f"Arguments:")
        print(f"   - Model: {args.model}")
        print(f"   - Output: {args.output}")
        print(f"   - Image size: {args.image_height}x{args.image_width}")
        print(f"   - Opset: {args.opset}")
        print(f"   - Dynamic batch: {not args.static_batch}")
        print(f"   - Test: {args.test}")

        onnx_path = export_vitpose_to_onnx(
            model_name=args.model,
            output_path=args.output,
            image_height=args.image_height,
            image_width=args.image_width,
            opset_version=args.opset,
            dynamic_batch=not args.static_batch
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

    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
