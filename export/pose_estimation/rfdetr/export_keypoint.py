import argparse
import sys


def main():
    parser = argparse.ArgumentParser('RF-DETR Keypoint Pose Estimation Export Script',
                                     description='Export RF-DETR keypoint pose estimation model to ONNX format')

    parser.add_argument('--output_dir', default=None, type=str,
                        help='Path to save exported model (default: current directory)')
    parser.add_argument('--opset_version', default=17, type=int,
                        help='ONNX opset version (default: 17)')
    parser.add_argument('--simplify', action='store_true',
                        help='Simplify ONNX model using onnxsim after export')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size for export (default: 1)')
    parser.add_argument('--input_size', default=640, type=int,
                        help='Input image size (default: 640)')
    parser.add_argument('--model_type', default='preview', type=str,
                        choices=['preview', 'nano', 'small', 'medium', 'large', 'xlarge'],
                        help='Model type (default: preview)')

    args = parser.parse_args()

    print("="*60)
    print("RF-DETR Keypoint Pose Estimation Model Export")
    print("="*60)

    print(f"\n[1/2] Loading RF-DETR Keypoint model ({args.model_type})...")
    model = None
    if args.model_type == 'preview':
        from rfdetr import RFDETRKeypointPreview
        model = RFDETRKeypointPreview()
    else:
        # Try named class first; fall back to Preview if not yet available
        class_map = {
            'nano': 'RFDETRKeypointNano',
            'small': 'RFDETRKeypointSmall',
            'medium': 'RFDETRKeypointMedium',
            'large': 'RFDETRKeypointLarge',
            'xlarge': 'RFDETRKeypointXLarge',
        }
        import rfdetr
        class_name = class_map[args.model_type]
        if hasattr(rfdetr, class_name):
            model = getattr(rfdetr, class_name)()
        else:
            print(f"  Warning: {class_name} not available in installed rfdetr; falling back to RFDETRKeypointPreview")
            from rfdetr import RFDETRKeypointPreview
            model = RFDETRKeypointPreview()

    export_kwargs = {
        'opset_version': args.opset_version,
        'batch_size': args.batch_size,
    }

    if args.output_dir:
        export_kwargs['output_dir'] = args.output_dir

    print("\n[2/2] Exporting to ONNX format...")
    print(f"  - Model type: {args.model_type}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Input size: {args.input_size}x{args.input_size}")
    print(f"  - ONNX opset: {args.opset_version}")
    print(f"  - Simplify: {args.simplify}")

    output_path = model.export(**export_kwargs)
    print(f"  - Output: {output_path}")

    if args.simplify:
        try:
            import onnx
            from onnxsim import simplify
            model_onnx = onnx.load(str(output_path))
            model_simplified, check = simplify(model_onnx)
            if not check:
                print("  Warning: onnxsim simplification check incomplete")
            onnx.save(model_simplified, str(output_path))
            print("  - onnxsim simplification applied")
        except ImportError:
            print("  Warning: onnxsim not installed; skipping simplification")
            print("  Install with: pip install onnxsim")

    print("\n" + "="*60)
    print("Export complete")
    print("="*60)
    print("\nModel outputs:")
    print("  - dets: Bounding boxes [batch, num_queries, 4]")
    print("  - labels: Class logits [batch, num_queries, num_classes]")
    print("  - keypoints: Keypoint coordinates and confidences [batch, num_queries, num_keypoints, 3]")
    print("\nNote: This is a keypoint pose estimation model.")
    print("="*60)


if __name__ == '__main__':
    main()
