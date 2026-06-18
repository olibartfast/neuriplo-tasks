import argparse


def main():
    parser = argparse.ArgumentParser('RF-DETR Keypoint Pose Estimation Export Script',
                                     description='Export RF-DETR keypoint pose estimation model to ONNX format')

    # Export options that will be passed to the model's export() method
    parser.add_argument('--output_dir', default=None, type=str,
                        help='Path to save exported model (default: current directory)')
    parser.add_argument('--opset_version', default=17, type=int,
                        help='ONNX opset version (default: 17)')
    parser.add_argument('--simplify', action='store_true',
                        help='Simplify ONNX model using onnxsim')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size for export (default: 1)')
    parser.add_argument('--input_size', default=640, type=int,
                        help='Input image size (default: 640)')
    parser.add_argument('--model_type', default='medium', type=str,
                        choices=['nano', 'small', 'medium', 'large', 'xlarge'],
                        help='Model type (default: medium)')

    args = parser.parse_args()

    print("="*60)
    print("RF-DETR Keypoint Pose Estimation Model Export")
    print("="*60)

    # Initialize the keypoint model
    print(f"\n[1/2] Loading RF-DETR Keypoint model ({args.model_type})...")
    model = None
    if args.model_type == 'nano':
        from rfdetr import RFDETRKeypointNano
        model = RFDETRKeypointNano()
    elif args.model_type == 'small':
        from rfdetr import RFDETRKeypointSmall
        model = RFDETRKeypointSmall()
    elif args.model_type == 'medium':
        from rfdetr import RFDETRKeypointMedium
        model = RFDETRKeypointMedium()
    elif args.model_type == 'large':
        from rfdetr import RFDETRKeypointLarge
        model = RFDETRKeypointLarge()
    elif args.model_type == 'xlarge':
        from rfdetr import RFDETRKeypointXLarge
        model = RFDETRKeypointXLarge()
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    # Build export kwargs from arguments
    export_kwargs = {
        'opset_version': args.opset_version,
        'simplify': args.simplify,
        'batch_size': args.batch_size,
    }

    # Add output_dir if specified
    if args.output_dir:
        export_kwargs['output_dir'] = args.output_dir

    # Export using the model's built-in export method
    print("\n[2/2] Exporting to ONNX format...")
    print(f"  - Model type: {args.model_type}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Input size: {args.input_size}x{args.input_size}")
    print(f"  - ONNX opset: {args.opset_version}")
    print(f"  - Simplify: {args.simplify}")

    model.export(**export_kwargs)

    print("\n" + "="*60)
    print("✓ Export complete!")
    print("="*60)
    print("\nModel outputs:")
    print("  - dets: Bounding boxes [batch, num_queries, 4]")
    print("  - labels: Class logits [batch, num_queries, num_classes]")
    print("  - keypoints: Keypoint coordinates and confidences [batch, num_queries, num_keypoints, 3]")
    print("\nNote: This is a keypoint pose estimation model.")
    print("="*60)


if __name__ == '__main__':
    main()
