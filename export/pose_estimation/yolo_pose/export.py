#!/usr/bin/env python3
"""
YOLO Pose Model Export Script

Export Ultralytics YOLO pose models (v5, v8, v11, v26) to ONNX format.

Output tensor format:
  YOLOv8/v11/v26 (no objectness): [batch, 56, 8400]  → channels = 4 + 1 + 17*3
  YOLOv5         (objectness):    [batch, 8400, 56]   → same channels, transposed

Usage:
    # YOLOv8 pose (recommended, ultralytics package)
    python export.py --model yolov8n-pose.pt --format onnx

    # YOLO11 pose
    python export.py --model yolo11s-pose.pt --format onnx --imgsz 640

    # YOLOv5 pose (requires yolov5 repository)
    python export.py --model yolov5s-pose.pt --version v5 --repo-dir ./repositories/yolov5 --format onnx

    # Download weights automatically
    python export.py --model yolov8n-pose --download-weights --format onnx

    # Export and test with ONNX Runtime
    python export.py --model yolov8n-pose.pt --format onnx --test
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ultralytics YOLO pose weight download URLs
POSE_WEIGHTS = {
    # YOLOv8 pose
    'yolov8n-pose': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt',
    'yolov8s-pose': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-pose.pt',
    'yolov8m-pose': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-pose.pt',
    'yolov8l-pose': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-pose.pt',
    'yolov8x-pose': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-pose.pt',
    'yolov8x-pose-p6': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-pose-p6.pt',
    # YOLO11 pose
    'yolo11n-pose': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt',
    'yolo11s-pose': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt',
    'yolo11m-pose': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt',
    'yolo11l-pose': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-pose.pt',
    'yolo11x-pose': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt',
    # YOLO26 pose
    'yolo26n-pose': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt',
    'yolo26s-pose': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-pose.pt',
    'yolo26m-pose': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-pose.pt',
    'yolo26l-pose': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l-pose.pt',
    'yolo26x-pose': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-pose.pt',
}


def detect_version(model_path: str) -> str:
    name = Path(model_path).stem.lower()
    if 'yolov5' in name:
        return 'v5'
    if 'yolov8' in name:
        return 'v8'
    if 'yolo11' in name:
        return 'v11'
    if 'yolo26' in name:
        return '26'
    # Default to ultralytics for unknown
    logger.warning(f"Could not auto-detect version for '{name}', defaulting to v8 (ultralytics)")
    return 'v8'


def download_weights(model_name: str, weights_dir: str = './weights') -> str:
    """Download pose model weights if not present."""
    weights_path = Path(weights_dir)
    weights_path.mkdir(exist_ok=True, parents=True)

    stem = Path(model_name).stem.lower()
    url = POSE_WEIGHTS.get(stem)

    if not url:
        logger.warning(f"No download URL for '{stem}'. Pass a local .pt file path instead.")
        return model_name

    dest = weights_path / f"{stem}.pt"
    if dest.exists():
        logger.info(f"Weights already present: {dest}")
        return str(dest)

    logger.info(f"Downloading {stem} from {url}")
    try:
        import urllib.request
        urllib.request.urlretrieve(url, dest)
        logger.info(f"Downloaded to {dest}")
        return str(dest)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return model_name


def export_ultralytics(model_path: str, output_dir: str, imgsz: int, batch_size: int,
                       opset: int, simplify: bool, dynamic: bool) -> str:
    """Export via ultralytics package (YOLOv8, YOLO11, YOLO26)."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Install with: pip install ultralytics>=8.3.0")
        sys.exit(1)

    logger.info(f"Loading model: {model_path}")
    model = YOLO(model_path)

    export_args = {
        'format': 'onnx',
        'imgsz': imgsz,
        'batch': batch_size,
        'simplify': simplify,
        'opset': opset,
        'dynamic': dynamic,
    }
    logger.info(f"Exporting with args: {export_args}")
    output_path = model.export(**export_args)

    if output_path and Path(output_path).exists():
        dest = Path(output_dir) / Path(output_path).name
        shutil.move(output_path, dest)
        logger.info(f"Saved to: {dest}")
        return str(dest)

    return str(output_path)


def export_yolov5_pose(model_path: str, output_dir: str, imgsz: int, batch_size: int,
                       opset: int, simplify: bool, dynamic: bool, repo_dir: str) -> str:
    """Export YOLOv5 pose using the yolov5 repository export script."""
    if not repo_dir or not Path(repo_dir).exists():
        logger.error("YOLOv5 pose export requires the yolov5 repository.")
        logger.info("Clone it with:")
        logger.info("  git clone https://github.com/ultralytics/yolov5.git repositories/yolov5")
        logger.info("Then re-run with: --repo-dir ./repositories/yolov5")
        sys.exit(1)

    repo_dir = Path(repo_dir).resolve()
    export_script = repo_dir / 'export.py'
    if not export_script.exists():
        logger.error(f"export.py not found in {repo_dir}")
        sys.exit(1)

    cmd = [
        sys.executable, 'export.py',
        '--weights', str(Path(model_path).resolve()),
        '--img-size', str(imgsz),
        '--batch-size', str(batch_size),
        '--include', 'onnx',
        '--opset', str(opset),
    ]
    if simplify:
        cmd.append('--simplify')
    if dynamic:
        cmd.append('--dynamic')

    logger.info(f"Running YOLOv5 export from {repo_dir}")
    subprocess.run(cmd, cwd=str(repo_dir), check=True)

    expected = Path(model_path).resolve().with_suffix('.onnx')
    if expected.exists():
        dest = Path(output_dir) / expected.name
        shutil.move(str(expected), str(dest))
        logger.info(f"Saved to: {dest}")
        return str(dest)

    return str(expected)


def test_onnx(onnx_path: str, imgsz: int, batch_size: int) -> bool:
    """Run a basic ONNX Runtime inference test."""
    try:
        import numpy as np
        import onnxruntime as ort

        logger.info(f"Testing ONNX model: {onnx_path}")
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        dummy = np.random.randn(batch_size, 3, imgsz, imgsz).astype(np.float32)
        outputs = session.run(None, {input_name: dummy})
        logger.info(f"Inference OK — output shapes: {[o.shape for o in outputs]}")
        return True
    except ImportError:
        logger.warning("onnxruntime not installed. Skipping test (pip install onnxruntime).")
        return False
    except Exception as e:
        logger.error(f"ONNX test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="YOLO Pose Model Export Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # YOLOv8 pose — download weights and export
    python export.py --model yolov8n-pose --download-weights --format onnx

    # YOLO11 pose — local weights
    python export.py --model yolo11s-pose.pt --format onnx

    # YOLOv5 pose — requires yolov5 repo
    python export.py --model yolov5s-pose.pt --version v5 --repo-dir ./repositories/yolov5 --format onnx

    # Export and run ONNX Runtime test
    python export.py --model yolov8n-pose.pt --format onnx --test
        """
    )

    parser.add_argument('--model', '-m', required=True,
                        help='Path to .pt weights or model name (e.g. yolov8n-pose)')
    parser.add_argument('--version', '-v', default='auto',
                        choices=['auto', 'v5', 'v8', 'v11', '26'],
                        help='YOLO version (default: auto-detect)')
    parser.add_argument('--format', '-f', default='onnx',
                        choices=['onnx', 'tensorrt'],
                        help='Export format (default: onnx)')
    parser.add_argument('--output-dir', '-o', default='./exported_models',
                        help='Output directory (default: ./exported_models)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input image size (default: 640)')
    parser.add_argument('--batch-size', '-b', type=int, default=1,
                        help='Batch size (default: 1)')
    parser.add_argument('--opset', type=int, default=12,
                        help='ONNX opset version (default: 12)')
    parser.add_argument('--no-simplify', action='store_true',
                        help='Skip ONNX simplification')
    parser.add_argument('--dynamic', action='store_true',
                        help='Enable dynamic batch size')
    parser.add_argument('--repo-dir', default=None,
                        help='Path to yolov5 repository (required for --version v5)')
    parser.add_argument('--download-weights', action='store_true',
                        help='Download model weights automatically')
    parser.add_argument('--weights-dir', default='./weights',
                        help='Directory for downloaded weights (default: ./weights)')
    parser.add_argument('--test', action='store_true',
                        help='Run ONNX Runtime inference test after export')

    args = parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    model_path = args.model
    if args.download_weights:
        model_path = download_weights(args.model, args.weights_dir)

    version = args.version
    if version == 'auto':
        version = detect_version(model_path)
    logger.info(f"YOLO version: {version}")

    simplify = not args.no_simplify

    if args.format == 'onnx':
        if version == 'v5':
            onnx_path = export_yolov5_pose(model_path, args.output_dir, args.imgsz,
                                           args.batch_size, args.opset, simplify,
                                           args.dynamic, args.repo_dir)
        else:
            onnx_path = export_ultralytics(model_path, args.output_dir, args.imgsz,
                                           args.batch_size, args.opset, simplify,
                                           args.dynamic)

        if onnx_path:
            logger.info(f"ONNX export complete: {onnx_path}")

        if args.test and onnx_path:
            test_onnx(onnx_path, args.imgsz, args.batch_size)

    elif args.format == 'tensorrt':
        logger.info("TensorRT: export to ONNX first, then convert with trtexec:")
        stem = Path(model_path).stem
        logger.info(f"  trtexec --onnx={stem}.onnx --saveEngine={stem}.engine --fp16")


if __name__ == '__main__':
    main()
