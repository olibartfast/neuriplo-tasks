#!/usr/bin/env python3
"""
YOLO-World Fixed-Vocabulary Export Script

Bake a custom class list into an Ultralytics YOLO-World model and export it to ONNX.
The text embeddings are frozen into the graph, so the exported model behaves like a
standard YOLOv8 detector and runs in neuriplo-tasks with model type "yoloworld".

Output tensor format:
  [batch, 4 + num_classes, 8400] at 640x640 (no objectness, NMS done by neuriplo-tasks)

The class order of --classes is the class_id order of the exported model. A
<model>_classes.txt file with one class per line is written next to the ONNX file.

Usage:
    # Export with an inline vocabulary
    python export.py --model yolov8s-worldv2.pt --classes person bus "traffic light"

    # Export with a vocabulary file (one class per line)
    python export.py --model yolov8m-worldv2.pt --classes-file my_classes.txt --imgsz 640

    # Export and check the output shape with ONNX Runtime
    python export.py --model yolov8s-worldv2.pt --classes person bus --test
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Only the v2 YOLO-World checkpoints support ONNX export in Ultralytics.
SUPPORTED_MODELS = [
    'yolov8s-worldv2',
    'yolov8m-worldv2',
    'yolov8l-worldv2',
    'yolov8x-worldv2',
]


def load_classes(args: argparse.Namespace) -> list:
    classes = list(args.classes or [])
    if args.classes_file:
        with open(args.classes_file, encoding='utf-8') as f:
            classes.extend(line.strip() for line in f if line.strip())
    if not classes:
        raise ValueError('No classes given: use --classes and/or --classes-file')
    if len(set(classes)) != len(classes):
        raise ValueError('Duplicate class names in vocabulary')
    return classes


def export(args: argparse.Namespace) -> Path:
    from ultralytics import YOLOWorld

    stem = Path(args.model).stem
    if stem not in SUPPORTED_MODELS:
        logger.warning('%s is not a known YOLO-World v2 checkpoint; export may fail', stem)

    classes = load_classes(args)
    logger.info('Vocabulary (%d classes): %s', len(classes), classes)

    model = YOLOWorld(args.model)
    model.set_classes(classes)

    exported = Path(model.export(format='onnx', imgsz=args.imgsz, opset=args.opset, dynamic=args.dynamic,
                                 simplify=not args.no_simplify))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / exported.name
    if exported.resolve() != onnx_path.resolve():
        shutil.move(str(exported), onnx_path)

    classes_path = output_dir / f'{onnx_path.stem}_classes.txt'
    classes_path.write_text('\n'.join(classes) + '\n', encoding='utf-8')

    logger.info('ONNX model:  %s', onnx_path)
    logger.info('Class list:  %s', classes_path)
    return onnx_path


def test_onnx(onnx_path: Path, num_classes: int, imgsz: int) -> None:
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    inp = session.get_inputs()[0]
    outputs = session.run(None, {inp.name: np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)})

    shape = outputs[0].shape
    logger.info('Input %s %s -> output %s', inp.name, inp.shape, shape)
    if len(shape) != 3 or shape[1] != 4 + num_classes:
        raise RuntimeError(f'Unexpected output shape {shape}, expected [1, {4 + num_classes}, anchors]')
    logger.info('Output matches the YOLOv8 detection layout neuriplo-tasks expects')


def main() -> int:
    parser = argparse.ArgumentParser(description='Export Ultralytics YOLO-World with a fixed vocabulary to ONNX')
    parser.add_argument('--model', default='yolov8s-worldv2.pt',
                        help=f'YOLO-World checkpoint (auto-downloaded if missing): {", ".join(SUPPORTED_MODELS)}')
    parser.add_argument('--classes', nargs='+', help='Class names, in class_id order')
    parser.add_argument('--classes-file', help='Text file with one class name per line, appended after --classes')
    parser.add_argument('--imgsz', type=int, default=640, help='Square input size (default: 640)')
    parser.add_argument('--opset', type=int, default=None, help='ONNX opset (default: Ultralytics default)')
    parser.add_argument('--dynamic', action='store_true', help='Export with a dynamic batch axis')
    parser.add_argument('--no-simplify', action='store_true', help='Skip ONNX simplification')
    parser.add_argument('--output-dir', default='./exported_models', help='Output directory')
    parser.add_argument('--test', action='store_true', help='Check the exported model with ONNX Runtime')
    args = parser.parse_args()

    try:
        onnx_path = export(args)
        if args.test:
            test_onnx(onnx_path, len(load_classes(args)), args.imgsz)
    except Exception as e:  # noqa: BLE001
        logger.error('Export failed: %s', e)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
