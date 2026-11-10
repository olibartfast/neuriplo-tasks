#!/usr/bin/env python3
"""UrbanOmniDetect monocular 3D cuboid export to ONNX.

Downloads a pretrained UrbanOmniDetect checkpoint from Hugging Face and
exports it with stock Ultralytics to ONNX (YOLO-pose-style `output0`).

Output tensor contract (matches Monocular3DDetectionTask, v1 checkpoints):
  input  images  [B,3,H,W] float32 (YOLO letterbox, [0,1], RGB)
  output output0 [B,4+nc+16,N] float32 (cx, cy, w, h in input pixels,
                 nc class scores, 8 x/y cuboid corners in input pixels;
                 kpt_shape [8, 2], i.e. no per-corner confidence)

UrbanOmniDetect-2 checkpoints use the YOLO26 end-to-end rows layout
[B,N,6+24]; the same export command applies, routed by model name.

Usage:
    python export.py --test
    python export.py --model urbanomnidetect_yolo11n-p2_640 --test
    python export.py --model checkpoints/urbanomnidetect_yolo11n-p2_640.pt --imgsz 1280

Requires: ultralytics, huggingface_hub, onnx, onnxruntime (for --test).
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ID = "mehmetkeremturkcan/UrbanOmniDetect"
CHECKPOINTS = {
    # Smallest, fastest; used for verification. Larger scales / 1280 / 1920
    # inputs trade latency for accuracy; see the export README.
    "urbanomnidetect_yolo11n-p2_640": "checkpoints/urbanomnidetect_yolo11n-p2_640.pt",
}


def resolve_model(model: str, weights_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    candidate = Path(model)
    if candidate.suffix == ".pt" and candidate.exists():
        return candidate
    key = candidate.stem
    if key not in CHECKPOINTS:
        raise ValueError(f"unknown model '{model}'; known: {sorted(CHECKPOINTS)}")
    local = hf_hub_download(REPO_ID, CHECKPOINTS[key], local_dir=str(weights_dir))
    logger.info("weights: %s", local)
    return Path(local)


def export_onnx(weights: Path, imgsz: int) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    logger.info("classes: %s", model.names)
    out = model.export(format="onnx", imgsz=imgsz)
    logger.info("exported %s", out)
    return Path(out)


def run_test(onnx_path: Path, image: Path | None, conf: float) -> None:
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    logger.info(
        "inputs=%s outputs=%s",
        [(i.name, i.shape) for i in sess.get_inputs()],
        [(o.name, o.shape) for o in sess.get_outputs()],
    )
    shape = sess.get_inputs()[0].shape
    size = int(shape[2]) if isinstance(shape[2], int) else 640
    if image is not None:
        import cv2

        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(image)
        h, w = img.shape[:2]
        scale = min(size / w, size / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((size, size, 3), np.uint8)
        canvas[(size - nh) // 2 : (size - nh) // 2 + nh, (size - nw) // 2 : (size - nw) // 2 + nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    else:
        rng = np.random.default_rng(0)
        tensor = rng.random((1, 3, size, size), dtype=np.float32)
    (out,) = sess.run(None, {"images": tensor})
    logger.info("output0 shape=%s dtype=%s", out.shape, out.dtype)
    batch, channels, anchors = out.shape
    nc3, nc2 = channels - 28, channels - 20
    if nc3 >= 1:
        scores = out[0, 4 : 4 + nc3, :].max(axis=0)  # kd=3: 4 box + nc scores + 24 kpt
    elif nc2 >= 1:
        scores = out[0, 4 : 4 + nc2, :].max(axis=0)  # kd=2: 4 box + nc scores + 16 kpt
    else:
        raise AssertionError(f"unexpected head width {channels} (want 4+nc+8*kd)")
    logger.info("detections above %.2f: %d (max score %.3f)", conf, int((scores > conf).sum()), scores.max())
    logger.info("test passed")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Export UrbanOmniDetect to ONNX")
    parser.add_argument("--model", default="urbanomnidetect_yolo11n-p2_640",
                        help="checkpoint name or local .pt path")
    parser.add_argument("--weights-dir", default="weights", help="where to download the checkpoint")
    parser.add_argument("--imgsz", type=int, default=640, help="export image size (match the checkpoint)")
    parser.add_argument("--image", default=None, help="optional road photo for --test")
    parser.add_argument("--conf", type=float, default=0.25, help="detection threshold for --test")
    parser.add_argument("--test", action="store_true", help="verify the export with onnxruntime")
    args = parser.parse_args(argv)

    weights = resolve_model(args.model, Path(args.weights_dir))
    onnx_path = export_onnx(weights, args.imgsz)
    if args.test:
        run_test(onnx_path, Path(args.image) if args.image else None, args.conf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
