#!/usr/bin/env python3
"""ETH-XGaze gaze estimation export to ONNX.

Downloads the public ptgaze ETH-XGaze resnet18 weights (safetensors),
rebuilds the timm resnet18 + 2-unit head, and exports it to ONNX with a
dynamic batch axis.

Output tensor contract (matches GazeEstimationTask):
  input  pixel_values [B,3,224,224] float32 (ImageNet-normalized face crop)
  output gaze         [B,2] float32 ([pitch, yaw] radians, ETH-XGaze order)

Usage:
    python export.py --test
    python export.py --image face.jpg --test
    python export.py --output ethxgaze_resnet18.onnx --opset 17

Requires: torch, timm, safetensors, huggingface_hub, onnx (+onnxscript),
          onnxruntime (for --test), opencv-python (for --image), numpy.
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ID = "hysts/ptgaze-eth-xgaze-resnet18"
WEIGHTS_FILE = "model.safetensors"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_model(weights_path: Path):
    import timm
    import torch
    from safetensors.torch import load_file

    state = load_file(str(weights_path))
    model = timm.create_model("resnet18", pretrained=False, num_classes=2)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing={missing} unexpected={unexpected}")
    return model.eval()


def download_weights(dest_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(REPO_ID, WEIGHTS_FILE, local_dir=str(dest_dir))
    logger.info("weights: %s", path)
    return Path(path)


def export_onnx(model, output: Path, opset: int) -> Path:
    import torch

    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["pixel_values"],
        output_names=["gaze"],
        dynamic_axes={"pixel_values": {0: "batch"}, "gaze": {0: "batch"}},
        opset_version=opset,
    )
    logger.info("exported %s", output)
    return output


def load_face(image_path: Path):
    import cv2
    import numpy as np

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    tensor = resized.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, np.float32).reshape(1, 3, 1, 1)
    std = np.array(IMAGENET_STD, np.float32).reshape(1, 3, 1, 1)
    return (tensor - mean) / std


def run_test(onnx_path: Path, image: Path | None) -> None:
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    logger.info(
        "inputs=%s outputs=%s",
        [(i.name, i.shape) for i in sess.get_inputs()],
        [(o.name, o.shape) for o in sess.get_outputs()],
    )
    if image is not None:
        face = load_face(image)
    else:
        rng = np.random.default_rng(0)
        face = rng.random((1, 3, 224, 224), dtype=np.float32)
    out = sess.run(["gaze"], {"pixel_values": face})[0]
    assert out.shape[1] == 2, f"expected [B,2], got {out.shape}"
    assert np.all(np.isfinite(out)), "non-finite gaze output"
    logger.info("gaze [pitch, yaw] rad = %s (deg = %s)", out[0], np.degrees(out[0]))
    logger.info("test passed")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Export ETH-XGaze gaze model to ONNX")
    parser.add_argument("--weights-dir", default="weights", help="where to download the safetensors")
    parser.add_argument("--output", default="ethxgaze_resnet18.onnx", help="output ONNX path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--image", default=None, help="optional face photo for --test")
    parser.add_argument("--test", action="store_true", help="verify the export with onnxruntime")
    args = parser.parse_args(argv)

    weights = download_weights(Path(args.weights_dir))
    model = build_model(weights)
    onnx_path = export_onnx(model, Path(args.output), args.opset)
    if args.test:
        run_test(onnx_path, Path(args.image) if args.image else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
