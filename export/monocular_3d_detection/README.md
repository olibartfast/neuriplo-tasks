# Monocular 3D Detection Export

UrbanOmniDetect-style models predict a 2D box plus the eight projected corners of a 3D cuboid for road users. The eight points are rigid-object geometry (not human keypoints), so they live in `Monocular3DDetection`, not `PoseEstimation`. See [pose task scope](../../docs/pose_estimation_scope.md).

Public checkpoints live on Hugging Face (`mehmetkeremturkcan/UrbanOmniDetect` for v1, `mehmetkeremturkcan/UrbanOmniDetect-2` for the YOLO26 hybrid). Checkpoints are distributed under AGPL-3.0; that governs the weights, not this MIT library, but be aware before redistributing them.

## Setup and export

`export.py` downloads a pretrained UrbanOmniDetect checkpoint from Hugging Face and exports it with stock Ultralytics to ONNX (YOLO-pose-style `output0`: box plus keypoints, no separate heads). No weights are committed.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install ultralytics huggingface_hub onnx onnxruntime opencv-python numpy

python export/monocular_3d_detection/export.py --test
python export/monocular_3d_detection/export.py --image street.jpg --test
```

Available scales cover YOLOv8/v9/v11/v12 (`n/s/m/l/x`) at 640/1280/1920 inputs; UrbanOmniDetect-2 adds YOLO26-based hybrid detection+pose checkpoints (see the [model hub](https://huggingface.co/mehmetkeremturkcan/UrbanOmniDetect-2)). Match `imgsz` to the checkpoint's training size and to `ModelInfo::input_shapes`.

Do not commit downloaded checkpoints or exported models.

## neuriplo-tasks contract

- TaskFactory model types: `urbanomni` / `urbanomnidetect` (v1, prefix), `urbanomnidetect2` (v2, YOLO26 end-to-end), `monocular3d` (prefix), `mono3d` (exact)
- Input: RGB float32 NCHW `[B,3,H,W]` (640/1280/1920), YOLO letterboxed, `[0,1]`, BGR→RGB
- Output: single `output0` tensor —
  - v1 (YOLOv8/v11 style): `[B, 4+nc+24, N]` (`cx, cy, w, h` input pixels, `nc` class scores, 8×`(x, y, conf)` corner triples in input pixels)
  - v1 rows (YOLOv5 style): `[B, N, 5+nc+24]` with objectness
  - v2 (YOLO26 end-to-end): `[B, N, 30]` (`x1, y1, x2, y2, score, class_id`, 8×`(x, y, conf)`)
- Result: `Monocular3DDetection` per surviving detection — `bbox` in original-frame pixels, `class_id`/`class_confidence`, 8 `projected_corners` in original-frame pixels, `geometry_confidence` as the mean corner confidence
- Corner order is ground-first (0-3 ground, 4-7 roof, `i+4` above `i`, 0/3 front face), as emitted by the released v1 and v2 checkpoints (verified empirically; the paper text states the opposite — the postprocessor has an opt-in remap for top-first outputs)
- Filtering: rows below `TaskConfig::confidence_threshold` (default 0.25) are dropped, then class-agnostic NMS with `TaskConfig::nms_threshold`

Letterbox padding is removed for both boxes and corners, so wide/tall frames map back to the original resolution.

See [docs/monocular_3d_detection.md](../../docs/monocular_3d_detection.md) for the corner convention and worked contract.
