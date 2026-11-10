# UrbanOmniDetect Export (v1 + v2)

`export.py` downloads a pretrained UrbanOmniDetect checkpoint from Hugging Face and exports it with stock Ultralytics to ONNX (YOLO-pose-style `output0`: box plus keypoints, no separate heads). No weights are committed.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install ultralytics huggingface_hub onnx onnxruntime opencv-python numpy

python export/monocular_3d_detection/urbanomnidetect/export.py --test
python export/monocular_3d_detection/urbanomnidetect/export.py --image street.jpg --test
```

Public checkpoints live on Hugging Face (`mehmetkeremturkcan/UrbanOmniDetect` for v1, `mehmetkeremturkcan/UrbanOmniDetect-2` for the YOLO26 hybrid). Checkpoints are distributed under AGPL-3.0; that governs the weights, not this MIT library, but be aware before redistributing them. Do not commit downloaded checkpoints or exported models.

## Verified contract

`urbanomnidetect_yolo11n-p2_640.pt` → `yolo export format=onnx imgsz=640`:

- Input tensor: `images` with shape `[1, 3, 640, 640]` float32 — YOLO letterbox, `[0, 1]`, RGB
- Output tensor: `output0` with shape `[1, 23, 34100]` float32 — `cx, cy, w, h` (input pixels), 3 class scores (`car`, `person`, `bike`), 8× `(x, y)` cuboid corners in input pixels (kpt_shape `[8, 2]`, no per-corner confidence)

## neuriplo-tasks contract

- TaskFactory model types: `urbanomni` / `urbanomnidetect` (v1, prefix), `urbanomnidetect2` (v2, YOLO26 end-to-end), `monocular3d` (prefix), `mono3d` (exact)
- Input: RGB float32 NCHW `[B,3,H,W]` (640/1280/1920), YOLO letterboxed, `[0,1]`, BGR→RGB
- Output: single `output0` tensor —
  - v1 channels-first: `[B, 4+nc+16, N]` (`cx, cy, w, h` input pixels, `nc` class scores, 8× `(x, y)` corners); kd=3 variants carry `(x, y, conf)` triples
  - v1 rows: `[B, N, 5+nc+8kd]` with objectness
  - v2 (YOLO26 end-to-end): `[B, N, 30]` (`x1, y1, x2, y2, score, class_id`, 8× `(x, y, conf)`)
- Result: `Monocular3DDetection` per surviving detection — `bbox` in original-frame pixels, `class_id`/`class_confidence`, 8 `projected_corners` in original-frame pixels, `geometry_confidence` as the mean corner confidence (falls back to the detection score when the export carries none)
- Corner order is ground-first (0-3 ground, 4-7 roof, `i+4` above `i`, 0/3 front face), as emitted by the released v1 and v2 checkpoints (verified empirically; the paper text states the opposite — the postprocessor has an opt-in remap for top-first outputs)
- Filtering: rows below `TaskConfig::confidence_threshold` (default 0.25) are dropped, then class-agnostic NMS with `TaskConfig::nms_threshold`

Letterbox padding is removed for both boxes and corners, so wide/tall frames map back to the original resolution.

See [Monocular3DDetection.md](../Monocular3DDetection.md) for the task contract and corner convention.
