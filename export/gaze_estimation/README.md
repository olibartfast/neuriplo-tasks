# Gaze Estimation Export

Appearance-based gaze models (ETH-XGaze-style, GazeTR-style) predict where a person is looking from a normalized face or eye crop. Face detection, landmarks, and calibration stay outside `neuriplo-tasks` — export only the gaze head.

## Setup and export

`export.py` downloads the public ptgaze ETH-XGaze resnet18 weights, rebuilds the timm resnet18 + 2-unit head, and exports it to ONNX. No weights are committed.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch timm safetensors huggingface_hub onnx onnxscript onnxruntime opencv-python numpy

python export/gaze_estimation/export.py --test
python export/gaze_estimation/export.py --image face.jpg --test
```

The ETH-XGaze dataset itself is distributed under a custom research agreement (non-commercial); the ptgaze weights are MIT but trained on it — assess commercial use yourself. Do not commit downloaded weights or exported models.

## neuriplo-tasks contract

- TaskFactory model types: `gaze`, `ethxgaze`, `gazetr` (prefix match, e.g. `ethxgaze-r50`, `gazetr-v1`)
- Input: RGB float32 NCHW `[B,3,224,224]`, resized from the face crop, `[0,1]` + ImageNet mean/std, BGR→RGB
- Output tensor 0: `[B,2]` pitch/yaw radians `[pitch, yaw]` or `[B,3]` xyz gaze vector (any float/int scalar type)
- Output tensor 1 (optional): `[B,1]`/`[B]` confidence in `[0,1]`
- Result: one `GazeEstimation` per batch row with `pitch`, `yaw`, normalized `direction`, `confidence`

Direction convention follows the ETH-XGaze / OpenGaze normalized camera (cf. ptgaze):

```text
direction = [-cos(pitch)*sin(yaw), -sin(pitch), -cos(pitch)*cos(yaw)]
pitch = asin(-y), yaw = atan2(-x, -z)
```

A horizontal flip of the input negates yaw and leaves pitch unchanged — the check to run when validating against normalized faces.

`[B,2]` angles are converted to `direction`; `[B,3]` vectors are normalized and back out `pitch`/`yaw`, so both families share one result schema.

See [docs/gaze_estimation.md](../../docs/gaze_estimation.md) for scope (face crop in, no detection/tracking inside the task).
