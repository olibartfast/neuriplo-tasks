# ETH-XGaze Gaze Export (ptgaze resnet18)

`export.py` downloads the public ptgaze ETH-XGaze resnet18 weights, rebuilds the timm resnet18 + 2-unit head, and exports it to ONNX. No weights are committed.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch timm safetensors huggingface_hub onnx onnxscript onnxruntime opencv-python numpy

python export/gaze_estimation/eth_xgaze/export.py --test
python export/gaze_estimation/eth_xgaze/export.py --image face.jpg --test
```

The ETH-XGaze dataset itself is distributed under a custom research agreement (non-commercial); the ptgaze weights are MIT but trained on it — assess commercial use yourself. Do not commit downloaded weights or exported models.

## Verified contract

`hysts/ptgaze-eth-xgaze-resnet18`, timm resnet18 strict state-dict load, opset 17 (kept at 18 by the exporter where required):

- Input tensor: `pixel_values` with shape `[B, 3, 224, 224]` float32 — RGB face crop, `[0, 1]` + ImageNet mean/std, BGR→RGB
- Output tensor: `gaze` with shape `[B, 2]` float32 — `[pitch, yaw]` radians, ETH-XGaze order

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

See [GazeEstimation.md](../GazeEstimation.md) for the task contract and input scope.
