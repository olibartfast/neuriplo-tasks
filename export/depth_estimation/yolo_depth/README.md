# YOLO Depth Export

Ultralytics YOLO26 depth models perform monocular metric-depth estimation. Official checkpoints use the `-depth` suffix, such as `yolo26n-depth.pt`, and are trained for a 768x768 input.

## Setup and export

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "ultralytics[export]"

yolo export model=yolo26n-depth.pt format=onnx imgsz=768
```

The checkpoint downloads automatically on first use. Equivalent Python:

```python
from ultralytics import YOLO

YOLO("yolo26n-depth.pt").export(format="onnx", imgsz=768)
```

Do not commit the downloaded checkpoint or exported ONNX model.

## neuriplo-tasks contract

- TaskFactory model type: `yolo-depth`, `yolo26n-depth`, or another YOLO-prefixed string containing `depth`
- Input: RGB float32 NCHW `[B,3,H,W]`, YOLO letterboxed and normalized to `[0,1]`
- Output: float depth map `[B,1,H,W]` in meters
- Result: one `DepthEstimation` per batch item with the metric map in `depth`, display-normalized values in `normalized_depth`, and observed `min_depth` / `max_depth`

The exported head is already upsampled to the model input size. Postprocessing removes YOLO letterbox padding before resizing the depth map to the original frame.

See the [Ultralytics depth task documentation](https://docs.ultralytics.com/tasks/depth/) for model variants and current accuracy metrics.
