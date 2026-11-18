# Depth Anything 3 Export

Depth Anything 3 (DA3) monocular relative-depth models, exported with the
upstream script in [depth-anything-3-cpp-inference](https://github.com/olibartfast/depth-anything-3-cpp-inference)
(`export_onnx.py`). neuriplo-tasks consumes relative-depth variants only;
metric, Nested, and camera-depth models are rejected — see below.

## Setup and export

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "torch>=2" torchvision xformers
python -m pip install -e /path/to/Depth-Anything-3

python /path/to/depth-anything-3-cpp-inference/export_onnx.py \
    --model-name da3-large --output da3-large.onnx --process-res 504
```

Supported `--model-name` values for this task (relative depth):
`da3-small`, `da3-base`, `da3-large`, `da3-giant`, `da3mono-large`
(prefer the retrained `-1.1` checkpoints for large/giant where available).
`da3metric-large`, `da3-nested-giant-large`, and the camera-depth family
(`d405`, `d435`, …) are **not** supported by this task: metric scaling needs a
focal length the task contract does not carry, and camera-depth models take a
4-channel RGB+sensor-depth input `export_onnx.py` cannot produce.

Do not commit downloaded checkpoints or exported ONNX models. Small, Base,
Mono, and Metric checkpoints are Apache-2.0; Large, Giant, and Nested are
CC BY-NC 4.0 (non-commercial) — prefer Apache-2.0 variants unless the use
allows it.

## neuriplo-tasks contract

- TaskFactory model types: `depth-anything-3`, `depth_anything_3`,
  `da3-small`, `da3mono-large`, or any type containing `depthanything3` or
  starting with `da3`
- Input: RGB float32 NCHW `[B,3,H,W]` (default 504x504), `[0,1]` + ImageNet
  mean/std, BGR→RGB. Export dims must be multiples of 14 (RoPE constraint);
  the export is shape-locked to the trace resolution, so run the size it was
  traced with (`--process-res`, or `--fixed-size` and that exact size)
- Output tensor 0: float depth map `[B,H,W]` (relative); an optional second
  `conf` map is currently ignored
- Result: one `DepthEstimation` per batch item with the raw map in `depth`,
  display-normalized values in `normalized_depth`, and observed `min_depth` /
  `max_depth`

See the [Depth Anything 3 project page](https://depth-anything-3.github.io/) for
model variants and accuracy metrics.
