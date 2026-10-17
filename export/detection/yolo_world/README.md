# YOLO-World (Fixed Vocabulary) ONNX Export

Export an [Ultralytics YOLO-World](https://docs.ultralytics.com/models/yolo-world/) model with a
class list baked in. The text embeddings are frozen into the graph, so the exported model has the
standard YOLOv8 detection output and runs through the existing YOLO postprocessor.

This is **not** runtime open-vocabulary detection: changing the class list means exporting again.
For prompts at inference time, use OWLv2 or Grounding DINO
(see [Open-Vocabulary Detection](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/open_vocab_detection/OWLv2.md)).

## Prerequisites

```bash
pip install ultralytics onnx onnxruntime onnxslim
```

`set_classes` uses CLIP to encode the class names; Ultralytics installs it on first use.
Only the `-worldv2` checkpoints (`yolov8s/m/l/x-worldv2.pt`) support ONNX export. Weights are
downloaded automatically.

## Export

```bash
git clone https://github.com/olibartfast/neuriplo-tasks.git
cd neuriplo-tasks/export/detection/yolo_world

# Inline vocabulary, checked with ONNX Runtime
python export.py --model yolov8s-worldv2.pt --classes person bus "traffic light" --test

# Vocabulary from a file (one class per line)
python export.py --model yolov8m-worldv2.pt --classes-file my_classes.txt --output-dir ./exported_models
```

Outputs, in `--output-dir`:

- `<model>.onnx` — the exported detector
- `<model>_classes.txt` — the class names, one per line, in `class_id` order. Use it as the labels
  file for your inference app.

Options: `--imgsz` (default 640), `--opset`, `--dynamic` (dynamic batch axis), `--no-simplify`.

## Using the model in neuriplo-tasks

- **Model type**: `yoloworld` (any string starting with `yolo` that does not contain `seg`,
  `pose`, or `depth` works, e.g. `yolov8s-worldv2`)
- **Input**: `images` `[1, 3, 640, 640]`, RGB, `[0, 1]`, letterboxed — same as YOLOv8
- **Output**: `output0` `[1, 4 + num_classes, 8400]` (cx, cy, w, h + one score per class, no objectness)

```cpp
auto task = TaskFactory::createTaskInstance("yoloworld", model_info);
```

## TensorRT

```bash
trtexec --onnx=yolov8s-worldv2.onnx --saveEngine=yolov8s-worldv2.engine --fp16
```
