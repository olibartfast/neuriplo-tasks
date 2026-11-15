# GGUF export for the GGML backend

This directory documents the ONNX → GGUF step consumed by the
[neuriplo](https://github.com/olibartfast/neuriplo) GGML backend
(`GGMLInfer`). The backend is strict-GGUF-only: it loads `.gguf` weights and
rejects any other extension. The converter lives in neuriplo
(`scripts/convert_onnx_to_gguf.py`); this guide records the contract and the
models it covers. Produce the input ONNX with the per-task exporters in this
repo first (torchvision classifier, ultralytics YOLO).

## Converter

```bash
python3 /path/to/neuriplo/scripts/convert_onnx_to_gguf.py \
    --input /path/to/model.onnx \
    --output /path/to/model.gguf
```

Requires Python 3 with `onnx` and `numpy`. The script verifies its output
with an independent read-back parse before exiting.

## Weight contract

- F32 tensors only. FLOAT16/DOUBLE initializers are widened to F32 with a
  note; non-float initializers are skipped with a note.
- Rank 1..4 only; rank-0 and rank->4 tensors are skipped with a note.
- Dims stored in ggml order (reversed ONNX dims), 32-byte aligned data
  section, GGUF v3.
- One `neuriplo.rank.<tensor-name>` int32 KV entry per tensor carries the
  ONNX rank (the GGUF C API exposes no rank accessor). The loader rejects
  files without these entries instead of guessing.
- Tensors are matched by exact name. The minimum loadable head is either the
  YOLO fixed DFL conv weight (`model.22.dfl.conv.weight`) or one 2D
  `[1000,512]`/`[1000,2048]` classifier weight plus its `[1000]` bias.

## Supported models

Fixed geometries only (batch 1). Numbers are from the verified conversion
runs (neuriplo GGUF exporter).

| Model | ONNX source | Tensors | `.gguf` bytes | Input |
|-------|-------------|---------|---------------|-------|
| ResNet-18 classifier | torchvision `resnet18` (`resnet18-v1-7.onnx`, see `export/classification/`) | 102 | 46,810,304 | 1,3,224,224 |
| ResNet-50 classifier | torchvision `resnet50` (`resnet50-v1-12.onnx`, see `export/classification/`) | 299 | 102,556,256 | 1,3,224,224 |
| YOLOv8n detection | ultralytics `yolov8n` (opset 12, static 640; see `export/detection/yolo/`) | 132 | 12,791,904 | 1,3,640,640 |
| YOLOv8n-seg | ultralytics `yolov8n-seg` (opset 12, static 640; see `export/detection/yolo/`) | 158 | 13,804,864 | 1,3,640,640 |

## Running the backend

`neuriplo-infer` gates GGUF models with
`scripts/ggml_app_gate.py` (defaults to the GGUF converter):

```bash
python3 scripts/ggml_app_gate.py \
    --onnx /tmp/yolov8n.onnx \
    --model-type yolov8 \
    --source data/dog.jpg \
    --input-sizes 1,3,640,640 \
    --labels labels/coco.names
```

## Limitations

- YOLO26 is blocked at the converter: its `model.23` head uses attention
  ops and lacks the fixed DFL conv weight the GGUF contract requires.
- Transformer models (RT-DETR family, ViT, RF-DETR, OWLv2) need loader-side
  graph support (multi-input, INT64) that the GGML backend does not have;
  no GGUF export is claimed for them.
