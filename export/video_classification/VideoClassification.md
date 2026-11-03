# Video Classification Export

Export tools for HuggingFace video classification models to ONNX format.

## Supported Models

| Model | HuggingFace Class | Default Frames | Default Model |
|-------|-------------------|----------------|---------------|
| VideoMAE | `VideoMAEForVideoClassification` | 16 | `MCG-NJU/videomae-base-finetuned-kinetics` |
| ViViT | `VivitForVideoClassification` | 32 | `google/vivit-b-16x2-kinetics400` |
| TimeSformer | `TimesformerForVideoClassification` | 8 | `facebook/timesformer-base-finetuned-k400` |
| V-JEPA 2 | `VJEPA2ForVideoClassification` | 16 (fpc16) / 32 / 64 | `facebook/vjepa2-vitl-fpc16-256-ssv2` |
| VideoPrism | `VideoPrismForVideoClassification` | 16 (base) / 8 (large) | `google/videoprism-base-f16r288` |
| PE Video | `PeVideoModel` + baked text embeddings | 8–16 | `facebook/pe-av-large` |

All models produce a 5D input tensor with shape `[batch, num_frames, 3, height, width]` and output logits `[batch, num_classes]` — except a raw PE Video export, which emits video/text embeddings (see its section).

## Usage

### VideoMAE

```bash
# Default export
python export/video_classification/videomae/export_videomae_to_onnx.py

# Custom model and settings
python export/video_classification/videomae/export_videomae_to_onnx.py \
    --model MCG-NJU/videomae-base-finetuned-kinetics \
    --output videomae.onnx \
    --num-frames 16 \
    --image-size 224 \
    --test
```

### ViViT

```bash
# Default export
python export/video_classification/vivit/export_vivit_to_onnx.py

# Custom model and settings
python export/video_classification/vivit/export_vivit_to_onnx.py \
    --model google/vivit-b-16x2-kinetics400 \
    --output vivit.onnx \
    --num-frames 32 \
    --image-size 224 \
    --test
```

### TimeSformer

```bash
# Default export
python export/video_classification/timesformer/export_timesformer_to_onnx.py

# Custom model and settings
python export/video_classification/timesformer/export_timesformer_to_onnx.py \
    --model facebook/timesformer-base-finetuned-k400 \
    --output timesformer.onnx \
    --num-frames 8 \
    --image-size 224 \
    --test
```

## Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | HuggingFace model name | Model-specific |
| `--output` | Output ONNX file path | Model-specific |
| `--num-frames` | Number of input frames | Model-specific |
| `--image-size` | Frame height and width | 224 |
| `--opset` | ONNX opset version | 14 |
| `--static-batch` | Disable dynamic batch size | Off |
| `--test` | Run ONNX Runtime inference test | Off |

### V-JEPA 2

```bash
# Pre-exported and verified on the Hub:
#   onnx-community/vjepa2-vitl-fpc32-256-diving48-ONNX
optimum-cli export onnx --model facebook/vjepa2-vitl-fpc16-256-ssv2 vjepa2_onnx/
```

Contract: `pixel_values_videos [1, T, 3, 256, 256]` float32 → `logits [1, num_labels]`, where
`T` is the checkpoint's `frames_per_clip` (16/32/64). The 384-crop checkpoints resize the shortest
edge to `384 * 256 / 224 = 438` before cropping. Route with any name starting with `vjepa`.

### VideoPrism

```bash
# google/videoprism-* publishes Flax weights only; export a PyTorch conversion, e.g.
optimum-cli export onnx --model MHRDYN7/videoprism-base-f16r288-finetuned-ucf101 videoprism_onnx/
```

Contract: `pixel_values_videos [1, 16, 3, 288, 288]` float32 → `logits [1, num_labels]`. The
processor resizes straight to 288x288 with **bicubic** sampling, rescales to `[0, 1]`, and applies
**no** mean/std. Large checkpoints take 8 frames instead of 16.

### PE Video

PE Video is a video–text contrastive encoder: `PeVideoModel` emits `logits_video_text` for a batch
of videos against a batch of texts, so a plain export cannot be driven by `preprocess(images)`.
Export a **fixed vocabulary** graph instead — embed the label list once and bake the text
embeddings as initializers, as the [YOLO-World fixed-vocabulary recipe](../detection/yolo_world/)
does — so the graph takes `pixel_values_videos [1, T, 3, 336, 336]` and emits
`logits [1, num_labels]`. Video processor settings: 336x336 direct resize, `1/255`, mean/std `0.5`.
A raw `PeVideoModel` export (embeddings out, text ids in) is not supported by this task.

## Requirements

```
torch
onnx
numpy
transformers
onnxruntime  # optional, for --test
```

## Preprocessing (neuriplo-tasks C++)

Each model uses a different preprocessing pipeline in the C++ inference code:

| Model | Resize | Crop | Rescale | Normalization |
|-------|--------|------|---------|---------------|
| VideoMAE | Direct 224x224 | None | 1/255 | mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] |
| ViViT | Shortest edge 256 | Center 224x224 | 1/127.5, offset -1 | mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] |
| TimeSformer | Shortest edge 224 | Center 224x224 | 1/255 | mean=[0.45,0.45,0.45], std=[0.225,0.225,0.225] |
| V-JEPA 2 | Shortest edge `crop*256/224` (292 for 256) | Center crop | 1/255 | mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] |
| VideoPrism | Direct 288x288, **bicubic** | None | 1/255 | **None** (processor sets `do_normalize=False`) |
| PE Video | Direct 336x336 | None | 1/255 | mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5] |

Frames per clip come from the model: V-JEPA 2 exposes `frames_per_clip` (16/32/64), VideoPrism 16 for base checkpoints and 8 for large, PE Video 8–16. When the ONNX input has a fixed time axis, `VideoClassificationTask` reads it from the 5D shape (`getRequiredFrames()`), so the C++ side needs no per-checkpoint setting.
