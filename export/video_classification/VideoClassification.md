# Video Classification Export

Export tools for HuggingFace video classification models to ONNX format.

## Supported Models

| Model | HuggingFace Class | Default Frames | Default Model |
|-------|-------------------|----------------|---------------|
| VideoMAE | `VideoMAEForVideoClassification` | 16 | `MCG-NJU/videomae-base-finetuned-kinetics` |
| ViViT | `VivitForVideoClassification` | 32 | `google/vivit-b-16x2-kinetics400` |
| TimeSformer | `TimesformerForVideoClassification` | 8 | `facebook/timesformer-base-finetuned-k400` |

All models produce a 5D input tensor with shape `[batch, num_frames, 3, height, width]` and output logits `[batch, num_classes]`.

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
