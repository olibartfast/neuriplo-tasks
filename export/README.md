# Vision-Core Export Tools

This directory contains generic model export utilities for various computer vision tasks. These tools are designed to work with any inference engine, not just Triton.

## Directory Structure

```
export/
├── classification/
│   ├── tensorflow/         # TensorFlow classifier export
│   ├── torchvision/        # Torchvision/timm classifier export
│   └── vit/                # Vision Transformer export
├── detection/
│   ├── yolo/               # Universal YOLO export (v5-v12, NAS)
│   ├── rtdetr/             # RT-DETR detection export
│   └── rfdetr/             # RF-DETR detection export
├── segmentation/
│   └── rfdetr/             # RF-DETR segmentation export
├── optical_flow/
│   └── raft/               # RAFT optical flow export
└── video_classification/
    └── timesformer/        # TimeSFormer video classification export
```

## Usage

Each subdirectory contains:
- Export scripts (.py)
- Setup/environment scripts (.sh)
- Requirements files (requirements.txt)
- Usage documentation

### Quick Start Examples

```bash
cd /path/to/vision-core

# Export YOLO model
python export/detection/yolo/export.py --model yolov8n.pt --format onnx

# Export Vision Transformer
python export/classification/vit/onnx/export.py

# Export RAFT optical flow
python export/optical_flow/raft/raft_exporter.py --model-type large --format onnx

# Export TimeSFormer
python export/video_classification/timesformer/export_timesformer_to_onnx.py
```

## Supported Export Formats

- **ONNX**: Cross-platform inference
- **TorchScript**: PyTorch native format
- **TensorRT**: NVIDIA GPU optimized (via ONNX conversion)
- **SavedModel**: TensorFlow native format

## Integration with Inference Engines

These exported models can be used with:
- Triton Inference Server
- ONNX Runtime
- TensorRT
- OpenVINO
- Custom inference implementations

For Triton-specific deployment tools, see [tritonic](https://github.com/olibartfast/tritonic/tree/master/deploy) repository.
