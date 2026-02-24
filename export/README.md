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
├── pose_estimation/
│   └── vitpose/            # ViTPose pose estimation export
├── depth_estimation/
│   └── depth_anything_v2/  # Depth Anything V2 depth export
└── video_classification/
    ├── videomae/           # VideoMAE video classification export
    ├── vivit/              # ViViT video classification export
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

# Export ViTPose
python export/pose_estimation/vitpose/export_vitpose_to_onnx.py

# Export Depth Anything V2
python export/depth_estimation/depth_anything_v2/export_depth_anything_v2_to_onnx.py

# Export VideoMAE
python export/video_classification/videomae/export_videomae_to_onnx.py

# Export ViViT
python export/video_classification/vivit/export_vivit_to_onnx.py

# Export TimeSFormer
python export/video_classification/timesformer/export_timesformer_to_onnx.py
```

## Export Integration with Inference Engines

These exported models can be used with:
- Triton Inference Server
- ONNX Runtime
- TensorRT
- OpenVINO
- Custom inference implementations

Refer to:
* [Object Detection](https://github.com/olibartfast/vision-core/blob/master/export/detection/ObjectDetection.md)
* [Classification](https://github.com/olibartfast/vision-core/blob/master/export/classification/Classification.md)

* [Instance Segmentation](https://github.com/olibartfast/vision-core/blob/master/export/segmentation/InstanceSegmentation.md)
* [Optical Flow](https://github.com/olibartfast/vision-core/blob/master/export/optical_flow/OpticalFlow.md)
* [Pose Estimation](https://github.com/olibartfast/vision-core/blob/master/export/pose_estimation/PoseEstimation.md)
* [Depth Estimation](https://github.com/olibartfast/vision-core/blob/master/export/depth_estimation/DepthEstimation.md)
* [Video Classification](https://github.com/olibartfast/vision-core/blob/master/export/video_classification/VideoClassification.md)


For Triton-specific deployment tools, see [tritonic](https://github.com/olibartfast/tritonic/tree/master/deploy) repository.
