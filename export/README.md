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
│   ├── rfdetr/             # RF-DETR detection export
│   └── edgecrafter/        # EdgeCrafter detection export
├── segmentation/
│   ├── rfdetr/             # RF-DETR segmentation export
│   └── edgecrafter/        # EdgeCrafter segmentation export
├── optical_flow/
│   └── raft/               # RAFT optical flow export
├── open_vocab_detection/
│   └── owlv2/              # OWLv2 open-vocabulary detection export
├── pose_estimation/
│   ├── yolo_pose/          # YOLO pose export (v5/v8/v11/v26)
│   ├── vitpose/            # ViTPose pose estimation export
│   └── edgecrafter/        # EdgeCrafter pose estimation export
├── depth_estimation/
│   └── depth_anything_v2/  # Depth Anything V2 depth export
├── gaussian_splatting/
│   └── lgm/                # LGM / GRM Gaussian Splatting export
├── video_classification/
│   ├── videomae/           # VideoMAE video classification export
│   ├── vivit/              # ViViT video classification export
│   └── timesformer/        # TimeSFormer video classification export
└── image_understanding/    # VLM model download (GGUF, no ONNX export)
```

## Usage

Each subdirectory contains task-specific documentation and, where applicable, export scripts.

Most tasks export to ONNX. The `image_understanding/` task is an exception — it uses GGUF models downloaded from Hugging Face and runs via the llama.cpp backend (no ONNX export step).

### Quick Start Examples

```bash
cd /path/to/neuriplo-tasks

# Export YOLO model
python export/detection/yolo/export.py --model yolov8n.pt --format onnx

# Export Vision Transformer
python export/classification/vit/onnx/export.py

# Export RAFT optical flow
python export/optical_flow/raft/raft_exporter.py --model-type large --format onnx

# Export YOLO pose (YOLOv8/YOLO11/YOLO26)
python export/pose_estimation/yolo_pose/export.py --model yolov8n-pose.pt --format onnx

# Export ViTPose
python export/pose_estimation/vitpose/export_vitpose_to_onnx.py

# Export Depth Anything V2
python export/depth_estimation/depth_anything_v2/export_depth_anything_v2_to_onnx.py

# Export OWLv2
python export/open_vocab_detection/owlv2/export_owlv2_to_onnx.py --test

# Export LGM Gaussian Splatting
python export/gaussian_splatting/lgm/export_lgm_to_onnx.py

# Export VideoMAE
python export/video_classification/videomae/export_videomae_to_onnx.py

# Export ViViT
python export/video_classification/vivit/export_vivit_to_onnx.py

# Export TimeSFormer
python export/video_classification/timesformer/export_timesformer_to_onnx.py

# Image Understanding (VLM) — download GGUF models, no export step needed
# See export/image_understanding/ImageUnderstanding.md for wget commands
```

## Batched inference (dynamic batch axis)

When exporting ONNX with a dynamic or fixed batch dimension on axis 0, align
`ModelInfo.input_shapes` / `max_batch_size_` with the engine config and use
neuriplo-tasks batch helpers for preprocess/postprocess split.

| Topic | Document |
|-------|----------|
| End-to-end consumer flow (`N=2` images, engine vs library) | [docs/batch_processing.md](../docs/batch_processing.md) |
| Per-task-family batch readiness | [docs/batch_support_matrix.md](../docs/batch_support_matrix.md) |
| Classification `[N,C]` output | [classification/TorchVisionClassification.md](classification/TorchVisionClassification.md) |
| YOLO `[N, 4+C, anchors]` output | [detection/ObjectDetection.md](detection/ObjectDetection.md) |

## Export Integration with Inference Engines

These exported models can be used with:
- Triton Inference Server
- ONNX Runtime
- TensorRT
- OpenVINO
- Custom inference implementations

Refer to:
* [Object Detection](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/detection/ObjectDetection.md)
* [Classification](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/classification/Classification.md)

* [Instance Segmentation](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/segmentation/InstanceSegmentation.md)
* [Optical Flow](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/optical_flow/OpticalFlow.md)
* [Pose Estimation](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/pose_estimation/PoseEstimation.md)
* [Depth Estimation](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/depth_estimation/DepthEstimation.md)
* [Open-Vocabulary Detection](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/open_vocab_detection/OWLv2.md)
* [Video Classification](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/video_classification/VideoClassification.md)
* [Image Understanding (VLM)](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/image_understanding/ImageUnderstanding.md)
* [EdgeCrafter Detection](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/detection/edgecrafter/README.md)
* [EdgeCrafter Segmentation](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/segmentation/edgecrafter/README.md)
* [EdgeCrafter Pose Estimation](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/pose_estimation/edgecrafter/README.md)


For Triton-specific deployment tools, see [tritonic](https://github.com/olibartfast/tritonic/tree/master/deploy) repository.
