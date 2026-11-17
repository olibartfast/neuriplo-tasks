## YOLOv8/YOLOv10/YOLO11/YOLOv12/YOLO26
* Install  [following Ultralytics official documentation (pip ultralytics package version >= 8.4.0)](https://docs.ultralytics.com/tasks/segment/) and export the model in different formats, you can use the following commands:

#### OnnxRuntime/Torchscript/Openvino

```
yolo export model=yolov8/yolo11/yolo12 n/s/m/x-seg.pt format=onnx/torchscript/openvino
```

#### GGUF (GGML backend)

Convert the exported ONNX to GGUF weights for the neuriplo GGML backend
(`GGMLInfer`, strict-GGUF-only loader) with the converter from neuriplo
(`scripts/convert_onnx_to_gguf.py`; requires Python 3 with `onnx` and `numpy`):

```bash
python3 /path/to/neuriplo/scripts/convert_onnx_to_gguf.py \
    --input /tmp/yolov8n-seg.onnx --output /tmp/yolov8n-seg.gguf
```

The converter keeps F32 tensors of rank 1..4 (other initializers are skipped
with a note) and matches tensors by exact name; a loadable YOLO-seg file must
contain the fixed DFL conv weight (`model.22.dfl.conv.weight`). Verified
conversion (fixed batch 1):

| Model | ONNX source | Tensors | `.gguf` bytes | Input |
|-------|-------------|---------|---------------|-------|
| YOLOv8n-seg | ultralytics `yolov8n-seg` (opset 12, static 640) | 158 | 13,804,864 | 1,3,640,640 |

---

## YOLOv5
#### OnnxRuntime/Torchscript
* Run from [yolov5 repo](https://github.com/ultralytics/yolov5) export script:  ```python export.py  --weights <yolov5seg_version>.pt  --include [onnx,torchscript]```

---


## RF-DETR

### ONNX Export for Instance Segmentation

For instance segmentation, use the export script from neuriplo-tasks:

```bash
# Clone neuriplo-tasks if not already available
git clone https://github.com/olibartfast/neuriplo-tasks.git
cd neuriplo-tasks/export/segmentation/rfdetr

python export.py --simplify --input_size 432
```



**Available Options:**
- `--output_dir`: Path to save exported model (default: current directory)
- `--opset_version`: ONNX opset version (default: 17)
- `--simplify`: Simplify ONNX model using onnxsim
- `--batch_size`: Batch size for export (default: 1)
- `--input_size`: Input image size (default: 640)

#### Using Python API

```python
from rfdetr import RFDETRSegPreview

model = RFDETRSegPreview(pretrain_weights=<CHECKPOINT_PATH>)

model.export(
    opset_version=17,
    simplify=True,
    batch_size=1
)
```

**Model Outputs:**
- `dets`: Bounding boxes `[batch, num_queries, 4]` in cxcywh format (normalized)
- `labels`: Class logits `[batch, num_queries, num_classes]`
- `masks`: Segmentation masks `[batch, num_queries, mask_h, mask_w]` (e.g., 108x108)

This command saves the ONNX segmentation model to the `output` directory.


### GGUF (GGML backend)

Convert the exported ONNX to GGUF weights for the neuriplo GGML backend
(`GGMLInfer`, strict-GGUF-only loader) with the converter from neuriplo
(`scripts/convert_onnx_to_gguf.py`; requires Python 3 with `onnx` and `numpy`):

```bash
python3 /path/to/neuriplo/scripts/convert_onnx_to_gguf.py \
    --input /tmp/rfdetr-seg.onnx --output /tmp/rfdetr-seg.gguf
```

The converter keeps F32 tensors of rank 1..4 (other initializers are skipped
with a note) and matches tensors by exact name. A loadable RF-DETR-seg file
must contain the 91-class head (`class_embed.bias` `[91]` plus a 2D `[256,91]`
class weight) and `segmentation_head.bias`, which keys the seg path (mask
branch) in the loader. Verified conversion (fixed batch 1):

| Model | ONNX source | Tensors | `.gguf` bytes | Input |
|-------|-------------|---------|---------------|-------|
| RF-DETR-seg | RF-DETR seg (`export/segmentation/rfdetr/export.py`) | 436 | 123,514,688 | 1,3,432,432 |

### TensorRT Export

For GPU deployment, you can convert the ONNX model to TensorRT format for optimized performance.

### Detection or Segmentation Models

```bash
trtexec --onnx=/path/to/model.onnx \
        --saveEngine=/path/to/model.engine \
        --memPoolSize=workspace:4096 \
        --fp16 \
        --useCudaGraph \
        --useSpinWait \
        --warmUp=500 \
        --avgRuns=1000 \
        --duration=10
```

### Using TensorRT Docker Container

```bash
export NGC_TAG_VERSION=25.06

docker run --rm -it --gpus=all \
    -v $(pwd)/exports:/exports \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $(pwd)/model.onnx:/workspace/model.onnx \
    -w /workspace \
    nvcr.io/nvidia/tensorrt:${NGC_TAG_VERSION}-py3 \
    /bin/bash -cx "trtexec --onnx=model.onnx \
                            --saveEngine=/exports/model.engine \
                            --memPoolSize=workspace:4096 \
                            --fp16 \
                            --useCudaGraph \
                            --useSpinWait \
                            --warmUp=500 \
                            --avgRuns=1000 \
                            --duration=10"
```
