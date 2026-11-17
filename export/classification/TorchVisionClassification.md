## Instructions

### Export the model for the inference
## Models from Torchvision Pytorch API

Use the provided export script for Torchvision models:

```bash
# Clone neuriplo-tasks if not already available
git clone https://github.com/olibartfast/neuriplo-tasks.git
cd neuriplo-tasks/export/classification/torchvision

# Export ResNet50 to ONNX
python export_torchvision_classifier.py --library torchvision --model resnet50 --export_format onnx --output_onnx resnet50.onnx

# Or export to TorchScript
python export_torchvision_classifier.py --library torchvision --model resnet50 --export_format torchscript --output_script resnet50.pt
```

Alternatively, you can manually export models from https://pytorch.org/vision/stable/models.html#classification:

#### OnnxRuntime
 ```python
import torch
import torchvision.models as models
import argparse
import os
resnet50 = models.resnet50(pretrained=True)
dummy_input = torch.randn(1, 3, 224, 224)
resnet50 = resnet50.eval()

torch.onnx.export(resnet50,
                    dummy_input,
                    args.save,
                    export_params=True,
                    opset_version=10,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size', 2: "height", 3: 'width'},
                                'output': {0: 'batch_size'}})
  ```

### Batch inference (`N > 1`)

- Export with dynamic batch on **input and output** (see `dynamic_axes` above) so the engine can run `[N,3,H,W]` → `[N,num_classes]`.
- In neuriplo-tasks, preprocess one buffer per image (`batchPreprocess`), stack for inference, then pass output shape `[N,C]` to `batchPostprocess` with `batch_size = N`. Postprocess returns **one top-1 class per batch index** when `N > 1` (full `top_k` applies only when `N = 1`).

 #### TensorRT
 Once you have your exported onnx model, using trtexec:
 ```bash
   trtexec --onnx=model.onnx --saveEngine=model.plan --explicitBatch --minShapes=input:1x3x224x224 --optShapes=input:1x3x224x224 --maxShapes=input:256x3x224x224

  ```

  * Pay attention to use tensorrt version corresponding to the one used on Triton image, or you could use a container like below (supposing you are using Triton release 23.08)
   ```bash
  docker run -it --gpus=all -v $(pwd):/workspace nvcr.io/nvidia/pytorch:23.08-py3 /bin/bash -cx \
   "trtexec --onnx=model.onnx --saveEngine=model.plan --explicitBatch --minShapes=input:1x3x224x224 --optShapes=input:1x3x224x224 --maxShapes=input:256x3x224x224 --fp16
  ```

  #### GGUF (GGML backend)
  Convert the exported ONNX to GGUF weights for the neuriplo GGML backend
  (`GGMLInfer`, strict-GGUF-only loader) with the converter from neuriplo
  (`scripts/convert_onnx_to_gguf.py`; requires Python 3 with `onnx` and `numpy`):

  ```bash
  python3 /path/to/neuriplo/scripts/convert_onnx_to_gguf.py \
      --input resnet50.onnx --output resnet50.gguf
  ```

  The converter keeps F32 tensors of rank 1..4 (other initializers are
  skipped with a note) and matches tensors by exact name. A loadable
  classifier file carries one 2D `[1000,512]`/`[1000,2048]` weight plus its
  `[1000]` bias. Verified conversions (fixed batch 1):

  | Model | ONNX source | Tensors | `.gguf` bytes | Input |
  |-------|-------------|---------|---------------|-------|
  | ResNet-18 | torchvision `resnet18` (`resnet18-v1-7.onnx`) | 102 | 46,810,304 | 1,3,224,224 |
  | ResNet-50 | torchvision `resnet50` (`resnet50-v1-12.onnx`) | 299 | 102,556,256 | 1,3,224,224 |
