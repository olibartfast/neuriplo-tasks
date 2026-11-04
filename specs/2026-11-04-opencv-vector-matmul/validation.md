# Validation before implementation

- Original versus adapted ONNX Runtime outputs on deterministic inputs, including batched MatMul and reused initializers.
- Check graph input/output contracts and collision avoidance, reject in-place output and old opsets.
- Original RT-DETRv4 fails OpenCV MatMul rank assertion; adapted graph must run and agree with original ORT outputs on identical dog.jpg tensors.
- Downstream Slurm job preserves hashes and render evidence; no release pin changes.

## Results (2026-11-04)

- Three Python tests pass: matrix/batched/vector input parity via the ONNX
  reference evaluator, shared initializer preservation, name collisions,
  idempotence, overridable initializer exclusion, and rejection of opset 12.
- Actual RT-DETRv4 model: three vector MatMul nodes adapted. Original versus
  adapted ORT maxima: labels 0, boxes 0.000091553 pixels, scores 0.000000954.
  Mathematically equivalent graph forms can select different FP kernels.
- Adapted OpenCV 5 versus original ORT: selected labels/scores agree; complete
  output error metrics retained in neuriplo-infer's evidence JSON. Renders
  inspected. The original file still fails OpenCV's vector MatMul shape check.
- Tasks CTest: 31/31 pass. No C++ task code, aliases, input/output contracts or
  model inventory changed; no generated README inventory change is needed.
- Slurm parallel rerun is pending because the existing controller went offline.
  Local artifacts: /tmp/blocked-slurm-20261104/dnn/models/rtdetrv4-opencv.onnx;
  references: /tmp/blocked-rtdetrv4-ort-in*.bin and corresponding output files.

Reproduce the real export:

```bash
python export/detection/rtdetr/prepare_opencv_onnx.py original.onnx prepared.onnx
python export/detection/rtdetr/test_prepare_opencv_onnx.py
```

Use the downstream `rtdetr` CLI type for prepared v4 weights, as in the existing
preset. `rtdetrv4` is a preset name, not a new TaskFactory alias.
