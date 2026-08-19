# Mission

*Living document — describes the project as it is now. If it disagrees with the
code, this file is wrong and should be fixed.*

## Who this is for

Authors of inference-engine projects that need the pixels-in / results-out half
of computer vision without adopting a particular runtime:

- [tritonic](https://github.com/olibartfast/tritonic) — Triton client
- [neuriplo-infer](https://github.com/olibartfast/neuriplo-infer) — multi-backend inference
- [neuriplo-track](https://github.com/olibartfast/neuriplo-track) — multi-object tracking

## The problem

An inference engine is mostly not inference. Letterboxing, channel order,
normalization statistics, anchor decoding, NMS, mask assembly, keypoint
rescaling — this is where model families differ, and it is the part every
consumer reimplements. Reimplemented per repository and per backend, it drifts:
the same ONNX file gives different boxes in two projects, and the difference is
a normalization constant nobody wrote down.

## What this is

A framework-agnostic C++17 static library holding that logic once, behind a
contract stable enough to depend on:

- `TaskInterface` — `preprocess(vector<Image>)` and
  `postprocess(Size, vector<Tensor>)`, in framework-neutral types.
- `TaskFactory` — a model-type string picks the task and its postprocessor.
  Supporting a new model family is a routing entry plus a postprocessor, not a
  change at every call site.
- `Result` — one variant schema across detection, segmentation, pose, depth,
  classification, optical flow, splatting, and VLM output.

The library never calls an inference runtime. Tensors go in, results come out;
what produced the tensors is the consumer's business.

## What this deliberately is not

- **Not an inference engine.** No ONNX Runtime, Triton, TensorRT, or LibTorch
  calls. Consumers own the session.
- **Not a model zoo.** `export/` holds Python export scripts; weights are never
  committed.
- **Not a plugin system.** The task registry is a compile-time table in
  `task_factory.cpp`. Runtime or third-party task registration is out of scope
  unless it is added as a product requirement — and then as a separate
  extension registry, not by growing that table without limit.
- **Not an OpenCV wrapper.** OpenCV is an optional interop adapter
  (`vision-opencv`, default OFF), never a required dependency.
- **Not a training or evaluation framework.**

## What success looks like

- A consumer switches model family by changing a string, and the result schema
  does not move under it.
- A preprocessing or decoding bug is fixed once, here, and every consumer gets
  the fix by moving a pinned tag.
- Task contracts and result schemas stay backward compatible across minor
  releases; breaking either is a deliberate, reviewed act — see
  `preserve_task_contracts` and `preserve_result_schema` in
  [REPO_META.yaml](../REPO_META.yaml).
