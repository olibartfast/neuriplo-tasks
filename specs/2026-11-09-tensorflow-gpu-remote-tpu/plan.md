# TPU-servable TensorFlow export - plan

**Packet:** `2026-11-09-tensorflow-gpu-remote-tpu`
**Branch:** `feature/tensorflow-gpu-remote-tpu`

## Group 0 - Probe the existing artifact

- [T-1] Export the existing ResNet-50 classifier SavedModel with
  `export_tf_saved_model_classifier.py`.
- [T-2] On a Cloud TPU VM, load the artifact with
  `tf.saved_model.load()` and run one inference inside a `TPUStrategy` scope.
- [T-3] Record the exact runtime, `libtpu` version, TPU type, and the failure
  point if the artifact is not directly usable. This decides whether the
  packet is validation-only or needs an exporter change.

## Group 1 - Validation helper

- [T-4] Add a CPU metadata mode that checks input/output names, dtypes, ranks,
  and shapes without a TPU.
- [T-5] Add a TPU execution mode that runs the same preprocessing as the C++
  task, executes through `TPUStrategy`, and prints top-5 classes and scores.
- [T-6] Add a CPU reference mode that runs the same artifact on CPU and prints
  the same top-5 data for comparison.
- [T-7] Make missing TPU runtime or missing artifact an explicit error, not a
  silent CPU fallback.

## Group 2 - Conditional exporter change

- [T-8] If [Q-1] proves the existing artifact cannot run on TPU, add an opt-in
  exporter path that produces a TPU-compatible SavedModel while preserving the
  current default export.
- [T-9] Add a regression test that the default exporter still produces the
  existing raw-BGR/logits contract.
- [T-10] If the existing artifact already runs, do not add an export mode; record
  the probe result and keep the change to validation and documentation.

## Group 3 - Tests and documentation

- [T-11] Add Python tests for the helper's metadata mode and error paths.
- [T-12] Update `export/classification/TensorflowClassification.md` with the
  TPU section, commands, expected top-5 values, and limits.
- [T-13] Update `export/classification/tensorflow/requirements.txt` only if the
  TPU probe needs an additional pinned package. Keep it isolated from the C++
  library.
- [T-14] Add the changelog entry after evidence exists, not before.

## Acceptance

The packet is complete when the CPU metadata check runs in CI, the TPU run is
recorded with exact hardware and commands, and the artifact contract is
unchanged. If the TPU run cannot be executed, the packet remains specified and
the unexecuted check is recorded as such.
