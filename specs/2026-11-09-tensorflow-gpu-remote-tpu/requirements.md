# TPU-servable TensorFlow export - requirements

**Date:** 2026-11-09
**Branch:** `feature/tensorflow-gpu-remote-tpu` from `develop`
**Status:** Implemented; pending TPU evidence
**Cross-repo packet:** `2026-11-09-tensorflow-gpu-remote-tpu` in `neuriplo`,
`neuriplo-infer`, and `neuriplo-platform`

## Goal

Make the existing TensorFlow classifier artifact provably usable by the remote
TPU serving path chosen for this program. The TPU service is an external KServe
V2 process; this repository owns only the model export and its validation
recipe. No C++ task contract changes are needed.

The first representative is the ResNet-50 TensorFlow classifier: raw Float32
NHWC BGR pixels in `[0, 255]`, logits out. Detection and segmentation exports
are follow-ups.

## In Scope

- A repeatable TPU validation helper for the existing TensorFlow SavedModel,
  under `export/classification/tensorflow/`.
- CPU-side metadata checks that run without a TPU: input/output names, dtype,
  rank, and shape.
- A TPU-side execution check that loads the SavedModel, runs it through a
  TPUStrategy scope, and compares top-5 classes and scores with the CPU
  reference.
- An opt-in exporter change only if the probe proves the current artifact cannot
  be run by the TPU runtime.
- Documentation in `export/classification/TensorflowClassification.md`,
  including exact commands, artifact naming, expected output, and known limits.
- Python tests for the new helper and any exporter change.

## Out of Scope

- Changes to `include/neuriplo/tasks/`, `src/`, task routing, preprocessing,
  postprocessing, result types, or the C++ library.
- StableHLO or PJRT export; that is not the selected TPU path.
- The TPU serving service itself. It lives outside this repository and speaks
  KServe V2 to `neuriplo-infer`.
- TPU training, multi-host TPU, and non-Google accelerators.
- GPU export or GPU validation. TensorFlow GPU is owned by `neuriplo`.
- Detection, segmentation, pose, or video TPU exports in this packet.

## Decisions

- [D-1] Preserve the existing SavedModel contract. The task still receives raw
  BGR pixels and returns logits; TPU serving must not add a second preprocessing
  convention.
- [D-2] Prefer a validation helper over a new exporter. Add an export mode only
  if the TPU probe shows the existing artifact cannot be loaded or executed.
- [D-3] Keep TPU-only execution out of normal CI. CPU metadata checks run in CI;
  the TPU run is manual or scheduled and recorded as attested evidence.
- [D-4] Keep TPU Python dependencies isolated under `export/`, not in the C++
  library or the task runtime.
- [D-5] The helper must fail clearly when `tensorflow` or the TPU runtime is
  missing; it must not silently fall back to CPU and report a TPU pass.

## Assumptions

- [A-1] A Cloud TPU VM with a TensorFlow runtime compatible with the existing
  SavedModel can execute the classifier through `TPUStrategy`.
- [A-2] The existing `export_tf_saved_model_classifier.py` artifact has no
  op that the TPU runtime rejects. Verify during the probe.
- [A-3] The TPU service can expose the same logits over KServe V2 without a
  task-layer change.

## Open Questions

- [Q-1] Does the current SavedModel run under `TPUStrategy` unchanged? If no,
  which part fails: unsupported op, variable restoration, or shape handling?
  The answer selects between validation-only and an exporter change.
- [Q-2] Should the TPU helper live in `export/classification/tensorflow/` or a
  shared `export/tpu/` directory? Recommendation: keep it beside the exporter
  until a second model family needs it.

## Requirements

- [R-1] The existing TensorFlow classifier export path must remain unchanged
  unless [Q-1] proves a TPU-specific export is required.
- [R-2] A TPU validation helper must load the SavedModel and verify the input
  signature is Float32 NHWC with the classifier's image shape.
- [R-3] The helper must verify the output signature is Float32 logits with the
  classifier's class dimension.
- [R-4] On a TPU VM, the helper must execute the model through a TPUStrategy
  scope and return top-5 classes and scores for a caller-supplied image. Record
  the exact image path and hash used for the comparison.
- [R-5] The TPU result must be compared with a CPU reference from the same
  artifact and input. Record the class and score differences; do not claim
  bitwise parity.
- [R-6] The helper must support a CPU-only metadata mode that runs in normal CI
  without TensorFlow TPU packages.
- [R-7] If an exporter change is needed, it must be opt-in, preserve the
  existing default output, and have a regression test for the default path.
- [R-8] No public C++ header, task type, result type, or routing string may
  change.
- [R-9] Documentation must state that TPU is a remote serving path, name the
  required runtime and artifact, and include the exact validation commands.
- [R-10] Existing TensorFlow export tests and the full `neuriplo-tasks` CTest
  suite must stay green.

## Contract

The artifact contract remains:

```text
input:  image, Float32, NHWC, [1, 224, 224, 3], BGR pixels in [0, 255]
output: logits, Float32, [1, 1000]
```

The TPU service is responsible for transport and device placement. The task
library is responsible only for the preprocessing and postprocessing it already
performs on the client side.
