# TPU-servable TensorFlow export - validation

Defined before implementation. TPU checks are manual or scheduled; normal CI
must remain TPU-free.

## Automated

- [x] [V-1] -> [R-1], [R-10]: Existing TensorFlow exporter test passes:
      `python tests/test_tensorflow_export.py`.
- [x] [V-2] -> [R-10]: Full repository test suite passes:
      `cmake -S . -B build -DBUILD_TESTS=ON`
      then `cmake --build build --parallel` and
      `ctest --test-dir build --output-on-failure`.
- [x] [V-3] -> [R-2], [R-3], [R-6]: Helper CPU metadata mode reports the
      expected input and output names, dtypes, ranks, and shapes without a TPU.
- [ ] [V-4] -> [R-7]: If an exporter change was needed, the default export
      still produces the existing raw-BGR/logits contract and its regression
      test passes.
- [ ] [V-5] -> [R-8]: No public header, task type, result type, or routing
      change; diff audit confirms the packet touches only `export/`, `tests/`,
      and documentation.

## Manual

- [ ] [M-1] -> [R-4], [R-5]: On a Cloud TPU VM, run the helper in TPU mode and
      record TPU type, TensorFlow version, `libtpu` version, command, and the
      top-5 classes/scores.
- [ ] [M-2] -> [R-5]: Run the CPU reference on the same artifact and input;
      compare top-5 classes and scores. Record the differences and tolerance.
- [ ] [M-3] -> [R-9]: Read the updated export guide as a new user would. Confirm
      it states that TPU is remote serving, not an in-process backend.
- [ ] [M-4] -> [R-9]: Verify all added relative links resolve and absolute links
      return HTTP 200.

## Acceptance Commands

```bash
# CPU metadata and existing regression
python tests/test_tensorflow_export.py
python export/classification/tensorflow/verify_tpu_saved_model.py \
  --saved-model /tmp/tf-regate-resnet50 \
  --image /path/to/dog.jpg \
  --mode metadata

# CPU reference
python export/classification/tensorflow/verify_tpu_saved_model.py \
  --saved-model /tmp/tf-regate-resnet50 \
  --image /path/to/dog.jpg \
  --mode cpu

# Cloud TPU VM only
python export/classification/tensorflow/verify_tpu_saved_model.py \
  --saved-model /tmp/tf-regate-resnet50 \
  --image /path/to/dog.jpg \
  --mode tpu
```

The helper lives at `export/classification/tensorflow/verify_tpu_saved_model.py`.
The image path is caller-supplied; `neuriplo-tasks` does not ship `data/dog.jpg`. Record
the final command and image hash in the evidence table.

## Evidence

| ID | Command/check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| V-1 | Exporter regression | pass (1/1) | 2026-11-09 | `/tmp/opencode/tf-tasks-venv/bin/python tests/test_tensorflow_export.py` -> `Ran 1 test ... OK` (TF 2.19.1). Exporter untouched. |
| V-2 | CTest suite | pass (31/31) | 2026-11-09 | `cmake -S . -B build -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release`, `cmake --build build --parallel`, `ctest --test-dir build --output-on-failure`: 100% passed, 0 failed out of 31. |
| V-3 | CPU metadata | pass | 2026-11-09 | With TF 2.19.1: `tests/test_verify_tpu_saved_model.py` -> `Ran 12 tests ... OK` (TF-gated cases executed). On TF-free host 2026-11-09: `Ran 37 tests ... OK (skipped=4)` — pure/ErrorPath/MockedTPU pass, 4 TF-gated skip. Metadata against the real tiny export: `verify_tpu_saved_model.py --saved-model /tmp/opencode/tiny-tf-sm --image /tmp/opencode/tiny.png --mode metadata --image-height 4 --image-width 6 --num-classes 2` -> input `image`, output `output_0`, both `float32`, ranks 4 and 2, shapes `[None, 4, 6, 3]` / `[None, 2]`. The default 224/1000 invocation correctly rejects this 4x6/2-class fixture with a shape-mismatch error. |
| V-4 | Default export preserved | not executed | 2026-11-09 | Conditional opt-in exporter path was not added; no TPU probe was possible on this host (Q-1 unresolved). Exporter unchanged and V-1 covers the default contract. |
| M-1 | TPU execution | not executed | - | No TPU hardware on this host; needs a Cloud TPU VM. |
| M-2 | CPU/TPU comparison | not executed | - | Needs M-1. |

## Deviations

- V-1/V-3 now run under `/tmp/opencode/tf-tasks-venv/bin/python` (TF 2.19.1);
  the earlier "no TF on host" entries are superseded.
- V-3 used a real tiny SavedModel with a 4x6 input and 2 classes, so the
  metadata check was run with explicit `--image-height/--image-width/
  --num-classes`; the default 224/1000 shape is validated by behavior (it
  rejects the mismatch) rather than by a full ResNet-50 export on this host.
- `Pillow` is a decode-only dependency, now listed in
  `export/classification/tensorflow/requirements.txt`. Resampling is
  implemented locally as half-pixel-center bilinear to match the C++
  `image_ops::resize(..., Interpolation::Linear)` path; Pillow's antialiased
  resampler is deliberately not used, so CPU/TPU preprocessing matches the task.
- The TPU execution path is architecturally corrected (single input replicated,
  execution inside `TPUStrategy`, explicit failure with no silent CPU fallback)
  but remains unverified on TPU hardware.
- `worker`/`reviewer` subagents unavailable (subagent depth limit); implemented
  and self-reviewed directly per handoff fallback.
- New helper tests: 37 tests total; 12 TF-gated executed with TF 2.19.1, all
  pass; on TF-free hosts the 4 TF-gated cases skip and the remaining 33 pass.
- No CHANGELOG entry (plan T-14: entry follows evidence; no TPU evidence on
  this host).
