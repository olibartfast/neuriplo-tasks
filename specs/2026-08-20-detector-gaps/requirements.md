# Detector gaps — requirements

**Date:** 2026-08-20
**Branch:** `fix/detector-gaps` → `develop`
**Pull request:** [#10](https://github.com/olibartfast/neuriplo-tasks/pull/10)

> **Written after the code.** This packet is a retroactive record. The fixes
> were made first, while diagnosing a downstream problem in `neuriplo-track`,
> and the packet was written afterwards to bring this repository under the
> spec-driven workflow. Its validation section therefore does **not** carry the
> guarantee the workflow normally provides — that success was defined before
> anything was built. It is marked accordingly, and the next packet here starts
> the proper way round.

## Goal

Three model families that the README claims are supported did not work through
the library. Make each one produce correct detections from its real exported
ONNX file.

## Context

Found while producing demo footage for
[neuriplo-track](https://github.com/olibartfast/neuriplo-track): `yolov10n`
returned nothing usable, so the demos fell back to YOLOv12s. Chasing that
uncovered two more failures in the same corner of the code. All three were
reproduced against actual model files in `~/model_repository`, not inferred
from reading the source.

| Model type | Symptom |
|------------|---------|
| `yolov10n` | zero usable detections — every box off-frame |
| `rtdetrv4` | ~5 detections, all misplaced |
| `deimv2` | throws `Unrecognized model type: dfine` before inference |

## The three causes

### 1. NMS-free YOLO coordinate convention

`[1, 300, 6]` output was assumed to be normalized and multiplied by the input
size. Probing the file showed a maximum coordinate of **640.348** on a 640×640
input: this export emits **input-space pixels**. Multiplying pushed every box
far outside the frame.

The assumption was not arbitrary — release 0.3.2 *added* that multiplication to
fix the opposite bug, where normalized output landed at negative coordinates.
Both fixes were correct about the file in front of them. **Both conventions
exist in the wild for the same tensor shape**, so neither assumption can be
right in general; the convention has to be established from the data.

### 2. RT-DETR family normalization

Misplaced `rtdetrv4` boxes came from **pre**processing, not decoding.
`RtDetrPreprocessor` applied ImageNet mean/std subtraction. RT-DETR, RT-DETRv2,
D-FINE, and DEIM take `images` scaled to `[0,1]` and nothing else — the extra
statistics shifted the input distribution, and the model reported what it was
shown. RF-DETR and EdgeCrafter, which shared the preprocessor, *do* want
ImageNet statistics.

### 3. Factory routing rejects D-FINE and DEIM

`ObjectDetectionTask::detectModelType` already mapped `dfine` and `deim` to
`RT_DETR_STYLE`, and the README documented them as supported — but
`TaskFactory::createTaskInstance` did not list them, and it runs first. The
name was rejected before the task that understood it was ever constructed.

## In scope

- Detect the NMS-free coordinate convention per tensor rather than assuming it.
- Split preprocessing so the RT-DETR family gets `[0,1]` scaling and the
  ImageNet-statistics families keep theirs.
- Route `rtdetrv2`, `dfine`, `deim`, and later DEIM revisions through the
  factory.
- Update the README `TASKFACTORY_MODEL_LIST` block, which
  `test_readme_model_types` enforces.
- Unit tests that fail against the pre-fix sources.

## Out of scope

- Any change to task contracts, result schema, tensor shapes, or existing model
  behaviour that was already correct.
- New model families, new tasks, or export scripts.
- Restructuring the preprocessor hierarchy beyond the split these fixes force.

## Decisions

1. **Detect the coordinate convention, do not switch on model type.** The
   convention varies by *export*, not by architecture — the same model type
   produces both. A per-tensor check is the only thing that is true for both
   files. Switching on model type would have re-broken the 0.3.2 case.

2. **Threshold at 1.5** (`kNormalizedCoordinateLimit`), applied to the maximum
   of all four coordinates over all detections. Normalized output is bounded by
   1 plus float noise; real pixel boxes on a 640×640 (or smaller) input do not
   have their *largest* coordinate below 1.5 — that would be a box under 0.25%
   of the frame width, in an export that also gave every other detection the
   same. Scanning all detections rather than the first makes a single degenerate
   box unable to flip the decision.

3. **Split the preprocessor; do not parameterize it.** `EdgeCrafterPreprocessor`
   is a new class that keeps ImageNet normalization; `RtDetrPreprocessor` and
   `DFinePreprocessor` drop it. A boolean flag would have kept the two families
   one argument apart, which is how they came to be confused in the first place.

4. **EdgeCrafter behaviour is preserved, not verified.** No EdgeCrafter model
   was available. Splitting the class keeps its current behaviour bit-for-bit
   rather than guessing what is right, and the untested normalization is
   recorded in [roadmap.md](../roadmap.md) as a known gap rather than left
   silent.

5. **Match DEIM and D-FINE by prefix**, not by equality, so `deimv2` and future
   revisions resolve without another patch. This is what the reported bug
   actually was: an alias one character away from a supported name.
