# Detector gaps — validation

> **This section was written after the implementation**, unlike every packet
> that follows. The workflow's guarantee is that success is defined before the
> work, so that "done" cannot be shaped to fit whatever got built; that
> guarantee does not apply here. What follows is an honest record of checks
> actually run, not a contract that was met.

## Checks

| # | Check | Result |
|---|-------|--------|
| 1 | New unit tests fail against the pre-fix sources | **pass** |
| 2 | Full test suite | **pass** — 28/28 suites |
| 3 | `yolov10n` end to end | **pass** |
| 4 | `rtdetrv4` end to end | **pass** |
| 5 | `deimv2` end to end | **pass** |
| 6 | No behaviour change for models that already worked | **pass** |
| 7 | Local gate (format, cppcheck, `-DWERROR`, ctest, Valgrind) | **pass** |
| 8 | CI on the pull request | **pass** — 8/8 checks |
| 9 | EdgeCrafter preprocessing | **not executed** |
| 10 | Accuracy measured against a labelled dataset | **not executed** |

## Detail

**1 — the tests fail without the fix.** A test written after a fix tends to
pass against the broken code too. Each new test was run with the `src/` changes
stashed and the tests kept, and each failed; then restored, and each passed.
This applies to `YoloNmsFreeAcceptsPixelCoordinates`,
`YoloNmsFreeKeepsConventionsSeparate`, and the four new factory aliases.

**3 — `yolov10n`.** Previously zero usable detections: raw maximum coordinate
640.348 on a 640×640 input, multiplied by 640, placing every box off-frame.
Now decoded unscaled and correctly placed.

**4 — `rtdetrv4`.** Previously ~5 detections, all misplaced. Now a full,
correctly placed set on the same image. The change was in preprocessing —
removing ImageNet mean/std — with the decoder untouched.

**5 — `deimv2`.** Previously `Unrecognized model type: dfine` from the factory.
Now constructs and runs; this model had never executed through the library.

**6 — no regression.** The convention check is a read-only scan that leaves the
normalized path exactly as it was, which is what protects the case release
0.3.2 fixed. The preprocessor split preserves EdgeCrafter's behaviour
bit-for-bit and changes only RT-DETR and D-FINE, both of which were wrong.
Covered by the existing suites passing unchanged.

**8 — CI.** Format Check, clang-tidy, cppcheck, Build & Test, Build with Strict
Warnings, Valgrind, Windows MSVC, Branch Policy. Two failed on the first
attempt and neither was a defect in the change: clang-format-18 rejected two
lines a newer local clang-format had left alone, and Branch Policy evaluated a
stale base ref while the pull request was being retargeted from `master` to
`develop`. Both green after the reformat and retarget.

**9 — EdgeCrafter is untested, not verified.** No EdgeCrafter model was
available, so `EdgeCrafterPreprocessor` was created to *preserve* the existing
normalization rather than to assert it is right. If EdgeCrafter's true
preprocessing is not ImageNet-normalized, that bug survives this change
untouched. Recorded in [roadmap.md](../roadmap.md).

**10 — no accuracy numbers.** Every end-to-end check is structural and visual:
detections exist, are placed on the objects, and are stable across frames. This
says the decoding is right, not that mAP is what upstream reports. Measuring
that needs a labelled dataset and an evaluation harness, neither of which this
repository has.

## Promoted to the constitution

Durable findings do not stay in this folder — see
[specs/README.md](../README.md#promotion-what-must-not-stay-in-a-packet):

- routing is two matchers and the factory runs first → [tech-stack.md](../tech-stack.md)
- NMS-free exports use both coordinate conventions → [tech-stack.md](../tech-stack.md)
- RT-DETR family takes no ImageNet statistics; RF-DETR and EdgeCrafter do
  → [tech-stack.md](../tech-stack.md)
