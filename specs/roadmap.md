# Roadmap

*Living document — what shipped, what is in flight, what is proposed. Updated
when a packet merges, not when one is imagined.*

## Shipped

Releases are cut from `master` and recorded in [CHANGELOG.md](../CHANGELOG.md),
which stays the authority on the detail. This is the shape of the arc:

| Release | Theme |
|---------|-------|
| 0.1.0 – 0.2.0 | Detection, segmentation, classification, pose; `TaskInterface` / `TaskFactory` established |
| 0.3.0 | Breadth: image understanding (VLM), Gaussian splatting, Grounding DINO; README ↔ factory contract test |
| 0.3.1 – 0.3.2 | Routing and README drift closed; NMS-free coordinate scaling |
| 0.4.0 | Rename `vision-core` → `neuriplo-tasks`; batch preprocess/postprocess; task pipelines; Valgrind in CI |
| 0.4.1 | Class-aware NMS for NMS-free detection |
| 0.5.0 | RF-DETR keypoint pose |
| 0.6.0 – 0.6.1 | **OpenCV removed as a dependency** — native `vision::Image`, optional `vision-stb` and `vision-opencv` adapters |
| 0.7.0 | Framework-neutral polygon output for instance segmentation |
| 0.8.0 | YOLO26 depth estimation; vision hot-loop fast paths |
| 0.8.1 | `vision::decodeImage` for in-memory encoded images; [detector gaps](2026-08-20-detector-gaps/requirements.md) — YOLO NMS-free coordinate convention, RT-DETR-family normalization, D-FINE/DEIM factory routing |

## In flight

Nothing on `develop` beyond the last release.

## Proposed

Candidates, **not commitments** — nothing here is scheduled, and the list is
the maintainer's to set.

- **EdgeCrafter preprocessing has no test.** `EdgeCrafterPreprocessor` was split
  off from `RtDetrPreprocessor` with its ImageNet normalization preserved,
  because no EdgeCrafter model was available to check it against. Its
  normalization is asserted by nobody. Obtaining a model and pinning the
  behaviour would close the last gap left by 2026-08-20-detector-gaps.
- **A detector-convention conformance suite.** Two coordinate-convention bugs
  (0.3.2, 0.8.1) and one normalization bug reached releases because per-family
  input and output conventions are encoded only in the code that implements
  them. A table-driven test over the families the README claims would catch the
  third instance before a consumer does.
- **Preprocessor ownership by family, not by shape.** RT-DETR, D-FINE, DEIM,
  RF-DETR, and EdgeCrafter share tensor shapes but not normalization. Sharing
  one preprocessor across them is what allowed the RT-DETR bug; the split has
  begun, and finishing it would make the mistake unrepresentable.

## How work gets scheduled

Pick an item, open a packet under `specs/YYYY-MM-DD-name/`, and follow
[specs/README.md](README.md). A packet that merges updates this file in the
same branch — otherwise the roadmap describes intentions rather than history.
