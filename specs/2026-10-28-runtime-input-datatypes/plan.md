# Plan

1. Preserve per-input runtime datatype tags in ModelInfo independently of PixelType.
2. Pack detector size buffers using that metadata; retain image normalization and dimension order.
3. Propagate all runtime datatype tags through neuriplo-infer for local and KServe metadata.
4. Add regressions covering multiple detector families, mixed input types, batch sizes and invalid metadata.
5. Validate candidate commits with real models across the applicable backend matrix before any consolidated release.
