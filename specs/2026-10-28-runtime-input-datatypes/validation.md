# Validation

The INT32 detector regression reproduced the previous fixed-INT64 packing failure before the change. The complete WERROR-enabled CTest suite passes: 29/29 entries. Tests cover six detector families with independent FP32/INT32/INT64 size metadata, exact values and buffer lengths, batched EdgeCrafter dimensions, invalid metadata and overflow. Open-vocabulary regressions compare INT64 reference tokens/masks with independently packed INT32 tokens and BOOL masks. Existing image-only behavior remains covered.

Pending: consumer metadata propagation tests and real-output validation across the applicable backend/model matrix. Unit success alone is not release readiness.
