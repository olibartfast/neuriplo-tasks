# Runtime input datatypes

Detector preprocessing must preserve the runtime model contract for each input independently. Image and size inputs may use different scalar types. Metadata from local runtimes or KServe must reach the task without narrowing auxiliary INT64 tensors to image pixel types.

RT-DETR, DEIM and EdgeCrafter must retain their existing image preprocessing and size-value conventions. Size, token ID and attention-mask buffer packing must use the declared runtime datatype, including INT32, INT64, FP32 and boolean masks. Missing or unsupported explicit metadata must fail rather than guessing from a model name. Existing callers using only input_types remain supported according to their declared type.
