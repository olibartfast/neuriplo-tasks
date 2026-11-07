# OpenCV-compatible vector MatMul export

The maintainer requests solving blocked backend coverage cells. RT-DETRv4 ONNX uses vector-right-hand MatMul, legal in ONNX but rejected by OpenCV 5.
R1: Provide an opt-in offline export adapter replacing constant-vector-right-hand MatMul with Unsqueeze/MatMul/Squeeze. Preserve model input/output names, dtypes, shapes and the mathematical operation. Do not modify source weights or files in place.
R2: Leave other nodes untouched, use collision-free intermediate names, and reject unsupported ONNX opsets/domains rather than guessing.
R3: Compare original/adapted graphs in ORT and adapted graph in OpenCV on identical real task inputs. Test the pass on a small deterministic model. No task routing, C++ inference logic or result schema changes.
