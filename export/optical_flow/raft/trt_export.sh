#!/bin/bash

export NGC_TAG_VERSION=25.12

export MODEL_ONNX_DIR=$HOME/model_repository/raft_large_onnx/1
export MODEL_TRT_DIR=$HOME/model_repository/raft_large_trt/1
export MIN_SHAPES="input1:1x3x256x256,input2:1x3x256x256"
export OPT_SHAPES="input1:1x3x520x960,input2:1x3x520x960"
export MAX_SHAPES="input1:1x3x1080x1920,input2:1x3x1080x1920"

docker run --rm -it --gpus=all \
    -v $MODEL_ONNX_DIR:/workspace/input \
    -v $MODEL_TRT_DIR:/workspace/output \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -w /workspace \
    nvcr.io/nvidia/tensorrt:${NGC_TAG_VERSION}-py3 \
    /bin/bash -c "trtexec --onnx=/workspace/input/model.onnx \
        --saveEngine=/workspace/output/model.plan \
        --memPoolSize=workspace:4096 \
        --fp16 \
        --useCudaGraph \
        --useSpinWait \
        --warmUp=500 \
        --avgRuns=1000 \
        --duration=10 \
        --minShapes=$MIN_SHAPES \
        --optShapes=$OPT_SHAPES \
        --maxShapes=$MAX_SHAPES"
