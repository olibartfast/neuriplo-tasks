# Image Understanding (VLM) Model Setup

Unlike other task types in `neuriplo-tasks`, the `ImageUnderstanding` task does **not** use ONNX export.
Models are downloaded as GGUF files and run via the [llama.cpp](https://github.com/ggerganov/llama.cpp) backend.

## Supported Models

| Model | HuggingFace repo | Notes |
|-------|-----------------|-------|
| Gemma 4 E2B IT | `unsloth/gemma-4-E2B-it-GGUF` | 2B params, edge/mobile |
| Gemma 4 E4B IT | `unsloth/gemma-4-E4B-it-GGUF` | 4B params, recommended |
| Gemma 4 E12B IT | `unsloth/gemma-4-E12B-it-GGUF` | 12B params, high quality |

Each model requires **two** GGUF files:
1. **Language model** — e.g. `gemma-4-E4B-it-Q4_K_M.gguf`
2. **Vision projector (mmproj)** — e.g. `mmproj-gemma4-E4B-F16.gguf`

## Download Instructions

### Gemma 4 E4B IT (recommended)

```bash
mkdir -p models/gemma4
# Language model
wget -O models/gemma4/gemma-4-E4B-it-Q4_K_M.gguf \
    "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf"

# Vision projector
wget -O models/gemma4/mmproj-gemma4-E4B-F16.gguf \
    "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/mmproj-gemma4-E4B-F16.gguf"
```

### Gemma 4 E2B IT (smallest)

```bash
mkdir -p models/gemma4
wget -O models/gemma4/gemma-4-E2B-it-Q4_K_M.gguf \
    "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf"

wget -O models/gemma4/mmproj-gemma4-E2B-F16.gguf \
    "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/mmproj-gemma4-E2B-F16.gguf"
```

## Passing the Model Path to neuriplo-infer

The `model_path` field encodes both GGUFs using the `|mmproj=` suffix convention:

```
<language-model-path>|mmproj=<projector-path>
```

Example:

```bash
./neuriplo-infer \
    --model_type gemma4 \
    --model_path "models/gemma4/gemma-4-E4B-it-Q4_K_M.gguf|mmproj=models/gemma4/mmproj-gemma4-E4B-F16.gguf" \
    --image input.jpg \
    --prompt "Describe this image."
```

## Two-Tensor Input Contract

`ImageUnderstandingTask::preprocess()` returns a two-element vector:

| Index | Content |
|-------|---------|
| `[0]` | UTF-8 encoded prompt bytes |
| `[1]` | 8-byte header `[uint32 width LE][uint32 height LE]` followed by `H×W×3` raw RGB bytes |

When no image is provided (text-only mode) only `[0]` is returned.

The llama.cpp backend (`LlamaCppInfer`) decodes both tensors and feeds them to `mtmd_image_tokens_get` / `llama_decode`.

## Backend Requirement

This task requires the `LLAMACPP` backend built with:

```cmake
cmake -Bbuild -H. \
    -DDEFAULT_BACKEND=LLAMACPP \
    -DLLAMACPP_DIR=/path/to/llamacpp/install
```

The install must expose `libllama.so`, `libmtmd.so`, and the `ggml` shared libraries.
See [neuriplo/cmake/LinkBackend.cmake](https://github.com/olibartfast/neuriplo/blob/master/cmake/LinkBackend.cmake) for the exact link flags.

## Context Length

Gemma 4 multimodal inference requires `n_ctx >= 8192`. If the context is too small the model will produce empty or truncated output.
