# Keypoint Matching ONNX Export

Export SuperGlue-family models to ONNX for the C++ `KeypointMatchingTask`.

> **Status: scaffold.** The C++ task routes these families, but decoding is not
> implemented yet — decoders land per family once a real export's output layout
> has been probed. This guide documents the export paths, the contracts seen in
> the wild, and how to verify a file before writing its decoder.

## Export shapes in the wild

There are two shapes, and only one fits this library's contract
(`preprocess(images) → tensors`, `postprocess(tensors) → Result`):

| Family | Export path | Inputs | Outputs | Fits the task? |
|--------|-------------|--------|---------|----------------|
| SuperGlue (original matcher) | [jadehh/SuperPoint-SuperGlue-ONNX](https://github.com/jadehh/SuperPoint-SuperGlue-ONNX), magicleap export scripts | `keypoints_0`, `scores_0`, `descriptors_0`, `keypoints_1`, `scores_1`, `descriptors_1` | `matches0 [1,N0]`, `matching_scores0 [1,N0]` | **No** — keypoints and descriptors go in, not images. Needs a separate front-end orchestration design. |
| LightGlue pipeline | [fabio-sim/LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX) `lightglue-onnx export` | interleaved pair `(2B, 1, H, W)` (SuperPoint extractor; 3-channel variants for other extractors) | `keypoints (2B, K, 2)`, `matches (M, 3)` (`[batch, idx_left, idx_right]`), `match_scores (M,)` | **Yes** — images in, matches out. |
| MatchAnything | HF [`zju-community/matchanything_eloftr`](https://huggingface.co/zju-community/matchanything_eloftr) (`AutoModelForKeypointMatching`); exportable with Optimum | `pixel_values`-style pair input; contract not pinned | `keypoints0` / `keypoints1` / `matching_scores` after HF postprocessing | **Likely** — verify the ONNX graph before implementing. |

The current C++ preprocessing is shared across families: resize to the model
input size, BGR → RGB, scale to `[0,1]`, NCHW float32. Whether a family needs
grayscale, `mean/std 0.5` normalization, or `[-1,1]` keypoint-space rescaling is
**open until probed** — see `specs/2026-10-15-keypoint-matching/requirements.md`
(open questions 1–3).

## Exporting a LightGlue pipeline

```bash
# Requires uv (https://docs.astral.sh/uv/) and a PyTorch install; see the
# LightGlue-ONNX README for extractor choices (superpoint, disk, aliked, ...).
uv run lightglue-onnx export superpoint \
  --num-keypoints 1024 \
  -b 2 -h 1024 -w 1024 \
  -o weights/superpoint_lightglue_pipeline.onnx
```

`-b 2` is the **pair** batch axis: the model expects the source and target
images interleaved on the batch dimension (`[left, right]`), not a plain pair of
separate inputs.

## Probing a file before implementing its decoder

Do not write a decoder from documentation alone. Print the graph contract first
and record it in the spec packet:

```python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
print("inputs:", [(i.name, i.shape, i.type) for i in session.get_inputs()])
print("outputs:", [(o.name, o.shape, o.type) for o in session.get_outputs()])
```

Then run one real image pair through the session and check, per output tensor:

- the dtype and shape (fixed vs dynamic `K`, `M`);
- what the values mean — indices vs coordinates, and which axis is the pair;
- the coordinate space of keypoints (model input pixels vs original image
  pixels) — a resize mismatch here is the silent-wrong-result failure mode this
  repo has been bitten by before;
- whether the graph expects normalized or un-normalized pixels.

Only then implement the family postprocessor in
`src/keypoint_matching/` and add its values to the C++ tests.

## Deploying

Exported files run in any ONNX-capable engine (ONNX Runtime, TensorRT,
OpenVINO, Triton). Feed `ModelInfo.input_shapes` / `input_formats` /
`input_names` from the probe above, and create the task with the family name
(`"superglue"`, `"lightglue"`, `"matchanything"`, or any prefixed revision).
