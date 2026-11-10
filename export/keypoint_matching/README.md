# Keypoint Matching ONNX Export

Export SuperGlue-family models to ONNX for the C++ `KeypointMatchingTask`.

> **Status: SuperGlue, LightGlue and FlexGlue decoded, MatchAnything pending.**
> All three decoders are implemented and verified against real exports (see
> below). The MatchAnything postprocessor still throws `std::logic_error`
> until its export is probed the same way.

## Export shapes in the wild

There are two shapes, and only one fits this library's contract
(`preprocess(images) → tensors`, `postprocess(tensors) → Result`):

| Family | Export path | Inputs | Outputs | Fits the task? |
|--------|-------------|--------|---------|----------------|
| SuperGlue (transformers / Optimum) | `optimum-cli export onnx` of `magic-leap-community/superglue_outdoor` (or `_indoor`); pre-exported and verified at [KnightsAnalytics/superglue_outdoor](https://huggingface.co/KnightsAnalytics/superglue_outdoor) | `pixel_values [1, 2, 3, 480, 640]` — pair on axis 1 | `keypoints`, `matches`, `matching_scores`, `mask` | **Yes** — images in, matches out. **Decoded.** |
| SuperGlue (original matcher) | [jadehh/SuperPoint-SuperGlue-ONNX](https://github.com/jadehh/SuperPoint-SuperGlue-ONNX), magicleap export scripts | `keypoints_0`, `scores_0`, `descriptors_0`, `keypoints_1`, `scores_1`, `descriptors_1` | `matches0 [1,N0]`, `matching_scores0 [1,N0]` | **No** — keypoints and descriptors go in, not images. Needs a separate front-end orchestration design. |
| LightGlue pipeline | [fabio-sim/LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX) `lightglue-onnx export`; pre-exported and verified at [thomasonzhou/superpoint-lightglue](https://huggingface.co/thomasonzhou/superpoint-lightglue) | interleaved pair `images [2, C, H, W]` (SuperPoint: grayscale; 3-channel for ALIKED/DISK) | `keypoints [2, K, 2]` int64, `matches [M, 3]`, `mscores [M]` | **Yes** — images in, matches out. **Decoded.** |
| MatchAnything | HF [`zju-community/matchanything_eloftr`](https://huggingface.co/zju-community/matchanything_eloftr) (`AutoModelForKeypointMatching`); exportable with Optimum | `pixel_values`-style pair input; contract not pinned | `keypoints0` / `keypoints1` / `matching_scores` after HF postprocessing | **Likely** — verify the ONNX graph before implementing. |
| FlexGlue sparse matcher | [cvg/FlexGlue](https://github.com/cvg/FlexGlue) v0.1 release (`flexglue_descriptor.onnx` + `flexglue_matcher.onnx`) | descriptor: `image [1, 3, 512, 512]` RGB `[0, 1]` per image; matcher: `features0/1 [1, 256, 64, 64]` + `kpts0/1 [1, N, 2]` (corner pixels) + `image_size0/1 [1, 2]` + scalar gates | `matches0 [1, N0]` int64 (`-1` unmatched), `mscores0 [1, N0]`, `covisibility_score [1]` | **Yes, with forwarding** — images in via the descriptor, externally detected keypoints forwarded with the matcher outputs. **Decoded.** |

Preprocessing is per family, because the exports disagree: SuperGlue wants a
single rank-5 pair tensor of grayscale values; LightGlue's SuperPoint pipeline
wants a `(2B, 1, H, W)` grayscale stack of its own; FlexGlue wants one RGB
`[0, 1]` buffer per image (the descriptor runs once per image); MatchAnything
is unpinned. `KeypointMatchingPreprocessor` emits one buffer per image for the
two-input families, and `SuperGluePreprocessor` overrides it with a single
packed buffer.

## Exporting SuperGlue (transformers / Optimum)

```bash
uv pip install "transformers>=4.48" "optimum[exporters]" torch torchvision
optimum-cli export onnx \
  --model magic-leap-community/superglue_outdoor \
  superglue_outdoor_onnx/
```

Verified against `KnightsAnalytics/superglue_outdoor` (transformers 5.20, opset 17).
The export is one graph — SuperPoint extractor and SuperGlue matcher fused — with:

| Tensor | Shape | Meaning |
|--------|-------|---------|
| `pixel_values` (in) | `[1, 2, 3, 480, 640]` float32 | pair on axis 1; grayscale replicated to 3 channels, `[0, 1]` |
| `keypoints` | `[1, 2, N, 2]` float32 | normalized `x / width`, `y / height`; `N` is dynamic (568–929 observed) |
| `matches` | `[1, 2, N]` int64 | for side 0, the side-1 keypoint index it matches, or `-1`; side 1 is the inverse |
| `matching_scores` | `[1, 2, N]` float32 | mutual-match probability |
| `mask` | `[1, 2, N]` int64 | `1` = detected keypoint, `0` = padding |

Preprocessing (what `SuperGluePreprocessor` emits): resize each image to
`640x480` bilinear, BGR → RGB, RGB → grayscale with torchvision's weights
(`0.2989, 0.5870, 0.1140`), scale to `[0, 1]`, replicate to three channels, and
pack the pair into the single input tensor. No mean/std normalization.

Decoding (what `SuperGluePostprocessor` does): drop keypoints where `mask == 0`,
scale the rest by the original `(width, height)` of their image, take side-0
matches with `mask[1][k] > 0`, and keep only scores above
`TaskConfig::confidence_threshold`. The export does not include the SuperPoint
detector score, so `Keypoint::confidence` stays `0`.

**Verification run (2026-11-02)** on the magicleap phototourism Capitol pair
(`1016x762` / `1048x626`): `N=903`, `795` / `903` valid keypoints, `233` matches.
The C++ decoder reproduced the Python `post_process_keypoint_matching` result on
all 233 matches (max coordinate/score delta `1.2e-6`). Preprocessed pixels match
the transformers pipeline to `mean |diff| = 0.004`, `[0,1]` scale — the residual
is PIL's antialiased bilinear downscale versus this library's plain bilinear.

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

**Verified contract** (`thomasonzhou/superpoint-lightglue`, 51 MB, 512x512 run):

| Tensor | Shape | Meaning |
|--------|-------|---------|
| `images` (in) | `[2, 1, H, W]` float32 | grayscale pair on the batch axis, `[0, 1]`, no mean/std |
| `keypoints` | `[2, K, 2]` int64 | pixel coordinates **in the model input space** (not normalized) |
| `matches` | `[M, 3]` int64 | rows of `[pair_index, source_index, target_index]`; pair index 0 applies to the task's single pair |
| `mscores` | `[M]` float32 | match score |

`LightGluePreprocessor` packs the pair into the single input and takes the
channel count from the model input shape (1 for SuperPoint, 3 for ALIKED/DISK).
`LightGluePostprocessor` rescales keypoints by `original / model input` per axis,
which is why it is constructed with the model input size.

**Verification run (2026-11-03)** on the magicleap Capitol pair (1016x762 /
1048x626 at 512x512 input): the C++ buffer fed to the export produced 107
matches (111 from a PIL-preprocessed buffer; the difference is the resize
kernel); above the default 0.25 threshold both yield 63. Decoding the same
tensors in C++ and Python gives identical rows (max score/coordinate delta
`5e-7`).

## Running a FlexGlue sparse matcher

No export step is needed: download the v0.1 release (SHA-pinned in
`onnx/match_pair.py` upstream) and run the two-model flow. Keypoints come from
any detector — the example uses OpenCV SIFT with the +0.5 corner offset
FlexGlue expects (LightGlue detectors need `origin="corner"`):

```bash
python probe_flexglue.py --descriptor flexglue_descriptor.onnx \
  --matcher flexglue_matcher.onnx --source a.jpg --target b.jpg
```

**Verified contract** (v0.1 release, berlin pair `1100x757` / `960x640`):

| Tensor | Shape | Meaning |
|--------|-------|---------|
| `image` (descriptor in) | `[1, 3, 512, 512]` float32 | RGB `[0, 1]`, squashed bilinear, no antialias |
| `features` (descriptor out) | `[1, 256, 64, 64]` float32 | dense map, once per image |
| `kpts0/1` (matcher in) | `[1, N, 2]` float32 | corner-origin pixels of the original images; both images need ≥ 1 keypoint |
| `image_size0/1` | `[1, 2]` float32 | `(w, h)` of the originals |
| `matches0` | `[1, N0]` int64 | index into kpts1, or `-1`; N0 equals the kpts0 count |
| `mscores0` | `[1, N0]` float32 | match confidence, `0.0` where unmatched |
| `covisibility_score` | `[1]` float32 | pair overlap probability |

`FlexGluePreprocessor` emits the two descriptor inputs (RGB, squashed to the
model size from `ModelInfo`). `FlexGluePostprocessor` takes the matcher
outputs **plus the forwarded `kpts0/kpts1` tensors** (in order), passes the
keypoints through unscaled, and keeps matches with scores above
`TaskConfig::confidence_threshold`. `min_covisibility` gating already happened
inside the graph; `max_distance_px = 0` rejects all matches in ONNX (unlike
PyTorch, where 0 disables the check).

**Verification run (2026-11-10)** on the upstream berlin pair (785 / 727 SIFT
keypoints): 5 valid matches at `filter_threshold=0.1`, covisibility 0.9932,
all match indices in range, unmatched scores `0.0`. The C++ test suite embeds
an excerpt of this run (`FlexGlueDecodesVerifiedExportValues`).

## Probing a file before implementing its decoder

Do not write a decoder from documentation alone. `probe_superglue.py` in this
directory does it for SuperGlue-style exports:

```bash
python probe_superglue.py --model superglue_outdoor_onnx/model.onnx \
  --source a.jpg --target b.jpg
```

It prints the graph contract plus the values the decoder depends on: coordinate
normalization range, mask packing, and the index space of `matches` (which
side's keypoints it indexes). Then run one real image pair through the session
and check, per output tensor:

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
(`"superglue"`, `"lightglue"`, `"flexglue"`, `"matchanything"`, or any prefixed revision).
