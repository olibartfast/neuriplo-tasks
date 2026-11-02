"""Probe a SuperGlue ONNX export before trusting the C++ decoder.

Prints the graph contract and, when given an image pair, the values the
`SuperGluePostprocessor` relies on: coordinate normalization, match index space,
mask semantics, and how many matches survive a score threshold.

Usage:
    python probe_superglue.py --model superglue.onnx
    python probe_superglue.py --model superglue.onnx --source a.jpg --target b.jpg

Requires: numpy, onnxruntime, Pillow (from transformers' SuperGlueImageProcessor
contract: resize to 640x480 bilinear, rescale 1/255, grayscale replicated to 3
channels).
"""

import argparse

import numpy as np
import onnxruntime as ort

INPUT_SIZE = (640, 480)  # width, height


def preprocess(path):
    from PIL import Image

    image = Image.open(path).convert("RGB").resize(INPUT_SIZE, Image.BILINEAR)
    array = np.asarray(image).astype(np.float32) / 255.0
    gray = 0.2989 * array[..., 0] + 0.5870 * array[..., 1] + 0.1140 * array[..., 2]
    return np.stack([gray, gray, gray])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source")
    parser.add_argument("--target")
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    print("inputs:")
    for item in session.get_inputs():
        print(f"  {item.name} {item.shape} {item.type}")
    print("outputs:")
    for item in session.get_outputs():
        print(f"  {item.name} {item.shape} {item.type}")

    if not (args.source and args.target):
        return

    from PIL import Image

    source_size = Image.open(args.source).size  # (width, height)
    target_size = Image.open(args.target).size
    inputs = np.stack([preprocess(args.source), preprocess(args.target)])[None].astype(np.float32)
    outputs = dict(zip([o.name for o in session.get_outputs()], session.run(None, {session.get_inputs()[0].name: inputs})))

    keypoints = outputs["keypoints"][0]
    matches = outputs["matches"][0][0]
    scores = outputs["matching_scores"][0][0]
    mask = outputs["mask"][0]

    count = keypoints.shape[1]
    valid0 = int(mask[0].sum())
    valid1 = int(mask[1].sum())
    print(f"\nN={count} keypoints (padded); valid source={valid0} target={valid1}")
    for side, label in enumerate(("source", "target")):
        valid = mask[side] > 0
        if valid.any():
            print(f"  {label} normalized x[{keypoints[side][valid][:, 0].min():.4f}, "
                  f"{keypoints[side][valid][:, 0].max():.4f}] "
                  f"y[{keypoints[side][valid][:, 1].min():.4f}, {keypoints[side][valid][:, 1].max():.4f}]")
        print(f"  {label} mask is prefix-packed: {bool(np.array_equal(np.sort(np.where(valid)[0]), np.where(valid)[0]))}")

    matched = int(((matches >= 0) & (scores > args.threshold)).sum())
    print(f"matches above threshold {args.threshold}: {matched}")
    if matches[matches >= 0].size:
        print(f"  match index range: {int(matches[matches >= 0].min())}..{int(matches[matches >= 0].max())}"
              f" (valid target slots: {valid1})")

    print("\nfirst matches (score, source px, target px):")
    shown = 0
    for index in range(count):
        target = int(matches[index])
        if target < 0 or mask[0][index] <= 0 or mask[1][target] <= 0 or scores[index] <= args.threshold:
            continue
        sx = float(keypoints[0][index][0] * source_size[0])
        sy = float(keypoints[0][index][1] * source_size[1])
        tx = float(keypoints[1][target][0] * target_size[0])
        ty = float(keypoints[1][target][1] * target_size[1])
        print(f"  {scores[index]:.6f} ({sx:.2f}, {sy:.2f}) -> ({tx:.2f}, {ty:.2f})")
        shown += 1
        if shown == 5:
            break


if __name__ == "__main__":
    main()
