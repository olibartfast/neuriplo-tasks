"""Probe FlexGlue ONNX exports before trusting the C++ decoder.

Runs the two-model sparse flow from upstream onnx/match_pair.py (descriptor per
image, then the sparse matcher on externally detected keypoints) and prints the
graph contracts plus the values `FlexGluePostprocessor` relies on: keypoint
coordinate space, match index space, covisibility gating, and the
max_distance_px edge behavior.

Usage:
    python probe_flexglue.py --descriptor flexglue_descriptor.onnx --matcher flexglue_matcher.onnx
    python probe_flexglue.py --descriptor flexglue_descriptor.onnx --matcher flexglue_matcher.onnx \
        --source a.jpg --target b.jpg

Requires: numpy, onnxruntime, opencv-python (SIFT keypoints, as upstream).
"""

import argparse

import numpy as np
import onnxruntime as ort

DESCRIBE_SIZE = 512  # descriptor input is S x S, squashed bilinear, RGB [0,1]


def print_contract(session, title):
    print(f"{title}: {session.get_inputs()[0].name!r} model")
    print("  inputs:")
    for item in session.get_inputs():
        print(f"    {item.name} {item.shape} {item.type}")
    print("  outputs:")
    for item in session.get_outputs():
        print(f"    {item.name} {item.shape} {item.type}")


def read_image(path):
    import cv2

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def detect_keypoints(image, num_keypoints):
    import cv2

    gray = cv2.cvtColor((image * 255).round().astype(np.uint8), cv2.COLOR_RGB2GRAY)
    keypoints = cv2.SIFT_create(nfeatures=num_keypoints).detect(gray, None)
    keypoints = sorted(keypoints, key=lambda k: -k.response)
    points = np.array([k.pt for k in keypoints], dtype=np.float32)
    _, first = np.unique(points, axis=0, return_index=True)
    points = points[np.sort(first)][:num_keypoints]
    # OpenCV centers the top-left pixel at (0,0); FlexGlue expects corner
    # convention (center at (0.5, 0.5)).
    return points + 0.5


def describe(session, image):
    import cv2

    height, width = session.get_inputs()[0].shape[-2:]
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    (features,) = session.run(None, {"image": resized.transpose(2, 0, 1)[None]})
    return features


def match(session, features0, features1, kpts0, kpts1, size0, size1, filter_threshold=0.1,
          max_distance_px=3.0, min_covisibility=0.0):
    feeds = {
        "features0": features0,
        "features1": features1,
        "kpts0": kpts0[None].astype(np.float32),
        "kpts1": kpts1[None].astype(np.float32),
        "image_size0": np.array([size0], dtype=np.float32),
        "image_size1": np.array([size1], dtype=np.float32),
        "filter_threshold": np.array(filter_threshold, dtype=np.float32),
        "max_distance_px": np.array(max_distance_px, dtype=np.float32),
        "min_covisibility": np.array(min_covisibility, dtype=np.float32),
    }
    matches0, mscores0, covisibility = session.run(None, feeds)
    return matches0[0], mscores0[0], float(covisibility[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptor", required=True)
    parser.add_argument("--matcher", required=True)
    parser.add_argument("--source")
    parser.add_argument("--target")
    parser.add_argument("--num_keypoints", type=int, default=1024)
    parser.add_argument("--filter_threshold", type=float, default=0.1)
    parser.add_argument("--max_distance_px", type=float, default=3.0)
    parser.add_argument("--min_covisibility", type=float, default=0.0)
    args = parser.parse_args()

    ort.set_default_logger_severity(3)
    descriptor = ort.InferenceSession(args.descriptor, providers=["CPUExecutionProvider"])
    matcher = ort.InferenceSession(args.matcher, providers=["CPUExecutionProvider"])
    print_contract(descriptor, "descriptor")
    print_contract(matcher, "matcher")

    if not (args.source and args.target):
        return

    images, keypoints, features = [], [], []
    for path in (args.source, args.target):
        image = read_image(path)
        kpts = detect_keypoints(image, args.num_keypoints)
        feats = describe(descriptor, image)
        print(f"{path}: {image.shape[1]}x{image.shape[0]}, {len(kpts)} keypoints, "
              f"features {list(feats.shape)} {feats.dtype}")
        print(f"  kpts range x[{kpts[:, 0].min():.1f},{kpts[:, 0].max():.1f}] "
              f"y[{kpts[:, 1].min():.1f},{kpts[:, 1].max():.1f}] (original pixels expected)")
        images.append(image)
        keypoints.append(kpts)
        features.append(feats)

    sizes = [(im.shape[1], im.shape[0]) for im in images]
    matches0, mscores0, covis = match(
        matcher, *features, *keypoints, *sizes,
        filter_threshold=args.filter_threshold,
        max_distance_px=args.max_distance_px,
        min_covisibility=args.min_covisibility)
    n0 = len(keypoints[0])
    valid = matches0 > -1
    print(f"matches0 shape {list(matches0.shape)} dtype {matches0.dtype}, "
          f"mscores0 shape {list(mscores0.shape)} dtype {mscores0.dtype}")
    print(f"N0={n0} N1={len(keypoints[1])}, valid={int(valid.sum())}, "
          f"covisibility={covis:.4f}")
    print(f"matches0[:10]={matches0[:10].tolist()} scores[:10]={mscores0[:10].tolist()}")
    oob = valid & ((matches0 < 0) | (matches0 >= len(keypoints[1])))
    print(f"out-of-range indices among valid: {int(oob.sum())}")
    # Decoder-relevant: unmatched score values, and whether keypoint coords
    # come back anywhere (they should not — matcher outputs carry no coords).
    print(f"score of unmatched entries: min={float(mscores0[~valid].min() if (~valid).any() else float('nan')):.4f}")

    # Edge behavior the C++ decoder must mirror.
    m_zero, _, _ = match(matcher, *features, *keypoints, *sizes,
                         filter_threshold=args.filter_threshold, max_distance_px=0.0)
    print(f"max_distance_px=0 -> valid={(m_zero > -1).sum()} (upstream: rejects all in ONNX)")
    m_gate, _, cov_gate = match(matcher, *features, *keypoints, *sizes,
                                filter_threshold=args.filter_threshold,
                                min_covisibility=max(covis + 0.01, 1.0))
    print(f"min_covisibility above score ({covis:.4f}) -> valid={(m_gate > -1).sum()} (upstream: all rejected)")


if __name__ == "__main__":
    main()
