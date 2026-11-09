"""Validate the TensorFlow classifier SavedModel for the remote TPU serving path.

The TPU service is an external KServe V2 process; this helper only validates the
exported artifact. It never changes the SavedModel input/output contract:

    input:  image, Float32, NHWC, [1, 224, 224, 3], BGR pixels in [0, 255]
    output: logits, Float32, [1, 1000]

Modes:
  metadata - CPU-side signature checks (names, dtypes, ranks, shapes).
             Needs TensorFlow installed but no TPU hardware.
  cpu      - Run the artifact on CPU and print top-5 classes/scores (reference).
  tpu      - Run the artifact through a TPUStrategy scope and print top-5.
             Fails explicitly when TensorFlow or the TPU runtime is unavailable;
             it never silently falls back to CPU.

This host has no TPU; TPU mode is expected to fail here with a clear message.

Pillow decodes the caller-supplied image file (JPEG/PNG/etc.). Resampling is
implemented locally as half-pixel-center bilinear to match the C++
`image_ops::resize(..., Interpolation::Linear)` used by the TensorFlow
preprocessor branch; Pillow's own resampler is antialiased and would not match.
`Pillow` is therefore a dependency of this helper, listed in
`export/classification/tensorflow/requirements.txt`.
"""

import argparse
import hashlib
import sys
from pathlib import Path

SERVING_TAG = "serving_default"
EXPECTED_INPUT_RANK = 4
EXPECTED_OUTPUT_RANK = 2
EXPECTED_INPUT_NAME = "image"
# CPU/TPU scores are compared with this absolute tolerance; classes must match
# exactly. This is not bitwise parity: TPU reductions can reorder float sums.
DEFAULT_SCORE_TOLERANCE = 1e-2
# Keras 3 exports name the serving output "output_0"; "logits" is accepted for
# hand-built or future exports that keep the contract name.
ACCEPTED_OUTPUT_NAMES = ("output_0", "logits")


class ValidationError(RuntimeError):
    """Raised when the artifact or runtime fails validation."""


def _require_tensorflow():
    try:
        import tensorflow as tf  # noqa: PLC0415
    except ImportError as exc:
        raise ValidationError(
            "TensorFlow is required for SavedModel validation but is not "
            "installed. Install it with: "
            "python -m pip install -r export/classification/tensorflow/requirements.txt"
        ) from exc
    return tf


def sha256_of_file(path):
    """Return the hex SHA-256 of a file (TF-free, used to pin evidence images)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def top_k_classes_scores(logits, k=5):
    """Return [(class_index, score), ...] for the top-k logits (TF-free).

    Ties keep their lowest-index-first order (stable sort).
    """
    import numpy as np  # noqa: PLC0415

    flat = np.asarray(logits, dtype=np.float64).reshape(-1)
    if k <= 0:
        raise ValidationError(f"top-k must be positive, got {k}")
    k = min(k, flat.size)
    order = sorted(range(flat.size), key=lambda i: flat[i], reverse=True)[:k]
    return [(int(i), float(flat[i])) for i in order]


def _resize_bilinear_uint8(pixels, height, width):
    """Resize an HxWxC uint8 array to (height, width) like the C++ preprocessor.

    `image_ops::resize(..., Interpolation::Linear)` uses half-pixel centers
    (`(dst + 0.5) * scale - 0.5`), edge clamping, and no antialias prefilter;
    the result is truncated back to uint8. Pillow's resampler antialiases
    downscales, so it cannot be used for parity.
    """
    import numpy as np  # noqa: PLC0415

    if height <= 0 or width <= 0:
        raise ValidationError(
            f"Resize target must be positive, got height={height}, width={width}."
        )
    source = np.asarray(pixels, dtype=np.uint8)
    if source.ndim != 3:
        raise ValidationError(
            f"Expected an HxWxC image array, got shape {list(source.shape)}."
        )
    src_height, src_width, channels = source.shape
    if src_height <= 0 or src_width <= 0:
        raise ValidationError(
            f"Source image must be non-empty, got shape {list(source.shape)}."
        )
    out = np.empty((height, width, channels), dtype=np.uint8)
    scale_y = src_height / height
    scale_x = src_width / width
    for dy in range(height):
        fy = (dy + 0.5) * scale_y - 0.5
        y0 = int(np.floor(fy))
        wy = fy - y0
        y0c = min(max(y0, 0), src_height - 1)
        y1c = min(max(y0 + 1, 0), src_height - 1)
        wy0 = 1.0 - wy
        row0 = source[y0c].astype(np.float64)
        row1 = source[y1c].astype(np.float64)
        for dx in range(width):
            fx = (dx + 0.5) * scale_x - 0.5
            x0 = int(np.floor(fx))
            wx = fx - x0
            x0c = min(max(x0, 0), src_width - 1)
            x1c = min(max(x0 + 1, 0), src_width - 1)
            wx0 = 1.0 - wx
            top = row0[x0c] * wx0 + row0[x1c] * wx
            bottom = row1[x0c] * wx0 + row1[x1c] * wx
            value = top * wy0 + bottom * wy
            out[dy, dx] = np.clip(value, 0.0, 255.0).astype(np.uint8)
    return out


def load_bgr_pixels(image_path, height, width):
    """Load an image file as Float32 NHWC BGR pixels in [0, 255] (TF-free).

    Mirrors the C++ `TensorflowPreprocessor` branch:
      * no BGR->RGB conversion (the model consumes BGR), so the decoded RGB
        file is flipped to BGR exactly once;
      * bilinear resize with the same half-pixel mapping as the C++ path;
      * no [0, 1] scaling (raw 0-255 values cast to Float32).
    """
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError as exc:
        raise ValidationError(
            "Pillow is required to load validation images but is not installed. "
            "Install it with: "
            "python -m pip install -r export/classification/tensorflow/requirements.txt"
        ) from exc
    import numpy as np  # noqa: PLC0415

    try:
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"))
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise ValidationError(
            f"Failed to decode validation image {image_path}: {exc}"
        ) from exc
    if rgb.size == 0:
        raise ValidationError(f"Decoded validation image {image_path} is empty.")
    # C++ receives BGR frames and does not swap for the TensorFlow branch.
    bgr = rgb[:, :, ::-1].copy()
    resized = _resize_bilinear_uint8(bgr, height, width)
    return resized.astype(np.float32).reshape(1, height, width, 3)


def _resolve_saved_model_dir(saved_model_path):
    path = Path(saved_model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"SavedModel not found: {path}. Export one with "
            "export/classification/tensorflow/export_tf_saved_model_classifier.py "
            "before validating."
        )
    return path


def _signature_io(function):
    """Return (inputs, outputs) spec dicts for a serving signature.

    Keras 3 tracks resource captures (dtype ``resource``) as extra signature
    inputs. Only the structured, non-resource input/output specs describe the
    model contract, so flat ``function.inputs``/``function.outputs`` must not be
    counted directly.
    """
    inputs = {}
    structured_inputs = function.structured_input_signature
    parts = structured_inputs if isinstance(structured_inputs, (list, tuple)) else (structured_inputs,)
    for part in parts:
        if isinstance(part, dict):
            for key, spec in part.items():
                if spec.dtype.name != "resource":
                    inputs[key] = spec
        elif isinstance(part, (list, tuple)):
            for index, spec in enumerate(part):
                if spec.dtype.name != "resource":
                    inputs[f"args_{index}"] = spec

    outputs = {}
    structured_outputs = function.structured_outputs
    if isinstance(structured_outputs, dict):
        entries = structured_outputs.items()
    elif isinstance(structured_outputs, (list, tuple)):
        entries = ((f"output_{index}", spec) for index, spec in enumerate(structured_outputs))
    else:
        entries = (("output_0", structured_outputs),)
    for key, spec in entries:
        if spec.dtype.name != "resource":
            outputs[key] = spec

    if len(inputs) != 1:
        raise ValidationError(
            f"Expected exactly 1 non-resource input in signature '{SERVING_TAG}', "
            f"got {len(inputs)}: {sorted(inputs)}. Resource captures are ignored."
        )
    if not outputs:
        raise ValidationError(
            f"Signature '{SERVING_TAG}' exposes no tensor outputs."
        )
    return inputs, outputs


def _select_output_tensor(outputs):
    """Return the logits tensor from a signature call result.

    Accepts the Keras 3 ``{"output_0": tensor}`` dict, a ``logits`` dict, or a
    bare single tensor. Raises ValidationError for unexpected output names.
    """
    if isinstance(outputs, dict):
        for name in ACCEPTED_OUTPUT_NAMES:
            if name in outputs:
                return outputs[name]
        raise ValidationError(
            f"Unexpected output names {sorted(outputs)}; expected one of "
            f"{list(ACCEPTED_OUTPUT_NAMES)}."
        )
    if isinstance(outputs, (list, tuple)):
        if len(outputs) != 1:
            raise ValidationError(
                f"Expected exactly 1 output tensor, got {len(outputs)}."
            )
        return outputs[0]
    return outputs


def _check_spec(spec, *, role, expected_dtype_name, expected_rank,
                expected_shape):
    dtype_name = spec.dtype.name
    shape = spec.shape.as_list()
    if dtype_name != expected_dtype_name:
        raise ValidationError(
            f"{role} dtype mismatch: expected {expected_dtype_name}, "
            f"got {dtype_name} (name={spec.name})."
        )
    if len(shape) != expected_rank:
        raise ValidationError(
            f"{role} rank mismatch: expected {expected_rank}, got {len(shape)} "
            f"(shape={shape}, name={spec.name})."
        )
    for axis, (got, want) in enumerate(zip(shape, expected_shape)):
        if want is None:
            continue
        if got is None:
            # A dynamic batch axis (axis 0) is legal for the serving signature;
            # every other axis must be concrete so a 224/1000 artifact cannot
            # pass validation with an unconstrained spatial or class dimension.
            if axis == 0:
                continue
            raise ValidationError(
                f"{role} shape has a dynamic axis {axis}: expected a concrete "
                f"dimension {want} in {expected_shape}, got {shape} "
                f"(name={spec.name})."
            )
        if got != want:
            raise ValidationError(
                f"{role} shape mismatch at axis {axis}: expected {expected_shape}, "
                f"got {shape} (name={spec.name})."
            )
    return {"name": spec.name, "dtype": dtype_name, "shape": shape}


def check_metadata(saved_model_path, image_shape=(1, 224, 224, 3),
                   num_classes=1000):
    """Verify the SavedModel signature without TPU hardware.

    Returns a dict with input/output names, dtypes, and shapes. Raises
    FileNotFoundError when the artifact is missing, ValidationError when
    TensorFlow is missing or the contract does not match.
    """
    path = _resolve_saved_model_dir(saved_model_path)
    tf = _require_tensorflow()
    try:
        model = tf.saved_model.load(str(path))
    except Exception as exc:
        raise ValidationError(f"Failed to load SavedModel at {path}: {exc}") from exc
    if SERVING_TAG not in model.signatures:
        raise ValidationError(
            f"Signature '{SERVING_TAG}' not found in SavedModel at {path}; "
            f"available: {sorted(model.signatures)}."
        )
    function = model.signatures[SERVING_TAG]
    inputs, outputs = _signature_io(function)

    image_name = next(iter(inputs))
    if image_name != EXPECTED_INPUT_NAME:
        raise ValidationError(
            f"Input name mismatch: expected '{EXPECTED_INPUT_NAME}', got "
            f"'{image_name}' (signature inputs: {sorted(inputs)})."
        )
    if len(outputs) != 1:
        raise ValidationError(
            f"Expected exactly 1 output tensor in signature '{SERVING_TAG}', "
            f"got {len(outputs)}: {sorted(outputs)}."
        )
    logits_name = next(iter(outputs))
    if logits_name not in ACCEPTED_OUTPUT_NAMES:
        raise ValidationError(
            f"Output name mismatch: expected one of {list(ACCEPTED_OUTPUT_NAMES)}, "
            f"got '{logits_name}'."
        )

    expected_input = list(image_shape)
    expected_output = [image_shape[0], num_classes]
    input_info = _check_spec(inputs[image_name], role="input",
                             expected_dtype_name="float32",
                             expected_rank=EXPECTED_INPUT_RANK,
                             expected_shape=expected_input)
    output_info = _check_spec(outputs[logits_name], role="output",
                              expected_dtype_name="float32",
                              expected_rank=EXPECTED_OUTPUT_RANK,
                              expected_shape=expected_output)
    input_info["signature_key"] = image_name
    output_info["signature_key"] = logits_name
    return {"saved_model": str(path), "input": input_info, "output": output_info}


def _run_logits(saved_model_path, image_path, image_shape, num_classes,
                executor):
    tf = _require_tensorflow()
    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Validation image not found: {image_file}.")
    height, width = image_shape[1], image_shape[2]
    pixels = load_bgr_pixels(str(image_file), height, width)
    logits = executor(str(_resolve_saved_model_dir(saved_model_path)), pixels)
    import numpy as np  # noqa: PLC0415

    logits = np.asarray(logits, dtype=np.float32)
    if logits.shape != (image_shape[0], num_classes):
        raise ValidationError(
            f"Logits shape mismatch: expected {[image_shape[0], num_classes]}, "
            f"got {list(logits.shape)}."
        )
    top5 = top_k_classes_scores(logits[0], k=5)
    return {
        "image": str(image_file),
        "sha256": sha256_of_file(str(image_file)),
        "top5": [{"class": index, "score": score} for index, score in top5],
    }


def _serving_function(model, path):
    """Return the serving signature function or raise a clear ValidationError."""
    signatures = getattr(model, "signatures", None) or {}
    try:
        return signatures[SERVING_TAG]
    except (KeyError, TypeError) as exc:
        raise ValidationError(
            f"Signature '{SERVING_TAG}' not found in SavedModel at {path}; "
            f"available: {sorted(signatures)}."
        ) from exc


def _cpu_executor(path, pixels):
    """Load and execute the artifact on CPU, returning the logits tensor."""
    tf = _require_tensorflow()
    with tf.device("/CPU:0"):
        model = tf.saved_model.load(path)
        function = _serving_function(model, path)
        outputs = function(tf.convert_to_tensor(pixels))
    return _select_output_tensor(outputs)


def run_cpu_reference(saved_model_path, image_path, image_shape=(1, 224, 224, 3),
                      num_classes=1000):
    """Run the artifact on CPU and return image hash plus top-5."""
    result = _run_logits(saved_model_path, image_path, image_shape, num_classes,
                         _cpu_executor)
    result["device"] = "CPU"
    return result


def _make_tpu_strategy(tf):
    """Connect to the TPU and return an initialized TPUStrategy.

    Raises ValidationError instead of silently falling back to CPU when the TPU
    runtime or device is unavailable.
    """
    if not hasattr(tf.distribute.cluster_resolver, "TPUClusterResolver"):
        raise ValidationError(
            "This TensorFlow build has no TPUClusterResolver; TPU mode cannot run. "
            "Use --mode cpu or --mode metadata on this host."
        )
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    except Exception as exc:
        raise ValidationError(
            "No TPU found: TPUClusterResolver could not connect. Set the TPU_NAME "
            f"environment variable on a Cloud TPU VM. Details: {exc}"
        ) from exc
    try:
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
    except Exception as exc:
        raise ValidationError(
            f"TPU initialization failed; refusing to fall back to CPU. Details: {exc}"
        ) from exc
    try:
        strategy = tf.distribute.TPUStrategy(resolver)
    except Exception as exc:
        raise ValidationError(
            "TPUStrategy could not be created; refusing to fall back to CPU. "
            f"Details: {exc}"
        ) from exc
    if not tf.config.list_logical_devices("TPU"):
        raise ValidationError(
            "TPUStrategy was created but no TPU logical devices are visible; "
            "refusing to report a CPU result as TPU."
        )
    return strategy


def _run_tpu_inference(tf, strategy, path, pixels):
    """Trace and execute one inference through TPUStrategy, returning logits.

    The single image is replicated to every core, inference runs via
    ``strategy.run`` (never on CPU outside the strategy), and the per-replica
    results are reduced to one tensor.
    """
    with strategy.scope():
        model = tf.saved_model.load(path)
        function = _serving_function(model, path)

        @tf.function
        def step(batch):
            return _select_output_tensor(function(batch))

        def replica_input(_context):
            return tf.convert_to_tensor(pixels, dtype=tf.float32)

        distributed = strategy.experimental_distribute_values_from_function(
            replica_input
        )
        per_replica = strategy.run(step, args=(distributed,))
        return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica, axis=None)


def _libtpu_version():
    """Return the installed libtpu version string, or None when unavailable."""
    try:
        import libtpu  # noqa: PLC0415
    except ImportError:
        return None
    for attribute in ("__version__", "version"):
        value = getattr(libtpu, attribute, None)
        if value is None:
            continue
        try:
            return str(value() if callable(value) else value)
        except Exception:
            continue
    return None


def _tpu_evidence(tf, strategy, saved_model_path):
    """Collect M-1 evidence for a TPU result (best effort, never fails the run)."""
    import os  # noqa: PLC0415

    try:
        devices = [device.name for device in tf.config.list_logical_devices("TPU")]
    except Exception:
        devices = []
    tpu_type = (
        os.environ.get("TPU_TYPE")
        or os.environ.get("TPU_ACCELERATOR_TYPE")
        or os.environ.get("TPU_NAME")
        or "unknown"
    )
    try:
        replicas = int(strategy.num_replicas_in_sync)
    except Exception:
        replicas = None
    return {
        "saved_model": str(saved_model_path),
        "tensorflow_version": getattr(tf, "__version__", "unknown"),
        "tpu_type": tpu_type,
        "tpu_devices": devices,
        "libtpu_version": _libtpu_version(),
        "num_replicas_in_sync": replicas,
    }


def compare_top5(cpu_top5, tpu_top5, score_tolerance=DEFAULT_SCORE_TOLERANCE):
    """Compare CPU and TPU top-5 results (classes exact, scores within tolerance).

    Returns a dict describing the per-rank class/score deltas. It never claims
    bitwise parity: only the class indices and the annotated score tolerance are
    part of the pass criterion.
    """
    if score_tolerance < 0:
        raise ValidationError(
            f"Score tolerance must be non-negative, got {score_tolerance}."
        )
    if len(cpu_top5) != len(tpu_top5):
        raise ValidationError(
            f"CPU/TPU top-k lengths differ: cpu={len(cpu_top5)}, "
            f"tpu={len(tpu_top5)}."
        )
    entries = []
    match = True
    for rank, (cpu, tpu) in enumerate(zip(cpu_top5, tpu_top5), start=1):
        class_delta = int(tpu["class"]) - int(cpu["class"])
        score_delta = float(tpu["score"]) - float(cpu["score"])
        class_match = class_delta == 0
        score_match = abs(score_delta) <= score_tolerance
        if not (class_match and score_match):
            match = False
        entries.append({
            "rank": rank,
            "cpu_class": int(cpu["class"]),
            "tpu_class": int(tpu["class"]),
            "class_match": class_match,
            "class_delta": class_delta,
            "cpu_score": float(cpu["score"]),
            "tpu_score": float(tpu["score"]),
            "score_delta": score_delta,
            "score_match": score_match,
        })
    return {
        "score_tolerance": float(score_tolerance),
        "match": match,
        "entries": entries,
    }


def _comparison_failure_message(comparison):
    details = []
    for entry in comparison["entries"]:
        problems = []
        if not entry["class_match"]:
            problems.append(
                f"class cpu={entry['cpu_class']} tpu={entry['tpu_class']} "
                f"(delta={entry['class_delta']})"
            )
        if not entry["score_match"]:
            problems.append(
                f"score cpu={entry['cpu_score']:.6f} "
                f"tpu={entry['tpu_score']:.6f} (delta={entry['score_delta']:+.6f})"
            )
        if problems:
            details.append(f"rank {entry['rank']}: " + ", ".join(problems))
    return (
        "CPU/TPU top-5 comparison failed "
        f"(score tolerance {comparison['score_tolerance']}): " + "; ".join(details)
    )


def run_tpu(saved_model_path, image_path, image_shape=(1, 224, 224, 3),
            num_classes=1000):
    """Run the artifact through a TPUStrategy scope and return top-5.

    Fails with ValidationError when TensorFlow/TPU support or the TPU itself is
    unavailable. Never falls back to CPU. Artifact and image paths are resolved
    before TPU initialization so missing inputs raise FileNotFoundError (exit 2)
    instead of being misreported as a TPU failure.
    """
    image_file = Path(image_path)
    resolved_model = _resolve_saved_model_dir(saved_model_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Validation image not found: {image_file}.")

    tf = _require_tensorflow()
    strategy = _make_tpu_strategy(tf)

    def executor(path, pixels):
        return _run_tpu_inference(tf, strategy, path, pixels)

    try:
        result = _run_logits(str(resolved_model), str(image_file), image_shape,
                             num_classes, executor)
    except (FileNotFoundError, OSError):
        # Filesystem/artifact errors are not TPU failures; keep exit 2 distinct.
        raise
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(f"TPU inference failed: {exc}") from exc
    result["device"] = "TPU"
    result["evidence"] = _tpu_evidence(tf, strategy, resolved_model)
    return result


def _format_result(result):
    lines = [f"device: {result.get('device', 'metadata')}"]
    if "saved_model" in result:
        lines.append(f"saved_model: {result['saved_model']}")
    if "input" in result:
        lines.append(f"input: {result['input']}")
        lines.append(f"output: {result['output']}")
    if "image" in result:
        lines.append(f"image: {result['image']}")
        lines.append(f"sha256: {result['sha256']}")
        lines.append("top5:")
        for entry in result["top5"]:
            lines.append(f"  class={entry['class']} score={entry['score']:.6f}")
    evidence = result.get("evidence")
    if evidence:
        lines.append("tpu_evidence:")
        for key in ("saved_model", "tensorflow_version", "tpu_type",
                    "tpu_devices", "libtpu_version", "num_replicas_in_sync"):
            if key in evidence:
                value = evidence[key]
                if key == "tpu_devices":
                    value = ", ".join(value) if value else "none"
                elif value is None:
                    value = "unavailable"
                lines.append(f"  {key}: {value}")
    comparison = result.get("comparison")
    if comparison:
        lines.append(
            f"comparison: top5 classes exact, scores within "
            f"tolerance {comparison['score_tolerance']} "
            f"(match={comparison['match']})"
        )
        for entry in comparison["entries"]:
            lines.append(
                f"  rank {entry['rank']}: "
                f"cpu_class={entry['cpu_class']} tpu_class={entry['tpu_class']} "
                f"class_delta={entry['class_delta']:+d} "
                f"cpu_score={entry['cpu_score']:.6f} "
                f"tpu_score={entry['tpu_score']:.6f} "
                f"score_delta={entry['score_delta']:+.6f}"
            )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-model", required=True,
                        help="Path to the exported SavedModel directory.")
    parser.add_argument("--image",
                        help="Validation image file (required for cpu/tpu modes).")
    parser.add_argument("--mode", choices=("metadata", "cpu", "tpu"),
                        default="metadata")
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--compare-cpu", action="store_true",
                        help="Run the CPU reference and compare it with TPU top-5 "
                             "(requires --mode tpu).")
    parser.add_argument("--score-tolerance", type=float,
                        default=DEFAULT_SCORE_TOLERANCE,
                        help="Absolute score tolerance for the CPU/TPU "
                             f"comparison (default {DEFAULT_SCORE_TOLERANCE}).")
    args = parser.parse_args(argv)

    image_shape = (1, args.image_height, args.image_width, 3)
    try:
        if args.compare_cpu and args.mode != "tpu":
            raise ValidationError("--compare-cpu requires --mode tpu.")
        if args.mode == "metadata":
            result = check_metadata(args.saved_model, image_shape,
                                    args.num_classes)
            result["device"] = "metadata (no device execution)"
        elif args.mode == "cpu":
            if not args.image:
                raise ValidationError("--image is required for --mode cpu.")
            result = run_cpu_reference(args.saved_model, args.image, image_shape,
                                       args.num_classes)
        else:
            if not args.image:
                raise ValidationError("--image is required for --mode tpu.")
            cpu_reference = None
            if args.compare_cpu:
                cpu_reference = run_cpu_reference(
                    args.saved_model, args.image, image_shape, args.num_classes
                )
            result = run_tpu(args.saved_model, args.image, image_shape,
                             args.num_classes)
            if cpu_reference is not None:
                comparison = compare_top5(
                    cpu_reference["top5"], result["top5"],
                    args.score_tolerance,
                )
                result["comparison"] = comparison
                if not comparison["match"]:
                    raise ValidationError(
                        _comparison_failure_message(comparison)
                    )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    print(_format_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
