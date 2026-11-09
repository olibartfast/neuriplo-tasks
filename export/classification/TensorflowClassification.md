## Models from TensorFlow/Keras API

The exporter uses pretrained Keras ResNet50 weights and produces a SavedModel for
`tensorflow-classifier`. Its input is Float32 NHWC **BGR pixels in [0, 255]**.
The graph subtracts the ResNet BGR channel means and returns **logits**; the task
applies softmax once. The exported graph contains the serving function's tracked
variables so TensorFlow C++ can restore it through `LoadSavedModel`.

```bash
# Clone neuriplo-tasks if not already available
git clone https://github.com/olibartfast/neuriplo-tasks.git
cd neuriplo-tasks/export/classification/tensorflow
python -m pip install -r requirements.txt
python export_tf_saved_model_classifier.py --output model.savedmodel
```

The supported recipe uses `keras.applications.ResNet50(weights="imagenet",
classifier_activation=None)` and `export_classifier()` from the script. Exporting
the unwrapped Keras model omits the raw-BGR preprocessing and normally retains a
softmax head, which does not match this task contract.

A lightweight regression test exports known weights without downloading a model,
loads the SavedModel through a TensorFlow Session, and checks its input dtype,
resource restoration, preprocessing and logits:

```bash
python tests/test_tensorflow_export.py
```

Run that test from the repository root in an environment with TensorFlow and NumPy.

## Remote TPU validation

TPU execution is a remote serving path, not an in-process backend: the TPU
service is an external KServe V2 process, and this repository owns only the
model export plus its validation recipe. The artifact contract is unchanged
(Float32 NHWC BGR pixels in `[0, 255]` in, Float32 logits out); no C++ task
contract changes are needed.

### Export the artifact used below

The TPU commands use `/tmp/tf-regate-resnet50`. Create that exact path from the
repository root; the exporter's default `--output` is `model.savedmodel`, so
pass the path explicitly instead of relying on the default:

```bash
python export/classification/tensorflow/export_tf_saved_model_classifier.py \
  --output /tmp/tf-regate-resnet50
```

The helper `export/classification/tensorflow/verify_tpu_saved_model.py` checks
an exported SavedModel in three modes:

```bash
# CPU metadata mode: signature names, dtypes, ranks, shapes. No TPU needed,
# but TensorFlow must be installed.
python export/classification/tensorflow/verify_tpu_saved_model.py \
  --saved-model /tmp/tf-regate-resnet50 \
  --mode metadata

# CPU reference mode: runs the artifact on CPU, prints the image SHA-256 and
# the top-5 classes/scores for comparison.
python export/classification/tensorflow/verify_tpu_saved_model.py \
  --saved-model /tmp/tf-regate-resnet50 \
  --image /path/to/dog.jpg \
  --mode cpu

# Cloud TPU VM only: runs the artifact inside a TPUStrategy scope and prints
# the top-5 classes/scores. Fails explicitly when TensorFlow or the TPU
# runtime is unavailable; it never falls back to CPU silently.
python export/classification/tensorflow/verify_tpu_saved_model.py \
  --saved-model /tmp/tf-regate-resnet50 \
  --image /path/to/dog.jpg \
  --mode tpu
```

Run these commands from the repository root. The image path is caller-supplied;
`neuriplo-tasks` does not ship a sample image. Record the exact image path and
hash with every comparison.

### Required runtime versions

- CPU metadata and CPU reference modes are exercised with
  `tensorflow-cpu==2.19.1` and `Pillow==13.0.0` (both listed in
  `requirements.txt`).
- TPU mode needs a Cloud TPU VM whose TensorFlow build can load the SavedModel.
  Use the TPU VM image's TensorFlow TPU build (the `2.19.x` line) together with
  the `libtpu` wheel built for that exact TensorFlow version; a TensorFlow
  `libtpu` mismatch makes `TPUClusterResolver` or `initialize_tpu_system` fail.
  Record the exact `tensorflow` and `libtpu` versions with
  `python -c "import tensorflow as tf; print(tf.__version__)"` and
  `python -m pip show libtpu` in the evidence table. No TPU runtime version has
  been validated on this branch because no TPU was available.

### Reading the output

Each mode prints the device, the artifact, the input SHA-256 (CPU/TPU modes),
and the top-5 class indices with their scores. The printed scores are the model's
raw logits, not softmax probabilities; apply softmax if a probability is needed.
Illustrative CPU output (the hash is the pinned 64-character SHA-256 of the
input file; scores are example values):

```text
device: CPU
image: /tmp/opencode/dog.jpg
sha256: 0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c4b5a69788796a5b4c3d2e1f0
top5:
  class=207 score=13.123456
  class=208 score=11.987654
  class=206 score=11.745012
  class=213 score=10.934221
  class=250 score=10.510377
```

TPU mode prints the same fields with `device: TPU`. Metadata mode prints the
resolved input and output spec instead of an image block.

### CPU/TPU comparison and tolerance

Run `--mode cpu` and `--mode tpu` against the same `--saved-model` and
`--image`. Both modes print the same `sha256`, which pins the input file.

1. Record whether the top-5 class sets and their order match.
2. Record the maximum absolute raw-logit difference over the top-5 entries and
   any class that changes rank.
3. Do not claim bitwise parity. Device arithmetic can reorder near-tied logits,
   so report the observed deltas rather than asserting an equality that the
   hardware cannot guarantee. Use a small absolute logit tolerance only as an
   initial screen (for example `atol=1e-2`) and record the tolerance actually
   applied with the evidence.

The CPU reference decoder is not the C++ `videocapture` decoder. Pillow decodes
the caller-supplied file without applying EXIF orientation, while
`videocapture`/OpenCV may honour it, so the two can differ at the JPEG/EXIF
boundary before any resize happens. Only the bilinear resize mapping is matched
to the C++ preprocessor; decode differences are out of scope for this helper.
That is another reason the procedure records class/score deltas and never
claims bitwise CPU/TPU or helper/`videocapture` parity.

`requirements.txt` includes Pillow, which the helper uses only to decode the
caller-supplied image. Resampling is done locally with the same half-pixel
bilinear mapping as the C++ preprocessor; Pillow's antialiased resampler is not
used, so decode and resize stay separate.

Known limits:

- TPU mode requires a Cloud TPU VM with a TensorFlow runtime compatible with
  the exported SavedModel (`TPUClusterResolver` must connect). On hosts
  without a TPU it exits non-zero with an explicit message.
- Missing TensorFlow or a missing artifact is an explicit error, not a silent
  pass.
- Detection, segmentation, pose, and video TPU exports are follow-ups; this
  packet covers the ResNet-50 classifier only.

Helper tests (no TPU hardware required; TensorFlow-dependent cases skip when
TensorFlow is absent):

```bash
python tests/test_verify_tpu_saved_model.py
```
