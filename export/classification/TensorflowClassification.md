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
