"""Verify the SavedModel contract with the Session loader used by TensorFlow C++."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tensorflow as tf

spec = importlib.util.spec_from_file_location(
    "tensorflow_export",
    Path(__file__).resolve().parents[1] / "export/classification/tensorflow/export_tf_saved_model_classifier.py",
)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class TensorFlowExportTests(unittest.TestCase):
    def test_session_restores_weights_and_applies_bgr_means_once(self):
        inputs = tf.keras.Input(shape=(4, 6, 3))
        features = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        outputs = tf.keras.layers.Dense(
            2, use_bias=False,
            kernel_initializer=tf.keras.initializers.Constant([[1, 2], [3, 4], [5, 6]]),
        )(features)
        classifier = tf.keras.Model(inputs, outputs)
        before = [weight.numpy().copy() for weight in classifier.weights]
        pixels = np.broadcast_to(np.array([110, 120, 130], np.float32), (1, 4, 6, 3)).copy()
        with tempfile.TemporaryDirectory() as path:
            exporter.export_classifier(classifier, path)
            with tf.compat.v1.Session(graph=tf.Graph()) as session:
                metadata = tf.compat.v1.saved_model.loader.load(session, ["serve"], path)
                signature = metadata.signature_def["serving_default"]
                self.assertEqual(len(signature.inputs), 1)
                self.assertEqual(len(signature.outputs), 1)
                image = next(iter(signature.inputs.values()))
                self.assertEqual(image.dtype, tf.float32.as_datatype_enum)
                logits = session.run(
                    next(iter(signature.outputs.values())).name, {image.name: pixels}
                )
                np.testing.assert_allclose(logits, [[47.324, 62.926]], rtol=0, atol=1e-4)
        for weight, expected in zip(classifier.weights, before):
            np.testing.assert_array_equal(weight.numpy(), expected)


if __name__ == "__main__":
    unittest.main()
