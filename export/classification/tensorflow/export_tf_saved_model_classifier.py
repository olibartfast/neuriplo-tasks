"""Export a ResNet50 accepting raw BGR pixels and returning logits."""
import argparse

import tensorflow as tf
from tensorflow import keras


def export_classifier(classifier, output_path):
    image = keras.Input(shape=classifier.input_shape[1:], dtype=tf.float32, name="image")
    # The task supplies BGR already; Keras ResNet needs BGR mean subtraction.
    pixels = keras.layers.Rescaling(
        1.0, offset=[-103.939, -116.779, -123.68], name="bgr_mean_subtraction"
    )(image)
    deployed = keras.Model(image, classifier(pixels, training=False))
    # Keras export tracks the resources used by its serving function. Wrapping
    # Keras variables in a plain tf.Module can leave TF1/C++ sessions uninitialized.
    deployed.export(str(output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="model.savedmodel")
    args = parser.parse_args()
    model = keras.applications.ResNet50(weights="imagenet", classifier_activation=None)
    export_classifier(model, args.output)
