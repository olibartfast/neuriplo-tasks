"""Tests for export/classification/tensorflow/verify_tpu_saved_model.py.

TPU hardware is never required: TF-dependent cases skip when TensorFlow is
absent, and error-path cases assert the helper fails clearly instead of
passing silently.
"""
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

HELPER = (
    Path(__file__).resolve().parents[1]
    / "export/classification/tensorflow/verify_tpu_saved_model.py"
)

spec = importlib.util.spec_from_file_location("verify_tpu_saved_model", HELPER)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)

try:
    import tensorflow as tf  # noqa: F401
    HAS_TF = True
except ImportError:
    HAS_TF = False


class _FakeDType:
    def __init__(self, name="float32"):
        self.name = name


class _FakeShape:
    def __init__(self, dims):
        self._dims = list(dims)

    def as_list(self):
        return list(self._dims)


class _FakeSpec:
    def __init__(self, shape, name="placeholder"):
        self.dtype = _FakeDType()
        self.shape = _FakeShape(shape)
        self.name = name


def _reference_resize_bilinear(pixels, height, width):
    """Independent scalar reference for the half-pixel-center bilinear resize."""
    source = np.asarray(pixels, dtype=np.float64)
    src_height, src_width, channels = source.shape
    out = np.empty((height, width, channels), dtype=np.uint8)
    for dy in range(height):
        fy = (dy + 0.5) * src_height / height - 0.5
        y0 = int(np.floor(fy))
        wy = fy - y0
        y0c = min(max(y0, 0), src_height - 1)
        y1c = min(max(y0 + 1, 0), src_height - 1)
        for dx in range(width):
            fx = (dx + 0.5) * src_width / width - 0.5
            x0 = int(np.floor(fx))
            wx = fx - x0
            x0c = min(max(x0, 0), src_width - 1)
            x1c = min(max(x0 + 1, 0), src_width - 1)
            top = source[y0c, x0c] * (1 - wx) + source[y0c, x1c] * wx
            bottom = source[y1c, x0c] * (1 - wx) + source[y1c, x1c] * wx
            value = top * (1 - wy) + bottom * wy
            out[dy, dx] = np.clip(value, 0, 255).astype(np.uint8)
    return out


class PureHelperTests(unittest.TestCase):
    def test_top_k_order_and_scores(self):
        top = helper.top_k_classes_scores([0.1, 5.0, 3.0, 5.0, -1.0], k=3)
        self.assertEqual([c for c, _ in top], [1, 3, 2])
        self.assertAlmostEqual(top[0][1], 5.0)

    def test_top_k_rejects_non_positive_k(self):
        with self.assertRaises(helper.ValidationError):
            helper.top_k_classes_scores([1.0, 2.0], k=0)

    def test_sha256_pins_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            handle.write(b"neuriplo-tpu")
            name = handle.name
        import hashlib
        self.assertEqual(helper.sha256_of_file(name), hashlib.sha256(b"neuriplo-tpu").hexdigest())

    def test_load_bgr_pixels_shape_dtype_and_channel_order(self):
        from PIL import Image
        red = np.zeros((4, 6, 3), np.uint8)
        red[:, :, 0] = 255
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            Image.fromarray(red, "RGB").save(handle.name)
            name = handle.name
        pixels = helper.load_bgr_pixels(name, 4, 6)
        self.assertEqual(pixels.shape, (1, 4, 6, 3))
        self.assertEqual(pixels.dtype, np.float32)
        # Pure red RGB -> BGR (0, 0, 255).
        np.testing.assert_array_equal(pixels[0, 0, 0], [0.0, 0.0, 255.0])

    def test_load_bgr_pixels_resizes(self):
        from PIL import Image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            Image.fromarray(np.zeros((8, 8, 3), np.uint8), "RGB").save(handle.name)
            name = handle.name
        self.assertEqual(helper.load_bgr_pixels(name, 4, 6).shape, (1, 4, 6, 3))

    def test_load_bgr_pixels_rejects_undecodable_file(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            handle.write(b"not an image")
            name = handle.name
        with self.assertRaises(helper.ValidationError):
            helper.load_bgr_pixels(name, 4, 6)

    def test_resize_bilinear_known_values(self):
        # 1x2 -> 1x3 with half-pixel centers gives the exact edge-clamped ramp.
        source = np.array([[[0], [100]]], dtype=np.uint8)
        result = helper._resize_bilinear_uint8(source, 1, 3)
        np.testing.assert_array_equal(result, np.array([[[0], [50], [100]]],
                                                       dtype=np.uint8))

    def test_resize_bilinear_known_downscale_value(self):
        # 2x2 -> 1x1 lands at (0.5, 0.5): 138.75 truncates to 138.
        source = np.array([[[0], [100]], [[200], [255]]], dtype=np.uint8)
        result = helper._resize_bilinear_uint8(source, 1, 1)
        np.testing.assert_array_equal(result, np.array([[[138]]], dtype=np.uint8))

    def test_resize_bilinear_matches_reference_array(self):
        rng = np.random.default_rng(1234)
        source = rng.integers(0, 256, size=(7, 5, 3), dtype=np.uint8)
        expected = _reference_resize_bilinear(source, 4, 3)
        np.testing.assert_array_equal(
            helper._resize_bilinear_uint8(source, 4, 3), expected
        )

    def test_resize_bilinear_rejects_zero_size_geometry(self):
        source = np.zeros((2, 2, 3), dtype=np.uint8)
        with self.assertRaises(helper.ValidationError):
            helper._resize_bilinear_uint8(source, 0, 2)
        with self.assertRaises(helper.ValidationError):
            helper._resize_bilinear_uint8(np.zeros((0, 2, 3), np.uint8), 1, 1)

    def test_check_spec_allows_dynamic_batch_axis(self):
        info = helper._check_spec(
            _FakeSpec([None, 224, 224, 3]),
            role="input",
            expected_dtype_name="float32",
            expected_rank=4,
            expected_shape=[1, 224, 224, 3],
        )
        self.assertEqual(info["shape"], [None, 224, 224, 3])

    def test_check_spec_rejects_dynamic_spatial_axis(self):
        with self.assertRaises(helper.ValidationError):
            helper._check_spec(
                _FakeSpec([None, None, 224, 3]),
                role="input",
                expected_dtype_name="float32",
                expected_rank=4,
                expected_shape=[1, 224, 224, 3],
            )
        with self.assertRaises(helper.ValidationError):
            helper._check_spec(
                _FakeSpec([None, 224, None, 3]),
                role="input",
                expected_dtype_name="float32",
                expected_rank=4,
                expected_shape=[1, 224, 224, 3],
            )

    def test_check_spec_rejects_dynamic_class_axis(self):
        with self.assertRaises(helper.ValidationError):
            helper._check_spec(
                _FakeSpec([None, None]),
                role="output",
                expected_dtype_name="float32",
                expected_rank=2,
                expected_shape=[1, 1000],
            )

    def test_compare_top5_match(self):
        cpu = [{"class": 3, "score": 0.9}, {"class": 1, "score": 0.5}]
        tpu = [{"class": 3, "score": 0.905}, {"class": 1, "score": 0.495}]
        comparison = helper.compare_top5(cpu, tpu, score_tolerance=1e-2)
        self.assertTrue(comparison["match"])
        self.assertEqual([e["class_delta"] for e in comparison["entries"]], [0, 0])
        self.assertAlmostEqual(comparison["entries"][0]["score_delta"], 0.005,
                               places=6)

    def test_compare_top5_class_mismatch_reports_delta(self):
        cpu = [{"class": 3, "score": 0.9}]
        tpu = [{"class": 4, "score": 0.9}]
        comparison = helper.compare_top5(cpu, tpu, score_tolerance=1e-2)
        self.assertFalse(comparison["match"])
        self.assertEqual(comparison["entries"][0]["class_delta"], 1)
        message = helper._comparison_failure_message(comparison)
        self.assertIn("rank 1", message)
        self.assertIn("class", message)

    def test_compare_top5_score_tolerance(self):
        cpu = [{"class": 3, "score": 0.9}]
        within = helper.compare_top5(cpu, [{"class": 3, "score": 0.909}],
                                     score_tolerance=1e-2)
        self.assertTrue(within["match"])
        outside = helper.compare_top5(cpu, [{"class": 3, "score": 0.95}],
                                      score_tolerance=1e-2)
        self.assertFalse(outside["match"])
        message = helper._comparison_failure_message(outside)
        self.assertIn("score", message)
        self.assertNotIn("bitwise", message)

    def test_compare_top5_rejects_negative_tolerance(self):
        with self.assertRaises(helper.ValidationError):
            helper.compare_top5([], [], score_tolerance=-1)


class ErrorPathTests(unittest.TestCase):
    def test_missing_artifact_is_explicit(self):
        with self.assertRaises(FileNotFoundError):
            helper.check_metadata("/tmp/neuriplo-tasks-missing-savedmodel")

    def test_missing_image_is_explicit(self):
        with self.assertRaises((FileNotFoundError, helper.ValidationError)):
            helper.run_cpu_reference("/tmp/neuriplo-tasks-missing-savedmodel",
                                     "/tmp/neuriplo-tasks-missing-image.jpg")

    def test_tpu_mode_never_passes_silently_without_tpu(self):
        # On a host without TF/TPU this must raise, never return CPU results.
        with self.assertRaises((helper.ValidationError, FileNotFoundError)):
            helper.run_tpu("/tmp/neuriplo-tasks-missing-savedmodel",
                           "/tmp/neuriplo-tasks-missing-image.jpg")

    def test_cli_missing_image_rejected_for_cpu(self):
        code = helper.main(["--saved-model", "/tmp/whatever", "--mode", "cpu"])
        self.assertNotEqual(code, 0)

    def test_cli_missing_artifact_is_nonzero(self):
        code = helper.main(["--saved-model", "/tmp/neuriplo-tasks-missing-savedmodel",
                            "--mode", "metadata"])
        self.assertNotEqual(code, 0)

    def test_run_tpu_resolves_artifact_before_touching_tpu(self):
        with mock.patch.object(helper, "_make_tpu_strategy") as make_strategy:
            with self.assertRaises(FileNotFoundError):
                helper.run_tpu("/tmp/neuriplo-tasks-missing-savedmodel",
                               "/tmp/neuriplo-tasks-missing-image.jpg")
        make_strategy.assert_not_called()

    def test_run_tpu_resolves_image_before_touching_tpu(self):
        with tempfile.TemporaryDirectory() as model_dir:
            with mock.patch.object(helper, "_make_tpu_strategy") as make_strategy:
                with self.assertRaises(FileNotFoundError):
                    helper.run_tpu(model_dir,
                                   "/tmp/neuriplo-tasks-missing-image.jpg")
            make_strategy.assert_not_called()

    def test_run_tpu_does_not_wrap_file_errors_as_tpu_failures(self):
        with tempfile.TemporaryDirectory() as model_dir:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as image:
                image.write(b"image")
                image_name = image.name
            with mock.patch.object(helper, "_require_tensorflow",
                                   return_value=types.SimpleNamespace()), \
                    mock.patch.object(helper, "_make_tpu_strategy",
                                      return_value=mock.Mock()), \
                    mock.patch.object(helper, "_run_logits",
                                      side_effect=FileNotFoundError("gone")):
                with self.assertRaises(FileNotFoundError):
                    helper.run_tpu(model_dir, image_name)

    def test_run_tpu_does_not_wrap_os_errors_as_tpu_failures(self):
        with tempfile.TemporaryDirectory() as model_dir:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as image:
                image.write(b"image")
                image_name = image.name
            with mock.patch.object(helper, "_require_tensorflow",
                                   return_value=types.SimpleNamespace()), \
                    mock.patch.object(helper, "_make_tpu_strategy",
                                      return_value=mock.Mock()), \
                    mock.patch.object(helper, "_run_logits",
                                      side_effect=OSError("corrupt")):
                with self.assertRaises(OSError):
                    helper.run_tpu(model_dir, image_name)

    def test_run_tpu_attaches_m1_evidence(self):
        with tempfile.TemporaryDirectory() as model_dir:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as image:
                image.write(b"image")
                image_name = image.name
            strategy = mock.Mock(num_replicas_in_sync=8)
            fake_tf = types.SimpleNamespace(
                __version__="2.19.1",
                config=types.SimpleNamespace(
                    list_logical_devices=mock.Mock(return_value=[
                        types.SimpleNamespace(name="/device:TPU:0"),
                        types.SimpleNamespace(name="/device:TPU:1"),
                    ])
                ),
            )
            with mock.patch.object(helper, "_require_tensorflow",
                                   return_value=fake_tf), \
                    mock.patch.object(helper, "_make_tpu_strategy",
                                      return_value=strategy), \
                    mock.patch.object(helper, "_run_logits",
                                      return_value={"top5": [], "image": "x",
                                                    "sha256": "abc"}):
                result = helper.run_tpu(model_dir, image_name)
        evidence = result["evidence"]
        self.assertEqual(result["device"], "TPU")
        self.assertEqual(evidence["saved_model"], model_dir)
        self.assertEqual(evidence["tensorflow_version"], "2.19.1")
        self.assertEqual(evidence["tpu_devices"],
                         ["/device:TPU:0", "/device:TPU:1"])
        self.assertEqual(evidence["num_replicas_in_sync"], 8)
        self.assertIn("libtpu_version", evidence)
        self.assertIn("tpu_type", evidence)


def _fake_tf(*, resolver_factory=None, initialize=None,
             strategy_factory=None, devices=None):
    if resolver_factory is None:
        cluster_resolver = types.SimpleNamespace()  # no TPUClusterResolver
    else:
        cluster_resolver = types.SimpleNamespace(
            TPUClusterResolver=resolver_factory
        )
    return types.SimpleNamespace(
        distribute=types.SimpleNamespace(
            cluster_resolver=cluster_resolver,
            TPUStrategy=strategy_factory or mock.Mock(),
            ReduceOp=types.SimpleNamespace(MEAN="MEAN"),
        ),
        config=types.SimpleNamespace(
            experimental_connect_to_cluster=mock.Mock(),
            list_logical_devices=mock.Mock(return_value=devices or []),
        ),
        tpu=types.SimpleNamespace(
            experimental=types.SimpleNamespace(
                initialize_tpu_system=initialize or mock.Mock()
            )
        ),
    )


class MockedTpuStrategyTests(unittest.TestCase):
    def test_missing_cluster_resolver_is_explicit(self):
        fake = _fake_tf()
        with self.assertRaises(helper.ValidationError):
            helper._make_tpu_strategy(fake)
        fake.distribute.TPUStrategy.assert_not_called()

    def test_resolver_failure_is_explicit(self):
        resolver = mock.Mock(side_effect=RuntimeError("no TPU name"))
        fake = _fake_tf(resolver_factory=resolver)
        with self.assertRaises(helper.ValidationError):
            helper._make_tpu_strategy(fake)
        fake.distribute.TPUStrategy.assert_not_called()

    def test_initialize_failure_is_explicit(self):
        resolver = mock.Mock(return_value=object())
        initialize = mock.Mock(side_effect=RuntimeError("init boom"))
        fake = _fake_tf(resolver_factory=resolver, initialize=initialize)
        with self.assertRaises(helper.ValidationError):
            helper._make_tpu_strategy(fake)
        fake.distribute.TPUStrategy.assert_not_called()

    def test_no_logical_tpu_devices_is_explicit(self):
        resolver = mock.Mock(return_value=object())
        strategy = mock.Mock()
        fake = _fake_tf(resolver_factory=resolver,
                        strategy_factory=mock.Mock(return_value=strategy),
                        devices=[])
        with self.assertRaises(helper.ValidationError):
            helper._make_tpu_strategy(fake)

    def test_successful_strategy_is_returned(self):
        resolver = mock.Mock(return_value=object())
        strategy = mock.Mock()
        fake = _fake_tf(resolver_factory=resolver,
                        strategy_factory=mock.Mock(return_value=strategy),
                        devices=[types.SimpleNamespace(name="/device:TPU:0")])
        self.assertIs(helper._make_tpu_strategy(fake), strategy)

    def test_strategy_construction_failure_is_explicit(self):
        resolver = mock.Mock(return_value=object())
        fake = _fake_tf(
            resolver_factory=resolver,
            strategy_factory=mock.Mock(side_effect=RuntimeError("strategy boom")),
            devices=[object()],
        )
        with self.assertRaises(helper.ValidationError):
            helper._make_tpu_strategy(fake)


@unittest.skipUnless(HAS_TF, "requires TensorFlow")
class MetadataWithTensorFlowTests(unittest.TestCase):
    def _export_tiny(self, path, height=4, width=6, classes=2):
        exporter_spec = importlib.util.spec_from_file_location(
            "tensorflow_export",
            Path(__file__).resolve().parents[1]
            / "export/classification/tensorflow/export_tf_saved_model_classifier.py",
        )
        exporter = importlib.util.module_from_spec(exporter_spec)
        exporter_spec.loader.exec_module(exporter)
        inputs = tf.keras.Input(shape=(height, width, 3))
        features = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        outputs = tf.keras.layers.Dense(classes, use_bias=False)(features)
        exporter.export_classifier(tf.keras.Model(inputs, outputs), path)

    def test_metadata_accepts_matching_contract(self):
        with tempfile.TemporaryDirectory() as path:
            self._export_tiny(path)
            result = helper.check_metadata(path, image_shape=(1, 4, 6, 3),
                                           num_classes=2)
        self.assertEqual(result["input"]["dtype"], "float32")
        self.assertEqual(len(result["input"]["shape"]), 4)
        self.assertEqual(result["output"]["dtype"], "float32")
        self.assertEqual(len(result["output"]["shape"]), 2)

    def test_metadata_rejects_shape_mismatch(self):
        with tempfile.TemporaryDirectory() as path:
            self._export_tiny(path)
            with self.assertRaises(helper.ValidationError):
                helper.check_metadata(path, image_shape=(1, 224, 224, 3),
                                      num_classes=1000)

    def test_cpu_executor_runs_tiny_model(self):
        with tempfile.TemporaryDirectory() as path:
            self._export_tiny(path, height=4, width=6, classes=3)
            pixels = np.zeros((1, 4, 6, 3), dtype=np.float32)
            logits = helper._cpu_executor(path, pixels)
            array = np.asarray(logits, dtype=np.float32)
        self.assertEqual(array.shape, (1, 3))
        selected = helper._select_output_tensor({"output_0": logits})
        self.assertEqual(np.asarray(selected).shape, (1, 3))

    def test_cpu_reference_end_to_end(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as path:
            self._export_tiny(path, height=4, width=6, classes=3)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                Image.fromarray(np.zeros((4, 6, 3), np.uint8), "RGB").save(handle.name)
                image_name = handle.name
            result = helper.run_cpu_reference(path, image_name,
                                              image_shape=(1, 4, 6, 3),
                                              num_classes=3)
        self.assertEqual(result["device"], "CPU")
        self.assertEqual(len(result["top5"]), 3)
        self.assertEqual(len(result["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
