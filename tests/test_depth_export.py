"""Exercise the real ONNX exporter without downloading HuggingFace weights."""
from collections import namedtuple
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import torch

spec = importlib.util.spec_from_file_location(
    "depth_export",
    Path(__file__).resolve().parents[1] / "export/depth_estimation/depth_anything_v2/export_depth_anything_v2_to_onnx.py",
)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)
DepthOutput = namedtuple("DepthOutput", "predicted_depth")


class TinyDepth(torch.nn.Module):
    config = types.SimpleNamespace(hidden_size=3)

    def forward(self, pixels):
        return DepthOutput(torch.nn.functional.gelu(pixels).mean(dim=1))


class DepthExportTests(unittest.TestCase):
    def test_native_gelu_and_compatible_opset_keep_depth_values(self):
        transformers = types.ModuleType("transformers")
        transformers.__version__ = "test-fixture"
        transformers.__spec__ = importlib.util.spec_from_loader("transformers", loader=None)
        transformers.AutoModelForDepthEstimation = types.SimpleNamespace(from_pretrained=Mock(return_value=TinyDepth()))
        transformers.AutoImageProcessor = types.SimpleNamespace(from_pretrained=Mock(return_value=object()))
        pixels = np.linspace(-3, 3, 72, dtype=np.float32).reshape(1, 3, 4, 6)
        expected = np.array([
            0.5 * float(x) * (1 + math.erf(float(x) / math.sqrt(2)))
            for x in pixels.flat
        ]).reshape(pixels.shape).mean(axis=1)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"transformers": transformers}):
            for opset in (18, 20):
                with self.subTest(opset=opset):
                    path = str(Path(directory) / f"depth-{opset}.onnx")
                    result = exporter.export_depth_anything_v2_to_onnx(
                        output_path=path, image_height=4, image_width=6,
                        dynamic_batch=False, **({"opset_version": 20} if opset == 20 else {}),
                    )
                    self.assertEqual(result, path)
                    graph = onnx.load(path)
                    self.assertEqual(graph.opset_import[0].version, opset)
                    if opset == 20:
                        self.assertIn("Gelu", [node.op_type for node in graph.graph.node])
                    actual = ReferenceEvaluator(graph).run(None, {"pixel_values": pixels})[0]
                    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
