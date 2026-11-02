"""Regression tests for preservation of trained classifier weights during export."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

spec = importlib.util.spec_from_file_location(
    "classifier_export",
    Path(__file__).resolve().parents[1] / "export/classification/torchvision/export_torchvision_classifier.py",
)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 2)

    def forward(self, x):
        return self.fc(x)


class ExportTests(unittest.TestCase):
    def test_both_formats_preserve_trained_head(self):
        for kind in ("torchscript", "onnx"):
            with self.subTest(kind=kind):
                model = Classifier()
                head = model.fc
                before = {key: value.clone() for key, value in model.state_dict().items()}
                with patch.object(torch.jit, "script"), patch.object(torch.onnx, "export"):
                    export = getattr(exporter, "export_to_" + kind)
                    export(model, "unused", 2, (1, 3), "images", "logits")
                self.assertIs(model.fc, head)
                self.assertFalse(model.training)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_mismatched_classes_fail_before_export(self):
        with patch.object(torch.onnx, "export") as export:
            with self.assertRaisesRegex(ValueError, "trained classifier"):
                exporter.export_to_onnx(Classifier(), "unused", 3, (1, 3), "images", "logits")
            export.assert_not_called()


if __name__ == "__main__":
    unittest.main()
