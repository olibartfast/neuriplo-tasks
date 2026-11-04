#!/usr/bin/env python3
import unittest
import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator
from prepare_opencv_onnx import prepare


def model(shape=(2, 3), opset=13):
    graph = helper.make_graph([
        helper.make_node('MatMul', ['image', 'weight'], ['output']),
        helper.make_node('Identity', ['weight'], ['unchanged_weight']),
    ], 'vector', [helper.make_tensor_value_info('image', TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, list(shape[:-1])),
         helper.make_tensor_value_info('unchanged_weight', TensorProto.FLOAT, [3])],
        [helper.make_tensor('weight', TensorProto.FLOAT, [3], [0.5, -2, 3]),
         helper.make_tensor('__opencv_vector_matmul_0_axes', TensorProto.INT64, [1], [0])])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', opset)], ir_version=8)


class PrepareTest(unittest.TestCase):
    def test_parity_contract_and_shared_initializer(self):
        for shape in ((2, 3), (2, 4, 3), (3,)):
            with self.subTest(shape=shape):
                original = model(shape)
                adapted = onnx.ModelProto()
                adapted.CopyFrom(original)
                self.assertEqual(prepare(adapted), 1)
                self.assertEqual(adapted.graph.input, original.graph.input)
                self.assertEqual(adapted.graph.output, original.graph.output)
                image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
                expected = ReferenceEvaluator(original).run(None, {'image': image})
                actual = ReferenceEvaluator(adapted).run(None, {'image': image})
                for before, after in zip(expected, actual):
                    np.testing.assert_array_equal(before, after)
                self.assertEqual(prepare(adapted), 0)
                self.assertTrue(any(node.name.startswith('__opencv_vector_matmul_1_') for node in adapted.graph.node))

    def test_rejects_old_opset(self):
        with self.assertRaisesRegex(ValueError, 'opset'):
            prepare(model(opset=12))

    def test_leaves_overridable_initializer_unchanged(self):
        graph = model()
        graph.graph.input.append(helper.make_tensor_value_info('weight', TensorProto.FLOAT, [3]))
        self.assertEqual(prepare(graph), 0)


if __name__ == '__main__':
    unittest.main()
