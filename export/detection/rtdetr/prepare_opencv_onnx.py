#!/usr/bin/env python3
"""Adapt constant-vector MatMul to OpenCV's matrix-only implementation.

Writes a separate ONNX artifact. External model inputs, outputs and dtypes stay
unchanged. Requires the existing ONNX export dependency, not a C++ dependency.
"""
import argparse
from pathlib import Path
import onnx
from onnx import helper


def prepare(model):
    onnx.checker.check_model(model)
    opset = next((item.version for item in model.opset_import if item.domain == ''), 0)
    if opset < 13:
        raise ValueError('OpenCV MatMul preparation requires standard ONNX opset >= 13')
    vectors = {item.name for item in model.graph.initializer if len(item.dims) == 1}
    # Graph-input initializers may be overridden at execution time; do not
    # assume those are constants.
    vectors.difference_update(item.name for item in model.graph.input)
    names = {name for node in model.graph.node for name in [node.name, *node.input, *node.output]}
    names.update(item.name for item in model.graph.initializer)
    names.update(item.name for item in [*model.graph.input, *model.graph.output, *model.graph.value_info])
    nodes = []
    changed = 0
    counter = 0
    for node in model.graph.node:
        if node.domain or node.op_type != 'MatMul' or node.input[1] not in vectors:
            nodes.append(node)
            continue
        while True:
            base = f'__opencv_vector_matmul_{counter}'
            counter += 1
            reserved = [base + suffix for suffix in ('_axes', '_column', '_product', '_expand', '_matmul', '_squeeze')]
            if not names.intersection(reserved):
                names.update(reserved)
                break
        axes, column, product, expand_name, matmul_name, squeeze_name = reserved
        model.graph.initializer.append(helper.make_tensor(axes, onnx.TensorProto.INT64, [1], [-1]))
        nodes.append(helper.make_node('Unsqueeze', [node.input[1], axes], [column], name=expand_name))
        matmul = onnx.NodeProto()
        matmul.CopyFrom(node)
        matmul.name = matmul_name
        matmul.input[1] = column
        matmul.output[0] = product
        nodes.append(matmul)
        nodes.append(helper.make_node('Squeeze', [product, axes], list(node.output), name=squeeze_name))
        changed += 1
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if (args.input.resolve() == args.output.resolve()
            or (args.output.exists() and args.output.samefile(args.input))):
        parser.error('output must differ from input; keep the original artifact')
    model = onnx.load(args.input)
    changed = prepare(model)
    onnx.save(model, args.output)
    print(f'Prepared {changed} vector MatMul nodes: {args.output}')


if __name__ == '__main__':
    main()
