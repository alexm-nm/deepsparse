# Tool to set input dimensions of an ONNX model to static values
# This is needed since frameworks like TensorFlow can export models with dynamic input sizes
#
# Example usage: You want to set the input of a model to dimensions [1,227,227,3]
# python utils/set_onnx_input_dims.py ~/Downloads/model.onnx -i 1 227 227 3

import argparse
import onnx
import os


parser = argparse.ArgumentParser()
parser.add_argument(
    "onnx_filename",
    help="The full filepath of the onnx model file",
)
parser.add_argument(
    "-i",
    "--input_shapes",
    type=str,
    nargs="+",
    action="append",
    help="Input shapes i.e. for three inputs of [1,128] do '-i 1 128 -i 1 128 -i 1 128'",
)

args = parser.parse_args()
onnx_filename = args.onnx_filename
input_shapes = args.input_shapes


model = onnx.load(onnx_filename)
all_inputs = model.graph.input
initializer_input_names = [node.name for node in model.graph.initializer]
external_inputs = [
    input for input in all_inputs if input.name not in initializer_input_names
]

for input_index, graph_input in enumerate(external_inputs):
    tensor_type = graph_input.type.tensor_type
    in_shape = [int(dim) for dim in input_shapes[input_index]]

    for dim_index, dim in enumerate(in_shape):
        print(
            f"Setting dim #{dim_index} to {dim} (original value {tensor_type.shape.dim[dim_index].dim_value})"
        )
        tensor_type.shape.dim[dim_index].dim_value = dim

name, ext = os.path.splitext(onnx_filename)
shaped_filename = f"{name}_static{ext}"
print(f"Saving static input onnx to: {shaped_filename}")
onnx.save(model, shaped_filename)
