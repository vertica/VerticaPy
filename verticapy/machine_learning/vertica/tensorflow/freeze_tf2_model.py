#!/usr/bin/env python3

###########################################################################################
# This script should be run in the python environment where you train your models.
#
# This script ingests a saved TensorFlow 2 model and outputs a folder with a frozen model
# and a model description json file. The format of the model description file is based on
# the input/output Vertica column type in prediction. Copy this folder to your Vertica
# cluster and use the IMPORT_MODELS function to import it. Use PREDICT_TENSORFLOW for
# primitive input/output columns and PREDICT_TENSORFLOW_SCALAR for complex input/output
# columns. See documentation for more details.
#
# To see how to use this script, run it without arguments.
###########################################################################################

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (  # pylint: disable=no-name-in-module
    convert_variables_to_constants_v2,
)
import numpy as np
import json, os, sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # suppress some TF noise


# https://www.tensorflow.org/api_docs/python/tf/dtypes
def get_str_from_dtype(dtype, is_input, idx):
    dtype_to_string = {
        tf.float16: "TF_HALF",
        tf.float32: "TF_FLOAT",
        tf.half: "TF_HALF",
        tf.float64: "TF_DOUBLE",
        tf.double: "TF_DOUBLE",
        tf.int8: "TF_INT8",
        tf.int16: "TF_INT16",
        tf.int32: "TF_INT32",
        tf.int64: "TF_INT64",
    }

    if dtype in dtype_to_string:
        dtype_str = dtype_to_string[dtype]
    else:
        print(
            "Only floats, doubles, and signed ints are currently supported as model inputs/outputs in Vertica, please modify your model."
        )
        sys.exit()

    in_or_out = "Input" if is_input else "Output"
    print(in_or_out, str(idx), "is of type:", dtype_str)

    return dtype_str


def freeze_model(model, save_dir, column_type):
    # shared parameters
    frozen_out_path = save_dir
    frozen_graph_file = "frozen_graph" + ".pb"  # name of the .pb file
    tf_model_desc_file = "tf_model_desc.json"

    if column_type == "0":  # standard (primitive) column types
        # Convert Keras model to ConcreteFunction
        full_model = tf.function(lambda x: model(x))

        print("Model Input:", model.input)

        if isinstance(model.input, list):
            no_of_inputs = len(model.input)
        else:
            no_of_inputs = 1

        tensor_input = []
        for i in range(no_of_inputs):
            tensor_input.append(
                tf.TensorSpec(
                    shape=[None] + model.input[i].shape.as_list(),
                    dtype=model.input[i].dtype,
                )
            )

        full_model = full_model.get_concrete_function((tensor_input))

        # Note: If this line is failing, you may need to change the argument that is being passed into
        # get_concrete_function. The input should be one or more TensorSpecs, and should match the inputs
        # to your model. Sometimes, models with custom functions do not have model.inputs/outputs and you
        # need to provide them yourself. For example, if you have a function that accepts multiple inputs,
        # you may need to write something like ('None' acts as a wildcard for the shape argument):
        #
        # full_model = full_model.get_concrete_function(
        #    (tf.TensorSpec(shape=None, dtype=tf.float32),
        #    tf.TensorSpec(shape=None, dtype=tf.int64),
        #    ...
        #    ))

        # Get frozen graph def
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()

        save_graph(frozen_func, frozen_out_path, frozen_graph_file)

        # Below, we assume that for multi input/output models col_start will be the same as input/output
        # index. Example, if you have 10 inputs and each input is a simple 1-dimensional tensor, then
        # each input must increment the col_start value, because if we always set the col_start = 0 then
        # each Tensor will be filled from the first column and have the same value.

        inputs = []
        col_start = 0
        for input_idx, inp in enumerate(frozen_func.inputs):
            # print(inp.op.name)
            # print(inp.get_shape())
            input_dims = [1 if e is None else e for e in list(inp.get_shape())]
            dtype = get_str_from_dtype(inp.dtype, True, input_idx)
            inputs.append(
                {
                    "op_name": inp.op.name,
                    "tensor_map": [
                        {
                            "idx": inp.value_index,
                            "dim": input_dims,
                            "col_start": col_start,
                            "data_type": dtype,
                        }
                    ],
                }
            )
            col_start += flat_dim(input_dims)

        outputs = []
        col_start = 0
        for output_idx, output in enumerate(frozen_func.outputs):
            # print(output.op.name)
            # print(output.get_shape())
            output_dims = [1 if e is None else e for e in list(output.get_shape())]
            dtype = get_str_from_dtype(output.dtype, False, output_idx)
            outputs.append(
                {
                    "op_name": output.op.name,
                    "tensor_map": [
                        {
                            "idx": output.value_index,
                            "dim": output_dims,
                            "col_start": col_start,
                            "data_type": dtype,
                        }
                    ],
                }
            )
            col_start += flat_dim(output_dims)

        model_info = {
            "frozen_graph": frozen_graph_file,
            "input_desc": inputs,
            "output_desc": outputs,
        }

        save_desc_file(model_info, frozen_out_path, tf_model_desc_file)

    elif column_type == "1":  # new row-type column type, aka array complex data type
        input_tensors = []
        if hasattr(model, "signatures"):
            input_tensors = model.signatures["serving_default"].inputs
        else:
            input_tensors = (
                model.inputs
            )  # if a model hasn't been saved it may not have any signatures
        input_tensors = list(
            filter(lambda t: t.shape.rank > 0, input_tensors)
        )  # remove resource-type tensors

        # Convert the TF model to a concrete function
        var_func = tf.function(lambda x: model(x))
        var_func = var_func.get_concrete_function(
            [tf.TensorSpec(t.shape.as_list(), t.dtype.name) for t in input_tensors]
        )

        # Generate the frozen graph file
        const_func = convert_variables_to_constants_v2(var_func)

        save_graph(const_func, frozen_out_path, frozen_graph_file)

        # Generate the model json file
        model_info = {
            "column_type": "complex",
            "frozen_graph": frozen_graph_file,
            "input_tensors": gen_tensor_list(const_func.inputs),
            "output_tensors": gen_tensor_list(const_func.outputs),
        }

        save_desc_file(model_info, frozen_out_path, tf_model_desc_file)

    else:
        print("Unrecognized column type flag")
        sys.exit()


def freeze_model_from_file(
    saved_model_path, save_dir="frozen_tfmodel", column_type="0"
):
    frozen_out_path = os.path.join(saved_model_path, save_dir)
    model = tf.keras.models.load_model(saved_model_path)
    freeze_model(model, frozen_out_path, column_type)


def save_graph(frozen_func, frozen_out_path, frozen_graph_file):
    print("Saving frozen model to: " + os.path.join(frozen_out_path, frozen_graph_file))
    tf.io.write_graph(
        graph_or_graph_def=frozen_func.graph,
        logdir=frozen_out_path,
        name=frozen_graph_file,
        as_text=False,
    )


def save_desc_file(model_info, frozen_out_path, tf_model_desc_file):
    print(
        "Saving model description file to: "
        + os.path.join(frozen_out_path, tf_model_desc_file)
    )
    with open(os.path.join(frozen_out_path, tf_model_desc_file), "w") as json_file:
        json.dump(model_info, json_file, indent=4, sort_keys=False)


def flat_dim(dims):
    flat_dim = 1
    for dim in dims:
        flat_dim *= dim
    return flat_dim


def gen_tensor_list(tensors):
    res = []
    for t in tensors:
        res.append(
            {
                "name": t.name,
                "data_type": "double" if t.dtype.name == "float64" else t.dtype.name,
                "dims": [-1 if e is None else e for e in t.shape.as_list()],
            }
        )
    return res


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 4 or sys.argv[1] in ["-h", "--help"]:
        print("There are three arguments to the script:")
        print("1. Path to your saved model directory")
        print(
            "2. (Optional) name of the folder to save the frozen model (default: frozen_tfmodel)"
        )
        print(
            "3. (Optional) Input/output Vertica column type in prediction (0 (default): primitive; 1: complex)"
        )
        print("   Use primitive if you want one value stored in each row/column cell.")
        print("   Use complex if you want to store the data in Vertica arrays.")
        print(
            "Example call: ./freeze_tf_model.py path/to/saved/model my_frozen_model 0"
        )
        sys.exit()

    saved_model_path = sys.argv[1]

    if len(sys.argv) == 2:
        freeze_model_from_file(str(sys.argv[1]))
    elif len(sys.argv) == 3:
        freeze_model_from_file(str(sys.argv[1]), str(sys.argv[2]))
    elif len(sys.argv) == 4:
        freeze_model_from_file(str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]))
    else:
        print("Invalid number of arguments.")  # unreachable, just here for completeness
