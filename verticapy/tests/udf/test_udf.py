# (c) Copyright [2018-2020] Micro Focus or one of its affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest, math, os, verticapy
from verticapy.udf import *

def normalize_titanic(age, fare):
    return (age - 30.15) / 14.44, (fare - 33.96) / 52.65

class TestUdf:

    def test_create_lib_udf(self):
        file_path = os.path.dirname(verticapy.__file__) + "/python_math_lib.py"
        pmath_path = os.path.dirname(verticapy.__file__) + "/tests/udf/pmath.py"
        udx_str, udx_sql = create_lib_udf([(math.exp, [float], float, {}, "python_exp"),
                                           (math.isclose, [float, float], bool, {"abs_tol": float}, "python_isclose"),
                                           (normalize_titanic, [float, float], {"norm_age": float, "norm_fare": float}, {}, "python_norm_titanic"),],
                                           library_name = "python_math",
                                           include_dependencies = pmath_path,
                                           file_path = file_path,
                                           create_file = False)
        assert udx_str == "import vertica_sdk\nimport math\nimport verticapy.tests.udf.test_udf\n\ndef normalize_titanic(age, fare):\n    return (age - 30.15) / 14.44, (fare - 33.96) / 52.65\n\nclass verticapy_python_exp(vertica_sdk.ScalarFunction):\n\n\tdef setup(self, server_interface, col_types):\n\t\tself.params = {}\n\n\tdef processBlock(self, server_interface, arg_reader, res_writer):\n\t\twhile(True):\n\t\t\tinputs  = []\n\t\t\tinputs += [arg_reader.getFloat(0)]\n\t\t\tresult = math.exp(*inputs, **self.params)\n\t\t\tres_writer.setFloat(result)\n\t\t\tres_writer.next()\n\t\t\tif not arg_reader.next():\n\t\t\t\tbreak\n\n\tdef destroy(self, server_interface, col_types):\n\t\tpass\n\nclass verticapy_python_exp_factory(vertica_sdk.ScalarFunctionFactory):\n\n\tdef createScalarFunction(self, srv):\n\t\treturn verticapy_python_exp()\n\n\tdef getPrototype(self, server_interface, arg_types, return_type):\n\t\targ_types.addFloat()\n\t\treturn_type.addFloat()\n\n\tdef getReturnType(self, server_interface, arg_types, return_type):\n\t\treturn_type.addFloat()\n\n\nclass verticapy_python_isclose(vertica_sdk.ScalarFunction):\n\n\tdef setup(self, server_interface, col_types):\n\t\tparams = server_interface.getParamReader()\n\t\tself.params = {}\n\t\tif params.containsParameter('abs_tol'):\n\t\t\tself.params['abs_tol'] = params.getFloat('abs_tol')\n\n\tdef processBlock(self, server_interface, arg_reader, res_writer):\n\t\twhile(True):\n\t\t\tinputs  = []\n\t\t\tinputs += [arg_reader.getFloat(0)]\n\t\t\tinputs += [arg_reader.getFloat(1)]\n\t\t\tresult = math.isclose(*inputs, **self.params)\n\t\t\tres_writer.setBool(result)\n\t\t\tres_writer.next()\n\t\t\tif not arg_reader.next():\n\t\t\t\tbreak\n\n\tdef destroy(self, server_interface, col_types):\n\t\tpass\n\nclass verticapy_python_isclose_factory(vertica_sdk.ScalarFunctionFactory):\n\n\tdef createScalarFunction(self, srv):\n\t\treturn verticapy_python_isclose()\n\n\tdef getPrototype(self, server_interface, arg_types, return_type):\n\t\targ_types.addFloat()\n\t\targ_types.addFloat()\n\t\treturn_type.addBool()\n\n\tdef getReturnType(self, server_interface, arg_types, return_type):\n\t\treturn_type.addBool()\n\n\tdef getParameterType(self, server_interface, parameterTypes):\n\t\tparameterTypes.addFloat('abs_tol')\n\nclass verticapy_python_norm_titanic(vertica_sdk.TransformFunction):\n\n\tdef setup(self, server_interface, col_types):\n\t\tself.params = {}\n\n\tdef processPartition(self, server_interface, arg_reader, res_writer):\n\t\twhile(True):\n\t\t\tinputs  = []\n\t\t\tinputs += [arg_reader.getFloat(0)]\n\t\t\tinputs += [arg_reader.getFloat(1)]\n\t\t\tresult = verticapy.tests.udf.test_udf.normalize_titanic(*inputs, **self.params)\n\t\t\tif len(result) == 1:\n\t\t\t\tresult = result[0]\n\t\t\tres_writer.setFloat(0, result[0])\n\t\t\tres_writer.setFloat(1, result[1])\n\t\t\tres_writer.next()\n\t\t\tif not arg_reader.next():\n\t\t\t\tbreak\n\n\tdef destroy(self, server_interface, col_types):\n\t\tpass\n\nclass verticapy_python_norm_titanic_factory(vertica_sdk.TransformFunctionFactory):\n\n\tdef createTransformFunction(self, srv):\n\t\treturn verticapy_python_norm_titanic()\n\n\tdef getPrototype(self, server_interface, arg_types, return_type):\n\t\targ_types.addFloat()\n\t\targ_types.addFloat()\n\t\treturn_type.addFloat()\n\t\treturn_type.addFloat()\n\n\tdef getReturnType(self, server_interface, arg_types, return_type):\n\t\treturn_type.addFloat('norm_age')\n\t\treturn_type.addFloat('norm_fare')\n\n"
        assert udx_sql == [f"CREATE OR REPLACE LIBRARY python_math AS '{file_path}' LANGUAGE 'Python';",
                           "CREATE OR REPLACE FUNCTION python_exp AS NAME 'verticapy_python_exp_factory' LIBRARY python_math;",
                           "CREATE OR REPLACE FUNCTION python_isclose AS NAME 'verticapy_python_isclose_factory' LIBRARY python_math;",
                           "CREATE OR REPLACE TRANSFORM FUNCTION python_norm_titanic AS NAME 'verticapy_python_norm_titanic_factory' LIBRARY python_math;"]
        