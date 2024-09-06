"""
Copyright  (c)  2018-2024 Open Text  or  one  of its
affiliates.  Licensed  under  the   Apache  License,
Version 2.0 (the  "License"); You  may  not use this
file except in compliance with the License.

You may obtain a copy of the License at:
http://www.apache.org/licenses/LICENSE-2.0

Unless  required  by applicable  law or  agreed to in
writing, software  distributed  under the  License is
distributed on an  "AS IS" BASIS,  WITHOUT WARRANTIES
OR CONDITIONS OF ANY KIND, either express or implied.
See the  License for the specific  language governing
permissions and limitations under the License.
"""

# Pytest
import pytest

# Standard Python Modules
import math, os

# VerticaPy
import verticapy
from verticapy.udf import generate_lib_udf


def scale_titanic(age, fare):
    return (age - 30.15) / 14.44, (fare - 33.96) / 52.65


class TestUdf:
    def test_generate_lib_udf(self):
        file_path = os.path.dirname(verticapy.__file__) + "/python_math_lib.py"
        pmath_path = os.path.dirname(verticapy.__file__) + "/tests/udf/pmath.py"
        udx_str, udx_sql = generate_lib_udf(
            [
                (math.exp, [float], float, {}, "python_exp"),
                (
                    math.isclose,
                    [float, float],
                    bool,
                    {"abs_tol": float},
                    "python_isclose",
                ),
                (
                    scale_titanic,
                    [float, float],
                    {"norm_age": float, "norm_fare": float},
                    {},
                    "python_norm_titanic",
                ),
            ],
            library_name="python_math",
            include_dependencies=pmath_path,
            file_path=file_path,
            create_file=False,
        )
        assert udx_str.split("\n")[0] == "import vertica_sdk"
        assert (
            udx_str.split("\n")[10] == "\tdef setup(self, server_interface, col_types):"
        )
        assert udx_sql == [
            f"CREATE OR REPLACE LIBRARY python_math AS '{file_path}' LANGUAGE 'Python';",
            "CREATE OR REPLACE FUNCTION python_exp AS NAME 'verticapy_python_exp_factory' LIBRARY python_math;",
            "CREATE OR REPLACE FUNCTION python_isclose AS NAME 'verticapy_python_isclose_factory' LIBRARY python_math;",
            "CREATE OR REPLACE TRANSFORM FUNCTION python_norm_titanic AS NAME 'verticapy_python_norm_titanic_factory' LIBRARY python_math;",
        ]
