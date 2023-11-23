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
import os
import warnings
from typing import Union

from verticapy._utils._sql._collect import save_verticapy_logs

from verticapy.sql.sys import _executeSQL, current_session, username

from verticapy.sdk.vertica.udf.gen import generate_lib_udf


@save_verticapy_logs
def import_lib_udf(
    udf_list: list,
    library_name: str,
    include_dependencies: Union[None, str, list[str]] = None,
) -> bool:
    """
    Install a library of Python functions in Vertica.
    This function will only work when it is executed
    directly in the server.

    Parameters
    ----------
    udf_list: list
        List of tuples that includes the different functions.

        **function**     :
                            [function]  Python   Function.

        **arg_types**    :
                            [dict/list] List or dictionary
                            of  the function input  types.

                            Example: {"input1": int,
                            "input2": float}  or
                            [int, float]

        **return_type**  :
                            [type/dict] Function output type.
                            In the case of many  outputs, it
                            must be a dictionary including
                            all the outputs types and names.

                            Example: {"result1": int,
                            "result2": float}

        **parameters**   :
                            [dict] Dictionary of the function
                            input optional parameters.

                            Example: {"param1": int,
                            "param2": str}

        **new_name**     :
                            [str] New   function   name  when
                            installed in Vertica.

    library_name: str
        Library Name.
    include_dependencies: str / list, optional
        Library files  dependencies. The function copies and
        pastes the  different files in the  UDF  definition.

    Returns
    -------
    bool
        True  if  the  installation  was  a  success,  False
        otherwise.

    Example
    -------
    Import the math module. This example will use the `math.exp` and `math.isclose` functions:

    .. code-block:: python

        import math

    .. important:: Python is type-agnostic, but Vertica requires specific data types. It's important to specify input, output, and parameter types when generating User-Defined Extensions (UDx). These functions will be automatically installed, allowing you to call them directly using SQL.

    .. code-block:: python

            from verticapy.sdk.vertica.udf import import_lib_udf

            import_lib_udf(
                [
                    (math.exp, [float], float, {}, "python_exp"),
                    (math.isclose, [float, float], bool, {"abs_tol": float}, "python_isclose"),
                ],
                library_name = "python_math"
            )

    .. important::  In this example, we utilized a standard Python function. If you wish to use a non-standard function, you'll need to install it on each node individually.
    """
    directory = os.path.expanduser("~")
    session_name = f"{current_session()}_{username()}"
    file_name = f"{library_name}_{session_name}.py"
    if os.path.exists(f"{directory}/{file_name}"):
        os.remove(f"{directory}/{file_name}")
    udx_str, sql = generate_lib_udf(
        udf_list, library_name, include_dependencies, f"{directory}/{file_name}"
    )
    with open(f"{directory}/{file_name}", "w", encoding="utf-8") as f:
        f.write(udx_str)
    try:
        for idx, query in enumerate(sql):
            _executeSQL(query, title=f"UDF installation. [step {idx}]")
        return True
    except Exception as e:
        warnings.warn(e, Warning)
        return False
    finally:
        os.remove(f"{directory}/{file_name}")
