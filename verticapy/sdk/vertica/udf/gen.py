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
from typing import Optional, Union

from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import format_type

from verticapy.sdk.vertica.udf.utils import get_set_add_function


@save_verticapy_logs
def generate_lib_udf(
    udf_list: list,
    library_name: str,
    include_dependencies: Union[None, str, list[str]] = None,
    file_path: Optional[str] = None,
    create_file: bool = False,
) -> tuple[str, str]:
    """
    Generates the code needed to install a library of
    Python functions.  It uses the Vertica SDK to
    create UDFs of the input functions.

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
        Library files dependencies. The function copies and
        pastes the different files in the UDF definition.
    file_path: str, optional
        Path to the UDF file.
    create_file: bool, optional
        If set to True,  instead of returning the str of the UDx,
        the function creates two files:  a UDF py file and
        a SQL file to install it.

    Returns
    -------
    udx_str, sql
        UDF py file, str needed to install the library.

    Example
    -------
    .. ipython:: python

        from verticapy.sdk.vertica.udf import generate_lib_udf
        @suppress
        import math

        udx_str, udx_sql = generate_lib_udf(
            [(math.exp, [float], float, {}, "python_exp"), (math.isclose, [float, float], bool, {"abs_tol": float}, "python_isclose"),],
            library_name = "python_math",
            file_path = "",
            create_file = False)

    Print the generated UDx Python code:

    .. ipython:: python

        print(udx_str)

    Print the SQL statements that install the function:

    .. ipython:: python

        print("\\n".join(udx_sql))
    """
    include_dependencies = format_type(include_dependencies, dtype=list)
    if not isinstance(include_dependencies, (list)):
        raise ValueError(
            "The parameter include_dependencies type must be <list>. "
            f"Found {type(include_dependencies)}."
        )
    if not isinstance(library_name, str):
        raise ValueError(
            f"The parameter library_name type must be <str>. Found {type(library_name)}."
        )
    if not isinstance(file_path, str):
        raise ValueError(
            f"The parameter file_path type must be <str>. Found {type(file_path)}."
        )
    if not isinstance(create_file, bool):
        raise ValueError(
            f"The parameter create_file type must be <bool>. Found {type(create_file)}."
        )
    udx_str = "import vertica_sdk\n"
    all_libraries = [udf[0].__module__ for udf in udf_list]
    all_libraries = list(dict.fromkeys(all_libraries))
    if "__main__" in all_libraries:
        all_libraries.remove("__main__")
    for udf in all_libraries:
        if udf:
            udx_str += f"import {udf}\n"
    if include_dependencies:
        for dep_file_path in include_dependencies:
            if not isinstance(dep_file_path, str):
                raise ValueError(
                    "The parameter include_dependencies type must be <list> of <str>. "
                    f"Found {type(dep_file_path)} inside."
                )
            with open(dep_file_path, "r", encoding="utf-8") as f:
                file_str = f.read()
                exec(file_str)
                udx_str += "\n" + file_str + "\n"
    udx_str += "\n"
    sql = []
    for udf in udf_list:
        tmp = generate_udf(*udf, **{"library_name": library_name})
        udx_str += tmp[0]
        sql += [tmp[1]]
    sql_path = f"{library_name}.sql"
    if file_path:
        sql_path = f"{os.path.dirname(file_path)}/{sql_path}"
    if not file_path:
        file_path = f"verticapy_{library_name}.py"
    sql = [
        f"CREATE OR REPLACE LIBRARY {library_name} AS '{file_path}' LANGUAGE 'Python';"
    ] + sql
    if create_file:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(udx_str)
        with open(sql_path, "w", encoding="utf-8") as f:
            f.write("\n".join(sql))
    else:
        return udx_str, sql


def generate_udf(
    function,
    arg_types: Union[list, dict],
    return_type: Union[type, dict],
    parameters: Optional[dict] = None,
    new_name: Optional[str] = None,
    library_name: Optional[str] = None,
) -> tuple[str, str]:
    """
    Generates the UDx Python code and the SQL
    statements needed to install it.
    """
    parameters = format_type(parameters, dtype=dict)
    if not hasattr(function, "__call__"):
        raise ValueError(
            f"The function parameter must be a Python function. Found {type(function)}."
        )
    if not new_name:
        new_name = function.__name__
    elif not isinstance(new_name, str):
        raise ValueError(
            f"The parameter new_name type must be <str>. Found {type(new_name)}."
        )
    module = function.__module__
    if module == "__main__":
        module = ""
    if not library_name:
        library_name = module
    if module:
        module += "."
    elif not isinstance(library_name, str):
        raise ValueError(
            f"The parameter library_name type must be <str>. Found {type(library_name)}."
        )
    if isinstance(arg_types, dict):
        arg_types = [arg_types[var] for var in arg_types]
    elif not isinstance(arg_types, list):
        raise ValueError(
            f"The arg_types parameter must be a <list> of <types>. Found {type(arg_types)}."
        )
    for idx, dtype in enumerate(arg_types):
        if not isinstance(dtype, type):
            raise ValueError(
                "Each element of arg_types parameter must be a <type>. "
                f"Found {type(dtype)} at index {idx}."
            )
    if isinstance(return_type, dict):
        if len(return_type) == 0:
            raise ValueError(
                "return_type is empty. The returned type must have at least one element."
            )
        elif len(return_type) == 1:
            return_type = [return_type[dtype] for dtype in return_type][0]

    # Main Function
    if isinstance(return_type, dict):
        is_udtf, process_function, ftype = (
            True,
            "processPartition",
            "TransformFunction",
        )
    elif isinstance(return_type, type):
        is_udtf, process_function, ftype = False, "processBlock", "ScalarFunction"
    else:
        raise ValueError(
            "return_type must be the dictionary of the returned types in case of "
            "Transform Function and only a type in case of Scalar Function. Can not"
            f" be of type {type(return_type)}."
        )
    udx_str = f"class verticapy_{new_name}(vertica_sdk.{ftype}):\n\n"

    # setup - For Optional parameters
    udx_str += "\tdef setup(self, server_interface, col_types):\n"
    if parameters:
        udx_str += "\t\tparams = server_interface.getParamReader()\n"
    udx_str += "\t\tself.params = {}\n"
    for param in parameters:
        udx_str += f"\t\tif params.containsParameter('{param}'):\n"
        func = get_set_add_function(parameters[param], func="get")
        udx_str += f"\t\t\tself.params['{param}'] = params.{func}('{param}')\n"
    udx_str += "\n"

    # processPartition / processBlock
    udx_str += (
        f"\tdef {process_function}(self, server_interface, arg_reader, res_writer):\n"
    )
    udx_str += "\t\twhile(True):\n"
    udx_str += "\t\t\tinputs  = []\n"
    for idx, var_type in enumerate(arg_types):
        func = get_set_add_function(var_type, func="get")
        func += f"({idx})"
        udx_str += f"\t\t\tinputs += [arg_reader.{func}]\n"
    udx_str += (
        f"\t\t\tresult = {module}{function.__qualname__}(*inputs, **self.params)\n"
    )
    if is_udtf:
        udx_str += "\t\t\tif len(result) == 1:\n"
        udx_str += "\t\t\t\tresult = result[0]\n"
        for i, var in enumerate(return_type):
            func = get_set_add_function(return_type[var], func="set")
            udx_str += f"\t\t\tres_writer.{func}({i}, result[{i}])\n"
    else:
        func = get_set_add_function(return_type, func="set")
        udx_str += f"\t\t\tres_writer.{func}(result)\n"
    udx_str += "\t\t\tres_writer.next()\n"
    udx_str += "\t\t\tif not arg_reader.next():\n"
    udx_str += "\t\t\t\tbreak\n\n"

    # destroy
    udx_str += "\tdef destroy(self, server_interface, col_types):\n"
    udx_str += "\t\tpass\n\n"

    # Factory Function
    udx_str += f"class verticapy_{new_name}_factory(vertica_sdk.{ftype}Factory):\n\n"

    # createScalarFunction / createTransformFunction
    udx_str += f"\tdef create{ftype}(self, srv):\n"
    udx_str += f"\t\treturn verticapy_{new_name}()\n\n"

    # getPrototype
    udx_str += "\tdef getPrototype(self, server_interface, arg_types, return_type):\n"
    for var_type in arg_types:
        func = get_set_add_function(var_type, func="add")
        udx_str += f"\t\targ_types.{func}()\n"
    if is_udtf:
        for var in return_type:
            func = get_set_add_function(return_type[var], func="add")
            udx_str += f"\t\treturn_type.{func}()\n"
        udx_str += "\n"
    else:
        func = get_set_add_function(return_type, func="add")
        udx_str += f"\t\treturn_type.{func}()\n\n"

    # getReturnType
    udx_str += "\tdef getReturnType(self, server_interface, arg_types, return_type):\n"
    if is_udtf:
        for var in return_type:
            func = get_set_add_function(return_type[var], func="add")
            udx_str += f"\t\treturn_type.{func}('{var}')\n"
    else:
        udx_str += f"\t\treturn_type.{func}()\n\n"

    # getParameterType
    if parameters:
        udx_str += "\tdef getParameterType(self, server_interface, parameterTypes):\n"
        for param in parameters:
            func = get_set_add_function(parameters[param], func="add")
            udx_str += f"\t\tparameterTypes.{func}('{param}')\n"

    udx_str += "\n"
    transform = "TRANSFORM " if is_udtf else ""
    sql = (
        f"CREATE OR REPLACE {transform}FUNCTION {new_name} AS NAME "
        f"'verticapy_{new_name}_factory' LIBRARY {library_name};"
    )

    return udx_str, sql
