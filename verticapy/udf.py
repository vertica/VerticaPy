# (c) Copyright [2018-2021] Micro Focus or one of its affiliates.
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
#
# |_     |~) _  _| _  /~\    _ |.
# |_)\/  |_)(_|(_||   \_/|_|(_|||
#    /
#              ____________       ______
#             / __        `\     /     /
#            |  \/         /    /     /
#            |______      /    /     /
#                   |____/    /     /
#          _____________     /     /
#          \           /    /     /
#           \         /    /     /
#            \_______/    /     /
#             ______     /     /
#             \    /    /     /
#              \  /    /     /
#               \/    /     /
#                    /     /
#                   /     /
#                   \    /
#                    \  /
#                     \/
#                    _
# \  / _  __|_. _ _ |_)
#  \/ (/_|  | |(_(_|| \/
#                     /
# VerticaPy is a Python library with scikit-like functionality to use to conduct
# data science projects on data stored in Vertica, taking advantage Vertica’s
# speed and built-in analytics and machine learning features. It supports the
# entire data science life cycle, uses a ‘pipeline’ mechanism to sequentialize
# data transformation operations, and offers beautiful graphical options.
#
# VerticaPy aims to solve all of these problems. The idea is simple: instead
# of moving data around for processing, VerticaPy brings the logic to the data.
#
#
# Modules
#
# Standard Python Modules
import datetime, decimal, inspect, os

# VerticaPy Modules
import verticapy
from verticapy.utilities import *
from verticapy.toolbox import *

#
# ---#
def import_lib_udf(
    udf_list: list, library_name: str, include_dependencies: list = [], cursor=None
):
    """
---------------------------------------------------------------------------
Install a library of Python functions in Vertica. This function will work only
when it is executed directly in the server.

Parameters
----------
udf_list: list
	List of tuples including the different functions.
		function     : [function] Python Function.
	    arg_types    : [dict/list] List or dictionary of the function input types.
	    			   Example: {"input1": int, "input2": float} or [int, float]
	    return_type  : [type/dict] Function output type. In case of many outputs, 
	    			   it must be a dictionary including all the outputs types and 
	    			   names. Example: {"result1": int, "result2": float}
	    parameters   : [dict] Dictionary of the function input optional parameters.
	    			   Example: {"param1": int, "param2": str}
	    new_name     : [str] New function name when installed in Vertica.
library_name: str
	Library Name.
include_dependencies: list, optional
	Library files dependencies. The function will copy paste the different files
	in the UDF definition.
cursor: DBcursor, optional
	Vertica database cursor.
	"""
    cursor, conn = check_cursor(cursor)[0:2]
    directory = os.path.dirname(verticapy.__file__)
    session_name = get_session(cursor)
    file_name = f"{library_name}_{session_name}.py"
    try:
        os.remove(directory + "/" + file_name)
    except:
        pass
    udx_str, sql = create_lib_udf(
        udf_list, library_name, include_dependencies, directory + "/" + file_name
    )
    f = open(directory + "/" + file_name, "w")
    f.write(udx_str)
    f.close()
    try:
        for query in sql:
            print(query)
            cursor.execute(query)
        os.remove(directory + "/" + file_name)
        return True
    except Exception as e:
        os.remove(directory + "/" + file_name)
        if conn:
            conn.close()
        raise e
    if conn:
        conn.close()


# ---#
def create_lib_udf(
    udf_list: list,
    library_name: str,
    include_dependencies: list = [],
    file_path: str = "",
    create_file: bool = False,
):
    """
---------------------------------------------------------------------------
Generates the code needed to install a library of Python functions. It will
use the Vertica SDK to create UDF of the input functions.

Parameters
----------
udf_list: list
	List of tuples including the different functions.
		function     : [function] Python Function.
	    arg_types    : [dict/list] List or dictionary of the function input types.
	    			   Example: {"input1": int, "input2": float} or [int, float]
	    return_type  : [type/dict] Function output type. In case of many outputs, 
	    			   it must be a dictionary including all the outputs types and 
	    			   names. Example: {"result1": int, "result2": float}
	    parameters   : [dict] Dictionary of the function input optional parameters.
	    			   Example: {"param1": int, "param2": str}
	    new_name     : [str] New function name when installed in Vertica.
library_name: str
	Library Name.
include_dependencies: list, optional
	Library files dependencies. The function will copy paste the different files
	in the UDF definition.
file_path: str, optional
	Path to the UDF file.
create_file: bool, optional
	If set to True, instead of returning the str of the UDx, the function will
	create two files: one UDF py file and one SQL file to install it.

Returns
-------
udx_str, sql
    UDF py file, str needed to install the library.
	"""
    if isinstance(include_dependencies, (str)):
        include_dependencies = [include_dependencies]
    if not (isinstance(include_dependencies, (list))):
        raise ValueError(
            f"The parameter include_dependencies type must be <list>. Found {type(include_dependencies)}."
        )
    if not (isinstance(library_name, str)):
        raise ValueError(
            f"The parameter library_name type must be <str>. Found {type(library_name)}."
        )
    if not (isinstance(file_path, str)):
        raise ValueError(
            f"The parameter file_path type must be <str>. Found {type(file_path)}."
        )
    if not (isinstance(create_file, bool)):
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
            if not (isinstance(dep_file_path, str)):
                raise ValueError(
                    f"The parameter include_dependencies type must be <list> of <str>. Found {type(dep_file_path)} inside."
                )
            f = open(dep_file_path)
            file_str = f.read()
            exec(file_str)
            udx_str += "\n" + file_str + "\n"
            f.close()
    udx_str += "\n"
    sql = []
    for udf in udf_list:
        tmp = create_udf(*udf, **{"library_name": library_name})
        udx_str += tmp[0]
        sql += [tmp[1]]
    sql_path = (
        os.path.dirname(file_path) + "/" + library_name + ".sql"
        if (file_path)
        else library_name + ".sql"
    )
    if not (file_path):
        file_path = f"verticapy_{library_name}.py"
    sql = [
        f"CREATE OR REPLACE LIBRARY {library_name} AS '{file_path}' LANGUAGE 'Python';"
    ] + sql
    if create_file:
        f = open(file_path, "w")
        f.write(udx_str)
        f.close()
        f = open(sql_path, "w")
        f.write("\n".join(sql))
        f.close()
    else:
        return udx_str, sql


# Functions used to create the 2 main ones.

# ---#
def create_udf(
    function,
    arg_types: (list, dict),
    return_type: (type, dict),
    parameters: dict = {},
    new_name: str = "",
    library_name: str = "",
):
    if not (hasattr(function, "__call__")):
        raise ValueError(
            f"The function parameter must be a Python function. Found {type(function)}."
        )
    if not (new_name):
        new_name = function.__name__
    elif not (isinstance(new_name, str)):
        raise ValueError(
            f"The parameter new_name type must be <str>. Found {type(new_name)}."
        )
    module = function.__module__
    if module == "__main__":
        module = ""
    if not (library_name):
        library_name = module
    if module:
        module += "."
    elif not (isinstance(library_name, str)):
        raise ValueError(
            f"The parameter library_name type must be <str>. Found {type(library_name)}."
        )
    if isinstance(arg_types, dict):
        arg_types = [arg_types[var] for var in arg_types]
    elif not (isinstance(arg_types, list)):
        raise ValueError(
            f"The arg_types parameter must be a <list> of <types>. Found {type(arg_types)}."
        )
    for idx, dtype in enumerate(arg_types):
        if not (isinstance(dtype, type)):
            raise ValueError(
                f"Each element of arg_types parameter must be a <type>. Found {type(dtype)} at index {idx}."
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
        is_udtf, process_function, ftype = True, "processPartition", "TransformFunction"
    elif isinstance(return_type, type):
        is_udtf, process_function, ftype = False, "processBlock", "ScalarFunction"
    else:
        raise ValueError(
            f"return_type must be the dictionary of the returned types in case of Transform Function and only a type in case of Scalar Function. Can not be of type {type(return_type)}."
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
    sql = f"CREATE OR REPLACE {transform}FUNCTION {new_name} AS NAME 'verticapy_{new_name}_factory' LIBRARY {library_name};"

    return udx_str, sql


# ---#
def get_func_info(func):
    # TO COMPLETE - GUESS FUNCTIONS TYPES
    name = func.__name__
    argspec = inspect.getfullargspec(func)[6]
    if "return" in argspec:
        return_type = argspec["return"]
        del argspec["return"]
    else:
        return_type = None
    arg_types = {}
    parameters = {}
    for param in argspec:
        if inspect.signature(func).parameters[param].default == inspect._empty:
            arg_types[param] = argspec[param]
        else:
            parameters[param] = argspec[param]
    return (func, arg_types, return_type, parameters)


# ---#
def get_module_func_info(module):
    # ---#
    def get_list(module):
        func_list = []
        for func in dir(module):
            if func[0] != "_":
                func_list += [func]
        return func_list

    # TO COMPLETE - TRY AND RAISE THE APPROPRIATE ERROR
    func_list = get_list(module)
    func_info = []
    for func in func_list:
        ldic = locals()
        exec(f"info = get_func_info(module.{func})", globals(), ldic)
        func_info += [ldic["info"]]
    return func_info


# ---#
def get_set_add_function(ftype, func="get"):
    # func = get / set / add
    func = func.lower()
    if ftype == bytes:
        return f"{func}Binary"
    elif ftype == bool:
        return f"{func}Bool"
    elif ftype == datetime.date:
        return f"{func}Date"
    elif ftype == float:
        return f"{func}Float"
    elif ftype == int:
        return f"{func}Int"
    elif ftype == datetime.timedelta:
        return f"{func}Interval"
    elif ftype == decimal.Decimal:
        return f"{func}Numeric"
    elif ftype == str:
        if func == "add":
            return f"{func}Varchar"
        else:
            return f"{func}String"
    elif ftype == datetime.time:
        return f"{func}Time"
    elif ftype == (datetime.time, datetime.tzinfo):
        return f"{func}TimeTz"
    elif ftype == datetime.datetime:
        return f"{func}Timestamp"
    elif ftype == (datetime.datetime, datetime.tzinfo):
        return f"{func}TimestampTz"
    else:
        raise "The input type is not managed by get_set_add_function. Check the Vertica Python SDK for more information."
