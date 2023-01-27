# (c) Copyright [2018-2023] Micro Focus or one of its affiliates.
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
# VerticaPy is a Python library with scikit-like functionality for conducting
# data science projects on data stored in Vertica, taking advantage Vertica’s
# speed and built-in analytics and machine learning features. It supports the
# entire data science life cycle, uses a ‘pipeline’ mechanism to sequentialize
# data transformation operations, and offers beautiful graphical options.
#
# VerticaPy aims to do all of the above. The idea is simple: instead of moving
# data around for processing, VerticaPy brings the logic to the data.
#
#
#
# Modules
#
# Standard Python Modules
import typing, warnings, sys, inspect
from functools import wraps

#
#
# Decorators
#
# ---#
def save_verticapy_logs(func):
    """
------------------------------------------------------------------------------------
save_verticapy_logs decorator. It simplifies the code and automatically
identifies which function to save to the QUERY PROFILES table.
    """

    @wraps(func)
    def func_prec_save_logs(*args, **kwargs):

        from verticapy.utilities import save_to_query_profile

        name = func.__name__
        path = func.__module__.replace("verticapy.", "")
        json_dict = {}
        var_names = func.__code__.co_varnames
        if len(args) == len(var_names):
            for idx, arg in enumerate(args):
                if var_names[idx] != "self":
                    if (
                        var_names[idx] == "steps"
                        and path == "pipeline"
                        and isinstance(arg, list)
                    ):
                        json_dict[var_names[idx]] = [item[1].type for item in arg]
                    else:
                        json_dict[var_names[idx]] = arg
                else:
                    path += "." + type(arg).__name__
        json_dict = {**json_dict, **kwargs}
        save_to_query_profile(name=name, path=path, json_dict=json_dict)

        return func(*args, **kwargs)

    return func_prec_save_logs


# ---#
def check_dtypes(func):
    """
------------------------------------------------------------------------------------
check_dtypes decorator. It simplifies the code by checking whether the
parameters passed to the function are of an expected data type.
    """

    @wraps(func)
    def func_prec_check_dtypes(*args, **kwargs):

        from verticapy.toolbox import str_sql
        from verticapy.vdataframe import vDataFrame

        python_version = [int(i) for i in sys.version.split(" ")[0].split(".")]

        hints = typing.get_type_hints(func)
        all_args = {**kwargs}

        args_name = inspect.getfullargspec(func)[0]
        n = len(args_name)
        for idx, var in enumerate(args):
            if idx < n:
                all_args[args_name[idx]] = var

        for var_name in hints:
            # get_args is only available for Python version greater than 3.7
            if python_version[0] > 3 or (
                python_version[0] == 3 and python_version[1] > 7
            ):
                dt = typing.get_args(hints[var_name])
            else:
                try:
                    dt = hints[var_name].__args__
                except:
                    dt = ()
            if not dt:
                dt = hints[var_name]
                dt_str_list = str(dt)
                single_type = True
            else:
                dt_str_list = "|".join([str(t) for t in dt])
                single_type = False
            if var_name in all_args:
                var = all_args[var_name]
                if isinstance(var, vDataFrame) and (
                    (isinstance(dt, tuple) and str_sql in dt) or dt == str_sql
                ):
                    pass
                elif not (isinstance(var, type(None))) and not (isinstance(var, dt)):
                    dt_var = type(var)
                    if single_type:
                        warning_message = f"Parameter '{var_name}' must be of type <{dt_str_list}>, found type <{dt_var}>."
                    else:
                        warning_message = f"Parameter '{var_name}' type must be one of the following: <{dt_str_list}>, found type <{dt_var}>."
                    warning_message = (
                        warning_message.replace("<class '", "")
                        .replace("'>", "")
                        .replace("verticapy.toolbox.", "")
                        .replace("str_sql", "str_sql|vDataFrame")
                    )
                    warnings.warn(warning_message, Warning)

        return func(*args, **kwargs)

    return func_prec_check_dtypes


# ---#
def check_minimum_version(func):
    """
------------------------------------------------------------------------------------
check_minimum_version decorator. It simplifies the code by checking if the
feature is available in the user's version.
    """

    @wraps(func)
    def func_prec_check_minimum_version(*args, **kwargs):

        import verticapy as vp
        from verticapy.utilities import vertica_version

        fun_name, object_name, condition = func.__name__, "", []
        if len(args) > 0:
            object_name = type(args[0]).__name__
        name = object_name if fun_name == "__init__" else fun_name
        vertica_version(vp.MINIMUM_VERSION[name])

        return func(*args, **kwargs)

    return func_prec_check_minimum_version
