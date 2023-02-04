"""
(c)  Copyright  [2018-2023]  OpenText  or one of its
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


def save_verticapy_logs(func):
    """
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


def check_minimum_version(func):
    """
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
