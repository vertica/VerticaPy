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
import datetime
import decimal
import inspect
from typing import Literal


def get_func_info(func) -> tuple:
    # TO COMPLETE - GUESS FUNCTIONS TYPES

    # name = func.__name__
    argspec = inspect.getfullargspec(func)[6]
    if "return" in argspec:
        return_type = argspec["return"]
        del argspec["return"]
    else:
        return_type = None

    arg_types, parameters = {}, {}
    for param in argspec:
        if inspect.signature(func).parameters[param].default == inspect._empty:
            arg_types[param] = argspec[param]
        else:
            parameters[param] = argspec[param]

    return (func, arg_types, return_type, parameters)


def get_module_func_info(module) -> list:
    # TO COMPLETE - TRY AND RAISE THE APPROPRIATE ERROR

    def get_list(module):
        func_list = []
        for func in dir(module):
            if func[0] != "_":
                func_list += [func]
        return func_list

    func_list = get_list(module)
    func_info = []
    for func in func_list:
        ldic = locals()
        exec(f"info = get_func_info(module.{func})", globals(), ldic)
        func_info += [ldic["info"]]
    return func_info


def get_set_add_function(
    ftype: type, func: Literal["get", "set", "add"] = "get"
) -> str:
    func = str(func).lower()
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
        raise ValueError(
            "The input type is not managed by get_set_add_function. "
            "Check the Vertica Python SDK for more information."
        )
