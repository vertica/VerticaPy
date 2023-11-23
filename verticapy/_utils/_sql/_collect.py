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
from functools import wraps
from typing import Any, Callable, Optional

from vertica_python.errors import QueryError

import verticapy._config.config as conf
from verticapy._utils._sql._format import format_type
from verticapy.connection.global_connection import get_global_connection
from verticapy.connection.connect import current_cursor


def _dict_to_json_string(
    name: Optional[str] = None,
    path: Optional[str] = None,
    json_dict: Optional[dict] = None,
    add_identifier: bool = False,
) -> str:
    gb_conn = get_global_connection()
    json = "{"
    if name:
        json += f'"verticapy_fname": "{name}", '
    if path:
        json += f'"verticapy_fpath": "{path}", '
    if add_identifier:
        json += f'"verticapy_id": "{gb_conn.vpy_session_identifier}", '
    for key in json_dict:
        object_type = None
        if hasattr(json_dict[key], "object_type"):
            object_type = json_dict[key].object_type
        json += f'"{key}": '
        if isinstance(json_dict[key], bool):
            json += "true" if json_dict[key] else "false"
        elif isinstance(json_dict[key], (float, int)):
            json += str(json_dict[key])
        elif json_dict[key] is None:
            json += "null"
        elif object_type == "vDataFrame":
            json_dict_str = json_dict[key].current_relation().replace('"', '\\"')
            json += f'"{json_dict_str}"'
        elif object_type == "VerticaModel":
            json += f'"{json_dict[key]._model_type}"'
        elif isinstance(json_dict[key], dict):
            json += _dict_to_json_string(json_dict=json_dict[key])
        elif isinstance(json_dict[key], list):
            json_dict_str = ";".join([str(item) for item in json_dict[key]])
            json += f'"{json_dict_str}"'
        else:
            json_dict_str = str(json_dict[key]).replace('"', '\\"')
            json += f'"{json_dict_str}"'
        json += ", "
    json = json[:-2] + "}"
    return json


def save_to_query_profile(
    name: str,
    path: Optional[str] = None,
    json_dict: Optional[dict] = None,
    query_label: str = "verticapy_json",
    return_query: bool = False,
    add_identifier: bool = True,
) -> bool:
    """
    Saves information about the specified VerticaPy
    method to the QUERY_PROFILES table in the Vertica
    database. It is used to collect usage statistics
    on methods and their parameters. This function
    generates a JSON string.

    Parameters
    ----------
    name: str
        Name of the method.
    path: str, optional
        Path to the function or method.
    json_dict: dict, optional
        Dictionary of the different parameters to
        store.
    query_label: str, optional
        Name to give to the identifier in the query
        profile table. If unspecified, the name of the
        method is used.
    return_query: bool, optional
        If set to True, the query is returned.
    add_identifier: bool, optional
        If set to True, the VerticaPy identifier is
        added to the generated json.

    Returns
    -------
    bool
        True if the operation succeeded, False otherwise.
    """
    json_dict = format_type(json_dict, dtype=dict)
    value = conf.get_option("save_query_profile")
    if not value:
        return False
    query_label_str = query_label.replace("'", "''")
    dict_to_json_string_str = _dict_to_json_string(
        name, path, json_dict, add_identifier
    ).replace("'", "''")
    query = f"SELECT /*+LABEL('{query_label_str}')*/ '{dict_to_json_string_str}'"
    if return_query:
        return query
    try:
        current_cursor().execute(query)
        return True
    except QueryError:
        return False


def save_verticapy_logs(func: Callable) -> Callable:
    """
    save_verticapy_logs decorator. It simplifies the code
    and automatically identifies which function to save to
    the QUERY_PROFILES table.
    """

    @wraps(func)
    def func_prec_save_logs(*args, **kwargs) -> Any:
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
                        json_dict[var_names[idx]] = [
                            item[1]._model_type for item in arg
                        ]
                    else:
                        json_dict[var_names[idx]] = arg
                else:
                    path += "." + type(arg).__name__
        json_dict = {**json_dict, **kwargs}
        save_to_query_profile(name=name, path=path, json_dict=json_dict)

        return func(*args, **kwargs)

    return func_prec_save_logs
