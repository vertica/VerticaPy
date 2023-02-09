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


def save_to_query_profile(
    name: str,
    path: str = "",
    json_dict: dict = {},
    query_label: str = "verticapy_json",
    return_query: bool = False,
    add_identifier: bool = True,
):
    """
Saves information about the specified VerticaPy method to the QUERY_PROFILES 
table in the Vertica database. It is used to collect usage statistics on 
methods and their parameters. This function generates a JSON string.

Parameters
----------
name: str
    Name of the method.
path: str, optional
    Path to the function or method.
json_dict: dict, optional
    Dictionary of the different parameters to store.
query_label: str, optional
    Name to give to the identifier in the query profile table. If 
    unspecified, the name of the method is used.
return_query: bool, optional
    If set to True, the query is returned.
add_identifier: bool, optional
    If set to True, the VerticaPy identifier is added to the generated json.

Returns
-------
bool
    True if the operation succeeded, False otherwise.
    """
    import verticapy as vp
    from verticapy.sql.read import _executeSQL

    if not (vp.OPTIONS["save_query_profile"]) or (
        isinstance(vp.OPTIONS["save_query_profile"], list)
        and name not in vp.OPTIONS["save_query_profile"]
    ):
        return False
    try:

        def dict_to_json_string(
            name: str = "",
            path: str = "",
            json_dict: dict = {},
            add_identifier: bool = False,
        ):
            from verticapy import vDataFrame
            from verticapy.learn.vmodel import vModel

            json = "{"
            if name:
                json += f'"verticapy_fname": "{name}", '
            if path:
                json += f'"verticapy_fpath": "{path}", '
            if add_identifier:
                json += f'"verticapy_id": "{vp.OPTIONS["identifier"]}", '
            for key in json_dict:
                json += f'"{key}": '
                if isinstance(json_dict[key], bool):
                    json += "true" if json_dict[key] else "false"
                elif isinstance(json_dict[key], (float, int)):
                    json += str(json_dict[key])
                elif json_dict[key] is None:
                    json += "null"
                elif isinstance(json_dict[key], vDataFrame):
                    json_dict_str = json_dict[key].__genSQL__().replace('"', '\\"')
                    json += f'"{json_dict_str}"'
                elif isinstance(json_dict[key], vModel):
                    json += f'"{json_dict[key].type}"'
                elif isinstance(json_dict[key], dict):
                    json += dict_to_json_string(json_dict=json_dict[key])
                elif isinstance(json_dict[key], list):
                    json_dict_str = ";".join([str(item) for item in json_dict[key]])
                    json += f'"{json_dict_str}"'
                else:
                    json_dict_str = str(json_dict[key]).replace('"', '\\"')
                    json += f'"{json_dict_str}"'
                json += ", "
            json = json[:-2] + "}"
            return json

        query_label_str = query_label.replace("'", "''")
        dict_to_json_string_str = dict_to_json_string(
            name, path, json_dict, add_identifier
        ).replace("'", "''")
        query = f"SELECT /*+LABEL('{query_label_str}')*/ '{dict_to_json_string_str}'"
        if return_query:
            return query
        _executeSQL(
            query=query,
            title="Sending query to save the information in query profile table.",
            print_time_sql=False,
        )
        return True
    except:
        return False
