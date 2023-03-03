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
from typing import Union

from verticapy._utils._sql._format import schema_relation
from verticapy._utils._sql._sys import _executeSQL


def does_model_exist(
    name: str, raise_error: bool = False, return_model_type: bool = False
) -> Union[bool, str]:
    """
    Checks if the model is stored in the Vertica DB.

    Parameters
    ----------
    name: str
        Model name.
    raise_error: bool, optional
        If set to True and an error occurs, it 
        raises the error.
    return_model_type: bool, optional
        If set to True, returns the model type.

    Returns
    -------
    bool
        True if the model is stored in the
        Vertica DB.
    """
    model_type = None
    schema, model_name = schema_relation(name)
    schema, model_name = schema[1:-1], model_name[1:-1]
    result = _executeSQL(
        query=f"""
            SELECT 
                /*+LABEL('learn.tools.does_model_exist')*/ 
                model_type 
            FROM MODELS 
            WHERE LOWER(model_name) = LOWER('{model_name}') 
              AND LOWER(schema_name) = LOWER('{schema}') 
            LIMIT 1""",
        method="fetchrow",
        print_time_sql=False,
    )
    if result:
        model_type = result[0]
        result = True
    else:
        result = False
    if raise_error and result:
        raise NameError(f"The model '{name}' already exists !")
    if return_model_type:
        return model_type
    return result
