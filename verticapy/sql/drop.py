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
from typing import Literal, Optional

from vertica_python.errors import MissingRelation, QueryError

from verticapy._utils._sql._format import (
    format_schema_table,
    schema_relation,
)
from verticapy._utils._sql._sys import _executeSQL


def drop(
    name: Optional[str] = None,
    method: Literal["table", "view", "model", "geo", "text", "auto", "schema"] = "auto",
    raise_error: bool = False,
) -> bool:
    """
    Drops the input relation. This can be a model,
    view, table, text index, schema, or geo index.

    Parameters
    ----------
    name: str, optional
        Relation name. If empty, the function drops
        all VerticaPy temporary elements.
    method: str, optional
        Method used to drop.

        **auto**   :
                    identifies the table / view /
                    index / model to drop.
                    It never drops an entire
                    schema  unless the  method is
                    set to 'schema'.

        **model**  :
                    drops the input model.

        **table**  :
                    drops the input table.

        **view**   :
                    drops the input view.

        **geo**    :
                    drops the input geo index.

        **text**   :
                    drops the input text index.

        **schema** :
                    drops the input schema.

    raise_error: bool, optional
        If  the object  couldn't be dropped,  this
        function raises an error.

    Returns
    -------
    bool
        True   if   the   relation   was  dropped,
        False otherwise.

    Examples
    --------
    .. ipython:: python
        :suppress:

        from verticapy.sql import create_table
        create_table(
            table_name = "table_example",
            schema = "public",
            dtype = {"name": "VARCHAR(60)"},
        )

    Drop the table:

    .. warning:: Dropping an element permanently removes it from the database. Please exercise caution, as this action is irreversible.

    .. ipython:: python

        from verticapy.sql import drop

        drop(name = "public.table_example")
    """
    schema, relation = schema_relation(name)
    schema, relation = schema[1:-1], relation[1:-1]
    if not name:
        method = "temp"
    if method == "auto":
        fail = False
        result = _executeSQL(
            query=f"""
            SELECT 
                /*+LABEL('drop')*/ * 
            FROM columns 
            WHERE table_schema = '{schema}' 
                AND table_name = '{relation}'""",
            print_time_sql=False,
            method="fetchrow",
        )
        if result:
            return drop(name=name, method="table", raise_error=raise_error)
        result = _executeSQL(
            query=f"""
            SELECT 
                /*+LABEL('drop')*/ * 
            FROM view_columns 
            WHERE table_schema = '{schema}' 
                AND table_name = '{relation}'""",
            print_time_sql=False,
            method="fetchrow",
        )
        if result:
            return drop(name=name, method="view", raise_error=raise_error)
        result = _executeSQL(
            query=f"""
            SELECT 
                /*+LABEL('drop')*/ * 
            FROM models 
            WHERE schema_name = '{schema}' 
                AND model_name = '{relation}'""",
            print_time_sql=False,
            method="fetchrow",
        )
        if result:
            return drop(name=name, method="model", raise_error=raise_error)
        result = _executeSQL(
            query=f"""
            SELECT 
                /*+LABEL('drop')*/ * 
            FROM 
                (SELECT STV_Describe_Index () OVER ()) x  
            WHERE name IN ('{schema}.{relation}',
                           '{relation}',
                           '\"{schema}\".\"{relation}\"',
                           '\"{relation}\"',
                           '{schema}.\"{relation}\"',
                           '\"{schema}\".{relation}')""",
            print_time_sql=False,
            method="fetchrow",
        )
        if result:
            return drop(name=name, method="geo", raise_error=raise_error)
        try:
            _executeSQL(
                query=f"""
                    SELECT 
                        /*+LABEL(\'utilities.drop\')*/ * 
                    FROM "{schema}"."{relation}" LIMIT 0;""",
                print_time_sql=False,
            )
            return drop(name=name, method="text", raise_error=raise_error)
        except QueryError:
            fail = True
        if fail:
            if raise_error:
                raise MissingRelation(f"No relation named '{name}' was detected.")
            return False
    query = ""
    if method == "model":
        query = f"DROP MODEL {name};"
    elif method == "table":
        query = f"DROP TABLE {name};"
    elif method == "view":
        query = f"DROP VIEW {name};"
    elif method == "geo":
        query = f"SELECT STV_Drop_Index(USING PARAMETERS index ='{name}') OVER ();"
    elif method == "text":
        query = f"DROP TEXT INDEX {name};"
    elif method == "schema":
        query = f"DROP SCHEMA {name} CASCADE;"
    if query:
        try:
            _executeSQL(query, title="Deleting the relation.")
            result = True
        except QueryError:
            if raise_error:
                raise
            result = False
    elif method == "temp":
        sql = """SELECT /*+LABEL('drop')*/
                    table_schema, table_name 
                 FROM columns 
                 WHERE LOWER(table_name) LIKE '%_verticapy_tmp_%' 
                 GROUP BY 1, 2;"""
        all_tables = result = _executeSQL(sql, print_time_sql=False, method="fetchall")
        for elem in all_tables:
            table = format_schema_table(
                elem[0].replace('"', '""'), elem[1].replace('"', '""')
            )
            drop(table, method="table")
        sql = """SELECT /*+LABEL('drop')*/
                    table_schema, table_name 
                 FROM view_columns 
                 WHERE LOWER(table_name) LIKE '%_verticapy_tmp_%' 
                 GROUP BY 1, 2;"""
        all_views = _executeSQL(sql, print_time_sql=False, method="fetchall")
        for elem in all_views:
            view = format_schema_table(
                elem[0].replace('"', '""'), elem[1].replace('"', '""')
            )
            drop(view, method="view")
        result = True
    else:
        result = True
    return result
