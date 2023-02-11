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
from typing import Literal

from verticapy.sql._utils._format import (
    quote_ident,
    schema_relation,
    format_schema_table,
)
from verticapy.utils._sql import _executeSQL


def drop(
    name: str = "",
    method: Literal["table", "view", "model", "geo", "text", "auto", "schema"] = "auto",
    raise_error: bool = False,
    **kwds,
):
    """
Drops the input relation. This can be a model, view, table, text index,
schema, or geo index.

Parameters
----------
name: str, optional
    Relation name. If empty, it will drop all VerticaPy temporary 
    elements.
method / relation_type: str, optional
    Method used to drop.
        auto   : identifies the table/view/index/model to drop. 
                 It will never drop an entire schema unless the 
                 method is set to 'schema'.
        model  : drops the input model.
        table  : drops the input table.
        view   : drops the input view.        
        geo    : drops the input geo index.
        text   : drops the input text index.
        schema : drops the input schema.
raise_error: bool, optional
    If the object couldn't be dropped, this function raises an error.

Returns
-------
bool
    True if the relation was dropped, False otherwise.
    """
    if "relation_type" in kwds and method == "auto":
        method = kwds["relation_type"]
    schema, relation = schema_relation(name)
    schema, relation = schema[1:-1], relation[1:-1]
    if not (name):
        method = "temp"
    if method == "auto":
        fail, end_conditions = False, False
        result = _executeSQL(
            query=f"""
            SELECT 
                /*+LABEL('utilities.drop')*/ * 
            FROM columns 
            WHERE table_schema = '{schema}' 
                AND table_name = '{relation}'""",
            print_time_sql=False,
            method="fetchrow",
        )
        if not (result):
            result = _executeSQL(
                query=f"""
                SELECT 
                    /*+LABEL('utilities.drop')*/ * 
                FROM view_columns 
                WHERE table_schema = '{schema}' 
                    AND table_name = '{relation}'""",
                print_time_sql=False,
                method="fetchrow",
            )
        elif not (end_conditions):
            method = "table"
            end_conditions = True
        if not (result):
            try:
                result = _executeSQL(
                    query=f"""
                    SELECT 
                        /*+LABEL('utilities.drop')*/ model_type 
                    FROM verticapy.models 
                    WHERE LOWER(model_name) = '{quote_ident(name).lower()}'""",
                    print_time_sql=False,
                    method="fetchrow",
                )
            except:
                result = []
        elif not (end_conditions):
            method = "view"
            end_conditions = True
        if not (result):
            result = _executeSQL(
                query=f"""
                SELECT 
                    /*+LABEL('utilities.drop')*/ * 
                FROM models 
                WHERE schema_name = '{schema}' 
                    AND model_name = '{relation}'""",
                print_time_sql=False,
                method="fetchrow",
            )
        elif not (end_conditions):
            method = "model"
            end_conditions = True
        if not (result):
            result = _executeSQL(
                query=f"""
                SELECT 
                    /*+LABEL('utilities.drop')*/ * 
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
        elif not (end_conditions):
            method = "model"
            end_conditions = True
        if not (result):
            try:
                _executeSQL(
                    query=f"""
                        SELECT 
                            /*+LABEL(\'utilities.drop\')*/ * 
                        FROM "{schema}"."{relation}" LIMIT 0;""",
                    print_time_sql=False,
                )
                method = "text"
            except:
                fail = True
        elif not (end_conditions):
            method = "geo"
            end_conditions = True
        if fail:
            if raise_error:
                raise MissingRelation(
                    f"No relation / index / view / model named '{name}' was detected."
                )
            return False
    query = ""
    if method == "model":
        model_type = kwds["model_type"] if "model_type" in kwds else None
        try:
            result = _executeSQL(
                query=f"""
                    SELECT 
                        /*+LABEL('utilities.drop')*/ model_type 
                    FROM verticapy.models 
                    WHERE LOWER(model_name) = '{quote_ident(name).lower()}'""",
                print_time_sql=False,
                method="fetchfirstelem",
            )
            is_in_verticapy_schema = True
            if not (model_type):
                model_type = result
        except:
            is_in_verticapy_schema = False
        if (
            model_type
            in (
                "DBSCAN",
                "LocalOutlierFactor",
                "CountVectorizer",
                "KernelDensity",
                "AutoDataPrep",
                "KNeighborsRegressor",
                "KNeighborsClassifier",
                "NearestCentroid",
            )
            or is_in_verticapy_schema
        ):
            if model_type in ("DBSCAN", "LocalOutlierFactor"):
                drop(name, method="table")
            elif model_type == "CountVectorizer":
                drop(name, method="text")
                if is_in_verticapy_schema:
                    res = _executeSQL(
                        query=f"""
                            SELECT 
                                /*+LABEL('utilities.drop')*/ value 
                            FROM verticapy.attr 
                            WHERE LOWER(model_name) = '{quote_ident(name).lower()}' 
                                AND attr_name = 'countvectorizer_table'""",
                        print_time_sql=False,
                        method="fetchrow",
                    )
                    if res and res[0]:
                        drop(res[0], method="table")
            elif model_type == "KernelDensity":
                table_name = name.replace('"', "") + "_KernelDensity_Map"
                drop(table_name, method="table")
                model_name = name.replace('"', "") + "_KernelDensity_Tree"
                drop(model_name, method="model")
            elif model_type == "AutoDataPrep":
                drop(name, method="table")
            if is_in_verticapy_schema:
                _executeSQL(
                    query=f"""
                        DELETE /*+LABEL('utilities.drop')*/ 
                        FROM verticapy.models 
                        WHERE LOWER(model_name) = '{quote_ident(name).lower()}';""",
                    title="Deleting vModel.",
                )
                _executeSQL("COMMIT;", title="Commit.")
                _executeSQL(
                    query=f"""
                        DELETE /*+LABEL('utilities.drop')*/ 
                        FROM verticapy.attr 
                        WHERE LOWER(model_name) = '{quote_ident(name).lower()}';""",
                    title="Deleting vModel attributes.",
                )
                _executeSQL("COMMIT;", title="Commit.")
        else:
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
        except:
            if raise_error:
                raise
            result = False
    elif method == "temp":
        sql = """SELECT /*+LABEL('utilities.drop')*/
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
        sql = """SELECT /*+LABEL('utilities.drop')*/
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
