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
from verticapy.errors import ConnectionError


def get_dblink_fun(query: str, symbol: str = "$"):
    import verticapy as vp
    from verticapy.utils._toolbox import clean_query

    assert symbol in vp.OPTIONS["external_connection"], ConnectionError(
        f"External Query detected but no corresponding Connection Identifier "
        f"Database is defined (Using the symbol '{symbol}'). Use the "
        "function connect.set_external_connection to set one with the correct symbol."
    )
    cid = vp.OPTIONS["external_connection"][symbol]["cid"].replace("'", "''")
    query = query.replace("'", "''")
    rowset = vp.OPTIONS["external_connection"][symbol]["rowset"]
    query = f"""
        SELECT 
            DBLINK(USING PARAMETERS 
                   cid='{cid}',
                   query='{query}',
                   rowset={rowset}) OVER ()"""
    return clean_query(query)


def replace_external_queries_in_query(query: str):
    import verticapy as vp
    from verticapy.utils._toolbox import gen_tmp_name, executeSQL

    sql_keyword = (
        "select ",
        "create ",
        "insert ",
        "drop ",
        "backup ",
        "alter ",
        "update ",
    )
    nb_external_queries = 0
    for s in vp.OPTIONS["external_connection"]:
        external_queries = re.findall(f"\\{s}\\{s}\\{s}(.*?)\\{s}\\{s}\\{s}", query)
        for external_query in external_queries:
            if external_query.strip().lower().startswith(sql_keyword):
                external_query_tmp = external_query
                subquery_flag = False
            else:
                external_query_tmp = f"SELECT * FROM {external_query}"
                subquery_flag = True
            query_dblink_template = get_dblink_fun(external_query_tmp, symbol=s)
            if " " in external_query.strip():
                alias = f"VERTICAPY_EXTERNAL_TABLE_{nb_external_queries}"
            else:
                alias = '"' + external_query.strip().replace('"', '""') + '"'
            if nb_external_queries >= 1:
                temp_table_name = '"' + gen_tmp_name(name=alias).replace('"', "") + '"'
                create_statement = f"""
                    CREATE LOCAL TEMPORARY TABLE {temp_table_name} 
                    ON COMMIT PRESERVE ROWS 
                    AS {query_dblink_template}"""
                executeSQL(
                    create_statement,
                    title=(
                        "Creating a temporary local table to "
                        f"store the {nb_external_queries} external table."
                    ),
                )
                query_dblink_template = f"v_temp_schema.{temp_table_name} AS {alias}"
            else:
                if subquery_flag:
                    query_dblink_template = f"({query_dblink_template}) AS {alias}"
            query = query.replace(
                f"{s}{s}{s}{external_query}{s}{s}{s}", query_dblink_template
            )
            nb_external_queries += 1
    return query
