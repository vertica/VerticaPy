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
import re

from verticapy._utils._sql._format import clean_query
from verticapy.connect import EXTERNAL_CONNECTION
from verticapy.errors import ConnectionError


def get_dblink_fun(query: str, symbol: str = "$"):
    if symbol not in EXTERNAL_CONNECTION:
        raise ConnectionError(
            "External Query detected but no corresponding "
            "Connection Identifier Database is defined (Using "
            f"the symbol '{symbol}'). Use the function connect."
            "set_external_connection to set one with the correct symbol."
        )
    cid = EXTERNAL_CONNECTION[symbol]["cid"].replace("'", "''")
    query = query.replace("'", "''")
    rowset = EXTERNAL_CONNECTION[symbol]["rowset"]
    query = f"""
        SELECT 
            DBLINK(USING PARAMETERS 
                   cid='{cid}',
                   query='{query}',
                   rowset={rowset}) OVER ()"""
    return clean_query(query)


def replace_external_queries_in_query(query: str):
    from verticapy._utils._gen import gen_tmp_name
    from verticapy._utils._sql._execute import _executeSQL

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
    for s in EXTERNAL_CONNECTION:
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
                _executeSQL(
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
