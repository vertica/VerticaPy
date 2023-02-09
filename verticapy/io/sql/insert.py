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
import warnings, sys, time
from verticapy.utils._decorators import save_verticapy_logs
from verticapy.errors import MissingRelation
from verticapy.io.sql._utils._format import (
    format_schema_table,
    clean_query,
    quote_ident,
)


@save_verticapy_logs
def insert_into(
    table_name: str,
    data: list,
    schema: str = "",
    column_names: list = [],
    copy: bool = True,
    genSQL: bool = False,
):
    """
Inserts the dataset into an existing Vertica table.

Parameters
----------
table_name: str
    Name of the table to insert into.
data: list
    The data to ingest.
schema: str, optional
    Schema name.
column_names: list, optional
    Name of the column(s) to insert into.
copy: bool, optional
    If set to True, the batch insert is converted to a COPY statement 
    with prepared statements. Otherwise, the INSERTs are performed
    sequentially.
genSQL: bool, optional
    If set to True, the SQL code that would be used to insert the data 
    is generated, but not executed.

Returns
-------
int
    The number of rows ingested.

See Also
--------
pandas_to_vertica : Ingests a pandas DataFrame into the Vertica database.
    """
    import verticapy as vp
    from verticapy.utils._toolbox import executeSQL

    if not (schema):
        schema = vp.OPTIONS["temp_schema"]
    input_relation = format_schema_table(schema, table_name)
    if not (column_names):
        result = executeSQL(
            query=f"""
                SELECT /*+LABEL('utilities.insert_into')*/
                    column_name
                FROM columns 
                WHERE table_name = '{table_name}' 
                    AND table_schema = '{schema}' 
                ORDER BY ordinal_position""",
            title=f"Getting the table {input_relation} column names.",
            method="fetchall",
        )
        column_names = [elem[0] for elem in result]
        assert column_names, MissingRelation(
            f"The table {input_relation} does not exist."
        )
    cols = [quote_ident(col) for col in column_names]
    if copy and not (genSQL):
        executeSQL(
            query=f"""
                INSERT INTO {input_relation} 
                ({", ".join(cols)})
                VALUES ({", ".join(["%s" for i in range(len(cols))])})""",
            title=(
                f"Insert new lines in the {table_name} table. "
                "The batch insert is converted into a COPY "
                "statement by using prepared statements."
            ),
            data=list(map(tuple, data)),
        )
        executeSQL("COMMIT;", title="Commit.")
        return len(data)
    else:
        if genSQL:
            sql = []
        i, n, total_rows = 0, len(data), 0
        header = f"""
            INSERT INTO {input_relation}
            ({", ".join(cols)}) VALUES """
        for i in range(n):
            sql_tmp = "("
            for d in data[i]:
                if isinstance(d, str):
                    d_str = d.replace("'", "''")
                    sql_tmp += f"'{d_str}'"
                elif d is None or d != d:
                    sql_tmp += "NULL"
                else:
                    sql_tmp += f"'{d}'"
                sql_tmp += ","
            sql_tmp = sql_tmp[:-1] + ");"
            query = header + sql_tmp
            if genSQL:
                sql += [clean_query(query)]
            else:
                try:
                    executeSQL(
                        query=query,
                        title=f"Insert a new line in the relation: {input_relation}.",
                    )
                    executeSQL("COMMIT;", title="Commit.")
                    total_rows += 1
                except Exception as e:
                    warning_message = f"Line {i} was skipped.\n{e}"
                    warnings.warn(warning_message, Warning)
        if genSQL:
            return sql
        else:
            return total_rows


def insert_verticapy_schema(
    model_name: str,
    model_type: str,
    model_save: dict,
    category: str = "VERTICAPY_MODELS",
):
    from verticapy.utils._toolbox import (
        executeSQL,
        quote_ident,
    )

    sql = "SELECT /*+LABEL(insert_verticapy_schema)*/ * FROM columns WHERE table_schema='verticapy';"
    result = executeSQL(sql, method="fetchrow", print_time_sql=False)
    if not (result):
        warning_message = (
            "The VerticaPy schema doesn't exist or is "
            "incomplete. The model can not be stored.\n"
            "Please use create_verticapy_schema function "
            "to set up the schema and the drop function to "
            "drop it if it is corrupted."
        )
        warnings.warn(warning_message, Warning)
    else:
        size = sys.getsizeof(model_save)
        create_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        try:
            model_name = quote_ident(model_name)
            result = executeSQL(
                query=f"""
                    SELECT 
                        /*+LABEL(insert_verticapy_schema)*/ * 
                    FROM verticapy.models
                    WHERE LOWER(model_name) = '{model_name.lower()}'""",
                method="fetchrow",
                print_time_sql=False,
            )
            if result:
                raise NameError(f"The model named {model_name} already exists.")
            else:
                executeSQL(
                    query=f"""
                        INSERT /*+LABEL(insert_verticapy_schema)*/ 
                        INTO verticapy.models(model_name, 
                                              category, 
                                              model_type, 
                                              create_time, 
                                              size) 
                                      VALUES ('{model_name}', 
                                              '{category}',
                                              '{model_type}',
                                              '{create_time}',
                                               {size});""",
                    print_time_sql=False,
                )
                executeSQL("COMMIT;", print_time_sql=False)
                for attr_name in model_save:
                    model_save_str = str(model_save[attr_name]).replace("'", "''")
                    executeSQL(
                        query=f"""
                            INSERT /*+LABEL(insert_verticapy_schema)*/
                            INTO verticapy.attr(model_name,
                                                attr_name,
                                                value) 
                                        VALUES ('{model_name}',
                                                '{attr_name}',
                                                '{model_save_str}');""",
                        print_time_sql=False,
                    )
                    executeSQL("COMMIT;", print_time_sql=False)
        except Exception as e:
            warning_message = f"The VerticaPy model could not be stored:\n{e}"
            warnings.warn(warning_message, Warning)
            raise
