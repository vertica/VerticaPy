"""
Copyright  (c)  2018-2025 Open Text  or  one  of its
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
from typing import Union, Optional

import verticapy._config.config as conf
from verticapy._utils._print import print_message
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import (
    clean_query,
    format_schema_table,
    format_type,
    quote_ident,
)
from verticapy._utils._sql._sys import _executeSQL
from verticapy.errors import MissingRelation


@save_verticapy_logs
def insert_into(
    table_name: str,
    data: list,
    schema: Optional[str] = None,
    column_names: Optional[list] = None,
    copy: bool = True,
    genSQL: bool = False,
) -> Union[int, str]:
    """
    Inserts the dataset into an existing Vertica
    table.

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
        If  set to  True, the  batch  insert  is
        converted  to  a   COPY  statement  with
        prepared   statements.  Otherwise,   the
        INSERTs   are  performed   sequentially.
    genSQL: bool, optional
        If  set to True, the SQL code that would
        be used to insert the data is generated,
        but not executed.

    Returns
    -------
    int
        The number of rows ingested.

    Examples
    --------
    For this example, we will use the Iris dataset.

    .. code-block:: python

        import verticapy.datasets as vpd

        data = vpd.load_iris()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_iris.html

    .. note::

        VerticaPy offers a wide range of sample
        datasets that are ideal for training
        and testing purposes. You can explore
        the full list of available datasets in
        the :ref:`api.datasets`, which provides
        detailed information on each dataset and
        how to use them effectively. These datasets
        are invaluable resources for honing your
        data analysis and machine learning skills
        within the VerticaPy environment.

    .. ipython:: python
        :suppress:

        import verticapy.datasets as vpd

        data = vpd.load_iris()

    We import the ``insert_into`` function and insert
    different element to the ``iris`` table.

    .. ipython:: python

        from verticapy.sql import insert_into

    You can insert all the elements at once with a single
    ``COPY`` statement by using the following command.

    .. ipython:: python

        insert_into(
            table_name = "iris",
            schema = "public",
            data = [
                [3.3, 4.5, 5.6, 7.8, "Iris-setosa"],
                [4.3, 4.7, 9.6, 1.8, "Iris-virginica"],
            ],
        )

    If you want to use multiple inserts to avoid a general
    failure and insert what you can, use the following approach.

    .. ipython:: python

        insert_into(
            table_name = "iris",
            schema = "public",
            data = [
                [3.3, 4.5, 5.6, 7.8, "Iris-setosa"],
                [4.3, 4.7, 9.6, 1.8, "Iris-virginica"],
            ],
            copy = False,
        )

    If you want to examine the generated SQL without executing it,
    use the following command.

    .. ipython:: python

        insert_into(
            table_name = "iris",
            schema = "public",
            data = [
                [3.3, 4.5, 5.6, 7.8, "Iris-setosa"],
                [4.3, 4.7, 9.6, 1.8, "Iris-virginica"],
            ],
            genSQL = True,
        )

    .. note::

        Set ``copy`` to ``False`` for multiple inserts.

    .. seealso::
        | :py:func:`~read_csv` : Ingests a CSV file using flex tables.
        | :py:func:`~read_json` : Ingests a JSON file using flex tables.
    """
    column_names = format_type(column_names, dtype=list)
    if not schema:
        schema = conf.get_option("temp_schema")
    input_relation = format_schema_table(schema, table_name)
    if not column_names:
        result = _executeSQL(
            query=f"""
                SELECT /*+LABEL('insert_into')*/
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
    cols = quote_ident(column_names)
    if copy and not genSQL:
        _executeSQL(
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
        _executeSQL("COMMIT;", title="Commit.")
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
                    _executeSQL(
                        query=query,
                        title=f"Insert a new line in the relation: {input_relation}.",
                    )
                    _executeSQL("COMMIT;", title="Commit.")
                    total_rows += 1
                except Exception as e:
                    warning_message = f"Line {i} was skipped.\n{e}"
                    print_message(warning_message, "warning")
        if genSQL:
            return sql
        else:
            return total_rows
