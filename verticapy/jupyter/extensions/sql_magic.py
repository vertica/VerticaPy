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
##
#  _____  _____ _      ___  ___  ___  _____ _____ _____
# /  ___||  _  | |     |  \/  | / _ \|  __ \_   _/  __ \
# \ `--. | | | | |     | .  . |/ /_\ \ |  \/ | | | /  \/
#  `--. \| | | | |     | |\/| ||  _  | | __  | | | |
# /\__/ /\ \/' / |____ | |  | || | | | |_\ \_| |_| \__/\
# \____/  \_/\_\_____/ \_|  |_/\_| |_/\____/\___/ \____/
#
##
import re
import time
import warnings
from typing import Optional, TYPE_CHECKING

from IPython.core.magic import needs_local_scope
from IPython.display import display, HTML

import verticapy._config.config as conf
from verticapy._utils._object import create_new_vdf
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._check import is_procedure
from verticapy._utils._sql._dblink import replace_external_queries
from verticapy._utils._sql._format import (
    clean_query,
    replace_vars_in_query,
)
from verticapy._utils._sql._sys import _executeSQL
from verticapy.connection import current_cursor
from verticapy.connection.global_connection import get_global_connection
from verticapy.errors import QueryError

from verticapy.jupyter.extensions._utils import get_magic_options

if conf.get_import_success("graphviz"):
    import graphviz
    from graphviz import Source

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame

SPECIAL_WORDS = (
    # ML Algos
    "ARIMA",
    "AUTOREGRESSOR",
    "BALANCE",
    "BISECTING_KMEANS",
    "CROSS_VALIDATE",
    "DETECT_OUTLIERS",
    "IFOREST",
    "IMPUTE",
    "KMEANS",
    "KPROTOTYPES",
    "LINEAR_REG",
    "LOGISTIC_REG",
    "MOVING_AVERAGE",
    "NAIVE_BAYES",
    "NORMALIZE",
    "NORMALIZE_FIT",
    "ONE_HOT_ENCODER_FIT",
    "PCA",
    "POISSON_REG",
    "RF_CLASSIFIER",
    "RF_REGRESSOR",
    "SVD",
    "SVM_CLASSIFIER",
    "SVM_REGRESSOR",
    "XGB_CLASSIFIER",
    "XGB_REGRESSOR",
    # ML Management
    "CHANGE_MODEL_STATUS",
    "EXPORT_MODELS",
    "IMPORT_MODELS",
    "REGISTER_MODEL",
    "UPGRADE_MODEL",
)


@save_verticapy_logs
@needs_local_scope
def sql_magic(
    line: str, cell: Optional[str] = None, local_ns: Optional[dict] = None
) -> "vDataFrame":
    """
    Executes SQL queries in the Jupyter cell.

    Parameters
    ----------
    -c / --command : str, optional
        SQL Command to execute.
    -f / --file : str, optional
        Input  File. You  can use this option
        if  you  want  to  execute the  input
        file.
    -ncols : int, optional
        Maximum number of columns to display.
    -nrows : int, optional
        Maximum  number  of rows to  display.
    -o / --output : str, optional
        Output File. You  can use this option
        if  you want to export the  result of
        the query to  the CSV or JSON format.

    Returns
    -------
    vDataFrame
        Result of the query

    Examples
    --------
    The following examples demonstrate:

    * Setting up the environment
    * Using SQL Magic
    * Getting the vDataFrame of a query
    * Using variables inside a query
    * Limiting the number of rows and columns
    * Exporting a query to JSON or CSV
    * Executing SQL files

    Setting up the environment
    ==========================
    If you don't already have a connection, create one:

    .. code-block:: python

        import verticapy as vp

        # Save a new connection
        vp.new_connection(
            {
                "host": "10.211.55.14",
                "port": "5433",
                "database": "testdb",
                "password": "XxX",
                "user": "dbadmin",
            },
            name = "VerticaDSN",
        )

    If you already have a connection in a connection
    file, you can use it by running the following
    command:

    .. code-block:: python

        # Connect using the VerticaDSN connection
        vp.connect("VerticaDSN")

    Load the extension:

    .. code-block:: python

        %load_ext verticapy.sql

    Load a sample dataset. These sample datasets
    are loaded into the public schema by default.
    You can specify a target schema with the 'name'
    and 'schema' parameters:

    .. code-block:: python

        from verticapy.datasets import load_titanic, load_iris

        titanic = load_titanic()
        iris = load_iris()

    SQL Magic
    =========
    Use '%%sql' to run a query on the dataset:

    .. code-block:: python

        %%sql
        SELECT
            survived,
            AVG(fare) AS avg_fare,
            AVG(age) AS avg_age
        FROM titanic
        GROUP BY 1;

    **Execution**: 0.006s

    .. ipython:: python
        :suppress:

        import verticapy as vp
        from verticapy.datasets import load_titanic, load_iris

        titanic = load_titanic()
        iris = load_iris()
        %load_ext verticapy.sql

    .. ipython:: python
        :suppress:

        %sql -c 'SELECT survived, AVG(fare) AS avg_fare, AVG(age) AS avg_age FROM titanic GROUP BY 1;'

    .. ipython:: python
        :suppress:

        t = _
        html_file = open("SPHINX_DIRECTORY/figures/jupyter_extensions_sql_magic_sql_magic.html", "w")
        html_file.write(t._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/jupyter_extensions_sql_magic_sql_magic.html

    You can also run queries with '%sql' and the '-c' option:

    .. code-block:: python

        %sql -c 'SELECT DISTINCT Species FROM iris;'

    **Execution**: 0.006s

    .. ipython:: python
        :suppress:

        %sql -c 'SELECT DISTINCT Species FROM iris;'

    .. ipython:: python
        :suppress:

        result = _
        html_file = open("SPHINX_DIRECTORY/figures/jupyter_extensions_sql_magic_sql_magic_2.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/jupyter_extensions_sql_magic_sql_magic_2.html

    You can use a single cell for multiple queries:

    .. warning:: Don't forget to include a semicolon at the end of each query.

    .. code-block:: python

        %%sql
        DROP TABLE IF EXISTS test;
        CREATE TABLE test AS SELECT 'Badr Ouali' AS name;
        SELECT * FROM test;

    **Execution**: 0.05s

    .. ipython:: python
        :suppress:
        :okwarning:

        %%sql
        DROP TABLE IF EXISTS test;
        CREATE TABLE test AS SELECT 'Badr Ouali' AS name;
        SELECT * FROM test;

    .. ipython:: python
        :suppress:

        test_table = _
        html_file = open("SPHINX_DIRECTORY/figures/jupyter_extensions_sql_magic_sql_magic_3.html", "w")
        html_file.write(test_table._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/jupyter_extensions_sql_magic_sql_magic_3.html

    To add comments to a query, use one of the following comment syntaxes:

    .. warning:: Vertica uses '/' and '/' for both comments and query hints. Whenever possible, use '--' to avoid conflicts.

    .. code-block:: python

        %%sql
        -- Comment Test
        /* My Vertica Version */
        SELECT version(); -- Select my current version

    **Execution**: 0.005s

    .. ipython:: python
        :suppress:

        %%sql
        -- Comment Test
        /* My Vertica Version */
        SELECT version(); -- Select my current version

    .. ipython:: python
        :suppress:

        test_comment = _
        html_file = open("SPHINX_DIRECTORY/figures/jupyter_extensions_sql_magic_sql_magic_4.html", "w")
        html_file.write(test_comment._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/jupyter_extensions_sql_magic_sql_magic_4.html

    Get the vDataFrame of a query
    =============================

    Results of a SQL Magic query are stored in a vDataFrame, which is assigned
    to a temporary variable called '_'. You can assign this temporary varaible
    to a new variable to save your results.

    .. code-block:: python

        %%sql
        SELECT
            age,
            fare,
            pclass
        FROM titanic
        WHERE age IS NOT NULL AND fare IS NOT NULL;

    **Execution**: 0.007s

    .. ipython:: python
        :suppress:

        %%sql
        SELECT
            age,
            fare,
            pclass
        FROM titanic
        WHERE age IS NOT NULL AND fare IS NOT NULL;

    .. ipython:: python
        :suppress:

        titanic_clean = _
        html_file = open("SPHINX_DIRECTORY/figures/jupyter_extensions_sql_magic_sql_magic_5.html", "w")
        html_file.write(titanic_clean._repr_html_())
        html_file.close()

    .. raw:: html
        :file: figures/jupyter_extensions_sql_magic_sql_magic_5.html

    Assign the results to a new variable:

    .. code-block:: python

        titanic_clean = _
        display(titanic_clean)

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/jupyter_extensions_sql_magic_sql_magic_5.html

    Temporary results are stored in a vDataFrame, allowing you to call
    vDataFrame methods:

    .. ipython:: python

        titanic_clean["age"].max()

    Using variables inside a query
    ==============================

    You can use variables in a SQL query with the ':' operator. This
    variable can be a vDataFrame, a TableSample, a pandas.DataFrame,
    or any standard Python type.

    .. code-block:: python

        import verticapy.sql.functions as vpf

        class_fare = titanic_clean.groupby(
            "pclass",
            [vpf.avg(titanic_clean["fare"])._as("avg_fare")]
        )
        class_fare

    .. ipython:: python
        :suppress:

        import verticapy.sql.functions as vpf
        class_fare = titanic_clean.groupby("pclass",
                                   [vpf.avg(titanic_clean["fare"])._as("avg_fare")])
        html_file = open("SPHINX_DIRECTORY/figures/jupyter_extensions_sql_magic_sql_magic_6.html", "w")
        html_file.write(class_fare._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/jupyter_extensions_sql_magic_sql_magic_6.html

    Use the 'class_fare' variable in a SQL query:

    .. code-block:: python

        %%sql
        SELECT
            x.*,
            y.avg_fare
        FROM titanic AS x LEFT JOIN (SELECT * FROM :class_fare) AS y
        ON x.pclass = y.pclass;

    **Execution**: 0.011s

    .. ipython:: python
        :suppress:

        %%sql
        SELECT
            x.*,
            y.avg_fare
        FROM titanic AS x LEFT JOIN (SELECT * FROM :class_fare) AS y
        ON x.pclass = y.pclass;

    .. ipython:: python
        :suppress:

        titanic_class = _
        html_file = open("SPHINX_DIRECTORY/figures/jupyter_extensions_sql_magic_sql_magic_7.html", "w")
        html_file.write(titanic_class._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/jupyter_extensions_sql_magic_sql_magic_7.html

    You can do the same with a TableSample:

    .. code-block:: python

        tb = {"name": ["Badr", "Arash"], "specialty": ["Python", "C++"]}
        tb = vp.TableSample(tb)

    .. code-block:: python

        %%sql
        SELECT * FROM :tb;

    **Execution**: 0.014s

    .. ipython:: python
        :suppress:

        tb = {"name": ["Badr", "Arash"], "specialty": ["Python", "C++"]}
        tb = vp.TableSample(tb)

    .. ipython:: python
        :suppress:

        %%sql
        SELECT * FROM :tb;

    .. ipython:: python
        :suppress:

        tb_test = _
        html_file = open("SPHINX_DIRECTORY/figures/jupyter_extensions_sql_magic_sql_magic_8.html", "w")
        html_file.write(tb_test._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/jupyter_extensions_sql_magic_sql_magic_8.html

    And with a pandas.DataFrame:

    .. ipython:: python

        titanic_pandas = titanic.to_pandas()
        titanic_pandas

    .. code-block:: python

        %%sql
        SELECT * FROM :titanic_pandas;

    .. ipython:: python
        :suppress:

        %%sql
        SELECT * FROM :titanic_pandas;

    .. ipython:: python
        :suppress:

        pandas_test = _
        html_file = open("SPHINX_DIRECTORY/figures/jupyter_extensions_sql_magic_sql_magic_9.html", "w")
        html_file.write(pandas_test._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/jupyter_extensions_sql_magic_sql_magic_9.html

    You can also use a sample loop with a variable:

    .. note:: VerticaPy will store the object in a temporary local table before executing the overall query, which facilitates integration with in-memory objects.

    .. code-block:: python

        %sql -c 'DROP TABLE IF EXISTS test;'
        %sql -c 'CREATE TABLE test (id INT);'
        for i in range(4):
            %sql -c 'INSERT INTO test(id) SELECT :i;'

    `DROP`

    **Execution**: 0.014s

    `CREATE`

    **Execution**: 0.008s

    `INSERT`

    **Execution**: 0.05s

    `INSERT`

    **Execution**: 0.015s

    `INSERT`

    **Execution**: 0.016s

    `INSERT`

    **Execution**: 0.013s

    .. ipython:: python

        %sql -c 'DROP TABLE IF EXISTS test;'
        %sql -c 'CREATE TABLE test (id INT);'
        for i in range(4):
            %sql -c 'INSERT INTO test(id) SELECT :i;'

    .. code-block:: python

        %%sql
        SELECT * FROM test;

    **Execution**: 0.005s

    .. ipython:: python
        :suppress:

        %%sql
        SELECT * FROM test;

    .. ipython:: python
        :suppress:

        loop_test = _
        html_file = open("SPHINX_DIRECTORY/figures/jupyter_extensions_sql_magic_sql_magic_10.html", "w")
        html_file.write(loop_test._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/jupyter_extensions_sql_magic_sql_magic_10.html

    Change the maximum number of rows/columns to display
    ====================================================

    Use the '-nrows' and '-ncols' option to limit the number of rows and columns displayed:

    .. code-block:: python

        %%sql -nrows 5 -ncols 2
        SELECT * FROM public.titanic;

    **Execution**: 0.008s

    .. ipython:: python
        :suppress:

        %%sql -nrows 5 -ncols 2
        SELECT * FROM public.titanic;

    .. ipython:: python
        :suppress:

        limit_test = _
        html_file = open("SPHINX_DIRECTORY/figures/jupyter_extensions_sql_magic_sql_magic_11.html", "w")
        html_file.write(limit_test._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/jupyter_extensions_sql_magic_sql_magic_11.html

    Export results to a JSON or CSV file
    ====================================

    To export the results of a query to a CSV file:

    .. code-block:: python

        %%sql -o titanic_age_clean.csv
        SELECT
            *
        FROM public.titanic
        WHERE age IS NOT NULL LIMIT 5;

    **Execution**: 0.008s

    .. ipython:: python
        :suppress:

        %%sql -o titanic_age_clean.csv
        SELECT
            *
        FROM public.titanic
        WHERE age IS NOT NULL LIMIT 5;

    .. ipython:: python
        :suppress:

        export_test = _
        html_file = open("SPHINX_DIRECTORY/figures/jupyter_extensions_sql_magic_sql_magic_12.html", "w")
        html_file.write(export_test._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/jupyter_extensions_sql_magic_sql_magic_12.html

    .. ipython:: python

        file = open("titanic_age_clean.csv", "r")
        print(file.read())
        file.close()

    To export the results of a query to a JSON file:

    .. code-block:: python

        %%sql -o titanic_age_clean.json
        SELECT
            *
        FROM public.titanic
        WHERE age IS NOT NULL LIMIT 5;

    **Execution**: 0.008s

    .. ipython:: python
        :suppress:

        %%sql -o titanic_age_clean.json
        SELECT
            *
        FROM public.titanic
        WHERE age IS NOT NULL LIMIT 5;

    .. ipython:: python
        :suppress:

        json_test = _
        html_file = open("SPHINX_DIRECTORY/figures/jupyter_extensions_sql_magic_sql_magic_13.html", "w")
        html_file.write(json_test._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/jupyter_extensions_sql_magic_sql_magic_13.html

    .. ipython:: python

        file = open("titanic_age_clean.json", "r")
        print(file.read())
        file.close()

    Execute SQL files
    =================

    To execute commands from a SQL file, use the following syntax:

    .. ipython:: python

        file = open("query.sql", "w+")
        file.write("SELECT version();")
        file.close()

    Using the ``-f`` option, we can easily read SQL files:

    .. code-block:: python

        %sql -f query.sql

    **Execution**: 0.006s

    .. ipython:: python
        :suppress:

        %sql -f query.sql

    .. ipython:: python
        :suppress:

        sql_test = _
        html_file = open("SPHINX_DIRECTORY/figures/jupyter_extensions_sql_magic_sql_magic_14.html", "w")
        html_file.write(sql_test._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/jupyter_extensions_sql_magic_sql_magic_14.html

    Connect to an external database
    ===============================

    Since v0.12.0, it is possible to connect to external Databases using the connection
    symbol. Detailled examples are available in
    `this notebook <https://www.vertica.com/python/workshop/full_stack/dblink_integration/>`_.
    """

    # We don't want to display the query/time twice if
    # the options are still on.
    # So we save the previous configuration and turn
    # them off.
    sql_on, time_on = conf.get_option("sql_on"), conf.get_option("time_on")
    conf.set_option("sql_on", False)
    conf.set_option("time_on", False)

    try:
        # Initialization
        queries = "" if (not cell and (line)) else cell

        # Options
        options = {}
        options_dict = get_magic_options(line)

        for option in options_dict:
            if option.lower() in (
                "-f",
                "--file",
                "-o",
                "--output",
                "-nrows",
                "-ncols",
                "-c",
                "--command",
            ):
                if option.lower() in ("-f", "--file"):
                    if "-f" in options:
                        raise ValueError("Duplicate option '-f'.")
                    options["-f"] = options_dict[option]
                elif option.lower() in ("-o", "--output"):
                    if "-o" in options:
                        raise ValueError("Duplicate option '-o'.")
                    options["-o"] = options_dict[option]
                elif option.lower() in ("-c", "--command"):
                    if "-c" in options:
                        raise ValueError("Duplicate option '-c'.")
                    options["-c"] = options_dict[option]
                elif option.lower() in ("-nrows",):
                    if "-nrows" in options:
                        raise ValueError("Duplicate option '-nrows'.")
                    options["-nrows"] = int(options_dict[option])
                elif option.lower() in ("-ncols",):
                    if "-ncols" in options:
                        raise ValueError("Duplicate option '-ncols'.")
                    options["-ncols"] = int(options_dict[option])

            elif conf.get_option("print_info"):
                warning_message = (
                    f"\u26A0 Warning : The option '{option}' doesn't "
                    "exist, it was skipped."
                )
                warnings.warn(warning_message, Warning)

        if "-f" in options and "-c" in options:
            raise ValueError(
                "Do not find which query to run: One of "
                "the options '-f' and '-c' must be empty."
            )

        if cell and ("-f" in options or "-c" in options):
            raise ValueError("Cell must be empty when using options '-f' or '-c'.")

        if "-f" in options:
            with open(options["-f"], "r", encoding="utf-8") as f:
                queries = f.read()

        elif "-c" in options:
            queries = options["-c"]

        # Case when it is a procedure
        if is_procedure(queries):
            current_cursor().execute(queries)
            print("CREATE")
            return

        # Cleaning the Query
        queries = clean_query(queries)
        queries = replace_vars_in_query(queries, locals()["local_ns"])
        queries = replace_external_queries(queries)

        # Looking at very specific external queries symbols
        gb_conn = get_global_connection()
        for s in gb_conn.special_symbols:
            external_queries = re.findall(
                f"\\{s}\\{s}\\{s}(.*?)\\{s}\\{s}\\{s}", queries
            )
            warning_message = (
                f"External Query detected but no corresponding Connection "
                "Identifier Database is defined (Using the symbol '{s}'). "
                "Use the function connect.set_external_connection to set "
                "one with the correct symbol."
            )

            if external_queries:
                warnings.warn(warning_message, Warning)

        n, i, all_split = len(queries), 0, []

        while i < n and queries[n - i - 1] in (";", " ", "\n"):
            i += 1

        queries = queries[: n - i]
        i, n = 0, n - i

        while i < n:
            if queries[i] == '"':
                i += 1
                while i < n and queries[i] != '"':
                    i += 1
            elif queries[i] == "'":
                i += 1
                while i < n and queries[i] != "'":
                    i += 1
            elif queries[i] == ";":
                all_split += [i]
            i += 1

        all_split = [0] + all_split + [n]
        m = len(all_split)
        queries = [queries[all_split[i] : all_split[i + 1]] for i in range(m - 1)]
        n = len(queries)

        for i in range(n):
            query = queries[i]
            while len(query) > 0 and query.endswith((";", " ")):
                query = query[0:-1]
            while len(query) > 0 and query.startswith((";", " ")):
                query = query[1:]
            queries[i] = query

        queries_tmp, i = [], 0

        while i < n:
            query = queries[i]
            if (i < n - 1) and (queries[i + 1].lower() == "end"):
                query += f"; {queries[i + 1]}"
                i += 1
            queries_tmp += [query]
            i += 1

        queries, n = queries_tmp, len(queries_tmp)
        result, start_time = None, time.time()

        # Executing the Queries

        for i in range(n):
            query = queries[i]

            query_words = query.split(" ")

            idx = 0 if query_words[0] else 1
            query_type = query_words[idx].upper().replace("(", "")
            if len(query_words) > 1:
                query_subtype = query_words[idx + 1].upper()
            else:
                query_subtype = "UNDEFINED"

            if len(query_type) > 1 and query_type.startswith(("/*", "--")):
                query_type = "undefined"

            if (query_type == "COPY") and ("from local" in query.lower()):
                query = re.split("from local", query, flags=re.IGNORECASE)
                if query[1].split(" ")[0]:
                    file_name = query[1].split(" ")[0]
                else:
                    file_name = query[1].split(" ")[1]
                query = (
                    "".join(query[0])
                    + "FROM"
                    + "".join(query[1]).replace(file_name, "STDIN")
                )
                if (file_name[0] == file_name[-1]) and (file_name[0] in ('"', "'")):
                    file_name = file_name[1:-1]

                _executeSQL(query, method="copy", path=file_name, print_time_sql=False)

            elif (
                (i < n - 1)
                or (
                    (i == n - 1)
                    and (
                        query_type.lower()
                        not in ("select", "show", "with", "undefined")
                    )
                )
            ) and query_type.lower() not in ("explain",):
                error = ""

                try:
                    _executeSQL(query, print_time_sql=False)

                except Exception as e:
                    error = str(e)

                if conf.get_option("print_info") and (
                    "Severity: ERROR, Message: User defined transform must return at least one column"
                    in error
                    and "DBLINK" in error
                ):
                    print(query_type)

                elif error:
                    raise QueryError(error)

                elif conf.get_option("print_info"):
                    print(query_type)

            else:
                error = ""

                if query_type.lower() in ("show",):
                    final_result = _executeSQL(
                        query, method="fetchall", print_time_sql=False
                    )
                    columns = [d.name for d in current_cursor().description]
                    result = create_new_vdf(
                        final_result,
                        usecols=columns,
                    )
                    continue
                elif query_type.lower() in ("explain",):
                    final_result = _executeSQL(
                        query, method="fetchall", print_time_sql=False
                    )
                    final_result = "\n".join([l[0] for l in final_result])
                    tree = "digraph G {" + final_result.split("\ndigraph G {")[1]
                    plan = final_result.split("\ndigraph G {")[0]
                    print(plan)
                    if i < n - 1 or not conf.get_import_success("graphviz"):
                        print(tree)
                    else:
                        res = graphviz.Source(tree)
                        if i == n - 1:
                            return res
                        else:
                            print(res)
                is_vdf = False
                if not (query_subtype.upper().startswith(SPECIAL_WORDS)):
                    try:
                        result = create_new_vdf(
                            query,
                            _is_sql_magic=True,
                        )
                        result._vars["sql_magic_result"] = True
                        # Display parameters
                        if "-nrows" in options:
                            result._vars["max_rows"] = options["-nrows"]
                        if "-ncols" in options:
                            result._vars["max_columns"] = options["-ncols"]
                        is_vdf = True
                    except:
                        pass  # we could not create a vDataFrame out of the query.

                if not (is_vdf):
                    try:
                        final_result = _executeSQL(
                            query, method="fetchfirstelem", print_time_sql=False
                        )
                        if final_result and conf.get_option("print_info"):
                            print(final_result)
                        elif (
                            query_subtype.upper().startswith(SPECIAL_WORDS)
                        ) and conf.get_option("print_info"):
                            print(query_subtype.upper())
                        elif conf.get_option("print_info"):
                            print(query_type)

                    except Exception as e:
                        error = str(e)

                # If it fails because no elements were returned in the DBLINK UDx
                # - we do not display the error message
                if (
                    "Severity: ERROR, Message: User defined transform must return at least one column"
                    in error
                    and "DBLINK" in error
                ):
                    if conf.get_option("print_info"):
                        print(query_type)

                elif error:
                    raise QueryError(error)

        # Exporting the result

        if (
            hasattr(result, "object_type")
            and (result.object_type == "vDataFrame")
            and ("-o" in options)
        ):
            if options["-o"][-4:] == "json":
                result.to_json(options["-o"])
            else:
                result.to_csv(options["-o"])

        # Displaying the time

        elapsed_time = round(time.time() - start_time, 3)

        if conf.get_option("print_info"):
            display(HTML(f"<div><b>Execution: </b> {elapsed_time}s</div>"))

        return result

    finally:
        # we load the previous configuration before returning the result.
        conf.set_option("sql_on", sql_on)
        conf.set_option("time_on", time_on)


def load_ipython_extension(ipython) -> None:
    ipython.register_magic_function(sql_magic, "cell", "sql")
    ipython.register_magic_function(sql_magic, "line", "sql")
