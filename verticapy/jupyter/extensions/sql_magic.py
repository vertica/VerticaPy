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
from verticapy._utils._sql._dblink import replace_external_queries
from verticapy._utils._sql._format import (
    clean_query,
    replace_vars_in_query,
)
from verticapy._utils._sql._sys import _executeSQL
from verticapy.connection.global_connection import get_global_connection
from verticapy.errors import QueryError

from verticapy.jupyter.extensions._utils import get_magic_options

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame


@save_verticapy_logs
@needs_local_scope
def sql_magic(
    line: str, cell: Optional[str] = None, local_ns: Optional[dict] = None
) -> "vDataFrame":
    """
    Executes SQL queries in the Jupyter cell.

    -c / --command : SQL Command to execute.

    -f  /   --file : Input  File. You  can use this option
                     if  you  want  to  execute the  input
                     file.

            -ncols : Maximum number of columns to display.

            -nrows : Maximum  number  of rows to  display.

     -o / --output : Output File. You  can use this option
                     if  you want to export the  result of
                     the query to  the CSV or JSON format.
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

            if query.split(" ")[0]:
                query_type = query.split(" ")[0].upper().replace("(", "")

            else:
                query_type = query.split(" ")[1].upper().replace("(", "")

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

            elif (i < n - 1) or (
                (i == n - 1)
                and (query_type.lower() not in ("select", "with", "undefined"))
            ):
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

                except:
                    try:
                        final_result = _executeSQL(
                            query, method="fetchfirstelem", print_time_sql=False
                        )
                        if final_result and conf.get_option("print_info"):
                            print(final_result)
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
