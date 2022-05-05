# (c) Copyright [2018-2022] Micro Focus or one of its affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# |_     |~) _  _| _  /~\    _ |.
# |_)\/  |_)(_|(_||   \_/|_|(_|||
#    /
#              ____________       ______
#             / __        `\     /     /
#            |  \/         /    /     /
#            |______      /    /     /
#                   |____/    /     /
#          _____________     /     /
#          \           /    /     /
#           \         /    /     /
#            \_______/    /     /
#             ______     /     /
#             \    /    /     /
#              \  /    /     /
#               \/    /     /
#                    /     /
#                   /     /
#                   \    /
#                    \  /
#                     \/
#                    _
# \  / _  __|_. _ _ |_)
#  \/ (/_|  | |(_(_|| \/
#                     /
# VerticaPy is a Python library with scikit-like functionality for conducting
# data science projects on data stored in Vertica, taking advantage Vertica’s
# speed and built-in analytics and machine learning features. It supports the
# entire data science life cycle, uses a ‘pipeline’ mechanism to sequentialize
# data transformation operations, and offers beautiful graphical options.
#
# VerticaPy aims to do all of the above. The idea is simple: instead of moving
# data around for processing, VerticaPy brings the logic to the data.
#
##
#  _____  _____ _      ___  ___  ___  _____ _____ _____
# /  ___||  _  | |     |  \/  | / _ \|  __ \_   _/  __ \
# \ `--. | | | | |     | .  . |/ /_\ \ |  \/ | | | /  \/
#  `--. \| | | | |     | |\/| ||  _  | | __  | | | |
# /\__/ /\ \/' / |____ | |  | || | | | |_\ \_| |_| \__/\
# \____/  \_/\_\_____/ \_|  |_/\_| |_/\____/\___/ \____/
#
##
#
# ---#
# Jupyter Modules
from IPython.core.magic import needs_local_scope
from IPython.core.display import HTML, display

# Standard Python Modules
import warnings, re, time

# Other modules
import pandas as pd

# VerticaPy Modules
import verticapy
from verticapy.errors import QueryError, ParameterError
from verticapy import (
    executeSQL,
    vDataFrameSQL,
    get_magic_options,
    vDataFrame,
    set_option,
    tablesample,
    clean_query,
    replace_vars_in_query,
)

# ---#
@needs_local_scope
def sql(line, cell="", local_ns=None):

    # We don't want to display the query/time twice if the options are still on
    # So we save the previous configuration and turn them off.
    sql_on, time_on = verticapy.options["sql_on"], verticapy.options["time_on"]
    set_option("sql_on", False)
    set_option("time_on", False)

    try:

        # Initialization
        queries = "" if (not (cell) and (line)) else cell

        # Options
        options = {}
        all_options_dict = get_magic_options(line)

        for option in all_options_dict:

            if option.lower() in ("-f", "--file", "-o", "--output", "-nrows", "-ncols", "-c", "--command"):

                if option.lower() in ("-f", "--file"):
                    if "-f" in options:
                        raise ParameterError("Duplicate option '-f'.")
                    options["-f"] = all_options_dict[option]
                elif option.lower() in ("-o", "--output"):
                    if "-o" in options:
                        raise ParameterError("Duplicate option '-o'.")
                    options["-o"] = all_options_dict[option]
                elif option.lower() in ("-c", "--command"):
                    if "-c" in options:
                        raise ParameterError("Duplicate option '-c'.")
                    options["-c"] = all_options_dict[option]
                elif option.lower() in ("-nrows",):
                    if "-nrows" in options:
                        raise ParameterError("Duplicate option '-nrows'.")
                    options["-nrows"] = int(all_options_dict[option])
                elif option.lower() in ("-ncols",):
                    if "-ncols" in options:
                        raise ParameterError("Duplicate option '-ncols'.")
                    options["-ncols"] = int(all_options_dict[option])

            elif verticapy.options["print_info"]:
                warning_message = (
                    f"\u26A0 Warning : The option '{option}' doesn't "
                    "exist, it was skipped."
                )
                warnings.warn(warning_message, Warning)

        if "-f" in options and "-c" in options:
            raise ParameterError("Do not find which query to run: One of "
                                 "the options '-f' and '-c' must be empty.")

        if cell and ("-f" in options or "-c" in options):
            raise ParameterError("Cell must be empty when using options '-f' or '-c'.")

        if "-f" in options:
            f = open(options["-f"], "r")
            queries = f.read()
            f.close()

        elif "-c" in options:
            queries = options["-c"]

        # Cleaning the Query
        queries = clean_query(queries)
        queries = replace_vars_in_query(queries, locals()["local_ns"])

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
            while len(query) > 0 and (query[-1] in (";", " ")):
                query = query[0:-1]
            while len(query) > 0 and (query[0] in (";", " ")):
                query = query[1:]
            queries[i] = query

        queries_tmp, i = [], 0

        while i < n:

            query = queries[i]
            if (i < n - 1) and (queries[i + 1].lower() == "end"):
                query += "; {}".format(queries[i + 1])
                i += 1
            queries_tmp += [query]
            i += 1

        queries, n = queries_tmp, len(queries_tmp)
        result, start_time = None, time.time()

        # Executing the Queries

        for i in range(n):

            query = queries[i]

            if query.split(" ")[0]:
                query_type = query.split(" ")[0].upper()
            else:
                query_type = query.split(" ")[1].upper()

            if len(query_type) > 1 and query_type[0:2] in ("/*", "--"):
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

                executeSQL(query, method="copy", path=file_name, print_time_sql=False)

            elif (i < n - 1) or (
                (i == n - 1)
                and (query_type.lower() not in ("select", "with", "undefined"))
            ):

                executeSQL(query, print_time_sql=False)
                if verticapy.options["print_info"]:
                    print(query_type)

            else:

                error = ""
                try:
                    result = vDataFrameSQL("({}) x".format(query))
                    # Display parameters
                    if "-nrows" in options:
                        result._VERTICAPY_VARIABLES_["max_rows"] = options["-nrows"]
                    if "-ncols" in options:
                        result._VERTICAPY_VARIABLES_["max_columns"] = options["-ncols"]

                except:
                    try:
                        final_result = executeSQL(
                            query, method="fetchfirstelem", print_time_sql=False
                        )
                        if final_result and verticapy.options["print_info"]:
                            print(final_result[0])
                        elif verticapy.options["print_info"]:
                            print(query_type)
                    except Exception as e:
                        error = e

                if error:
                    raise QueryError(error)

        # Displaying the information

        elapsed_time = round(time.time() - start_time, 3)

        if verticapy.options["print_info"]:
            display(HTML(f"<div><b>Execution: </b> {elapsed_time}s</div>"))

        # Exporting the result

        if isinstance(result, vDataFrame) and "-o" in options:

            if options["-o"][-4:] == "json":
                result.to_json(options["-o"])
            else:
                result.to_csv(options["-o"])

        # we load the previous configuration before returning the result.
        set_option("sql_on", sql_on)
        set_option("time_on", time_on)

        return result

    except:

        # If it fails, we load the previous configuration before raising the error.
        set_option("sql_on", sql_on)
        set_option("time_on", time_on)
        
        raise


# ---#
def load_ipython_extension(ipython):
    ipython.register_magic_function(sql, "cell")
    ipython.register_magic_function(sql, "line")
