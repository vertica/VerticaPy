# (c) Copyright [2018-2021] Micro Focus or one of its affiliates.
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
import warnings

# VerticaPy Modules
import verticapy
from verticapy.errors import QueryError
from verticapy import (
    executeSQL,
    vdf_from_relation,
    get_magic_options,
    vDataFrame,
    set_option,
    tablesample,
)


import re, time


@needs_local_scope
def sql(line, cell="", local_ns=None):

    # We don't want to display the query/time twice if the options are still on
    # So we save the previous configuration and turn them off.
    sql_on, time_on = verticapy.options["sql_on"], verticapy.options["time_on"]
    set_option("sql_on", False)
    set_option("time_on", False)

    try:

        queries = line if (not (cell) and (line)) else cell

        has_option = (len(queries) > 1 and queries[0] == "-" and queries[1] != "-") or (
            (cell) and (line)
        )
        options = {}

        if has_option:

            all_options_dict = get_magic_options(line)

            for option in all_options_dict:

                if option.lower() in ("-i", "-o",):
                    x = option.lower()[1:]
                    options[x] = all_options_dict[option]

                elif verticapy.options["print_info"]:
                    warning_message = (
                        f"\u26A0 Warning : The option '{option}' doesn't "
                        "exist, it was skipped."
                    )
                    warnings.warn(warning_message, Warning)

        if "i" in options:
            f = open(options["i"], "r")
            queries = f.read()
            f.close()

        queries = re.sub("--.+\n", "", queries)
        queries = queries.replace("\t", " ").replace("\n", " ")
        queries = re.sub(" +", " ", queries)
        variables = re.findall(":[A-Za-z0-9_]+", queries)
        for v in variables:
            val = locals()["local_ns"][v[1:]]
            try:
                import pandas as pd
                pandas_import = True
            except:
                pandas_import = False
            if isinstance(val, vDataFrame):
                val = val.__genSQL__()
            elif isinstance(val, tablesample):
                val = "({0}) VERTICAPY_SUBTABLE".format(val.to_sql())
            elif pandas_import and isinstance(val, pd.DataFrame):
                val = pandas_to_vertica(val).__genSQL__()
            queries = queries.replace(v, str(val))

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
                    result = vdf_from_relation("({}) x".format(query))
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

        elapsed_time = round(time.time() - start_time, 3)

        if verticapy.options["print_info"]:
            display(HTML(f"<div><b>Execution: </b> {elapsed_time}s</div>"))

        if isinstance(result, vDataFrame) and "o" in options:

            if options["o"][-4:] == "json":
                result.to_json(options["o"])
            else:
                result.to_csv(options["o"])

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
