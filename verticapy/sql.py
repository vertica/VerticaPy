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
# VerticaPy is a Python library with scikit-like functionality to use to conduct
# data science projects on data stored in Vertica, taking advantage Vertica’s
# speed and built-in analytics and machine learning features. It supports the
# entire data science life cycle, uses a ‘pipeline’ mechanism to sequentialize
# data transformation operations, and offers beautiful graphical options.
#
# VerticaPy aims to solve all of these problems. The idea is simple: instead
# of moving data around for processing, VerticaPy brings the logic to the data.
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
def sql(line, cell=""):
    import verticapy
    from verticapy.connect import read_auto_connect
    from verticapy.utilities import readSQL
    from verticapy.utilities import vdf_from_relation
    from IPython.core.display import HTML, display
    import time
    import re
    import vertica_python
    from verticapy.errors import QueryError

    version = vertica_python.__version__.split(".")
    version = [int(elem) for elem in version]
    conn = read_auto_connect()
    cursor = conn.cursor()
    queries = line if (not (cell) and (line)) else cell
    options = {"limit": 100, "vdf": False}
    queries = queries.replace("\t", " ")
    queries = queries.replace("\n", " ")
    queries = re.sub(" +", " ", queries)
    if (cell) and (line):
        line = re.sub(" +", " ", line)
        all_options_tmp = line.split(" ")
        all_options = []
        for elem in all_options_tmp:
            if elem != "":
                all_options += [elem]
        n, i, all_options_dict = len(all_options), 0, {}
        while i < n:
            all_options_dict[all_options[i]] = all_options[i + 1]
            i += 2
        for option in all_options_dict:
            if option.lower() == "-limit":
                options["limit"] = int(all_options_dict[option])
            elif option.lower() == "-vdf":
                options["vdf"] = bool(all_options_dict[option])
            elif verticapy.options["print_info"]:
                print(
                    "\u26A0 Warning : The option '{}' doesn't exist, it was skipped.".format(
                        option
                    )
                )
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
    start_time = time.time()
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
    result = None
    for i in range(n):
        query = queries[i]
        query_type = (
            query.split(" ")[0].upper()
            if (query.split(" ")[0])
            else query.split(" ")[1].upper()
        )
        if (
            (query_type == "COPY")
            and ("from local" in query.lower())
            and (version[0] == 0)
            and (version[1] < 11)
        ):
            query = re.split("from local", query, flags=re.IGNORECASE)
            file_name = (
                query[1].split(" ")[0]
                if (query[1].split(" ")[0])
                else query[1].split(" ")[1]
            )
            query = (
                "".join(query[0])
                + "FROM"
                + "".join(query[1]).replace(file_name, "STDIN")
            )
            if (file_name[0] == file_name[-1]) and (file_name[0] in ('"', "'")):
                file_name = file_name[1:-1]
            with open(file_name, "r") as fs:
                cursor.copy(query, fs)
        elif (i < n - 1) or ((i == n - 1) and (query_type.lower() != "select")):
            cursor.execute(query)
            if verticapy.options["print_info"]:
                print(query_type)
        else:
            error = ""
            try:
                if options["vdf"]:
                    result = vdf_from_relation("({}) x".format(query), cursor=cursor)
                else:
                    result = readSQL(query, cursor=cursor, limit=options["limit"],)
            except:
                try:
                    cursor.execute(query)
                    final_result = cursor.fetchone()
                    if final_result and verticapy.options["print_info"]:
                        print(final_result[0])
                    elif verticapy.options["print_info"]:
                        print(query_type)
                except Exception as e:
                    error = e
            if error:
                raise QueryError(error)
    if not (options["vdf"]):
        conn.close()
    elapsed_time = time.time() - start_time
    if verticapy.options["print_info"]:
        display(HTML("<div><b>Execution: </b> {}s</div>".format(round(elapsed_time, 3))))
    return result


# ---#
def load_ipython_extension(ipython):
    ipython.register_magic_function(sql, "cell")
    ipython.register_magic_function(sql, "line")
