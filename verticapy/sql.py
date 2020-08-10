# (c) Copyright [2018-2020] Micro Focus or one of its affiliates.
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
    from verticapy.connections.connect import read_auto_connect
    from verticapy.utilities import readSQL
    from IPython.core.display import HTML, display
    import time
    import re
    import vertica_python

    version = vertica_python.__version__.split(".")
    version = [int(elem) for elem in version]
    conn = read_auto_connect()
    cursor = conn.cursor()
    queries = line if (not (cell) and (line)) else cell
    options = {"limit": 100}
    queries = queries.replace("\t", " ")
    queries = queries.replace("\n", " ")
    queries = re.sub(" +", " ", queries)
    if (cell) and (line):
        all_options = line.split(" ")
        options["limit"] = int(all_options[0])
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
    for i in range(m - 2):
        query = queries[all_split[i] : all_split[i + 1]]
        if query[-1] in (";", " "):
            query = query[0:-1]
        if query[0] in (";", " "):
            query = query[1:]
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
        else:
            cursor.execute(query)
        print(query_type)
    query = queries[all_split[m - 2] : all_split[m - 1]]
    if query[-1] in (";", " "):
        query = query[0:-1]
    if query[0] in (";", " "):
        query = query[1:]
    query_type = (
        query.split(" ")[0].upper()
        if (query.split(" ")[0])
        else query.split(" ")[1].upper()
    )
    try:
        result = readSQL(query, cursor=cursor, limit=options["limit"])
    except:
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
        else:
            cursor.execute(query)
        print(query_type)
        result = None
    conn.close()
    elapsed_time = time.time() - start_time
    display(HTML("<div><b>Execution: </b> {}s</div>".format(round(elapsed_time, 3))))
    return result


# ---#
def load_ipython_extension(ipython):
    ipython.register_magic_function(sql, "cell")
    ipython.register_magic_function(sql, "line")
