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
#        _____ _                _        ___  ___            _
#       /  __ \ |              | |       |  \/  |           (_)
# __   _| /  \/ |__   __ _ _ __| |_ ___  | .  . | __ _  __ _ _  ___
# \ \ / / |   | '_ \ / _` | '__| __/ __| | |\/| |/ _` |/ _` | |/ __|
#  \ V /| \__/\ | | | (_| | |  | |_\__ \ | |  | | (_| | (_| | | (__
#   \_/  \____/_| |_|\__,_|_|   \__|___/ \_|  |_/\__,_|\__, |_|\___|
#                                                       __/ |
#                                                      |___/
##
#
# ---#
def vCharts(line, cell):
    from verticapy.connect import read_auto_connect
    from verticapy.hchart import hchartSQL
    from IPython.core.display import HTML, display
    import time
    import re

    conn = read_auto_connect()
    cursor = conn.cursor()
    options = {"type": "auto"}
    query = cell
    if line == "":
        option = "auto"
    else:
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
            if option.lower() == "-type":
                options["type"] = all_options_dict[option]
            else:
                print(
                    "\u26A0 Warning : option '{}' doesn't exist, it was skipped.".format(
                        option
                    )
                )
    query = query.replace("\t", " ")
    query = query.replace("\n", " ")
    query = re.sub(" +", " ", query)
    while len(query) > 0 and (query[-1] in (";", " ")):
        query = query[0:-1]
    while len(query) > 0 and (query[0] in (";", " ")):
        query = query[1:]
    start_time = time.time()
    chart = hchartSQL(query, cursor, options["type"])
    elapsed_time = time.time() - start_time
    display(HTML("<div><b>Execution: </b> {}s</div>".format(round(elapsed_time, 3))))
    conn.close()
    return chart


# ---#
def load_ipython_extension(ipython):
    ipython.register_magic_function(vCharts, "cell")
