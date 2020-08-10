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
    from verticapy.connections.connect import read_auto_connect
    from verticapy.hchart import hchartSQL

    conn = read_auto_connect()
    cursor = conn.cursor()
    if line == "":
        line = "auto"
    chart = hchartSQL(cell, cursor, line)
    conn.close()
    return chart


# ---#
def load_ipython_extension(ipython):
    ipython.register_magic_function(vCharts, "cell")
