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
# ---#
# Jupyter Modules
from IPython.core.magic import needs_local_scope
from IPython.core.display import HTML, display

# Standard Python Modules
import re, time

# Other Modules
import pandas as pd

# VerticaPy
from verticapy import (
    vDataFrame,
    tablesample,
    clean_query,
    replace_vars_in_query,
    get_magic_options,
)
from verticapy.highchart import hchartSQL

# ---#
@needs_local_scope
def hchart(line, cell, local_ns=None):

    # Initialization
    options = {"type": "auto"}
    query = cell

    if line:

        all_options_dict = get_magic_options(line)

        for option in all_options_dict:

            if option.lower() == "-type":
                options["type"] = all_options_dict[option]

            else:
                print(
                    f"\u26A0 Warning : option '{option}'"
                    " doesn't exist, it was skipped."
                )

    # Cleaning the Query
    query = clean_query(query)
    query = replace_vars_in_query(query, locals()["local_ns"])

    while len(query) > 0 and (query[-1] in (";", " ")):
        query = query[0:-1]

    while len(query) > 0 and (query[0] in (";", " ")):
        query = query[1:]

    # Drawing the graphic and displaying the info
    start_time = time.time()
    chart = hchartSQL(query, options["type"])
    elapsed_time = time.time() - start_time
    display(HTML("<div><b>Execution: </b> {0}s</div>".format(round(elapsed_time, 3))))

    return chart


# ---#
def load_ipython_extension(ipython):
    ipython.register_magic_function(vCharts, "cell")
