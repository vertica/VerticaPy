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
# ---#
# Jupyter Modules
from IPython.core.magic import needs_local_scope
from IPython.core.display import HTML, display

# Standard Python Modules
import re, time, warnings

# Other Modules
import pandas as pd

# VerticaPy
import verticapy
from verticapy.errors import ParameterError
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
def hchart(line, cell="", local_ns=None):

    # Initialization
    query = "" if (not (cell) and (line)) else cell

    # Options
    options = {}
    all_options_dict = get_magic_options(line)

    for option in all_options_dict:

        if option.lower() in ("-f", "--file", "-o", "--output", "-c", "--command", "-k", "--kind"):

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
            elif option.lower() in ("-k", "--kind"):
                if "-k" in options:
                    raise ParameterError("Duplicate option '-k'.")
                options["-k"] = all_options_dict[option]

        elif verticapy.options["print_info"]:
            warning_message = (
                f"\u26A0 Warning : The option '{option}' doesn't "
                "exist - skipping."
            )
            warnings.warn(warning_message, Warning)

    if "-f" in options and "-c" in options:
        raise ParameterError("Could not find a query to run; the options"
                             "'-f' and '-c' cannot be used together.")

    if cell and ("-f" in options or "-c" in options):
        raise ParameterError("Cell must be empty when using options '-f' or '-c'.")

    if "-f" in options:
        f = open(options["-f"], "r")
        query = f.read()
        f.close()

    elif "-c" in options:
        query = options["-c"]

    if "-k" not in options:
        options["-k"] = "auto"

    # Cleaning the Query
    query = clean_query(query)
    query = replace_vars_in_query(query, locals()["local_ns"])

    # Drawing the graphic and displaying the info
    start_time = time.time()
    chart = hchartSQL(query, options["-k"])
    elapsed_time = time.time() - start_time
    display(HTML("<div><b>Execution: </b> {0}s</div>".format(round(elapsed_time, 3))))

    # export graphic
    if "-o" in options:
        chart.save_file(options["-o"])

    return chart


# ---#
def load_ipython_extension(ipython):
    ipython.register_magic_function(hchart, "cell")
    ipython.register_magic_function(hchart, "line")
