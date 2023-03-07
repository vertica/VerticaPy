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
import time, warnings
from typing import Optional, Literal, Union

from IPython.core.magic import needs_local_scope
from IPython.display import display, HTML

from vertica_highcharts import Highstock, Highchart

import verticapy._config.config as conf
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import clean_query, replace_vars_in_query
from verticapy._utils._sql._sys import _executeSQL
from verticapy.connection.connect import current_cursor
from verticapy.errors import ParameterError

from verticapy.core.tablesample.base import TableSample
from verticapy.core.vdataframe.base import vDataFrame

from verticapy.jupyter.extensions._utils import get_magic_options


def hchartSQL(
    query: str,
    kind: Literal[
        "area",
        "area_range",
        "area_ts",
        "bar",
        "biserial",
        "boxplot",
        "bubble",
        "candlestick",
        "cramer",
        "donut",
        "donut3d",
        "heatmap",
        "hist",
        "kendall",
        "line",
        "multi_area",
        "multi_line",
        "multi_spline",
        "negative_bar",
        "pearson",
        "pie",
        "pie3d",
        "pie_half",
        "scatter",
        "spearman",
        "spearmand",
        "spider",
        "spline",
        "stacked_bar",
        "stacked_hist",
    ] = "auto",
    width: int = 600,
    height: int = 400,
    options: dict = {},
) -> Union[Highstock, Highchart]:
    """
    Helper Function:
    Draws a custom High Chart graphic using the 
    input SQL query.
    """

    from verticapy.core.vdataframe.base import vDataFrame

    aggregate, stock = False, False
    data = _executeSQL(
        query=f"""
            SELECT 
                /*+LABEL('highchart.hchartSQL')*/ * 
            FROM ({query}) VERTICAPY_SUBTABLE LIMIT 0""",
        method="fetchall",
        print_time_sql=False,
    )
    names = [desc[0] for desc in current_cursor().description]
    vdf = vDataFrame(query)
    allnum = vdf.numcol()
    if kind == "auto":
        if len(names) == 1:
            kind = "pie"
        elif (len(names) == len(allnum)) and (len(names) < 5):
            kind = "scatter"
        elif len(names) == 2:
            if vdf[names[0]].isdate() and vdf[names[1]].isnum():
                kind = "line"
            else:
                kind = "bar"
        elif len(names) == 3:
            if vdf[names[0]].isdate() and vdf[names[1]].isnum():
                kind = "line"
            elif vdf[names[2]].isnum():
                kind = "hist"
            else:
                kind = "boxplot"
        else:
            kind = "boxplot"
    if kind in (
        "pearson",
        "kendall",
        "cramer",
        "biserial",
        "spearman",
        "spearmand",
        "boxplot",
    ):
        x, y, z, c = allnum, None, None, None
    elif kind == "scatter":
        if len(names) < 2:
            raise ValueError("Scatter Plots need at least 2 columns.")
        x, y, z, c = names[0], names[1], None, None
        if len(names) == 3 and len(allnum) == 3:
            z = names[2]
        elif len(names) == 3:
            c = names[2]
        elif len(names) > 3:
            z, c = names[2], names[3]
    elif kind == "bubble":
        if len(names) < 3:
            raise ValueError("Bubble Plots need at least 3 columns.")
        x, y, z, c = names[0], names[1], names[2], None
        if len(names) > 3:
            c = names[3]
    elif kind in (
        "area",
        "area_ts",
        "spline",
        "line",
        "area_range",
        "spider",
        "candlestick",
    ):
        if vdf[names[0]].isdate():
            stock = True
        if len(names) < 2:
            raise ValueError(f"{kind} Plots need at least 2 columns.")
        x, y, z, c = names[0], names[1:], None, None
        if kind == "candlestick":
            aggregate = True
    else:
        if len(names) == 1:
            aggregate = True
            x, y, z, c = names[0], "COUNT(*) AS cnt", None, None
        else:
            x, y, z, c = names[0], names[1], None, None
        if len(names) > 2:
            z = names[2]
    return vdf.hchart(
        x=x,
        y=y,
        z=z,
        c=c,
        aggregate=aggregate,
        kind=kind,
        width=width,
        height=height,
        options=options,
        max_cardinality=100,
        stock=stock,
    )


@save_verticapy_logs
@needs_local_scope
def hchart_magic(
    line: str, cell: str = "", local_ns: Optional[dict] = None
) -> Union[Highstock, Highchart]:
    """
    Draws  responsive charts using the High Chart  API: 
    https://api.highcharts.com/highcharts/ 
    The returned object can be customized using the API 
    parameters and the 'set_dict_options' method.

    -c / --command : SQL Command to execute.

    -f  /   --file : Input File. You can use this option 
                     if  you  want to execute the  input 
                     file.

    -k  /  --kind  : Chart  Type.  Can  be  one  of  the 
                     following.
                     area  / area_range  / area_ts / bar
                     biserial   /   boxplot   /   bubble 
                     candlestick   /   cramer  /   donut 
                     donut3d  / heatmap / hist / kendall 
                     line / negative_bar / pearson / pie
                     pie_half / pie3d / scatter / spider
                     spline / stacked_bar / stacked_hist
                     spearman

     -o / --output : Output File. You can use this option 
                     if  you want to export the result of 
                     the query to the HTML format.

    """

    # Initialization
    query = "" if (not (cell) and (line)) else cell

    # Options
    options = {}
    options_dict = get_magic_options(line)

    for option in options_dict:

        if option.lower() in (
            "-f",
            "--file",
            "-o",
            "--output",
            "-c",
            "--command",
            "-k",
            "--kind",
        ):

            if option.lower() in ("-f", "--file"):
                if "-f" in options:
                    raise ParameterError("Duplicate option '-f'.")
                options["-f"] = options_dict[option]
            elif option.lower() in ("-o", "--output"):
                if "-o" in options:
                    raise ParameterError("Duplicate option '-o'.")
                options["-o"] = options_dict[option]
            elif option.lower() in ("-c", "--command"):
                if "-c" in options:
                    raise ParameterError("Duplicate option '-c'.")
                options["-c"] = options_dict[option]
            elif option.lower() in ("-k", "--kind"):
                if "-k" in options:
                    raise ParameterError("Duplicate option '-k'.")
                options["-k"] = options_dict[option]

        elif conf.get_option("print_info"):
            warning_message = (
                f"\u26A0 Warning : The option '{option}' doesn't exist - skipping."
            )
            warnings.warn(warning_message, Warning)

    if "-f" in options and "-c" in options:
        raise ParameterError(
            "Could not find a query to run; the options"
            "'-f' and '-c' cannot be used together."
        )

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

    # Drawing the graphic
    start_time = time.time()
    chart = hchartSQL(query, options["-k"])

    # Exporting the result
    if "-o" in options:
        chart.save_file(options["-o"])

    # Displaying the time
    elapsed_time = round(time.time() - start_time, 3)
    display(HTML(f"<div><b>Execution: </b> {elapsed_time}s</div>"))

    return chart


def load_ipython_extension(ipython) -> None:
    ipython.register_magic_function(hchart_magic, "cell", "hchart")
    ipython.register_magic_function(hchart_magic, "line", "hchart")
    return None
