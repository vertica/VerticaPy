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
import time
import warnings
from typing import Optional, Literal, Union

from IPython.core.magic import needs_local_scope
from IPython.display import display, HTML

from vertica_highcharts import Highstock, Highchart

import verticapy._config.config as conf
from verticapy._typing import PlottingObject
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import clean_query, replace_vars_in_query

from verticapy.core.vdataframe.base import vDataFrame

from verticapy.jupyter.extensions._utils import get_magic_options

from verticapy.plotting._utils import PlottingUtils

CLASS_NAME_MAP = {
    "auto": None,
    "area": "MultiLinePlot",
    "area_percent": "MultiLinePlot",
    "area_stacked": "MultiLinePlot",
    "bar": "BarChart",
    "bar2D": "BarChart2D",
    "barh": "HorizontalBarChart",
    "barh2D": "HorizontalBarChart2D",
    "biserial": None,
    "boxplot": "BoxPlot",
    "bubble": "ScatterPlot",
    "candlestick": "CandleStick",
    "cramer": None,
    "fstacked_barh": "HorizontalBarChart2D",
    "drilldown_bar": "DrillDownBarChart",
    "drilldown_barh": "DrillDownHorizontalBarChart",
    "donut": "PieChart",
    "heatmap": "HeatMap",
    "hist": "Histogram",
    "kendall": None,
    "line": "LinePlot",
    "multi_area": "MultiLinePlot",
    "multi_line": "MultiLinePlot",
    "multi_spline": "MultiLinePlot",
    "multi_step": "MultiLinePlot",
    "negative_bar": "HorizontalBarChart2D",
    "nested_pie": "NestedPieChart",
    "outliers": "OutliersPlot",
    "pearson": None,
    "pie": "PieChart",
    "rose": "PieChart",
    "scatter": "ScatterPlot",
    "scatter_matrix": "ScatterMatrix",
    "spearman": None,
    "spearmand": None,
    "spider": "SpiderChart",
    "spline": "LinePlot",
    "stacked_bar": "BarChart2D",
    "stacked_barh": "HorizontalBarChart2D",
    "step": "LinePlot",
}


def get_kind_option(kind: Literal[tuple(CLASS_NAME_MAP)] = "auto") -> dict:
    kind_option = {}
    other_params = {}
    if kind in [
        "area",
        "area_percent",
        "area_stacked",
        "bubble",
        "donut",
        "line",
        "rose",
        "spline",
        "step",
    ]:
        kind_option["kind"] = kind
        if kind == "bubble":
            other_params["kind"] = "bubble"
    elif kind in ["area"]:
        kind_option["kind"] = "area_stacked"
    elif kind in ["multi_area", "multi_line", "multi_spline", "multi_step"]:
        kind_option["kind"] = kind[6:]
    elif kind == "negative_bar":
        kind_option["kind"] = "density"
        kind_option["method"] = "density"
    elif kind in ["fstacked_bar", "fstacked_barh"]:
        kind_option["kind"] = "fully_stacked"
    elif kind in ["stacked_bar", "stacked_barh"]:
        kind_option["kind"] = "stacked"
    elif kind in ["bar2D", "barh2D", "pie"]:
        kind_option["kind"] = "regular"
    return {"misc_layout": kind_option, **other_params}


def chartSQL(
    query: str,
    kind: Literal[tuple(CLASS_NAME_MAP)] = "auto",
) -> PlottingObject:
    """
    Helper Function:
    Draws a custom High Chart graphic using the
    input SQL query.
    """
    if kind in [
        "auto",
        "bar",
        "barh",
        "biserial",
        "cramer",
        "donut",
        "line",
        "kendall",
        "pearson",
        "pie",
        "rose",
        "spearman",
        "spearmand",
        "spline",
        "step",
    ]:
        vdf = vDataFrame(input_relation=query)
        cols = vdf.get_columns()
        if kind == "auto":
            if len(cols) == 1:
                if vdf[cols[0]].isnum():
                    return vdf[cols[0]].boxplot()
                else:
                    return vdf[cols[0]].pie()
            elif len(cols) > 1 and vdf[cols[0]].isdate():
                kind = "line"
            elif len(cols) == 2 and vdf[cols[0]].isnum() and vdf[cols[1]].isnum():
                kind = "hist"
            elif len(cols) == 2 and vdf[cols[1]].isnum():
                kind = "barh"
            elif (
                len(cols) == 3
                and not vdf[cols[0]].isnum()
                and not vdf[cols[1]].isnum()
                and vdf[cols[2]].isnum()
            ):
                kind = "barh2D"
            elif len(cols) > 4:
                return vdf.boxplot()
            else:
                kind = "scatter"
        elif kind in [
            "biserial",
            "cramer",
            "kendall",
            "pearson",
            "spearman",
            "spearmand",
        ]:
            return vdf.corr(method=kind)
        elif kind in ["line", "spline", "step"]:
            if len(cols) > 3 or ((len(cols) == 3) and (vdf[cols[2]].isnum())):
                kind = "multi_" + kind
        elif kind in ["bar", "barh"]:
            if len(cols) > 2:
                kind = kind + "2D"
        elif kind in ["donut", "pie", "rose"]:
            if len(cols) > 2:
                kind = "nested_pie"
    class_name = CLASS_NAME_MAP[kind]
    vpy_plt, kwargs = PlottingUtils().get_plotting_lib(class_name=class_name)
    graph = getattr(vpy_plt, class_name)
    return graph(query=query, **get_kind_option(kind)).draw(**kwargs)


@save_verticapy_logs
@needs_local_scope
def chart_magic(
    line: str, cell: Optional[str] = None, local_ns: Optional[dict] = None
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

    -k  /  --kind  : Chart Type, one  of  the following:
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
    query = "" if (not cell and (line)) else cell

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
            elif option.lower() in ("-k", "--kind"):
                if "-k" in options:
                    raise ValueError("Duplicate option '-k'.")
                options["-k"] = options_dict[option]

        elif conf.get_option("print_info"):
            warning_message = (
                f"\u26A0 Warning : The option '{option}' doesn't exist - skipping."
            )
            warnings.warn(warning_message, Warning)

    if "-f" in options and "-c" in options:
        raise ValueError(
            "Could not find a query to run; the options"
            "'-f' and '-c' cannot be used together."
        )

    if cell and ("-f" in options or "-c" in options):
        raise ValueError("Cell must be empty when using options '-f' or '-c'.")

    if "-f" in options:
        with open(options["-f"], "r", encoding="utf-8") as f:
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
    chart = chartSQL(query, options["-k"])

    # Exporting the result
    if "-o" in options:
        chart.save_file(options["-o"])

    # Displaying the time
    elapsed_time = round(time.time() - start_time, 3)
    display(HTML(f"<div><b>Execution: </b> {elapsed_time}s</div>"))

    return chart


def load_ipython_extension(ipython) -> None:
    ipython.register_magic_function(chart_magic, "cell", "chart")
    ipython.register_magic_function(chart_magic, "line", "chart")
    ipython.register_magic_function(chart_magic, "cell", "plot")
    ipython.register_magic_function(chart_magic, "line", "plot")
