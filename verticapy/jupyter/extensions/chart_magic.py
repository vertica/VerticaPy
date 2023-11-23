"""
Copyright  (c)  2018-2024 Open Text  or  one  of its
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
    Draws responsive charts using the Matplotlib,
    Plotly, or Highcharts library.

    Different cutomization parameters are available for Plotly, Highcharts,
    and Matplotlib.
    For a comprehensive list of customization features, please
    consult the documentation for the respective plotting libraries:
    `plotly <https://plotly.com/python-api-reference/>`_,
    `matplotlib <https://matplotlib.org/stable/api/matplotlib_configuration_api.html>`_
    and `highcharts <https://api.highcharts.com/highcharts/>`_.

    Parameters
    ----------
    -c / --command : str, optional
        SQL Command to execute.
    -f / --file : str, optional
        Input File. You can use this option
        if  you  want to execute the  input
        file.
    -k / --kind  : str, optional
        Chart Type, one  of  the following:

        **area**         :
                            Area Chart

        **area_range**   :
                            Area Range Chart

        **area_ts**      :
                            Area Chart with Time Series Design

        **bar**          :
                            Bar Chart

        **biserial**     :
                            Biserial Point Matrix (Correlation
                            between binary variables and numerical)

        **boxplot**      :
                            Box Plot

        **bubble**       :
                            Bubble Plot

        **candlestick**  :
                            Candlestick and Volumes (Time Series
                            Special Plot)

        **cramer**       :
                            Cramer's V Matrix (Correlation between
                            categories)

        **donut**        :
                            Donut Chart

        **donut3d**      :
                            3D Donut Chart

        **heatmap**      :
                            Heatmap

        **hist**         :
                            Histogram

        **kendall**      :
                            Kendall Correlation Matrix

        .. Warning::
            This method  uses a CROSS JOIN  during  computation and is
            therefore computationally expensive at  O(n * n),  where n
            is the  total  count of  the :py:class:`vDataFrame`.

        **line**         :
                            Line Plot

        **negative_bar** :
                            Multi-Bar Chart for binary classes

        **pearson**      :
                            Pearson Correlation Matrix

        **pie**          :
                            Pie Chart

        **pie_half**     :
                            Half Pie Chart

        **pie3d**        :
                            3D Pie Chart

        **scatter**      :
                            Scatter Plot

        **spider**       :
                            Spider Chart

        **spline**       :
                            Spline Plot

        **stacked_bar**  :
                            Stacker Bar Chart

        **stacked_hist** :
                            Stacked Histogram

        **spearman**     :
                            Spearman Correlation Matrix

    -o / --output : str, optional
        Output File. You can use this option
        if  you want to export the result of
        the query to the HTML format.

    Returns
    -------
    Chart Object

    Examples
    --------
    The following examples demonstrate:

    * Setting up the environment
    * Drawing graphics
    * Exporting to HTML
    * Using variables
    * Using SQL files

    .. hint:: To see more examples, please refer to the ref:chart_gallery.guide.

    Setting up the environment
    ==========================

    If you don't already have one, create a new connection:

    .. code-block:: python

        import verticapy as vp

        # Save a new connection
        vp.new_connection(
            {
                "host": "10.211.55.14",
                "port": "5433",
                "database": "testdb",
                "password": "XxX",
                "user": "dbadmin",
            },
            name = "VerticaDSN",
        )

    Otherwise, to use an existing connection:

    .. code-block:: python

        vp.connect("VerticaDSN")

    Load the chart extension:

    .. ipython:: python
        :suppress:

        import verticapy as vp

    .. ipython:: python
        :suppress:

        %load_ext verticapy.chart

    Run the following to load some sample datasets. Once loaded, these datasets are
    stored in the 'public' schema. You can change the target schema with the
    'schema' parameter:

    .. ipython:: python

        from verticapy.datasets import load_titanic, load_amazon, load_iris

        titanic = load_titanic()
        amazon = load_amazon()
        iris = load_iris()

    Use the :py:func:`set_option` function to set your desired plotting library:

    .. ipython:: python

        vp.set_option("plotting_lib","plotly")

    Drawing graphics
    ================

    The following examples draw various responsive charts from SQL queries.

    Pie Chart
    ^^^^^^^^^

    .. code-block:: python

        %chart -k pie -c "SELECT pclass, AVG(age) AS av_avg FROM titanic GROUP BY 1;"

    .. ipython:: python
        :suppress:

        %%chart -k pie
        SELECT pclass, AVG(age) AS av_avg FROM titanic GROUP BY 1;

    .. ipython:: python
        :suppress:

        pie_chart = _
        pie_chart.write_html("SPHINX_DIRECTORY/figures/jupyter_extensions_chart_magic_chart_magic.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/jupyter_extensions_chart_magic_chart_magic.html

    Line Plot
    ^^^^^^^^^

    .. code-block:: python

        %%chart -k line
        SELECT
            date,
            AVG(number) AS number
        FROM amazon
        GROUP BY 1;

    .. ipython:: python
        :suppress:

        %%chart -k line
        SELECT
            date,
            AVG(number) AS number
        FROM amazon
        GROUP BY 1;

    .. ipython:: python
        :suppress:

        line_chart = _
        line_chart.write_html("SPHINX_DIRECTORY/figures/jupyter_extensions_chart_magic_chart_magic_2.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/jupyter_extensions_chart_magic_chart_magic_2.html

    Correlation Matrix
    ^^^^^^^^^^^^^^^^^^

    .. code-block:: python

        %%chart -k pearson
        SELECT
            *
        FROM titanic;

    .. ipython:: python
        :suppress:

        %%chart -k pearson
        SELECT * FROM titanic;

    .. ipython:: python
        :suppress:

        heatmap = _
        heatmap.write_html("SPHINX_DIRECTORY/figures/jupyter_extensions_chart_magic_chart_magic_3.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/jupyter_extensions_chart_magic_chart_magic_3.html

    Bar Chart
    ^^^^^^^^^

    .. code-block:: python

        %%chart -k bar
        SELECT
            pclass,
            SUM(survived)
        FROM titanic GROUP BY 1;

    .. ipython:: python
        :suppress:

        %%chart -k bar
        SELECT
            pclass,
            SUM(survived)
        FROM titanic GROUP BY 1;

    .. ipython:: python
        :suppress:

        bar = _
        bar.write_html("SPHINX_DIRECTORY/figures/jupyter_extensions_chart_magic_chart_magic_4.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/jupyter_extensions_chart_magic_chart_magic_4.html

    Scatter Plot
    ^^^^^^^^^^^^

    .. code-block:: python

        %%chart -k scatter
        SELECT
            PetalLengthCm,
            PetalWidthCm,
            Species
        FROM iris;

    .. ipython:: python
        :suppress:

        %%chart -k scatter
        SELECT
            PetalLengthCm,
            PetalWidthCm,
            Species
        FROM iris;

    .. ipython:: python
        :suppress:

        scatter = _
        scatter.write_html("SPHINX_DIRECTORY/figures/jupyter_extensions_chart_magic_chart_magic_5.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/jupyter_extensions_chart_magic_chart_magic_5.html

    Boxplot
    ^^^^^^^

    .. code-block:: python

        %%chart -k boxplot
        SELECT * FROM titanic;

    .. ipython:: python
        :suppress:

        %%chart -k boxplot
        SELECT * FROM titanic;

    .. ipython:: python
        :suppress:

        boxplot = _
        boxplot.write_html("SPHINX_DIRECTORY/figures/jupyter_extensions_chart_magic_chart_magic_6.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/jupyter_extensions_chart_magic_chart_magic_6.html

    Exporting to HTML
    =================
    Export a chart to HTML:

    .. code-block:: python

        %%chart -k scatter -o "my_graphic"
        SELECT * FROM titanic;

    .. ipython:: python
        :suppress:

        %%chart -k spearman
        SELECT * FROM titanic;

    .. ipython:: python
        :suppress:

        export = _
        export.write_html("SPHINX_DIRECTORY/figures/jupyter_extensions_chart_magic_chart_magic_7.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/jupyter_extensions_chart_magic_chart_magic_7.html

    The following lines open the HTML file:

    .. note:: The HTML graphic can be embedded in an external environment, such as a website.

    .. code-block:: python

        file = open("my_graphic.html", "r")
        file.read()
        file.close()

    Using Variables
    ===============
    You can use variables in charts with the ':' operator:

    .. ipython:: python

        import verticapy.sql.functions as vpf

        class_fare = titanic.groupby(
            "pclass",
            [vpf.avg(titanic["fare"])._as("avg_fare")]
        )

    .. ipython:: python
        :suppress:

        html_file = open("SPHINX_DIRECTORY/figures/jupyter_extensions_chart_magic_chart_magic_8.html", "w")
        html_file.write(class_fare._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/jupyter_extensions_chart_magic_chart_magic_8.html

    You can then use the variable in the query:

    .. note:: In this example, we use a vDataFrame, but it's also possible to use a pandas.DataFrame, a numpy.array, and many other in-memory objects.

    .. code-block:: python

        %%chart -k bar
        SELECT * FROM :class_fare;

    .. ipython:: python
        :suppress:

        %%chart -k bar
        SELECT * FROM :class_fare;

    .. ipython:: python
        :suppress:

        chart = _
        chart.write_html("SPHINX_DIRECTORY/figures/jupyter_extensions_chart_magic_chart_magic_9.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/jupyter_extensions_chart_magic_chart_magic_9.html

    Using SQL files
    ===============
    Create charts from a SQL file:

    .. ipython:: python

        file = open("query.sql", "w+")
        file.write("SELECT PetalLengthCm, PetalWidthCm, Species FROM iris;")
        file.close()

    Using the ``-f`` option, we can easily read the above SQL file:

    .. code-block:: python

        %chart -f query.sql -k scatter

    .. ipython:: python
        :suppress:

        %chart -f query.sql -k scatter

    .. ipython:: python
        :suppress:

        sql_file = _
        sql_file.write_html("SPHINX_DIRECTORY/figures/jupyter_extensions_chart_magic_chart_magic_10.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/jupyter_extensions_chart_magic_chart_magic_10.html
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
