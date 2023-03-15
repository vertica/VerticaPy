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
import datetime, math
from typing import Literal, Optional, Union
from collections.abc import Iterable

from matplotlib.axes import Axes

from verticapy._config.colors import get_colors
import verticapy._config.config as conf
from verticapy._typing import (
    ArrayLike,
    PythonNumber,
    PythonScalar,
    SQLColumns,
    SQLExpression,
)
from verticapy._utils._gen import gen_tmp_name
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._sys import _executeSQL

from verticapy.core.tablesample.base import TableSample

from verticapy.plotting.base import PlottingBase
from verticapy.plotting._highcharts.base import hchart_from_vdf
import verticapy.plotting._matplotlib as vpy_matplotlib_plt
import verticapy.plotting._plotly as vpy_plotly_plt


class vDFPlot:
    @staticmethod
    def _get_plotting_lib(
        matplotlib_kwargs: dict = {}, plotly_kwargs: dict = {}, style_kwargs: dict = {}
    ) -> tuple[Literal[vpy_plotly_plt, vpy_matplotlib_plt], dict]:
        if conf.get_option("plotting_lib") == "plotly":
            vpy_plt = vpy_plotly_plt
            kwargs = {**plotly_kwargs, **style_kwargs}
        else:
            vpy_plt = vpy_matplotlib_plt
            kwargs = {**matplotlib_kwargs, **style_kwargs}
        return vpy_plt, kwargs

    @save_verticapy_logs
    def animated(
        self,
        ts: str,
        columns: Union[list] = [],
        by: str = "",
        start_date: PythonScalar = "",
        end_date: PythonScalar = "",
        kind: Literal["auto", "bar", "bubble", "ts", "pie"] = "auto",
        limit_over: int = 6,
        limit: int = 1000000,
        limit_labels: int = 6,
        ts_steps: dict = {"window": 100, "step": 5},
        bubble_img: dict = {"bbox": [], "img": ""},
        fixed_xy_lim: bool = False,
        date_in_title: bool = False,
        date_f=None,
        date_style_dict: dict = {},
        interval: int = 300,
        repeat: bool = True,
        return_html: bool = True,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ):
        """
    Draws the animated chart.

    Parameters
    ----------
    ts: str
        TS (Time Series) vDataColumn to use to order the data. The vDataColumn type must be
        date like (date, datetime, timestamp...) or numerical.
    columns: SQLColumns, optional
        List of the vDataColumns names.
    by: str, optional
        Categorical vDataColumn used in the partition.
    start_date: str / date, optional
        Input Start Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is lesser than November 1993 the 3rd.
    end_date: str / date, optional
        Input End Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is greater than November 1993 the 3rd.
    kind: str, optional
        Animation Type.
            auto   : Pick up automatically the type.
            bar    : Animated Bar Race.
            bubble : Animated Bubble Plot.
            pie    : Animated Pie Chart.
            ts     : Animated Time Series.
    limit_over: int, optional
        Limited number of elements to consider for each category.
    limit: int, optional
        Maximum number of data points to use.
    limit_labels: int, optional
        [Only used when kind = 'bubble']
        Maximum number of text labels to draw.
    ts_steps: dict, optional
        [Only used when kind = 'ts']
        dictionary including 2 keys.
            step   : number of elements used to update the time series.
            window : size of the window used to draw the time series.
    bubble_img: dict, optional
        [Only used when kind = 'bubble']
        dictionary including 2 keys.
            img  : Path to the image to display as background.
            bbox : List of 4 elements to delimit the boundaries of the final Plot.
                   It must be similar the following list: [xmin, xmax, ymin, ymax]
    fixed_xy_lim: bool, optional
        If set to True, the xlim and ylim will be fixed.
    date_in_title: bool, optional
        If set to True, the ts vDataColumn will be displayed in the title section.
    date_f: function, optional
        Function used to display the ts vDataColumn.
    date_style_dict: dict, optional
        Style Dictionary used to display the ts vDataColumn when date_in_title = False.
    interval: int, optional
        Number of ms between each update.
    repeat: bool, optional
        If set to True, the animation will be repeated.
    return_html: bool, optional
        If set to True and if using a Jupyter notebook, the HTML of the animation will be 
        generated.
    ax: Axes, optional
        [Only for MATPLOTLIB]
        The axes to plot on.
    **style_kwargs
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    animation
        Matplotlib animation object
        """
        if isinstance(columns, str):
            columns = [columns]
        if kind == "auto":
            if len(columns) > 3 or len(columns) <= 1:
                kind = "ts"
            elif len(columns) == 2:
                kind = "bar"
            else:
                kind = "bubble"
        assert kind == "ts" or columns, ParameterError(
            f"Parameter 'columns' can not be empty when using kind = '{kind}'."
        )
        assert (
            2 <= len(columns) <= 4
            and self[columns[0]].isnum()
            and self[columns[1]].isnum()
        ) or kind != "bubble", ParameterError(
            f"Parameter 'columns' must include at least 2 numerical vDataColumns and maximum 4 vDataColumns when using kind = '{kind}'."
        )
        columns, ts, by = self._format_colnames(columns, ts, by)
        if kind == "bubble":
            if len(columns) == 3 and not (self[columns[2]].isnum()):
                label_name = columns[2]
                columns = columns[0:2]
            elif len(columns) >= 4:
                if not (self[columns[3]].isnum()):
                    label_name = columns[3]
                    columns = columns[0:3]
                else:
                    label_name = columns[2]
                    columns = columns[0:2] + [columns[3]]
            else:
                label_name = ""
            if "img" not in bubble_img:
                bubble_img["img"] = ""
            if "bbox" not in bubble_img:
                bubble_img["bbox"] = []
            return vpy_matplotlib_plt.AnimatedBubblePlot().draw(
                self,
                order_by=ts,
                columns=columns,
                label_name=label_name,
                by=by,
                order_by_start=start_date,
                order_by_end=end_date,
                limit_over=limit_over,
                limit=limit,
                lim_labels=limit_labels,
                fixed_xy_lim=fixed_xy_lim,
                date_in_title=date_in_title,
                date_f=date_f,
                date_style_dict=date_style_dict,
                interval=interval,
                repeat=repeat,
                return_html=return_html,
                img=bubble_img["img"],
                bbox=bubble_img["bbox"],
                ax=ax,
                **style_kwargs,
            )
        elif kind in ("bar", "pie"):
            return vpy_matplotlib_plt.AnimatedBarChart().draw(
                self,
                order_by=ts,
                columns=columns,
                by=by,
                order_by_start=start_date,
                order_by_end=end_date,
                limit_over=limit_over,
                limit=limit,
                fixed_xy_lim=fixed_xy_lim,
                date_in_title=date_in_title,
                date_f=date_f,
                date_style_dict=date_style_dict,
                interval=interval,
                repeat=repeat,
                return_html=return_html,
                pie=(kind == "pie"),
                ax=ax,
                **style_kwargs,
            )
        else:
            if by:
                assert len(columns) == 1, ParameterError(
                    "Parameter 'columns' can not be empty when using kind = 'ts' and when parameter 'by' is not empty."
                )
                vdf = self.pivot(index=ts, columns=by, values=columns[0])
            else:
                vdf = self
            columns = vdf.numcol()[0:limit_over]
            if "step" not in ts_steps:
                ts_steps["step"] = 5
            if "window" not in ts_steps:
                ts_steps["window"] = 100
            return vpy_matplotlib_plt.AnimatedLinePlot().draw(
                vdf,
                order_by=ts,
                columns=columns,
                order_by_start=start_date,
                order_by_end=end_date,
                limit=limit,
                fixed_xy_lim=fixed_xy_lim,
                window_size=ts_steps["window"],
                step=ts_steps["step"],
                interval=interval,
                repeat=repeat,
                return_html=return_html,
                ax=ax,
                **style_kwargs,
            )

    @save_verticapy_logs
    def bar(
        self,
        columns: SQLColumns,
        method: str = "density",
        of: Optional[str] = None,
        max_cardinality: tuple[int, int] = (6, 6),
        h: tuple[PythonNumber, PythonNumber] = (None, None),
        bar_type: Literal["auto", "stacked"] = "auto",
        ax: Optional[Axes] = None,
        **style_kwargs,
    ):
        """
    Draws the histogram of the input vDataColumns based on an aggregation.

    Parameters
    ----------
    columns: SQLColumns
        List of the vDataColumns names. The list must have one or two elements.
    method: str, optional
        The method to use to aggregate the data.
            count   : Number of elements.
            density : Percentage of the distribution.
            mean    : Average of the vDataColumn 'of'.
            min     : Minimum of the vDataColumn 'of'.
            max     : Maximum of the vDataColumn 'of'.
            sum     : Sum of the vDataColumn 'of'.
            q%      : q Quantile of the vDataColumn 'of' (ex: 50% to get the median).
        It can also be a cutomized aggregation (ex: AVG(column1) + 5).
    of: str, optional
        The vDataColumn to use to compute the aggregation.
    max_cardinality: tuple, optional
        Maximum number of distinct elements for vDataColumns 1 and 2 to be used as 
        categorical (No h will be picked or computed)
    h: tuple, optional
        Interval width of the vDataColumns 1 and 2 bars. It is only valid if the 
        vDataColumns are numerical. Optimized h will be computed if the parameter 
        is empty or invalid.
    bar_type: str, optional
        The BarChart Type.
            auto    : Regular BarChart based on 1 or 2 vDataColumns.
            stacked : Stacked BarChart based on 2 vDataColumns.
    ax: Axes, optional
        [Only for MATPLOTLIB]
        The axes to plot on.
    **style_kwargs
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Axes

    See Also
    --------
    vDataFrame.bar         : Draws the bar chart of the input vDataColumns based on an aggregation.
    vDataFrame.boxplot     : Draws the Box Plot of the input vDataColumns.
    vDataFrame.pivot_table : Draws the pivot table of vDataColumns based on an aggregation.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns, of = self._format_colnames(columns, of, expected_nb_of_cols=[1, 2])
        if len(columns) == 1:
            return self[columns[0]].bar(
                method=method,
                of=of,
                max_cardinality=max_cardinality[0],
                h=h[0],
                **style_kwargs,
            )
        else:
            vpy_plt, kwargs = self._get_plotting_lib(
                matplotlib_kwargs={
                    "ax": ax,
                    "stacked": (bar_type.lower() == "stacked"),
                },
                style_kwargs=style_kwargs,
            )
            return vpy_plt.BarChart2D(
                vdf=self,
                columns=columns,
                method=method,
                of=of,
                h=h,
                max_cardinality=max_cardinality,
            ).draw(**kwargs)

    @save_verticapy_logs
    def barh(
        self,
        columns: SQLColumns,
        method: str = "density",
        of: Optional[str] = None,
        max_cardinality: tuple[int, int] = (6, 6),
        h: tuple[PythonNumber, PythonNumber] = (None, None),
        bar_type: Literal[
            "auto",
            "fully_stacked",
            "stacked",
            "fully",
            "fully stacked",
            "pyramid",
            "density",
        ] = "auto",
        ax: Optional[Axes] = None,
        **style_kwargs,
    ):
        """
    Draws the bar chart of the input vDataColumns based on an aggregation.

    Parameters
    ----------
    columns: SQLColumns
        List of the vDataColumns names. The list must have one or two elements.
    method: str, optional
        The method to use to aggregate the data.
            count   : Number of elements.
            density : Percentage of the distribution.
            mean    : Average of the vDataColumn 'of'.
            min     : Minimum of the vDataColumn 'of'.
            max     : Maximum of the vDataColumn 'of'.
            sum     : Sum of the vDataColumn 'of'.
            q%      : q Quantile of the vDataColumn 'of' (ex: 50% to get the median).
        It can also be a cutomized aggregation (ex: AVG(column1) + 5).
    of: str, optional
         The vDataColumn to use to compute the aggregation.
    max_cardinality: tuple, optional
        Maximum number of distinct elements for vDataColumns 1 and 2 to be used as 
        categorical (No h will be picked or computed)
    h: tuple, optional
        Interval width of the vDataColumns 1 and 2 bars. It is only valid if the 
        vDataColumns are numerical. Optimized h will be computed if the parameter 
        is empty or invalid.
    bar_type: str, optional
        The BarChart Type.
            auto          : Regular Bar Chart based on 1 or 2 vDataColumns.
            pyramid       : Pyramid Density Bar Chart. Only works if one of
                            the two vDataColumns is binary and the 'method' is 
                            set to 'density'.
            stacked       : Stacked Bar Chart based on 2 vDataColumns.
            fully_stacked : Fully Stacked Bar Chart based on 2 vDataColumns.
    ax: Axes, optional
        [Only for MATPLOTLIB]
        The axes to plot on.
    **style_kwargs
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Axes

     See Also
     --------
     vDataFrame.boxplot     : Draws the Box Plot of the input vDataColumns.
     vDataFrame.bar         : Draws the Bar Chart of the input vDataColumns based 
                              on an aggregation.
     vDataFrame.pivot_table : Draws the pivot table of vDataColumns based on an 
                              aggregation.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns, of = self._format_colnames(columns, of, expected_nb_of_cols=[1, 2])
        if len(columns) == 1:
            return self[columns[0]].barh(
                method=method,
                of=of,
                max_cardinality=max_cardinality[0],
                h=h[0],
                ax=ax,
                **style_kwargs,
            )
        else:
            if bar_type in ("fully", "fully stacked"):
                bar_type = "fully_stacked"
            elif bar_type == "pyramid":
                bar_type = "density"
            vpy_plt, kwargs = self._get_plotting_lib(
                matplotlib_kwargs={"ax": ax, "bar_type": bar_type,},
                style_kwargs=style_kwargs,
            )
            return vpy_plt.HorizontalBarChart2D(
                vdf=self,
                columns=columns,
                method=method,
                of=of,
                max_cardinality=max_cardinality,
                h=h,
            ).draw(**kwargs)

    @save_verticapy_logs
    def hist(
        self,
        columns: SQLColumns,
        method: str = "density",
        of: Optional[str] = None,
        h: PythonNumber = None,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ):
        """
    Draws the histogram of the input vDataColumns based on an aggregation.

    Parameters
    ----------
    columns: SQLColumns
        List of the vDataColumns names. The list must have less than 5 elements.
    method: str, optional
        The method to use to aggregate the data.
            count   : Number of elements.
            density : Percentage of the distribution.
            mean    : Average of the vDataColumn 'of'.
            min     : Minimum of the vDataColumn 'of'.
            max     : Maximum of the vDataColumn 'of'.
            sum     : Sum of the vDataColumn 'of'.
            q%      : q Quantile of the vDataColumn 'of' (ex: 50% to get the median).
        It can also be a cutomized aggregation (ex: AVG(column1) + 5).
    of: str, optional
        The vDataColumn to use to compute the aggregation.
    h: tuple, optional
        Interval width of the input vDataColumns. Optimized h will be computed 
        if the parameter is empty or invalid.
    ax: Axes, optional
        [Only for MATPLOTLIB]
        The axes to plot on.
    **style_kwargs
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Axes

    See Also
    --------
    vDataFrame.bar         : Draws the bar chart of the input vDataColumns based on an aggregation.
    vDataFrame.boxplot     : Draws the Box Plot of the input vDataColumns.
    vDataFrame.pivot_table : Draws the pivot table of vDataColumns based on an aggregation.
        """
        vpy_plt, kwargs = self._get_plotting_lib(
            matplotlib_kwargs={"ax": ax}, style_kwargs=style_kwargs,
        )
        return vpy_plt.Histogram(
            vdf=self, columns=columns, method=method, of=of, h=h,
        ).draw(**kwargs)

    @save_verticapy_logs
    def boxplot(
        self,
        columns: SQLColumns = [],
        q: tuple = (0.25, 0.75),
        ax: Optional[Axes] = None,
        **style_kwargs,
    ):
        """
    Draws the Box Plot of the input vDataColumns. 

    Parameters
    ----------
    columns: SQLColumns, optional
        List of the vDataColumns names. If empty, all numerical vDataColumns will 
        be used.
    q: tuple, optional
        Tuple including the 2 quantiles used to draw the BoxPlot.
    ax: Axes, optional
        [Only for MATPLOTLIB]
        The axes to plot on.
    **style_kwargs
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Axes

    See Also
    --------
    vDataFrame.bar         : Draws the bar chart of the input vDataColumns based 
                             on an aggregation.
    vDataFrame.boxplot     : Draws the vDataColumn box plot.
    vDataFrame.bar         : Draws the Bar Chart of the input vDataColumns based 
                             on an aggregation.
    vDataFrame.pivot_table : Draws the pivot table of vDataColumns based on an 
                             aggregation.
        """
        vpy_plt, kwargs = self._get_plotting_lib(
            matplotlib_kwargs={"ax": ax}, style_kwargs=style_kwargs,
        )
        return vpy_plt.BoxPlot(vdf=self, columns=columns, q=q,).draw(**kwargs)

    @save_verticapy_logs
    def contour(
        self,
        columns: SQLColumns,
        func,
        nbins: int = 100,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ):
        """
    Draws the contour plot of the input function two input vDataColumns.

    Parameters
    ----------
    columns: SQLColumns
        List of the vDataColumns names. The list must have two elements.
    func: function / str
        Function used to compute the contour score. It can also be a SQL
        expression.
    nbins: int, optional
        Number of bins used to discretize the two input numerical vDataColumns.
    ax: Axes, optional
        [Only for MATPLOTLIB]
        The axes to plot on.
    **style_kwargs
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Axes

     See Also
     --------
     vDataFrame.boxplot     : Draws the Box Plot of the input vDataColumns.
     vDataFrame.bar         : Draws the Bar Chart of the input vDataColumns based on an aggregation.
     vDataFrame.pivot_table : Draws the pivot table of vDataColumns based on an aggregation.
        """
        columns = self._format_colnames(columns, expected_nb_of_cols=2)
        return vpy_matplotlib_plt.ContourPlot().draw(
            self, columns, func, nbins, ax=ax, **style_kwargs,
        )

    @save_verticapy_logs
    def density(
        self,
        columns: SQLColumns = [],
        bandwidth: float = 1.0,
        kernel: Literal["gaussian", "logistic", "sigmoid", "silverman"] = "gaussian",
        nbins: int = 50,
        xlim: tuple = None,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ):
        """
    Draws the vDataColumns Density Plot.

    Parameters
    ----------
    columns: SQLColumns, optional
        List of the vDataColumns names. If empty, all numerical vDataColumns will 
        be selected.
    bandwidth: float, optional
        The bandwidth of the kernel.
    kernel: str, optional
        The method used for the plot.
            gaussian  : Gaussian Kernel.
            logistic  : Logistic Kernel.
            sigmoid   : Sigmoid Kernel.
            silverman : Silverman Kernel.
    nbins: int, optional
        Maximum number of points to use to evaluate the approximate density function.
        Increasing this parameter will increase the precision but will also increase 
        the time of the learning and the scoring phases.
    xlim: tuple, optional
        Set the x limits of the current axes.
    ax: Axes, optional
        [Only for MATPLOTLIB]
        The axes to plot on.
    **style_kwargs
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Axes

    See Also
    --------
    vDataFrame[].bar : Draws the Bar Chart of the vDataColumn based on an aggregation.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns = self._format_colnames(columns)
        if not (columns):
            columns = self.numcol()
        else:
            for column in columns:
                assert self[column].isnum(), TypeError(
                    f"vDataColumn {column} is not numerical to draw KDE"
                )
        assert columns, EmptyParameter("No Numerical Columns found to draw KDE.")
        colors = get_colors()
        min_max = self.agg(func=["min", "max"], columns=columns)
        if not xlim:
            xmin = min(min_max["min"])
            xmax = max(min_max["max"])
        else:
            xmin, xmax = xlim
        custom_lines = []
        for idx, column in enumerate(columns):
            param = {"color": colors[idx % len(colors)]}
            ax = self[column].density(
                bandwidth=bandwidth,
                kernel=kernel,
                nbins=nbins,
                xlim=(xmin, xmax),
                ax=ax,
                **PlottingBase._update_dict(param, style_kwargs, idx),
            )
            custom_lines += [
                Line2D([0], [0], color=colors[idx % len(colors)], lw=4),
            ]
        ax.legend(custom_lines, columns, loc="center left", bbox_to_anchor=[1, 0.5])
        ax.set_ylim(bottom=0)
        return ax

    @save_verticapy_logs
    def hchart(
        self,
        x: SQLExpression = None,
        y: SQLExpression = None,
        z: SQLExpression = None,
        c: SQLExpression = None,
        aggregate: bool = True,
        kind: Literal[
            "area",
            "area_range",
            "area_ts",
            "bar",
            "boxplot",
            "bubble",
            "candlestick",
            "donut",
            "donut3d",
            "heatmap",
            "hist",
            "line",
            "negative_bar",
            "pie",
            "pie_half",
            "pie3d",
            "scatter",
            "spider",
            "spline",
            "stacked_bar",
            "stacked_hist",
            "pearson",
            "kendall",
            "cramer",
            "biserial",
            "spearman",
            "spearmand",
        ] = "boxplot",
        width: int = 600,
        height: int = 400,
        options: dict = {},
        h: float = -1,
        max_cardinality: int = 10,
        limit: int = 10000,
        drilldown: bool = False,
        stock: bool = False,
        alpha: float = 0.25,
    ):
        """
    [Beta Version]
    Draws responsive charts using the High Chart API: 
    https://api.highcharts.com/highcharts/

    The returned object can be customized using the API parameters and the 
    'set_dict_options' method.

    \u26A0 Warning : This function uses the unsupported HighChart Python API. 
                     For more information, see python-hicharts repository:
                     https://github.com/kyper-data/python-highcharts

    Parameters
    ----------
    x / y / z / c: SQLExpression
        The vDataColumns and aggregations used to draw the chart. These will depend 
        on the chart type. You can also specify an expression, but it must be a SQL 
        statement. For example: AVG(column1) + SUM(column2) AS new_name.

            area / area_ts / line / spline
                x: numerical or type date like vDataColumn.
                y: a single expression or list of expressions used to draw the plot
                z: [OPTIONAL] vDataColumn representing the different categories 
                    (only if y is a single vDataColumn)
            area_range
                x: numerical or date type vDataColumn.
                y: list of three expressions [expression, lower bound, upper bound]
            bar (single) / donut / donut3d / hist (single) / pie / pie_half / pie3d
                x: vDataColumn used to compute the categories.
                y: [OPTIONAL] numerical expression representing the categories values. 
                    If empty, COUNT(*) is used as the default aggregation.
            bar (double / drilldown) / hist (double / drilldown) / pie (drilldown) 
            / stacked_bar / stacked_hist
                x: vDataColumn used to compute the first category.
                y: vDataColumn used to compute the second category.
                z: [OPTIONAL] numerical expression representing the different categories 
                    values. 
                    If empty, COUNT(*) is used as the default aggregation.
            biserial / boxplot / pearson / kendall / pearson / spearman / spearmanD
                x: list of the vDataColumns used to draw the Chart.
            bubble / scatter
                x: numerical vDataColumn.
                y: numerical vDataColumn.
                z: numerical vDataColumn (bubble size in case of bubble plot, third 
                     dimension in case of scatter plot)
                c: [OPTIONAL] vDataColumn used to compute the different categories.
            candlestick
                x: date type vDataColumn.
                y: Can be a numerical vDataColumn or list of 5 expressions 
                    [last quantile, maximum, minimum, first quantile, volume]
            negative_bar
                x: binary vDataColumn used to compute the first category.
                y: vDataColumn used to compute the second category.
                z: [OPTIONAL] numerical expression representing the categories values. 
                    If empty, COUNT(*) is used as the default aggregation.
            spider
                x: vDataColumn used to compute the different categories.
                y: [OPTIONAL] Can be a list of the expressions used to draw the Plot 
                    or a single expression. 
                    If empty, COUNT(*) is used as the default aggregation.
    aggregate: bool, optional
        If set to True, the input vDataColumns will be aggregated.
    kind: str, optional
        Chart Type.
            area         : Area Chart
            area_range   : Area Range Chart
            area_ts      : Area Chart with Time Series Design
            bar          : Bar Chart
            biserial     : Biserial Point Matrix (Correlation between binary
                             variables and numerical)
            boxplot      : Box Plot
            bubble       : Bubble Plot
            candlestick  : Candlestick and Volumes (Time Series Special Plot)
            cramer       : Cramer's V Matrix (Correlation between categories)
            donut        : Donut Chart
            donut3d      : 3D Donut Chart
            heatmap      : Heatmap
            hist         : BarChart
            kendall      : Kendall Correlation Matrix. The method will compute the Tau-B 
                           coefficients.
                           \u26A0 Warning : This method uses a CROSS JOIN during computation 
                                            and is therefore computationally expensive at 
                                            O(n * n), where n is the total count of the 
                                            vDataFrame.
            line         : Line Plot
            negative_bar : Multi Bar Chart for binary classes
            pearson      : Pearson Correlation Matrix
            pie          : Pie Chart
            pie_half     : Half Pie Chart
            pie3d        : 3D Pie Chart
            scatter      : Scatter Plot
            spider       : Spider Chart
            spline       : Spline Plot
            stacked_bar  : Stacked Bar Chart
            stacked_hist : Stacked BarChart
            spearman     : Spearman's Correlation Matrix
            spearmanD    : Spearman's Correlation Matrix using the DENSE RANK
                           function instead of the RANK function.
    width: int, optional
        Chart Width.
    height: int, optional
        Chart Height.
    options: dict, optional
        High Chart Dictionary to use to customize the Chart. Look at the API 
        documentation to know the different options.
    h: float, optional
        Interval width of the bar. If empty, an optimized value will be used.
    max_cardinality: int, optional
        Maximum number of the vDataColumn distinct elements.
    limit: int, optional
        Maximum number of elements to draw.
    drilldown: bool, optional
        Drilldown Chart: Only possible for Bars, BarCharts, donuts and pies.
                          Instead of drawing 2D charts, this option allows you
                          to add a drilldown effect to 1D Charts.
    stock: bool, optional
        Stock Chart: Only possible for Time Series. The design of the Time
                     Series is dragable and have multiple options.
    alpha: float, optional
        Value used to determine the position of the upper and lower quantile 
        (Used when kind is set to 'candlestick')

    Returns
    -------
    Highchart
        Chart Object
        """
        kind = str(kind).lower()
        params = [
            self,
            x,
            y,
            z,
            c,
            aggregate,
            kind,
            width,
            height,
            options,
            h,
            max_cardinality,
            limit,
            drilldown,
            stock,
            alpha,
        ]
        try:
            return hchart_from_vdf(*params)
        except:
            params[5] = not (params[5])
            return hchart_from_vdf(*params)

    @save_verticapy_logs
    def heatmap(
        self,
        columns: SQLColumns,
        method: str = "count",
        of: Optional[str] = None,
        h: tuple = (None, None),
        ax: Optional[Axes] = None,
        **style_kwargs,
    ):
        """
    Draws the Heatmap of the two input vDataColumns.

    Parameters
    ----------
    columns: SQLColumns
        List of the vDataColumns names. The list must have two elements.
    method: str, optional
        The method to use to aggregate the data.
            count   : Number of elements.
            density : Percentage of the distribution.
            mean    : Average of the vDataColumn 'of'.
            min     : Minimum of the vDataColumn 'of'.
            max     : Maximum of the vDataColumn 'of'.
            sum     : Sum of the vDataColumn 'of'.
            q%      : q Quantile of the vDataColumn 'of (ex: 50% to get the median).
        It can also be a cutomized aggregation (ex: AVG(column1) + 5).
    of: str, optional
        The vDataColumn to use to compute the aggregation.
    h: tuple, optional
        Interval width of the vDataColumns 1 and 2 bars. Optimized h will be computed 
        if the parameter is empty or invalid.
    ax: Axes, optional
        [Only for MATPLOTLIB]
        The axes to plot on.
    **style_kwargs
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Axes

    See Also
    --------
    vDataFrame.pivot_table  : Draws the pivot table of vDataColumns based on an aggregation.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns, of = self._format_colnames(columns, of, expected_nb_of_cols=2)
        for column in columns:
            assert self[column].isnum(), TypeError(
                f"vDataColumn {column} must be numerical to draw the Heatmap."
            )
        min_max = self.agg(func=["min", "max"], columns=columns).transpose()
        vpy_plt, kwargs = self._get_plotting_lib(
            matplotlib_kwargs={
                "ax": ax,
                "show": True,
                "with_numbers": with_numbers,
                "fill_none": 0.0,
                "return_ax": True,
                "extent": min_max[columns[0]] + min_max[columns[1]],
            },
            style_kwargs=style_kwargs,
        )
        return vpy_plt.HeatMap(
            vdf=self,
            columns=columns,
            method=method,
            of=of,
            h=h,
            max_cardinality=(0, 0),
        ).draw(**kwargs)

    @save_verticapy_logs
    def hexbin(
        self,
        columns: SQLColumns,
        method: Literal["density", "count", "avg", "min", "max", "sum"] = "count",
        of: Optional[str] = None,
        bbox: list = [],
        img: str = "",
        ax: Optional[Axes] = None,
        **style_kwargs,
    ):
        """
    Draws the Hexbin of the input vDataColumns based on an aggregation.

    Parameters
    ----------
    columns: SQLColumns
        List of the vDataColumns names. The list must have two elements.
    method: str, optional
        The method to use to aggregate the data.
            count   : Number of elements.
            density : Percentage of the distribution.
            mean    : Average of the vDataColumn 'of'.
            min     : Minimum of the vDataColumn 'of'.
            max     : Maximum of the vDataColumn 'of'.
            sum     : Sum of the vDataColumn 'of'.
            q%      : q Quantile of the vDataColumn 'of (ex: 50% to get the median).
    of: str, optional
        The vDataColumn to use to compute the aggregation.
    bbox: list, optional
        List of 4 elements to delimit the boundaries of the final Plot. 
        It must be similar the following list: [xmin, xmax, ymin, ymax]
    img: str, optional
         Path to the image to display as background.
    ax: Axes, optional
        [Only for MATPLOTLIB]
        The axes to plot on.
    **style_kwargs
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Axes

    See Also
    --------
    vDataFrame.pivot_table : Draws the pivot table of vDataColumns based on an aggregation.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns, of = self._format_colnames(columns, of, expected_nb_of_cols=2)
        vpy_plt, kwargs = self._get_plotting_lib(
            matplotlib_kwargs={"ax": ax, "bbox": bbox, "img": img},
            style_kwargs=style_kwargs,
        )
        return vpy_plt.HexbinMap(vdf=self, columns=columns, method=method, of=of,).draw(
            **kwargs
        )

    @save_verticapy_logs
    def outliers_plot(
        self,
        columns: SQLColumns,
        threshold: float = 3.0,
        color: str = "orange",
        outliers_color: str = "black",
        inliers_color: str = "white",
        inliers_border_color: str = "red",
        max_nb_points: int = 500,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ):
        """
    Draws the global outliers plot of one or two columns based on their ZSCORE.

    Parameters
    ----------
    columns: SQLColumns
        List of one or two vDataColumn names.
    threshold: float, optional
        ZSCORE threshold used to detect outliers.
    color: str, optional
        Inliers Area color.
    outliers_color: str, optional
        Outliers color.
    inliers_color: str, optional
        Inliers color.
    inliers_border_color: str, optional
        Inliers border color.
    max_nb_points: int, optional
        Maximum number of points to display.
    ax: Axes, optional
        [Only for MATPLOTLIB]
        The axes to plot on.
    **style_kwargs
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax: Axes, optional
        [Only for MATPLOTLIB]
        The axes to plot on.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns = self._format_colnames(columns, expected_nb_of_cols=[1, 2])
        return vpy_matplotlib_plt.OutliersPlot().draw(
            self,
            columns,
            color=color,
            threshold=threshold,
            outliers_color=outliers_color,
            inliers_color=inliers_color,
            inliers_border_color=inliers_border_color,
            max_nb_points=max_nb_points,
            ax=ax,
            **style_kwargs,
        )

    @save_verticapy_logs
    def pie(
        self,
        columns: SQLColumns,
        max_cardinality: Union[int, tuple, list] = None,
        h: Union[float, tuple] = None,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ):
        """
    Draws the nested density pie chart of the input vDataColumns.

    Parameters
    ----------
    columns: SQLColumns
        List of the vDataColumns names.
    max_cardinality: int / tuple / list, optional
        Maximum number of the vDataColumn distinct elements to be used as categorical 
        (No h will be picked or computed).
        If of type tuple, it must represent each column 'max_cardinality'.
    h: float/tuple, optional
        Interval width of the bar. If empty, an optimized h will be computed.
        If of type tuple, it must represent each column 'h'.
    ax: Axes, optional
        [Only for MATPLOTLIB]
        The axes to plot on.
    **style_kwargs
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Axes

    See Also
    --------
    vDataFrame[].pie : Draws the Pie Chart of the vDataColumn based on an aggregation.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns = self._format_colnames(columns)
        return vpy_matplotlib_plt.NestedPieChart().draw(
            self, columns, max_cardinality, h, ax=ax, **style_kwargs
        )

    @save_verticapy_logs
    def pivot_table(
        self,
        columns: SQLColumns,
        method: str = "count",
        of: Optional[str] = None,
        max_cardinality: tuple[int, int] = (20, 20),
        h: tuple[PythonNumber, PythonNumber] = (None, None),
        show: bool = True,
        with_numbers: bool = True,
        fill_none: float = 0.0,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ):
        """
    Draws the pivot table of one or two columns based on an aggregation.

    Parameters
    ----------
    columns: SQLColumns
        List of the vDataColumns names. The list must have one or two elements.
    method: str, optional
        The method to use to aggregate the data.
            count   : Number of elements.
            density : Percentage of the distribution.
            mean    : Average of the vDataColumn 'of'.
            min     : Minimum of the vDataColumn 'of'.
            max     : Maximum of the vDataColumn 'of'.
            sum     : Sum of the vDataColumn 'of'.
            q%      : q Quantile of the vDataColumn 'of (ex: 50% to get the median).
        It can also be a cutomized aggregation (ex: AVG(column1) + 5).
    of: str, optional
        The vDataColumn to use to compute the aggregation.
    max_cardinality: tuple, optional
        Maximum number of distinct elements for vDataColumns 1 and 2 to be used as 
        categorical (No h will be picked or computed)
    h: tuple, optional
        Interval width of the vDataColumns 1 and 2 bars. It is only valid if the 
        vDataColumns are numerical. Optimized h will be computed if the parameter 
        is empty or invalid.
    show: bool, optional
        If set to True, the result will be drawn using Matplotlib.
    with_numbers: bool, optional
        If set to True, no number will be displayed in the final drawing.
    fill_none: float, optional
        The empty values of the pivot table will be filled by this number.
    ax: Axes, optional
        [Only for MATPLOTLIB]
        The axes to plot on.
    **style_kwargs
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    TableSample
        An object containing the result. For more information, see
        utilities.TableSample.

    See Also
    --------
    vDataFrame.hexbin : Draws the Hexbin Plot of 2 vDataColumns based on an aggregation.
    vDataFrame.pivot  : Returns the Pivot of the vDataFrame using the input aggregation.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns, of = self._format_colnames(columns, of, expected_nb_of_cols=[1, 2])
        vpy_plt, kwargs = self._get_plotting_lib(
            matplotlib_kwargs={"ax": ax, "with_numbers": with_numbers},
            style_kwargs=style_kwargs,
        )
        plt_obj = vpy_plt.HeatMap(
            vdf=self,
            columns=columns,
            method=method,
            of=of,
            h=h,
            max_cardinality=max_cardinality,
            fill_none=fill_none,
        )
        if show:
            return plt_obj.draw(**kwargs)
        values = {"index": plt_obj.layout["x_labels"]}
        if len(plt_obj.data["X"].shape) == 1:
            values[plt_obj.layout["aggregate"]] = list(plt_obj.data["X"])
        else:
            for idx in range(plt_obj.data["X"].shape[1]):
                values[plt_obj.layout["y_labels"][idx]] = list(
                    plt_obj.data["X"][:, idx]
                )
        return TableSample(values=values)

    @save_verticapy_logs
    def plot(
        self,
        ts: str,
        columns: SQLColumns = [],
        start_date: PythonScalar = None,
        end_date: PythonScalar = None,
        step: bool = False,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ):
        """
    Draws the time series.

    Parameters
    ----------
    ts: str
        TS (Time Series) vDataColumn to use to order the data. The vDataColumn type must be
        date like (date, datetime, timestamp...) or numerical.
    columns: SQLColumns, optional
        List of the vDataColumns names. If empty, all numerical vDataColumns will be 
        used.
    start_date: PythonScalar, optional
        Input Start Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is lesser than November 1993 the 3rd.
    end_date: PythonScalar, optional
        Input End Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is greater than November 1993 the 3rd.
    step: bool, optional
        If set to True, draw a Step Plot.
    ax: Axes, optional
        [Only for MATPLOTLIB]
        The axes to plot on.
    **style_kwargs
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Axes

    See Also
    --------
    vDataFrame[].plot : Draws the Time Series of one vDataColumn.
        """
        vpy_plt, kwargs = self._get_plotting_lib(
            matplotlib_kwargs={"ax": ax, "kind": "step" if step else "line",},
            style_kwargs=style_kwargs,
        )
        return vpy_plt.MultiLinePlot(
            vdf=self,
            order_by=ts,
            columns=columns,
            order_by_start=start_date,
            order_by_end=end_date,
        ).draw(**kwargs)

    @save_verticapy_logs
    def range_plot(
        self,
        columns: SQLColumns,
        ts: str,
        q: tuple = (0.25, 0.75),
        start_date: PythonScalar = None,
        end_date: PythonScalar = None,
        plot_median: bool = False,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ):
        """
    Draws the range plot of the input vDataColumns. The aggregations used are the median 
    and two input quantiles.

    Parameters
    ----------
    columns: SQLColumns
        List of vDataColumns names.
    ts: str
        TS (Time Series) vDataColumn to use to order the data. The vDataColumn type must be
        date like (date, datetime, timestamp...) or numerical.
    q: tuple, optional
        Tuple including the 2 quantiles used to draw the Plot.
    start_date: str / PythonNumber / date, optional
        Input Start Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is lesser than November 1993 the 3rd.
    end_date: str / PythonNumber / date, optional
        Input End Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is greater than November 1993 the 3rd.
    plot_median: bool, optional
        If set to True, the Median will be drawn.
    ax: Axes, optional
        [Only for MATPLOTLIB]
        The axes to plot on.
    **style_kwargs
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Axes

    See Also
    --------
    vDataFrame.plot : Draws the time series.
        """
        vpy_plt, kwargs = self._get_plotting_lib(
            matplotlib_kwargs={"ax": ax, "plot_median": plot_median,},
            style_kwargs=style_kwargs,
        )
        return vpy_plt.RangeCurve(
            vdf=self,
            columns=columns,
            order_by=ts,
            q=q,
            order_by_start=start_date,
            order_by_end=end_date,
        ).draw(**kwargs)

    @save_verticapy_logs
    def scatter(
        self,
        columns: SQLColumns,
        catcol: Optional[str] = None,
        cmap_col: Optional[str] = None,
        size_bubble_col: Optional[str] = None,
        max_cardinality: int = 6,
        cat_priority: Union[None, PythonScalar, ArrayLike] = None,
        max_nb_points: int = 20000,
        dimensions: tuple = None,
        bbox: Optional[tuple] = None,
        img: Optional[str] = None,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ):
        """
    Draws the scatter plot of the input vDataColumns.

    Parameters
    ----------
    columns: SQLColumns
        List of the vDataColumns names. 
    catcol: str, optional
        Categorical vDataColumn to use to label the data.
    cmap_col: str, optional
        Numerical column used with a color map as color.
    size_bubble_col: str
        Numerical vDataColumn to use to represent the Bubble size.
    max_cardinality: int, optional
        Maximum number of distinct elements for 'catcol' to be used as 
        categorical. The less frequent elements will be gathered together to 
        create a new category: 'Others'.
    cat_priority: PythonScalar / ArrayLike, optional
        ArrayLike of the different categories to consider when labeling the data using
        the vDataColumn 'catcol'. The other categories will be filtered.
    max_nb_points: int, optional
        Maximum number of points to display.
    dimensions: tuple, optional
        Tuple of two elements representing the IDs of the PCA's components.
        If empty and the number of input columns is greater than 3, the
        first and second PCA will be drawn.
    bbox: list, optional
        Tuple of 4 elements to delimit the boundaries of the final Plot. 
        It must be similar the following list: [xmin, xmax, ymin, ymax]
    img: str, optional
        Path to the image to display as background.
    ax: Axes, optional
        [Only for MATPLOTLIB]
        The axes to plot on.
    **style_kwargs
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Axes

    See Also
    --------
    vDataFrame.bubble      : Draws the bubble plot of the input vDataColumns.
    vDataFrame.pivot_table : Draws the pivot table of vDataColumns based on an aggregation.
        """
        from verticapy.machine_learning.vertica.decomposition import PCA

        if img and not (bbox) and len(columns) == 2:
            aggr = self.agg(columns=columns, func=["min", "max"])
            bbox = (
                aggr.values["min"][0],
                aggr.values["max"][0],
                aggr.values["min"][1],
                aggr.values["max"][1],
            )
        if len(columns) > 3 and dimensions == None:
            dimensions = (1, 2)
        if isinstance(dimensions, Iterable):
            model_name = gen_tmp_name(
                schema=conf.get_option("temp_schema"), name="pca_plot"
            )
            model = PCA(model_name)
            model.drop()
            try:
                model.fit(self, columns)
                ax = model.transform(self).scatter(
                    columns=["col1", "col2"],
                    catcol=catcol,
                    cmap_col=cmap_col,
                    size_bubble_col=size_bubble_col,
                    max_cardinality=max_cardinality,
                    cat_priority=cat_priority,
                    max_nb_points=max_nb_points,
                    bbox=bbox,
                    img=img,
                    ax=ax,
                    **style_kwargs,
                )
                for idx, fun in enumerate([ax.set_xlabel, ax.set_ylabel]):
                    if not (model.explained_variance_[dimensions[idx] - 1]):
                        dimension2 = ""
                    else:
                        x2 = round(
                            model.explained_variance_[dimensions[idx] - 1] * 100, 1
                        )
                        dimension2 = f"({x2}%)"
                    fun(f"Dim{dimensions[idx]} {dimension2}")
            finally:
                model.drop()
            return ax
        vpy_plt, kwargs = self._get_plotting_lib(
            matplotlib_kwargs={"ax": ax, "bbox": bbox, "img": img,},
            style_kwargs=style_kwargs,
        )
        return vpy_plt.ScatterPlot(
            vdf=self,
            columns=columns,
            catcol=catcol,
            cmap_col=cmap_col,
            size_bubble_col=size_bubble_col,
            max_cardinality=max_cardinality,
            cat_priority=cat_priority,
            max_nb_points=max_nb_points,
        ).draw(**kwargs)

    @save_verticapy_logs
    def scatter_matrix(
        self, columns: SQLColumns = [], max_nb_points: int = 1000, **style_kwargs
    ):
        """
    Draws the scatter matrix of the vDataFrame.

    Parameters
    ----------
    columns: SQLColumns, optional
        List of the vDataColumns names. If empty, all numerical vDataColumns will be 
        used.
    max_nb_points: int, optional
        Maximum number of points to display for each scatter plot.
    **style_kwargs
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Axes

    See Also
    --------
    vDataFrame.scatter : Draws the scatter plot of the input vDataColumns.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns = self._format_colnames(columns)
        vpy_plt, kwargs = self._get_plotting_lib(style_kwargs=style_kwargs,)
        return vpy_plt.ScatterMatrix(
            vdf=self, columns=columns, max_nb_points=max_nb_points
        ).draw(**kwargs)

    @save_verticapy_logs
    def stacked_area(
        self,
        ts: str,
        columns: SQLColumns = [],
        start_date: PythonScalar = None,
        end_date: PythonScalar = None,
        fully: bool = False,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ):
        """
    Draws the stacked area chart of the time series.

    Parameters
    ----------
    ts: str
        TS (Time Series) vDataColumn to use to order the data. The vDataColumn type must be
        date like (date, datetime, timestamp...) or numerical.
    columns: SQLColumns, optional
        List of the vDataColumns names. If empty, all numerical vDataColumns will be 
        used. They must all include only positive values.
    start_date: PythonScalar, optional
        Input Start Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is lesser than November 1993 the 3rd.
    end_date: PythonScalar, optional
        Input End Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is greater than November 1993 the 3rd.
    fully: bool, optional
        If set to True, a Fully Stacked Area Chart will be drawn.
    ax: Axes, optional
        [Only for MATPLOTLIB]
        The axes to plot on.
    **style_kwargs
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Axes
        """
        if isinstance(columns, str):
            columns = [columns]
        elif not (columns):
            columns = self.numcol()
        assert min(self.min(columns)["min"]) >= 0, ValueError(
            "Columns having negative values can not be "
            "processed by the 'stacked_area' method."
        )
        columns, ts = self._format_colnames(columns, ts)
        vpy_plt, kwargs = self._get_plotting_lib(
            matplotlib_kwargs={
                "ax": ax,
                "kind": "area_percent" if fully else "area_stacked",
            },
            style_kwargs=style_kwargs,
        )
        return vpy_plt.MultiLinePlot(
            vdf=self,
            order_by=ts,
            columns=columns,
            order_by_start=start_date,
            order_by_end=end_date,
        ).draw(**kwargs)


class vDCPlot:
    def numh(
        self, method: Literal["sturges", "freedman_diaconis", "fd", "auto"] = "auto"
    ):
        """
    Computes the optimal vDataColumn bar width.

    Parameters
    ----------
    method: str, optional
        Method to use to compute the optimal h.
            auto              : Combination of Freedman Diaconis and Sturges.
            freedman_diaconis : Freedman Diaconis [2 * IQR / n ** (1 / 3)]
            sturges           : Sturges [CEIL(log2(n)) + 1]

    Returns
    -------
    float
        optimal bar width.
        """
        if method == "auto":
            pre_comp = self._parent._get_catalog_value(self._alias, "numh")
            if pre_comp != "VERTICAPY_NOT_PRECOMPUTED":
                return pre_comp
        assert self.isnum() or self.isdate(), ParameterError(
            "numh is only available on type numeric|date"
        )
        if self.isnum():
            result = (
                self._parent.describe(
                    method="numerical", columns=[self._alias], unique=False
                )
                .transpose()
                .values[self._alias]
            )
            (
                count,
                vDataColumn_min,
                vDataColumn_025,
                vDataColumn_075,
                vDataColumn_max,
            ) = (
                result[0],
                result[3],
                result[4],
                result[6],
                result[7],
            )
        elif self.isdate():
            result = _executeSQL(
                f"""
                SELECT 
                    /*+LABEL('vDataColumn.numh')*/ COUNT({self._alias}) AS NAs, 
                    MIN({self._alias}) AS min, 
                    APPROXIMATE_PERCENTILE({self._alias} 
                        USING PARAMETERS percentile = 0.25) AS Q1, 
                    APPROXIMATE_PERCENTILE({self._alias} 
                        USING PARAMETERS percentile = 0.75) AS Q3, 
                    MAX({self._alias}) AS max 
                FROM 
                    (SELECT 
                        DATEDIFF('second', 
                                 '{self.min()}'::timestamp, 
                                 {self._alias}) AS {self._alias} 
                    FROM {self._parent._genSQL()}) VERTICAPY_OPTIMAL_H_TABLE""",
                title="Different aggregations to compute the optimal h.",
                method="fetchrow",
                sql_push_ext=self._parent._vars["sql_push_ext"],
                symbol=self._parent._vars["symbol"],
            )
            (
                count,
                vDataColumn_min,
                vDataColumn_025,
                vDataColumn_075,
                vDataColumn_max,
            ) = result
        sturges = max(
            float(vDataColumn_max - vDataColumn_min)
            / int(math.floor(math.log(count, 2) + 2)),
            1e-99,
        )
        fd = max(
            2.0 * (vDataColumn_075 - vDataColumn_025) / (count) ** (1.0 / 3.0), 1e-99
        )
        if method.lower() == "sturges":
            best_h = sturges
        elif method.lower() in ("freedman_diaconis", "fd"):
            best_h = fd
        else:
            best_h = max(sturges, fd)
            self._parent._update_catalog({"index": ["numh"], self._alias: [best_h]})
        if self.category() == "int":
            best_h = max(math.floor(best_h), 1)
        return best_h

    @save_verticapy_logs
    def bar(
        self,
        method: str = "density",
        of: Optional[str] = None,
        max_cardinality: int = 6,
        nbins: int = 0,
        h: PythonNumber = 0,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ):
        """
    Draws the histogram of the vDataColumn based on an aggregation.

    Parameters
    ----------
    method: str, optional
        The method to use to aggregate the data.
            count   : Number of elements.
            density : Percentage of the distribution.
            mean    : Average of the vDataColumn 'of'.
            min     : Minimum of the vDataColumn 'of'.
            max     : Maximum of the vDataColumn 'of'.
            sum     : Sum of the vDataColumn 'of'.
            q%      : q Quantile of the vDataColumn 'of' (ex: 50% to get the median).
        It can also be a cutomized aggregation (ex: AVG(column1) + 5).
    of: str, optional
        The vDataColumn to use to compute the aggregation.
    max_cardinality: int, optional
        Maximum number of the vDataColumn distinct elements to be used as categorical 
        (No h will be picked or computed)
    nbins: int, optional
        Number of bins. If empty, an optimized number of bins will be computed.
    h: PythonNumber, optional
        Interval width of the bar. If empty, an optimized h will be computed.
    ax: Axes, optional
        [Only for MATPLOTLIB]
        The axes to plot on.
    **style_kwargs
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Axes

    See Also
    --------
    vDataFrame[].bar : Draws the Bar Chart of vDataColumn based on an aggregation.
        """
        vpy_plt, kwargs = self._parent._get_plotting_lib(
            matplotlib_kwargs={"ax": ax}, style_kwargs=style_kwargs
        )
        return vpy_plt.BarChart(
            vdc=self,
            method=method,
            of=of,
            max_cardinality=max_cardinality,
            nbins=nbins,
            h=h,
        ).draw(**kwargs)

    @save_verticapy_logs
    def barh(
        self,
        method: str = "density",
        of: Optional[str] = None,
        max_cardinality: int = 6,
        nbins: int = 0,
        h: PythonNumber = 0,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ):
        """
    Draws the bar chart of the vDataColumn based on an aggregation.

    Parameters
    ----------
    method: str, optional
        The method to use to aggregate the data.
            count   : Number of elements.
            density : Percentage of the distribution.
            mean    : Average of the vDataColumn 'of'.
            min     : Minimum of the vDataColumn 'of'.
            max     : Maximum of the vDataColumn 'of'.
            sum     : Sum of the vDataColumn 'of'.
            q%      : q Quantile of the vDataColumn 'of' (ex: 50% to get the median).
        It can also be a cutomized aggregation (ex: AVG(column1) + 5).
    of: str, optional
        The vDataColumn to use to compute the aggregation.
    max_cardinality: int, optional
        Maximum number of the vDataColumn distinct elements to be used as categorical 
        (No h will be picked or computed)
    nbins: int, optional
        Number of nbins. If empty, an optimized number of nbins will be computed.
    h: PythonNumber, optional
        Interval width of the bar. If empty, an optimized h will be computed.
    ax: Axes, optional
        [Only for MATPLOTLIB]
        The axes to plot on.
    **style_kwargs
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Axes

    See Also
    --------
    vDataFrame[].bar : Draws the bar chart of the vDataColumn based on an aggregation.
        """
        vpy_plt, kwargs = self._parent._get_plotting_lib(
            matplotlib_kwargs={"ax": ax}, style_kwargs=style_kwargs
        )
        return vpy_plt.HorizontalBarChart(
            vdc=self,
            method=method,
            of=of,
            max_cardinality=max_cardinality,
            nbins=nbins,
            h=h,
        ).draw(**kwargs)

    @save_verticapy_logs
    def hist(
        self,
        by: Optional[str] = None,
        method: str = "density",
        of: Optional[str] = None,
        h: PythonNumber = None,
        h_by: PythonNumber = 0,
        max_cardinality: int = 8,
        cat_priority: Union[None, PythonScalar, ArrayLike] = None,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ):
        """
    Draws the histogram of the input vDataColumn based on an aggregation.

    Parameters
    ----------
    by: str, optional
        vDataColumn to use to partition the data.
    method: str, optional
        The method to use to aggregate the data.
            count   : Number of elements.
            density : Percentage of the distribution.
            mean    : Average of the vDataColumn 'of'.
            min     : Minimum of the vDataColumn 'of'.
            max     : Maximum of the vDataColumn 'of'.
            sum     : Sum of the vDataColumn 'of'.
            q%      : q Quantile of the vDataColumn 'of' (ex: 50% to get the median).
        It can also be a cutomized aggregation (ex: AVG(column1) + 5).
    of: str, optional
        The vDataColumn to use to compute the aggregation.
    h: PythonNumber, optional
        Interval width of the input vDataColumns. Optimized h will be computed 
        if the parameter is empty or invalid.
    h_by: PythonNumber, optional
        Interval width if the 'by' vDataColumn is numerical or of type date like. Optimized 
        h will be computed if the parameter is empty or invalid.
    max_cardinality: int, optional
        Maximum number of vDataColumn distinct elements to be used as categorical. 
        The less frequent elements will be gathered together to create a new 
        category : 'Others'.
    cat_priority: PythonScalar / ArrayLike, optional
        ArrayLike of the different categories to consider when drawing the box plot.
        The other categories will be filtered.
    ax: Axes, optional
        [Only for MATPLOTLIB]
        The axes to plot on.
    **style_kwargs
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Axes

    See Also
    --------
    vDataFrame[].bar : Draws the bar chart of the vDataColumn based on an aggregation.
        """
        vpy_plt, kwargs = self._parent._get_plotting_lib(
            matplotlib_kwargs={"ax": ax}, style_kwargs=style_kwargs,
        )
        return vpy_plt.Histogram(
            vdf=self._parent,
            columns=[self._alias],
            by=by,
            method=method,
            of=of,
            h=h,
            h_by=h_by,
            max_cardinality=max_cardinality,
            cat_priority=cat_priority,
        ).draw(**kwargs)

    @save_verticapy_logs
    def boxplot(
        self,
        by: Optional[str] = None,
        q: tuple = (0.25, 0.75),
        h: PythonNumber = 0,
        max_cardinality: int = 8,
        cat_priority: Union[None, PythonScalar, ArrayLike] = None,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ):
        """
    Draws the box plot of the vDataColumn.

    Parameters
    ----------
    by: str, optional
        vDataColumn to use to partition the data.
    q: tuple, optional
        Tuple including the 2 quantiles used to draw the BoxPlot.
    h: PythonNumber, optional
        Interval width if the 'by' vDataColumn is numerical or of type date like. Optimized 
        h will be computed if the parameter is empty or invalid.
    max_cardinality: int, optional
        Maximum number of vDataColumn distinct elements to be used as categorical. 
        The less frequent elements will be gathered together to create a new 
        category : 'Others'.
    cat_priority: PythonScalar / ArrayLike, optional
        ArrayLike of the different categories to consider when drawing the box plot.
        The other categories will be filtered.
    ax: Axes, optional
        [Only for MATPLOTLIB]
        The axes to plot on.
    **style_kwargs
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Axes

    See Also
    --------
    vDataFrame.boxplot : Draws the Box Plot of the input vDataColumns. 
        """
        vpy_plt, kwargs = self._parent._get_plotting_lib(
            matplotlib_kwargs={"ax": ax}, style_kwargs=style_kwargs
        )
        return vpy_plt.BoxPlot(
            vdf=self._parent,
            columns=[self._alias],
            by=by,
            q=q,
            h=h,
            max_cardinality=max_cardinality,
            cat_priority=cat_priority,
        ).draw(**kwargs)

    @save_verticapy_logs
    def density(
        self,
        by: str = "",
        bandwidth: PythonNumber = 1.0,
        kernel: Literal["gaussian", "logistic", "sigmoid", "silverman"] = "gaussian",
        nbins: int = 200,
        xlim: tuple = None,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ):
        """
    Draws the vDataColumn Density Plot.

    Parameters
    ----------
    by: str, optional
        vDataColumn to use to partition the data.
    bandwidth: PythonNumber, optional
        The bandwidth of the kernel.
    kernel: str, optional
        The method used for the plot.
            gaussian  : Gaussian kernel.
            logistic  : Logistic kernel.
            sigmoid   : Sigmoid kernel.
            silverman : Silverman kernel.
    nbins: int, optional
        Maximum number of points to use to evaluate the approximate density function.
        Increasing this parameter will increase the precision but will also increase 
        the time of the learning and scoring phases.
    xlim: tuple, optional
        Set the x limits of the current axes.
    ax: Axes, optional
        [Only for MATPLOTLIB]
        The axes to plot on.
    **style_kwargs
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Axes

    See Also
    --------
    vDataFrame[].bar : Draws the bar chart of the vDataColumn based on an aggregation.
        """
        from verticapy.machine_learning.vertica import KernelDensity

        if by:
            by = self._parent._format_colnames(by)
            colors = get_colors()
            if not xlim:
                xmin = self.min()
                xmax = self.max()
            else:
                xmin, xmax = xlim
            custom_lines = []
            columns = self._parent[by].distinct()
            for idx, column in enumerate(columns):
                param = {"color": colors[idx % len(colors)]}
                ax = self._parent.search(f"{self._parent[by]._alias} = '{column}'")[
                    self._alias
                ].density(
                    bandwidth=bandwidth,
                    kernel=kernel,
                    nbins=nbins,
                    xlim=(xmin, xmax),
                    ax=ax,
                    **PlottingBase._update_dict(param, style_kwargs, idx),
                )
                custom_lines += [
                    Line2D(
                        [0],
                        [0],
                        color=PlottingBase._update_dict(param, style_kwargs, idx)[
                            "color"
                        ],
                        lw=4,
                    ),
                ]
            ax.set_title("KernelDensity")
            ax.legend(
                custom_lines,
                columns,
                title=by,
                loc="center left",
                bbox_to_anchor=[1, 0.5],
            )
            ax.set_xlabel(self._alias)
            return ax
        kernel = kernel.lower()
        schema = conf.get_option("temp_schema")
        if not (schema):
            schema = "public"
        name = gen_tmp_name(schema=schema, name="kde")
        if isinstance(xlim, (tuple, list)):
            xlim_tmp = [xlim]
        else:
            xlim_tmp = []
        model = KernelDensity(
            name,
            bandwidth=bandwidth,
            kernel=kernel,
            nbins=nbins,
            xlim=xlim_tmp,
            store=False,
        )
        try:
            result = model.fit(self._parent._genSQL(), [self._alias]).plot(
                ax=ax, **style_kwargs
            )
            return result
        finally:
            model.drop()

    @save_verticapy_logs
    def geo_plot(self, *args, **kwargs):
        """
    Draws the Geospatial object.

    Parameters
    ----------
    *args / **kwargs
        Any optional parameter to pass to the geopandas plot function.
        For more information, see: 
        https://geopandas.readthedocs.io/en/latest/docs/reference/api/
                geopandas.GeoDataFrame.plot.html
    
    Returns
    -------
    ax
        Axes
        """
        columns = [self._alias]
        check = True
        if len(args) > 0:
            column = args[0]
        elif "column" in kwargs:
            column = kwargs["column"]
        else:
            check = False
        if check:
            column = self._parent._format_colnames(column)
            columns += [column]
            if not ("cmap" in kwargs):
                kwargs["cmap"] = PlottingBase().get_cmap(idx=0)
        else:
            if not ("color" in kwargs):
                kwargs["color"] = get_colors(idx=0)
        if not ("legend" in kwargs):
            kwargs["legend"] = True
        if not ("figsize" in kwargs):
            kwargs["figsize"] = (14, 10)
        return self._parent[columns].to_geopandas(self._alias).plot(*args, **kwargs)

    @save_verticapy_logs
    def pie(
        self,
        method: str = "density",
        of: Optional[str] = None,
        max_cardinality: int = 6,
        h: PythonNumber = 0,
        pie_type: Literal["auto", "donut", "rose"] = "auto",
        ax: Optional[Axes] = None,
        **style_kwargs,
    ):
        """
    Draws the pie chart of the vDataColumn based on an aggregation.

    Parameters
    ----------
    method: str, optional
        The method to use to aggregate the data.
            count   : Number of elements.
            density : Percentage of the distribution.
            mean    : Average of the vDataColumn 'of'.
            min     : Minimum of the vDataColumn 'of'.
            max     : Maximum of the vDataColumn 'of'.
            sum     : Sum of the vDataColumn 'of'.
            q%      : q Quantile of the vDataColumn 'of' (ex: 50% to get the median).
        It can also be a cutomized aggregation (ex: AVG(column1) + 5).
    of: str, optional
        The vDataColumn to use to compute the aggregation.
    max_cardinality: int, optional
        Maximum number of the vDataColumn distinct elements to be used as categorical 
        (No h will be picked or computed)
    h: PythonNumber, optional
        Interval width of the bar. If empty, an optimized h will be computed.
    pie_type: str, optional
        The type of pie chart.
            auto   : Regular pie chart.
            donut  : Donut chart.
            rose   : Rose chart.
        It can also be a cutomized aggregation (ex: AVG(column1) + 5).
    ax: Axes, optional
        [Only for MATPLOTLIB]
        The axes to plot on.
    **style_kwargs
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Axes

    See Also
    --------
    vDataFrame.donut : Draws the donut chart of the vDataColumn based on an aggregation.
        """
        vpy_plt, kwargs = self._parent._get_plotting_lib(
            matplotlib_kwargs={"ax": ax, "pie_type": pie_type},
            style_kwargs=style_kwargs,
        )
        return vpy_plt.PieChart(
            vdc=self,
            method=method,
            of=of,
            max_cardinality=max_cardinality,
            h=h,
            pie=True,
        ).draw(**kwargs)

    @save_verticapy_logs
    def plot(
        self,
        ts: str,
        by: str = "",
        start_date: PythonScalar = None,
        end_date: PythonScalar = None,
        area: bool = False,
        step: bool = False,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ):
        """
    Draws the Time Series of the vDataColumn.

    Parameters
    ----------
    ts: str
        TS (Time Series) vDataColumn to use to order the data. The vDataColumn type must be
        date like (date, datetime, timestamp...) or numerical.
    by: str, optional
        vDataColumn to use to partition the TS.
    start_date: str / PythonNumber / date, optional
        Input Start Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is lesser than November 1993 the 3rd.
    end_date: str / PythonNumber / date, optional
        Input End Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is greater than November 1993 the 3rd.
    area: bool, optional
        If set to True, draw an Area Plot.
    step: bool, optional
        If set to True, draw a Step Plot.
    ax: Axes, optional
        [Only for MATPLOTLIB]
        The axes to plot on.
    **style_kwargs
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Axes

    See Also
    --------
    vDataFrame.plot : Draws the time series.
        """
        ts, by = self._parent._format_colnames(ts, by)
        vpy_plt, kwargs = self._parent._get_plotting_lib(
            matplotlib_kwargs={"ax": ax, "area": area, "step": step},
            style_kwargs=style_kwargs,
        )
        return vpy_plt.LinePlot(
            vdf=self._parent,
            order_by=ts,
            columns=[self._alias, by] if by else [self._alias],
            order_by_start=start_date,
            order_by_end=end_date,
        ).draw(**kwargs)

    @save_verticapy_logs
    def range_plot(
        self,
        ts: str,
        q: tuple = (0.25, 0.75),
        start_date: PythonScalar = None,
        end_date: PythonScalar = None,
        plot_median: bool = False,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ):
        """
    Draws the range plot of the vDataColumn. The aggregations used are the median 
    and two input quantiles.

    Parameters
    ----------
    ts: str
        TS (Time Series) vDataColumn to use to order the data. The vDataColumn type must be
        date like (date, datetime, timestamp...) or numerical.
    q: tuple, optional
        Tuple including the 2 quantiles used to draw the Plot.
    start_date: str / PythonNumber / date, optional
        Input Start Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is lesser than November 1993 the 3rd.
    end_date: str / PythonNumber / date, optional
        Input End Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is greater than November 1993 the 3rd.
    plot_median: bool, optional
        If set to True, the Median will be drawn.
    ax: Axes, optional
        [Only for MATPLOTLIB]
        The axes to plot on.
    **style_kwargs
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Axes

    See Also
    --------
    vDataFrame.plot : Draws the time series.
        """
        return self._parent.range_plot(
            columns=[self._alias],
            ts=ts,
            q=q,
            start_date=start_date,
            end_date=end_date,
            plot_median=plot_median,
            ax=ax,
            **style_kwargs,
        )

    @save_verticapy_logs
    def spider(
        self,
        by: str = "",
        method: str = "density",
        of: Optional[str] = None,
        max_cardinality: tuple[int, int] = (6, 6),
        h: tuple[PythonNumber, PythonNumber] = (None, None),
        ax: Optional[Axes] = None,
        **style_kwargs,
    ):
        """
    Draws the spider plot of the input vDataColumn based on an aggregation.

    Parameters
    ----------
    by: str, optional
        vDataColumn to use to partition the data.
    method: str, optional
        The method to use to aggregate the data.
            count   : Number of elements.
            density : Percentage of the distribution.
            mean    : Average of the vDataColumn 'of'.
            min     : Minimum of the vDataColumn 'of'.
            max     : Maximum of the vDataColumn 'of'.
            sum     : Sum of the vDataColumn 'of'.
            q%      : q Quantile of the vDataColumn 'of' (ex: 50% to get the median).
        It can also be a cutomized aggregation (ex: AVG(column1) + 5).
    of: str, optional
        The vDataColumn to use to compute the aggregation.
    h: PythonNumber / tuple / list, optional
        Interval width of the vDataColumns 1 and 2 bars. It is only valid if the 
        vDataColumns are numerical. Optimized h will be computed if the parameter 
        is empty or invalid.
    max_cardinality: tuple, optional
        Maximum number of distinct elements for vDataColumns 1 and 2 to be used as 
        categorical (No h will be picked or computed)
    ax: Axes, optional
        [Only for MATPLOTLIB]
        The axes to plot on.
    **style_kwargs
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Axes

    See Also
    --------
    vDataFrame.bar : Draws the Bar Chart of the input vDataColumns based on an aggregation.
        """
        by, of = self._parent._format_colnames(by, of)
        columns = [self._alias]
        if by:
            columns += [by]
        vpy_plt, kwargs = self._parent._get_plotting_lib(
            matplotlib_kwargs={"ax": ax}, style_kwargs=style_kwargs
        )
        return vpy_plt.SpiderChart(
            vdf=self._parent,
            columns=columns,
            method=method,
            of=of,
            max_cardinality=max_cardinality,
            h=h,
        ).draw(**kwargs)
