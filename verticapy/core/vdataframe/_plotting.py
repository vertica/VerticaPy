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
import datetime, copy, math
from typing import Callable, Literal, Optional, Union
from collections.abc import Iterable
import numpy as np

from matplotlib.axes import Axes

from verticapy._config.colors import get_colors
import verticapy._config.config as conf
from verticapy._typing import (
    ArrayLike,
    ColorType,
    PlottingMethod,
    PlottingObject,
    PythonNumber,
    PythonScalar,
    SQLColumns,
    SQLExpression,
)
from verticapy._utils._gen import gen_tmp_name
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._sys import _executeSQL

from verticapy.core.tablesample.base import TableSample

from verticapy.plotting._utils import PlottingUtils
from verticapy.plotting._highcharts.base import hchart_from_vdf


class vDFPlot(PlottingUtils):

    # Boxplots.

    @save_verticapy_logs
    def boxplot(
        self,
        columns: SQLColumns = [],
        q: tuple[float, float] = (0.25, 0.75),
        max_nb_fliers: int = 30,
        whis: float = 1.5,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the Box Plot of the input vDataColumns. 

        Parameters
        ----------
        columns: SQLColumns, optional
            List  of the vDataColumns names.  If  empty, all 
            numerical vDataColumns will be used.
        q: tuple, optional
            Tuple including the 2 quantiles used to draw the 
            BoxPlot.
        max_nb_fliers: int, optional
            Maximum number of points to use to represent the 
            fliers  of each category.  Drawing  fliers  will 
            slow down the graphic computation.
        whis: float, optional
            The position of the whiskers.
        ax: Axes, optional
            [Only for MATPLOTLIB]
            The axes to plot on.
        **style_kwargs
            Any  optional parameter to  pass to the plotting 
            functions.

        Returns
        -------
        obj
            Plotting Object.
        """
        vpy_plt, kwargs = self._get_plotting_lib(
            class_name="BoxPlot",
            matplotlib_kwargs={"ax": ax},
            style_kwargs=style_kwargs,
        )
        return vpy_plt.BoxPlot(
            vdf=self, columns=columns, q=q, whis=whis, max_nb_fliers=max_nb_fliers,
        ).draw(**kwargs)

    # 2D / ND CHARTS.

    @save_verticapy_logs
    def bar(
        self,
        columns: SQLColumns,
        method: PlottingMethod = "density",
        of: Optional[str] = None,
        max_cardinality: tuple[int, int] = (6, 6),
        h: tuple[PythonNumber, PythonNumber] = (None, None),
        bar_type: Literal["auto", "stacked"] = "auto",
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the bar chart of the input vDataColumns based 
        on an aggregation.

        Parameters
        ----------
        columns: SQLColumns
            List of  the vDataColumns names.  The list must 
            have one or two elements.
        method: str, optional
            The method to use to aggregate the data.
                count   : Number of elements.
                density : Percentage  of  the  distribution.
                mean    : Average  of the  vDataColumn 'of'.
                min     : Minimum  of the  vDataColumn 'of'.
                max     : Maximum  of the  vDataColumn 'of'.
                sum     : Sum of the vDataColumn 'of'.
                q%      : q Quantile of the vDataColumn 'of' 
                          (ex: 50% to get the median).
            It can also be a cutomized aggregation 
            (ex: AVG(column1) + 5).
        of: str, optional
            The  vDataColumn to use to compute the  aggregation.
        max_cardinality: tuple, optional
            Maximum number of distinct elements for vDataColumns 
            1  and  2  to be used as categorical (No  h will  be 
            picked or computed)
        h: tuple, optional
            Interval width of  the vDataColumns 1 and 2 bars. It 
            is  only  valid if the  vDataColumns are  numerical. 
            Optimized  h will be  computed  if the parameter  is 
            empty or invalid.
        bar_type: str, optional
            The BarChart Type.
                auto    : Regular  BarChart  based  on  1  or  2 
                          vDataColumns.
                stacked : Stacked    BarChart    based    on   2 
                          vDataColumns.
        ax: Axes, optional
            [Only for MATPLOTLIB]
            The axes to plot on.
        **style_kwargs
            Any  optional  parameter  to  pass  to  the plotting 
            functions.

        Returns
        -------
        obj
            Plotting Object.
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
                class_name="BarChart2D",
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
        method: PlottingMethod = "density",
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
    ) -> PlottingObject:
        """
        Draws  the  horizontal  bar  chart  of  the  input 
        vDataColumns based on an aggregation.

        Parameters
        ----------
        columns: SQLColumns
            List of  the vDataColumns names.  The list must 
            have one or two elements.
        method: str, optional
            The method to use to aggregate the data.
                count   : Number of elements.
                density : Percentage  of  the  distribution.
                mean    : Average  of the  vDataColumn 'of'.
                min     : Minimum  of the  vDataColumn 'of'.
                max     : Maximum  of the  vDataColumn 'of'.
                sum     : Sum of the vDataColumn 'of'.
                q%      : q Quantile of the vDataColumn 'of' 
                          (ex: 50% to get the median).
            It can also be a cutomized aggregation 
            (ex: AVG(column1) + 5).
        of: str, optional
            The  vDataColumn to use to compute the  aggregation.
        max_cardinality: tuple, optional
            Maximum number of distinct elements for vDataColumns 
            1  and  2  to be used as categorical (No  h will  be 
            picked or computed)
        h: tuple, optional
            Interval width of  the vDataColumns 1 and 2 bars. It 
            is  only  valid if the  vDataColumns are  numerical. 
            Optimized  h will be  computed  if the parameter  is 
            empty or invalid.
        bar_type: str, optional
            The BarChart Type.
                auto          : Regular Bar Chart  based on 1 or 2 
                                vDataColumns.
                pyramid       : Pyramid  Density  Bar  Chart. Only 
                                works if one of
                                the two vDataColumns is binary and 
                                the 'method' is set to 'density'.
                stacked       : Stacked  Bar  Chart   based  on  2 
                                vDataColumns.
                fully_stacked : Fully Stacked Bar Chart based on 2 
                                vDataColumns.
        ax: Axes, optional
            [Only for MATPLOTLIB]
            The axes to plot on.
        **style_kwargs
            Any  optional  parameter  to  pass  to  the plotting 
            functions.

        Returns
        -------
        obj
            Plotting Object.
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
                class_name="HorizontalBarChart2D",
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
    def pie(
        self,
        columns: SQLColumns,
        max_cardinality: Union[None, int, tuple] = None,
        h: Union[None, int, tuple] = None,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the nested density pie chart of the input 
        vDataColumns.

        Parameters
        ----------
        columns: SQLColumns
            List of the vDataColumns names.
        max_cardinality: int / tuple, optional
            Maximum number of the vDataColumn distinct 
            elements  to  be   used   as   categorical 
            (No h will be picked or computed).
            If  of type tuple,  it must represent each 
            column 'max_cardinality'.
        h: int / tuple, optional
            Interval  width  of the bar. If empty,  an 
            optimized h will be computed.
            If  of type tuple, it must represent  each 
            column 'h'.
        ax: Axes, optional
            [Only for MATPLOTLIB]
            The axes to plot on.
        **style_kwargs
            Any  optional  parameter  to  pass to  the 
            plotting functions.

        Returns
        -------
        obj
            Plotting Object.
        """
        vpy_plt, kwargs = self._get_plotting_lib(
            class_name="NestedPieChart",
            matplotlib_kwargs={"ax": ax,},
            style_kwargs=style_kwargs,
        )
        return vpy_plt.NestedPieChart(
            vdf=self,
            columns=columns,
            max_cardinality=max_cardinality,
            h=h,
            method="count",
        ).draw(**kwargs)

    # Histogram & Density.

    @save_verticapy_logs
    def hist(
        self,
        columns: SQLColumns,
        method: PlottingMethod = "density",
        of: Optional[str] = None,
        h: PythonNumber = None,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws  the  histograms  of  the  input vDataColumns 
        based on an aggregation.

        Parameters
        ----------
        columns: SQLColumns
            List of  the vDataColumns names.  The list must 
            have less than 5 elements.
        method: str, optional
            The method to use to aggregate the data.
                count   : Number of elements.
                density : Percentage  of  the  distribution.
                mean    : Average  of the  vDataColumn 'of'.
                min     : Minimum  of the  vDataColumn 'of'.
                max     : Maximum  of the  vDataColumn 'of'.
                sum     : Sum of the vDataColumn 'of'.
                q%      : q Quantile of the vDataColumn 'of' 
                          (ex: 50% to get the median).
            It can also be a cutomized aggregation 
            (ex: AVG(column1) + 5).
        of: str, optional
            The  vDataColumn to use to compute the  aggregation.
        max_cardinality: tuple, optional
            Maximum number of distinct elements for vDataColumns 
            1  and  2  to be used as categorical (No  h will  be 
            picked or computed)
        h: tuple, optional
            Interval width of the  input vDataColumns. Optimized 
            h  will be  computed if  the  parameter  is empty or 
            invalid.
        ax: Axes, optional
            [Only for MATPLOTLIB]
            The axes to plot on.
        **style_kwargs
            Any  optional  parameter  to  pass  to  the plotting 
            functions.

        Returns
        -------
        obj
            Plotting Object.
        """
        vpy_plt, kwargs = self._get_plotting_lib(
            class_name="Histogram",
            matplotlib_kwargs={"ax": ax},
            style_kwargs=style_kwargs,
        )
        return vpy_plt.Histogram(
            vdf=self, columns=columns, method=method, of=of, h=h,
        ).draw(**kwargs)

    @save_verticapy_logs
    def density(
        self,
        columns: SQLColumns = [],
        bandwidth: float = 1.0,
        kernel: Literal["gaussian", "logistic", "sigmoid", "silverman"] = "gaussian",
        nbins: int = 50,
        xlim: list[tuple[float, float]] = None,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the vDataColumns Density Plot.

        Parameters
        ----------
        columns: SQLColumns, optional
            List  of the vDataColumns names.  If  empty, 
            all numerical vDataColumns will be selected.
        bandwidth: float, optional
            The bandwidth of the kernel.
        kernel: str, optional
            The method used for the plot.
                gaussian  : Gaussian Kernel.
                logistic  : Logistic Kernel.
                sigmoid   : Sigmoid Kernel.
                silverman : Silverman Kernel.
        nbins: int, optional
            Maximum  number of  points to use to  evaluate 
            the approximate density function.
            Increasing  this  parameter will increase  the 
            precision  but will  also increase the time of 
            the learning and the scoring phases.
        xlim: list of tuple, optional
            Set the x limits of the current axes.
        ax: Axes, optional
            [Only for MATPLOTLIB]
            The axes to plot on.
        **style_kwargs
            Any optional parameter to pass to the plotting 
            functions.

        Returns
        -------
        obj
            Plotting Object.
        """
        from verticapy.machine_learning.vertica import KernelDensity

        if isinstance(columns, str):
            columns = [columns]
        columns = self._format_colnames(columns)
        if not (columns):
            columns = self.numcol()
        if not (columns):
            raise ValueError("No numerical columns found.")
        name = gen_tmp_name(schema=conf.get_option("temp_schema"), name="kde")
        if not xlim:
            xmin = min(self[columns].min()["min"])
            xmax = max(self[columns].max()["max"])
            xlim_ = [(xmin, xmax)]
        elif isinstance(xlim, tuple):
            xlim_ = [xlim]
        else:
            xlim_ = xlim
        model = KernelDensity(
            name=name,
            bandwidth=bandwidth,
            kernel=kernel,
            nbins=nbins,
            xlim=xlim_,
            store=False,
        )
        if len(columns) == 1:
            try:
                model.fit(self, columns)
                return model.plot(ax=ax, **style_kwargs)
            finally:
                model.drop()
        else:
            custom_lines, X, Y = [], [], []
            for column in columns:
                try:
                    model.fit(self, [column])
                    data, layout = model._compute_plot_params()
                    X += [data["x"]]
                    Y += [data["y"]]
                finally:
                    model.drop()
            X = np.column_stack(X)
            Y = np.column_stack(Y)
            vpy_plt, kwargs = self._get_plotting_lib(
                class_name="MultiDensityPlot",
                matplotlib_kwargs={"ax": ax},
                style_kwargs=style_kwargs,
            )
            data = {"X": X, "Y": Y}
            layout = {
                "title": "KernelDensity",
                "x_label": None,
                "y_label": "density",
                "labels": np.array(columns),
                "labels_title": None,
            }
            return vpy_plt.MultiDensityPlot(data=data, layout=layout).draw(**kwargs)

    # Time Series.

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
    ) -> PlottingObject:
        """
        Draws the time series.

        Parameters
        ----------
        ts: str
            TS (Time Series)  vDataColumn to use to order 
            the data.  The vDataColumn type must be  date 
            like   (date,   datetime,   timestamp...)  or 
            numerical.
        columns: SQLColumns, optional
            List of the vDataColumns names. If empty, all 
            numerical vDataColumns will be used.
        start_date: PythonScalar, optional
            Input   Start  Date.  For  example,   time  = 
            '03-11-1993'  will  filter the data when 'ts' 
            is lesser than November 1993 the 3rd.
        end_date: PythonScalar, optional
            Input   End   Date.   For   example,   time = 
            '03-11-1993'   will  filter  the  data   when 
            'ts'  is greater than November 1993 the  3rd.
        step: bool, optional
            If set to True, draw a Step Plot.
        ax: Axes, optional
            [Only for MATPLOTLIB]
            The axes to plot on.
        **style_kwargs
            Any   optional  parameter  to   pass  to  the  
            plotting functions.

        Returns
        -------
        obj
            Plotting Object.
        """
        vpy_plt, kwargs = self._get_plotting_lib(
            class_name="MultiLinePlot",
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
        q: tuple[float, float] = (0.25, 0.75),
        start_date: PythonScalar = None,
        end_date: PythonScalar = None,
        plot_median: bool = False,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the range plot of the input vDataColumns. The 
        aggregations  used  are  the median and  two  input 
        quantiles.

        Parameters
        ----------
        columns: SQLColumns
            List of vDataColumns names.
        ts: str
            TS (Time Series) vDataColumn to use to order the 
            data.  The  vDataColumn  type must be date  like 
            (date, datetime, timestamp...) or numerical.
        q: tuple, optional
            Tuple including the 2 quantiles used to draw the 
            Plot.
        start_date: str / PythonNumber / date, optional
            Input Start Date. For example, time = '03-11-1993' 
            will  filter  the data when 'ts' is  lesser  than 
            November 1993 the 3rd.
        end_date: str / PythonNumber / date, optional
            Input End Date.  For example, time = '03-11-1993' 
            will  filter the  data when 'ts' is greater than 
            November 1993 the 3rd.
        plot_median: bool, optional
            If set to True, the Median will be drawn.
        ax: Axes, optional
            [Only for MATPLOTLIB]
            The axes to plot on.
        **style_kwargs
            Any  optional parameter to pass to the  plotting 
            functions.

        Returns
        -------
        obj
            Plotting Object.
        """
        vpy_plt, kwargs = self._get_plotting_lib(
            class_name="RangeCurve",
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
    def stacked_area(
        self,
        ts: str,
        columns: SQLColumns = None,
        start_date: PythonScalar = None,
        end_date: PythonScalar = None,
        fully: bool = False,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the stacked area chart of the time series.

        Parameters
        ----------
        ts: str
            TS (Time Series)  vDataColumn  to use to order 
            the  data.  The vDataColumn  type must be date 
            like   (date,   datetime,   timestamp...)   or 
            numerical.
        columns: SQLColumns, optional
            List of the vDataColumns  names. If empty, all 
            numerical vDataColumns will be used. They must 
            all include only positive values.
        start_date: PythonScalar, optional
            Input   Start  Date.   For  example,  time  = 
            '03-11-1993' will filter the data when 'ts' is 
            lesser than November 1993 the 3rd.
        end_date: PythonScalar, optional
            Input End Date. For example, time = '03-11-1993' 
            will  filter the data when 'ts' is greater than 
            November 1993 the 3rd.
        fully: bool, optional
            If set to True, a Fully Stacked Area Chart will 
            be drawn.
        ax: Axes, optional
            [Only for MATPLOTLIB]
            The axes to plot on.
        **style_kwargs
            Any  optional parameter to pass to the plotting 
            functions.

        Returns
        -------
        obj
            Plotting Object.
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
            class_name="MultiLinePlot",
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

    # 2D MAP.

    @save_verticapy_logs
    def _pivot_table(
        self,
        columns: SQLColumns,
        method: PlottingMethod = "count",
        of: Optional[str] = None,
        max_cardinality: tuple[int, int] = (20, 20),
        h: tuple[PythonNumber, PythonNumber] = (None, None),
        fill_none: float = 0.0,
    ) -> TableSample:
        """
        Computes and  returns the pivot table of one or two 
        columns based on an aggregation.

        Parameters
        ----------
        columns: SQLColumns
            List  of the vDataColumns names.  The list  must 
            have one or two elements.
        method: str, optional
            The method to use to aggregate the data.
                count   : Number of elements.
                density : Percentage of the distribution.
                mean    : Average of the vDataColumn 'of'.
                min     : Minimum of the vDataColumn 'of'.
                max     : Maximum of the vDataColumn 'of'.
                sum     : Sum of the vDataColumn 'of'.
                q%      : q Quantile of the vDataColumn 'of 
                          (ex: 50% to get the median).
            It can also be a cutomized aggregation 
            (ex: AVG(column1) + 5).
        of: str, optional
            The   vDataColumn   to   use  to  compute   the 
            aggregation.
        max_cardinality: tuple, optional
            Maximum   number   of  distinct  elements   for 
            vDataColumns 1 and 2 to  be used as categorical 
            (No h will be picked or computed)
        h: tuple, optional
            Interval width of the vDataColumns 1 and 2 bars. 
            It  is  only  valid   if  the  vDataColumns  are 
            numerical. 
            Optimized h will be computed if the parameter is 
            empty or invalid.
        fill_none: float, optional
            The  empty  values  of the pivot table  will  be 
            filled by this number.

        Returns
        -------
        obj
            TableSample.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns, of = self._format_colnames(columns, of, expected_nb_of_cols=[1, 2])
        vpy_plt = self._get_plotting_lib(class_name="HeatMap")[0]
        plt_obj = vpy_plt.HeatMap(
            vdf=self,
            columns=columns,
            method=method,
            of=of,
            h=h,
            max_cardinality=max_cardinality,
            fill_none=fill_none,
        )
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
    def pivot_table(
        self,
        columns: SQLColumns,
        method: PlottingMethod = "count",
        of: Optional[str] = None,
        max_cardinality: tuple[int, int] = (20, 20),
        h: tuple[PythonNumber, PythonNumber] = (None, None),
        fill_none: float = 0.0,
        with_numbers: bool = True,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the pivot table of one or two columns based on 
        an aggregation.

        Parameters
        ----------
        columns: SQLColumns
            List  of the vDataColumns names.  The list  must 
            have one or two elements.
        method: str, optional
            The method to use to aggregate the data.
                count   : Number of elements.
                density : Percentage of the distribution.
                mean    : Average of the vDataColumn 'of'.
                min     : Minimum of the vDataColumn 'of'.
                max     : Maximum of the vDataColumn 'of'.
                sum     : Sum of the vDataColumn 'of'.
                q%      : q Quantile of the vDataColumn 'of 
                          (ex: 50% to get the median).
            It can also be a cutomized aggregation 
            (ex: AVG(column1) + 5).
        of: str, optional
            The   vDataColumn   to   use  to  compute   the 
            aggregation.
        max_cardinality: tuple, optional
            Maximum   number   of  distinct  elements   for 
            vDataColumns 1 and 2 to  be used as categorical 
            (No h will be picked or computed)
        h: tuple, optional
            Interval width of the vDataColumns 1 and 2 bars. 
            It  is  only  valid   if  the  vDataColumns  are 
            numerical. 
            Optimized h will be computed if the parameter is 
            empty or invalid.
        fill_none: float, optional
            The  empty  values  of the pivot table  will  be 
            filled by this number.
        with_numbers: bool, optional
            If  set to True, no number will be  displayed in 
            the final drawing.
        ax: Axes, optional
            [Only for MATPLOTLIB]
            The axes to plot on.
        **style_kwargs
            Any  optional parameter to pass to the  plotting 
            functions.

        Returns
        -------
        obj
            Plotting Object.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns, of = self._format_colnames(columns, of, expected_nb_of_cols=[1, 2])
        vpy_plt, kwargs = self._get_plotting_lib(
            class_name="HeatMap",
            matplotlib_kwargs={"ax": ax, "with_numbers": with_numbers},
            style_kwargs=style_kwargs,
        )
        return vpy_plt.HeatMap(
            vdf=self,
            columns=columns,
            method=method,
            of=of,
            h=h,
            max_cardinality=max_cardinality,
            fill_none=fill_none,
        ).draw(**kwargs)

    @save_verticapy_logs
    def contour(
        self,
        columns: SQLColumns,
        func: Union[Callable, str],
        nbins: int = 100,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws  the  contour  plot of the input function  two 
        input vDataColumns.

        Parameters
        ----------
        columns: SQLColumns
            List  of the  vDataColumns  names. The list must 
            have two elements.
        func: function / str
            Function  used to compute  the contour score. It 
            can also be a SQL expression.
        nbins: int, optional
            Number of bins used to  discretize the two input 
            numerical vDataColumns.
        ax: Axes, optional
            [Only for MATPLOTLIB]
            The axes to plot on.
        **style_kwargs
            Any  optional parameter to pass to  the plotting 
            functions.

        Returns
        -------
        obj
            Plotting Object.
        """
        vpy_plt, kwargs = self._get_plotting_lib(
            class_name="ContourPlot",
            matplotlib_kwargs={"ax": ax},
            style_kwargs=style_kwargs,
        )
        func_name = None
        if "func_name" in kwargs:
            func_name = kwargs["func_name"]
            del kwargs["func_name"]
        return vpy_plt.ContourPlot(
            vdf=self, columns=columns, func=func, nbins=nbins, func_name=func_name,
        ).draw(**kwargs)

    @save_verticapy_logs
    def heatmap(
        self,
        columns: SQLColumns,
        method: PlottingMethod = "count",
        of: Optional[str] = None,
        h: tuple = (None, None),
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the Heatmap of  the two input vDataColumns.

        Parameters
        ----------
        columns: SQLColumns
            List of the vDataColumns names. The list must 
            have two elements.
        method: str, optional
            The method to use to aggregate the data.
                count   : Number of elements.
                density : Percentage of the distribution.
                mean    : Average of the vDataColumn 'of'.
                min     : Minimum of the vDataColumn 'of'.
                max     : Maximum of the vDataColumn 'of'.
                sum     : Sum of the vDataColumn 'of'.
                q%      : q  Quantile  of the  vDataColumn 
                          'of (ex: 50% to get the median).
            It can also be a cutomized aggregation 
            (ex: AVG(column1) + 5).
        of: str, optional
            The   vDataColumn  to   use  to  compute   the 
            aggregation.
        h: tuple, optional
            Interval width  of  the vDataColumns 1  and  2 
            bars.  Optimized  h  will  be computed if  the 
            parameter is empty or invalid.
        ax: Axes, optional
            [Only for MATPLOTLIB]
            The axes to plot on.
        **style_kwargs
            Any optional parameter to pass to the plotting 
            functions.

        Returns
        -------
        obj
            Plotting Object.
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
            class_name="HeatMap",
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
        method: PlottingMethod = "count",
        of: Optional[str] = None,
        bbox: list = [],
        img: str = "",
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the Hexbin of the  input vDataColumns based 
        on an aggregation.

        Parameters
        ----------
        columns: SQLColumns
            List of the vDataColumns names. The list must 
            have two elements.
        method: str, optional
            The method to use to aggregate the data.
                count   : Number of elements.
                density : Percentage of the distribution.
                mean    : Average of  the vDataColumn 'of'.
                min     : Minimum of  the vDataColumn 'of'.
                max     : Maximum of  the vDataColumn 'of'.
                sum     : Sum of the vDataColumn 'of'.
                q%      : q Quantile of the vDataColumn 'of 
                          (ex: 50% to get the median).
        of: str, optional
            The   vDataColumn   to  use  to   compute   the 
            aggregation.
        bbox: list, optional
            List of 4 elements  to delimit the boundaries of 
            the final Plot. It must be similar the following 
            list: [xmin, xmax, ymin, ymax]
        img: str, optional
            Path  to the  image to  display  as  background.
        ax: Axes, optional
            [Only for MATPLOTLIB]
            The axes to plot on.
        **style_kwargs
            Any  optional parameter to pass to the  plotting 
            functions.

        Returns
        -------
        obj
            Plotting Object.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns, of = self._format_colnames(columns, of, expected_nb_of_cols=2)
        vpy_plt, kwargs = self._get_plotting_lib(
            class_name="HexbinMap",
            matplotlib_kwargs={"ax": ax, "bbox": bbox, "img": img},
            style_kwargs=style_kwargs,
        )
        return vpy_plt.HexbinMap(vdf=self, columns=columns, method=method, of=of,).draw(
            **kwargs
        )

    # Scatters.

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
    ) -> PlottingObject:
        """
        Draws the scatter plot of the input vDataColumns.

        Parameters
        ----------
        columns: SQLColumns
            List of the vDataColumns names. 
        catcol: str, optional
            Categorical  vDataColumn  to  use to label  the 
            data.
        cmap_col: str, optional
            Numerical  column  used  with  a  color  map as 
            color.
        size_bubble_col: str
            Numerical  vDataColumn to use to represent  the 
            Bubble size.
        max_cardinality: int, optional
            Maximum number of distinct elements for 'catcol' 
            to  be  used as categorical.  The less  frequent 
            elements will  be gathered together  to create a 
            new category: 'Others'.
        cat_priority: PythonScalar / ArrayLike, optional
            ArrayLike of the different categories to consider 
            when  labeling  the  data using  the  vDataColumn 
            'catcol'.  The other categories will be filtered.
        max_nb_points: int, optional
            Maximum number of points to display.
        dimensions: tuple, optional
            Tuple of two  elements representing the IDs of the 
            PCA's components. If empty and the number of input 
            columns  is greater  than 3, the first and  second 
            PCA will be drawn.
        bbox: list, optional
            Tuple  of 4 elements to delimit the boundaries  of 
            the  final Plot. It must be similar the  following 
            list: [xmin, xmax, ymin, ymax]
        img: str, optional
            Path to the image to display as background.
        ax: Axes, optional
            [Only for MATPLOTLIB]
            The axes to plot on.
        **style_kwargs
            Any  optional  parameter  to pass to the  plotting 
            functions.

        Returns
        -------
        obj
            Plotting Object.
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
            class_name="ScatterPlot",
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
    ) -> PlottingObject:
        """
        Draws the scatter matrix of the vDataFrame.

        Parameters
        ----------
        columns: SQLColumns, optional
            List of the vDataColumns names. If empty, 
            all numerical  vDataColumns will be used.
        max_nb_points: int, optional
            Maximum  number of points to display for 
            each scatter plot.
        **style_kwargs
            Any  optional  parameter  to pass to the  
            plotting functions.

        Returns
        -------
        obj
            Plotting Object.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns = self._format_colnames(columns)
        vpy_plt, kwargs = self._get_plotting_lib(
            class_name="ScatterMatrix", style_kwargs=style_kwargs,
        )
        return vpy_plt.ScatterMatrix(
            vdf=self, columns=columns, max_nb_points=max_nb_points
        ).draw(**kwargs)

    @save_verticapy_logs
    def outliers_plot(
        self,
        columns: SQLColumns,
        threshold: float = 3.0,
        color: ColorType = "orange",
        outliers_color: ColorType = "black",
        inliers_color: ColorType = "white",
        inliers_border_color: ColorType = "red",
        max_nb_points: int = 500,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the global  outliers plot of one or two 
        columns based on their ZSCORE.

        Parameters
        ----------
        columns: SQLColumns
            List  of  one or two  vDataColumn  names.
        threshold: float, optional
            ZSCORE threshold used to detect outliers.
        color: ColorType, optional
            Inliers Area color.
        outliers_color: ColorType, optional
            Outliers color.
        inliers_color: ColorType, optional
            Inliers color.
        inliers_border_color: ColorType, optional
            Inliers border color.
        max_nb_points: int, optional
            Maximum number of points to display.
        ax: Axes, optional
            [Only for MATPLOTLIB]
            The axes to plot on.
        **style_kwargs
            Any  optional  parameter to pass to  the  
            plotting functions.

        Returns
        -------
        obj
            Plotting Object.
        """
        vpy_plt, kwargs = self._get_plotting_lib(
            class_name="OutliersPlot",
            matplotlib_kwargs={
                "ax": ax,
                "color": color,
                "outliers_color": outliers_color,
                "inliers_color": inliers_color,
                "inliers_border_color": inliers_border_color,
            },
            style_kwargs=style_kwargs,
        )
        return vpy_plt.OutliersPlot(
            vdf=self, columns=columns, threshold=threshold, max_nb_points=max_nb_points,
        ).draw(**kwargs)

    # DEPRECATED.

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


class vDCPlot:

    # Special Methods.

    def numh(
        self, method: Literal["sturges", "freedman_diaconis", "fd", "auto"] = "auto"
    ) -> float:
        """
        Computes the optimal vDataColumn bar width.

        Parameters
        ----------
        method: str, optional
            Method to use to compute the optimal h.
                auto              : Combination of Freedman Diaconis 
                                    and Sturges.
                freedman_diaconis : Freedman Diaconis 
                                    [2 * IQR / n ** (1 / 3)]
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

    # Boxplots.

    @save_verticapy_logs
    def boxplot(
        self,
        by: Optional[str] = None,
        q: tuple[float, float] = (0.25, 0.75),
        h: PythonNumber = 0,
        max_cardinality: int = 8,
        cat_priority: Union[None, PythonScalar, ArrayLike] = None,
        max_nb_fliers: int = 30,
        whis: float = 1.5,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ):
        """
        Draws the box plot of the vDataColumn.

        Parameters
        ----------
        by: str, optional
            vDataColumn  to use to partition  the  data.
        q: tuple, optional
            Tuple including the 2 quantiles used to draw 
            the BoxPlot.
        h: PythonNumber, optional
            Interval  width  if  the 'by'  vDataColumn is 
            numerical or 'of type  date like. Optimized h 
            will be computed if the parameter is empty or 
            invalid.
        max_cardinality: int, optional
            Maximum   number   of   vDataColumn  distinct 
            elements to be used as categorical. 
            The less frequent  elements  will be gathered 
            together to create a new category : 'Others'.
        cat_priority: PythonScalar / ArrayLike, optional
            ArrayLike  of  the  different  categories  to 
            consider when drawing the box plot. The other 
            categories will be filtered.
        max_nb_fliers: int, optional
            Maximum  number of points to use to represent 
            the fliers of each category.
            Drawing  fliers  will  slow down the  graphic 
            computation.
        whis: float, optional
            The position of the whiskers.
        ax: Axes, optional
            [Only for MATPLOTLIB]
            The axes to plot on.
        **style_kwargs
            Any  optional  parameter  to  pass to the  plotting 
            functions.

        Returns
        -------
        obj
            Plotting Object.
        """
        vpy_plt, kwargs = self._parent._get_plotting_lib(
            class_name="BoxPlot",
            matplotlib_kwargs={"ax": ax},
            style_kwargs=style_kwargs,
        )
        return vpy_plt.BoxPlot(
            vdf=self._parent,
            columns=[self._alias],
            by=by,
            q=q,
            h=h,
            max_cardinality=max_cardinality,
            cat_priority=cat_priority,
            max_nb_fliers=max_nb_fliers,
            whis=whis,
        ).draw(**kwargs)

    # 1D CHARTS.

    @save_verticapy_logs
    def bar(
        self,
        method: PlottingMethod = "density",
        of: Optional[str] = None,
        max_cardinality: int = 6,
        nbins: int = 0,
        h: PythonNumber = 0,
        categorical: bool = True,
        bargap: float = 0.06,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the bar chart of the vDataColumn based on an 
        aggregation.

        Parameters
        ----------
        method: str, optional
            The method to use to aggregate the data.
                count   : Number of elements.
                density : Percentage of the distribution.
                mean    : Average of the  vDataColumn 'of'.
                min     : Minimum of the  vDataColumn 'of'.
                max     : Maximum of the  vDataColumn 'of'.
                sum     : Sum of the vDataColumn 'of'.
                q%      : q Quantile of the vDataColumn 'of 
                          (ex: 50% to get the median).
            It can also be a cutomized aggregation 
            (ex: AVG(column1) + 5).
        of: str, optional
            The vDataColumn  to use to compute the aggregation.
        max_cardinality: int, optional
            Maximum number of the vDataColumn distinct elements 
            to be used as categorical (No  h will  be picked or 
            computed)
        nbins: int, optional
            Number  of  bins. If empty, an  optimized number of 
            bins will be computed.
        h: PythonNumber, optional
            Interval width of the bar. If empty, an optimized h 
            will be computed.
        categorical: bool, optional
            If  set to False and the  vDataColumn is numerical,
            the parmater  'max_cardinality' will be ignored and
            the bar  chart will be represented as an histogram.
        bargap: float, optional
            A float between  0 exclusive and 1 inclusive which
            represents the proportion  taken out from each bar
            to render the chart.  This proportion  will create
            gaps  between  each bar.  The  bigger  it is,  the 
            bigger the gap will be.
        ax: Axes, optional
            [Only for MATPLOTLIB]
            The axes to plot on.
        **style_kwargs
            Any  optional  parameter  to  pass to the  plotting 
            functions.

        Returns
        -------
        obj
            Plotting Object.
        """
        vpy_plt, kwargs = self._parent._get_plotting_lib(
            class_name="BarChart",
            matplotlib_kwargs={"ax": ax},
            style_kwargs=style_kwargs,
        )
        return vpy_plt.BarChart(
            vdc=self,
            method=method,
            of=of,
            max_cardinality=max_cardinality,
            nbins=nbins,
            h=h,
            pie=categorical,
            bargap=bargap,
        ).draw(**kwargs)

    @save_verticapy_logs
    def barh(
        self,
        method: str = "density",
        of: Optional[str] = None,
        max_cardinality: int = 6,
        nbins: int = 0,
        h: PythonNumber = 0,
        categorical: bool = True,
        bargap: float = 0.06,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws  the horizontal bar  chart of the vDataColumn 
        based on an aggregation.

        Parameters
        ----------
        method: str, optional
            The method to use to aggregate the data.
                count   : Number of elements.
                density : Percentage of the distribution.
                mean    : Average of the  vDataColumn 'of'.
                min     : Minimum of the  vDataColumn 'of'.
                max     : Maximum of the  vDataColumn 'of'.
                sum     : Sum of the vDataColumn 'of'.
                q%      : q Quantile of the vDataColumn 'of 
                          (ex: 50% to get the median).
            It can also be a cutomized aggregation 
            (ex: AVG(column1) + 5).
        of: str, optional
            The vDataColumn  to use to compute the aggregation.
        max_cardinality: int, optional
            Maximum number of the vDataColumn distinct elements 
            to be used as categorical (No  h will  be picked or 
            computed)
        nbins: int, optional
            Number  of  bins. If empty, an  optimized number of 
            bins will be computed.
        h: PythonNumber, optional
            Interval width of the bar. If empty, an optimized h 
            will be computed.
        categorical: bool, optional
            If  set to False and the  vDataColumn is numerical,
            the parmater  'max_cardinality' will be ignored and
            the bar  chart will be represented as an histogram.
        bargap: float, optional
            A float between  0 exclusive and 1 inclusive which
            represents the proportion  taken out from each bar
            to render the chart.  This proportion  will create
            gaps  between  each bar.  The  bigger  it is,  the 
            bigger the gap will be.
        ax: Axes, optional
            [Only for MATPLOTLIB]
            The axes to plot on.
        **style_kwargs
            Any  optional  parameter  to  pass to the  plotting 
            functions.

        Returns
        -------
        obj
            Plotting Object.
        """
        vpy_plt, kwargs = self._parent._get_plotting_lib(
            class_name="HorizontalBarChart",
            matplotlib_kwargs={"ax": ax},
            style_kwargs=style_kwargs,
        )
        return vpy_plt.HorizontalBarChart(
            vdc=self,
            method=method,
            of=of,
            max_cardinality=max_cardinality,
            nbins=nbins,
            h=h,
            pie=categorical,
            bargap=bargap,
        ).draw(**kwargs)

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
    ) -> PlottingObject:
        """
        Draws the pie chart of the vDataColumn based on an 
        aggregation.

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
                q%      : q Quantile of the vDataColumn 'of' 
                          (ex: 50% to get the median).
            It   can  also  be  a  cutomized   aggregation 
            (ex: AVG(column1) + 5).
        of: str, optional
            The vDataColumn to use to compute the aggregation.
        max_cardinality: int, optional
            Maximum   number  of  the  vDataColumn   distinct 
            elements to be used as  categorical (No h will be 
            picked or computed)
        h: PythonNumber, optional
            Interval width of the bar. If empty, an optimized 
            h will be computed.
        pie_type: str, optional
            The type of pie chart.
                auto   : Regular pie chart.
                donut  : Donut chart.
                rose   : Rose chart.
            It   can    also   be  a  cutomized   aggregation 
            (ex: AVG(column1) + 5).
        ax: Axes, optional
            [Only for MATPLOTLIB]
            The axes to plot on.
        **style_kwargs
            Any  optional parameter to pass to  the  plotting 
            functions.

        Returns
        -------
        obj
            Plotting Object.
        """
        vpy_plt, kwargs = self._parent._get_plotting_lib(
            class_name="PieChart",
            matplotlib_kwargs={"ax": ax, "pie_type": pie_type},
            plotly_kwargs={"pie_type": pie_type},
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
    def spider(
        self,
        by: str = "",
        method: str = "density",
        of: Optional[str] = None,
        max_cardinality: tuple[int, int] = (6, 6),
        h: tuple[PythonNumber, PythonNumber] = (None, None),
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the spider plot of the input vDataColumn based on 
        an aggregation.

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
                q%      : q Quantile of the vDataColumn 'of' 
                          (ex: 50% to get the median).
            It   can  also  be  a  cutomized   aggregation 
            (ex: AVG(column1) + 5).
        of: str, optional
            The vDataColumn to use to compute the aggregation.
        max_cardinality: int, optional
            Maximum   number  of  the  vDataColumn   distinct 
            elements to be used as  categorical (No h will be 
            picked or computed)
        h: PythonNumber, optional
            Interval width of the bar. If empty, an optimized 
            h will be computed.
        ax: Axes, optional
            [Only for MATPLOTLIB]
            The axes to plot on.
        **style_kwargs
            Any  optional parameter to pass to  the  plotting 
            functions.

        Returns
        -------
        obj
            Plotting Object.
        """
        by, of = self._parent._format_colnames(by, of)
        columns = [self._alias]
        if by:
            columns += [by]
        vpy_plt, kwargs = self._parent._get_plotting_lib(
            class_name="SpiderChart",
            matplotlib_kwargs={"ax": ax},
            style_kwargs=style_kwargs,
        )
        return vpy_plt.SpiderChart(
            vdf=self._parent,
            columns=columns,
            method=method,
            of=of,
            max_cardinality=max_cardinality,
            h=h,
        ).draw(**kwargs)

    # Histogram & Density.

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
    ) -> PlottingObject:
        """
        Draws  the histogram of the input vDataColumn based 
        on an aggregation.

        Parameters
        ----------
        by: str, optional
            vDataColumn  to  use  to  partition  the  data.
        method: str, optional
            The method to use to aggregate the data.
                count   : Number of elements.
                density : Percentage of the distribution.
                mean    : Average of the  vDataColumn 'of'.
                min     : Minimum of the  vDataColumn 'of'.
                max     : Maximum of the  vDataColumn 'of'.
                sum     : Sum of the vDataColumn 'of'.
                q%      : q Quantile of the vDataColumn 'of 
                          (ex: 50% to get the median).
            It can also be a cutomized aggregation 
            (ex: AVG(column1) + 5).
        of: str, optional
            The  vDataColumn  to use to compute the  aggregation.
        h: PythonNumber, optional
            Interval  width of the  input vDataColumns. Optimized 
            h  will be  computed  if  the  parameter is empty  or 
            invalid.
        h_by: PythonNumber, optional
            Interval  width if the 'by' vDataColumn is  numerical 
            or of type  date like.  Optimized  h will be computed 
            if the parameter is empty or invalid.
        max_cardinality: int, optional
            Maximum  number  of vDataColumn distinct  elements to 
            be used as categorical. 
            The less frequent  elements will be gathered together 
            to create a new category : 'Others'.
        cat_priority: PythonScalar / ArrayLike, optional
            ArrayLike  of  the different  categories to  consider 
            when drawing the box plot.  The other categories will 
            be filtered.
        ax: Axes, optional
            [Only for MATPLOTLIB]
            The axes to plot on.
        **style_kwargs
            Any  optional  parameter  to  pass  to  the  plotting 
            functions.

        Returns
        -------
        obj
            Plotting Object.
        """
        vpy_plt, kwargs = self._parent._get_plotting_lib(
            class_name="Histogram",
            matplotlib_kwargs={"ax": ax},
            style_kwargs=style_kwargs,
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
    def density(
        self,
        by: Optional[str] = None,
        bandwidth: PythonNumber = 1.0,
        kernel: Literal["gaussian", "logistic", "sigmoid", "silverman"] = "gaussian",
        nbins: int = 200,
        xlim: Optional[tuple[float, float]] = None,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the vDataColumn Density Plot.

        Parameters
        ----------
        by: str, optional
            vDataColumn  to use to partition  the data.
        bandwidth: PythonNumber, optional
            The bandwidth of the kernel.
        kernel: str, optional
            The method used for the plot.
                gaussian  : Gaussian kernel.
                logistic  : Logistic kernel.
                sigmoid   : Sigmoid kernel.
                silverman : Silverman kernel.
        nbins: int, optional
            Maximum number of points  to use to evaluate 
            the approximate density function.
            Increasing this  parameter will increase the 
            precision but will also increase the time of 
            the learning and scoring phases.
        xlim: tuple, optional
            Set the x limits of the current axes.
        ax: Axes, optional
            [Only for MATPLOTLIB]
            The axes to plot on.
        **style_kwargs
            Any  optional  parameter  to  pass   to  the  
            plotting functions.

        Returns
        -------
        obj
            Plotting Object.
        """
        from verticapy.machine_learning.vertica import KernelDensity

        name = gen_tmp_name(schema=conf.get_option("temp_schema"), name="kde")
        by = self._parent._format_colnames(by)
        if not xlim:
            xlim_ = [(self.min(), self.max())]
        else:
            xlim_ = [xlim]
        model = KernelDensity(
            name=name,
            bandwidth=bandwidth,
            kernel=kernel,
            nbins=nbins,
            xlim=xlim_,
            store=False,
        )
        if not (by):
            try:
                model.fit(self._parent, [self._alias])
                return model.plot(ax=ax, **style_kwargs)
            finally:
                model.drop()
        else:
            custom_lines = []
            categories = self._parent[by].distinct()
            X, Y = [], []
            for idx, cat in enumerate(categories):
                vdf = self._parent[by].isin(cat)
                try:
                    model.fit(vdf, [self._alias])
                    data, layout = model._compute_plot_params()
                    X += [data["x"]]
                    Y += [data["y"]]
                finally:
                    model.drop()
            X = np.column_stack(X)
            Y = np.column_stack(Y)
            vpy_plt, kwargs = self._parent._get_plotting_lib(
                class_name="MultiDensityPlot",
                matplotlib_kwargs={"ax": ax},
                style_kwargs=style_kwargs,
            )
            data = {"X": X, "Y": Y}
            layout = {
                "title": "KernelDensity",
                "x_label": self._alias,
                "y_label": "density",
                "labels_title": by,
                "labels": np.array(categories),
            }
            return vpy_plt.MultiDensityPlot(data=data, layout=layout).draw(**kwargs)

    # Time Series.

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
    ) -> PlottingObject:
        """
        Draws the Time Series of the vDataColumn.

        Parameters
        ----------
        ts: str
            TS  (Time Series)  vDataColumn  to use   to order 
            the  data.  The  vDataColumn  type must  be  date 
            like (date, datetime, timestamp...) or  numerical.
        by: str, optional
            vDataColumn to use to partition the TS.
        start_date: str / PythonNumber / date, optional
            Input Start Date. For example, time = '03-11-1993' 
            will  filter  the  data  when 'ts' is lesser than 
            November 1993 the 3rd.
        end_date: str / PythonNumber / date, optional
            Input  End  Date. For example, time = '03-11-1993' 
            will filter  the data when 'ts' is  greater  than 
            November 1993 the 3rd.
        area: bool, optional
            If set to True, draw an Area Plot.
        step: bool, optional
            If set to True, draw a Step Plot.
        ax: Axes, optional
            [Only for MATPLOTLIB]
            The axes to plot on.
        **style_kwargs
            Any  optional  parameter  to pass to the  plotting 
            functions.

        Returns
        -------
        obj
            Plotting Object.
        """
        ts, by = self._parent._format_colnames(ts, by)
        vpy_plt, kwargs = self._parent._get_plotting_lib(
            class_name="LinePlot",
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
        q: tuple[float, float] = (0.25, 0.75),
        start_date: PythonScalar = None,
        end_date: PythonScalar = None,
        plot_median: bool = False,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws  the  range   plot  of  the  vDataColumn.  The 
        aggregations  used  are  the  median  and two  input 
        quantiles.

        Parameters
        ----------
        ts: str
            TS  (Time Series)  vDataColumn  to use   to order 
            the  data.  The  vDataColumn  type must  be  date 
            like (date, datetime, timestamp...) or  numerical.
        q: tuple, optional
            Tuple including the  2 quantiles used to draw the 
            Plot.
        start_date: str / PythonNumber / date, optional
            Input Start Date. For example, time = '03-11-1993' 
            will  filter  the  data  when 'ts' is lesser than 
            November 1993 the 3rd.
        end_date: str / PythonNumber / date, optional
            Input  End  Date. For example, time = '03-11-1993' 
            will filter  the data when 'ts' is  greater  than 
            November 1993 the 3rd.
        plot_median: bool, optional
            If set to True, the Median will be drawn.
        ax: Axes, optional
            [Only for MATPLOTLIB]
            The axes to plot on.
        **style_kwargs
            Any  optional  parameter  to pass to the  plotting 
            functions.

        Returns
        -------
        obj
            Plotting Object.
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

    # Geospatial.

    @save_verticapy_logs
    def geo_plot(self, *args, **kwargs):
        """
        Draws the Geospatial object.

        Parameters
        ----------
        *args / **kwargs
            Any optional parameter to pass to the geopandas 
            plot function.
            For more information, see: 
            https://geopandas.readthedocs.io/en/latest/
            docs/reference/api/geopandas.GeoDataFrame.plot.html
        
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
