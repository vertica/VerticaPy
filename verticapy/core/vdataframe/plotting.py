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
from typing import Union, Literal
import verticapy.plotting._matplotlib as plt
from verticapy._utils._gen import gen_tmp_name


class vDFPLOT:
    @save_verticapy_logs
    def animated(
        self,
        ts: str,
        columns: Union[list] = [],
        by: str = "",
        start_date: Union[str, int, float, datetime.datetime, datetime.date] = "",
        end_date: Union[str, int, float, datetime.datetime, datetime.date] = "",
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
        ax=None,
        **style_kwds,
    ):
        """
    Draws the animated chart.

    Parameters
    ----------
    ts: str
        TS (Time Series) vColumn to use to order the data. The vColumn type must be
        date like (date, datetime, timestamp...) or numerical.
    columns: str / list, optional
        List of the vColumns names.
    by: str, optional
        Categorical vColumn used in the partition.
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
        If set to True, the ts vColumn will be displayed in the title section.
    date_f: function, optional
        Function used to display the ts vColumn.
    date_style_dict: dict, optional
        Style Dictionary used to display the ts vColumn when date_in_title = False.
    interval: int, optional
        Number of ms between each update.
    repeat: bool, optional
        If set to True, the animation will be repeated.
    return_html: bool, optional
        If set to True and if using a Jupyter notebook, the HTML of the animation will be 
        generated.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
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
            f"Parameter 'columns' must include at least 2 numerical vColumns and maximum 4 vColumns when using kind = '{kind}'."
        )
        columns, ts, by = self.format_colnames(columns, ts, by)
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
            return plt.animated_bubble_plot(
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
                **style_kwds,
            )
        elif kind in ("bar", "pie"):
            return plt.animated_bar(
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
                **style_kwds,
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
            return plt.animated_ts_plot(
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
                **style_kwds,
            )

    @save_verticapy_logs
    def bar(
        self,
        columns: Union[str, list],
        method: str = "density",
        of: str = "",
        max_cardinality: tuple = (6, 6),
        h: tuple = (None, None),
        hist_type: Literal[
            "auto",
            "fully_stacked",
            "stacked",
            "fully",
            "fully stacked",
            "pyramid",
            "density",
        ] = "auto",
        ax=None,
        **style_kwds,
    ):
        """
    Draws the bar chart of the input vColumns based on an aggregation.

    Parameters
    ----------
    columns: str / list
        List of the vColumns names. The list must have one or two elements.
    method: str, optional
        The method to use to aggregate the data.
            count   : Number of elements.
            density : Percentage of the distribution.
            mean    : Average of the vColumn 'of'.
            min     : Minimum of the vColumn 'of'.
            max     : Maximum of the vColumn 'of'.
            sum     : Sum of the vColumn 'of'.
            q%      : q Quantile of the vColumn 'of' (ex: 50% to get the median).
        It can also be a cutomized aggregation (ex: AVG(column1) + 5).
    of: str, optional
         The vColumn to use to compute the aggregation.
    h: tuple, optional
        Interval width of the vColumns 1 and 2 bars. It is only valid if the 
        vColumns are numerical. Optimized h will be computed if the parameter 
        is empty or invalid.
    max_cardinality: tuple, optional
        Maximum number of distinct elements for vColumns 1 and 2 to be used as 
        categorical (No h will be picked or computed)
    hist_type: str, optional
        The Histogram Type.
            auto          : Regular Bar Chart based on 1 or 2 vColumns.
            pyramid       : Pyramid Density Bar Chart. Only works if one of
                            the two vColumns is binary and the 'method' is 
                            set to 'density'.
            stacked       : Stacked Bar Chart based on 2 vColumns.
            fully_stacked : Fully Stacked Bar Chart based on 2 vColumns.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object

     See Also
     --------
     vDataFrame.boxplot     : Draws the Box Plot of the input vColumns.
     vDataFrame.hist        : Draws the histogram of the input vColumns based 
                              on an aggregation.
     vDataFrame.pivot_table : Draws the pivot table of vColumns based on an 
                              aggregation.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns, of = self.format_colnames(columns, of, expected_nb_of_cols=[1, 2])
        if len(columns) == 1:
            return self[columns[0]].bar(method, of, 6, 0, 0, ax=ax, **style_kwds)
        else:
            stacked, fully_stacked, density = False, False, False
            if hist_type in ("fully", "fully stacked", "fully_stacked"):
                fully_stacked = True
            elif hist_type == "stacked":
                stacked = True
            elif hist_type in ("pyramid", "density"):
                density = True
            return plt.bar2D(
                self,
                columns,
                method,
                of,
                max_cardinality,
                h,
                stacked,
                fully_stacked,
                density,
                ax=ax,
                **style_kwds,
            )

    @save_verticapy_logs
    def boxplot(self, columns: Union[str, list] = [], ax=None, **style_kwds):
        """
    Draws the Box Plot of the input vColumns. 

    Parameters
    ----------
    columns: str / list, optional
        List of the vColumns names. If empty, all numerical vColumns will 
        be used.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object

    See Also
    --------
    vDataFrame.bar         : Draws the bar chart of the input vColumns based 
                             on an aggregation.
    vDataFrame.boxplot     : Draws the vColumn box plot.
    vDataFrame.hist        : Draws the histogram of the input vColumns based 
                             on an aggregation.
    vDataFrame.pivot_table : Draws the pivot table of vColumns based on an 
                             aggregation.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns = self.format_colnames(columns) if (columns) else self.numcol()
        return plt.boxplot2D(self, columns, ax=ax, **style_kwds)

    @save_verticapy_logs
    def bubble(
        self,
        columns: Union[str, list],
        size_bubble_col: str = "",
        catcol: str = "",
        cmap_col: str = "",
        max_nb_points: int = 20000,
        bbox: list = [],
        img: str = "",
        ax=None,
        **style_kwds,
    ):
        """
    Draws the bubble plot of the input vColumns.

    Parameters
    ----------
    columns: str / list
        List of the vColumns names. The list must have two elements.
    size_bubble_col: str
        Numerical vColumn to use to represent the Bubble size.
    catcol: str, optional
        Categorical column used as color.
    cmap_col: str, optional
        Numerical column used with a color map as color.
    max_nb_points: int, optional
        Maximum number of points to display.
    bbox: list, optional
        List of 4 elements to delimit the boundaries of the final Plot. 
        It must be similar the following list: [xmin, xmax, ymin, ymax]
    img: str, optional
        Path to the image to display as background.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
       Matplotlib axes object

    See Also
    --------
    vDataFrame.scatter : Draws the scatter plot of the input vColumns.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns, catcol, size_bubble_col, cmap_col = self.format_colnames(
            columns, catcol, size_bubble_col, cmap_col, expected_nb_of_cols=2
        )
        return plt.bubble(
            self,
            columns + [size_bubble_col] if size_bubble_col else columns,
            catcol,
            cmap_col,
            max_nb_points,
            bbox,
            img,
            ax=ax,
            **style_kwds,
        )

    @save_verticapy_logs
    def contour(self, columns: list, func, nbins: int = 100, ax=None, **style_kwds):
        """
    Draws the contour plot of the input function two input vColumns.

    Parameters
    ----------
    columns: list
        List of the vColumns names. The list must have two elements.
    func: function / str
        Function used to compute the contour score. It can also be a SQL
        expression.
    nbins: int, optional
        Number of bins used to discretize the two input numerical vColumns.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object

     See Also
     --------
     vDataFrame.boxplot     : Draws the Box Plot of the input vColumns.
     vDataFrame.hist        : Draws the histogram of the input vColumns based on an aggregation.
     vDataFrame.pivot_table : Draws the pivot table of vColumns based on an aggregation.
        """
        columns = self.format_colnames(columns, expected_nb_of_cols=2)
        return plt.contour_plot(self, columns, func, nbins, ax=ax, **style_kwds,)

    @save_verticapy_logs
    def heatmap(
        self,
        columns: Union[str, list],
        method: str = "count",
        of: str = "",
        h: tuple = (None, None),
        ax=None,
        **style_kwds,
    ):
        """
    Draws the Heatmap of the two input vColumns.

    Parameters
    ----------
    columns: str / list
        List of the vColumns names. The list must have two elements.
    method: str, optional
        The method to use to aggregate the data.
            count   : Number of elements.
            density : Percentage of the distribution.
            mean    : Average of the vColumn 'of'.
            min     : Minimum of the vColumn 'of'.
            max     : Maximum of the vColumn 'of'.
            sum     : Sum of the vColumn 'of'.
            q%      : q Quantile of the vColumn 'of (ex: 50% to get the median).
        It can also be a cutomized aggregation (ex: AVG(column1) + 5).
    of: str, optional
        The vColumn to use to compute the aggregation.
    h: tuple, optional
        Interval width of the vColumns 1 and 2 bars. Optimized h will be computed 
        if the parameter is empty or invalid.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object

    See Also
    --------
    vDataFrame.pivot_table  : Draws the pivot table of vColumns based on an aggregation.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns, of = self.format_colnames(columns, of, expected_nb_of_cols=2)
        for column in columns:
            assert self[column].isnum(), TypeError(
                f"vColumn {column} must be numerical to draw the Heatmap."
            )
        min_max = self.agg(func=["min", "max"], columns=columns).transpose()
        ax = plt.pivot_table(
            vdf=self,
            columns=columns,
            method=method,
            of=of,
            h=h,
            max_cardinality=(0, 0),
            show=True,
            with_numbers=False,
            fill_none=0.0,
            ax=ax,
            return_ax=True,
            extent=min_max[columns[0]] + min_max[columns[1]],
            **style_kwds,
        )
        return ax

    @save_verticapy_logs
    def hexbin(
        self,
        columns: Union[str, list],
        method: Literal["density", "count", "avg", "min", "max", "sum"] = "count",
        of: str = "",
        bbox: list = [],
        img: str = "",
        ax=None,
        **style_kwds,
    ):
        """
    Draws the Hexbin of the input vColumns based on an aggregation.

    Parameters
    ----------
    columns: str / list
        List of the vColumns names. The list must have two elements.
    method: str, optional
        The method to use to aggregate the data.
            count   : Number of elements.
            density : Percentage of the distribution.
            mean    : Average of the vColumn 'of'.
            min     : Minimum of the vColumn 'of'.
            max     : Maximum of the vColumn 'of'.
            sum     : Sum of the vColumn 'of'.
    of: str, optional
        The vColumn to use to compute the aggregation.
    bbox: list, optional
        List of 4 elements to delimit the boundaries of the final Plot. 
        It must be similar the following list: [xmin, xmax, ymin, ymax]
    img: str, optional
         Path to the image to display as background.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object

    See Also
    --------
    vDataFrame.pivot_table : Draws the pivot table of vColumns based on an aggregation.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns, of = self.format_colnames(columns, of, expected_nb_of_cols=2)
        return plt.hexbin(self, columns, method, of, bbox, img, ax=ax, **style_kwds)

    @save_verticapy_logs
    def hist(
        self,
        columns: Union[str, list],
        method: str = "density",
        of: str = "",
        max_cardinality: tuple = (6, 6),
        h: Union[int, float, tuple] = (None, None),
        hist_type: Literal["auto", "multi", "stacked"] = "auto",
        ax=None,
        **style_kwds,
    ):
        """
    Draws the histogram of the input vColumns based on an aggregation.

    Parameters
    ----------
    columns: str / list
        List of the vColumns names. The list must have less than 5 elements.
    method: str, optional
        The method to use to aggregate the data.
            count   : Number of elements.
            density : Percentage of the distribution.
            mean    : Average of the vColumn 'of'.
            min     : Minimum of the vColumn 'of'.
            max     : Maximum of the vColumn 'of'.
            sum     : Sum of the vColumn 'of'.
            q%      : q Quantile of the vColumn 'of' (ex: 50% to get the median).
        It can also be a cutomized aggregation (ex: AVG(column1) + 5).
    of: str, optional
        The vColumn to use to compute the aggregation.
    h: int/float/tuple, optional
        Interval width of the vColumns 1 and 2 bars. It is only valid if the 
        vColumns are numerical. Optimized h will be computed if the parameter 
        is empty or invalid.
    max_cardinality: tuple, optional
        Maximum number of distinct elements for vColumns 1 and 2 to be used as 
        categorical (No h will be picked or computed)
    hist_type: str, optional
        The Histogram Type.
            auto    : Regular Histogram based on 1 or 2 vColumns.
            multi   : Multiple Regular Histograms based on 1 to 5 vColumns.
            stacked : Stacked Histogram based on 2 vColumns.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object

    See Also
    --------
    vDataFrame.bar         : Draws the bar chart of the input vColumns based on an aggregation.
    vDataFrame.boxplot     : Draws the Box Plot of the input vColumns.
    vDataFrame.pivot_table : Draws the pivot table of vColumns based on an aggregation.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns, of = self.format_colnames(
            columns, of, expected_nb_of_cols=[1, 2, 3, 4, 5]
        )
        stacked = True if (hist_type.lower() == "stacked") else False
        multi = True if (hist_type.lower() == "multi") else False
        if len(columns) == 1:
            return self[columns[0]].hist(method, of, 6, 0, 0, **style_kwds)
        else:
            if multi:
                if isinstance(h, (int, float)):
                    h_0 = h
                else:
                    h_0 = h[0] if (h[0]) else 0
                return plt.multiple_hist(
                    self, columns, method, of, h_0, ax=ax, **style_kwds,
                )
            else:
                return plt.hist2D(
                    self,
                    columns,
                    method,
                    of,
                    max_cardinality,
                    h,
                    stacked,
                    ax=ax,
                    **style_kwds,
                )

    @save_verticapy_logs
    def outliers_plot(
        self,
        columns: Union[str, list],
        threshold: float = 3.0,
        color: str = "orange",
        outliers_color: str = "black",
        inliers_color: str = "white",
        inliers_border_color: str = "red",
        max_nb_points: int = 500,
        ax=None,
        **style_kwds,
    ):
        """
    Draws the global outliers plot of one or two columns based on their ZSCORE.

    Parameters
    ----------
    columns: str / list
        List of one or two vColumn names.
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
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax: Matplotlib axes object, optional
        The axes to plot on.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns = self.format_colnames(columns, expected_nb_of_cols=[1, 2])
        return plt.outliers_contour_plot(
            self,
            columns,
            color=color,
            threshold=threshold,
            outliers_color=outliers_color,
            inliers_color=inliers_color,
            inliers_border_color=inliers_border_color,
            max_nb_points=max_nb_points,
            ax=ax,
            **style_kwds,
        )

    @save_verticapy_logs
    def pie(
        self,
        columns: Union[str, list],
        max_cardinality: Union[int, tuple, list] = None,
        h: Union[float, tuple] = None,
        ax=None,
        **style_kwds,
    ):
        """
    Draws the nested density pie chart of the input vColumns.

    Parameters
    ----------
    columns: list
        List of the vColumns names.
    max_cardinality: int / tuple / list, optional
        Maximum number of the vColumn distinct elements to be used as categorical 
        (No h will be picked or computed).
        If of type tuple, it must represent each column 'max_cardinality'.
    h: float/tuple, optional
        Interval width of the bar. If empty, an optimized h will be computed.
        If of type tuple, it must represent each column 'h'.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object

    See Also
    --------
    vDataFrame[].pie : Draws the Pie Chart of the vColumn based on an aggregation.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns = self.format_colnames(columns)
        return plt.nested_pie(self, columns, max_cardinality, h, ax=None, **style_kwds)

    @save_verticapy_logs
    def pivot_table(
        self,
        columns: Union[str, list],
        method: str = "count",
        of: str = "",
        max_cardinality: tuple = (20, 20),
        h: tuple = (None, None),
        show: bool = True,
        with_numbers: bool = True,
        fill_none: float = 0.0,
        ax=None,
        **style_kwds,
    ):
        """
    Draws the pivot table of one or two columns based on an aggregation.

    Parameters
    ----------
    columns: str / list
        List of the vColumns names. The list must have one or two elements.
    method: str, optional
        The method to use to aggregate the data.
            count   : Number of elements.
            density : Percentage of the distribution.
            mean    : Average of the vColumn 'of'.
            min     : Minimum of the vColumn 'of'.
            max     : Maximum of the vColumn 'of'.
            sum     : Sum of the vColumn 'of'.
            q%      : q Quantile of the vColumn 'of (ex: 50% to get the median).
        It can also be a cutomized aggregation (ex: AVG(column1) + 5).
    of: str, optional
        The vColumn to use to compute the aggregation.
    max_cardinality: tuple, optional
        Maximum number of distinct elements for vColumns 1 and 2 to be used as 
        categorical (No h will be picked or computed)
    h: tuple, optional
        Interval width of the vColumns 1 and 2 bars. It is only valid if the 
        vColumns are numerical. Optimized h will be computed if the parameter 
        is empty or invalid.
    show: bool, optional
        If set to True, the result will be drawn using Matplotlib.
    with_numbers: bool, optional
        If set to True, no number will be displayed in the final drawing.
    fill_none: float, optional
        The empty values of the pivot table will be filled by this number.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.hexbin : Draws the Hexbin Plot of 2 vColumns based on an aggregation.
    vDataFrame.pivot  : Returns the Pivot of the vDataFrame using the input aggregation.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns, of = self.format_colnames(columns, of, expected_nb_of_cols=[1, 2])
        return plt.pivot_table(
            self,
            columns,
            method,
            of,
            h,
            max_cardinality,
            show,
            with_numbers,
            fill_none,
            ax=ax,
            **style_kwds,
        )

    @save_verticapy_logs
    def plot(
        self,
        ts: str,
        columns: list = [],
        start_date: Union[str, int, float, datetime.datetime, datetime.date] = "",
        end_date: Union[str, int, float, datetime.datetime, datetime.date] = "",
        step: bool = False,
        ax=None,
        **style_kwds,
    ):
        """
    Draws the time series.

    Parameters
    ----------
    ts: str
        TS (Time Series) vColumn to use to order the data. The vColumn type must be
        date like (date, datetime, timestamp...) or numerical.
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.
    start_date: str / int / float / date, optional
        Input Start Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is lesser than November 1993 the 3rd.
    end_date: str / int / float / date, optional
        Input End Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is greater than November 1993 the 3rd.
    step: bool, optional
        If set to True, draw a Step Plot.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object

    See Also
    --------
    vDataFrame[].plot : Draws the Time Series of one vColumn.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns, ts = self.format_colnames(columns, ts)
        kind = "step" if step else "line"
        return plt.multi_ts_plot(
            self, ts, columns, start_date, end_date, kind, ax=ax, **style_kwds,
        )

    @save_verticapy_logs
    def scatter(
        self,
        columns: Union[str, list],
        catcol: str = "",
        max_cardinality: int = 6,
        cat_priority: list = [],
        with_others: bool = True,
        max_nb_points: int = 20000,
        dimensions: tuple = None,
        bbox: list = [],
        img: str = "",
        ax=None,
        **style_kwds,
    ):
        """
    Draws the scatter plot of the input vColumns.

    Parameters
    ----------
    columns: str, list
        List of the vColumns names. 
    catcol: str, optional
        Categorical vColumn to use to label the data.
    max_cardinality: int, optional
        Maximum number of distinct elements for 'catcol' to be used as 
        categorical. The less frequent elements will be gathered together to 
        create a new category: 'Others'.
    cat_priority: list, optional
        List of the different categories to consider when labeling the data using
        the vColumn 'catcol'. The other categories will be filtered.
    with_others: bool, optional
        If set to false and the cardinality of the vColumn 'catcol' is too big then
        the less frequent element will not be merged to another category and they 
        will not be drawn.
    max_nb_points: int, optional
        Maximum number of points to display.
    dimensions: tuple, optional
        Tuple of two elements representing the IDs of the PCA's components.
        If empty and the number of input columns is greater than 3, the
        first and second PCA will be drawn.
    bbox: list, optional
        List of 4 elements to delimit the boundaries of the final Plot. 
        It must be similar the following list: [xmin, xmax, ymin, ymax]
    img: str, optional
        Path to the image to display as background.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object

    See Also
    --------
    vDataFrame.bubble      : Draws the bubble plot of the input vColumns.
    vDataFrame.pivot_table : Draws the pivot table of vColumns based on an aggregation.
        """
        from verticapy.machine_learning.vertica.decomposition import PCA

        if len(columns) > 3 and dimensions == None:
            dimensions = (1, 2)
        if isinstance(dimensions, Iterable):
            model_name = gen_tmp_name(schema=OPTIONS["temp_schema"], name="pca_plot")
            model = PCA(model_name)
            model.drop()
            try:
                model.fit(self, columns)
                ax = model.transform(self).scatter(
                    columns=["col1", "col2"],
                    catcol=catcol,
                    max_cardinality=100,
                    max_nb_points=max_nb_points,
                    ax=ax,
                    **style_kwds,
                )
                explained_variance = model.explained_variance_["explained_variance"]
                for idx, fun in enumerate([ax.set_xlabel, ax.set_ylabel]):
                    if not (explained_variance[dimensions[idx] - 1]):
                        dimension2 = ""
                    else:
                        x2 = round(explained_variance[dimensions[idx] - 1] * 100, 1)
                        dimension2 = f"({x2}%)"
                    fun(f"Dim{dimensions[idx]} {dimension2}")
            finally:
                model.drop()
            return ax
        args = [
            self,
            columns,
            catcol,
            max_cardinality,
            cat_priority,
            with_others,
            max_nb_points,
            bbox,
            img,
        ]
        return plt.scatter(*args, ax=ax, **style_kwds,)

    @save_verticapy_logs
    def scatter_matrix(self, columns: Union[str, list] = [], **style_kwds):
        """
    Draws the scatter matrix of the vDataFrame.

    Parameters
    ----------
    columns: str / list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object

    See Also
    --------
    vDataFrame.scatter : Draws the scatter plot of the input vColumns.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns = self.format_colnames(columns)
        return plt.scatter_matrix(self, columns, **style_kwds)

    @save_verticapy_logs
    def stacked_area(
        self,
        ts: str,
        columns: list = [],
        start_date: Union[int, float, str, datetime.datetime, datetime.date] = "",
        end_date: Union[int, float, str, datetime.datetime, datetime.date] = "",
        fully: bool = False,
        ax=None,
        **style_kwds,
    ):
        """
    Draws the stacked area chart of the time series.

    Parameters
    ----------
    ts: str
        TS (Time Series) vColumn to use to order the data. The vColumn type must be
        date like (date, datetime, timestamp...) or numerical.
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used. They must all include only positive values.
    start_date: int / float / str / date, optional
        Input Start Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is lesser than November 1993 the 3rd.
    end_date: int / float / str / date, optional
        Input End Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is greater than November 1993 the 3rd.
    fully: bool, optional
        If set to True, a Fully Stacked Area Chart will be drawn.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object
        """
        if isinstance(columns, str):
            columns = [columns]
        if fully:
            kind = "area_percent"
        else:
            kind = "area_stacked"
        assert min(self.min(columns)["min"]) >= 0, ValueError(
            "Columns having negative values can not be "
            "processed by the 'stacked_area' method."
        )
        columns, ts = self.format_colnames(columns, ts)
        return plt.multi_ts_plot(
            self, ts, columns, start_date, end_date, kind=kind, ax=ax, **style_kwds,
        )
