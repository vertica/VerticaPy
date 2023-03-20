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
import copy, math
from typing import Optional

from matplotlib.axes import Axes

from verticapy._typing import (
    PythonScalar,
    SQLColumns,
)
from verticapy._utils._sql._collect import save_verticapy_logs

from verticapy.plotting._utils import PlottingUtils


class vDFAnimatedPlot(PlottingUtils):
    @save_verticapy_logs
    def animated_bar(
        self,
        ts: str,
        columns: SQLColumns,
        by: Optional[str] = None,
        start_date: PythonScalar = None,
        end_date: PythonScalar = None,
        limit_over: int = 6,
        limit: int = 1000000,
        fixed_xy_lim: bool = False,
        date_in_title: bool = False,
        date_f=None,
        date_style_dict: dict = {},
        interval: int = 300,
        repeat: bool = True,
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
    columns: SQLColumns
        List of the vDataColumns names.
    by: str, optional
        Categorical vDataColumn used in the partition.
    start_date: PythonScalar, optional
        Input Start Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is lesser than November 1993 the 3rd.
    end_date: PythonScalar, optional
        Input End Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is greater than November 1993 the 3rd.
    limit_over: int, optional
        Limited number of elements to consider for each category.
    limit: int, optional
        Maximum number of data points to use.
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
        columns, ts, by = self._format_colnames(columns, ts, by)
        vpy_plt, kwargs = self._get_plotting_lib(
            matplotlib_kwargs={
                "ax": ax,
                "fixed_xy_lim": fixed_xy_lim,
                "date_in_title": date_in_title,
                "date_f": date_f,
                "date_style_dict": date_style_dict,
                "interval": interval,
                "repeat": repeat,
            },
            style_kwargs=style_kwargs,
        )
        cols = copy.deepcopy(columns)
        if by:
            cols += [by]
        return vpy_plt.AnimatedBarChart(
            vdf=self,
            order_by=ts,
            columns=cols,
            order_by_start=start_date,
            order_by_end=end_date,
            limit_over=limit_over,
            limit=limit,
        ).draw(**kwargs)

    @save_verticapy_logs
    def animated_pie(
        self,
        ts: str,
        columns: SQLColumns,
        by: Optional[str] = None,
        start_date: PythonScalar = None,
        end_date: PythonScalar = None,
        limit_over: int = 6,
        limit: int = 1000000,
        fixed_xy_lim: bool = False,
        date_in_title: bool = False,
        date_f=None,
        date_style_dict: dict = {},
        interval: int = 300,
        repeat: bool = True,
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
    columns: SQLColumns
        List of the vDataColumns names.
    by: str, optional
        Categorical vDataColumn used in the partition.
    start_date: PythonScalar, optional
        Input Start Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is lesser than November 1993 the 3rd.
    end_date: PythonScalar, optional
        Input End Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is greater than November 1993 the 3rd.
    limit_over: int, optional
        Limited number of elements to consider for each category.
    limit: int, optional
        Maximum number of data points to use.
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
        columns, ts, by = self._format_colnames(columns, ts, by)
        vpy_plt, kwargs = self._get_plotting_lib(
            matplotlib_kwargs={
                "ax": ax,
                "fixed_xy_lim": fixed_xy_lim,
                "date_in_title": date_in_title,
                "date_f": date_f,
                "date_style_dict": date_style_dict,
                "interval": interval,
                "repeat": repeat,
            },
            style_kwargs=style_kwargs,
        )
        cols = copy.deepcopy(columns)
        if by:
            cols += [by]
        return vpy_plt.AnimatedPieChart(
            vdf=self,
            order_by=ts,
            columns=cols,
            order_by_start=start_date,
            order_by_end=end_date,
            limit_over=limit_over,
            limit=limit,
        ).draw(**kwargs)

    @save_verticapy_logs
    def animated_plot(
        self,
        ts: str,
        columns: SQLColumns = [],
        by: Optional[str] = None,
        start_date: PythonScalar = None,
        end_date: PythonScalar = None,
        limit_over: int = 6,
        limit: int = 1000000,
        window_size: int = 100,
        step: int = 5,
        fixed_xy_lim: bool = False,
        interval: int = 300,
        repeat: bool = True,
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
    start_date: PythonScalar, optional
        Input Start Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is lesser than November 1993 the 3rd.
    end_date: PythonScalar, optional
        Input End Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is greater than November 1993 the 3rd.
    limit_over: int, optional
        Limited number of elements to consider for each category.
    limit: int, optional
        Maximum number of data points to use.
    step: int, optional
        Number of elements used to update the time series.
    window_size: int, optional
        Size of the window used to draw the time series.
    fixed_xy_lim: bool, optional
        If set to True, the xlim and ylim will be fixed.
    interval: int, optional
        Number of ms between each update.
    repeat: bool, optional
        If set to True, the animation will be repeated.
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
        columns, ts, by = self._format_colnames(columns, ts, by)
        if by:
            if len(columns) != 1:
                raise ValueError(
                    "Parameter columns must include only one element when using parameter 'by'."
                )
            vdf = self.pivot(index=ts, columns=by, values=columns[0])
            columns = vdf.numcol()[0:limit_over]
        else:
            vdf = self
            if not (columns):
                columns = vdf.numcol()
        vpy_plt, kwargs = self._get_plotting_lib(
            matplotlib_kwargs={
                "ax": ax,
                "fixed_xy_lim": fixed_xy_lim,
                "window_size": window_size,
                "step": step,
                "interval": interval,
                "repeat": repeat,
            },
            style_kwargs=style_kwargs,
        )
        return vpy_plt.AnimatedLinePlot(
            vdf=vdf,
            order_by=ts,
            columns=columns,
            order_by_start=start_date,
            order_by_end=end_date,
            limit=limit,
        ).draw(**kwargs)

    @save_verticapy_logs
    def animated_bubble(
        self,
        ts: str,
        columns: SQLColumns,
        by: Optional[str] = None,
        start_date: PythonScalar = None,
        end_date: PythonScalar = None,
        limit_over: int = 6,
        limit: int = 1000000,
        limit_labels: int = 6,
        bbox: list = [],
        img: str = "",
        fixed_xy_lim: bool = False,
        date_in_title: bool = False,
        date_f=None,
        date_style_dict: dict = {},
        interval: int = 300,
        repeat: bool = True,
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
    columns: SQLColumns
        List of the vDataColumns names.
    by: str, optional
        Categorical vDataColumn used in the partition.
    start_date: PythonScalar, optional
        Input Start Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is lesser than November 1993 the 3rd.
    end_date: PythonScalar, optional
        Input End Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is greater than November 1993 the 3rd.
    limit_over: int, optional
        Limited number of elements to consider for each category.
    limit: int, optional
        Maximum number of data points to use.
    limit_labels: int, optional
        Maximum number of text labels to draw.
    img: str, optional
        Path to the image to display as background.
    bbox: list, optional
        List of 4 elements to delimit the boundaries of the final Plot.
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
        if not (
            2 <= len(columns) <= 4
            and self[columns[0]].isnum()
            and self[columns[1]].isnum()
        ):
            raise ValueError(
                f"Parameter 'columns' must include at least 2 numerical vDataColumns"
                " and maximum 4 vDataColumns."
            )
        columns, ts, by = self._format_colnames(columns, ts, by)
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
        vpy_plt, kwargs = self._get_plotting_lib(
            matplotlib_kwargs={
                "ax": ax,
                "img": img,
                "bbox": bbox,
                "fixed_xy_lim": fixed_xy_lim,
                "date_in_title": date_in_title,
                "date_f": date_f,
                "date_style_dict": date_style_dict,
                "interval": interval,
                "repeat": repeat,
            },
            style_kwargs=style_kwargs,
        )
        return vpy_plt.AnimatedBubblePlot().draw(
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
            img=img,
            bbox=bbox,
            fixed_xy_lim=fixed_xy_lim,
            date_in_title=date_in_title,
            date_f=date_f,
            date_style_dict=date_style_dict,
            interval=interval,
            repeat=repeat,
            ax=ax,
            **style_kwargs,
        )
