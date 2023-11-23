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
import copy
from typing import Callable, Optional

from verticapy._typing import (
    PlottingObject,
    PythonScalar,
    SQLColumns,
)
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import format_type

from verticapy.core.vdataframe._plotting import vDFPlot


class vDFAnimatedPlot(vDFPlot):
    # 1D CHARTS.

    @save_verticapy_logs
    def animated_bar(
        self,
        ts: str,
        columns: SQLColumns,
        by: Optional[str] = None,
        start_date: Optional[PythonScalar] = None,
        end_date: Optional[PythonScalar] = None,
        limit_over: int = 6,
        limit: int = 1000000,
        fixed_xy_lim: bool = False,
        date_in_title: bool = False,
        date_f: Optional[Callable] = None,
        date_style_dict: Optional[dict] = None,
        interval: int = 300,
        repeat: bool = True,
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the animated bar chart (bar race).

        Parameters
        ----------
        ts: str
            TS (Time Series) vDataColumn used to order the data.
            The vDataColumn type must be date (date, datetime,
            timestamp...) or numerical.
        columns: SQLColumns
            List of the vDataColumns names.
        by: str, optional
            Categorical vDataColumn used in the partition.
        start_date: PythonScalar, optional
            Input Start Date. For example, time = '03-11-1993' will
            filter the data when  'ts' is less than the 3rd of
            November 1993.
        end_date: PythonScalar, optional
            Input End Date.  For  example, time = '03-11-1993' will
            filter the data when 'ts' is greater than the 3rd of
            November 1993.
        limit_over: int, optional
            Limits the number of elements to consider for each
            category.
        limit: int, optional
            Maximum number of data points to use.
        fixed_xy_lim: bool, optional
            If set to True, the xlim and ylim are fixed.
        date_in_title: bool, optional
            If  set to True, the ts vDataColumn is displayed in
            the title section.
        date_f: function, optional
            Function used to display the ts vDataColumn.
        date_style_dict: dict, optional
            Style Dictionary used to display the ts vDataColumn when
            date_in_title = False.
        interval: int, optional
            Number of ms between each update.
        repeat: bool, optional
            If set to True, the animation is repeated.
        chart: PlottingObject, optional
            The chart object used to plot.
        **style_kwargs
            Any optional parameter to pass to the plotting functions.

        Returns
        -------
        obj
            Plotting Object.

        Examples
        ---------

        .. note::

            The below example is a very basic one. For
            other more detailed examples and customization
            options, please see :ref:`chart_gallery.animated`_

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp
            import verticapy.datasets as vpd

        Let's import the dataset:

        .. ipython:: python

            pop_growth = vpd.load_pop_growth()

        .. code-block:: python

          pop_growth.animated_bar(
            ts = "year",
            columns = ["city", "population"],
            by = "continent",
            start_date = 1970,
            end_date = 1980,
          )

        .. ipython:: python
          :suppress:

          fig = pop_growth.animated_bar(
            ts = "year",
            columns = ["city", "population"],
            by = "continent",
            start_date = 1970,
            end_date = 1980,
          )

          with open("SPHINX_DIRECTORY/figures/code_vdataframe_plotting_animated_bar.html", "w") as file:
            file.write(fig.__html__())

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/code_vdataframe_plotting_animated_bar.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.animated_pie` : Animated Pie Chart.
            | :py:meth:`verticapy.vDataColumn.bar` : Bar Chart.
        """
        columns = format_type(columns, dtype=list)
        date_style_dict = format_type(date_style_dict, dtype=dict)
        columns, ts, by = self.format_colnames(columns, ts, by)
        vpy_plt, kwargs = self.get_plotting_lib(
            class_name="AnimatedBarChart",
            chart=chart,
            matplotlib_kwargs={
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
        start_date: Optional[PythonScalar] = None,
        end_date: Optional[PythonScalar] = None,
        limit_over: int = 6,
        limit: int = 1000000,
        fixed_xy_lim: bool = False,
        date_in_title: bool = False,
        date_f: Optional[Callable] = None,
        date_style_dict: Optional[dict] = None,
        interval: int = 300,
        repeat: bool = True,
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the animated pie chart.

        Parameters
        ----------
        ts: str
            TS (Time Series) vDataColumn used to order the data.
            The vDataColumn type must be date (date, datetime,
            timestamp...) or numerical.
        columns: SQLColumns
            List of the vDataColumns names.
        by: str, optional
            Categorical vDataColumn used in the partition.
        start_date: PythonScalar, optional
            Input Start Date. For example, time = '03-11-1993' will
            filter the data when  'ts' is less than the 3rd of
            November 1993.
        end_date: PythonScalar, optional
            Input End Date.  For  example, time = '03-11-1993' will
            filter the data when 'ts' is greater than the 3rd of
            November 1993.
        limit_over: int, optional
            Limited number of elements to consider for each category.
        limit: int, optional
            Maximum number of data points to use.
        fixed_xy_lim: bool, optional
            If set to True, the xlim and ylim are fixed.
        date_in_title: bool, optional
            If  set to True, the ts vDataColumn is displayed in
            the title section.
        date_f: function, optional
            Function used to display the ts vDataColumn.
        date_style_dict: dict, optional
            Style Dictionary used to display the ts vDataColumn when
            date_in_title = False.
        interval: int, optional
            Number of ms between each update.
        repeat: bool, optional
            If set to True, the animation is repeated.
        chart: PlottingObject, optional
            The chart object used to plot.
        **style_kwargs
            Any optional parameter to pass to the plotting functions.

        Returns
        -------
        obj
            Plotting Object.

        Examples
        ---------

        .. note::

            The below example is a very basic one. For
            other more detailed examples and customization
            options, please see :ref:`chart_gallery.animated`_

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp
            import verticapy.datasets as vpd

        Let's import the dataset:

        .. ipython:: python

            pop_growth = vpd.load_pop_growth()

        .. code-block:: python

          pop_growth.animated_pie(
            ts = "year",
            columns = ["city", "population"],
            by = "continent",
            start_date = 1970,
            end_date = 1980,
          )

        .. ipython:: python
          :suppress:

          fig = pop_growth.animated_pie(
            ts = "year",
            columns = ["city", "population"],
            by = "continent",
            start_date = 1970,
            end_date = 1980,
          )

          with open("SPHINX_DIRECTORY/figures/code_vdataframe_plotting_animated_pie.html", "w") as file:
            file.write(fig.__html__())

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/code_vdataframe_plotting_animated_pie.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.animated_bar` : Animated Bar Chart.
            | :py:meth:`verticapy.vDataColumn.pie` : Pie Chart.
        """
        columns = format_type(columns, dtype=list)
        date_style_dict = format_type(date_style_dict, dtype=dict)
        columns, ts, by = self.format_colnames(columns, ts, by)
        vpy_plt, kwargs = self.get_plotting_lib(
            class_name="AnimatedPieChart",
            chart=chart,
            matplotlib_kwargs={
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

    # Time Series.

    @save_verticapy_logs
    def animated_plot(
        self,
        ts: str,
        columns: Optional[SQLColumns] = None,
        by: Optional[str] = None,
        start_date: Optional[PythonScalar] = None,
        end_date: Optional[PythonScalar] = None,
        limit_over: int = 6,
        limit: int = 1000000,
        window_size: int = 100,
        step: int = 5,
        fixed_xy_lim: bool = False,
        interval: int = 300,
        repeat: bool = True,
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the animated line plot.

        Parameters
        ----------
        ts: str
            TS  (Time Series)  vDataColumn used to order the
            data. The vDataColumn type must be date (date,
            datetime, timestamp...) or numerical.
        columns: SQLColumns, optional
            List of the vDataColumns names.
        by: str, optional
            Categorical  vDataColumn  used  in  the partition.
        start_date: PythonScalar, optional
            Input Start Date. For example, time = '03-11-1993'
            will  filter  the data when  'ts' is less  than
            the 3rd of November 1993.
        end_date: PythonScalar, optional
            Input  End Date. For example,  time = '03-11-1993'
            will  filter  the data when 'ts' is greater  than
            the 3rd of November 1993.
        limit_over: int, optional
            Limited number of  elements to consider for  each
            category.
        limit: int, optional
            Maximum number of data points to use.
        step: int, optional
            Number of elements used to update the time series.
        window_size: int, optional
            Size  of the window used to draw the time  series.
        fixed_xy_lim: bool, optional
            If set  to True, the xlim and ylim are fixed.
        interval: int, optional
            Number of ms between each update.
        repeat: bool, optional
            If set  to  True, the animation is repeated.
        chart: PlottingObject, optional
            The chart object used to plot.
        **style_kwargs
            Any  optional  parameter to pass to the  plotting
            functions.

        Returns
        -------
        obj
            Plotting Object.

        Examples
        ---------

        .. note::

            The below example is a very basic one. For
            other more detailed examples and customization
            options, please see :ref:`chart_gallery.animated`_

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp
            import verticapy.datasets as vpd

        Let's import the dataset:

        .. ipython:: python

            commodities = vpd.load_commodities()

        Now we can finally create the plot:

        .. code-block:: python

          commodities.animated_plot(ts = "date")

        .. ipython:: python
          :suppress:

          fig = commodities.animated_plot(ts = "date")

          with open("SPHINX_DIRECTORY/figures/code_vdataframe_plotting_animated_plot.html", "w") as file:
            file.write(fig.__html__())

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/code_vdataframe_plotting_animated_plot.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.animated_pie` : Animated Pie Chart.
            | :py:meth:`verticapy.vDataFrame.plot` : Line Plot.
        """
        columns = format_type(columns, dtype=list)
        columns, ts, by = self.format_colnames(columns, ts, by)
        if by:
            if len(columns) != 1:
                raise ValueError(
                    "Parameter columns must include only one element"
                    " when using parameter 'by'."
                )
            vdf = self.pivot(index=ts, columns=by, values=columns[0])
            columns = vdf.numcol()[0:limit_over]
        else:
            vdf = self
            if not columns:
                columns = vdf.numcol()
        vpy_plt, kwargs = self.get_plotting_lib(
            class_name="AnimatedLinePlot",
            chart=chart,
            matplotlib_kwargs={
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

    # Scatters.

    @save_verticapy_logs
    def animated_scatter(
        self,
        ts: str,
        columns: SQLColumns,
        by: Optional[str] = None,
        start_date: Optional[PythonScalar] = None,
        end_date: Optional[PythonScalar] = None,
        limit_over: int = 6,
        limit: int = 1000000,
        limit_labels: int = 6,
        bbox: Optional[list] = None,
        img: Optional[str] = None,
        fixed_xy_lim: bool = False,
        date_in_title: bool = False,
        date_f=None,
        date_style_dict: Optional[dict] = None,
        interval: int = 300,
        repeat: bool = True,
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the animated scatter plot.

        Parameters
        ----------
        ts: str
            TS  (Time Series)  vDataColumn used to order the
            data. The vDataColumn type must be date (date,
            datetime, timestamp...) or numerical.
        columns: SQLColumns, optional
            List of the vDataColumns names.
        by: str, optional
            Categorical  vDataColumn  used  in  the partition.
        start_date: PythonScalar, optional
            Input Start Date. For example, time = '03-11-1993'
            will  filter  the data when  'ts' is less  than
            the 3rd of November 1993.
        end_date: PythonScalar, optional
            Input  End Date. For example,  time = '03-11-1993'
            will  filter  the data when 'ts' is greater  than
            the 3rd of November 1993.
        limit_over: int, optional
            Limited number of  elements to consider for  each
            category.
        limit: int, optional
            Maximum   number   of    data   points   to   use.
        limit_labels: int, optional
            Maximum   number   of    text   labels  to   draw.
        img: str, optional
            Path  to  the  image  to  display  as  background.
        bbox: list, optional
            List  of 4 elements to delimit the boundaries  of
            the final Plot.  It must be similar the following
            list: [xmin, xmax, ymin, ymax]
        fixed_xy_lim: bool, optional
            If  set to True, the xlim  and ylim are fixed.
        date_in_title: bool, optional
            If  set  to  True,  the  ts  vDataColumn  is
            displayed in the title section.
        date_f: function, optional
            Function  used  to  display  the  ts  vDataColumn.
        date_style_dict: dict, optional
            Style   Dictionary   used   to   display  the  ts
            vDataColumn when date_in_title = False.
        interval: int, optional
            Number of ms between each update.
        repeat: bool, optional
            If  set to True, the animation is repeated.
        chart: PlottingObject, optional
            The chart object used to plot.
        **style_kwargs
            Any  optional parameter  to  pass to the plotting
            functions.

        Returns
        -------
        obj
            Plotting Object.

        Examples
        ---------

        .. note::

            The below example is a very basic one. For
            other more detailed examples and customization
            options, please see :ref:`chart_gallery.animated`_

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp
            import verticapy.datasets as vpd

        Let's import the dataset:

        .. ipython:: python

            gapminder = vpd.load_gapminder()

        .. code-block:: python

          gapminder.animated_scatter(
            ts = "year",
            columns = ["lifeExp", "gdpPercap"],
            by = "country",
            start_date = 1952,
            end_date = 2000,
          )

        .. ipython:: python
          :suppress:

          fig = gapminder.animated_scatter(
            ts = "year",
            columns = ["lifeExp", "gdpPercap"],
            by = "country",
            start_date = 1952,
            end_date = 2000,
          )

          with open("SPHINX_DIRECTORY/figures/code_vdataframe_plotting_animated_scatter.html", "w") as file:
            file.write(fig.__html__())

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/code_vdataframe_plotting_animated_scatter.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.animated_pie` : Animated Pie Chart.
            | :py:meth:`verticapy.vDataFrame.scatter` : Scatter Plot.
        """
        columns, bbox = format_type(columns, bbox, dtype=list)
        date_style_dict = format_type(date_style_dict, dtype=dict)
        if not (
            2 <= len(columns) <= 4
            and self[columns[0]].isnum()
            and self[columns[1]].isnum()
        ):
            raise ValueError(
                f"Parameter 'columns' must include at least 2 numerical "
                "vDataColumns and maximum 4 vDataColumns."
            )
        columns, ts, by = self.format_colnames(columns, ts, by)
        if len(columns) == 3 and not self[columns[2]].isnum():
            catcol = columns[2]
            columns = columns[0:2]
        elif len(columns) >= 4:
            if not self[columns[3]].isnum():
                catcol = columns[3]
                columns = columns[0:3]
            else:
                catcol = columns[2]
                columns = columns[0:2] + [columns[3]]
        else:
            catcol = None
        vpy_plt, kwargs = self.get_plotting_lib(
            class_name="AnimatedBubblePlot",
            chart=chart,
            matplotlib_kwargs={
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
        return vpy_plt.AnimatedBubblePlot(
            vdf=self,
            order_by=ts,
            columns=columns,
            catcol=catcol,
            by=by,
            order_by_start=start_date,
            order_by_end=end_date,
            limit_over=limit_over,
            limit=limit,
        ).draw(**kwargs)
