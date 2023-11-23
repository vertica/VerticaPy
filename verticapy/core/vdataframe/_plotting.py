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
import math
import warnings
from collections.abc import Iterable
from typing import Callable, Literal, Optional, Union
import numpy as np

import verticapy._config.config as conf
from verticapy._utils._object import get_vertica_mllib
from verticapy._typing import (
    ArrayLike,
    ColorType,
    NoneType,
    PlottingMethod,
    PlottingObject,
    PythonNumber,
    PythonScalar,
    SQLColumns,
)
from verticapy._utils._gen import gen_tmp_name
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import format_type
from verticapy._utils._sql._sys import _executeSQL

from verticapy.core.tablesample.base import TableSample

from verticapy.core.vdataframe._machine_learning import vDFMachineLearning
from verticapy.core.vdataframe._scaler import vDCScaler

from verticapy.plotting.base import PlottingBase


class vDFPlot(vDFMachineLearning):
    # Boxplots.

    @save_verticapy_logs
    def boxplot(
        self,
        columns: Optional[SQLColumns] = None,
        q: tuple[float, float] = (0.25, 0.75),
        max_nb_fliers: int = 30,
        whis: float = 1.5,
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the Box Plot of the input vDataColumns.

        Parameters
        ----------
        columns: SQLColumns, optional
            List  of the vDataColumns names.  If  empty, all
            numerical vDataColumns are used.
        q: tuple, optional
            Tuple including the 2 quantiles used to draw the
            BoxPlot.
        max_nb_fliers: int, optional
            Maximum number of points to use to represent the
            fliers  of each category.  Drawing  fliers  will
            slow down the graphic computation.
        whis: float, optional
            The position of the whiskers.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any  optional parameter to  pass to the plotting
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
            options, please see :ref:`chart_gallery.boxplot`_

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        Let's also import `numpy` to create a dataset.

        .. ipython:: python

            import numpy as np

        We can create a variable ``N`` to fix the size:

        .. ipython:: python

            N = 50

        Let's generate a dataset using the following data.

        .. ipython:: python

            data = vp.vDataFrame(
                {
                    "score1": np.random.normal(5, 1, N),
                    "score2": np.random.normal(8, 1.5, N),
                    "score3": np.random.normal(10, 2, N),
                }
            )

        Below are examples of two types of boxplot:

        - Single (for one column)
        - Multi (for more than one column)

        Check out the tabs below for specific examples.

        .. tab:: Single

            .. code-block:: python

                data.boxplot(["score1"])

            .. ipython:: python
                :suppress:

                vp.set_option("plotting_lib", "plotly")
                fig = data.boxplot(["score1"], width = 600)
                fig.write_html("SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_boxplot_single.html")

            .. raw:: html
                :file: SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_boxplot_single.html

        .. tab:: Multi

            .. code-block:: python

                data.boxplot(columns = ["score1", "score2", "score3"])

            .. ipython:: python
                :suppress:

                vp.set_option("plotting_lib", "plotly")
                fig = data.boxplot(columns = ["score1", "score2", "score3"], width = 600)
                fig.write_html("SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_boxplot_multi.html")

            .. raw:: html
                :file: SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_boxplot_multi.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.outliers_plot` : Outliers Plot.
            | :py:meth:`verticapy.vDataColumn.boxplot` : Box Plot.

        """
        columns = format_type(columns, dtype=list)
        vpy_plt, kwargs = self.get_plotting_lib(
            class_name="BoxPlot",
            chart=chart,
            style_kwargs=style_kwargs,
        )
        return vpy_plt.BoxPlot(
            vdf=self,
            columns=columns,
            q=q,
            whis=whis,
            max_nb_fliers=max_nb_fliers,
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
        kind: Literal["auto", "drilldown", "stacked"] = "auto",
        chart: Optional[PlottingObject] = None,
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
            The method used to aggregate the data.

            - count:
                Number of elements.
            - density:
                Percentage  of  the  distribution.
            - mean:
                Average  of the  vDataColumn ``of``.
            - min:
                Minimum  of the  vDataColumn ``of``.
            - max:
                Maximum  of the  vDataColumn ``of``.
            - sum:
                Sum of the vDataColumn ``of``.
            - q%:
                q Quantile of the vDataColumn ``of``
                (ex: 50% to get the median).

            It can also be a cutomized aggregation, for example:
            AVG(column1) + 5
        of: str, optional
            The  vDataColumn used to compute the  aggregation.
        max_cardinality: tuple, optional
            Maximum number of distinct elements for vDataColumns
            1  and  2  to be used as categorical. For these
            elements, no  h is picked or computed.
        h: tuple, optional
            Interval width of  the vDataColumns 1 and 2 bars.
            Only  valid if the  vDataColumns are  numerical.
            Optimized  h will be  computed  if the parameter  is
            empty or invalid.
        kind: str, optional
            The BarChart Type.

            - auto:
                Regular Bar Chart  based on 1 or 2 vDataColumns.
            - pyramid:
                Pyramid  Density  Bar  Chart. Only works if one
                of the two vDataColumns is binary and the
                'method' is set to 'density'.
            - stacked:
                Stacked  Bar  Chart   based  on  2 vDataColumns.
            - fully_stacked:
                Fully Stacked Bar Chart based on 2 vDataColumns.

        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any  optional  parameter  to  pass  to  the plotting
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
            options, please see :ref:`chart_gallery.bar`_

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        Let's also import `numpy` to create a dataset.

        .. ipython:: python

            import numpy as np

        Let's generate a dataset using the following data.

        .. ipython:: python

            data = vp.vDataFrame(
                {
                    "gender": ['M', 'M', 'M', 'F', 'F', 'F', 'F'],
                    "grade": ['A','B','C','A','B','B', 'B'],
                }
            )

        Below are examples of two types of bar plots:

        - 1D
        - 2D

        .. tab:: 1D

            .. code-block:: python

                data.bar(["grade"])

            .. ipython:: python
                :suppress:

                vp.set_option("plotting_lib", "plotly")
                fig = data.bar(["grade"])
                fig.write_html("SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_bar_1d.html")

            .. raw:: html
                :file: SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_bar_1d.html

        .. tab:: 2D

            .. code-block:: python

                data.bar(columns = ["grade", "gender"])

            .. ipython:: python
                :suppress:

                vp.set_option("plotting_lib", "plotly")
                fig = data.bar(columns = ["grade", "gender"])
                fig.write_html("SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_bar_2d.html")

            .. raw:: html
                :file: SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_bar_2d.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.barh` : Horizontal Bar Chart.
            | :py:meth:`verticapy.vDataColumn.bar` : Bar Chart.
            | :py:meth:`verticapy.vDataColumn.barh` : Horizontal Bar Chart.

        """
        columns = format_type(columns, dtype=list)
        columns, of = self.format_colnames(columns, of, expected_nb_of_cols=[1, 2])
        if not (isinstance(max_cardinality, Iterable)):
            max_cardinality = (max_cardinality, max_cardinality)
        if not (isinstance(h, Iterable)):
            h = (h, h)

        if len(columns) == 1:
            return self[columns[0]].bar(
                method=method,
                of=of,
                max_cardinality=max_cardinality[0],
                h=h[0],
                **style_kwargs,
            )
        elif kind == "drilldown":
            vpy_plt, kwargs = self.get_plotting_lib(
                class_name="DrillDownBarChart",
                chart=chart,
                style_kwargs=style_kwargs,
            )
            return vpy_plt.DrillDownBarChart(
                vdf=self,
                columns=columns,
                method=method,
                of=of,
                h=h,
                max_cardinality=max_cardinality,
            ).draw(**kwargs)
        else:
            vpy_plt, kwargs = self.get_plotting_lib(
                class_name="BarChart2D",
                chart=chart,
                style_kwargs=style_kwargs,
            )
            return vpy_plt.BarChart2D(
                vdf=self,
                columns=columns,
                method=method,
                of=of,
                h=h,
                max_cardinality=max_cardinality,
                misc_layout={"kind": kind},
            ).draw(**kwargs)

    @save_verticapy_logs
    def barh(
        self,
        columns: SQLColumns,
        method: PlottingMethod = "density",
        of: Optional[str] = None,
        max_cardinality: tuple[int, int] = (6, 6),
        h: tuple[PythonNumber, PythonNumber] = (None, None),
        kind: Literal[
            "auto",
            "fully_stacked",
            "stacked",
            "fully",
            "fully stacked",
            "pyramid",
            "density",
        ] = "auto",
        chart: Optional[PlottingObject] = None,
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
            The method used to aggregate the data.

            - count:
                Number of elements.
            - density:
                Percentage  of  the  distribution.
            - mean:
                Average  of the  vDataColumn ``of``.
            - min:
                Minimum  of the  vDataColumn ``of``.
            - max:
                Maximum  of the  vDataColumn ``of``.
            - sum:
                Sum of the vDataColumn ``of``.
            - q%:
                q Quantile of the vDataColumn ``of``
                (ex: 50% to get the median).

            It can also be a cutomized aggregation, for example:
            AVG(column1) + 5
        of: str, optional
            The  vDataColumn used to compute the  aggregation.
        max_cardinality: tuple, optional
            Maximum number of distinct elements for vDataColumns
            1  and  2  to be used as categorical. For these
            elements, no  h is picked or computed.
        h: tuple, optional
            Interval width of  the vDataColumns 1 and 2 bars.
            Only  valid if the  vDataColumns are  numerical.
            Optimized  h will be  computed  if the parameter  is
            empty or invalid.
        kind: str, optional
            The BarChart Type.

            - auto:
                Regular Bar Chart  based on 1 or 2 vDataColumns.
            - pyramid:
                Pyramid  Density  Bar  Chart. Only works if one
                of the two vDataColumns is binary and the
                'method' is set to 'density'.
            - stacked:
                Stacked  Bar  Chart   based  on  2 vDataColumns.
            - fully_stacked:
                Fully Stacked Bar Chart based on 2 vDataColumns.

        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any  optional  parameter  to  pass  to  the plotting
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
            options, please see :ref:`chart_gallery.barh`_

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        Let's also import `numpy` to create a dataset.

        .. ipython:: python

            import numpy as np

        Let's generate a dataset using the following data.

        .. ipython:: python

            data = vp.vDataFrame(
                {
                    "gender": ['M', 'M', 'M', 'F', 'F', 'F', 'F'],
                    "grade": ['A','B','C','A','B','B', 'B'],
                }
            )

        Below are examples of two types of barh plots:

        - 1D
        - 2D

        .. tab:: 1D

            .. code-block:: python

                data.barh(["grade"])

            .. ipython:: python
                :suppress:

                vp.set_option("plotting_lib", "plotly")
                fig = data.barh(["grade"], width = 600)
                fig.write_html("SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_barh_1d.html")

            .. raw:: html
                :file: SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_barh_1d.html

        .. tab:: 2D

            .. code-block:: python

                data.barh(columns = ["grade", "gender"])

            .. ipython:: python
                :suppress:

                vp.set_option("plotting_lib", "plotly")
                fig = data.barh(columns = ["grade", "gender"], width = 600)
                fig.write_html("SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_barh_2d.html")

            .. raw:: html
                :file: SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_barh_2d.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.bar` : Bar Chart.
            | :py:meth:`verticapy.vDataColumn.barh` : Horizontal Bar Chart.
            | :py:meth:`verticapy.vDataColumn.bar` : Bar Chart.


        """
        columns = format_type(columns, dtype=list)
        columns, of = self.format_colnames(columns, of, expected_nb_of_cols=[1, 2])
        if not (isinstance(max_cardinality, Iterable)):
            max_cardinality = (max_cardinality, max_cardinality)
        if not (isinstance(h, Iterable)):
            h = (h, h)

        if len(columns) == 1:
            return self[columns[0]].barh(
                method=method,
                of=of,
                max_cardinality=max_cardinality[0],
                h=h[0],
                chart=chart,
                **style_kwargs,
            )
        elif kind == "drilldown":
            vpy_plt, kwargs = self.get_plotting_lib(
                class_name="DrillDownHorizontalBarChart",
                chart=chart,
                style_kwargs=style_kwargs,
            )
            return vpy_plt.DrillDownHorizontalBarChart(
                vdf=self,
                columns=columns,
                method=method,
                of=of,
                h=h,
                max_cardinality=max_cardinality,
            ).draw(**kwargs)
        else:
            if kind in ("fully", "fully stacked"):
                kind = "fully_stacked"
            elif kind == "pyramid":
                kind = "density"
            vpy_plt, kwargs = self.get_plotting_lib(
                class_name="HorizontalBarChart2D",
                chart=chart,
                style_kwargs=style_kwargs,
            )
            return vpy_plt.HorizontalBarChart2D(
                vdf=self,
                columns=columns,
                method=method,
                of=of,
                max_cardinality=max_cardinality,
                h=h,
                misc_layout={"kind": kind},
            ).draw(**kwargs)

    @save_verticapy_logs
    def pie(
        self,
        columns: SQLColumns,
        max_cardinality: Union[None, int, tuple] = None,
        h: Union[None, int, tuple] = None,
        chart: Optional[PlottingObject] = None,
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
            Maximum number of distinct elements for
            vDataColumns 1  and  2  to be used as
            categorical. For these elements, no  h
            is picked or computed.
            If  of type tuple, represents the
            'max_cardinality' of each column.
        h: int / tuple, optional
            Interval  width  of the bar. If empty,  an
            optimized h will be computed.
            If  of type tuple, it must represent  each
            column's 'h'.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any  optional  parameter  to  pass to  the
            plotting functions.

        Returns
        -------
        obj
            Plotting Object.

        Examples
        ---------

        .. note::

            The below example is a very basic one. For
            other more detailed examples and customization
            options, please see :ref:`chart_gallery.pie`_

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        Let's also import `numpy` to create a dataset.

        .. ipython:: python

            import numpy as np

        Let's generate a dataset using the following data.

        .. ipython:: python

            data = vp.vDataFrame(
                {
                    "gender": ['M', 'M', 'M', 'F', 'F', 'F', 'F'],
                    "grade": ['A','B','C','A','B','B', 'B'],
                }
            )

        Below are examples of two types of pie plots:

        - Regular
        - Nested

        .. tab:: Regular

            .. code-block:: python

                data.pie(["grade"])

            .. ipython:: python
                :suppress:

                vp.set_option("plotting_lib", "plotly")
                fig = data.pie(["grade"], width = 600)
                fig.write_html("SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_pie_1d.html")

            .. raw:: html
                :file: SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_pie_1d.html

        .. tab:: Nested

            .. code-block:: python

                data.pie(columns = ["grade", "gender"])

            .. ipython:: python
                :suppress:

                vp.set_option("plotting_lib", "plotly")
                fig = data.pie(columns = ["grade", "gender"], width = 600)
                fig.write_html("SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_pie_2d.html")

            .. raw:: html
                :file: SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_pie_2d.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.hist` : Histogram.
            | :py:meth:`verticapy.vDataColumn.bar` : Bar Chart.
            | :py:meth:`verticapy.vDataColumn.pie` : Pie Chart.

        """
        vpy_plt, kwargs = self.get_plotting_lib(
            class_name="NestedPieChart",
            chart=chart,
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
        h: Optional[PythonNumber] = None,
        chart: Optional[PlottingObject] = None,
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
            The method used to aggregate the data.

            - count:
                Number of elements.
            - density:
                Percentage  of  the  distribution.
            - mean:
                Average  of the  vDataColumn ``of``.
            - min:
                Minimum  of the  vDataColumn ``of``.
            - max:
                Maximum  of the  vDataColumn ``of``.
            - sum:
                Sum of the vDataColumn ``of``.
            - q%:
                q Quantile of the vDataColumn ``of``
                (ex: 50% to get the median).

            It can also be a cutomized aggregation, for example:
            AVG(column1) + 5
        of: str, optional
            The  vDataColumn used to compute the  aggregation.
        h: tuple, optional
            Interval width of the  input vDataColumns. Optimized
            h  will be  computed if  the  parameter  is empty or
            invalid.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any  optional  parameter  to  pass  to  the plotting
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
            options, please see :ref:`chart_gallery.hist`_

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        Let's also import `numpy` to create a dataset.

        .. ipython:: python

            import numpy as np

        We can create a variable ``N`` to fix the size:

        .. ipython:: python

            N = 50

        Let's generate a dataset using the following data.

        .. ipython:: python

            data = vp.vDataFrame(
                {
                    "score1": np.random.normal(5, 1, N),
                    "score2": np.random.normal(8, 1.5, N),
                    "score3": np.random.normal(10, 2, N),
                }
            )

        Below are examples of two types of hist plots:

        - Single
        - Multi

        .. tab:: Single

            .. code-block:: python

                data.hist(["score1"])

            .. ipython:: python
                :suppress:

                vp.set_option("plotting_lib", "plotly")
                fig = data.hist(["score1"], width = 600)
                fig.write_html("SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_hist_1d.html")

            .. raw:: html
                :file: SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_hist_1d.html

        .. tab:: Multi

            .. code-block:: python

                data.hist(columns = ["score1", "score2"])

            .. ipython:: python
                :suppress:

                vp.set_option("plotting_lib", "plotly")
                fig = data.hist(columns = ["score1", "score2"], width = 600)
                fig.write_html("SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_hist_2d.html")

            .. raw:: html
                :file: SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_hist_2d.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.bar` : Bar Chart.
            | :py:meth:`verticapy.vDataFrame.barh` : Horizontal Bar Chart.
            | :py:meth:`verticapy.vDataColumn.hist` : Histogram.

        """
        vpy_plt, kwargs = self.get_plotting_lib(
            class_name="Histogram",
            chart=chart,
            style_kwargs=style_kwargs,
        )
        return vpy_plt.Histogram(
            vdf=self,
            columns=columns,
            method=method,
            of=of,
            h=h,
        ).draw(**kwargs)

    @save_verticapy_logs
    def density(
        self,
        columns: Optional[SQLColumns] = None,
        bandwidth: float = 1.0,
        kernel: Literal["gaussian", "logistic", "sigmoid", "silverman"] = "gaussian",
        nbins: int = 50,
        xlim: list[tuple[float, float]] = None,
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the vDataColumns Density Plot.

        Parameters
        ----------
        columns: SQLColumns, optional
            List  of the vDataColumns names.  If  empty,
            all numerical vDataColumns are selected.
        bandwidth: float, optional
            The bandwidth of the kernel.
        kernel: str, optional
            The method used for the plot.

            - gaussian:
                Gaussian Kernel.
            - logistic:
                Logistic Kernel.
            - sigmoid:
                Sigmoid Kernel.
            - silverman:
                Silverman Kernel.

        nbins: int, optional
            Maximum  number of  points used to  evaluate
            the approximate density function.
            Increasing  this  parameter increases  the
            precision  but also increases the time of the
            learning and the scoring phases.
        xlim: list of tuple, optional
            Set the x limits of the current axes.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any optional parameter to pass to the plotting
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
            options, please see :ref:`chart_gallery.density`_

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        Let's also import `numpy` to create a dataset.

        .. ipython:: python

            import numpy as np

        We can create a variable ``N`` to fix the size:

        .. ipython:: python

            N = 50

        Let's generate a dataset using the following data.

        .. ipython:: python

            data = vp.vDataFrame(
                {
                    "score1": np.random.normal(5, 1, N),
                    "score2": np.random.normal(8, 1.5, N),
                    "score3": np.random.normal(10, 2, N),
                }
            )

        Below are examples of two types of density plots:

        - Single
        - Multi

        .. tab:: Single

            .. code-block:: python

                data.density(["score1"])

            .. ipython:: python
                :suppress:

                vp.set_option("plotting_lib", "plotly")
                fig = data.density(["score1"], width = 600)
                fig.write_html("SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_density_1d.html")

            .. raw:: html
                :file: SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_density_1d.html

        .. tab:: Multi

            .. code-block:: python

                data.density(columns = ["score1", "score2"])

            .. ipython:: python
                :suppress:

                vp.set_option("plotting_lib", "plotly")
                fig = data.density(columns = ["score1", "score2"], width = 600)
                fig.write_html("SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_density_2d.html")

            .. raw:: html
                :file: SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_density_2d.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.hist` : Histogram.
            | :py:meth:`verticapy.vDataFrame.range_plot` : Range Plot.
            | :py:meth:`verticapy.vDataColumn.density` : Density Plot.

        """
        vml = get_vertica_mllib()
        columns = format_type(columns, dtype=list)
        columns = self.format_colnames(columns)
        if not columns:
            columns = self.numcol()
        if not columns:
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
        model = vml.KernelDensity(
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
                return model.plot(chart=chart, **style_kwargs)
            finally:
                model.drop()
        else:
            X, Y = [], []
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
            vpy_plt, kwargs = self.get_plotting_lib(
                class_name="MultiDensityPlot",
                chart=chart,
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
        columns: Optional[SQLColumns] = None,
        start_date: Optional[PythonScalar] = None,
        end_date: Optional[PythonScalar] = None,
        kind: Literal[
            "area_percent", "area_stacked", "line", "spline", "step"
        ] = "line",
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the time series.

        Parameters
        ----------
        ts: str
            TS (Time Series)  vDataColumn used to order
            the data.  The vDataColumn type must be  date
            (date, datetime, timestamp...) or numerical.
        columns: SQLColumns, optional
            List of the vDataColumns names. If empty, all
            numerical vDataColumns are used.
        start_date: PythonScalar, optional
            Input   Start  Date.  For  example,   time  =
            '03-11-1993'  will  filter the data when 'ts'
            is less than the 3rd of November 1993.
        end_date: PythonScalar, optional
            Input   End   Date.   For   example,   time =
            '03-11-1993'   will  filter  the  data   when
            'ts' is greater than the 3rd of November 1993.
        kind: str, optional
            The plot type.

            - line:
                Line Plot.
            - spline:
                Spline Plot.
            - step:
                Step Plot.
            - area_stacked:
                Stacked Area Plot.
            - area_percent:
                Fully Stacked Area Plot.

        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any   optional  parameter  to   pass  to  the
            plotting functions.

        Returns
        -------
        obj
            Plotting Object.

        Examples
        ---------

        .. note::

            The below example is a very basic one. For
            other more detailed examples and customization
            options, please see :ref:`chart_gallery.line`_

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        Let's also import `numpy` to create a dataset.

        .. ipython:: python

            import numpy as np

        Let's generate a dataset using the following data.

        .. ipython:: python

            data = vp.vDataFrame(
                {
                    "date": [1900, 1950, 2000],
                    "Asia": [947, 1402, 3634],
                    "Africa": [133, 221, 767],
                    "Europe": [408, 547, 729],
                    "America": [156, 339, 818],
                    "Oceania": [6, 13, 30],
                }
            )

        Below are examples of two types of plot plots:

        - Single
        - Multi

        .. tab:: Single

            .. code-block:: python

                data.plot(columns = ["Asia"], ts = "date", kind = "spline")

            .. ipython:: python
                :suppress:

                vp.set_option("plotting_lib", "plotly")
                fig = data.plot(columns = ["Asia"], ts = "date", kind = "spline", width = 600)
                fig.write_html("SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_plot_1d.html")

            .. raw:: html
                :file: SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_plot_1d.html

        .. tab:: Multi

            .. code-block:: python

                data.plot(columns = ["Asia", "Africa", "Europe", "America", "Oceania"], ts = "date")

            .. ipython:: python
                :suppress:

                vp.set_option("plotting_lib", "plotly")
                fig = data.plot(columns = ["Asia", "Africa", "Europe", "America", "Oceania"], ts = "date", width = 600)
                fig.write_html("SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_plot_2d.html")

            .. raw:: html
                :file: SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_plot_2d.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.range_plot` : Range Plot.
            | :py:meth:`verticapy.vDataColumn.plot` : Line Plot.

        """
        columns = format_type(columns, dtype=list)
        vpy_plt, kwargs = self.get_plotting_lib(
            class_name="MultiLinePlot",
            chart=chart,
            style_kwargs=style_kwargs,
        )
        return vpy_plt.MultiLinePlot(
            vdf=self,
            order_by=ts,
            columns=columns,
            order_by_start=start_date,
            order_by_end=end_date,
            misc_layout={"kind": kind},
        ).draw(**kwargs)

    @save_verticapy_logs
    def range_plot(
        self,
        columns: SQLColumns,
        ts: str,
        q: tuple[float, float] = (0.25, 0.75),
        start_date: Optional[PythonScalar] = None,
        end_date: Optional[PythonScalar] = None,
        plot_median: bool = False,
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the range plot of the input vDataColumns. The
        aggregations used to draw the plot are the median
        and the two user-specified quantiles.

        Parameters
        ----------
        columns: SQLColumns
            List of vDataColumns names.
        ts: str
            TS (Time Series) vDataColumn used to order the
            data.  The  vDataColumn  type must be date
            (date, datetime, timestamp...) or numerical.
        q: tuple, optional
            Tuple that includes the 2 quantiles used to draw
            the Plot.
        start_date: str / PythonNumber / date, optional
            Input Start Date. For example, time = '03-11-1993'
            will  filter  the data when 'ts' is  less  than
            the 3rd of November 1993.
        end_date: str / PythonNumber / date, optional
            Input End Date.  For example, time = '03-11-1993'
            will  filter the  data when 'ts' is greater than
            the 3rd of November 1993.
        plot_median: bool, optional
            If set to True, the Median is drawn.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any  optional parameter to pass to the  plotting
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
            options, please see :ref:`chart_gallery.range`_

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        Let's also import `numpy` to create a dataset.

        .. ipython:: python

            import numpy as np

        We can create a variable ``N`` to fix the size:

        .. ipython:: python

            N = 30

        Let's generate a dataset using the following data.

        .. ipython:: python

            data = vp.vDataFrame(
                {
                    "date": [1990 + i for i in range(N)] * 5,
                    "population1": [100 + i for i in range(N)] + [300 + i * 2 for i in range(N)] + [200 + i ** 2 - 3 * i for i in range(N)] + [50 + i ** 2 - 6 * i for i in range(N)] + [700 + i ** 2 - 10 * i for i in range(N)],
                    "population2": [200 + i ** 2 - i for i in range(N)] + [1000 + i * 2 for i in range(N)] + [500 + i ** 2 - 5 * i for i in range(N)] + [900 + i ** 2 + 3 * i for i in range(N)] + [100 + i ** 2 - 0.5 * i for i in range(N)],
                }
            )

        Below are examples of two types of range_plot plots:

        - Single
        - Multi

        .. tab:: Single

            .. code-block:: python

                data.range_plot(columns = ["population1", "population2"], ts = "date")

            .. ipython:: python
                :suppress:

                vp.set_option("plotting_lib", "plotly")
                fig = data.range_plot(columns = ["population1"], ts = "date", width = 600)
                fig.write_html("SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_range_plot_1d.html")

            .. raw:: html
                :file: SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_range_plot_1d.html

        .. tab:: Multi

            .. code-block:: python

                data.range_plot(columns = ["population1", "population2"], ts = "date")

            .. ipython:: python
                :suppress:

                vp.set_option("plotting_lib", "plotly")
                fig = data.range_plot(columns = ["population1", "population2"], ts = "date", width = 600)
                fig.write_html("SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_range_plot_2d.html")

            .. raw:: html
                :file: SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_range_plot_2d.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.plot` : Line Plot.
            | :py:meth:`verticapy.vDataColumn.range_plot` : Range Plot.

        """
        vpy_plt, kwargs = self.get_plotting_lib(
            class_name="RangeCurve",
            chart=chart,
            matplotlib_kwargs={
                "plot_median": plot_median,
            },
            plotly_kwargs={
                "plot_median": plot_median,
            },
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
            The method used to aggregate the data.

            - count:
                Number of elements.
            - density:
                Percentage  of  the  distribution.
            - mean:
                Average  of the  vDataColumn ``of``.
            - min:
                Minimum  of the  vDataColumn ``of``.
            - max:
                Maximum  of the  vDataColumn ``of``.
            - sum:
                Sum of the vDataColumn ``of``.
            - q%:
                q Quantile of the vDataColumn ``of``
                (ex: 50% to get the median).

            It can also be a cutomized aggregation,
            (ex: AVG(column1) + 5).
        of: str, optional
            The vDataColumn used to compute the aggregation.
        max_cardinality: tuple, optional
            Maximum number of distinct elements for vDataColumns
            1  and  2  to be used as categorical. For these
            elements, no  h is picked or computed.
        h: tuple, optional
            Interval width of the vDataColumns 1 and 2 bars.
            Only valid if  the  vDataColumns  are numerical.
            Optimized h will be computed if the parameter is
            empty or invalid.
        fill_none: float, optional
            The  empty  values  of the pivot table are
            filled by this number.

        Returns
        -------
        obj
            TableSample.

        """
        columns = format_type(columns, dtype=list)
        columns, of = self.format_colnames(columns, of, expected_nb_of_cols=[1, 2])
        vpy_plt = self.get_plotting_lib(class_name="HeatMap")[0]
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
        mround: int = 3,
        with_numbers: bool = True,
        chart: Optional[PlottingObject] = None,
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
            The method used to aggregate the data.

            - count:
                Number of elements.
            - density:
                Percentage  of  the  distribution.
            - mean:
                Average  of the  vDataColumn ``of``.
            - min:
                Minimum  of the  vDataColumn ``of``.
            - max:
                Maximum  of the  vDataColumn ``of``.
            - sum:
                Sum of the vDataColumn ``of``.
            - q%:
                q Quantile of the vDataColumn ``of``
                (ex: 50% to get the median).

            It can also be a cutomized aggregation
            (ex: AVG(column1) + 5).
        of: str, optional
            The vDataColumn used to compute the aggregation.
        max_cardinality: tuple, optional
            Maximum number of distinct elements for vDataColumns
            1  and  2  to be used as categorical. For these
            elements, no  h is picked or computed.
        h: tuple, optional
            Interval width of the vDataColumns 1 and 2 bars.
            Only valid if the vDataColumns  are numerical.
            Optimized h will be computed if the parameter is
            empty or invalid.
        fill_none: float, optional
            The  empty  values  of the pivot table  are
            filled by this number.
        mround: int, optional
            Rounds the coefficient using the input number of
            digits.  It  is only  used to display the  final
            pivot table.
        with_numbers: bool, optional
            If  set to True, no number is displayed in
            the final drawing.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any  optional parameter to pass to the  plotting
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
            options, please see :ref:`chart_gallery.pivot`_

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        Let's also import `numpy` to create a dataset.

        .. ipython:: python

            import numpy as np

        We can create a variable ``N`` to fix the size:

        .. ipython:: python

            N = 30

        Let's generate a dataset using the following data.

        .. ipython:: python

            data = vp.vDataFrame(
                {
                    "category1": [np.random.choice(['A','B','C']) for _ in range(N)],
                    "category2": [np.random.choice(['D','E']) for _ in range(N)],
                }
            )

        Below are examples of one types of pivot_table plots:

        - Pivot Plot

        .. tab:: Pivot Plot

            .. code-block:: python

                data.pivot_table(columns = ["category1", "category2"])

            .. ipython:: python
                :suppress:

                vp.set_option("plotting_lib", "plotly")
                fig = data.pivot_table(columns = ["category1", "category2"])
                fig.write_html("SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_pivot_table_1d.html")

            .. raw:: html
                :file: SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_pivot_table_1d.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.contour` : Contour Plot.
            | :py:meth:`verticapy.vDataFrame.scatter_matrix` : Scatter Matrix.

        """
        columns = format_type(columns, dtype=list)
        columns, of = self.format_colnames(columns, of, expected_nb_of_cols=[1, 2])
        vpy_plt, kwargs = self.get_plotting_lib(
            class_name="HeatMap",
            chart=chart,
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
            misc_layout={
                "mround": mround,
                "with_numbers": with_numbers,
            },
        ).draw(**kwargs)

    @save_verticapy_logs
    def contour(
        self,
        columns: SQLColumns,
        func: Union[Callable, str],
        nbins: int = 100,
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws  the  contour  plot of the input function
        using two input vDataColumns.

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
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any  optional parameter to pass to  the plotting
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
            options, please see :ref:`chart_gallery.contour`_

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        Let's also import `numpy` to create a dataset.

        .. ipython:: python

            import numpy as np

        We can create a variable ``N`` to fix the size:

        .. ipython:: python

            N = 30

        For contour plots, we also need a function to apply:

        .. ipython:: python

            def f(x, y):
                return x ** 2 - y + 1

        Let's generate a dataset using the following data.

        .. ipython:: python

            data = vp.vDataFrame(
                {
                    "x": np.random.normal(5, 1, N),
                    "y": np.random.normal(8, 1.5, N),
                }
            )

        Below is an examples of one type of contour plots:

        - Contour Plot

        .. tab:: Contour Plot

            .. code-block:: python

                data.contour(columns = ["x", "y"], func = f)

            .. ipython:: python
                :suppress:

                vp.set_option("plotting_lib", "plotly")
                fig = data.contour(columns = ["x", "y"], func = f)
                fig.write_html("SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_contour_1d.html")

            .. raw:: html
                :file: SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_contour_1d.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.hexbin` : Hexbin Plot.
            | :py:meth:`verticapy.vDataFrame.heatmap` : Heatmap.

        """
        vpy_plt, kwargs = self.get_plotting_lib(
            class_name="ContourPlot",
            chart=chart,
            style_kwargs=style_kwargs,
        )
        func_name = None
        if "func_name" in kwargs:
            func_name = kwargs["func_name"]
            del kwargs["func_name"]
        return vpy_plt.ContourPlot(
            vdf=self,
            columns=columns,
            func=func,
            nbins=nbins,
            func_name=func_name,
        ).draw(**kwargs)

    @save_verticapy_logs
    def heatmap(
        self,
        columns: SQLColumns,
        method: PlottingMethod = "count",
        of: Optional[str] = None,
        h: tuple = (None, None),
        chart: Optional[PlottingObject] = None,
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
            The method used to aggregate the data.

            - count:
                Number of elements.
            - density:
                Percentage  of  the  distribution.
            - mean:
                Average  of the  vDataColumn ``of``.
            - min:
                Minimum  of the  vDataColumn ``of``.
            - max:
                Maximum  of the  vDataColumn ``of``.
            - sum:
                Sum of the vDataColumn ``of``.
            - q%:
                q Quantile of the vDataColumn ``of``
                (ex: 50% to get the median).

            It can also be a cutomized aggregation
            (ex: AVG(column1) + 5).
        of: str, optional
            The vDataColumn used to compute the aggregation.
        h: tuple, optional
            Interval width  of  the vDataColumns 1  and  2
            bars.  Optimized  h  will  be computed if  the
            parameter is empty or invalid.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any optional parameter to pass to the plotting
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
            options, please see :ref:`chart_gallery`_

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        Let's also import `numpy` to create a dataset.

        .. ipython:: python

            import numpy as np

        We can create a variable ``N`` to fix the size:

        .. ipython:: python

            N = 30

        Let's generate a dataset using the following data.

        .. ipython:: python

            data = vp.vDataFrame(
                {
                    "x": np.random.normal(5, 1, N),
                    "y": np.random.normal(8, 1.5, N),
                }
            )

        Below is an examples of one type of heatmap plots:

        - Heatmap

        .. tab:: Heatmap

            .. code-block:: python

                data.heatmap(columns = ["x", "y"])

            .. ipython:: python
                :suppress:

                vp.set_option("plotting_lib", "plotly")
                fig = data.heatmap(columns = ["x", "y"])
                fig.write_html("SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_heatmap_1d.html")

            .. raw:: html
                :file: SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_heatmap_1d.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.hexbin` : Hexbin Plot.
            | :py:meth:`verticapy.vDataFrame.contour` : Contour Plot.

        """
        columns = format_type(columns, dtype=list)
        columns, of = self.format_colnames(columns, of, expected_nb_of_cols=2)
        for column in columns:
            assert self[column].isnum(), TypeError(
                f"vDataColumn {column} must be numerical to draw the Heatmap."
            )
        min_max = self.agg(func=["min", "max"], columns=columns).transpose()
        vpy_plt, kwargs = self.get_plotting_lib(
            class_name="HeatMap",
            chart=chart,
            matplotlib_kwargs={
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
            fill_none=0.0,
            misc_layout={
                "with_numbers": False,
            },
        ).draw(**kwargs)

    @save_verticapy_logs
    def hexbin(
        self,
        columns: SQLColumns,
        method: PlottingMethod = "count",
        of: Optional[str] = None,
        bbox: Optional[list] = None,
        img: Optional[str] = None,
        chart: Optional[PlottingObject] = None,
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
            The method used to aggregate the data.

            - count:
                Number of elements.
            - density:
                Percentage  of  the  distribution.
            - mean:
                Average  of the  vDataColumn ``of``.
            - min:
                Minimum  of the  vDataColumn ``of``.
            - max:
                Maximum  of the  vDataColumn ``of``.
            - sum:
                Sum of the vDataColumn ``of``.
            - q%:
                q Quantile of the vDataColumn ``of``
                (ex: 50% to get the median).

        of: str, optional
            The vDataColumn used to compute the aggregation.
        bbox: list, optional
            List of 4 elements  to delimit the boundaries of
            the final Plot. It must be similar the following
            list: [xmin, xmax, ymin, ymax]
        img: str, optional
            Path  to the  image used as a background.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any  optional parameter to pass to the  plotting
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
            options, please see :ref:`chart_gallery`_

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        Let's also import `numpy` to create a dataset.

        .. ipython:: python

            import numpy as np

        We can create a variable ``N`` to fix the size:

        .. ipython:: python

            N = 30

        Let's generate a dataset using the following data.

        .. ipython:: python

            data = vp.vDataFrame(
                {
                    "x": np.random.normal(5, 1, N),
                    "y": np.random.normal(8, 1.5, N),
                }
            )

        Below is an examples of one type of hexbin plots:

        - Hexbin

        .. tab:: Hexbin

            .. ipython:: python

                @suppress
                vp.set_option("plotting_lib", "matplotlib")

                @savefig core_vdataframe_plotting_vdf_hexbin_1.png
                data.hexbin(columns = ["x", "y"])

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.heatmap` : Heatmap.
            | :py:meth:`verticapy.vDataFrame.contour` : Contour Plot.

        """
        columns, bbox = format_type(columns, bbox, dtype=list)
        columns, of = self.format_colnames(columns, of, expected_nb_of_cols=2)
        vpy_plt, kwargs = self.get_plotting_lib(
            class_name="HexbinMap",
            chart=chart,
            matplotlib_kwargs={"bbox": bbox, "img": img},
            style_kwargs=style_kwargs,
        )
        return vpy_plt.HexbinMap(
            vdf=self,
            columns=columns,
            method=method,
            of=of,
        ).draw(**kwargs)

    # Scatters.

    @save_verticapy_logs
    def scatter(
        self,
        columns: SQLColumns,
        by: Optional[str] = None,
        size: Optional[str] = None,
        cmap_col: Optional[str] = None,
        max_cardinality: int = 6,
        cat_priority: Union[None, PythonScalar, ArrayLike] = None,
        max_nb_points: int = 20000,
        dimensions: tuple = None,
        bbox: Optional[tuple] = None,
        img: Optional[str] = None,
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the scatter plot of the input vDataColumns.

        Parameters
        ----------
        columns: SQLColumns
            List of the vDataColumns names.
        by: str, optional
            Categorical vDataColumn used to label the data.
        size: str
            Numerical  vDataColumn used to represent  the
            Bubble size.
        cmap_col: str, optional
            Numerical  column used  to represent the  color
            map.
        max_cardinality: int, optional
            Maximum  number  of  distinct elements for  'by'
            to  be  used as categorical.  The less  frequent
            elements are gathered together  to create a
            new category: 'Others'.
        cat_priority: PythonScalar / ArrayLike, optional
            ArrayLike list of the different categories to
            consider when  labeling  the  data using  the
            vDataColumn 'by'.  The  other  categories  are
            filtered.
        max_nb_points: int, optional
            Maximum number of points to display.
        dimensions: tuple, optional
            Tuple of two  elements representing the IDs of the
            PCA's components. If empty and the number of input
            columns  is greater  than 3, the first and  second
            PCA are drawn.
        bbox: list, optional
            Tuple  of 4 elements to delimit the boundaries  of
            the  final Plot. It must be similar the  following
            list: [xmin, xmax, ymin, ymax]
        img: str, optional
            Path to the image to display as background.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any  optional  parameter  to pass to the  plotting
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
            options, please see :ref:`chart_gallery.scatter`_

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        Let's also import `numpy` to create a dataset.

        .. ipython:: python

            import numpy as np

        We can create a variable ``N`` to fix the size:

        .. ipython:: python

            N = 30

        Let's generate a dataset using the following data.

        .. ipython:: python

            data = vp.vDataFrame(
                {
                    "category": [np.random.choice(['A','B','C']) for _ in range(N)],
                    "x": np.random.normal(5, 1, N),
                    "y": np.random.normal(8, 1.5, N),
                    "z": np.random.normal(10, 2, N),
                }
            )

        Below are examples of two types of scatter plots:

        - 2D
        - 3D

        .. tab:: 2D

            .. code-block:: python

                data.scatter(columns = ["x", "y"], by = "category")

            .. ipython:: python
                :suppress:

                vp.set_option("plotting_lib", "plotly")
                fig = data.scatter(columns = ["x", "y"], by = "category")
                fig.write_html("SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_scatter_2d.html")

            .. raw:: html
                :file: SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_scatter_2d.html

        .. tab:: 3D

            .. code-block:: python

                data.scatter(columns = ["x", "y", "z"])

            .. ipython:: python
                :suppress:

                vp.set_option("plotting_lib", "plotly")
                fig = data.scatter(columns = ["x", "y", "z"])
                fig.write_html("SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_scatter_3d.html")

            .. raw:: html
                :file: SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_scatter_3d.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.density` : Density Plot.
            | :py:meth:`verticapy.vDataFrame.outliers_plot` : Outliers Plot.

        """
        vml = get_vertica_mllib()
        if img and not bbox and len(columns) == 2:
            aggr = self.agg(columns=columns, func=["min", "max"])
            bbox = (
                aggr.values["min"][0],
                aggr.values["max"][0],
                aggr.values["min"][1],
                aggr.values["max"][1],
            )
        if len(columns) > 3 and isinstance(dimensions, NoneType):
            dimensions = (1, 2)
        if isinstance(dimensions, Iterable):
            model_name = gen_tmp_name(
                schema=conf.get_option("temp_schema"), name="pca_plot"
            )
            model = vml.PCA(model_name)
            model.drop()
            try:
                model.fit(self, columns)
                vdf = model.transform(self)
                ev_1 = model.explained_variance_[dimensions[0] - 1]
                x_label = f"Dim{dimensions[0]} ({ev_1}%)"
                ev_2 = model.explained_variance_[dimensions[1] - 1]
                y_label = f"Dim{dimensions[1]} ({ev_2}%)"
                vdf[f"col{dimensions[0]}"].rename(x_label)
                vdf[f"col{dimensions[1]}"].rename(y_label)
                chart = vdf.scatter(
                    columns=[x_label, y_label],
                    by=by,
                    cmap_col=cmap_col,
                    size=size,
                    max_cardinality=max_cardinality,
                    cat_priority=cat_priority,
                    max_nb_points=max_nb_points,
                    bbox=bbox,
                    img=img,
                    chart=chart,
                    **style_kwargs,
                )
            finally:
                model.drop()
            return chart
        vpy_plt, kwargs = self.get_plotting_lib(
            class_name="ScatterPlot",
            chart=chart,
            matplotlib_kwargs={
                "bbox": bbox,
                "img": img,
            },
            style_kwargs=style_kwargs,
        )
        return vpy_plt.ScatterPlot(
            vdf=self,
            columns=columns,
            by=by,
            cmap_col=cmap_col,
            size=size,
            max_cardinality=max_cardinality,
            cat_priority=cat_priority,
            max_nb_points=max_nb_points,
        ).draw(**kwargs)

    @save_verticapy_logs
    def scatter_matrix(
        self,
        columns: Optional[SQLColumns] = None,
        max_nb_points: int = 1000,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the scatter matrix of the vDataFrame.

        Parameters
        ----------
        columns: SQLColumns, optional
            List of the vDataColumns names. If empty,
            all numerical  vDataColumns are used.
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

        Examples
        ---------

        .. note::

            The below example is a very basic one. For
            other more detailed examples and customization
            options, please see :ref:`chart_gallery`_

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        Let's also import `numpy` to create a dataset.

        .. ipython:: python

            import numpy as np

        We can create a variable ``N`` to fix the size:

        .. ipython:: python

            N = 30

        Let's generate a dataset using the following data.

        .. ipython:: python

            data = vp.vDataFrame(
                {
                    "x": np.random.normal(5, 1, N),
                    "y": np.random.normal(8, 1.5, N),
                }
            )

        Below is an examples of one type of scatter_matrix plots:

        - Scatter Matrix

        .. tab:: Scatter Matrix

            .. ipython:: python

                @suppress
                vp.set_option("plotting_lib", "matplotlib")

                @savefig core_vdataframe_plotting_vdf_scatter_matrix.png
                data.scatter_matrix(columns = ["x", "y"])

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.contour` : Contour Plot.
            | :py:meth:`verticapy.vDataFrame.heatmap` : Heatmap.

        """
        columns = format_type(columns, dtype=list)
        columns = self.format_colnames(columns)
        vpy_plt, kwargs = self.get_plotting_lib(
            class_name="ScatterMatrix",
            style_kwargs=style_kwargs,
        )
        return vpy_plt.ScatterMatrix(
            vdf=self, columns=columns, max_nb_points=max_nb_points
        ).draw(**kwargs)

    @save_verticapy_logs
    def outliers_plot(
        self,
        columns: SQLColumns,
        threshold: float = 3.0,
        max_nb_points: int = 500,
        color: ColorType = "orange",
        outliers_color: ColorType = "black",
        inliers_color: ColorType = "white",
        inliers_border_color: ColorType = "red",
        chart: Optional[PlottingObject] = None,
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
        max_nb_points: int, optional
            Maximum number of points to display.
        color: ColorType, optional
            Inliers Area color.
        outliers_color: ColorType, optional
            Outliers color.
        inliers_color: ColorType, optional
            Inliers color.
        inliers_border_color: ColorType, optional
            Inliers border color.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any  optional  parameter to pass to  the
            plotting functions.

        Returns
        -------
        obj
            Plotting Object.

        Examples
        ---------

        .. note::

            The below example is a very basic one. For
            other more detailed examples and customization
            options, please see :ref:`chart_gallery.outliers`_

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        Let's also import `numpy` to create a dataset.

        .. ipython:: python

            import numpy as np

        We can create a variable ``N`` to fix the size:

        .. ipython:: python

            N = 30

        Let's generate a dataset using the following data.

        .. ipython:: python

            # Normal Distributions
            x = np.random.normal(5, 1, round(N / 2))
            y = np.random.normal(3, 1, round(N / 2))

            # Creating a vDataFrame with a few outliers
            data = vp.vDataFrame(
                {
                    "x": np.concatenate([x, [15]]),
                    "y": np.concatenate([y, [12]]),
                }
            )

        Below are examples of two types of outliers_plot plots:

        - 1D
        - 2D

        .. tab:: 1D

            .. code-block:: python

                data.outliers_plot(columns = ["x"])

            .. ipython:: python
                :suppress:

                vp.set_option("plotting_lib", "plotly")
                fig = data.outliers_plot(columns = ["x"])
                fig.write_html("SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_outliers_plot_1d.html")

            .. raw:: html
                :file: SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_outliers_plot_1d.html

        .. tab:: 2D

            .. code-block:: python

                data.outliers_plot(columns = ["x", "y"])

            .. ipython:: python
                :suppress:

                vp.set_option("plotting_lib", "plotly")
                fig = data.outliers_plot(columns = ["x", "y"])
                fig.write_html("SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_outliers_plot_2d.html")

            .. raw:: html
                :file: SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdf_outliers_plot_2d.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.scatter` : Scatter Plot.
            | :py:meth:`verticapy.vDataFrame.boxplot` : Box Plot.

        """
        vpy_plt, kwargs = self.get_plotting_lib(
            class_name="OutliersPlot",
            chart=chart,
            style_kwargs=style_kwargs,
        )
        return vpy_plt.OutliersPlot(
            vdf=self,
            columns=columns,
            threshold=threshold,
            max_nb_points=max_nb_points,
            misc_layout={
                "color": color,
                "outliers_color": outliers_color,
                "inliers_color": inliers_color,
                "inliers_border_color": inliers_border_color,
            },
        ).draw(**kwargs)


class vDCPlot(vDCScaler):
    # Special Methods.

    def numh(
        self, method: Literal["sturges", "freedman_diaconis", "fd", "auto"] = "auto"
    ) -> float:
        """
        Computes the optimal vDataColumn bar width.

        Parameters
        ----------
        method: str, optional
            Method used to compute the optimal h.
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
        assert self.isnum() or self.isdate(), ValueError(
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
                    /*+LABEL('vDataColumn.numh')*/ COUNT({self}) AS NAs, 
                    MIN({self}) AS min, 
                    APPROXIMATE_PERCENTILE({self} 
                        USING PARAMETERS percentile = 0.25) AS Q1, 
                    APPROXIMATE_PERCENTILE({self} 
                        USING PARAMETERS percentile = 0.75) AS Q3, 
                    MAX({self}) AS max 
                FROM 
                    (SELECT 
                        DATEDIFF('second', 
                                 '{self.min()}'::timestamp, 
                                 {self}) AS {self} 
                    FROM {self._parent}) VERTICAPY_OPTIMAL_H_TABLE""",
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
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the box plot of the vDataColumn.

        Parameters
        ----------
        by: str, optional
            vDataColumn used to partition  the  data.
        q: tuple, optional
            Tuple including the 2 quantiles used to draw
            the BoxPlot.
        h: PythonNumber, optional
            Interval  width  if  the 'by'  vDataColumn is
            numerical or of a date-like type. Optimized h
            will be computed if the parameter is empty or
            invalid.
        max_cardinality: int, optional
            Maximum   number   of   distinct vDataColumn
            elements to be used as categorical.
            The less frequent  elements  are gathered
            together to create a new category : 'Others'.
        cat_priority: PythonScalar / ArrayLike, optional
            ArrayLike list of the different categories to
            consider when drawing the box plot. The other
            categories are filtered.
        max_nb_fliers: int, optional
            Maximum  number of points used to represent
            the fliers of each category.
            Drawing fliers slows down the  graphic
            computation.
        whis: float, optional
            The position of the whiskers.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any  optional  parameter  to  pass to the  plotting
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
            options, please see :ref:`chart_gallery.boxplot`_

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        Let's also import `numpy` to create a dataset.

        .. ipython:: python

            import numpy as np

        We can create a variable ``N`` to fix the size:

        .. ipython:: python

            N = 50

        Let's generate a dataset using the following data.

        .. ipython:: python

            data = vp.vDataFrame(
                {
                    "score1": np.random.normal(5, 1, N),
                }
            )

        Now we are ready to draw the plot:

        .. code-block:: python

            data["score1"].boxplot()

        .. ipython:: python
            :suppress:

            vp.set_option("plotting_lib", "plotly")
            fig = data["score1"].boxplot(width = 600)
            fig.write_html("SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdc_boxplot_single.html")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdc_boxplot_single.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.boxplot` : Box Plot.
            | :py:meth:`verticapy.vDataColumn.outliers_plot` : Outliers Plot.

        """
        vpy_plt, kwargs = self._parent.get_plotting_lib(
            class_name="BoxPlot",
            chart=chart,
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
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the bar chart of the vDataColumn based on an
        aggregation.

        Parameters
        ----------
        method: str, optional
            The method used to aggregate the data.

            - count:
                Number of elements.
            - density:
                Percentage  of  the  distribution.
            - mean:
                Average  of the  vDataColumn ``of``.
            - min:
                Minimum  of the  vDataColumn ``of``.
            - max:
                Maximum  of the  vDataColumn ``of``.
            - sum:
                Sum of the vDataColumn ``of``.
            - q%:
                q Quantile of the vDataColumn ``of``
                (ex: 50% to get the median).

            It can also be a cutomized aggregation
            (ex: AVG(column1) + 5).
        of: str, optional
            The vDataColumn  used to compute the aggregation.
        max_cardinality: int, optional
            Maximum number of distinct vDataColumns elements
            to be used as categorical. For these elements, no
            h is picked or computed.
        nbins: int, optional
            Number  of  bins. If empty, an  optimized number of
            bins is computed.
        h: PythonNumber, optional
            Interval width of the bar. If empty, an optimized h
            is computed.
        categorical: bool, optional
            If  set to False and the  vDataColumn is numerical,
            the parmater  'max_cardinality' is ignored and
            the bar  chart is represented as a histogram.
        bargap: float, optional
            A float between  (0, 1] that represents the
            proportion  taken out of each bar to render the
            chart. This proportion creates gaps between each
            bar. The bigger the value, the bigger the gap.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any  optional  parameter  to  pass to the  plotting
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
            options, please see :ref:`chart_gallery.bar`_

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        Let's also import `numpy` to create a dataset.

        .. ipython:: python

            import numpy as np

        Let's generate a dataset using the following data.

        .. ipython:: python

            data = vp.vDataFrame(
                {
                    "gender": ['M', 'M', 'M', 'F', 'F', 'F', 'F'],
                    "grade": ['A','B','C','A','B','B', 'B'],
                }
            )

        Now we are ready to draw the plot:

        .. code-block:: python

            data["grade"].bar()

        .. ipython:: python
            :suppress:

            vp.set_option("plotting_lib", "plotly")
            fig = data["grade"].bar()
            fig.write_html("SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdc_bar_1d.html")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdc_bar_1d.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.bar` : Bar Chart.
            | :py:meth:`verticapy.vDataFrame.barh` : Horizontal Bar Chart.
            | :py:meth:`verticapy.vDataColumn.barh` : Horizontal Bar Chart.

        """
        vpy_plt, kwargs = self._parent.get_plotting_lib(
            class_name="BarChart",
            chart=chart,
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
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws  the horizontal bar  chart of the vDataColumn
        based on an aggregation.

        Parameters
        ----------
        method: str, optional
            The method used to aggregate the data.

            - count:
                Number of elements.
            - density:
                Percentage  of  the  distribution.
            - mean:
                Average  of the  vDataColumn ``of``.
            - min:
                Minimum  of the  vDataColumn ``of``.
            - max:
                Maximum  of the  vDataColumn ``of``.
            - sum:
                Sum of the vDataColumn ``of``.
            - q%:
                q Quantile of the vDataColumn ``of``
                (ex: 50% to get the median).

            It can also be a cutomized aggregation
            (ex: AVG(column1) + 5).
        of: str, optional
            The vDataColumn  used to compute the aggregation.
        max_cardinality: int, optional
            Maximum number of distinct elements for vDataColumns
            to be used as categorical. For these elements, no
            h is picked or computed.
        nbins: int, optional
            Number  of  bins. If empty, an  optimized number of
            bins is computed.
        h: PythonNumber, optional
            Interval width of the bar. If empty, an optimized h
            is computed.
        categorical: bool, optional
            If  set to False and the  vDataColumn is numerical,
            the parmater  'max_cardinality' is ignored and
            the bar  chart is represented as a histogram.
        bargap: float, optional
            A float between  (0, 1] that represent the
            proportion  taken out of each bar to render the
            chart. This proportion creates between  each bar.
            The  bigger the value,  the bigger the gap.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any  optional  parameter  to  pass to the  plotting
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
            options, please see :ref:`chart_gallery.bar`_

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        Let's also import `numpy` to create a dataset.

        .. ipython:: python

            import numpy as np

        Let's generate a dataset using the following data.

        .. ipython:: python

            data = vp.vDataFrame(
                {
                    "gender": ['M', 'M', 'M', 'F', 'F', 'F', 'F'],
                    "grade": ['A','B','C','A','B','B', 'B'],
                }
            )

        Now we are ready to draw the plot:

        .. code-block:: python

            data["grade"].barh()

        .. ipython:: python
            :suppress:

            vp.set_option("plotting_lib", "plotly")
            fig = data["grade"].barh()
            fig.write_html("SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdc_barh_1d.html")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdc_barh_1d.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.bar` : Bar Chart.
            | :py:meth:`verticapy.vDataFrame.barh` : Horizontal Bar Chart.
            | :py:meth:`verticapy.vDataColumn.bar` : Bar Chart.

        """
        vpy_plt, kwargs = self._parent.get_plotting_lib(
            class_name="HorizontalBarChart",
            chart=chart,
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
        kind: Literal["auto", "donut", "rose", "3d"] = "auto",
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the pie chart of the vDataColumn based on an
        aggregation.

        Parameters
        ----------
        method: str, optional
            The method used to aggregate the data.

            - count:
                Number of elements.
            - density:
                Percentage  of  the  distribution.
            - mean:
                Average  of the  vDataColumn ``of``.
            - min:
                Minimum  of the  vDataColumn ``of``.
            - max:
                Maximum  of the  vDataColumn ``of``.
            - sum:
                Sum of the vDataColumn ``of``.
            - q%:
                q Quantile of the vDataColumn ``of``
                (ex: 50% to get the median).
            It   can  also  be  a  cutomized   aggregation
            (ex: AVG(column1) + 5).
        of: str, optional
            The vDataColumn used to compute the aggregation.
        max_cardinality: int, optional
            Maximum number of distinct elements for vDataColumns
            to be used as categorical. For these elements, no
            h is picked or computed.
        h: PythonNumber, optional
            Interval width of the bar. If empty, an optimized
            h is computed.
        kind: str, optional
            The type of pie chart.
                auto   : Regular pie chart.
                donut  : Donut chart.
                rose   : Rose chart.
                3d     : 3D Pie.
            It   can    also   be  a  cutomized   aggregation
            (ex: AVG(column1) + 5).
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any  optional parameter to pass to  the  plotting
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
            options, please see :ref:`chart_gallery.pie`_

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        Let's also import `numpy` to create a dataset.

        .. ipython:: python

            import numpy as np

        Let's generate a dataset using the following data.

        .. ipython:: python

            data = vp.vDataFrame(
                {
                    "gender": ['M', 'M', 'M', 'F', 'F', 'F', 'F'],
                    "grade": ['A','B','C','A','B','B', 'B'],
                }
            )

        Now we are ready to draw the plot:

        .. code-block:: python

            data["grade"].pie()

        .. ipython:: python
            :suppress:

            vp.set_option("plotting_lib", "plotly")
            fig = data["grade"].pie(width = 600)
            fig.write_html("SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdc_pie_1d.html")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdc_pie_1d.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.hist` : Histogram.
            | :py:meth:`verticapy.vDataFrame.pie` : Pie Chart.
            | :py:meth:`verticapy.vDataColumn.bar` : Bar Chart.

        """
        vpy_plt, kwargs = self._parent.get_plotting_lib(
            class_name="PieChart",
            chart=chart,
            style_kwargs=style_kwargs,
        )
        return vpy_plt.PieChart(
            vdc=self,
            method=method,
            of=of,
            max_cardinality=max_cardinality,
            h=h,
            pie=True,
            misc_layout={"kind": kind},
        ).draw(**kwargs)

    @save_verticapy_logs
    def spider(
        self,
        by: Optional[str] = None,
        method: str = "density",
        of: Optional[str] = None,
        max_cardinality: tuple[int, int] = (6, 6),
        h: tuple[PythonNumber, PythonNumber] = (None, None),
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the spider plot of the input vDataColumn based on
        an aggregation.

        Parameters
        ----------
        by: str, optional
            vDataColumn used to partition the data.
        method: str, optional
            The method used to aggregate the data.

            - count:
                Number of elements.
            - density:
                Percentage  of  the  distribution.
            - mean:
                Average  of the  vDataColumn ``of``.
            - min:
                Minimum  of the  vDataColumn ``of``.
            - max:
                Maximum  of the  vDataColumn ``of``.
            - sum:
                Sum of the vDataColumn ``of``.
            - q%:
                q Quantile of the vDataColumn ``of``
                (ex: 50% to get the median).

            It   can  also  be  a  cutomized   aggregation
            (ex: AVG(column1) + 5).
        of: str, optional
            The vDataColumn used to compute the aggregation.
        max_cardinality: int, optional
            Maximum number of distinct elements for vDataColumns
            to be used as categorical. For these elements, no
            h is picked or computed.
        h: PythonNumber, optional
            Interval width of the bar. If empty, an optimized
            h is computed.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any  optional parameter to pass to  the  plotting
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
            options, please see :ref:`chart_gallery.spider`_

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        Let's also import `numpy` to create a dataset.

        .. ipython:: python

            import numpy as np

        Let's generate a dataset using the following data.

        .. ipython:: python

            data = vp.vDataFrame(
                {
                    "category": [np.random.choice(['A','B','C']) for _ in range(N)],
                    "score1": np.random.normal(5, 1, N),
                }
            )

        Now we are ready to draw the plot:

        .. code-block:: python

            data["score1"].spider()

        .. ipython:: python
            :suppress:

            vp.set_option("plotting_lib", "plotly")
            fig = data["score1"].spider(width = 600)
            fig.write_html("SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdc_spider_1d.html")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdc_spider_1d.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.pie` : Pie Chart.
            | :py:meth:`verticapy.vDataColumn.pie` : Pie Chart.

        """
        by, of = self._parent.format_colnames(by, of)
        columns = [self._alias]
        if by:
            columns += [by]
        vpy_plt, kwargs = self._parent.get_plotting_lib(
            class_name="SpiderChart",
            chart=chart,
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
        h: Optional[PythonNumber] = None,
        h_by: PythonNumber = 0,
        max_cardinality: int = 8,
        cat_priority: Union[None, PythonScalar, ArrayLike] = None,
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws  the histogram of the input vDataColumn based
        on an aggregation.

        Parameters
        ----------
        by: str, optional
            vDataColumn  used  to  partition  the  data.
        method: str, optional
            The method used to aggregate the data.

            - count:
                Number of elements.
            - density:
                Percentage  of  the  distribution.
            - mean:
                Average  of the  vDataColumn ``of``.
            - min:
                Minimum  of the  vDataColumn ``of``.
            - max:
                Maximum  of the  vDataColumn ``of``.
            - sum:
                Sum of the vDataColumn ``of``.
            - q%:
                q Quantile of the vDataColumn ``of``
                (ex: 50% to get the median).

            It can also be a cutomized aggregation
            (ex: AVG(column1) + 5).
        of: str, optional
            The  vDataColumn  used to compute the  aggregation.
        h: PythonNumber, optional
            Interval  width of the  input vDataColumns. Optimized
            h  will be  computed  if  the  parameter is empty  or
            invalid.
        h_by: PythonNumber, optional
            Interval  width if the 'by' vDataColumn is  numerical
            or of a date-like type. Optimized  h will be computed
            if the parameter is empty or invalid.
        max_cardinality: int, optional
            Maximum number of distinct elements for vDataColumns
            to be used as categorical.
            The less frequent  elements are gathered together
            to create a new category : 'Others'.
            This parameter is used to discretize the vDataColumn
            'by' when the main input nvDataColumn is nnumerical.
            Otherwise, it  is  used  to   discretize    all the
            vDataColumn inputs.
        cat_priority: PythonScalar / ArrayLike, optional
            ArrayLike list of the different categories to consider
            when drawing the box plot.  The other categories are
            filtered.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any  optional  parameter  to  pass  to  the  plotting
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
            options, please see :ref:`chart_gallery.hist`_

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        Let's also import `numpy` to create a dataset.

        .. ipython:: python

            import numpy as np

        We can create a variable ``N`` to fix the size:

        .. ipython:: python

            N = 50

        Let's generate a dataset using the following data.

        .. ipython:: python

            data = vp.vDataFrame(
                {
                    "score1": np.random.normal(5, 1, N),
                }
            )

        Now we are ready to draw the plot:

        .. code-block:: python

            data["score1"].hist()

        .. ipython:: python
            :suppress:

            vp.set_option("plotting_lib", "plotly")
            fig = data["score1"].hist(width = 600)
            fig.write_html("SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdc_hist_1d.html")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdc_hist_1d.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.hist` : Histogram.
            | :py:meth:`verticapy.vDataFrame.barh` : Horizontal Bar Chart.
            | :py:meth:`verticapy.vDataColumn.bar` : Bar Chart.

        """
        if self.isnum() and not (self.isbool()):
            vpy_plt, kwargs = self._parent.get_plotting_lib(
                class_name="Histogram",
                chart=chart,
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
        else:
            warning_message = (
                f"The Virtual Column {self._alias} is not "
                "numerical. A bar chart will be drawn instead."
            )
            warnings.warn(warning_message, Warning)
            if by:
                return self._parent.bar(
                    columns=[self._alias, by],
                    method=method,
                    of=of,
                    max_cardinality=(max_cardinality, max_cardinality),
                    h=(h, h),
                    chart=chart,
                    **style_kwargs,
                )
            else:
                return self.bar(
                    method=method,
                    of=of,
                    max_cardinality=max_cardinality,
                    h=h,
                    chart=chart,
                    **style_kwargs,
                )

    @save_verticapy_logs
    def density(
        self,
        by: Optional[str] = None,
        bandwidth: PythonNumber = 1.0,
        kernel: Literal["gaussian", "logistic", "sigmoid", "silverman"] = "gaussian",
        nbins: int = 200,
        xlim: Optional[tuple[float, float]] = None,
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the vDataColumn Density Plot.

        Parameters
        ----------
        by: str, optional
            vDataColumn used to partition  the data.
        bandwidth: PythonNumber, optional
            The bandwidth of the kernel.
        kernel: str, optional
            The method used for the plot.
                gaussian  : Gaussian kernel.
                logistic  : Logistic kernel.
                sigmoid   : Sigmoid kernel.
                silverman : Silverman kernel.
        nbins: int, optional
            Maximum number of points  used to evaluate
            the approximate density function.
            Increasing this  parameter increases the
            precision but also increases the time of
            the learning and scoring phases.
        xlim: tuple, optional
            Set the x limits of the current axes.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any  optional  parameter  to  pass   to  the
            plotting functions.

        Returns
        -------
        obj
            Plotting Object.

        Examples
        ---------

        .. note::

            The below example is a very basic one. For
            other more detailed examples and customization
            options, please see :ref:`chart_gallery.density`_

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        Let's also import `numpy` to create a dataset.

        .. ipython:: python

            import numpy as np

        We can create a variable ``N`` to fix the size:

        .. ipython:: python

            N = 50

        Let's generate a dataset using the following data.

        .. ipython:: python

            data = vp.vDataFrame(
                {
                    "score1": np.random.normal(5, 1, N),
                }
            )

        Now we are ready to draw the plot:

        .. code-block:: python

            data["score1"].density()

        .. ipython:: python
            :suppress:

            vp.set_option("plotting_lib", "plotly")
            fig = data["score1"].density(width = 600)
            fig.write_html("SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdc_density_1d.html")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdc_density_1d.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.density` : Density Plot.
            | :py:meth:`verticapy.vDataFrame.range_plot` : Range Plot.
            | :py:meth:`verticapy.vDataColumn.hist` : Histogram.

        """
        vml = get_vertica_mllib()
        name = gen_tmp_name(schema=conf.get_option("temp_schema"), name="kde")
        by = self._parent.format_colnames(by)
        if not xlim:
            xlim_ = [(self.min(), self.max())]
        else:
            xlim_ = [xlim]
        model = vml.KernelDensity(
            name=name,
            bandwidth=bandwidth,
            kernel=kernel,
            nbins=nbins,
            xlim=xlim_,
            store=False,
        )
        if not by:
            try:
                model.fit(self._parent, [self._alias])
                return model.plot(chart=chart, **style_kwargs)
            finally:
                model.drop()
        else:
            categories = self._parent[by].distinct()
            X, Y = [], []
            for cat in categories:
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
            vpy_plt, kwargs = self._parent.get_plotting_lib(
                class_name="MultiDensityPlot",
                chart=chart,
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
    def candlestick(
        self,
        ts: str,
        method: str = "sum",
        q: tuple[float, float] = (0.25, 0.75),
        start_date: Optional[PythonScalar] = None,
        end_date: Optional[PythonScalar] = None,
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the Time Series of the vDataColumn.

        Parameters
        ----------
        ts: str
            TS  (Time Series)  vDataColumn used to order the
            data.  The  vDataColumn  type must  be  date
            like (date, datetime, timestamp...) or  numerical.
        method: str, optional
            The method used to aggregate the data.

            - count:
                Number of elements.
            - density:
                Percentage  of  the  distribution.
            - mean:
                Average  of the  vDataColumn ``of``.
            - min:
                Minimum  of the  vDataColumn ``of``.
            - max:
                Maximum  of the  vDataColumn ``of``.
            - sum:
                Sum of the vDataColumn ``of``.
            - q%:
                q Quantile of the vDataColumn ``of``
                (ex: 50% to get the median).

            It   can  also  be  a  cutomized   aggregation
            (ex: AVG(column1) + 5).
        q: tuple, optional
            Tuple including the  2 quantiles used to draw the
            Plot.
        start_date: str / PythonNumber / date, optional
            Input Start Date. For example, time = '03-11-1993'
            will  filter  the  data  when 'ts' is less than
            the 3rd of November 1993.
        end_date: str / PythonNumber / date, optional
            Input  End  Date. For example, time = '03-11-1993'
            will filter  the data when 'ts' is  greater  than
            the 3rd of November 1993.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any  optional  parameter  to pass to the  plotting
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
            options, please see :ref:`chart_gallery.candlestick`_

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        Let's also import `numpy` to create a dataset.

        .. ipython:: python

            import numpy as np

        Let's generate a dataset using the following data.

        .. ipython:: python

            data = vp.vDataFrame(
                {
                    "date": [1990 + i for i in range(N)] * 5,
                    "population": [100 + i for i in range(N)] + [300 + i * 2 for i in range(N)] + [200 + i ** 2 - 3 * i for i in range(N)] + [50 + i ** 2 - 6 * i for i in range(N)] + [700 + i ** 2 - 10 * i for i in range(N)],
                }
            )

        Now we are ready to draw the plot:

        .. code-block:: python

            data["population"].candlestick(ts = "date")


        .. ipython:: python
            :suppress:

            vp.set_option("plotting_lib", "plotly")
            fig = data["population"].candlestick(ts = "date", width = 600)
            fig.write_html("SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdc_cadnlestick_1d.html")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdc_cadnlestick_1d.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.range_plot` : Range Plot.
            | :py:meth:`verticapy.vDataColumn.range_plot` : Range Plot.

        """
        ts = self._parent.format_colnames(ts)
        vpy_plt, kwargs = self._parent.get_plotting_lib(
            class_name="CandleStick",
            chart=chart,
            style_kwargs=style_kwargs,
        )
        return vpy_plt.CandleStick(
            vdf=self._parent,
            order_by=ts,
            method=method,
            q=q,
            column=self._alias,
            order_by_start=start_date,
            order_by_end=end_date,
        ).draw(**kwargs)

    @save_verticapy_logs
    def plot(
        self,
        ts: str,
        by: Optional[str] = None,
        start_date: Optional[PythonScalar] = None,
        end_date: Optional[PythonScalar] = None,
        kind: Literal[
            "area", "area_percent", "area_stacked", "line", "spline", "step"
        ] = "line",
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the Time Series of the vDataColumn.

        Parameters
        ----------
        ts: str
            TS  (Time Series)  vDataColumn used to order the
            data.  The  vDataColumn  type must  be  date
            like (date, datetime, timestamp...) or  numerical.
        by: str, optional
            vDataColumn used to partition the TS.
        start_date: str / PythonNumber / date, optional
            Input Start Date. For example, time = '03-11-1993'
            will  filter  the  data  when 'ts' is less than
            the 3rd of November 1993.
        end_date: str / PythonNumber / date, optional
            Input  End  Date. For example, time = '03-11-1993'
            will filter  the data when 'ts' is  greater  than
            the 3rd of November 1993.
        kind: str, optional
            The plot type.

            - line:
                Line Plot.
            - spline:
                Spline Plot.
            - step:
                Step Plot.
            - area_stacked:
                Stacked Area Plot.
            - area_percent:
                Fully Stacked Area Plot.

        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any  optional  parameter  to pass to the  plotting
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
            options, please see :ref:`chart_gallery.line`_

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        Let's also import `numpy` to create a dataset.

        .. ipython:: python

            import numpy as np

        Let's generate a dataset using the following data.

        .. ipython:: python

            data = vp.vDataFrame(
                {
                    "date": [1900, 1950, 2000],
                    "Asia": [947, 1402, 3634],
                    "Africa": [133, 221, 767],
                    "Europe": [408, 547, 729],
                    "America": [156, 339, 818],
                    "Oceania": [6, 13, 30],
                }
            )

        Now we are ready to draw the plot:

        .. code-block:: python

            data["Asia"].plot(ts = "date", kind = "spline")

        .. ipython:: python
            :suppress:

            vp.set_option("plotting_lib", "plotly")
            fig = data["Asia"].plot(ts = "date", kind = "spline", width = 600)
            fig.write_html("SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdc_plot_1d.html")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdc_plot_1d.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.plot` : Line Plot.
            | :py:meth:`verticapy.vDataColumn.range_plot` : Range Plot.

        """
        ts, by = self._parent.format_colnames(ts, by)
        vpy_plt, kwargs = self._parent.get_plotting_lib(
            class_name="LinePlot",
            chart=chart,
            style_kwargs=style_kwargs,
        )
        return vpy_plt.LinePlot(
            vdf=self._parent,
            order_by=ts,
            columns=[self._alias, by] if by else [self._alias],
            order_by_start=start_date,
            order_by_end=end_date,
            misc_layout={"kind": kind},
        ).draw(**kwargs)

    @save_verticapy_logs
    def range_plot(
        self,
        ts: str,
        q: tuple[float, float] = (0.25, 0.75),
        start_date: Optional[PythonScalar] = None,
        end_date: Optional[PythonScalar] = None,
        plot_median: bool = False,
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws  the  range   plot  of  the  vDataColumn.  The
        aggregations  used  to draw the plot are  the  median
        and the two user-specified quantiles.

        Parameters
        ----------
        ts: str
            TS  (Time Series)  vDataColumn  used  to order
            the  data.  The  vDataColumn  type must  be  date
            like (date, datetime, timestamp...) or  numerical.
        q: tuple, optional
            Tuple including the  2 quantiles used to draw the
            Plot.
        start_date: str / PythonNumber / date, optional
            Input Start Date. For example, time = '03-11-1993'
            will  filter  the  data  when 'ts' is less than
            the 3rd of November 1993.
        end_date: str / PythonNumber / date, optional
            Input  End  Date. For example, time = '03-11-1993'
            will filter  the data when 'ts' is  greater  than
            the 3rd of November 1993.
        plot_median: bool, optional
            If set to True, the Median is drawn.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any  optional  parameter  to pass to the  plotting
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
            options, please see :ref:`chart_gallery.range`_

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        Let's also import `numpy` to create a dataset.

        .. ipython:: python

            import numpy as np

        We can create a variable ``N`` to fix the size:

        .. ipython:: python

            N = 30

        Let's generate a dataset using the following data.

        .. ipython:: python

            data = vp.vDataFrame(
                {
                    "date": [1990 + i for i in range(N)] * 5,
                    "population1": [100 + i for i in range(N)] + [300 + i * 2 for i in range(N)] + [200 + i ** 2 - 3 * i for i in range(N)] + [50 + i ** 2 - 6 * i for i in range(N)] + [700 + i ** 2 - 10 * i for i in range(N)],
                    "population2": [200 + i ** 2 - i for i in range(N)] + [1000 + i * 2 for i in range(N)] + [500 + i ** 2 - 5 * i for i in range(N)] + [900 + i ** 2 + 3 * i for i in range(N)] + [100 + i ** 2 - 0.5 * i for i in range(N)],
                }
            )

        Now we are ready to draw the plot:

        .. code-block:: python

            data["population1"].range_plot(ts = "date")

        .. ipython:: python
            :suppress:

            vp.set_option("plotting_lib", "plotly")
            fig = data["population1"].range_plot(ts = "date", width = 600)
            fig.write_html("SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdc_range_plot_1d.html")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vdataframe_plotting_vdc_range_plot_1d.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.range_plot` : Range Plot.
            | :py:meth:`verticapy.vDataColumn.plot` : Line Plot.

        """
        return self._parent.range_plot(
            columns=[self._alias],
            ts=ts,
            q=q,
            start_date=start_date,
            end_date=end_date,
            plot_median=plot_median,
            chart=chart,
            **style_kwargs,
        )

    # Geospatial.

    @save_verticapy_logs
    def geo_plot(self, *args, **kwargs) -> PlottingObject:
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

        Examples
        ---------

        .. note::

            The below example is a very basic one. For
            other more detailed examples and customization
            options, please see :ref:`chart_gallery.geo`_

        Let's begin by importing the dataset module of `VerticaPy`.
        It provides a range of datasets for both training and
        exploring VerticaPy's capabilities.

        .. ipython:: python

            import verticapy.datasets as vpd

        Let's utilize the World dataset to demonstrate geospatial capabilities.

        .. code-block:: python

            import verticapy.datasets as vpd

            world = vpd.load_world()

            # We filter to select only the African continent
            africa = world[world["continent"] == "Africa"]

        .. ipython:: python
            :suppress:

            import verticapy as vp
            import verticapy.datasets as vpd

            vp.set_option("plotting_lib", "matplotlib")

            world = vpd.load_world()

            # We filter to select only the African continent
            africa = world[world["continent"] == "Africa"]

        Now we can draw the plot:

        .. ipython:: python
            :okwarning:

            @savefig core_vdataframe_plotting_vdc_geo_plot.png
            africa["geometry"].geo_plot(edgecolor = "black", color = "white")

        .. seealso::

            | :py:meth:`verticapy.vDataColumn.plot` : Line Plot.

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
            column = self._parent.format_colnames(column)
            columns += [column]
            if "cmap" not in kwargs:
                kwargs["cmap"] = PlottingBase().get_cmap(idx=0)
        else:
            if "color" not in kwargs:
                kwargs["color"] = PlottingBase().get_colors(idx=0)
        if "legend" not in kwargs:
            kwargs["legend"] = True
        if "figsize" not in kwargs:
            kwargs["figsize"] = (14, 10)
        return self._parent[columns].to_geopandas(self._alias).plot(*args, **kwargs)
