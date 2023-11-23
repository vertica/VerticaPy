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
import warnings
from typing import Literal, Optional

from matplotlib.axes import Axes
from matplotlib.pyplot import Figure as mFigure
from plotly.graph_objs._figure import Figure
from vertica_highcharts import Highchart, Highstock

import verticapy._config.config as conf
from verticapy._utils._sql._format import format_type
from verticapy._typing import PlottingObject

import verticapy.plotting._matplotlib as vpy_matplotlib_plt
import verticapy.plotting._plotly as vpy_plotly_plt
import verticapy.plotting._highcharts as vpy_highcharts_plt


class PlottingUtils:
    @staticmethod
    def _first_available_lib(
        class_name: str,
    ) -> Literal["highcharts", "matplotlib", "plotly"]:
        lookup_table = {
            vpy_highcharts_plt: "highcharts",
            vpy_matplotlib_plt: "matplotlib",
            vpy_plotly_plt: "plotly",
        }
        for lib in [vpy_plotly_plt, vpy_highcharts_plt, vpy_matplotlib_plt]:
            if hasattr(lib, class_name):
                return lookup_table[lib]
        raise NotImplementedError("This graphic is not yet implemented.")

    @staticmethod
    def _is_available(
        class_name: str, lib: Literal["highcharts", "plotly", "matplotlib"]
    ) -> bool:
        lookup_table = {
            "highcharts": vpy_highcharts_plt,
            "matplotlib": vpy_matplotlib_plt,
            "plotly": vpy_plotly_plt,
        }
        return hasattr(lookup_table[lib], class_name)

    def get_plotting_lib(
        self,
        class_name: Optional[str] = None,
        chart: Optional[PlottingObject] = None,
        matplotlib_kwargs: Optional[dict] = None,
        plotly_kwargs: Optional[dict] = None,
        highchart_kwargs: Optional[dict] = None,
        style_kwargs: Optional[dict] = None,
    ) -> tuple[Literal[vpy_highcharts_plt, vpy_matplotlib_plt, vpy_plotly_plt], dict]:
        """
        Returns the first available library (Plotly, Matplotlib, or
        Highcharts) to draw a specific graphic. If the graphic is not
        available for a library, function tries the next plotting
        library. The style applied to the graphic corresponds to the
        input style. The final graphic is drawn using the input
        'chart' object.
        """
        highchart_kwargs, plotly_kwargs, matplotlib_kwargs, style_kwargs = format_type(
            highchart_kwargs, plotly_kwargs, matplotlib_kwargs, style_kwargs, dtype=dict
        )
        chart_not_none = True
        if isinstance(chart, (Axes, mFigure)):
            lib = "matplotlib"
        elif isinstance(chart, (Highchart, Highstock)):
            lib = "highcharts"
        elif isinstance(chart, Figure):
            lib = "plotly"
        else:
            chart_not_none = False
            lib = conf.get_option("plotting_lib")
        if not self._is_available(class_name=class_name, lib=lib):
            if chart_not_none:
                raise NotImplementedError(
                    f"The type of the input parameter 'chart' ('{lib}' object)"
                    " is not supported for the current library. Leave it empty"
                    " if you want to use the first available plotting library."
                )
            last_lib = lib
            lib = self._first_available_lib(class_name=class_name)
            warning_message = (
                f"This plot is not yet available using the '{last_lib}' module.\n"
                f"This plot will be drawn by using the '{lib}' module.\n"
                "You can switch to any graphical library by using the 'set_option' "
                "function.\nThe following example sets matplotlib as graphical library:\n"
                "import verticapy\nverticapy.set_option('plotting_lib', 'matplotlib')"
            )
            warnings.warn(warning_message, Warning)
        if lib == "plotly":
            vpy_plt = vpy_plotly_plt
            kwargs = {"fig": chart, **plotly_kwargs, **style_kwargs}
        elif lib == "highcharts":
            vpy_plt = vpy_highcharts_plt
            kwargs = {"chart": chart, **highchart_kwargs, **style_kwargs}
        elif lib == "matplotlib":
            vpy_plt = vpy_matplotlib_plt
            kwargs = {"ax": chart, **matplotlib_kwargs, **style_kwargs}
        else:
            raise ModuleNotFoundError(f"Unrecognized library: '{lib}'.")
        return vpy_plt, kwargs
