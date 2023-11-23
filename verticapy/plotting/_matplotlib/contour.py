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
from typing import Literal, Optional

from matplotlib.axes import Axes

from verticapy._utils._sql._format import format_type

from verticapy.plotting._matplotlib.base import MatplotlibBase


class ContourPlot(MatplotlibBase):
    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["contour"]:
        return "contour"

    @property
    def _compute_method(self) -> Literal["contour"]:
        return "contour"

    @property
    def _dimension_bounds(self) -> tuple[int, int]:
        return (2, 2)

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {"linewidths": 0.5, "levels": 14, "colors": "k"}
        self.init_style_cf = {
            "cmap": self.get_cmap(
                color=[self.get_colors(idx=2), "#FFFFFF", self.get_colors(idx=0)]
            ),
            "levels": 14,
        }

    def _update_style(
        self, cf: bool = False, style_kwargs: Optional[dict] = None
    ) -> dict:
        style_kwargs = format_type(style_kwargs, dtype=dict)
        if cf:
            kwargs = self._update_dict(self.init_style_cf, style_kwargs)
            for s in ["colors", "color", "linewidths", "linestyles"]:
                if s in kwargs:
                    del kwargs[s]
        else:
            kwargs = self._update_dict(self.init_style, style_kwargs)
            if "cmap" in kwargs:
                del kwargs["cmap"]
        return kwargs

    # Draw.

    def draw(
        self,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a contour plot using the Matplotlib API.
        """
        ax, fig, style_kwargs = self._get_ax_fig(
            ax, size=(8, 6), set_axis_below=False, grid=False, style_kwargs=style_kwargs
        )
        ax.contour(
            self.data["X"],
            self.data["Y"],
            self.data["Z"],
            **self._update_style(cf=False, style_kwargs=style_kwargs),
        )
        cp = ax.contourf(
            self.data["X"],
            self.data["Y"],
            self.data["Z"],
            **self._update_style(cf=True, style_kwargs=style_kwargs),
        )
        fig.colorbar(cp).set_label(self.layout["func_repr"])
        ax.set_xlabel(self.layout["columns"][0])
        ax.set_ylabel(self.layout["columns"][1])
        return ax
