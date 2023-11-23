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

import numpy as np

from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from verticapy.plotting._matplotlib.base import MatplotlibBase


class DensityPlot(MatplotlibBase):
    # Properties.

    @property
    def _category(self) -> Literal["graph"]:
        return "graph"

    @property
    def _kind(self) -> Literal["density"]:
        return "density"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "color": self.get_colors(idx=0),
        }
        self.init_style_alpha = {
            "alpha": 0.7,
        }

    # Draw.

    def draw(
        self,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a density plot using the Matplotlib API.
        """
        style_kwargs = self._fix_color_style_kwargs(style_kwargs)
        ax, fig, style_kwargs = self._get_ax_fig(
            ax, size=(7, 5), set_axis_below=True, grid=True, style_kwargs=style_kwargs
        )
        ax.plot(
            self.data["x"],
            self.data["y"],
            **self._update_dict(self.init_style, style_kwargs),
        )
        color = self._get_final_color(style_kwargs=style_kwargs)
        ax.fill_between(
            self.data["x"],
            self.data["y"],
            facecolor=color,
            **self.init_style_alpha,
        )
        ax.set_xlim(np.nanmin(self.data["x"]), np.nanmax(self.data["x"]))
        ax.set_ylim(bottom=0)
        ax.set_xlabel(self._clean_quotes(self.layout["x_label"]))
        ax.set_ylabel(self._clean_quotes(self.layout["y_label"]))
        return ax


class MultiDensityPlot(DensityPlot):
    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style_alpha = {
            "alpha": 0.5,
        }

    # Draw.

    def draw(
        self,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a multi-density plot using the Matplotlib API.
        """
        style_kwargs = self._fix_color_style_kwargs(style_kwargs)
        ax, fig, style_kwargs = self._get_ax_fig(
            ax, size=(7, 5), set_axis_below=True, grid=True, style_kwargs=style_kwargs
        )
        n, m = self.data["X"].shape
        custom_lines = []
        for i in range(m):
            color = self._get_final_color(style_kwargs=style_kwargs, idx=i)
            kwargs = self._get_final_style_kwargs(style_kwargs=style_kwargs, idx=i)
            ax.plot(
                self.data["X"][:, i],
                self.data["Y"][:, i],
                **kwargs,
            )
            ax.fill_between(
                self.data["X"][:, i],
                self.data["Y"][:, i],
                facecolor=color,
                **self.init_style_alpha,
            )
            custom_lines += [
                Line2D(
                    [0],
                    [0],
                    color=color,
                    lw=4,
                ),
            ]
        ax.set_title(self.layout["title"])
        ax.set_xlim(np.nanmin(self.data["X"]), np.nanmax(self.data["X"]))
        ax.set_ylim(bottom=0)
        ax.legend(
            custom_lines,
            self.layout["labels"],
            title=self.layout["labels_title"],
            loc="center left",
            bbox_to_anchor=[1, 0.5],
        )
        ax.set_xlabel(self._clean_quotes(self.layout["x_label"]))
        ax.set_ylabel(self._clean_quotes(self.layout["y_label"]))
        return ax


class DensityPlot2D(MatplotlibBase):
    # Properties.

    @property
    def _category(self) -> Literal["map"]:
        return "map"

    @property
    def _kind(self) -> Literal["density"]:
        return "density"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "cmap": "Reds",
            "origin": "lower",
            "interpolation": "bilinear",
        }

    # Draw.

    def draw(
        self,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a density plot using the Matplotlib API.
        """
        style_kwargs = self._fix_color_style_kwargs(style_kwargs)
        ax, fig, style_kwargs = self._get_ax_fig(
            ax, size=(8, 6), set_axis_below=False, grid=False, style_kwargs=style_kwargs
        )
        im = ax.imshow(
            self.data["X"],
            extent=self.layout["extent"],
            **self._update_dict(self.init_style, style_kwargs),
        )
        fig.colorbar(im, ax=ax)
        ax.set_xlabel(self._clean_quotes(self.layout["x_label"]))
        ax.set_ylabel(self._clean_quotes(self.layout["y_label"]))
        return ax
