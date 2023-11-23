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
from typing import Any, Literal, Optional

import numpy as np

from matplotlib.axes import Axes

from verticapy.plotting._matplotlib.base import MatplotlibBase


class LinePlot(MatplotlibBase):
    # Properties.

    @property
    def _category(self) -> Literal["graph"]:
        return "graph"

    @property
    def _kind(self) -> Literal["line"]:
        return "line"

    @property
    def _compute_method(self) -> Literal["line"]:
        return "line"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "color": self.get_colors(idx=0),
            "linewidth": 2,
            "marker": "o",
            "markevery": 0.05,
            "markersize": 7,
            "markeredgecolor": "black",
        }
        if len(self.data["x"]) < 20:
            self.init_style["markerfacecolor"] = "white"
        self.init_style_fill = {
            "alpha": 0.2,
        }

    def _get_style(self, idx: int = 0) -> dict[str, Any]:
        colors = self.get_colors()
        kwargs = {
            "color": colors[idx % len(colors)],
            "markerfacecolor": colors[idx % len(colors)],
        }
        if len(self.data["x"]) < 20:
            kwargs = {
                **self.init_style,
                **kwargs,
            }
        return kwargs

    # Draw.

    def draw(
        self,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a time series plot using the Matplotlib API.
        """
        style_kwargs = self._fix_color_style_kwargs(style_kwargs)
        colors = self.get_colors()
        ax, fig, style_kwargs = self._get_ax_fig(
            ax, size=(8, 6), set_axis_below=True, grid="y", style_kwargs=style_kwargs
        )
        plot_fun = ax.step if (self.layout["kind"] == "step") else ax.plot
        if not self.layout["has_category"]:
            args = [self.data["x"], self.data["Y"][:, 0]]
            kwargs = self._update_dict(self.init_style, style_kwargs)
            plot_fun(*args, **kwargs)
            if self.layout["kind"] == "area":
                if "color" in self._update_dict(kwargs, style_kwargs):
                    color = self._update_dict(kwargs, style_kwargs)["color"]
                else:
                    color = colors[0]
                ax.fill_between(*args, facecolor=color, **self.init_style_fill)
            ax.set_xlim(min(self.data["x"]), max(self.data["x"]))
        else:
            uniques = np.unique(self.data["z"])
            for i, c in enumerate(uniques):
                x = self.data["x"][self.data["z"] == c]
                y = self.data["Y"][:, 0][self.data["z"] == c]
                kwargs = self._update_dict(self._get_style(idx=i), style_kwargs, i)
                plot_fun(x, y, label=c, **kwargs)
        ax.set_xlabel(self.layout["order_by"])
        ax.set_ylabel(self.layout["columns"][0])
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
        if self.layout["has_category"]:
            ax.legend(
                title=self.layout["order_by"],
                loc="center left",
                bbox_to_anchor=[1, 0.5],
            )
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        return ax


class MultiLinePlot(MatplotlibBase):
    # Properties.

    @property
    def _category(self) -> Literal["graph"]:
        return "graph"

    @property
    def _kind(self) -> Literal["line"]:
        return "line"

    @property
    def _compute_method(self) -> Literal["line"]:
        return "line"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "color": self.get_colors(idx=0),
            "linewidth": 2,
            "marker": "o",
            "markevery": 0.05,
            "markersize": 7,
            "markeredgecolor": "black",
        }
        if len(self.data["x"]) < 20:
            self.init_style["markerfacecolor"] = "white"

    def _get_style(
        self,
        idx: int = 0,
    ) -> dict[str, Any]:
        colors = self.get_colors()
        kwargs = {"linewidth": 1}
        kwargs_small = {
            "marker": "o",
            "markevery": 0.05,
            "markerfacecolor": colors[idx],
            "markersize": 7,
        }
        if self.layout["kind"] in ("line", "step"):
            color = colors[idx]
            if self.data["Y"].shape[0] < 20:
                kwargs = {
                    **kwargs_small,
                    "markeredgecolor": "black",
                }
            kwargs["label"] = self.layout["columns"][idx]
            kwargs["linewidth"] = 2
        elif self.layout["kind"] == "area_percent":
            color = "white"
            if self.data["Y"].shape[0] < 20:
                kwargs = {
                    **kwargs_small,
                    "markeredgecolor": "white",
                }
        else:
            color = "black"
            if self.data["Y"].shape[0] < 20:
                kwargs = {
                    **kwargs_small,
                    "markeredgecolor": "black",
                }
        kwargs["color"] = color
        return kwargs

    # Draw.

    def draw(
        self,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a multi-time series plot using the Matplotlib API.
        """
        style_kwargs = self._fix_color_style_kwargs(style_kwargs)
        colors = self.get_colors()
        ax, fig, style_kwargs = self._get_ax_fig(
            ax, size=(8, 6), set_axis_below=True, grid="y", style_kwargs=style_kwargs
        )
        n, m = self.data["Y"].shape
        plot_fun = ax.step if (self.layout["kind"] == "step") else ax.plot
        prec = [0 for j in range(n)]
        for i in range(0, m):
            if self.layout["kind"] in ("area_percent", "area_stacked"):
                points = np.sum(self.data["Y"][:, : i + 1], axis=1).astype(float)
                if self.layout["kind"] == "area_percent":
                    points /= np.sum(self.data["Y"], axis=1).astype(float)
            else:
                points = self.data["Y"][:, i].astype(float)
            kwargs = self._get_style(idx=i)
            if "color" in style_kwargs and n < 20:
                kwargs["markerfacecolor"] = self.get_colors(d=style_kwargs, idx=i)
            kwargs = self._update_dict(kwargs, style_kwargs, i)
            plot_fun(self.data["x"], points, **kwargs)
            if self.layout["kind"] not in ("line", "step"):
                args = [self.data["x"], prec, points]
                kwargs = {
                    "label": self.layout["columns"][i],
                    "color": colors[i],
                    **style_kwargs,
                }
                if not isinstance(kwargs["color"], str):
                    kwargs["color"] = kwargs["color"][i % len(kwargs["color"])]
                ax.fill_between(*args, **kwargs)
                prec = points
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
        if self.layout["kind"] == "area_percent":
            ax.set_ylim(0, 1)
        elif self.layout["kind"] == "area_stacked":
            ax.set_ylim(0)
        ax.set_xlim(min(self.data["x"]), max(self.data["x"]))
        ax.set_xlabel(self.layout["order_by"])
        ax.legend(loc="center left", bbox_to_anchor=[1, 0.5])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        return ax
