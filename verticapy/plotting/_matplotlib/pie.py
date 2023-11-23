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
from typing import Any, Callable, Literal, Optional

import numpy as np

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from matplotlib.axes import Axes

from verticapy._typing import ArrayLike

from verticapy.plotting._matplotlib.base import MatplotlibBase


class PieChart(MatplotlibBase):
    # Properties.

    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["pie"]:
        return "pie"

    @property
    def _compute_method(self) -> Literal["1D"]:
        return "1D"

    # Formatting Methods.

    @staticmethod
    def _make_autopct(values: ArrayLike, category: str) -> Callable:
        def my_autopct(pct: Any) -> str:
            total = sum(values)
            val = float(pct) * float(total) / 100.0
            if category == "int":
                val = int(round(val))
                return "{v:d}".format(v=val)
            else:
                return "{v:f}".format(v=val)

        return my_autopct

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "shadow": True,
            "startangle": 290,
            "textprops": {"color": "w"},
            "normalize": True,
        }
        self.init_style_donut = {
            **self.init_style,
            "wedgeprops": dict(width=0.4, edgecolor="w"),
            "explode": None,
            "pctdistance": 0.8,
        }
        self.init_style_text = {
            "rotation_mode": "anchor",
            "alpha": 1,
            "color": "black",
        }

    # Draw.

    def draw(
        self,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a pie chart using the Matplotlib API.
        """
        style_kwargs = self._fix_color_style_kwargs(style_kwargs)
        if "color" in style_kwargs and isinstance(style_kwargs["color"], str):
            raise ValueError("The color input should be a list.")
        colors = (
            style_kwargs["color"] + self.get_colors()
            if "color" in style_kwargs
            else self.get_colors()
        )
        if "color" in style_kwargs:
            del style_kwargs["color"]
        self.init_style["colors"] = colors
        self.init_style_donut["colors"] = colors
        n = len(self.data["y"])
        explode = [0 for i in range(n)]
        explode[max(zip(self.data["y"], range(n)))[1]] = 0.13
        current_explode = 0.15
        total_count = sum(self.data["y"])
        for idx, item in enumerate(self.data["y"]):
            if (item < 0.05) or (
                (item > 1) and (float(item) / float(total_count) < 0.05)
            ):
                current_explode = min(0.9, current_explode * 1.4)
                explode[idx] = current_explode
        if self.layout["method"].lower() == "density":
            autopct = "%1.1f%%"
        else:
            if (self.layout["method"].lower() in ["sum", "count"]) or (
                (self.layout["method"].lower() in ["min", "max"])
                and (self.layout["of_cat"] == "int")
            ):
                category = "int"
            else:
                category = None
            autopct = self._make_autopct(self.data["y"], category)
        if self.layout["kind"] != "rose":
            ax, fig, style_kwargs = self._get_ax_fig(
                ax,
                size=(8, 6),
                set_axis_below=False,
                grid=False,
                style_kwargs=style_kwargs,
            )
            if self.layout["kind"] == "donut":
                kwargs = {**self.init_style_donut, "autopct": autopct}
            else:
                kwargs = {
                    **self.init_style,
                    "autopct": autopct,
                    "explode": explode,
                }
            ax.pie(
                self.data["y"],
                labels=self.layout["labels"],
                **self._update_dict(kwargs, style_kwargs),
            )
            handles, labels = ax.get_legend_handles_labels()
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
            ax.legend(
                handles,
                labels,
                title=self.layout["column"],
                loc="center left",
                bbox_to_anchor=[1, 0.5],
            )
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        else:
            try:
                y, labels = zip(
                    *sorted(
                        zip(self.data["y"], self.layout["labels"]), key=lambda t: t[0]
                    )
                )
            except:
                y, labels = self.data["y"], self.layout["labels"]
            N = len(labels)
            width = 2 * np.pi / N
            rad = np.cumsum([width] * N)
            fig = plt.figure()
            if not ax:
                ax = fig.add_subplot(111, polar=True)
            ax.grid(False)
            ax.spines["polar"].set_visible(False)
            ax.set_yticks([])
            ax.set_thetagrids([])
            ax.set_theta_zero_location("N")
            kwargs = {
                "color": colors,
            }
            colors = self._update_dict(kwargs, style_kwargs, -1)["color"]
            if isinstance(colors, str):
                colors = [colors]
            colors = colors + self.get_colors()
            style_kwargs["color"] = colors
            ax.bar(
                rad,
                y,
                width=width,
                **self._update_dict(kwargs, style_kwargs, -1),
            )
            for i in np.arange(N):
                ax.text(
                    rad[i] + 0.1,
                    [yi * 1.02 for yi in y][i],
                    [round(yi, 2) for yi in y][i],
                    rotation=rad[i] * 180 / np.pi,
                    **self.init_style_text,
                )
            try:
                labels, colors = zip(
                    *sorted(zip(labels, colors[:N]), key=lambda t: t[0])
                )
            except:
                pass
            ax.legend(
                [Line2D([0], [0], color=color) for color in colors],
                labels,
                bbox_to_anchor=[1.1, 0.5],
                loc="center left",
                title=self.layout["column"],
                labelspacing=1,
            )
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        return ax


class NestedPieChart(MatplotlibBase):
    # Properties.

    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["pie"]:
        return "pie"

    @property
    def _compute_method(self) -> Literal["rollup"]:
        return "rollup"

    # Styling Methods.

    def _get_final_style(self, style_kwargs: dict) -> tuple:
        wedgeprops = dict(width=0.3, edgecolor="w")
        kwargs = {}
        for s in style_kwargs:
            if s not in ("color", "colors", "wedgeprops"):
                kwargs[s] = style_kwargs[s]
        if "wedgeprops" in style_kwargs:
            wedgeprops = style_kwargs["wedgeprops"]
        if "colors" in style_kwargs:
            colors = style_kwargs["colors"]
        elif "color" in style_kwargs:
            colors = style_kwargs["color"]
        else:
            colors = self.get_colors()
        return colors, wedgeprops, kwargs

    # Draw.

    def draw(
        self,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a nested pie chart using the Matplotlib API.
        """
        style_kwargs = self._fix_color_style_kwargs(style_kwargs)
        if "color" in style_kwargs and isinstance(style_kwargs["color"], str):
            raise ValueError("The color input should be a list.")
        n = len(self.layout["columns"])
        wedgeprops = dict(width=0.3, edgecolor="w")
        colors, wedgeprops, kwargs = self._get_final_style(style_kwargs=style_kwargs)
        m, k = len(colors), 0
        ax, fig, style_kwargs = self._get_ax_fig(
            ax,
            size=(12, 8),
            set_axis_below=False,
            grid=False,
            style_kwargs=style_kwargs,
        )
        all_colors_dict, all_categories, all_categories_col = {}, {}, []
        for i in range(0, n):
            if i in [0]:
                pctdistance = 0.77
            elif i > 2:
                pctdistance = 0.9
            elif i > 1:
                pctdistance = 0.88
            else:
                pctdistance = 0.85
            all_colors_dict[i] = {}
            all_categories[i] = list(dict.fromkeys(self.data["groups"][i][-2]))
            all_categories_col += [self.layout["columns"][n - i - 1]]
            for c in all_categories[i]:
                all_colors_dict[i][c] = colors[k % m]
                k += 1
            group = [int(c) for c in self.data["groups"][i][-1]]
            tmp_colors = [all_colors_dict[i][j] for j in self.data["groups"][i][-2]]
            autopct = None if len(group) > 16 else "%1.1f%%"
            ax.pie(
                group,
                radius=0.3 * (i + 2),
                colors=tmp_colors,
                wedgeprops=wedgeprops,
                autopct=autopct,
                pctdistance=pctdistance,
                **kwargs,
            )
            legend_colors = [all_colors_dict[i][c] for c in all_colors_dict[i]]
            if n == 1:
                bbox_to_anchor = [0.5, 1]
            elif n < 4:
                bbox_to_anchor = [0.4 + n * 0.23, 0.5 + 0.15 * i]
            else:
                bbox_to_anchor = [0.2 + n * 0.23, 0.5 + 0.15 * i]
            legend = ax.legend(
                [Line2D([0], [0], color=color, lw=4) for color in legend_colors],
                all_categories[i],
                bbox_to_anchor=bbox_to_anchor,
                loc="upper left",
                title=all_categories_col[i],
                labelspacing=1,
                ncol=len(all_categories[i]),
            )
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            plt.gca().add_artist(legend)
        return ax
