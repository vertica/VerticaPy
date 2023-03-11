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
from typing import Literal, Optional, TYPE_CHECKING
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes

from verticapy._config.colors import get_colors
import verticapy._config.config as conf
from verticapy._typing import PythonNumber, SQLColumns

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame, vDataColumn

from verticapy.plotting._matplotlib.base import MatplotlibBase


class PieChart(MatplotlibBase):
    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["pie"]:
        return "pie"

    @property
    def _compute_method(self) -> Literal["1D"]:
        return "1D"

    @staticmethod
    def _make_autopct(values, category):
        def my_autopct(pct):
            total = sum(values)
            val = float(pct) * float(total) / 100.0
            if category == "int":
                val = int(round(val))
                return "{v:d}".format(v=val)
            else:
                return "{v:f}".format(v=val)

        return my_autopct

    def draw(
        self,
        vdc: "vDataColumn",
        method: str = "density",
        of: Optional[str] = None,
        max_cardinality: int = 6,
        h: PythonNumber = 0,
        donut: bool = False,
        rose: bool = False,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a pie chart using the Matplotlib API.
        """
        colors = get_colors()
        self._compute_plot_params(
            vdc, max_cardinality=max_cardinality, method=method, of=of, pie=True
        )
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
        if method.lower() == "density":
            autopct = "%1.1f%%"
        else:
            if (method.lower() in ["sum", "count"]) or (
                (method.lower() in ["min", "max"])
                and (vdc._parent[of].category == "int")
            ):
                category = "int"
            else:
                category = None
            autopct = self._make_autopct(self.data["y"], category)
        if not (rose):
            ax, fig = self._get_ax_fig(
                ax, size=(8, 6), set_axis_below=False, grid=False
            )
            param = {
                "autopct": autopct,
                "colors": colors,
                "shadow": True,
                "startangle": 290,
                "explode": explode,
                "textprops": {"color": "w"},
                "normalize": True,
            }
            if donut:
                param["wedgeprops"] = dict(width=0.4, edgecolor="w")
                param["explode"] = None
                param["pctdistance"] = 0.8
            ax.pie(
                self.data["y"],
                labels=self.data["labels"],
                **self._update_dict(param, style_kwargs),
            )
            handles, labels = ax.get_legend_handles_labels()
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
            ax.legend(
                handles,
                labels,
                title=vdc._alias,
                loc="center left",
                bbox_to_anchor=[1, 0.5],
            )
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        else:
            try:
                y, labels = zip(
                    *sorted(
                        zip(self.data["y"], self.data["labels"]), key=lambda t: t[0]
                    )
                )
            except:
                y, labels = self.data["y"], self.data["labels"]
            N = len(labels)
            width = 2 * np.pi / N
            rad = np.cumsum([width] * N)

            fig = plt.figure()
            if not (ax):
                ax = fig.add_subplot(111, polar=True)
            ax.grid(False)
            ax.spines["polar"].set_visible(False)
            ax.set_yticks([])
            ax.set_thetagrids([])
            ax.set_theta_zero_location("N")
            param = {
                "color": colors,
            }
            colors = self._update_dict(param, style_kwargs, -1)["color"]
            if isinstance(colors, str):
                colors = [colors]
            colors = colors + get_colors()
            style_kwargs["color"] = colors
            ax.bar(
                rad, y, width=width, **self._update_dict(param, style_kwargs, -1),
            )
            for i in np.arange(N):
                ax.text(
                    rad[i] + 0.1,
                    [yi * 1.02 for yi in y][i],
                    [round(yi, 2) for yi in y][i],
                    rotation=rad[i] * 180 / np.pi,
                    rotation_mode="anchor",
                    alpha=1,
                    color="black",
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
                title=vdc._alias,
                labelspacing=1,
            )
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        return ax


class NestedPieChart(MatplotlibBase):
    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["pie"]:
        return "pie"

    def draw(
        self,
        vdf: "vDataFrame",
        columns: SQLColumns,
        max_cardinality: Optional[int] = None,
        h: PythonNumber = None,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a nested pie chart using the Matplotlib API.
        """
        if isinstance(columns, str):
            columns = [columns]
        wedgeprops = dict(width=0.3, edgecolor="w")
        tmp_style = {}
        for elem in style_kwargs:
            if elem not in ("color", "colors", "wedgeprops"):
                tmp_style[elem] = style_kwargs[elem]
        if "wedgeprops" in style_kwargs:
            wedgeprops = style_kwargs["wedgeprops"]
        if "colors" in style_kwargs:
            colors, n = style_kwargs["colors"], len(columns)
        elif "color" in style_kwargs:
            colors, n = style_kwargs["color"], len(columns)
        else:
            colors, n = get_colors(), len(columns)
        m, k = len(colors), 0
        if isinstance(h, (int, float, type(None))):
            h = (h,) * n
        if isinstance(max_cardinality, (int, float, type(None))):
            if max_cardinality == None:
                max_cardinality = (6,) * n
            else:
                max_cardinality = (max_cardinality,) * n
        vdf_tmp = vdf[columns]
        for idx, column in enumerate(columns):
            vdf_tmp[column].discretize(h=h[idx])
            vdf_tmp[column].discretize(method="topk", k=max_cardinality[idx])
        if not (ax):
            fig, ax = plt.subplots()
            if conf._get_import_success("jupyter"):
                fig.set_size_inches(8, 6)
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
            result = (
                vdf_tmp.groupby(columns[: n - i], ["COUNT(*) AS cnt"])
                .sort(columns[: n - i])
                .to_numpy()
                .T
            )
            all_colors_dict[i] = {}
            all_categories[i] = list(dict.fromkeys(result[-2]))
            all_categories_col += [columns[n - i - 1]]
            for elem in all_categories[i]:
                all_colors_dict[i][elem] = colors[k % m]
                k += 1
            group = [int(elem) for elem in result[-1]]
            tmp_colors = [all_colors_dict[i][j] for j in result[-2]]
            if len(group) > 16:
                autopct = None
            else:
                autopct = "%1.1f%%"
            ax.pie(
                group,
                radius=0.3 * (i + 2),
                colors=tmp_colors,
                wedgeprops=wedgeprops,
                autopct=autopct,
                pctdistance=pctdistance,
                **tmp_style,
            )
            legend_colors = [all_colors_dict[i][elem] for elem in all_colors_dict[i]]
            if n == 1:
                bbox_to_anchor = [0.5, 1]
            elif n < 4:
                bbox_to_anchor = [0.4 + n * 0.23, 0.5 + 0.15 * i]
            else:
                bbox_to_anchor = [0.2 + n * 0.23, 0.5 + 0.15 * i]
            legend = plt.legend(
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
