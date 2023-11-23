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
from typing import Callable, Literal

from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from verticapy.plotting._matplotlib.animated.bar import AnimatedBarChart


class AnimatedPieChart(AnimatedBarChart):
    # Properties.

    @property
    def _kind(self) -> Literal["animated_pie"]:
        return "animated_pie"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_date_style_dict = {
            "fontsize": 50,
            "alpha": 0.6,
            "color": "gray",
            "ha": "right",
            "va": "center",
        }
        self.init_pie_style = {
            "wedgeprops": {"edgecolor": "white", "alpha": 0.5},
            "textprops": {"fontsize": 10, "fontweight": "bold"},
            "autopct": "%1.1f%%",
        }

    # Draw.

    def _animate(
        self,
        bar_values: dict,
        m: int,
        date_f: Callable,
        fixed_xy_lim: bool,
        date_in_title: bool,
        date_style_dict: dict,
        style_kwargs: dict,
        ax: Axes,
    ) -> Callable:
        def animate(i: int) -> tuple[Axes]:
            ax.clear()
            pie_chart = ax.pie(
                x=bar_values[i]["width"],
                labels=bar_values[i]["y"],
                colors=bar_values[i]["c"],
                **self._update_dict(self.init_pie_style, style_kwargs),
            )
            for p in pie_chart[2]:
                p.set_fontweight("normal")
            if date_in_title:
                ax.set_title(date_f(bar_values[i]["date"]))
            else:
                ax.text(1.8, 1, date_f(bar_values[i]["date"]), **date_style_dict)
            all_categories, custom_lines = [], []
            if m >= 3:
                for idx, c in enumerate(bar_values[i]["x"]):
                    if c not in all_categories:
                        all_categories += [c]
                        custom_lines += [
                            Line2D(
                                [0],
                                [0],
                                color=bar_values[i]["c"][idx],
                                lw=6,
                                alpha=self._update_dict(
                                    self.init_pie_style, style_kwargs
                                )["wedgeprops"]["alpha"],
                            )
                        ]
                ax.legend(
                    custom_lines,
                    all_categories,
                    title=self.layout["columns"][-1],
                    loc="center left",
                    bbox_to_anchor=[1, 0.5],
                )
            return (ax,)

        return animate
