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
from typing import Literal, Optional
import numpy as np

from matplotlib.axes import Axes
import matplotlib.patches as mpatches

from verticapy.plotting._matplotlib.base import MatplotlibBase


class ImportanceBarChart(MatplotlibBase):

    # Properties.

    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["importance"]:
        return "importance"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {"alpha": 0.86}
        return None

    # Draw.

    def draw(self, ax: Optional[Axes] = None, **style_kwargs,) -> Axes:
        """
        Draws a coeff importance bar chart using the Matplotlib API.
        """
        coef_names = np.array(self.layout["columns"])
        importances = abs(self.data["importance"]).astype(float)
        coef_names = coef_names[importances != np.nan]
        importances = importances[importances != np.nan]
        importances = importances[coef_names != None]
        coef_names = coef_names[coef_names != None]
        signs = np.sign(importances)
        x_label = self.layout["x_label"] if "x_label" in self.layout else "Importance"
        y_label = self.layout["y_label"] if "y_label" in self.layout else "Features"
        importances, coef_names, signs = zip(
            *sorted(zip(importances, coef_names, signs))
        )
        ax, fig = self._get_ax_fig(
            ax, size=(12, int(len(importances) / 2) + 1), set_axis_below=True, grid=True
        )
        plus = self.get_colors(d=style_kwargs, idx=0)
        minus = self.get_colors(d=style_kwargs, idx=1)
        style_kwargs = self._update_dict(self.init_style, style_kwargs)
        style_kwargs["color"] = [
            self.get_colors(d=style_kwargs, idx={-1: 1, 0: 0, 1: 0}[int(i)])
            for i in signs
        ]
        ax.barh(range(0, len(importances)), importances, 0.9, **style_kwargs)
        signs = np.array(signs)
        if len(signs[signs == -1]) != 0:
            color_plus = mpatches.Patch(color=plus, label="sign +")
            color_minus = mpatches.Patch(color=minus, label="sign -")
            ax.legend(
                handles=[color_plus, color_minus],
                loc="center left",
                bbox_to_anchor=[1, 0.5],
            )
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_yticks(range(0, len(importances)))
        ax.set_yticklabels(coef_names)
        return ax
