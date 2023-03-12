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
import copy
from typing import Literal, Optional, Union
import numpy as np

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from verticapy._typing import ArrayLike

from verticapy.plotting._matplotlib.base import MatplotlibBase


class ImportanceBarChart(MatplotlibBase):
    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["importance"]:
        return "importance"

    def draw(
        self,
        coeff_importances: Union[dict, ArrayLike],
        coeff_sign: Union[dict, ArrayLike] = {},
        print_legend: bool = True,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a coeff importance bar chart using the Matplotlib API.
        """
        if isinstance(coeff_importances, dict):
            coefficients, importances, signs = [], [], []
            for coeff in coeff_importances:
                coefficients += [coeff]
                importances += [coeff_importances[coeff]]
                signs += [coeff_sign[coeff]] if (coeff in coeff_sign) else [1]
        else:
            coefficients = copy.deepcopy(coeff_importances)
            importances = abs(coeff_sign)
            signs = np.sign(coeff_sign)
        importances, coefficients, signs = zip(
            *sorted(zip(importances, coefficients, signs))
        )
        ax, fig = self._get_ax_fig(
            ax, size=(12, int(len(importances) / 2) + 1), set_axis_below=True, grid=True
        )
        color = []
        for item in signs:
            color += (
                [self.get_colors(d=style_kwargs, idx=0)]
                if (item == 1)
                else [self.get_colors(d=style_kwargs, idx=1)]
            )
        plus, minus = (
            self.get_colors(d=style_kwargs, idx=0),
            self.get_colors(d=style_kwargs, idx=1),
        )
        param = {"alpha": 0.86}
        style_kwargs = self._update_dict(param, style_kwargs)
        style_kwargs["color"] = color
        ax.barh(range(0, len(importances)), importances, 0.9, **style_kwargs)
        if print_legend:
            orange = mpatches.Patch(color=minus, label="sign -")
            blue = mpatches.Patch(color=plus, label="sign +")
            ax.legend(
                handles=[blue, orange], loc="center left", bbox_to_anchor=[1, 0.5]
            )
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.set_ylabel("Features")
        ax.set_xlabel("Importance")
        ax.set_yticks(range(0, len(importances)))
        ax.set_yticklabels(coefficients)
        return ax
