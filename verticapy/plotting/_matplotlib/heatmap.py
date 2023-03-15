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
from typing import Literal, Optional
import numpy as np

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from verticapy.plotting._matplotlib.base import MatplotlibBase


class HeatMap(MatplotlibBase):

    # Properties.

    @property
    def _category(self) -> Literal["map"]:
        return "map"

    @property
    def _kind(self) -> Literal["heatmap"]:
        return "heatmap"

    @property
    def _compute_method(self) -> Literal["2D"]:
        return "2D"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {"cmap": self.get_cmap(idx=0), "interpolation": "nearest"}
        return None

    # Draw.

    def draw(
        self,
        colorbar: str = "",
        with_numbers: bool = True,
        mround: int = 3,
        extent: Optional[list] = None,
        is_pivot: bool = False,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a heatmap using the Matplotlib API.
        """
        if len(self.data["X"].shape) == 1:
            n, m = self.data["X"].shape[0], 1
            X = np.transpose(np.array([self.data["X"]]))
            is_vector = True
        else:
            n, m = self.data["X"].shape
            X = copy.deepcopy(self.data["X"])
            is_vector = False
        x_labels = list(self.layout["x_labels"])
        y_labels = list(self.layout["y_labels"])
        if is_pivot and not (is_vector):
            np.flip(X, axis=1)
            x_labels.reverse()
        ax, fig = self._get_ax_fig(
            ax, size=(min(m, 500), min(n, 500)), set_axis_below=False, grid=False
        )
        kwargs = {
            **self.init_style,
            "extent": extent,
        }
        if "vmax" in self.layout:
            kwargs["vmax"] = self.layout["vmax"]
        if "vmin" in self.layout:
            kwargs["vmin"] = self.layout["vmin"]
        kwargs = self._update_dict(kwargs, style_kwargs)
        try:
            im = ax.imshow(X, **kwargs)
        except:
            if kwargs["extent"] != None:
                kwargs["extent"] = None
                im = ax.imshow(X, **kwargs)
            else:
                raise
        fig.colorbar(im, ax=ax).set_label(colorbar)
        if not (extent):
            ax.set_yticks([i for i in range(0, n)])
            ax.set_xticks([i for i in range(0, m)])
            ax.set_xticklabels(y_labels, rotation=90)
            ax.set_yticklabels(x_labels, rotation=0)
        if with_numbers:
            X = X.round(mround)
            for y_index in range(n):
                for x_index in range(m):
                    label = X[y_index][x_index]
                    ax.text(
                        x_index, y_index, label, color="black", ha="center", va="center"
                    )
        if "columns" in self.layout:
            if len(self.layout["columns"]) > 0:
                ax.set_ylabel(self.layout["columns"][0])
            if len(self.layout["columns"]) > 1:
                ax.set_xlabel(self.layout["columns"][1])
        return ax
