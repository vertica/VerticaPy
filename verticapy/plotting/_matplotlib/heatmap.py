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
import copy
from typing import Literal, Optional

import numpy as np

from matplotlib.axes import Axes

from verticapy._typing import NoneType

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
        self.init_style_text = {"color": "black", "ha": "center", "va": "center"}

    def _get_cmap_style(self, style_kwargs: dict) -> dict:
        if (
            "cmap" not in style_kwargs
            and "method" in self.layout
            and (
                self.layout["method"]
                in (
                    "pearson",
                    "spearman",
                    "spearmand",
                    "kendall",
                    "biserial",
                )
            )
        ):
            return {"cmap": self.get_cmap(idx=1)}
        elif "cmap" not in style_kwargs:
            return {"cmap": self.get_cmap(idx=0)}
        else:
            return {}

    # Draw.

    def draw(
        self,
        colorbar: Optional[str] = None,
        extent: Optional[list] = None,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a heatmap using the Matplotlib API.
        """
        X = copy.deepcopy(self.data["X"])
        x_labels = list(self.layout["x_labels"])
        y_labels = list(self.layout["y_labels"])
        n, m = self.data["X"].shape
        ax, fig, style_kwargs = self._get_ax_fig(
            ax,
            size=(min(n, 500), min(m, 500)),
            set_axis_below=False,
            grid=False,
            style_kwargs=style_kwargs,
        )
        kwargs = {
            **self.init_style,
            **self._get_cmap_style(style_kwargs=style_kwargs),
            "extent": extent,
        }
        if "vmax" in self.layout:
            kwargs["vmax"] = self.layout["vmax"]
        if "vmin" in self.layout:
            kwargs["vmin"] = self.layout["vmin"]
        kwargs = self._update_dict(kwargs, style_kwargs)
        try:
            im = ax.imshow(np.transpose(X), **kwargs)
        except:
            if not isinstance(kwargs["extent"], NoneType):
                kwargs["extent"] = None
                im = ax.imshow(np.transpose(X), **kwargs)
            else:
                raise
        fig.colorbar(im, ax=ax).set_label(colorbar)
        if not extent:
            ax.set_xticks([i for i in range(0, n)])
            ax.set_yticks([i for i in range(0, m)])
            ax.set_xticklabels(x_labels, rotation=90)
            ax.set_yticklabels(y_labels, rotation=0)
        if "with_numbers" in self.layout and self.layout["with_numbers"]:
            X = X.round(self.layout["mround"])
            for x_index in range(n):
                for y_index in range(m):
                    label = X[x_index][y_index]
                    ax.text(
                        x_index,
                        y_index,
                        label,
                        **self.init_style_text,
                    )
        ax.set_xlabel(self.layout["columns"][0])
        if len(self.layout["columns"]) > 1:
            ax.set_ylabel(self.layout["columns"][1])
        return ax
