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
import matplotlib.pyplot as plt

from verticapy._utils._sql._format import format_type

from verticapy.plotting._matplotlib.base import MatplotlibBase


class HexbinMap(MatplotlibBase):
    # Properties.

    @property
    def _category(self) -> Literal["map"]:
        return "map"

    @property
    def _kind(self) -> Literal["hexbin"]:
        return "hexbin"

    @property
    def _compute_method(self) -> Literal["aggregate"]:
        return "aggregate"

    @property
    def _dimension_bounds(self) -> tuple[int, int]:
        return (2, 2)

    @property
    def _only_standard(self) -> Literal[True]:
        return True

    # Styling Methods.

    def _init_style(self) -> None:
        """Must be overridden in child class"""
        self.init_style = {
            "cmap": self.get_cmap(idx=0),
            "gridsize": 10,
            "mincnt": 1,
            "edgecolors": None,
        }

    # Draw.

    def draw(
        self,
        bbox: Optional[list] = None,
        img: Optional[str] = None,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a hexbin plot using the Matplotlib API.
        """
        bbox = format_type(bbox, dtype=list)
        matrix = self.data["X"]
        matrix = matrix[(matrix != np.array(None)).all(axis=1)].astype(float)
        ax, fig, style_kwargs = self._get_ax_fig(
            ax, size=(9, 7), set_axis_below=False, grid=False, style_kwargs=style_kwargs
        )
        if bbox:
            ax.set_xlim(bbox[0], bbox[1])
            ax.set_ylim(bbox[2], bbox[3])
        if img:
            im = plt.imread(img)
            if not bbox:
                bbox = (
                    min(matrix[:, 0]),
                    max(matrix[:, 0]),
                    min(matrix[:, 1]),
                    max(matrix[:, 1]),
                )
                ax.set_xlim(bbox[0], bbox[1])
                ax.set_ylim(bbox[2], bbox[3])
            ax.imshow(im, extent=bbox)
        ax.set_xlabel(self.layout["columns"][0])
        ax.set_ylabel(self.layout["columns"][1])
        imh = ax.hexbin(
            matrix[:, 0],
            matrix[:, 1],
            C=matrix[:, 2],
            reduce_C_function=self.layout["aggregate_fun"],
            **self._update_dict(self.init_style, style_kwargs),
        )
        fig.colorbar(imh).set_label(self.layout["method_of"])
        return ax
