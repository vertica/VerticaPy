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

from matplotlib.axes import Axes
import matplotlib.patches as mpatches

from verticapy.plotting._matplotlib.base import MatplotlibBase


class ROCCurve(MatplotlibBase):
    # Properties.

    @property
    def _category(self) -> Literal["graph"]:
        return "graph"

    @property
    def _kind(self) -> Literal["roc"]:
        return "roc"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "alpha": 0.1,
        }

    # Draw.

    def draw(
        self,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a machine learning ROC curve using the Matplotlib API.
        """
        ax, fig, style_kwargs = self._get_ax_fig(
            ax, size=(8, 6), set_axis_below=True, grid=True, style_kwargs=style_kwargs
        )
        ax.plot(
            self.data["x"],
            self.data["y"],
            **self._get_final_style_kwargs(style_kwargs=style_kwargs, idx=0),
        )
        ax.fill_between(
            self.data["x"],
            self.data["x"],
            self.data["y"],
            facecolor=self._get_final_color(style_kwargs=style_kwargs, idx=0),
            **self.init_style,
        )
        ax.plot(
            [0, 1],
            [0, 1],
            **self._get_final_style_kwargs(style_kwargs=style_kwargs, idx=1),
        )
        ax.fill_between(
            [0, 1],
            [0, 0],
            [0, 1],
            facecolor=self._get_final_color(style_kwargs=style_kwargs, idx=1),
            **self.init_style,
        )
        ax.set_xlabel(self.layout["x_label"])
        ax.set_xlim(0, 1)
        ax.set_ylabel(self.layout["y_label"])
        ax.set_ylim(0, 1)
        ax.set_title(self.layout["title"])
        auc = self.data["auc"]
        ax.text(
            0.995,
            0,
            f"AUC = {round(auc, 4) * 100}%",
            verticalalignment="bottom",
            horizontalalignment="right",
            fontsize=11.5,
        )
        return ax


class CutoffCurve(ROCCurve):
    # Properties.

    @property
    def _kind(self) -> Literal["cutoff"]:
        return "cutoff"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "facecolor": "black",
            "alpha": 0.02,
        }

    # Draw.

    def draw(
        self,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a machine cutoff curve using the Matplotlib API.
        """
        ax, fig, style_kwargs = self._get_ax_fig(
            ax, size=(8, 6), set_axis_below=True, grid=True, style_kwargs=style_kwargs
        )
        ax.plot(
            self.data["x"],
            self.data["y"],
            label=self.layout["y_label"],
            **self._get_final_style_kwargs(style_kwargs=style_kwargs, idx=0),
        )
        ax.plot(
            self.data["x"],
            self.data["z"],
            label=self.layout["z_label"],
            **self._get_final_style_kwargs(style_kwargs=style_kwargs, idx=1),
        )
        ax.fill_between(
            self.data["x"],
            self.data["y"],
            self.data["z"],
            **self.init_style,
        )
        ax.set_xlabel(self.layout["x_label"])
        ax.set_ylabel("Values")
        ax.set_title(self.layout["title"])
        ax.legend(loc="center left", bbox_to_anchor=[1, 0.5])
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        return ax


class PRCCurve(ROCCurve):
    # Properties.

    @property
    def _kind(self) -> Literal["prc"]:
        return "prc"

    # Draw.

    def draw(
        self,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a machine learning PRC curve using the Matplotlib API.
        """
        ax, fig, style_kwargs = self._get_ax_fig(
            ax, size=(8, 6), set_axis_below=True, grid=True, style_kwargs=style_kwargs
        )
        kwargs = {"color": self.get_colors(idx=0)}
        ax.plot(
            self.data["x"],
            self.data["y"],
            **self._get_final_style_kwargs(style_kwargs=style_kwargs, idx=0),
        )
        ax.fill_between(
            self.data["x"],
            [0 for x in self.data["x"]],
            self.data["y"],
            facecolor=self._get_final_color(style_kwargs=style_kwargs, idx=0),
            **self.init_style,
        )
        ax.set_xlabel(self.layout["x_label"])
        ax.set_xlim(0, 1)
        ax.set_ylabel(self.layout["y_label"])
        ax.set_ylim(0, 1)
        ax.set_title(self.layout["title"])
        auc = self.data["auc"]
        ax.text(
            0.995,
            0,
            f"AUC = {round(auc, 4) * 100}%",
            verticalalignment="bottom",
            horizontalalignment="right",
            fontsize=11.5,
        )
        return ax


class LiftChart(ROCCurve):
    # Properties.

    @property
    def _kind(self) -> Literal["lift"]:
        return "lift"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "alpha": 0.2,
        }

    # Draw.

    def draw(
        self,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a machine learning lift chart using the Matplotlib API.
        """
        ax, fig, style_kwargs = self._get_ax_fig(
            ax, size=(8, 6), set_axis_below=True, grid=True, style_kwargs=style_kwargs
        )
        ax.set_xlabel(self.layout["x_label"])
        ax.plot(
            self.data["x"],
            self.data["z"],
            **self._get_final_style_kwargs(style_kwargs=style_kwargs, idx=0),
        )
        ax.plot(
            self.data["x"],
            self.data["y"],
            **self._get_final_style_kwargs(style_kwargs=style_kwargs, idx=1),
        )
        ax.fill_between(
            self.data["x"],
            self.data["y"],
            self.data["z"],
            facecolor=self._get_final_color(style_kwargs=style_kwargs, idx=0),
            alpha=0.2,
        )
        ax.fill_between(
            self.data["x"],
            [0 for x in self.data["x"]],
            self.data["y"],
            facecolor=self._get_final_color(style_kwargs=style_kwargs, idx=1),
            **self.init_style,
        )
        ax.set_title(self.layout["title"])
        c0 = mpatches.Patch(
            color=self._get_final_color(style_kwargs=style_kwargs, idx=0),
            label=self.layout["z_label"],
        )
        c1 = mpatches.Patch(
            color=self._get_final_color(style_kwargs=style_kwargs, idx=1),
            label=self.layout["y_label"],
        )
        ax.legend(handles=[c0, c1], loc="center left", bbox_to_anchor=[1, 0.5])
        ax.set_ylabel("Values")
        ax.set_xlim(0, 1)
        ax.set_ylim(0)
        return ax
