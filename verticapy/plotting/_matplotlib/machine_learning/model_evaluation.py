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

from matplotlib.axes import Axes
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

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
        return None

    # Draw.

    def draw(self, ax: Optional[Axes] = None, **style_kwargs,) -> Axes:
        """
        Draws a Machine Learning Bubble Plot using the Matplotlib API.
        """
        ax, fig = self._get_ax_fig(ax, size=(8, 6), set_axis_below=True, grid=True,)
        ax.set_xlabel("False Positive Rate (1-Specificity)")
        ax.set_ylabel("True Positive Rate (Sensitivity)")
        kwargs = {"color": self.get_colors(idx=0)}
        ax.plot(
            self.data["x"],
            self.data["y"],
            **self._update_dict(kwargs, style_kwargs, 0),
        )
        ax.fill_between(
            self.data["x"],
            self.data["x"],
            self.data["y"],
            facecolor=self.get_colors(idx=0),
            **self.init_style,
        )
        ax.fill_between(
            [0, 1], [0, 0], [0, 1], facecolor=self.get_colors(idx=1), **self.init_style
        )
        ax.plot([0, 1], [0, 1], color=self.get_colors(idx=1))
        ax.set_title("ROC Curve")
        auc = self.data["auc"]
        ax.text(
            0.995,
            0,
            f"AUC = {round(auc, 4) * 100}%",
            verticalalignment="bottom",
            horizontalalignment="right",
            fontsize=11.5,
        )
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
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
        return None

    # Draw.

    def draw(self, ax: Optional[Axes] = None, **style_kwargs,) -> Axes:
        """
        Draws a Machine Learning Bubble Plot using the Matplotlib API.
        """
        ax, fig = self._get_ax_fig(ax, size=(8, 6), set_axis_below=True, grid=True,)
        kwargs = {"color": self.get_colors(idx=0)}
        ax.plot(
            self.data["x"],
            1 - self.data["y"],
            label="Specificity",
            **self._update_dict(kwargs, style_kwargs, 0),
        )
        kwargs = {"color": self.get_colors(idx=1)}
        ax.plot(
            self.data["x"],
            self.data["z"],
            label="Sensitivity",
            **self._update_dict(kwargs, style_kwargs, 1),
        )
        ax.fill_between(
            self.data["x"], 1 - self.data["y"], self.data["z"], **self.init_style,
        )
        ax.set_xlabel("Decision Boundary")
        ax.set_title("Cutoff Curve")
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

    def draw(self, ax: Optional[Axes] = None, **style_kwargs,) -> Axes:
        """
        Draws a Machine Learning Bubble Plot using the Matplotlib API.
        """
        ax, fig = self._get_ax_fig(ax, size=(8, 6), set_axis_below=True, grid=True,)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        kwargs = {"color": self.get_colors(idx=0)}
        ax.plot(
            self.data["x"],
            self.data["y"],
            **self._update_dict(kwargs, style_kwargs, 0),
        )
        ax.fill_between(
            self.data["x"],
            [0 for x in self.data["x"]],
            self.data["y"],
            facecolor=self.get_colors(idx=0),
            **self._init_style,
        )
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_title("PRC Curve")
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


class LiftChart(MatplotlibBase):

    # Properties.

    @property
    def _category(self) -> Literal["graph"]:
        return "graph"

    @property
    def _kind(self) -> Literal["lift"]:
        return "lift"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "alpha": 0.2,
        }
        return None

    # Draw.

    def draw(self, ax: Optional[Axes] = None, **style_kwargs,) -> Axes:
        """
        Draws a Machine Learning Bubble Plot using the Matplotlib API.
        """
        ax, fig = self._get_ax_fig(ax, size=(8, 6), set_axis_below=True, grid=True,)
        ax.set_xlabel("Cumulative Data Fraction")
        color1, color2 = self.get_colors(idx=0), self.get_colors(idx=1)
        kwargs = {"color": color1}
        ax.plot(self.data["x"], self.data["z"], **self._update_dict(kwargs, style_kwargs, 0))
        kwargs = {"color": color2}
        ax.plot(
            self.data["x"],
            self.data["y"],
            **self._update_dict(kwargs, style_kwargs, 1),
        )
        ax.fill_between(
            self.data["x"], self.data["y"], self.data["z"], facecolor=color1, alpha=0.2,
        )
        ax.fill_between(
            self.data["x"],
            [0 for x in self.data["x"]],
            self.data["y"],
            facecolor=color2,
            **self.init_style,
        )
        ax.set_title("Lift Table")
        color1 = mpatches.Patch(color=color1, label="Cumulative Lift")
        color2 = mpatches.Patch(color=color2, label="Cumulative Capture Rate")
        ax.legend(handles=[color1, color2], loc="center left", bbox_to_anchor=[1, 0.5])
        ax.set_xlim(0, 1)
        ax.set_ylim(0)
        return ax
