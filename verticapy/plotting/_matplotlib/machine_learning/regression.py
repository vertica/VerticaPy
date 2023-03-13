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
import matplotlib.pyplot as plt

import verticapy._config.config as conf
from verticapy._utils._sql._sys import _executeSQL
from verticapy.errors import ParameterError

from verticapy.plotting._matplotlib.base import MatplotlibBase


class RegressionPlot(MatplotlibBase):
    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["regression"]:
        return "regression"

    def draw(
        self,
        X: list,
        y: str,
        input_relation: str,
        coefficients: list,
        max_nb_points: int = 50,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a regression plot using the Matplotlib API.
        """
        param = {
            "marker": "o",
            "color": self.get_colors(idx=0),
            "s": 50,
            "edgecolors": "black",
        }
        if len(X) == 1:
            all_points = _executeSQL(
                query=f"""
                SELECT 
                    /*+LABEL('plotting._matplotlib.regression_plot')*/ 
                    {X[0]}, 
                    {y} 
                FROM {input_relation} 
                WHERE {X[0]} IS NOT NULL 
                  AND {y} IS NOT NULL LIMIT {int(max_nb_points)}""",
                method="fetchall",
                print_time_sql=False,
            )
            ax, fig = self._get_ax_fig(ax, size=(8, 6), set_axis_below=True, grid=True)
            x0, y0 = (
                [float(item[0]) for item in all_points],
                [float(item[1]) for item in all_points],
            )
            min_reg, max_reg = min(x0), max(x0)
            x_reg = [min_reg, max_reg]
            y_reg = [coefficients[0] + coefficients[1] * item for item in x_reg]
            ax.plot(x_reg, y_reg, alpha=1, color="black")
            ax.scatter(
                x0, y0, **self._update_dict(param, style_kwargs, 0),
            )
            ax.set_xlabel(X[0])
            ax.set_ylabel(y)
        elif len(X) == 2:
            all_points = _executeSQL(
                query=f"""
                (SELECT 
                    /*+LABEL('plotting._matplotlib.regression_plot')*/ 
                    {X[0]}, 
                    {X[1]}, 
                    {y} 
                 FROM {input_relation} 
                 WHERE {X[0]} IS NOT NULL 
                   AND {X[1]} IS NOT NULL 
                   AND {y} IS NOT NULL 
                 LIMIT {int(max_nb_points)})""",
                method="fetchall",
                print_time_sql=False,
            )
            x0, y0, z0 = (
                [float(item[0]) for item in all_points],
                [float(item[1]) for item in all_points],
                [float(item[2]) for item in all_points],
            )
            min_reg_x, max_reg_x = min(x0), max(x0)
            step_x = (max_reg_x - min_reg_x) / 40.0
            min_reg_y, max_reg_y = min(y0), max(y0)
            step_y = (max_reg_y - min_reg_y) / 40.0
            X_reg = (
                np.arange(min_reg_x - 5 * step_x, max_reg_x + 5 * step_x, step_x)
                if (step_x > 0)
                else [max_reg_x]
            )
            Y_reg = (
                np.arange(min_reg_y - 5 * step_y, max_reg_y + 5 * step_y, step_y)
                if (step_y > 0)
                else [max_reg_y]
            )
            X_reg, Y_reg = np.meshgrid(X_reg, Y_reg)
            Z_reg = coefficients[0] + coefficients[1] * X_reg + coefficients[2] * Y_reg
            if not (ax):
                if conf._get_import_success("jupyter"):
                    plt.figure(figsize=(8, 6))
                ax = plt.axes(projection="3d")
            ax.plot_surface(
                X_reg, Y_reg, Z_reg, rstride=1, cstride=1, alpha=0.5, color="gray"
            )
            ax.scatter(
                x0, y0, z0, **self._update_dict(param, style_kwargs, 0),
            )
            ax.set_xlabel(X[0])
            ax.set_ylabel(X[1])
            ax.set_zlabel(y + " = f(" + X[0] + ", " + X[1] + ")")
        else:
            raise ParameterError("The number of predictors is too big.")
        return ax
