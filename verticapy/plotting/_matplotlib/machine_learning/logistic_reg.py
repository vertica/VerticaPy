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
import math
from typing import Optional
import numpy as np

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from verticapy._config.colors import get_colors
import verticapy._config.config as conf
from verticapy._typing import ArrayLike, SQLColumns
from verticapy._utils._sql._sys import _executeSQL
from verticapy.errors import ParameterError

from verticapy.plotting._matplotlib.base import MatplotlibBase


class LogisticRegressionPlot(MatplotlibBase):
    def logit_plot(
        self,
        X: SQLColumns,
        y: str,
        input_relation: str,
        coefficients: ArrayLike,
        max_nb_points: int = 50,
        ax: Optional[Axes] = None,
        **style_kwds,
    ) -> Axes:
        """
        Draws a Logistic Regression plot using the Matplotlib API.
        """
        if isinstance(X, str):
            X = [X]
        param0 = {
            "marker": "o",
            "s": 50,
            "color": get_colors()[0],
            "edgecolors": "black",
            "alpha": 0.8,
        }
        param1 = {
            "marker": "o",
            "s": 50,
            "color": get_colors()[1],
            "edgecolors": "black",
        }

        def logit(x):
            return 1 / (1 + math.exp(-x))

        if len(X) == 1:
            query = f"""
                (SELECT 
                    /*+LABEL('plotting._matplotlib.logit_plot')*/ 
                    {X[0]}, 
                    {y} 
                 FROM {input_relation} 
                 WHERE {X[0]} IS NOT NULL 
                   AND {y} = {{}} 
                LIMIT {int(max_nb_points / 2)})"""
            all_points = _executeSQL(
                query=f"{query.format(0)} UNION ALL {query.format(1)}",
                method="fetchall",
                print_time_sql=False,
            )
            ax, fig = self._get_ax_fig(ax, size=(8, 6), set_axis_below=True, grid=True)
            x0, x1 = [], []
            for idx, item in enumerate(all_points):
                if item[1] == 0:
                    x0 += [float(item[0])]
                else:
                    x1 += [float(item[0])]
            min_logit, max_logit = min(x0 + x1), max(x0 + x1)
            step = (max_logit - min_logit) / 40.0
            x_logit = (
                np.arange(min_logit - 5 * step, max_logit + 5 * step, step)
                if (step > 0)
                else [max_logit]
            )
            y_logit = [
                logit(coefficients[0] + coefficients[1] * item) for item in x_logit
            ]
            ax.plot(x_logit, y_logit, alpha=1, color="black")
            all_scatter = [
                ax.scatter(
                    x0,
                    [logit(coefficients[0] + coefficients[1] * item) for item in x0],
                    **self.updated_dict(param1, style_kwds, 1),
                )
            ]
            all_scatter += [
                ax.scatter(
                    x1,
                    [logit(coefficients[0] + coefficients[1] * item) for item in x1],
                    **self.updated_dict(param0, style_kwds, 0),
                )
            ]
            ax.set_xlabel(X[0])
            ax.set_ylabel(y)
            ax.legend(
                all_scatter,
                [0, 1],
                scatterpoints=1,
                loc="center left",
                bbox_to_anchor=[1, 0.5],
            )
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        elif len(X) == 2:
            query = f"""
                (SELECT 
                    /*+LABEL('plotting._matplotlib.logit_plot')*/ 
                    {X[0]}, 
                    {X[1]}, 
                    {y} 
                 FROM {input_relation} 
                 WHERE {X[0]} IS NOT NULL
                   AND {X[1]} IS NOT NULL
                   AND {y} = {{}} LIMIT {int(max_nb_points / 2)})"""
            all_points = _executeSQL(
                query=f"{query.format(0)} UNION {query.format(1)}",
                method="fetchall",
                print_time_sql=False,
            )
            x0, x1, y0, y1 = [], [], [], []
            for idx, item in enumerate(all_points):
                if item[2] == 0:
                    x0 += [float(item[0])]
                    y0 += [float(item[1])]
                else:
                    x1 += [float(item[0])]
                    y1 += [float(item[1])]
            min_logit_x, max_logit_x = min(x0 + x1), max(x0 + x1)
            step_x = (max_logit_x - min_logit_x) / 40.0
            min_logit_y, max_logit_y = min(y0 + y1), max(y0 + y1)
            step_y = (max_logit_y - min_logit_y) / 40.0
            X_logit = (
                np.arange(min_logit_x - 5 * step_x, max_logit_x + 5 * step_x, step_x)
                if (step_x > 0)
                else [max_logit_x]
            )
            Y_logit = (
                np.arange(min_logit_y - 5 * step_y, max_logit_y + 5 * step_y, step_y)
                if (step_y > 0)
                else [max_logit_y]
            )
            X_logit, Y_logit = np.meshgrid(X_logit, Y_logit)
            Z_logit = 1 / (
                1
                + np.exp(
                    -(
                        coefficients[0]
                        + coefficients[1] * X_logit
                        + coefficients[2] * Y_logit
                    )
                )
            )
            if not (ax):
                if conf._get_import_success("jupyter"):
                    plt.figure(figsize=(8, 6))
                ax = plt.axes(projection="3d")
            ax.plot_surface(
                X_logit, Y_logit, Z_logit, rstride=1, cstride=1, alpha=0.5, color="gray"
            )
            all_scatter = [
                ax.scatter(
                    x0,
                    y0,
                    [
                        logit(
                            coefficients[0]
                            + coefficients[1] * x0[i]
                            + coefficients[2] * y0[i]
                        )
                        for i in range(len(x0))
                    ],
                    **self.updated_dict(param1, style_kwds, 1),
                )
            ]
            all_scatter += [
                ax.scatter(
                    x1,
                    y1,
                    [
                        logit(
                            coefficients[0]
                            + coefficients[1] * x1[i]
                            + coefficients[2] * y1[i]
                        )
                        for i in range(len(x1))
                    ],
                    **self.updated_dict(param0, style_kwds, 0),
                )
            ]
            ax.set_xlabel(X[0])
            ax.set_ylabel(X[1])
            ax.set_zlabel(y)
            ax.legend(
                all_scatter,
                [0, 1],
                scatterpoints=1,
                loc="center left",
                bbox_to_anchor=[1.1, 0.5],
                title=y,
                ncol=2,
                fontsize=8,
            )
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        else:
            raise ParameterError("The number of predictors is too big.")
        return ax
