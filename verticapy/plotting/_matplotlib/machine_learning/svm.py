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
from typing import Optional
import numpy as np

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from verticapy._config.colors import get_colors
import verticapy._config.config as conf
from verticapy._utils._sql._sys import _executeSQL
from verticapy.errors import ParameterError

from verticapy.plotting._matplotlib.base import MatplotlibBase


class SVMClassifierPlot(MatplotlibBase):
    def svm_classifier_plot(
        self,
        X: list,
        y: str,
        input_relation: str,
        coefficients: list,
        max_nb_points: int = 500,
        ax: Optional[Axes] = None,
        **style_kwds,
    ) -> Axes:
        """
        Draws a SVM Classifier plot using the Matplotlib API.
        """
        param0 = {
            "marker": "o",
            "color": get_colors()[0],
            "s": 50,
            "edgecolors": "black",
        }
        param1 = {
            "marker": "o",
            "color": get_colors()[1],
            "s": 50,
            "edgecolors": "black",
        }
        if len(X) == 1:
            query = f"""
                (SELECT 
                    /*+LABEL('plotting._matplotlib.svm_classifier_plot')*/ 
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
            x_svm, y_svm = (
                [
                    -coefficients[0] / coefficients[1],
                    -coefficients[0] / coefficients[1],
                ],
                [-1, 1],
            )
            ax.plot(x_svm, y_svm, alpha=1, color="black")
            all_scatter = [
                ax.scatter(
                    x0, [0 for item in x0], **self.updated_dict(param1, style_kwds, 1)
                )
            ]
            all_scatter += [
                ax.scatter(
                    x1, [0 for item in x1], **self.updated_dict(param0, style_kwds, 0)
                )
            ]
            ax.set_xlabel(X[0])
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
                    /*+LABEL('plotting._matplotlib.svm_classifier_plot')*/ 
                    {X[0]}, 
                    {X[1]}, 
                    {y} 
                 FROM {input_relation} 
                 WHERE {X[0]} IS NOT NULL 
                   AND {X[1]} IS NOT NULL 
                   AND {y} = {{}} 
                 LIMIT {int(max_nb_points / 2)})"""
            all_points = _executeSQL(
                query=f"{query.format(0)} UNION {query.format(1)}",
                method="fetchall",
                print_time_sql=False,
            )
            ax, fig = self._get_ax_fig(ax, size=(8, 6), set_axis_below=True, grid=True)
            x0, x1, y0, y1 = [], [], [], []
            for idx, item in enumerate(all_points):
                if item[2] == 0:
                    x0 += [float(item[0])]
                    y0 += [float(item[1])]
                else:
                    x1 += [float(item[0])]
                    y1 += [float(item[1])]
            min_svm, max_svm = min(x0 + x1), max(x0 + x1)
            x_svm, y_svm = (
                [min_svm, max_svm],
                [
                    -(coefficients[0] + coefficients[1] * min_svm) / coefficients[2],
                    -(coefficients[0] + coefficients[1] * max_svm) / coefficients[2],
                ],
            )
            ax.plot(x_svm, y_svm, alpha=1, color="black")
            all_scatter = [
                ax.scatter(x0, y0, **self.updated_dict(param1, style_kwds, 1))
            ]
            all_scatter += [
                ax.scatter(x1, y1, **self.updated_dict(param0, style_kwds, 0))
            ]
            ax.set_xlabel(X[0])
            ax.set_ylabel(X[1])
            ax.legend(
                all_scatter,
                [0, 1],
                scatterpoints=1,
                loc="center left",
                bbox_to_anchor=[1, 0.5],
            )
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        elif len(X) == 3:
            query = f"""
                (SELECT 
                    /*+LABEL('plotting._matplotlib.svm_classifier_plot')*/ 
                    {X[0]}, 
                    {X[1]}, 
                    {X[2]}, 
                    {y} 
                 FROM {input_relation} 
                 WHERE {X[0]} IS NOT NULL 
                   AND {X[1]} IS NOT NULL 
                   AND {X[2]} IS NOT NULL 
                   AND {y} = {{}} 
                 LIMIT {int(max_nb_points / 2)})"""
            all_points = _executeSQL(
                query=f"{query.format(0)} UNION {query.format(1)}",
                method="fetchall",
                print_time_sql=False,
            )
            x0, x1, y0, y1, z0, z1 = [], [], [], [], [], []
            for idx, item in enumerate(all_points):
                if item[3] == 0:
                    x0 += [float(item[0])]
                    y0 += [float(item[1])]
                    z0 += [float(item[2])]
                else:
                    x1 += [float(item[0])]
                    y1 += [float(item[1])]
                    z1 += [float(item[2])]
            min_svm_x, max_svm_x = min(x0 + x1), max(x0 + x1)
            step_x = (max_svm_x - min_svm_x) / 40.0
            min_svm_y, max_svm_y = min(y0 + y1), max(y0 + y1)
            step_y = (max_svm_y - min_svm_y) / 40.0
            X_svm = (
                np.arange(min_svm_x - 5 * step_x, max_svm_x + 5 * step_x, step_x)
                if (step_x > 0)
                else [max_svm_x]
            )
            Y_svm = (
                np.arange(min_svm_y - 5 * step_y, max_svm_y + 5 * step_y, step_y)
                if (step_y > 0)
                else [max_svm_y]
            )
            X_svm, Y_svm = np.meshgrid(X_svm, Y_svm)
            Z_svm = coefficients[0] + coefficients[1] * X_svm + coefficients[2] * Y_svm
            if not (ax):
                if conf._get_import_success("jupyter"):
                    plt.figure(figsize=(8, 6))
                ax = plt.axes(projection="3d")
            ax.plot_surface(
                X_svm, Y_svm, Z_svm, rstride=1, cstride=1, alpha=0.5, color="gray"
            )
            param0["alpha"] = 0.8
            all_scatter = [
                ax.scatter(x0, y0, z0, **self.updated_dict(param1, style_kwds, 1))
            ]
            all_scatter += [
                ax.scatter(x1, y1, z1, **self.updated_dict(param0, style_kwds, 0))
            ]
            ax.set_xlabel(X[0])
            ax.set_ylabel(X[1])
            ax.set_zlabel(X[2])
            ax.legend(
                all_scatter,
                [0, 1],
                scatterpoints=1,
                title=y,
                loc="center left",
                bbox_to_anchor=[1.1, 0.5],
                ncol=1,
                fontsize=8,
            )
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        else:
            raise ParameterError("The number of predictors is too big.")
        return ax
