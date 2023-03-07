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

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from verticapy._config.colors import get_colors
import verticapy._config.config as conf
from verticapy._utils._sql._sys import _executeSQL

from verticapy.plotting.base import PlottingBase


class RegressionTreePlot(PlottingBase):
    def regression_tree_plot(
        self,
        X: list,
        y: str,
        input_relation: str,
        max_nb_points: int = 10000,
        ax: Optional[Axes] = None,
        **style_kwds,
    ) -> Axes:
        """
        Draws a regression tree plot using the Matplotlib API.
        """
        all_points = _executeSQL(
            query=f"""
            SELECT 
                /*+LABEL('plotting._matplotlib.regression_tree_plot')*/ 
                {X[0]}, 
                {X[1]}, 
                {y} 
            FROM {input_relation} 
            WHERE {X[0]} IS NOT NULL 
              AND {X[1]} IS NOT NULL 
              AND {y} IS NOT NULL 
            ORDER BY RANDOM() 
            LIMIT {int(max_nb_points)}""",
            method="fetchall",
            print_time_sql=False,
        )
        if not (ax):
            fig, ax = plt.subplots()
            if conf._get_import_success("jupyter"):
                fig.set_size_inches(8, 6)
            ax.set_axisbelow(True)
            ax.grid()
        x0, x1, y0, y1 = (
            [float(item[0]) for item in all_points],
            [float(item[0]) for item in all_points],
            [float(item[2]) for item in all_points],
            [float(item[1]) for item in all_points],
        )
        x0, y0 = zip(*sorted(zip(x0, y0)))
        x1, y1 = zip(*sorted(zip(x1, y1)))
        color = "black"
        if "color" in style_kwds:
            if (
                not (isinstance(style_kwds["color"], str))
                and len(style_kwds["color"]) > 1
            ):
                color = style_kwds["color"][1]
        ax.step(x1, y1, color=color)
        param = {
            "marker": "o",
            "color": get_colors()[0],
            "s": 50,
            "edgecolors": "black",
        }
        ax.scatter(x0, y0, **self.updated_dict(param, style_kwds))
        ax.set_xlabel(X[0])
        ax.set_ylabel(y)
        return ax
