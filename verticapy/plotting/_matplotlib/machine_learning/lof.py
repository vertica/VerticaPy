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
from verticapy._typing import PythonNumber, SQLColumns
from verticapy._utils._sql._format import quote_ident
from verticapy._utils._sql._sys import _executeSQL

from verticapy.plotting.base import PlottingBase


class LOFPlot(PlottingBase):
    def lof_plot(
        self,
        input_relation: str,
        columns: SQLColumns,
        lof: str,
        TableSample: PythonNumber = -1,
        ax: Optional[Axes] = None,
        **style_kwds,
    ) -> Axes:
        """
        Draws a Local Outlier Plot using the Matplotlib API.
        """
        if isinstance(columns, str):
            columns = [columns]
        TableSample = f"TABLESAMPLE({TableSample})" if (0 < TableSample < 100) else ""
        colors = []
        if "color" in style_kwds:
            if isinstance(style_kwds["color"], str):
                colors = [style_kwds["color"]]
            else:
                colors = style_kwds["color"]
            del style_kwds["color"]
        elif "colors" in style_kwds:
            if isinstance(style_kwds["colors"], str):
                colors = [style_kwds["colors"]]
            else:
                colors = style_kwds["colors"]
            del style_kwds["colors"]
        colors += get_colors()
        param = {
            "s": 50,
            "edgecolors": "black",
            "color": colors[0],
        }
        if len(columns) == 1:
            column = quote_ident(columns[0])
            query_result = _executeSQL(
                query=f"""
                    SELECT 
                        /*+LABEL('plotting._matplotlib.lof_plot')*/ 
                        {column}, 
                        {lof} 
                    FROM {input_relation} {TableSample} 
                    WHERE {column} IS NOT NULL""",
                method="fetchall",
                print_time_sql=False,
            )
            column1, lof = (
                [item[0] for item in query_result],
                [item[1] for item in query_result],
            )
            column2 = [0] * len(column1)
            if not (ax):
                fig, ax = plt.subplots()
                if conf._get_import_success("jupyter"):
                    fig.set_size_inches(8, 2)
                ax.set_axisbelow(True)
                ax.grid()
            ax.set_xlabel(column)
            radius = [
                2 * 1000 * (item - min(lof)) / (max(lof) - min(lof)) for item in lof
            ]
            ax.scatter(
                column1,
                column2,
                label="Data points",
                **self.updated_dict(param, style_kwds, 0),
            )
            ax.scatter(
                column1,
                column2,
                s=radius,
                label="Outlier scores",
                facecolors="none",
                color=colors[1],
            )
        elif len(columns) == 2:
            columns = [quote_ident(column) for column in columns]
            query_result = _executeSQL(
                query=f"""
                SELECT 
                    /*+LABEL('plotting._matplotlib.lof_plot')*/ 
                    {columns[0]}, 
                    {columns[1]}, 
                    {lof} 
                FROM {input_relation} {TableSample} 
                WHERE {columns[0]} IS NOT NULL 
                  AND {columns[1]} IS NOT NULL""",
                method="fetchall",
                print_time_sql=False,
            )
            column1, column2, lof = (
                [item[0] for item in query_result],
                [item[1] for item in query_result],
                [item[2] for item in query_result],
            )
            if not (ax):
                fig, ax = plt.subplots()
                if conf._get_import_success("jupyter"):
                    fig.set_size_inches(8, 6)
                ax.set_axisbelow(True)
                ax.grid()
            ax.set_ylabel(columns[1])
            ax.set_xlabel(columns[0])
            radius = [1000 * (item - min(lof)) / (max(lof) - min(lof)) for item in lof]
            ax.scatter(
                column1,
                column2,
                label="Data points",
                **self.updated_dict(param, style_kwds, 0),
            )
            ax.scatter(
                column1,
                column2,
                s=radius,
                label="Outlier scores",
                facecolors="none",
                color=colors[1],
            )
        elif len(columns) == 3:
            query_result = _executeSQL(
                query=f"""
                SELECT 
                    /*+LABEL('plotting._matplotlib.lof_plot')*/ 
                    {columns[0]}, 
                    {columns[1]}, 
                    {columns[2]}, 
                    {lof} 
                FROM {input_relation} {TableSample} 
                WHERE {columns[0]} IS NOT NULL 
                  AND {columns[1]} IS NOT NULL 
                  AND {columns[2]} IS NOT NULL""",
                method="fetchall",
                print_time_sql=False,
            )
            column1, column2, column3, lof = (
                [float(item[0]) for item in query_result],
                [float(item[1]) for item in query_result],
                [float(item[2]) for item in query_result],
                [float(item[3]) for item in query_result],
            )
            if not (ax):
                if conf._get_import_success("jupyter"):
                    plt.figure(figsize=(8, 6))
                ax = plt.axes(projection="3d")
            ax.set_xlabel(columns[0])
            ax.set_ylabel(columns[1])
            ax.set_zlabel(columns[2])
            radius = [1000 * (item - min(lof)) / (max(lof) - min(lof)) for item in lof]
            ax.scatter(
                column1,
                column2,
                column3,
                label="Data points",
                **self.updated_dict(param, style_kwds, 0),
            )
            ax.scatter(
                column1, column2, column3, s=radius, facecolors="none", color=colors[1],
            )
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        else:
            raise Exception(
                "LocalOutlierFactor Plot is available for a maximum of 3 columns"
            )
        return ax
