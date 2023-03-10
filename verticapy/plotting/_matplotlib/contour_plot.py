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
from typing import Callable, Optional, TYPE_CHECKING
import numpy as np

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from verticapy._config.colors import get_cmap, get_colors
from verticapy._typing import PythonScalar, SQLColumns
from verticapy._utils._sql._format import quote_ident

from verticapy.core.string_sql.base import StringSQL

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame

from verticapy.plotting._matplotlib.base import MatplotlibBase


class ContourPlot(MatplotlibBase):
    def contour_plot(
        self,
        vdf: "vDataFrame",
        columns: SQLColumns,
        func: Callable,
        nbins: int = 100,
        cbar_title: str = "",
        pos_label: PythonScalar = None,
        ax: Optional[Axes] = None,
        **style_kwds,
    ) -> Axes:
        """
        Draws a contour plot using the Matplotlib API.
        """

        from verticapy.datasets.generators import gen_meshgrid

        if not (cbar_title) and str(type(func)) in (
            "<class 'function'>",
            "<class 'method'>",
        ):
            cbar_title = func.__name__
        all_agg = vdf.agg(["min", "max"], columns)
        min_x, min_y = all_agg["min"]
        max_x, max_y = all_agg["max"]
        if str(type(func)) in ("<class 'function'>", "<class 'method'>"):
            xlist = np.linspace(min_x, max_x, nbins)
            ylist = np.linspace(min_y, max_y, nbins)
            X, Y = np.meshgrid(xlist, ylist)
            Z = func(X, Y)
        else:
            vdf_tmp = gen_meshgrid(
                {
                    quote_ident(columns[1])[1:-1]: {
                        "type": float,
                        "range": [min_y, max_y],
                        "nbins": nbins,
                    },
                    quote_ident(columns[0])[1:-1]: {
                        "type": float,
                        "range": [min_x, max_x],
                        "nbins": nbins,
                    },
                }
            )
            y = "verticapy_predict"
            if isinstance(func, (str, StringSQL)):
                vdf_tmp["verticapy_predict"] = func
            else:
                if func._model_type in (
                    "XGBClassifier",
                    "RandomForestClassifier",
                    "NaiveBayes",
                    "NearestCentroid",
                    "KNeighborsClassifier",
                ):
                    if func._model_type == "KNeighborsClassifier":
                        vdf_tmp = func._predict_proba(
                            vdf=vdf_tmp,
                            X=columns,
                            name="verticapy_predict",
                            inplace=False,
                            key_columns=None,
                        )
                        y = f"verticapy_predict_{pos_label}"
                    else:
                        vdf_tmp = func.predict_proba(
                            vdf=vdf_tmp,
                            X=columns,
                            name="verticapy_predict",
                            pos_label=pos_label,
                        )
                else:
                    if func._model_type == "KNeighborsRegressor":
                        vdf_tmp = func._predict(
                            vdf=vdf_tmp,
                            X=columns,
                            name="verticapy_predict",
                            inplace=False,
                            key_columns=None,
                        )
                    else:
                        vdf_tmp = func.predict(
                            vdf=vdf_tmp, X=columns, name="verticapy_predict"
                        )
            dataset = vdf_tmp[[columns[1], columns[0], y]].sort(columns).to_numpy()
            i, y_start, y_new = 0, dataset[0][1], dataset[0][1]
            n = len(dataset)
            X, Y, Z = [], [], []
            while i < n:
                x_tmp, y_tmp, z_tmp = [], [], []
                j, last_non_null_value = 0, 0
                while y_start == y_new and i < n and j < nbins:
                    if dataset[i][2] != None:
                        last_non_null_value = float(dataset[i][2])
                    x_tmp += [float(dataset[i][0])]
                    y_tmp += [float(dataset[i][1])]
                    z_tmp += [
                        float(
                            dataset[i][2]
                            if (dataset[i][2] != None)
                            else last_non_null_value
                        )
                    ]
                    y_new = dataset[i][1]
                    j += 1
                    i += 1
                    if j == nbins:
                        while y_start == y_new and i < n:
                            y_new = dataset[i][1]
                            i += 1
                y_start = y_new
                X += [x_tmp]
                Y += [y_tmp]
                Z += [z_tmp]
            X, Y, Z = np.array(Y), np.array(X), np.array(Z)
        ax, fig = self._get_ax_fig(ax, size=(8, 6), set_axis_below=False, grid=False)
        param = {"linewidths": 0.5, "levels": 14, "colors": "k"}
        param = self.updated_dict(param, style_kwds)
        if "cmap" in param:
            del param["cmap"]
        ax.contour(X, Y, Z, **param)
        param = {
            "cmap": get_cmap([get_colors()[2], "#FFFFFF", get_colors()[0]]),
            "levels": 14,
        }
        param = self.updated_dict(param, style_kwds)
        for elem in ["colors", "color", "linewidths", "linestyles"]:
            if elem in param:
                del param[elem]
        cp = ax.contourf(X, Y, Z, **param)
        fig.colorbar(cp).set_label(cbar_title)
        ax.set_xlabel(columns[0])
        ax.set_ylabel(columns[1])
        return ax
