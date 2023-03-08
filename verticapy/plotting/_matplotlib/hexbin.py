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
import statistics
from typing import Optional, TYPE_CHECKING

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from verticapy._config.colors import get_cmap
import verticapy._config.config as conf
from verticapy._typing import SQLColumns
from verticapy._utils._sql._format import quote_ident
from verticapy._utils._sql._sys import _executeSQL
from verticapy.errors import ParameterError

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame

from verticapy.plotting.base import PlottingBase


class HexbinPlot(PlottingBase):
    def hexbin(
        self,
        vdf: "vDataFrame",
        columns: SQLColumns,
        method: str = "count",
        of: str = "",
        bbox: list = [],
        img: str = "",
        ax: Optional[Axes] = None,
        **style_kwds,
    ) -> Axes:
        """
        Draws an hexbin plot using the Matplotlib API.
        """
        if len(columns) != 2:
            raise ParameterError(
                "The parameter 'columns' must be exactly of size 2 to draw the hexbin"
            )
        if method.lower() == "mean":
            method = "avg"
        if (
            (method.lower() in ["avg", "min", "max", "sum"])
            and (of)
            and ((of in vdf.get_columns()) or (quote_ident(of) in vdf.get_columns()))
        ):
            aggregate = f"{method}({of})"
            others_aggregate = method
            if method.lower() == "avg":
                reduce_C_function = statistics.mean
            elif method.lower() == "min":
                reduce_C_function = min
            elif method.lower() == "max":
                reduce_C_function = max
            elif method.lower() == "sum":
                reduce_C_function = sum
        elif method.lower() in ("count", "density"):
            aggregate = "count(*)"
            reduce_C_function = sum
        else:
            raise ParameterError(
                "The parameter 'method' must be in [avg|mean|min|max|sum|median]"
            )
        count = vdf.shape()[0]
        if method.lower() == "density":
            over = "/" + str(float(count))
        else:
            over = ""
        query_result = _executeSQL(
            query=f"""
                SELECT
                    /*+LABEL('plotting._matplotlib.hexbin')*/
                    {columns[0]},
                    {columns[1]},
                    {aggregate}{over}
                FROM {vdf._genSQL()}
                GROUP BY {columns[0]}, {columns[1]}""",
            title="Grouping all the elements for the Hexbin Plot",
            method="fetchall",
        )
        column1, column2, column3 = [], [], []
        for item in query_result:
            if (item[0] != None) and (item[1] != None) and (item[2] != None):
                column1 += [float(item[0])] * 2
                column2 += [float(item[1])] * 2
                if reduce_C_function in [min, max, statistics.mean]:
                    column3 += [float(item[2])] * 2
                else:
                    column3 += [float(item[2]) / 2] * 2
        if not (ax):
            fig, ax = plt.subplots()
            if conf._get_import_success("jupyter"):
                fig.set_size_inches(9, 7)
            ax.set_facecolor("white")
        else:
            fig = plt
        if bbox:
            ax.set_xlim(bbox[0], bbox[1])
            ax.set_ylim(bbox[2], bbox[3])
        if img:
            im = plt.imread(img)
            if not (bbox):
                bbox = (min(column1), max(column1), min(column2), max(column2))
                ax.set_xlim(bbox[0], bbox[1])
                ax.set_ylim(bbox[2], bbox[3])
            ax.imshow(im, extent=bbox)
        ax.set_ylabel(columns[1])
        ax.set_xlabel(columns[0])
        param = {"cmap": get_cmap()[0], "gridsize": 10, "mincnt": 1, "edgecolors": None}
        imh = ax.hexbin(
            column1,
            column2,
            C=column3,
            reduce_C_function=reduce_C_function,
            **self.updated_dict(param, style_kwds),
        )
        if method.lower() == "density":
            fig.colorbar(imh).set_label(method)
        else:
            fig.colorbar(imh).set_label(aggregate)
        return ax
