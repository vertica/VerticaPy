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
import warnings
from typing import Optional, TYPE_CHECKING

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from verticapy._config.colors import get_colors
import verticapy._config.config as conf
from verticapy._typing import SQLColumns
from verticapy.errors import ParameterError

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame

from verticapy.plotting.base import PlottingBase


class Histogram(PlottingBase):
    def hist(
        self,
        vdf: "vDataFrame",
        method: str = "density",
        of: Optional[str] = None,
        max_cardinality: int = 6,
        nbins: int = 0,
        h: float = 0.0,
        ax: Optional[Axes] = None,
        **style_kwds,
    ) -> Axes:
        """
        Draws a histogram using the Matplotlib API.
        """
        x, y, z, h, is_categorical = self._compute_plot_params(
            vdf, method, of, max_cardinality, nbins, h
        )
        is_numeric = vdf.isnum()
        if not (ax):
            fig, ax = plt.subplots()
            if conf._get_import_success("jupyter"):
                fig.set_size_inches(min(int(len(x) / 1.8) + 1, 600), 6)
            ax.set_axisbelow(True)
            ax.yaxis.grid()
        param = {"color": get_colors()[0], "alpha": 0.86}
        ax.bar(x, y, h, **self.updated_dict(param, style_kwds))
        ax.set_xlabel(vdf._alias)
        if is_categorical:
            if not (is_numeric):
                new_z = []
                for item in z:
                    new_z += [item[0:47] + "..."] if (len(str(item)) > 50) else [item]
            else:
                new_z = z
            ax.set_xticks(x)
            ax.set_xticklabels(new_z, rotation=90)
        else:
            L = [elem - round(h / 2 / 0.94, 10) for elem in x]
            ax.set_xticks(L)
            ax.set_xticklabels(L, rotation=90)
        if method.lower() == "density":
            ax.set_ylabel("Density")
        elif (
            method.lower() in ["avg", "min", "max", "sum", "mean"]
            or ("%" == method[-1])
        ) and (of != None):
            aggregate = f"{method}({of})"
            ax.set_ylabel(method)
        elif method.lower() == "count":
            ax.set_ylabel("Frequency")
        else:
            ax.set_ylabel(method)
        return ax

    def hist2D(
        self,
        vdf: "vDataFrame",
        columns: SQLColumns,
        method: str = "density",
        of: str = "",
        max_cardinality: tuple[int, int] = (6, 6),
        h: tuple[Optional[float], Optional[float]] = (None, None),
        stacked: bool = False,
        ax: Optional[Axes] = None,
        **style_kwds,
    ) -> Axes:
        """
        Draws a 2D Histogram using the Matplotlib API.
        """
        if isinstance(columns, str):
            columns = [columns]
        colors = get_colors()
        all_columns = vdf.pivot_table(
            columns,
            method=method,
            of=of,
            h=h,
            max_cardinality=max_cardinality,
            show=False,
        ).values
        all_columns = [[column] + all_columns[column] for column in all_columns]
        n, m = len(all_columns), len(all_columns[0])
        n_groups = m - 1
        bar_width = 0.5
        if not (ax):
            fig, ax = plt.subplots()
            if conf._get_import_success("jupyter"):
                fig.set_size_inches(min(600, 3 * m) / 2 + 1, 6)
            ax.set_axisbelow(True)
            ax.yaxis.grid()
        for i in range(1, n):
            current_column = all_columns[i][1:m]
            for idx, item in enumerate(current_column):
                try:
                    current_column[idx] = float(item)
                except:
                    current_column[idx] = 0
            current_label = str(all_columns[i][0])
            param = {
                "alpha": 0.86,
                "color": colors[(i - 1) % len(colors)],
            }
            if stacked:
                if i == 1:
                    last_column = [0 for item in all_columns[i][1:m]]
                else:
                    for idx, item in enumerate(all_columns[i - 1][1:m]):
                        try:
                            last_column[idx] += float(item)
                        except:
                            last_column[idx] += 0
                ax.bar(
                    [elem for elem in range(n_groups)],
                    current_column,
                    bar_width,
                    label=current_label,
                    bottom=last_column,
                    **self.updated_dict(param, style_kwds, i - 1),
                )
            else:
                ax.bar(
                    [elem + (i - 1) * bar_width / (n - 1) for elem in range(n_groups)],
                    current_column,
                    bar_width / (n - 1),
                    label=current_label,
                    **self.updated_dict(param, style_kwds, i - 1),
                )
        if stacked:
            ax.set_xticks([elem for elem in range(n_groups)])
            ax.set_xticklabels(all_columns[0][1:m], rotation=90)
        else:
            ax.set_xticks(
                [
                    elem + bar_width / 2 - bar_width / 2 / (n - 1)
                    for elem in range(n_groups)
                ],
            )
            ax.set_xticklabels(all_columns[0][1:m], rotation=90)
        ax.set_xlabel(columns[0])
        if method.lower() == "mean":
            method = "avg"
        if method.lower() == "density":
            ax.set_ylabel("Density")
        elif (method.lower() in ["avg", "min", "max", "sum"]) and (of != None):
            ax.set_ylabel(f"{method}({of})")
        elif method.lower() == "count":
            ax.set_ylabel("Frequency")
        else:
            ax.set_ylabel(method)
        ax.legend(title=columns[1], loc="center left", bbox_to_anchor=[1, 0.5])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        return ax

    def multiple_hist(
        self,
        vdf: "vDataFrame",
        columns: SQLColumns,
        method: str = "density",
        of: str = "",
        h: float = 0.0,
        ax: Optional[Axes] = None,
        **style_kwds,
    ) -> Axes:
        """
        Draws a muli-histogram chart using the Matplotlib API.
        """
        if isinstance(columns, str):
            columns = [columns]
        colors = get_colors()
        if len(columns) > 5:
            raise ParameterError(
                "The number of column must be <= 5 to use 'multiple_hist' method"
            )
        else:
            if not (ax):
                fig, ax = plt.subplots()
                if conf._get_import_success("jupyter"):
                    fig.set_size_inches(8, 6)
                ax.set_axisbelow(True)
                ax.yaxis.grid()
            alpha, all_columns, all_h = 1, [], []
            if h <= 0:
                for idx, column in enumerate(columns):
                    all_h += [vdf[column].numh()]
                h = min(all_h)
            for idx, column in enumerate(columns):
                if vdf[column].isnum():
                    [x, y, z, h, is_categorical] = self._compute_plot_params(
                        vdf[column], method=method, of=of, max_cardinality=1, h=h
                    )
                    h = h / 0.94
                    param = {"color": colors[idx % len(colors)]}
                    plt.bar(
                        x,
                        y,
                        h,
                        label=column,
                        alpha=alpha,
                        **self.updated_dict(param, style_kwds, idx),
                    )
                    alpha -= 0.2
                    all_columns += [columns[idx]]
                else:
                    if vdf._vars["display"]["print_info"]:
                        warning_message = (
                            f"The Virtual Column {column} is not numerical."
                            " Its histogram will not be drawn."
                        )
                        warnings.warn(warning_message, Warning)
            ax.set_xlabel(", ".join(all_columns))
            if method.lower() == "density":
                ax.set_ylabel("Density")
            elif (
                method.lower() in ["avg", "min", "max", "sum", "mean"]
                or ("%" == method[-1])
            ) and (of):
                ax.set_ylabel(f"{method}({of})")
            elif method.lower() == "count":
                ax.set_ylabel("Frequency")
            else:
                ax.set_ylabel(method)
            ax.legend(title="columns", loc="center left", bbox_to_anchor=[1, 0.5])
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            return ax
