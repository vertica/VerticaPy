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
from typing import Literal, TYPE_CHECKING
import numpy as np

from verticapy.core.tablesample.base import TableSample

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame


class PlottingBaseSQL:
    @staticmethod
    def _get_vdataframe_from_query(query: str) -> "vDataFrame":
        from verticapy.core.vdataframe.base import vDataFrame

        return vDataFrame(input_relation=query)

    # 1D AGG Graphics: BAR / PIE ...

    def _compute_plot_params_sql(self, query: str, bargap: float = 0.06) -> None:
        tbs = TableSample().read_sql(query)
        columns = tbs.get_columns()
        by = columns[0]
        agg = columns[1]
        X = tbs.to_numpy()
        n, m = tbs.shape()
        dt = int if tbs.category(column=columns[1]) == "int" else float
        self.data = {
            "x": [0.4 * i + 0.2 for i in range(n)],
            "y": X[:, 1].astype(dt),
            "width": None,
            "adj_width": 0.4 * (1 - bargap),
            "bargap": bargap,
            "is_categorical": True,
        }
        self.layout = {
            "labels": X[:, 0],
            "column": by,
            "method": agg,
            "method_of": agg,
            "of": None,
            "of_cat": tbs.category(column=agg),
            "aggregate": agg,
            "aggregate_fun": None,
            "is_standard": False,
        }
        return None

    # HISTOGRAMS

    def _compute_hists_params_sql(self, query: str) -> None:
        tbs = TableSample().read_sql(query)
        columns = tbs.get_columns()
        by = columns[0]
        agg = columns[1]
        X = tbs.to_numpy()
        n, m = tbs.shape()
        dt = int if tbs.category(column=columns[1]) == "int" else float
        self.data = {
            by: {
                "x": X[:, 0],
                "y": X[:, 1].astype(dt),
                "width": None,
                "adj_width": 1.0,
                "bargap": 1.0,
                "is_categorical": True,
            },
            "width": 1.0,
        }
        self.layout = {
            "columns": [by],
            "categories": None,
            "by": None,
            "method": agg,
            "method_of": agg,
            "of": None,
            "of_cat": tbs.category(column=agg),
            "aggregate": agg,
            "aggregate_fun": None,
            "is_standard": False,
            "has_category": False,
        }

    # 2D AGG Graphics: BAR / PIE / HEATMAP / CONTOUR / HEXBIN ...

    def _compute_pivot_table_sql(self, query: str, stacked: bool = False) -> None:
        tbs = TableSample().read_sql(query)
        columns = tbs.get_columns()
        X = tbs.to_numpy()
        x_labels = list(np.unique(X[:, 0]))
        y_labels = list(np.unique(X[:, 1]))
        Xf = np.array([[np.nan for i in y_labels] for j in x_labels])
        for ci, cj, val in X:
            i = x_labels.index(ci)
            j = y_labels.index(cj)
            Xf[i][j] = val
        self.data = {
            "X": Xf.astype(float),
        }
        self.layout = {
            "x_labels": x_labels,
            "y_labels": y_labels,
            "vmax": None,
            "vmin": None,
            "columns": columns[0:2],
            "method": columns[2],
            "method_of": columns[2],
            "of": None,
            "of_cat": None,
            "aggregate": columns[2],
            "aggregate_fun": None,
            "is_standard": False,
            "stacked": stacked,
        }
        return None

    # TIME SERIES: LINE / RANGE

    def _filter_line_sql(
        self,
        query: str,
        kind: Literal[
            "area", "area_percent", "area_stacked", "line", "spline", "step"
        ] = "line",
    ) -> None:
        tbs = TableSample().read_sql(query)
        columns = tbs.get_columns()
        order_by = columns[0]
        columns = columns[1:]
        X = tbs.to_numpy()
        self.data = {
            "x": X[:, 0],
        }
        if tbs.category(column=columns[-1]) in ("text", "int", "bool"):
            self.data["Y"] = X[:, 1:-1]
            self.data["z"] = X[:, -1]
            has_category = True
        else:
            self.data["Y"] = X[:, 1:]
            has_category = False
        self.layout = {
            "columns": columns,
            "order_by": order_by,
            "order_by_cat": tbs.category(column=order_by),
            "has_category": has_category,
            "limit": -1,
            "limit_over": -1,
            "kind": kind,
        }
        return None

    def _compute_range_sql(self, query: str,) -> None:
        tbs = TableSample().read_sql(query)
        columns = tbs.get_columns()
        order_by = columns[0]
        n = len(columns)
        X = tbs.to_numpy()
        self.data = {
            "x": self._parse_datetime(X[:, 0]),
            "Y": X[:, 1:].astype(float),
            "q": None,
        }
        self.layout = {
            "columns": [columns[i] for i in range(1, n, 3)],
            "order_by": order_by,
            "order_by_cat": tbs.category(column=order_by),
        }
        return None

    # SAMPLE: SCATTERS / OUTLIERS / ML PLOTS ...

    def _sample_sql(
        self, query: str, kind: Literal["scatter", "bubble"] = "scatter",
    ) -> None:
        tbs = TableSample().read_sql(query)
        columns = tbs.get_columns()
        X = tbs.to_numpy()
        if tbs.category(column=columns[-1]) in ("text", "int", "bool"):
            has_category = True
            c_name = columns[-1]
            c = X[:, -1]
            Xi = X[:, :-1].astype(float)
        else:
            has_category = False
            c_name = None
            c = None
            Xi = X.astype(float)
        if kind == "bubble":
            has_size = True
            idx = -2 if has_category else -1
            s_name = columns[idx]
            s = Xi[:, -1]
            Xi = Xi[:, :-1]
        else:
            has_size = False
            s_name = None
            s = None
        self.data = {"X": Xi, "s": s, "c": c}
        self.layout = {
            "columns": columns,
            "size": s_name,
            "c": c_name,
            "has_category": has_category,
            "has_cmap": False,
            "has_size": has_size,
        }
        return None
