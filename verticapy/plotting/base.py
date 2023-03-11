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
import copy, math
from typing import Literal, Optional, TYPE_CHECKING
import numpy as np

import verticapy._config.config as conf
from verticapy._typing import ArrayLike, SQLColumns
from verticapy._utils._sql._cast import to_varchar
from verticapy._utils._sql._format import quote_ident
from verticapy._utils._sql._sys import _executeSQL
from verticapy.errors import ParameterError

from verticapy.core.tablesample.base import TableSample

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame, vDataColumn

if conf._get_import_success("dateutil"):
    from dateutil.parser import parse


class PlottingBase:

    # Properties.

    @property
    def _compute_method(self) -> Literal[None]:
        """Must be overridden in child class"""
        return None

    def __init__(*args, **kwargs) -> None:
        return None

    # Formatting Methods.

    @staticmethod
    def _map_method(method: str, of: str) -> tuple[str, str, bool]:
        is_standard = True
        if method.lower() == "median":
            method = "50%"
        elif method.lower() == "mean":
            method = "avg"
        if (
            method.lower() not in ["avg", "min", "max", "sum", "density", "count"]
            and "%" != method[-1]
        ) and of:
            raise ParameterError(
                "Parameter 'of' must be empty when using customized aggregations."
            )
        if (
            (method.lower() in ["avg", "min", "max", "sum"])
            or (method.lower() and method[-1] == "%")
        ) and (of):
            if method.lower() in ["avg", "min", "max", "sum"]:
                aggregate = f"{method.upper()}({quote_ident(of)})"
            elif method and method[-1] == "%":
                aggregate = f"""
                    APPROXIMATE_PERCENTILE({quote_ident(of)} 
                        USING PARAMETERS
                        percentile = {float(method[0:-1]) / 100})"""
            else:
                raise ValueError(
                    "The parameter 'method' must be in [avg|mean|min|max|sum|"
                    f"median|q%] or a customized aggregation. Found {method}."
                )
        elif method.lower() in ["density", "count"]:
            aggregate = "count(*)"
        elif isinstance(method, str):
            aggregate = method
            is_standard = False
        else:
            raise ParameterError(
                "The parameter 'method' must be in [avg|mean|min|max|sum|"
                f"median|q%] or a customized aggregation. Found {method}."
            )
        return method, aggregate, is_standard

    @staticmethod
    def _parse_datetime(D: list) -> list:
        """
        Parses the list and casts the value to the datetime
        format if possible.
        """
        try:
            return [parse(d) for d in D]
        except:
            return copy.deepcopy(D)

    @staticmethod
    def _update_dict(d1: dict, d2: dict, color_idx: int = 0,) -> dict:
        """
        Updates the input dictionary using another one.
        """
        d = {}
        for elem in d1:
            d[elem] = d1[elem]
        for elem in d2:
            if elem == "color":
                if isinstance(d2["color"], str):
                    d["color"] = d2["color"]
                elif color_idx < 0:
                    d["color"] = [elem for elem in d2["color"]]
                else:
                    d["color"] = d2["color"][color_idx % len(d2["color"])]
            else:
                d[elem] = d2[elem]
        return d

    # Attributes Computations.

    def _compute_plot_params(
        self,
        vdc: "vDataColumn",
        method: str = "density",
        of: str = "",
        max_cardinality: int = 6,
        nbins: int = 0,
        h: float = 0.0,
        pie: bool = False,
    ) -> None:
        """
	    Computes the aggregations needed to draw a 1D graphic 
	    using the Matplotlib API.
	    """
        other_columns = ""
        method, aggregate, is_standard = self._map_method(method, of)
        if not (is_standard):
            other_columns = ", " + ", ".join(
                vdc._parent.get_columns(exclude_columns=[vdc._alias])
            )
        # depending on the cardinality, the type, the vDataColumn
        # can be treated as categorical or not
        cardinality, count, is_numeric, is_date, is_categorical = (
            vdc.nunique(True),
            vdc._parent.shape()[0],
            vdc.isnum() and not (vdc.isbool()),
            (vdc.category() == "date"),
            False,
        )
        rotation = 0 if ((is_numeric) and (cardinality > max_cardinality)) else 90
        # case when categorical
        if (((cardinality <= max_cardinality) or not (is_numeric)) or pie) and not (
            is_date
        ):
            if (is_numeric) and not (pie):
                query = f"""
	                SELECT 
	                    {vdc._alias},
	                    {aggregate}
	                FROM {vdc._parent._genSQL()} 
	                WHERE {vdc._alias} IS NOT NULL 
	                GROUP BY {vdc._alias} 
	                ORDER BY {vdc._alias} ASC 
	                LIMIT {max_cardinality}"""
            else:
                table = vdc._parent._genSQL()
                if (pie) and (is_numeric):
                    enum_trans = (
                        vdc.discretize(h=h, return_enum_trans=True)[0].replace(
                            "{}", vdc._alias
                        )
                        + " AS "
                        + vdc._alias
                    )
                    if of:
                        enum_trans += f" , {of}"
                    table = (
                        f"(SELECT {enum_trans + other_columns} FROM {table}) enum_table"
                    )
                cast_alias = to_varchar(vdc.category(), vdc._alias)
                query = f"""
	                (SELECT 
	                    /*+LABEL('plotting._matplotlib._compute_plot_params')*/ 
	                    {cast_alias} AS {vdc._alias},
	                    {aggregate}
	                 FROM {table} 
	                 GROUP BY {cast_alias} 
	                 ORDER BY 2 DESC 
	                 LIMIT {max_cardinality})"""
                if cardinality > max_cardinality:
                    query += f"""
	                    UNION 
	                    (SELECT 
	                        'Others',
	                        {aggregate} 
	                     FROM {table}
	                     WHERE {vdc._alias} NOT IN
	                     (SELECT 
	                        {vdc._alias} 
	                      FROM {table}
	                      GROUP BY {vdc._alias}
	                      ORDER BY {aggregate} DESC
	                      LIMIT {max_cardinality}))"""
            query_result = _executeSQL(
                query=query, title="Computing the histogram heights", method="fetchall"
            )
            if query_result[-1][1] == None:
                del query_result[-1]
            y = (
                [
                    item[1] / float(count) if item[1] != None else 0
                    for item in query_result
                ]
                if (method.lower() == "density")
                else [item[1] if item[1] != None else 0 for item in query_result]
            )
            x = [0.4 * i + 0.2 for i in range(0, len(y))]
            adj_width = 0.39
            labels = [item[0] for item in query_result]
            is_categorical = True
        # case when date
        elif is_date:
            if (h <= 0) and (nbins <= 0):
                h = vdc.numh()
            elif nbins > 0:
                query_result = _executeSQL(
                    query=f"""
	                    SELECT 
	                        /*+LABEL('plotting._matplotlib._compute_plot_params')*/
	                        DATEDIFF('second', MIN({vdc._alias}), MAX({vdc._alias}))
	                    FROM {vdc._parent._genSQL()}""",
                    title="Computing the histogram interval",
                    method="fetchrow",
                )
                h = float(query_result[0]) / nbins
            min_date = vdc.min()
            converted_date = f"DATEDIFF('second', '{min_date}', {vdc._alias})"
            query_result = _executeSQL(
                query=f"""
	                SELECT 
	                    /*+LABEL('plotting._matplotlib._compute_plot_params')*/
	                    FLOOR({converted_date} / {h}) * {h}, 
	                    {aggregate} 
	                FROM {vdc._parent._genSQL()}
	                WHERE {vdc._alias} IS NOT NULL 
	                GROUP BY 1 
	                ORDER BY 1""",
                title="Computing the histogram heights",
                method="fetchall",
            )
            x = [float(item[0]) for item in query_result]
            y = (
                [item[1] / float(count) for item in query_result]
                if (method.lower() == "density")
                else [item[1] for item in query_result]
            )
            query = "("
            for idx, item in enumerate(query_result):
                query_tmp = f"""
	                (SELECT 
	                    {{}}
	                    TIMESTAMPADD('second',
	                                 {math.floor(h * idx)},
	                                 '{min_date}'::timestamp))"""
                if idx == 0:
                    query += query_tmp.format(
                        "/*+LABEL('plotting._matplotlib._compute_plot_params')*/"
                    )
                else:
                    query += f" UNION {query_tmp.format('')}"
            query += ")"
            query_result = _executeSQL(
                query, title="Computing the datetime intervals.", method="fetchall"
            )
            adj_width = 0.94 * h
            labels = [item[0] for item in query_result]
            labels.sort()
            is_categorical = True
        # case when numerical
        else:
            if (h <= 0) and (nbins <= 0):
                h = vdc.numh()
            elif nbins > 0:
                h = float(vdc.max() - vdc.min()) / nbins
            if (vdc.ctype == "int") or (h == 0):
                h = max(1.0, h)
            query_result = _executeSQL(
                query=f"""
	                SELECT
	                    /*+LABEL('plotting._matplotlib._compute_plot_params')*/
	                    FLOOR({vdc._alias} / {h}) * {h},
	                    {aggregate} 
	                FROM {vdc._parent._genSQL()}
	                WHERE {vdc._alias} IS NOT NULL
	                GROUP BY 1
	                ORDER BY 1""",
                title="Computing the histogram heights",
                method="fetchall",
            )
            y = (
                [item[1] / float(count) for item in query_result]
                if (method.lower() == "density")
                else [item[1] for item in query_result]
            )
            x = [float(item[0]) + h / 2 for item in query_result]
            adj_width = 0.94 * h
            labels = None
        if pie:
            y.reverse()
            labels.reverse()
        self.data = {
            "x": x,
            "y": y,
            "labels": labels,
            "width": h,
            "adj_width": adj_width,
            "is_categorical": is_categorical,
        }
        return None

    def _compute_pivot_table(
        self,
        vdf: "vDataFrame",
        columns: SQLColumns,
        method: str = "count",
        of: str = "",
        h: tuple[Optional[float], Optional[float]] = (None, None),
        max_cardinality: tuple[int, int] = (20, 20),
        fill_none: float = 0.0,
    ) -> None:
        """
        Draws a pivot table using the Matplotlib API.
        """
        other_columns = ""
        method, aggregate, is_standard = self._map_method(method, of)
        if not (is_standard):
            other_columns = ", " + ", ".join(vdf.get_columns(exclude_columns=columns))
        columns, of = vdf._format_colnames(columns, of)
        is_column_date = [False, False]
        timestampadd = ["", ""]
        matrix = []
        for idx, column in enumerate(columns):
            is_numeric = vdf[column].isnum() and (vdf[column].nunique(True) > 2)
            is_date = vdf[column].isdate()
            where = []
            if is_numeric:
                if h[idx] == None:
                    interval = vdf[column].numh()
                    if interval > 0.01:
                        interval = round(interval, 2)
                    elif interval > 0.0001:
                        interval = round(interval, 4)
                    elif interval > 0.000001:
                        interval = round(interval, 6)
                    if vdf[column].category() == "int":
                        interval = int(max(math.floor(interval), 1))
                else:
                    interval = h[idx]
                if vdf[column].category() == "int":
                    floor_end = "-1"
                    interval = int(max(math.floor(interval), 1))
                else:
                    floor_end = ""
                expr = f"""'[' 
                          || FLOOR({column} 
                                 / {interval}) 
                                 * {interval} 
                          || ';' 
                          || (FLOOR({column} 
                                  / {interval}) 
                                  * {interval} 
                                  + {interval}{floor_end}) 
                          || ']'"""
                if (interval > 1) or (vdf[column].category() == "float"):
                    matrix += [expr]
                else:
                    matrix += [f"FLOOR({column}) || ''"]
                order_by = f"""ORDER BY MIN(FLOOR({column} 
                                          / {interval}) * {interval}) ASC"""
                where += [f"{column} IS NOT NULL"]
            elif is_date:
                if h[idx] == None:
                    interval = vdf[column].numh()
                else:
                    interval = max(math.floor(h[idx]), 1)
                min_date = vdf[column].min()
                matrix += [
                    f"""FLOOR(DATEDIFF('second',
                                       '{min_date}',
                                       {column})
                            / {interval}) * {interval}"""
                ]
                is_column_date[idx] = True
                sql = f"""TIMESTAMPADD('second', {columns[idx]}::int, 
                                                 '{min_date}'::timestamp)"""
                timestampadd[idx] = sql
                order_by = "ORDER BY 1 ASC"
                where += [f"{column} IS NOT NULL"]
            else:
                matrix += [column]
                order_by = "ORDER BY 1 ASC"
                distinct = vdf[column].topk(max_cardinality[idx]).values["index"]
                distinct = ["'" + str(c).replace("'", "''") + "'" for c in distinct]
                if len(distinct) < max_cardinality[idx]:
                    cast = to_varchar(vdf[column].category(), column)
                    where += [f"({cast} IN ({', '.join(distinct)}))"]
                else:
                    where += [f"({column} IS NOT NULL)"]
        where = f" WHERE {' AND '.join(where)}"
        over = "/" + str(vdf.shape()[0]) if (method == "density") else ""
        if len(columns) == 1:
            cast = to_varchar(vdf[columns[0]].category(), matrix[-1])
            res = TableSample.read_sql(
                query=f"""
                    SELECT 
                        {cast} AS {columns[0]},
                        {aggregate}{over} 
                    FROM {vdf._genSQL()}
                    {where}
                    GROUP BY 1 {order_by}"""
            ).to_numpy()
            matrix = res[:, 1].astype(float)
            x_labels = list(res[:, 0])
            return matrix, x_labels, [method], min(matrix), max(matrix), aggregate
        aggr = f", {of}" if (of) else ""
        cols, cast = [], []
        for i in range(2):
            if is_column_date[0]:
                cols += [f"{timestampadd[i]} AS {columns[i]}"]
            else:
                cols += [columns[i]]
            cast += [to_varchar(vdf[columns[i]].category(), columns[i])]
        query_result = _executeSQL(
            query=f"""
                SELECT 
                    /*+LABEL('plotting._matplotlib.pivot_table')*/
                    {cast[0]} AS {columns[0]},
                    {cast[1]} AS {columns[1]},
                    {aggregate}{over}
                FROM (SELECT 
                          {cols[0]},
                          {cols[1]}
                          {aggr}
                          {other_columns} 
                      FROM 
                          (SELECT 
                              {matrix[0]} AS {columns[0]},
                              {matrix[1]} AS {columns[1]}
                              {aggr}
                              {other_columns} 
                           FROM {vdf._genSQL()}{where}) 
                           pivot_table) pivot_table_date
                WHERE {columns[0]} IS NOT NULL 
                  AND {columns[1]} IS NOT NULL
                GROUP BY {columns[0]}, {columns[1]}
                ORDER BY {columns[0]}, {columns[1]} ASC""",
            title="Grouping the features to compute the pivot table",
            method="fetchall",
        )
        agg = [item[2] for item in query_result]
        matrix_categories = []
        for i in range(2):
            L = list(set([str(item[i]) for item in query_result]))
            L.sort()
            try:
                try:
                    order = []
                    for item in L:
                        order += [float(item.split(";")[0].split("[")[1])]
                except:
                    order = [float(item) for item in L]
                L = [x for _, x in sorted(zip(order, L))]
            except:
                pass
            matrix_categories += [copy.deepcopy(L)]
        x_labels, y_labels = matrix_categories
        matrix = np.array([[fill_none for item in x_labels] for item in y_labels])
        for item in query_result:
            j = x_labels.index(str(item[0]))
            i = y_labels.index(str(item[1]))
            matrix[i][j] = item[2]
        self.data = {"x_labels": x_labels, "y_labels": y_labels, "matrix": matrix}
