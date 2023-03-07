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
import math, copy
from typing import Optional, TYPE_CHECKING

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from verticapy._typing import SQLColumns
from verticapy._utils._sql._cast import to_varchar
from verticapy._utils._sql._sys import _executeSQL
from verticapy.errors import ParameterError

from verticapy.core.tablesample.base import TableSample

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame

from verticapy.plotting.base import PlottingBase
from verticapy.plotting._matplotlib.scatter import HeatMap


class PivotTable(PlottingBase, HeatMap):
    def pivot_table(
        self,
        vdf: "vDataFrame",
        columns: SQLColumns,
        method: str = "count",
        of: str = "",
        h: tuple[Optional[float], Optional[float]] = (None, None),
        max_cardinality: tuple[int, int] = (20, 20),
        show: bool = True,
        with_numbers: bool = True,
        fill_none: float = 0.0,
        ax: Optional[Axes] = None,
        return_ax: bool = False,
        extent: list = [],
        **style_kwds,
    ) -> Axes:
        """
        Draws a pivot table using the Matplotlib API.
        """
        columns, of = vdf._format_colnames(columns, of)
        other_columns = ""
        method = method.lower()
        if method == "median":
            method = "50%"
        elif method == "mean":
            method = "avg"
        if (
            method not in ["avg", "min", "max", "sum", "density", "count"]
            and "%" != method[-1]
        ) and of:
            raise ParameterError(
                "Parameter 'of' must be empty when using customized aggregations."
            )
        if (method in ["avg", "min", "max", "sum"]) and (of):
            aggregate = f"{method.upper()}({of})"
        elif method and method[-1] == "%":
            aggregate = f"""APPROXIMATE_PERCENTILE({of} 
                                                   USING PARAMETERS 
                                                   percentile = {float(method[0:-1]) / 100})"""
        elif method in ["density", "count"]:
            aggregate = "COUNT(*)"
        elif isinstance(method, str):
            aggregate = method
            other_columns = vdf.get_columns(exclude_columns=columns)
            other_columns = ", " + ", ".join(other_columns)
        else:
            raise ParameterError(
                "The parameter 'method' must be in [count|density|avg|mean|min|max|sum|q%]"
            )
        is_column_date = [False, False]
        timestampadd = ["", ""]
        all_columns = []
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
                    all_columns += [expr]
                else:
                    all_columns += [f"FLOOR({column}) || ''"]
                order_by = f"""ORDER BY MIN(FLOOR({column} 
                                          / {interval}) * {interval}) ASC"""
                where += [f"{column} IS NOT NULL"]
            elif is_date:
                if h[idx] == None:
                    interval = vdf[column].numh()
                else:
                    interval = max(math.floor(h[idx]), 1)
                min_date = vdf[column].min()
                all_columns += [
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
                all_columns += [column]
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
            cast = to_varchar(vdf[columns[0]].category(), all_columns[-1])
            return TableSample.read_sql(
                query=f"""
                    SELECT 
                        {cast} AS {columns[0]},
                        {aggregate}{over} 
                    FROM {vdf._genSQL()}
                    {where}
                    GROUP BY 1 {order_by}"""
            )
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
                              {all_columns[0]} AS {columns[0]},
                              {all_columns[1]} AS {columns[1]}
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
        all_columns_categories = []
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
            all_columns_categories += [copy.deepcopy(L)]
        all_column0_categories, all_column1_categories = all_columns_categories
        all_columns = [
            [fill_none for item in all_column0_categories]
            for item in all_column1_categories
        ]
        for item in query_result:
            j, i = (
                all_column0_categories.index(str(item[0])),
                all_column1_categories.index(str(item[1])),
            )
            all_columns[i][j] = item[2]
        all_columns = [
            [all_column1_categories[i]] + all_columns[i]
            for i in range(0, len(all_columns))
        ]
        all_columns = [
            [columns[0] + "/" + columns[1]] + all_column0_categories
        ] + all_columns
        if show:
            all_count = [item[2] for item in query_result]
            ax = self.cmatrix(
                all_columns,
                all_column0_categories,
                all_column1_categories,
                len(all_column0_categories),
                len(all_column1_categories),
                vmax=max(all_count),
                vmin=min(all_count),
                title="",
                colorbar=aggregate,
                x_label=columns[1],
                y_label=columns[0],
                with_numbers=with_numbers,
                inverse=True,
                extent=extent,
                ax=ax,
                is_pivot=True,
                **style_kwds,
            )
            if return_ax:
                return ax
        values = {all_columns[0][0]: all_columns[0][1 : len(all_columns[0])]}
        del all_columns[0]
        for column in all_columns:
            values[column[0]] = column[1 : len(column)]
        return TableSample(values=values)
