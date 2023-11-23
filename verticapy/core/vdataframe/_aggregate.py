"""
Copyright  (c)  2018-2024 Open Text  or  one  of its
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
import decimal
import multiprocessing
import warnings
from typing import Literal, Optional, Union
from tqdm.auto import tqdm

from vertica_python.errors import QueryError

import verticapy._config.config as conf
from verticapy._typing import (
    ArrayLike,
    NoneType,
    PythonNumber,
    PythonScalar,
    SQLColumns,
    SQLExpression,
    TYPE_CHECKING,
)
from verticapy._utils._map import verticapy_agg_name
from verticapy._utils._object import create_new_vdf
from verticapy._utils._sql._cast import to_varchar
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import (
    format_magic,
    format_type,
    quote_ident,
)
from verticapy._utils._sql._sys import _executeSQL
from verticapy._utils._sql._vertica_version import vertica_version
from verticapy.connection import current_cursor
from verticapy.errors import (
    EmptyParameter,
    FunctionError,
)

from verticapy.core.tablesample.base import TableSample

from verticapy.core.vdataframe._eval import vDFEval, vDCEval

from verticapy.core.vdataframe._multiprocessing import (
    aggregate_parallel_block,
    describe_parallel_block,
)

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame


class vDFAgg(vDFEval):
    # Main Aggregate Functions.

    @save_verticapy_logs
    def aggregate(
        self,
        func: SQLExpression,
        columns: Optional[SQLColumns] = None,
        ncols_block: int = 20,
        processes: int = 1,
    ) -> TableSample:
        """
        Aggregates the vDataFrame using the input functions.

        Parameters
        ----------
        func: SQLExpression
            List of the different aggregations:

             - aad:
                average absolute deviation.
             - approx_median:
                approximate median.
             - approx_q%:
                approximate q quantile (ex: approx_50%
                for the approximate median).
             - approx_unique:
                approximative cardinality.
             - count:
                number of non-missing elements.
             - cvar:
                conditional value at risk.
             - dtype:
                virtual column type.
             - iqr:
                interquartile range.
             - kurtosis:
                kurtosis.
             - jb:
                Jarque-Bera index.
             - mad:
                median absolute deviation.
             - max:
                maximum.
             - mean:
                average.
             - median:
                median.
             - min:
                minimum.
             - mode:
                most occurent element.
             - percent:
                percent of non-missing elements.
             - q%:
                q quantile (ex: 50% for the median)
                Use the ``approx_q%`` (approximate quantile)
                aggregation to get better performance.
             - prod:
                product.
             - range:
                difference between the max and the min.
             - sem:
                standard error of the mean.
             - skewness:
                skewness.
             - sum:
                sum.
             - std:
                standard deviation.
             - topk:
                kth most occurent element (ex: top1 for the mode)
             - topk_percent:
                kth most occurent element density.
             - unique:
                cardinality (count distinct).
             - var:
                variance.

            Other aggregations will work if supported by your database
            version.

        columns: SQLColumns, optional
            List of  the vDataColumn's names. If empty,  depending on the
            aggregations, all or only numerical vDataColumns are used.
        ncols_block: int, optional
            Number  of columns  used per query.  Setting  this  parameter
            divides  what  would otherwise be  one large query into  many
            smaller  queries  called "blocks", whose size is determine by
            the size of ncols_block.
        processes: int, optional
            Number  of child processes  to  create. Setting  this  with  the
            ncols_block  parameter lets you parallelize a  single query into
            many smaller  queries, where each child process creates  its own
            connection to the database and sends one query. This can improve
            query performance, but consumes  more resources. If processes is
            set to 1, the queries are sent iteratively from a single process.

        Returns
        -------
        TableSample
            result.

        Examples
        --------
        For this example, we will use the following dataset:

        .. code-block:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )

        With the ``aggregate`` method, you have the flexibility to select specific
        aggregates and the columns you wish to include in the query. This
        allows for more precise control over the aggregation process and helps
        tailor the results to your specific needs.

        .. code-block:: python

            data.aggregate(
                func = ["min", "approx_10%", "approx_50%", "approx_90%", "max"],
                columns = ["x", "y", "z"],
            )

        .. ipython:: python
            :suppress:

            import verticapy as vp
            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            result = data.aggregate(
                func = ["min", "approx_10%", "approx_50%", "approx_90%", "max"],
                columns = ["x", "y", "z"],
            )
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_aggregate_table.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_aggregate_table.html

        .. note:: All the calculations are pushed to the database.

        .. hint::

            When the vDataFrame includes a large number of columns and many aggregates
            need to be computed, it can be resource-intensive for the database. To
            address this, you can use the ``ncols_block`` parameter to control the number
            of blocks of aggregates to use and the ``processes`` parameter to manage
            the number of processes. These blocks consist of specific columns, and
            their aggregates are calculated first (or in parallel), then the subsequent
            ones, and the results are combined at the end.

        .. seealso::
            | :py:meth:`verticapy.vDataColumn.aggregate` :
              Aggregations for a specific column.
            | :py:meth:`verticapy.vDataColumn.describe` :
              Summarizes the information within the column.
            | :py:meth:`verticapy.vDataFrame.describe` :
              Summarizes the information for specific columns.
        """
        columns, func = format_type(columns, func, dtype=list)
        if len(columns) == 0:
            columns = self.get_columns()
            cat_agg = [
                "count",
                "unique",
                "approx_unique",
                "approximate_count_distinct",
                "dtype",
                "percent",
            ]
            for fun in func:
                if ("top" not in fun) and (fun not in cat_agg):
                    columns = self.numcol()
                    break
        else:
            columns = self.format_colnames(columns)

        # Some aggregations are not compatibles, we need to pre-compute them.

        agg_unique = []
        agg_approx = []
        agg_exact_percent = []
        agg_percent = []
        other_agg = []

        for fun in func:
            if fun[-1] == "%":
                if (len(fun.lower()) >= 8) and fun.startswith("approx_"):
                    agg_approx += [fun.lower()]
                else:
                    agg_exact_percent += [fun.lower()]

            elif fun.lower() in ("approx_unique", "approximate_count_distinct"):
                agg_approx += [fun.lower()]

            elif fun.lower() == "unique":
                agg_unique += [fun.lower()]

            else:
                other_agg += [fun.lower()]

        exact_percent, uniques = {}, {}

        if agg_exact_percent and (other_agg or agg_percent or agg_approx or agg_unique):
            exact_percent = self.aggregate(
                func=agg_exact_percent,
                columns=columns,
                ncols_block=ncols_block,
                processes=processes,
            ).transpose()

        if agg_unique and agg_approx:
            uniques = self.aggregate(
                func=["unique"],
                columns=columns,
                ncols_block=ncols_block,
                processes=processes,
            ).transpose()

        # Some aggregations are using some others. We need to precompute them.

        for fun in func:
            if fun.lower() in [
                "kurtosis",
                "kurt",
                "skewness",
                "skew",
                "jb",
            ]:
                count_avg_stddev = (
                    self.aggregate(func=["count", "avg", "stddev"], columns=columns)
                    .transpose()
                    .values
                )
                break

        # Computing iteratively aggregations using block of columns.

        if ncols_block < len(columns) and processes <= 1:
            if conf.get_option("tqdm"):
                loop = tqdm(range(0, len(columns), ncols_block))
            else:
                loop = range(0, len(columns), ncols_block)
            for i in loop:
                res_tmp = self.aggregate(
                    func=func,
                    columns=columns[i : i + ncols_block],
                    ncols_block=ncols_block,
                )
                if i == 0:
                    result = res_tmp
                else:
                    result.append(res_tmp)
            return result

        # Computing the aggregations using multiple queries at the same time.

        elif ncols_block < len(columns):
            parameters = []
            for i in range(0, len(columns), ncols_block):
                parameters += [(self, func, columns, ncols_block, i)]
            a_pool = multiprocessing.Pool(processes)
            L = a_pool.starmap(func=aggregate_parallel_block, iterable=parameters)
            result = L[0]
            for i in range(1, len(L)):
                result.append(L[i])
            return result

        agg = [[] for i in range(len(columns))]
        nb_precomputed = 0

        # Computing all the other aggregations.

        for idx, column in enumerate(columns):
            cast = "::int" if (self[column].isbool()) else ""
            for fun in func:
                pre_comp = self._get_catalog_value(column, fun)

                if pre_comp != "VERTICAPY_NOT_PRECOMPUTED":
                    nb_precomputed += 1
                    if isinstance(pre_comp, NoneType) or pre_comp != pre_comp:
                        expr = "NULL"
                    elif isinstance(pre_comp, (int, float)):
                        expr = pre_comp
                    else:
                        pre_comp_str = str(pre_comp).replace("'", "''")
                        expr = f"'{pre_comp_str}'"

                elif fun.lower().endswith("_percent") and fun.lower().startswith("top"):
                    n = fun.lower().replace("top", "").replace("_percent", "")
                    if n == "":
                        n = 1
                    try:
                        n = int(n)
                        assert n >= 1
                    except:
                        raise FunctionError(
                            f"The aggregation '{fun}' doesn't exist. To"
                            " compute the frequency of the n-th most "
                            "occurent element, use 'topk_percent' with "
                            "k > 0. For example: top2_percent computes "
                            "the frequency of the second most occurent "
                            "element."
                        )
                    try:
                        expr = str(
                            self[column]
                            .topk(k=n, dropna=False)
                            .values["percent"][n - 1]
                        )
                    except:
                        expr = "0.0"

                elif (len(fun.lower()) > 2) and (fun.lower().startswith("top")):
                    n = fun.lower()[3:] if (len(fun.lower()) > 3) else 1
                    try:
                        n = int(n)
                        assert n >= 1
                    except:
                        raise FunctionError(
                            f"The aggregation '{fun}' doesn't exist. To"
                            " compute the n-th most occurent element, use "
                            "'topk' with n > 0. For example: "
                            "top2 computes the second most occurent element."
                        )
                    expr = format_magic(self[column].mode(n=n))

                elif fun.lower() == "mode":
                    expr = format_magic(self[column].mode(n=1))

                elif fun.lower() in ("kurtosis", "kurt"):
                    count, avg, std = count_avg_stddev[column]
                    if (
                        count == 0
                        or (std != std)
                        or (avg != avg)
                        or isinstance(std, NoneType)
                        or isinstance(avg, NoneType)
                    ):
                        expr = "NULL"
                    elif (count == 1) or (std == 0):
                        expr = "-3"
                    else:
                        expr = f"AVG(POWER(({column}{cast} - {avg}) / {std}, 4))"
                        if count > 3:
                            expr += f"""
                                * {count * count * (count + 1) / (count - 1) / (count - 2) / (count - 3)} 
                                - 3 * {(count - 1) * (count - 1) / (count - 2) / (count - 3)}"""
                        else:
                            expr += "* - 3"
                            expr += (
                                f"* {count * count / (count - 1) / (count - 2)}"
                                if (count == 3)
                                else ""
                            )

                elif fun.lower() in ("skewness", "skew"):
                    count, avg, std = count_avg_stddev[column]
                    if (
                        count == 0
                        or (std != std)
                        or (avg != avg)
                        or isinstance(std, NoneType)
                        or isinstance(avg, NoneType)
                    ):
                        expr = "NULL"
                    elif (count == 1) or (std == 0):
                        expr = "0"
                    else:
                        expr = f"AVG(POWER(({column}{cast} - {avg}) / {std}, 3))"
                        if count >= 3:
                            expr += f"* {count * count / (count - 1) / (count - 2)}"

                elif fun.lower() == "jb":
                    count, avg, std = count_avg_stddev[column]
                    if (count < 4) or (std == 0):
                        expr = "NULL"
                    else:
                        expr = f"""
                            {count} / 6 * (POWER(AVG(POWER(({column}{cast} - {avg}) 
                            / {std}, 3)) * {count * count / (count - 1) / (count - 2)}, 
                            2) + POWER(AVG(POWER(({column}{cast} - {avg}) / {std}, 4)) 
                            - 3 * {count * count / (count - 1) / (count - 2)}, 2) / 4)"""

                elif fun.lower() == "dtype":
                    expr = f"'{self[column].ctype()}'"

                elif fun.lower() == "range":
                    expr = f"MAX({column}{cast}) - MIN({column}{cast})"

                elif fun.lower() == "unique":
                    if column in uniques:
                        expr = format_magic(uniques[column][0])
                    else:
                        expr = f"COUNT(DISTINCT {column})"

                elif fun.lower() in ("approx_unique", "approximate_count_distinct"):
                    expr = f"APPROXIMATE_COUNT_DISTINCT({column})"

                elif fun.lower() == "count":
                    expr = f"COUNT({column})"

                elif fun.lower() in ("approx_median", "approximate_median"):
                    expr = f"APPROXIMATE_MEDIAN({column}{cast})"

                elif fun.lower() == "median":
                    expr = f"MEDIAN({column}{cast}) OVER ()"

                elif fun.lower() in ("std", "stddev", "stdev"):
                    expr = f"STDDEV({column}{cast})"

                elif fun.lower() in ("var", "variance"):
                    expr = f"VARIANCE({column}{cast})"

                elif fun.lower() in ("mean", "avg"):
                    expr = f"AVG({column}{cast})"

                elif fun.lower() == "iqr":
                    expr = f"""
                        APPROXIMATE_PERCENTILE({column}{cast} 
                                               USING PARAMETERS
                                               percentile = 0.75) 
                      - APPROXIMATE_PERCENTILE({column}{cast}
                                               USING PARAMETERS 
                                               percentile = 0.25)"""

                elif "%" == fun[-1]:
                    try:
                        if (len(fun.lower()) >= 8) and fun.startswith("approx_"):
                            percentile = float(fun[7:-1]) / 100
                            expr = f"""
                                APPROXIMATE_PERCENTILE({column}{cast} 
                                                       USING PARAMETERS 
                                                       percentile = {percentile})"""
                        else:
                            if column in exact_percent:
                                expr = format_magic(exact_percent[column][0])
                            else:
                                percentile = float(fun[0:-1]) / 100
                                expr = f"""
                                    PERCENTILE_CONT({percentile}) 
                                                    WITHIN GROUP 
                                                    (ORDER BY {column}{cast}) 
                                                    OVER ()"""
                    except:
                        raise FunctionError(
                            f"The aggregation '{fun}' doesn't exist. If you "
                            "want to compute the percentile x of the element "
                            "please write 'x%' with x > 0. Example: 50% for "
                            "the median or approx_50% for the approximate median."
                        )

                elif fun.lower() == "cvar":
                    q95 = self[column].quantile(0.95)
                    expr = f"""AVG(
                                CASE 
                                    WHEN {column}{cast} >= {q95} 
                                        THEN {column}{cast} 
                                    ELSE NULL 
                                END)"""

                elif fun.lower() == "sem":
                    expr = f"STDDEV({column}{cast}) / SQRT(COUNT({column}))"

                elif fun.lower() == "aad":
                    mean = self[column].avg()
                    expr = f"SUM(ABS({column}{cast} - {mean})) / COUNT({column})"

                elif fun.lower() == "mad":
                    median = self[column].median()
                    expr = f"APPROXIMATE_MEDIAN(ABS({column}{cast} - {median}))"

                elif fun.lower() in ("prod", "product"):
                    expr = f"""
                        DECODE(ABS(MOD(SUM(
                            CASE 
                                WHEN {column}{cast} < 0 THEN 1 
                                ELSE 0 
                            END), 
                        2)), 0, 1, -1) * 
                        POWER(10, SUM(LOG(ABS({column}{cast}))))"""

                elif fun.lower() in ("percent", "count_percent"):
                    if self.shape()[0] == 0:
                        expr = "100.0"
                    else:
                        expr = f"ROUND(COUNT({column}) / {self.shape()[0]} * 100, 3)::float"

                elif "{}" not in fun:
                    expr = f"{fun.upper()}({column}{cast})"

                else:
                    expr = fun.replace("{}", column)

                agg[idx] += [expr]

        for idx, elem in enumerate(func):
            if "AS " in str(elem).upper():
                try:
                    func[idx] = (
                        str(elem)
                        .lower()
                        .split("as ")[1]
                        .replace("'", "")
                        .replace('"', "")
                    )
                except IndexError:
                    pass
        values = {"index": func}

        try:
            if nb_precomputed == len(func) * len(columns):
                res = _executeSQL(
                    query=f"""
                        SELECT 
                            /*+LABEL('vDataframe.aggregate')*/ 
                            {", ".join([str(item) for sublist in agg for item in sublist])}""",
                    print_time_sql=False,
                    method="fetchrow",
                )
            else:
                res = _executeSQL(
                    query=f"""
                        SELECT 
                            /*+LABEL('vDataframe.aggregate')*/ 
                            {", ".join([str(item) for sublist in agg for item in sublist])} 
                        FROM {self} 
                        LIMIT 1""",
                    title="Computing the different aggregations.",
                    method="fetchrow",
                    sql_push_ext=self._vars["sql_push_ext"],
                    symbol=self._vars["symbol"],
                )
            result = list(res)
            try:
                result = [float(item) for item in result]
            except (TypeError, ValueError):
                pass
            values = {"index": func}
            i = 0
            for column in columns:
                values[column] = result[i : i + len(func)]
                i += len(func)

        except QueryError:
            try:
                query = [
                    "SELECT {0} FROM vdf_table LIMIT 1".format(
                        ", ".join(
                            [
                                format_magic(item, cast_float_int_to_str=True)
                                for item in elem
                            ]
                        )
                    )
                    for elem in agg
                ]
                query = (
                    " UNION ALL ".join([f"({q})" for q in query])
                    if (len(query) != 1)
                    else query[0]
                )
                query = f"""
                    WITH vdf_table AS 
                        (SELECT 
                            /*+LABEL('vDataframe.aggregate')*/ * 
                         FROM {self}) {query}"""
                if nb_precomputed == len(func) * len(columns):
                    result = _executeSQL(query, print_time_sql=False, method="fetchall")
                else:
                    result = _executeSQL(
                        query,
                        title="Computing the different aggregations using UNION ALL.",
                        method="fetchall",
                        sql_push_ext=self._vars["sql_push_ext"],
                        symbol=self._vars["symbol"],
                    )

                for idx, elem in enumerate(result):
                    values[columns[idx]] = list(elem)

            except QueryError:
                try:
                    for i, elem in enumerate(agg):
                        pre_comp_val = []
                        for fun in func:
                            pre_comp = self._get_catalog_value(columns[i], fun)
                            if pre_comp == "VERTICAPY_NOT_PRECOMPUTED":
                                columns_str = ", ".join(
                                    [
                                        format_magic(item, cast_float_int_to_str=True)
                                        for item in elem
                                    ]
                                )
                                _executeSQL(
                                    query=f"""
                                        SELECT 
                                            /*+LABEL('vDataframe.aggregate')*/ 
                                            {columns_str} 
                                        FROM {self}""",
                                    title=(
                                        "Computing the different aggregations one "
                                        "vDataColumn at a time."
                                    ),
                                    sql_push_ext=self._vars["sql_push_ext"],
                                    symbol=self._vars["symbol"],
                                )
                                pre_comp_val = []
                                break
                            pre_comp_val += [pre_comp]
                        if pre_comp_val:
                            values[columns[i]] = pre_comp_val
                        else:
                            values[columns[i]] = [
                                elem for elem in current_cursor().fetchone()
                            ]
                except QueryError:
                    for i, elem in enumerate(agg):
                        values[columns[i]] = []
                        for j, agg_fun in enumerate(elem):
                            pre_comp = self._get_catalog_value(columns[i], func[j])
                            if pre_comp == "VERTICAPY_NOT_PRECOMPUTED":
                                result = _executeSQL(
                                    query=f"""
                                        SELECT 
                                            /*+LABEL('vDataframe.aggregate')*/ 
                                            {agg_fun} 
                                        FROM {self}""",
                                    title=(
                                        "Computing the different aggregations one "
                                        "vDataColumn & one agg at a time."
                                    ),
                                    method="fetchfirstelem",
                                    sql_push_ext=self._vars["sql_push_ext"],
                                    symbol=self._vars["symbol"],
                                )
                            else:
                                result = pre_comp
                            values[columns[i]] += [result]

        for elem in values:
            for idx in range(len(values[elem])):
                if isinstance(values[elem][idx], str) and "top" not in elem:
                    try:
                        values[elem][idx] = float(values[elem][idx])
                    except (TypeError, ValueError):
                        pass

        self._update_catalog(values)
        return TableSample(values=values).decimal_to_float().transpose()

    agg = aggregate

    @save_verticapy_logs
    def describe(
        self,
        method: Literal[
            "numerical",
            "categorical",
            "statistics",
            "length",
            "range",
            "all",
            "auto",
        ] = "auto",
        columns: Optional[SQLColumns] = None,
        unique: bool = False,
        ncols_block: int = 20,
        processes: int = 1,
    ) -> TableSample:
        """
        This function aggregates the vDataFrame using multiple statistical
        aggregations such as minimum (min), maximum (max), median, cardinality
        (unique), and other relevant statistics. The specific aggregations
        applied depend on the data types of the vDataColumns. For example,
        numeric columns are aggregated with numerical aggregations (min, median,
        max...), while categorical columns are aggregated using categorical ones
        (cardinality, mode...). This versatile function provides valuable insights
        into the dataset's statistical properties and can be customized to meet
        specific analytical requirements.

        .. note::

            This function can offer faster performance compared to the
            :py:meth:`verticapy.vDataFrame.aggregate` method, as it
            leverages specialized and optimized backend functions.

        Parameters
        ----------
        method: str, optional
            The describe method.

             - all:
                Aggregates all statistics for all vDataColumns.
                The exact method depends on the vDataColumn type
                (numerical  dtype:  numerical; timestamp  dtype:
                range; categorical dtype: length)
             - auto:
                Sets the method  to  ``numerical`` if  at least
                one vDataColumn of the vDataFrame is numerical,
                ``categorical`` otherwise.
             - categorical:
                Uses only categorical aggregations.
             - length:
                Aggregates the vDataFrame using numerical
                aggregation on the length of all selected
                vDataColumns.
             - numerical:
                Uses only numerical descriptive statistics,
                which are  computed faster than the `aggregate`
                method.
             - range:
                Aggregates  the  vDataFrame   using  multiple
                statistical aggregations - min, max, range...
             - statistics:
                Aggregates  the  vDataFrame  using   multiple
                statistical aggregations - kurtosis, skewness,
                min, max...

        columns: SQLColumns, optional
            List of the vDataColumns names.  If empty, the  vDataColumns are
            selected depending on the parameter ``method``.
        unique: bool, optional
            If set to True, computes the cardinality of each element.
        ncols_block: int, optional
            Number of columns used per query.  Setting this parameter divides
            what would otherwise be one large query into many smaller queries
            called "blocks", whose size is determined by the ncols_block
            parmeter.
        processes: int, optional
            Number  of child  processes to  create.  Setting  this with  the
            ncols_block  parameter lets you parallelize a single query  into
            many smaller  queries, where each  child process creates its own
            connection to the database and sends one query. This can improve
            query performance,  but consumes more resources. If processes is
            set to 1, the queries are sent iteratively from a single process.

        Returns
        -------
        TableSample
            result.

        Examples
        --------
        For this example, we will use the following dataset:

        .. code-block:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                    "c": ['A', 'A', 'A', 'A', 'B', 'B', 'C', 'D'],
                }
            )

        The ``describe`` method provides you with a variety of statistical
        methods.

        The ``numerical`` parameter allows for the computation of numerical
        aggregations.

        .. code-block:: python

            data.describe(
                columns = ["x", "y", "z"],
                method = "numerical",
            )

        .. ipython:: python
            :suppress:

            import verticapy as vp
            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                    "c": ['A', 'A', 'A', 'A', 'B', 'B', 'C', 'D'],
                }
            )
            result = data.describe(
                columns = ["x", "y", "z"],
                method = "numerical",
            )
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_describe_num_table.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_describe_num_table.html

        The ``categorical`` parameter allows for the computation of categorical
        aggregations.

        .. code-block:: python

            data.describe(
                columns = ["x", "y", "z", "c"],
                method = "categorical",
            )

        .. ipython:: python
            :suppress:

            import verticapy as vp
            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                    "c": ['A', 'A', 'A', 'A', 'B', 'B', 'C', 'D'],
                }
            )
            result = data.describe(
                columns = ["x", "y", "z", "c"],
                method = "categorical",
            )
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_describe_cat_table.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_describe_cat_table.html

        The ``all`` parameter allows for the computation of both categorical
        and numerical aggregations.

        .. code-block:: python

            data.describe(
                columns = ["x", "y", "z", "c"],
                method = "all",
            )

        .. ipython:: python
            :suppress:

            import verticapy as vp
            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                    "c": ['A', 'A', 'A', 'A', 'B', 'B', 'C', 'D'],
                }
            )
            result = data.describe(
                columns = ["x", "y", "z", "c"],
                method = "all",
            )
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_describe_all_table.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_describe_all_table.html

        .. note::

            Many other methods are available, and their cost in terms of computation can vary.

        .. note:: All the calculations are pushed to the database.

        .. seealso::
            | :py:meth:`verticapy.vDataColumn.aggregate` : Aggregations for a specific column.
            | :py:meth:`verticapy.vDataFrame.aggregate` : Aggregations for specific columns.
            | :py:meth:`verticapy.vDataColumn.describe` :
              Summarizes the information within the column.
        """
        if method == "auto":
            method = "numerical" if (len(self.numcol()) > 0) else "categorical"
        columns = format_type(columns, dtype=list)
        columns = self.format_colnames(columns)
        for i in range(len(columns)):
            columns[i] = quote_ident(columns[i])
        dtype, percent = {}, {}

        if method == "numerical":
            if not columns:
                columns = self.numcol()
            else:
                for column in columns:
                    assert self[column].isnum(), TypeError(
                        f"vDataColumn {column} must be numerical to run describe"
                        " using parameter method = 'numerical'"
                    )
            assert columns, EmptyParameter(
                "No Numerical Columns found to run describe using parameter"
                " method = 'numerical'."
            )
            if ncols_block < len(columns) and processes <= 1:
                if conf.get_option("tqdm"):
                    loop = tqdm(range(0, len(columns), ncols_block))
                else:
                    loop = range(0, len(columns), ncols_block)
                for i in loop:
                    res_tmp = self.describe(
                        method=method,
                        columns=columns[i : i + ncols_block],
                        unique=unique,
                        ncols_block=ncols_block,
                    )
                    if i == 0:
                        result = res_tmp
                    else:
                        result.append(res_tmp)
                return result
            elif ncols_block < len(columns):
                parameters = []
                for i in range(0, len(columns), ncols_block):
                    parameters += [(self, method, columns, unique, ncols_block, i)]
                a_pool = multiprocessing.Pool(processes)
                L = a_pool.starmap(func=describe_parallel_block, iterable=parameters)
                result = L[0]
                for i in range(1, len(L)):
                    result.append(L[i])
                return result
            try:
                vertica_version(condition=[9, 0, 0])
                idx = [
                    "index",
                    "count",
                    "mean",
                    "std",
                    "min",
                    "approx_25%",
                    "approx_50%",
                    "approx_75%",
                    "max",
                ]
                values = {}
                for key in idx:
                    values[key] = []
                col_to_compute = []
                for column in columns:
                    if self[column].isnum():
                        for fun in idx[1:]:
                            pre_comp = self._get_catalog_value(column, fun)
                            if pre_comp == "VERTICAPY_NOT_PRECOMPUTED":
                                col_to_compute += [column]
                                break
                    elif conf.get_option("print_info"):
                        warning_message = (
                            f"The vDataColumn {column} is not numerical, it was ignored."
                            "\nTo get statistical information about all different "
                            "variables, please use the parameter method = 'categorical'."
                        )
                        warnings.warn(warning_message, Warning)
                for column in columns:
                    if column not in col_to_compute:
                        values["index"] += [column.replace('"', "")]
                        for fun in idx[1:]:
                            values[fun] += [self._get_catalog_value(column, fun)]
                if col_to_compute:
                    cols_to_compute_str = [
                        col if not self[col].isbool() else f"{col}::int"
                        for col in col_to_compute
                    ]
                    cols_to_compute_str = ", ".join(cols_to_compute_str)
                    query_result = _executeSQL(
                        query=f"""
                            SELECT 
                                /*+LABEL('vDataframe.describe')*/ 
                                SUMMARIZE_NUMCOL({cols_to_compute_str}) OVER () 
                            FROM {self}""",
                        title=(
                            "Computing the descriptive statistics of all numerical "
                            "columns using SUMMARIZE_NUMCOL."
                        ),
                        method="fetchall",
                    )

                    # Formatting - to have the same columns' order than the input one.
                    for i, key in enumerate(idx):
                        values[key] += [elem[i] for elem in query_result]
                    tb = TableSample(values).transpose()
                    vals = {"index": tb["index"]}
                    for col in columns:
                        vals[col] = tb[col]
                    values = TableSample(vals).transpose().values

            except QueryError:
                values = self.aggregate(
                    [
                        "count",
                        "mean",
                        "std",
                        "min",
                        "approx_25%",
                        "approx_50%",
                        "approx_75%",
                        "max",
                    ],
                    columns=columns,
                    ncols_block=ncols_block,
                    processes=processes,
                ).values

        elif method == "categorical":
            func = ["dtype", "count", "top", "top_percent"]
            values = self.aggregate(
                func,
                columns=columns,
                ncols_block=ncols_block,
                processes=processes,
            ).values

        elif method == "statistics":
            func = [
                "dtype",
                "percent",
                "count",
                "avg",
                "stddev",
                "min",
                "approx_1%",
                "approx_10%",
                "approx_25%",
                "approx_50%",
                "approx_75%",
                "approx_90%",
                "approx_99%",
                "max",
                "skewness",
                "kurtosis",
            ]
            values = self.aggregate(
                func=func,
                columns=columns,
                ncols_block=ncols_block,
                processes=processes,
            ).values

        elif method == "length":
            if not columns:
                columns = self.get_columns()
            func = [
                "dtype",
                "percent",
                "count",
                "SUM(CASE WHEN LENGTH({}::varchar) = 0 THEN 1 ELSE 0 END) AS empty",
                "AVG(LENGTH({}::varchar)) AS avg_length",
                "STDDEV(LENGTH({}::varchar)) AS stddev_length",
                "MIN(LENGTH({}::varchar))::int AS min_length",
                """APPROXIMATE_PERCENTILE(LENGTH({}::varchar) 
                        USING PARAMETERS percentile = 0.25)::int AS '25%_length'""",
                """APPROXIMATE_PERCENTILE(LENGTH({}::varchar)
                        USING PARAMETERS percentile = 0.5)::int AS '50%_length'""",
                """APPROXIMATE_PERCENTILE(LENGTH({}::varchar) 
                        USING PARAMETERS percentile = 0.75)::int AS '75%_length'""",
                "MAX(LENGTH({}::varchar))::int AS max_length",
            ]
            values = self.aggregate(
                func=func,
                columns=columns,
                ncols_block=ncols_block,
                processes=processes,
            ).values

        elif method == "range":
            if not columns:
                columns = []
                all_cols = self.get_columns()
                for idx, column in enumerate(all_cols):
                    if self[column].isnum() or self[column].isdate():
                        columns += [column]
            func = ["dtype", "percent", "count", "min", "max", "range"]
            values = self.aggregate(
                func=func,
                columns=columns,
                ncols_block=ncols_block,
                processes=processes,
            ).values

        elif method == "all":
            datecols, numcol, catcol = [], [], []
            if not columns:
                columns = self.get_columns()
            for elem in columns:
                if self[elem].isnum():
                    numcol += [elem]
                elif self[elem].isdate():
                    datecols += [elem]
                else:
                    catcol += [elem]
            values = self.aggregate(
                func=[
                    "dtype",
                    "percent",
                    "count",
                    "top",
                    "top_percent",
                    "avg",
                    "stddev",
                    "min",
                    "approx_25%",
                    "approx_50%",
                    "approx_75%",
                    "max",
                    "range",
                ],
                columns=numcol,
                ncols_block=ncols_block,
                processes=processes,
            ).values
            values["empty"] = [None] * len(numcol)
            if datecols:
                tmp = self.aggregate(
                    func=[
                        "dtype",
                        "percent",
                        "count",
                        "top",
                        "top_percent",
                        "min",
                        "max",
                        "range",
                    ],
                    columns=datecols,
                    ncols_block=ncols_block,
                    processes=processes,
                ).values
                for elem in [
                    "index",
                    "dtype",
                    "percent",
                    "count",
                    "top",
                    "top_percent",
                    "min",
                    "max",
                    "range",
                ]:
                    values[elem] += tmp[elem]
                for elem in [
                    "avg",
                    "stddev",
                    "approx_25%",
                    "approx_50%",
                    "approx_75%",
                    "empty",
                ]:
                    values[elem] += [None] * len(datecols)
            if catcol:
                tmp = self.aggregate(
                    func=[
                        "dtype",
                        "percent",
                        "count",
                        "top",
                        "top_percent",
                        "AVG(LENGTH({}::varchar)) AS avg",
                        "STDDEV(LENGTH({}::varchar)) AS stddev",
                        "MIN(LENGTH({}::varchar))::int AS min",
                        """APPROXIMATE_PERCENTILE(LENGTH({}::varchar) 
                                USING PARAMETERS percentile = 0.25)::int AS 'approx_25%'""",
                        """APPROXIMATE_PERCENTILE(LENGTH({}::varchar) 
                                USING PARAMETERS percentile = 0.5)::int AS 'approx_50%'""",
                        """APPROXIMATE_PERCENTILE(LENGTH({}::varchar) 
                                USING PARAMETERS percentile = 0.75)::int AS 'approx_75%'""",
                        "MAX(LENGTH({}::varchar))::int AS max",
                        "MAX(LENGTH({}::varchar))::int - MIN(LENGTH({}::varchar))::int AS range",
                        "SUM(CASE WHEN LENGTH({}::varchar) = 0 THEN 1 ELSE 0 END) AS empty",
                    ],
                    columns=catcol,
                    ncols_block=ncols_block,
                    processes=processes,
                ).values
                for elem in [
                    "index",
                    "dtype",
                    "percent",
                    "count",
                    "top",
                    "top_percent",
                    "avg",
                    "stddev",
                    "min",
                    "approx_25%",
                    "approx_50%",
                    "approx_75%",
                    "max",
                    "range",
                    "empty",
                ]:
                    values[elem] += tmp[elem]
            for i in range(len(values["index"])):
                dtype[values["index"][i]] = values["dtype"][i]
                percent[values["index"][i]] = values["percent"][i]

        if unique:
            values["unique"] = self.aggregate(
                ["unique"],
                columns=columns,
                ncols_block=ncols_block,
                processes=processes,
            ).values["unique"]

        self._update_catalog(TableSample(values).transpose().values)
        values["index"] = quote_ident(values["index"])
        result = TableSample(values, percent=percent, dtype=dtype).decimal_to_float()
        if method == "all":
            result = result.transpose()

        return result

    @save_verticapy_logs
    def groupby(
        self,
        columns: SQLColumns,
        expr: Optional[SQLExpression] = None,
        rollup: Union[bool, list[bool]] = False,
        having: Optional[str] = None,
    ) -> "vDataFrame":
        """
        This method facilitates the aggregation of the vDataFrame by
        grouping its elements based on one or more specified criteria.
        Grouping is a critical operation in data analysis, as it allows
        us to segment data into subsets, making it easier to apply
        various aggregation functions and gain insights specific to
        each group.

        The ``groupby`` method can be applied to one or more columns, and
        it is particularly valuable when we want to calculate aggregate
        statistics or perform operations within distinct categories or
        segments of our data. By grouping the elements, we can perform
        custom analyses, create summary statistics, or uncover patterns
        that might not be apparent when looking at the entire dataset as
        a whole. It is a foundational method in data analysis and is used
        extensively to explore and understand data dynamics in numerous
        domains.

        Parameters
        ----------
        columns: SQLColumns
            List  of the  vDataColumns  used  to group the elements  or a
            customized expression.  If rollup is set to True, this can be
            a list of tuples.
        expr: SQLExpression, optional
            List of  the  different  aggregations  in  pure SQL.  Aliases
            can  be  used.  For  example, ``SUM(column)``  or  ``AVG(column)
            AS  my_new_alias``  are  valid  whereas ``AVG``  is  invalid.
            Aliases  are recommended to keep the track  of  the  features
            and  to  prevent  ambiguous  names.  For  example,  the  MODE
            function  does  not exist,  but can  be replicated  by  using
            the ``analytic`` method and then grouping the result.
        rollup: bool / list of bools, optional
            If set to True, the rollup operator is used.
            If  set to a list of bools, the  rollup  operator is  used on
            the  matching indexes  and the  length of ``rollup`` must match
            the length of ``columns``.
            For example, for ``columns = ['col1', ('col2', 'col3'), 'col4']``
            and ``rollup = [False, True, True]``, the rollup operator is used
            on the set ('col2', 'col3') and on 'col4'.
        having: str, optional
            Expression used to filter the result.

        Returns
        -------
        vDataFrame
            object result of the grouping.

        Examples
        --------
        For this example, we will use the following dataset:

        .. code-block:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C'],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )

        You can perform grouping using a direct SQL statement.

        .. code-block:: python

            data.groupby(
                columns = ["x"],
                expr = ["AVG(y) AS avg_y", "MIN(z) AS min_z"],
            )

        .. ipython:: python
            :suppress:

            import verticapy as vp
            data = vp.vDataFrame(
                {
                    "x": ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C'],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            result = data.groupby(
                columns = ["x"],
                expr = ["AVG(y) AS avg_y", "MIN(z) AS min_z"],
            )
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_groupby_table.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_groupby_table.html

        Alternatively, you can achieve grouping using VerticaPy SQL
        functions, which offer a more Pythonic approach.

        .. code-block:: python

            import verticapy.sql.functions as vpf

            data.groupby(
                columns = ["x"],
                expr = [
                    vpf.avg(data["y"])._as("avg_y"),
                    vpf.min(data["z"])._as("min_z"),
                ],
            )

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_groupby_table.html

        You can also perform rollup aggregations.

        .. code-block:: python

            data.groupby(
                columns = ["x"],
                expr = [
                    vpf.avg(data["y"])._as("avg_y"),
                    vpf.min(data["z"])._as("min_z"),
                ],
                rollup = True,
            )

        .. ipython:: python
            :suppress:

            import verticapy.sql.functions as vpf
            result = data.groupby(
                columns = ["x"],
                expr = [
                    vpf.avg(data["y"])._as("avg_y"),
                    vpf.min(data["z"])._as("min_z"),
                ],
                rollup = True,
            )
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_groupby_table_2.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_groupby_table_2.html

        .. note:: All the calculations are pushed to the database.

        .. hint:: For additional aggregation options, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataColumn.aggregate` : Aggregations for a specific column.
            | :py:meth:`verticapy.vDataFrame.aggregate` : Aggregates for particular columns.
        """
        columns, expr = format_type(columns, expr, dtype=list)
        assert not isinstance(rollup, list) or len(rollup) == len(columns), ValueError(
            "If parameter 'rollup' is of type list, it should have "
            "the same length as the 'columns' parameter."
        )
        columns_to_select = []
        if isinstance(rollup, bool) and rollup:
            rollup_expr = "ROLLUP(" if rollup else ""
        else:
            rollup_expr = ""
        for idx, elem in enumerate(columns):
            if isinstance(elem, tuple) and rollup:
                if isinstance(rollup, bool) and rollup:
                    rollup_expr += "("
                elif isinstance(rollup[idx], bool) and rollup[idx]:
                    rollup_expr += "ROLLUP("
                elif not isinstance(rollup[idx], bool):
                    raise ValueError(
                        "When parameter 'rollup' is not a boolean, it "
                        "has to be a list of booleans."
                    )
                for item in elem:
                    colname = self.format_colnames(item)
                    if colname:
                        rollup_expr += colname
                        columns_to_select += [colname]
                    else:
                        rollup_expr += str(item)
                        columns_to_select += [item]
                    rollup_expr += ", "
                rollup_expr = rollup_expr[:-2] + "), "
            elif isinstance(elem, str):
                colname = self.format_colnames(elem)
                if colname:
                    if (
                        not isinstance(rollup, bool)
                        and isinstance(rollup[idx], bool)
                        and (rollup[idx])
                    ):
                        rollup_expr += "ROLLUP(" + colname + ")"
                    else:
                        rollup_expr += colname
                    columns_to_select += [colname]
                else:
                    if (
                        not isinstance(rollup, bool)
                        and isinstance(rollup[idx], bool)
                        and (rollup[idx])
                    ):
                        rollup_expr += "ROLLUP(" + str(elem) + ")"
                    else:
                        rollup_expr += str(elem)
                    columns_to_select += [elem]
                rollup_expr += ", "
            else:
                raise ValueError(
                    "Parameter 'columns' must be a string; list of strings "
                    "or tuples (only when rollup is set to True)."
                )
        rollup_expr = rollup_expr[:-2]
        if isinstance(rollup, bool) and rollup:
            rollup_expr += ")"
        if having:
            having = f" HAVING {having}"
        else:
            having = ""
        columns_str = ", ".join(
            [str(elem) for elem in columns_to_select] + [str(elem) for elem in expr]
        )
        if not rollup:
            rollup_expr_str = ", ".join(
                [
                    str(i + 1)
                    for i in range(len([str(elem) for elem in columns_to_select]))
                ],
            )
        else:
            rollup_expr_str = rollup_expr
        query = f"""
            SELECT 
                {columns_str} 
            FROM {self} 
            GROUP BY {rollup_expr_str}{having}"""
        if not rollup:
            rollup_expr_str = ", ".join([str(c) for c in columns_to_select])
        else:
            rollup_expr_str = rollup_expr
        return create_new_vdf(query)

    # Single Aggregate Functions.

    @save_verticapy_logs
    def aad(
        self,
        columns: Optional[SQLColumns] = None,
        **agg_kwargs,
    ) -> TableSample:
        """
        Utilizes the ``aad`` (Average Absolute Deviation) aggregation
        method to analyze the vDataColumn. ``AAD`` measures the average
        absolute deviation of data points from their mean, offering
        valuable insights into data variability and dispersion.
        When we aggregate the vDataFrame using ``aad``, we gain an
        understanding of how data points deviate from the mean on
        average, which is particularly useful for assessing data
        spread and the magnitude of deviations.

        This method is valuable in scenarios where we want to evaluate
        data variability while giving equal weight to all data points,
        regardless of their direction of deviation. Calculating ``aad``
        provides us with information about the overall data consistency
        and can be useful in various analytical and quality assessment
        contexts.

        .. warning::

            To compute aad, VerticaPy needs to execute multiple
            queries. It necessitates, at a minimum, a query that
            includes a subquery to perform this type of aggregation.
            This complexity is the reason why calculating aad
            is typically slower than some other types of aggregations.

        Parameters
        ----------
        columns: SQLColumns, optional
            List of the  vDataColumns names.  If empty, all
            vDataColumns are used.
        **agg_kwargs
            Any optional parameter to pass to the Aggregate
            function.

        Returns
        -------
        TableSample
            result.

        Examples
        --------
        For this example, we will use the following dataset:

        .. code-block:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )

        Now, let's calculate the average absolute deviation for
        specific columns.

        .. code-block:: python

            data.aad(
                columns = ["x", "y", "z"],
            )

        .. ipython:: python
            :suppress:

            import verticapy as vp
            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            result = data.aad(
                columns = ["x", "y", "z"],
            )
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_aad_table.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_aad_table.html

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataColumn.aad` :
              Average Absolute Deviation for a specific column.
            | :py:meth:`verticapy.vDataFrame.std` :
              Standard Deviation for particular columns.
        """
        return self.aggregate(func=["aad"], columns=columns, **agg_kwargs)

    @save_verticapy_logs
    def all(
        self,
        columns: SQLColumns,
        **agg_kwargs,
    ) -> TableSample:
        """
        Applies the ``BOOL_AND`` aggregation method to the vDataFrame.
        ``BOOL_AND``, or Boolean AND, evaluates whether all the
        conditions within a set of Boolean values are ``true``.
        This is useful when you need to ascertain if every condition
        holds. It is particularly handy when working with binary data
        or to ensure that all specified conditions are met within the
        dataset.

        Parameters
        ----------
        columns: SQLColumns, optional
            List of the  vDataColumns names.  If empty, all
            vDataColumns are used.
        **agg_kwargs
            Any optional parameter to pass to the Aggregate
            function.

        Returns
        -------
        TableSample
            result.

        Examples
        --------
        For this example, we will use the following dataset:

        .. code-block:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [True, False, False],
                    "y": [False, False, False],
                    "z": [True, True, True],
                }
            )

        Now, let's use the ``all`` aggregator for specific columns.

        .. code-block:: python

            data.all(
                columns = ["x", "y", "z"],
            )

        .. ipython:: python
            :suppress:

            import verticapy as vp
            data = vp.vDataFrame(
                {
                    "x": [True, False, False],
                    "y": [False, False, False],
                    "z": [True, True, True],
                }
            )
            result = data.all(
                columns = ["x", "y", "z"],
            )
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_all_table.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_all_table.html

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataFrame.any` : Boolean OR Aggregation.
        """
        return self.aggregate(func=["bool_and"], columns=columns, **agg_kwargs)

    @save_verticapy_logs
    def any(
        self,
        columns: SQLColumns,
        **agg_kwargs,
    ) -> TableSample:
        """
        Uses the ``BOOL_OR`` aggregation method in the vDataFrame.
        This method checks if at least one ``true`` condition exists
        within a set of Boolean values. It's particularly handy
        for situations involving binary data or when you need to
        determine if any of the conditions are met.

        Parameters
        ----------
        columns: SQLColumns, optional
            List of the  vDataColumns names.  If empty, all
            vDataColumns are used.
        **agg_kwargs
            Any optional parameter to pass to the Aggregate
            function.

        Returns
        -------
        TableSample
            result.

        Examples
        --------
        For this example, we will use the following dataset:

        .. code-block:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [True, False, False],
                    "y": [False, False, False],
                    "z": [True, True, True],
                }
            )

        Now, let's use the ``any`` aggregator for specific columns.

        .. code-block:: python

            data.any(
                columns = ["x", "y", "z"],
            )

        .. ipython:: python
            :suppress:

            import verticapy as vp
            data = vp.vDataFrame(
                {
                    "x": [True, False, False],
                    "y": [False, False, False],
                    "z": [True, True, True],
                }
            )
            result = data.any(
                columns = ["x", "y", "z"],
            )
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_any_table.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_any_table.html

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataFrame.all` : Boolean AND Aggregation.
        """
        return self.aggregate(func=["bool_or"], columns=columns, **agg_kwargs)

    @save_verticapy_logs
    def avg(
        self,
        columns: Optional[SQLColumns] = None,
        **agg_kwargs,
    ) -> TableSample:
        """
        This operation aggregates the vDataFrame using the ``AVG``
        aggregation, which calculates the average value for the
        selected column or columns. It provides insights into the
        central tendency of the data and is a fundamental statistical
        measure often used in data analysis and reporting.

        Parameters
        ----------
        columns: SQLColumns, optional
            List of the  vDataColumns names.  If empty, all
            vDataColumns are used.
        **agg_kwargs
            Any optional parameter to pass to the Aggregate
            function.

        Returns
        -------
        TableSample
            result.

        Examples
        --------
        For this example, we will use the following dataset:

        .. code-block:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )

        Now, let's calculate the averages for specific columns.

        .. code-block:: python

            data.avg(
                columns = ["x", "y", "z"],
            )

        .. ipython:: python
            :suppress:

            import verticapy as vp
            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            result = data.avg(
                columns = ["x", "y", "z"],
            )
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_avg_table.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_avg_table.html

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataColumn.avg` : Aggregations for a specific column.
            | :py:meth:`verticapy.vDataFrame.max` : Maximum for particular columns.
            | :py:meth:`verticapy.vDataFrame.min` : Minimum for particular columns.
        """
        return self.aggregate(func=["avg"], columns=columns, **agg_kwargs)

    mean = avg

    @save_verticapy_logs
    def count(
        self,
        columns: Optional[SQLColumns] = None,
        **agg_kwargs,
    ) -> TableSample:
        """
        This operation aggregates the vDataFrame using the
        ``COUNT`` aggregation, providing the count of non-missing
        values for specified columns. This is valuable for
        assessing data completeness and quality.

        Parameters
        ----------
        columns: SQLColumns, optional
            List of the  vDataColumns names.  If empty, all
            vDataColumns are used.
        **agg_kwargs
            Any optional parameter to pass to the Aggregate
            function.

        Returns
        -------
        TableSample
            result.

        Examples
        --------
        For this example, we will use the following dataset:

        .. code-block:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )

        Now, let`s calculate the count for specific columns.

        .. code-block:: python

            data.count(
                columns = ["x", "y", "z"],
            )

        .. ipython:: python
            :suppress:

            import verticapy as vp
            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            result = data.count(
                columns = ["x", "y", "z"],
            )
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_count_table.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_count_table.html

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataColumn.count` : Count for a specific column.
            | :py:meth:`verticapy.vDataFrame.count_percent` : Count Percent for particular columns.
        """
        return self.aggregate(func=["count"], columns=columns, **agg_kwargs)

    @save_verticapy_logs
    def kurtosis(
        self,
        columns: Optional[SQLColumns] = None,
        **agg_kwargs,
    ) -> TableSample:
        """
        Calculates the kurtosis of the vDataFrame to obtain a measure
        of the data's peakedness or tailness. The kurtosis statistic
        helps us understand the shape of the data distribution.
        It quantifies whether the data has heavy tails or is more peaked
        relative to a normal distribution.

        By aggregating the vDataFrame with kurtosis, we can gain valuable
        insights into the data's distribution characteristics.

        .. warning::

            To compute kurtosis, VerticaPy needs to execute multiple
            queries. It necessitates, at a minimum, a query that
            includes a subquery to perform this type of aggregation.
            This complexity is the reason why calculating kurtosis
            is typically slower than some other types of aggregations.

        Parameters
        ----------
        columns: SQLColumns, optional
            List of the vDataColumns names. If empty, all vDataColumns
            are used.
        **agg_kwargs
            Any  optional parameter to pass to the Aggregate  function.

        Returns
        -------
        TableSample
            result.

        Examples
        --------
        For this example, we will use the following dataset:

        .. code-block:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )

        Now, let's calculate the kurtosis for specific columns.

        .. code-block:: python

            data.kurtosis(
                columns = ["x", "y", "z"],
            )

        .. ipython:: python
            :suppress:

            import verticapy as vp
            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            result = data.kurtosis(
                columns = ["x", "y", "z"],
            )
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_kurtosis_table.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_kurtosis_table.html

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataColumn.kurtosis` : Kurtosis for a specific column.
            | :py:meth:`verticapy.vDataFrame.skewness` : Skewness for particular columns.
            | :py:meth:`verticapy.vDataFrame.std` : Standard Deviation for particular columns.
        """
        return self.aggregate(func=["kurtosis"], columns=columns, **agg_kwargs)

    kurt = kurtosis

    @save_verticapy_logs
    def mad(
        self,
        columns: Optional[SQLColumns] = None,
        **agg_kwargs,
    ) -> TableSample:
        """
        Utilizes the ``mad`` (Median Absolute Deviation) aggregation
        method with the vDataFrame. ``MAD`` measures the dispersion
        of data points around the median, and it is particularly
        valuable for assessing the robustness of data in the
        presence of outliers. When we aggregate the vDataFrame
        using ``mad``, we gain insights into the variability and
        the degree to which data points deviate from the median.

        This is especially useful for datasets where we want to
        understand the spread of values while being resistant to
        the influence of extreme outliers. Calculating ``mad`` can
        involve robust statistical computations, making it a useful
        tool for outlier-robust analysis and data quality evaluation.

        .. warning::

            To compute mad, VerticaPy needs to execute multiple
            queries. It necessitates, at a minimum, a query that
            includes a subquery to perform this type of aggregation.
            This complexity is the reason why calculating mad
            is typically slower than some other types of aggregations.

        Parameters
        ----------
        columns: SQLColumns, optional
            List of the vDataColumns names. If empty, all vDataColumns
            are used.
        **agg_kwargs
            Any  optional parameter to pass to the Aggregate  function.

        Returns
        -------
        TableSample
            result.

        Examples
        --------
        For this example, we will use the following dataset:

        .. code-block:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )

        Now, let's calculate the median absolute deviation for
        specific columns.

        .. code-block:: python

            data.mad(
                columns = ["x", "y", "z"],
            )

        .. ipython:: python
            :suppress:

            import verticapy as vp
            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            result = data.mad(
                columns = ["x", "y", "z"],
            )
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_mad_table.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_mad_table.html

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataFrame.std` : Mean Absolute Deviation for particular columns.
            | :py:meth:`verticapy.vDataColumn.mad` : Standard Deviation for a specific column.
        """
        return self.aggregate(func=["mad"], columns=columns, **agg_kwargs)

    @save_verticapy_logs
    def max(
        self,
        columns: Optional[SQLColumns] = None,
        **agg_kwargs,
    ) -> TableSample:
        """
        Aggregates the vDataFrame by applying the ``MAX`` aggregation,
        which calculates the maximum value, for the specified
        columns. This aggregation provides insights into the highest
        values within the dataset, aiding in understanding the data
        distribution.

        Parameters
        ----------
        columns: SQLColumns, optional
            List of the vDataColumns names. If empty, all vDataColumns
            are used.
        **agg_kwargs
            Any optional parameter  to pass to  the Aggregate function.

        Returns
        -------
        TableSample
            result.

        Examples
        --------
        For this example, we will use the following dataset:

        .. code-block:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )

        Now, let's calculate the maximum for specific columns.

        .. code-block:: python

            data.max(
                columns = ["x", "y", "z"],
            )

        .. ipython:: python
            :suppress:

            import verticapy as vp
            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            result = data.max(
                columns = ["x", "y", "z"],
            )
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_max_table.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_max_table.html

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataFrame.min` : Minimum for particular columns.
            | :py:meth:`verticapy.vDataColumn.max` : Maximum for a specific column.
        """
        return self.aggregate(func=["max"], columns=columns, **agg_kwargs)

    @save_verticapy_logs
    def median(
        self,
        columns: Optional[SQLColumns] = None,
        approx: bool = True,
        **agg_kwargs,
    ) -> TableSample:
        """
        Aggregates the vDataFrame using the ``MEDIAN`` or ``APPROX_MEDIAN``
        aggregation, which calculates the median value for the specified
        columns. The median is a robust measure of central tendency and
        helps in understanding the distribution of data, especially in
        the presence of outliers.

        .. warning::

            When you set ``approx`` to True, the approximate median is
            computed, which is significantly faster than the exact
            calculation. However, be cautious when setting ``approx``
            to False, as it can significantly slow down the performance.

        Parameters
        ----------
        columns: SQLColumns, optional
            List of the vDataColumns names. If empty, all numerical vDataColumns are
            used.
        approx: bool, optional
            If set to True, the approximate median is returned. By setting this
            parameter to False, the function`s performance can drastically decrease.
        **agg_kwargs
            Any optional parameter to pass to the Aggregate function.

        Returns
        -------
        TableSample
            result.

        Examples
        --------
        For this example, we will use the following dataset:

        .. code-block:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )

        Now, let's calculate the median for specific columns.

        .. code-block:: python

            data.median(
                columns = ["x", "y", "z"],
                approx = True,
            )

        .. ipython:: python
            :suppress:

            import verticapy as vp
            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            result = data.median(
                columns = ["x", "y", "z"],
                approx = True,
            )
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_median_table.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_median_table.html

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataFrame.max` : Maximum for particular columns.
            | :py:meth:`verticapy.vDataFrame.min` : Maximum for particular columns.
            | :py:meth:`verticapy.vDataColumn.mean` : Mean for a specific column.
        """
        return self.quantile(
            0.5,
            columns=columns,
            approx=approx,
            **agg_kwargs,
        )

    @save_verticapy_logs
    def min(
        self,
        columns: Optional[SQLColumns] = None,
        **agg_kwargs,
    ) -> TableSample:
        """
        Aggregates the vDataFrame by applying the ``MIN`` aggregation,
        which calculates the minimum value, for the specified
        columns. This aggregation provides insights into the lowest
        values within the dataset, aiding in understanding the data
        distribution.

        Parameters
        ----------
        columns: SQLColumns, optional
            List of the vDataColumns names. If empty, all numerical vDataColumns are
            used.
        **agg_kwargs
            Any optional parameter to pass to the Aggregate function.

        Returns
        -------
        TableSample
            result.

        Examples
        --------
        For this example, we will use the following dataset:

        .. code-block:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )

        Now, let's calculate the minimum for specific columns.

        .. code-block:: python

            data.min(
                columns = ["x", "y", "z"],
            )

        .. ipython:: python
            :suppress:

            import verticapy as vp
            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            result = data.min(
                columns = ["x", "y", "z"],
            )
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_min_table.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_min_table.html

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataColumn.min` : Minimum for a specific column.
            | :py:meth:`verticapy.vDataFrame.max` : Maximum for particular columns.
        """
        return self.aggregate(func=["min"], columns=columns, **agg_kwargs)

    @save_verticapy_logs
    def product(
        self,
        columns: Optional[SQLColumns] = None,
        **agg_kwargs,
    ) -> TableSample:
        """
        Aggregates the vDataFrame by applying the ``product``
        aggregation function. This function computes the
        product of values within the dataset, providing
        insights into the multiplication of data points.

        The ``product`` aggregation can be particularly useful
        when we need to assess cumulative effects or when
        multiplying values is a key aspect of the analysis.
        This operation can be relevant in various domains,
        such as finance, economics, and engineering, where
        understanding the combined impact of values is
        critical for decision-making and modeling.

        .. note::

            Since ``product`` is not a conventional SQL
            aggregation, we employ a unique approach by
            combining the sum of logarithms and the
            exponential function for its computation.
            This non-standard methodology is utilized to
            derive the product of values within the dataset,
            offering a distinctive way to understand the
            multiplicative effects of data points.

        Parameters
        ----------
        columns: SQLColumns, optional
            List  of the vDataColumn  names.  If empty, all
            numerical vDataColumns are used.
        **agg_kwargs
            Any optional parameter to pass to the Aggregate
            function.

        Returns
        -------
        TableSample
            result.

        Examples
        --------
        For this example, we will use the following dataset:

        .. code-block:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )

        Now, let's calculate the product for specific columns.

        .. code-block:: python

            data.product(
                columns = ["x", "y", "z"],
            )

        .. ipython:: python
            :suppress:

            import verticapy as vp
            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            result = data.product(
                columns = ["x", "y", "z"],
            )
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_product_table.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_product_table.html

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataFrame.aggregate` : Aggregates for particular columns.
            | :py:meth:`verticapy.vDataFrame.quantile` : Quantile Aggregates for particular columns.
        """
        return self.aggregate(func=["prod"], columns=columns, **agg_kwargs)

    prod = product

    @save_verticapy_logs
    def quantile(
        self,
        q: Union[PythonNumber, ArrayLike],
        columns: Optional[SQLColumns] = None,
        approx: bool = True,
        **agg_kwargs,
    ) -> TableSample:
        """
        Aggregates the vDataFrame using specified ``quantile``.
        The ``quantile`` function is an indispensable tool for
        comprehending data distribution. By providing a quantile
        value as input, this aggregation method helps us identify
        the data point below which a certain percentage of the data
        falls. This can be pivotal for tasks like analyzing data
        distributions, assessing skewness, and determining essential
        percentiles such as medians or quartiles.

        .. warning::

            It's important to note that the ``quantile`` aggregation
            operates in two distinct modes, allowing flexibility in
            computation. Depending on the ``approx`` parameter, it can
            use either ``APPROXIMATE_QUANTILE`` or ``QUANTILE`` methods
            to derive the final aggregation. The ``APPROXIMATE_QUANTILE``
            method provides faster results by estimating the quantile
            values with an approximation technique, while ``QUANTILE``
            calculates precise quantiles through rigorous computation.
            This choice empowers users to strike a balance between
            computational efficiency and the level of precision
            required for their specific data analysis tasks.

        Parameters
        ----------
        q: PythonNumber / ArrayLike
            List of  the  different quantiles. They must be
            numbers between 0 and 1.
            For example [0.25, 0.75] will return  Q1 and Q3.
        columns: SQLColumns, optional
            List  of  the   vDataColumns  names.  If  empty,
            all numerical vDataColumns are used.
        approx: bool, optional
            If  set  to  True,  the approximate quantile is
            returned. By  setting  this parameter to  False,
            the  function's  performance   can  drastically
            decrease.
        **agg_kwargs
            Any optional parameter to pass to the Aggregate
            function.

        Returns
        -------
        TableSample
            result.

        Examples
        --------
        For this example, we will use the following dataset:

        .. code-block:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )

        Now, let's calculate some approximate quantiles for
        specific columns.

        .. code-block:: python

            data.quantile(
                q = [0.1, 0.2, 0.5, 0.9],
                columns = ["x", "y", "z"],
                approx = True,
            )

        .. ipython:: python
            :suppress:

            import verticapy as vp
            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            result = data.quantile(
                q = [0.1, 0.2, 0.5, 0.9],
                columns = ["x", "y", "z"],
                approx = True,
            )
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_quantile_table.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_quantile_table.html

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataColumn.aggregate` : Aggregations for a specific column.
            | :py:meth:`verticapy.vDataFrame.aggregate` : Aggregates for particular columns.
        """
        if isinstance(q, (int, float)):
            q = [q]
        prefix = "approx_" if approx else ""
        return self.aggregate(
            func=[verticapy_agg_name(prefix + f"{float(item) * 100}%") for item in q],
            columns=columns,
            **agg_kwargs,
        )

    @save_verticapy_logs
    def sem(
        self,
        columns: Optional[SQLColumns] = None,
        **agg_kwargs,
    ) -> TableSample:
        """
        Leverages the ``sem`` (Standard Error of the Mean) aggregation
        technique to perform analysis and aggregation on the vDataFrame.
        Standard Error of the Mean is a valuable statistical measure used
        to estimate the precision of the sample mean as an approximation
        of the population mean.

        When we aggregate the vDataFrame using ``sem``, we gain insights
        into the variability or uncertainty associated with the sample
        mean. This measure helps us assess the reliability of the sample
        mean as an estimate of the true population mean.

        It is worth noting that computing the Standard Error of the Mean
        requires statistical calculations and can be particularly useful
        when evaluating the precision of sample statistics or making
        inferences about a larger dataset based on a sample.

        .. warning::

            To compute sem, VerticaPy needs to execute multiple
            queries. It necessitates, at a minimum, a query that
            includes a subquery to perform this type of aggregation.
            This complexity is the reason why calculating sem is
            typically slower than some other types of aggregations.

        Parameters
        ----------
        columns: SQLColumns, optional
            List  of the vDataColumns names. If empty,  all
            numerical vDataColumns are used.
        **agg_kwargs
            Any optional parameter to pass to the Aggregate
            function.

        Returns
        -------
        TableSample
            result.

        Examples
        --------
        For this example, we will use the following dataset:

        .. code-block:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )

        Now, let's calculate the standard error of the mean
        for specific columns.

        .. code-block:: python

            data.sem(
                columns = ["x", "y", "z"],
            )

        .. ipython:: python
            :suppress:

            import verticapy as vp
            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            result = data.sem(
                columns = ["x", "y", "z"],
            )
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_sem_table.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_sem_table.html

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataColumn.sem` : Standard Error of Mean for a specific column.
            | :py:meth:`verticapy.vDataFrame.mad` : Mean Absolute Deviation for particular columns.
        """
        return self.aggregate(func=["sem"], columns=columns, **agg_kwargs)

    @save_verticapy_logs
    def skewness(
        self,
        columns: Optional[SQLColumns] = None,
        **agg_kwargs,
    ) -> TableSample:
        """
        Utilizes the ``skewness`` aggregation method to analyze and
        aggregate the vDataFrame. Skewness, a measure of the asymmetry
        in the data's distribution, helps us understand the data's
        deviation from a perfectly symmetrical distribution. When we
        aggregate the vDataFrame using skewness, we gain insights into
        the data's tendency to be skewed to the left or right, or if
        it follows a normal distribution.

        .. warning::

            To compute skewness, VerticaPy needs to execute multiple
            queries. It necessitates, at a minimum, a query that
            includes a subquery to perform this type of aggregation.
            This complexity is the reason why calculating skewness
            is typically slower than some other types of aggregations.

        Parameters
        ----------
        columns: SQLColumns, optional
            List of the  vDataColumns names.  If empty, all
            numerical vDataColumns are used.
        **agg_kwargs
            Any optional parameter to pass to the Aggregate
            function.

        Returns
        -------
        TableSample
            result.

        Examples
        --------
        For this example, we will use the following dataset:

        .. code-block:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )

        Now, let's calculate the skewness for specific columns.

        .. code-block:: python

            data.skewness(
                columns = ["x", "y", "z"],
            )

        .. ipython:: python
            :suppress:

            import verticapy as vp
            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            result = data.skewness(
                columns = ["x", "y", "z"],
            )
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_skewness_table.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_skewness_table.html

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataColumn.kurtosis` : Kurtosis for a specific column.
            | :py:meth:`verticapy.vDataColumn.skewness` : Skewness for a specific column.
            | :py:meth:`verticapy.vDataFrame.std` : Standard Deviation for particular columns.
        """
        return self.aggregate(func=["skewness"], columns=columns, **agg_kwargs)

    skew = skewness

    @save_verticapy_logs
    def std(
        self,
        columns: Optional[SQLColumns] = None,
        **agg_kwargs,
    ) -> TableSample:
        """
        Aggregates the vDataFrame using ``STDDEV`` aggregation
        (Standard Deviation), providing insights into the
        spread or variability of data for the selected columns.
        The standard deviation is a measure of how much individual
        data points deviate from the mean, helping to assess data
        consistency and variation.

        Parameters
        ----------
        columns: SQLColumns, optional
            List  of the vDataColumns names.  If empty, all
            numerical vDataColumns are used.
        **agg_kwargs
            Any optional parameter to pass to the Aggregate
            function.

        Returns
        -------
        TableSample
            result.

        Examples
        --------
        For this example, we will use the following dataset:

        .. code-block:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )

        Now, let's calculate the standard deviation for specific columns.

        .. code-block:: python

            data.std(
                columns = ["x", "y", "z"],
            )

        .. ipython:: python
            :suppress:

            import verticapy as vp
            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            result = data.std(
                columns = ["x", "y", "z"],
            )
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_std_table.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_std_table.html

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataColumn.kurtosis` : Kurtosis for a specific column.
            | :py:meth:`verticapy.vDataFrame.skewness` : Skewness for particular columns.
            | :py:meth:`verticapy.vDataColumn.std` : Standard Deviation for a specific column.
        """
        return self.aggregate(func=["stddev"], columns=columns, **agg_kwargs)

    stddev = std

    @save_verticapy_logs
    def sum(
        self,
        columns: Optional[SQLColumns] = None,
        **agg_kwargs,
    ) -> TableSample:
        """
        Aggregates the vDataFrame using ``SUM`` aggregation, which
        computes the total sum of values for the specified columns,
        providing a cumulative view of numerical data.

        Parameters
        ----------
        columns: SQLColumns, optional
            List of the  vDataColumns names.  If empty, all
            numerical vDataColumns are used.
        **agg_kwargs
            Any optional parameter to pass to the Aggregate
            function.

        Returns
        -------
        TableSample
            result.

        Examples
        --------
        For this example, we will use the following dataset:

        .. code-block:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )

        Now, let's calculate the sum for specific columns.

        .. code-block:: python

            data.sum(
                columns = ["x", "y", "z"],
            )

        .. ipython:: python
            :suppress:

            import verticapy as vp
            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            result = data.sum(
                columns = ["x", "y", "z"],
            )
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_sum_table.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_sum_table.html

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataColumn.sum` : Sum for a specific column.
            | :py:meth:`verticapy.vDataFrame.max` : Maximum for particular columns.
        """
        return self.aggregate(func=["sum"], columns=columns, **agg_kwargs)

    @save_verticapy_logs
    def var(
        self,
        columns: Optional[SQLColumns] = None,
        **agg_kwargs,
    ) -> TableSample:
        """
        Aggregates the vDataFrame using ``VAR`` aggregation
        (Variance), providing insights into the spread or
        variability of data for the selected columns.
        The variance is a measure of how much individual
        data points deviate from the mean, helping to assess
        data consistency and variation.

        Parameters
        ----------
        columns: SQLColumns, optional
            List of the  vDataColumns  names. If empty, all
            numerical vDataColumns are used.
        **agg_kwargs
            Any optional parameter to pass to the Aggregate
            function.

        Returns
        -------
        TableSample
            result.

        Examples
        --------
        For this example, we will use the following dataset:

        .. code-block:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )

        Now, let's calculate the variance for specific columns.

        .. code-block:: python

            data.var(
                columns = ["x", "y", "z"],
            )

        .. ipython:: python
            :suppress:

            import verticapy as vp
            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            result = data.var(
                columns = ["x", "y", "z"],
            )
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_var_table.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_var_table.html

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataColumn.kurtosis` : Kurtosis for a specific column.
            | :py:meth:`verticapy.vDataColumn.skewness` : Skewness for a specific column.
            | :py:meth:`verticapy.vDataFrame.std` : Standard Deviation for particular columns.
        """
        return self.aggregate(func=["variance"], columns=columns, **agg_kwargs)

    variance = var

    # TOPK.

    @save_verticapy_logs
    def count_percent(
        self,
        columns: Optional[SQLColumns] = None,
        sort_result: bool = True,
        desc: bool = True,
        **agg_kwargs,
    ) -> TableSample:
        """
        Performs aggregation on the vDataFrame using a list of
        aggregate functions, including ``count`` and ``percent``.
        The ``count`` function computes the number of non-missing
        (non-null) values within the dataset, providing us with
        an understanding of the data's completeness.

        On the other hand, the ``percent`` function calculates the
        percentage of non-missing values in relation to the total
        dataset size, offering insights into data integrity and
        completeness as a proportion.

        Parameters
        ----------
        columns: SQLColumns, optional
            List of vDataColumn names. If empty, all vDataColumns
            are used.
        sort_result: bool, optional
            If set to True, the result is sorted.
        desc: bool, optional
            If  set  to  True and ``sort_result`` is  set  to  True,
            the result is sorted in descending order.
        **agg_kwargs
            Any  optional  parameter  to  pass  to  the Aggregate
            function.

        Returns
        -------
        TableSample
            result.

        Examples
        --------
        For this example, we will use the following dataset:

        .. code-block:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )

        Now, let's calculate the count percentage for specific columns.

        .. code-block:: python

            data.count_percent(
                columns = ["x", "y", "z"],
            )

        .. ipython:: python
            :suppress:

            import verticapy as vp
            data = vp.vDataFrame(
                {
                    "x": [None, None, 4, 9, None, 15, None, 22],
                    "y": [1, 2, 1, 2, None, 1, 2, 1],
                    "z": [10, None, 2, 1, 9, 8, None, 3],
                }
            )
            result = data.count_percent(
                columns = ["x", "y", "z"],
            )
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_count_percent_table.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_count_percent_table.html

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataColumn.count` : Count for a specific column.
            | :py:meth:`verticapy.vDataFrame.count` : Count for particular columns.
        """
        result = self.aggregate(
            func=["count", "percent"],
            columns=columns,
            **agg_kwargs,
        )
        if sort_result:
            result.sort("count", desc)
        return result

    # Distincts.

    @save_verticapy_logs
    def nunique(
        self,
        columns: Optional[SQLColumns] = None,
        approx: bool = True,
        **agg_kwargs,
    ) -> TableSample:
        """
        When aggregating the vDataFrame using `nunique` (cardinality),
        VerticaPy employs the COUNT DISTINCT function to determine the
        number of unique values in a particular column. It also offers
        the option to use APPROXIMATE_COUNT_DISTINCT, a more efficient
        approximation method for calculating cardinality.

        .. hint::

            This flexibility allows you to optimize the computation
            based on your specific requirements, keeping in mind
            that using APPROXIMATE_COUNT_DISTINCT can significantly
            improve performance when cardinality estimation is sufficient
            for your analysis.

        .. important::

            To calculate the exact cardinality of a column, you should
            set the parameter `approx` to False. This will ensure that
            the cardinality is computed accurately rather than using the
            approximate method.

        Parameters
        ----------
        columns: SQLColumns, optional
            List of the vDataColumns names. If empty, all vDataColumns
            are used.
        approx: bool, optional
            If set to True, the  approximate cardinality  is  returned.
            By  setting  this  parameter   to  False,  the  function's
            performance can drastically decrease.
        **agg_kwargs
            Any   optional   parameter  to   pass   to  the  Aggregate
            function.

        Returns
        -------
        TableSample
            result.

        Examples
        --------
        For this example, we will use the following dataset:

        .. code-block:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )

        Now, let's calculate the cardinality for specific columns.

        .. code-block:: python

            data.nunique(
                columns = ["x", "y", "z"],
            )

        .. ipython:: python
            :suppress:

            import verticapy as vp
            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            result = data.nunique(
                columns = ["x", "y", "z"],
            )
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_nunique_table.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_nunique_table.html

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataFrame.duplicated` : Duplicate Values for particular columns.
            | :py:meth:`verticapy.vDataColumn.nunique` : Cardinaility for a specific column.
        """
        func = ["approx_unique"] if approx else ["unique"]
        return self.aggregate(func=func, columns=columns, **agg_kwargs)

    # Deals with duplicates.

    @save_verticapy_logs
    def duplicated(
        self, columns: Optional[SQLColumns] = None, count: bool = False, limit: int = 30
    ) -> TableSample:
        """
        This function returns a list or set of values that occur more
        than once within the dataset. It identifies and provides you
        with insight into which specific values or entries are duplicated
        in the dataset, helping to detect and manage data redundancy and
        potential issues related to duplicate information.

        .. warning::

            This function employs the ``ROW_NUMBER`` SQL function with
            multiple partition criteria. It's essential to note that
            as the number of partition columns increases, the
            computational cost can rise significantly. The ``ROW_NUMBER``
            function assigns a unique rank to each row within its
            partition, which means that the more columns are involved
            in partitioning, the more complex and resource-intensive
            the operation becomes. Therefore, when using a large number
            of columns for partitioning, it's important to be mindful
            of potential performance implications, as it may become
            computationally expensive.

        Parameters
        ----------
        columns: SQLColumns, optional
            List of the vDataColumns names. If empty, all vDataColumns
            are selected.
        count: bool, optional
            If set to  True, the  method also returns the count of
            each duplicate.
        limit: int, optional
            Sets a limit on the number of elements to be displayed.

        Returns
        -------
        TableSample
            result.

        Examples
        --------
        For this example, we will use the following dataset:

        .. code-block:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 15, 1, 15, 20, 1],
                    "y": [1, 2, 1, 1, 1, 1, 2, 1],
                    "z": [10, 12, 9, 10, 9, 8, 1, 10],
                }
            )

        Now, let's find duplicated rows.

        .. code-block:: python

            data.duplicated(
                columns = ["x", "y", "z"],
            )

        .. ipython:: python
            :suppress:

            import verticapy as vp
            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 15, 1, 15, 20, 1],
                    "y": [1, 2, 1, 1, 1, 1, 2, 1],
                    "z": [10, 12, 9, 10, 9, 8, 1, 10],
                }
            )
            result = data.duplicated(
                columns = ["x", "y", "z"],
            )
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_duplicated_table.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_duplicated_table.html

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataColumn.nunique` : Cardinality for a specific column.
            | :py:meth:`verticapy.vDataFrame.nunique` : Cardinality for particular columns.
        """
        columns = format_type(columns, dtype=list)
        if len(columns) == 0:
            columns = self.get_columns()
        columns = self.format_colnames(columns)
        columns = ", ".join(columns)
        main_table = f"""
            (SELECT 
                *, 
                ROW_NUMBER() OVER (PARTITION BY {columns}) AS duplicated_index 
             FROM {self}) duplicated_index_table 
             WHERE duplicated_index > 1"""
        if count:
            total = _executeSQL(
                query=f"""
                    SELECT 
                        /*+LABEL('vDataframe.duplicated')*/ COUNT(*) 
                    FROM {main_table}""",
                title="Computing the number of duplicates.",
                method="fetchfirstelem",
                sql_push_ext=self._vars["sql_push_ext"],
                symbol=self._vars["symbol"],
            )
            return total
        result = TableSample.read_sql(
            query=f"""
                SELECT 
                    {columns},
                    MAX(duplicated_index) AS occurrence 
                FROM {main_table} 
                GROUP BY {columns} 
                ORDER BY occurrence DESC LIMIT {limit}""",
            sql_push_ext=self._vars["sql_push_ext"],
            symbol=self._vars["symbol"],
        )
        result.count = _executeSQL(
            query=f"""
                SELECT 
                    /*+LABEL('vDataframe.duplicated')*/ COUNT(*) 
                FROM 
                    (SELECT 
                        {columns}, 
                        MAX(duplicated_index) AS occurrence 
                     FROM {main_table} 
                     GROUP BY {columns}) t""",
            title="Computing the number of distinct duplicates.",
            method="fetchfirstelem",
            sql_push_ext=self._vars["sql_push_ext"],
            symbol=self._vars["symbol"],
        )
        return result


class vDCAgg(vDCEval):
    # Main Aggregate Functions.

    @save_verticapy_logs
    def aggregate(self, func: list) -> TableSample:
        """
        Aggregates the vDataFrame using the input functions.

        Parameters
        ----------
        func: SQLExpression
            List of the different aggregations:

             - aad:
                average absolute deviation.
             - approx_median:
                approximate median.
             - approx_q%:
                approximate q quantile (ex: approx_50%
                for the approximate median).
             - approx_unique:
                approximative cardinality.
             - count:
                number of non-missing elements.
             - cvar:
                conditional value at risk.
             - dtype:
                virtual column type.
             - iqr:
                interquartile range.
             - kurtosis:
                kurtosis.
             - jb:
                Jarque-Bera index.
             - mad:
                median absolute deviation.
             - max:
                maximum.
             - mean:
                average.
             - median:
                median.
             - min:
                minimum.
             - mode:
                most occurent element.
             - percent:
                percent of non-missing elements.
             - q%:
                q quantile (ex: 50% for the median)
                Use the ``approx_q%`` (approximate quantile)
                aggregation to get better performance.
             - prod:
                product.
             - range:
                difference between the max and the min.
             - sem:
                standard error of the mean.
             - skewness:
                skewness.
             - sum:
                sum.
             - std:
                standard deviation.
             - topk:
                kth most occurent element (ex: top1 for the mode).
             - topk_percent:
                kth most occurent element density.
             - unique:
                cardinality (count distinct).
             - var:
                variance.

            Other aggregations will work if supported by your database
            version.
        columns: SQLColumns, optional
            List of  the vDataColumn's names. If empty,  depending on the
            aggregations, all or only numerical vDataColumns are used.
        ncols_block: int, optional
            Number  of columns  used per query.  Setting  this  parameter
            divides  what  would otherwise be  one large query into  many
            smaller  queries  called "blocks", whose size is determine by
            the size of ncols_block.
        processes: int, optional
            Number  of child processes  to  create. Setting  this  with  the
            ncols_block  parameter lets you parallelize a  single query into
            many smaller  queries, where each child process creates  its own
            connection to the database and sends one query. This can improve
            query performance, but consumes  more resources. If processes is
            set to 1, the queries are sent iteratively from a single process.

        Returns
        -------
        TableSample
            result.

        Examples
        --------
        For this example, we will use the following dataset:

        .. code-block:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )

        With the ``aggregate`` method, you have the flexibility to select specific
        aggregates you wish to include in the query. This allows for more precise
        control over the aggregation process and helps tailor the results to your
        specific needs.

        .. code-block:: python

            data["x"].aggregate(
                func = ["min", "approx_10%", "approx_50%", "approx_90%", "max"],
            )

        .. ipython:: python
            :suppress:

            import verticapy as vp
            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            result = data["x"].aggregate(
                func = ["min", "approx_10%", "approx_50%", "approx_90%", "max"],
            )
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDCAgg_aggregate_table.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDCAgg_aggregate_table.html

        .. note:: All the calculations are pushed to the database.

        .. seealso::
            | :py:meth:`verticapy.vDataFrame.aggregate` : Aggregations for specific columns.
            | :py:meth:`verticapy.vDataColumn.describe` :
              Summarizes the information within the column.
            | :py:meth:`verticapy.vDataFrame.describe` :
              Summarizes the information for specific columns.
        """
        return self._parent.aggregate(func=func, columns=[self._alias]).transpose()

    agg = aggregate

    @save_verticapy_logs
    def describe(
        self,
        method: Literal["auto", "numerical", "categorical", "cat_stats"] = "auto",
        max_cardinality: int = 6,
        numcol: Optional[str] = None,
    ) -> TableSample:
        """
        This function aggregates the vDataColumn using multiple statistical
        aggregations such as minimum (min), maximum (max), median, cardinality
        (unique), and other relevant statistics. The specific aggregations
        applied depend on the data types of the vDataColumn. For example,
        numeric columns are aggregated with numerical aggregations (min, median,
        max...), while categorical columns are aggregated using categorical ones
        (cardinality, mode...). This versatile function provides valuable insights
        into the dataset's statistical properties and can be customized to meet
        specific analytical requirements.

        Parameters
        ----------
        method: str, optional
            The describe method.

             - auto:
                Sets  the  method to  ``numerical`` if
                the   vDataColumn    is    numerical,
                ``categorical`` otherwise.
             - categorical:
                Uses  only categorical  aggregations
                during the computation.
             - cat_stats:
                Computes  statistics  of a numerical
                column for each vDataColumn category.
                In this case,  the parameter ``numcol``
                must be defined.
             - numerical:
                Uses  popular numerical aggregations
                during the computation.

        max_cardinality: int, optional
            Cardinality  threshold  to  use  to  determine  if the
            vDataColumn is considered as categorical.
        numcol: str, optional
            Numerical  vDataColumn  to  use  when  the   parameter
            method is set to ``cat_stats``.

        Returns
        -------
        TableSample
            result.

        Examples
        --------
        For this example, we will use the following dataset:

        .. code-block:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                    "c": ['A', 'A', 'A', 'A', 'B', 'B', 'C', 'D'],
                }
            )

        The ``describe`` method provides you with a variety of statistical
        methods.

        The ``numerical`` parameter allows for the computation of numerical
        aggregations.

        .. code-block:: python

            data["x"].describe(method = "numerical")

        .. ipython:: python
            :suppress:

            import verticapy as vp
            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                    "c": ['A', 'A', 'A', 'A', 'B', 'B', 'C', 'D'],
                }
            )
            result = data["x"].describe(method = "numerical")
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDCAgg_describe_num_table.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDCAgg_describe_num_table.html

        The ``categorical`` parameter allows for the computation of categorical
        aggregations.

        .. code-block:: python

            data["x"].describe(method = "categorical")

        .. ipython:: python
            :suppress:

            import verticapy as vp
            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                    "c": ['A', 'A', 'A', 'A', 'B', 'B', 'C', 'D'],
                }
            )
            result = data["x"].describe(method = "categorical")
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDCAgg_describe_cat_table.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDCAgg_describe_cat_table.html

        The ``cat_stats`` parameter enables grouping by a categorical column and computing
        various aggregations on a numerical one.

        .. code-block:: python

            data["c"].describe(
                method = "cat_stats",
                numcol = "x"
            )

        .. ipython:: python
            :suppress:

            import verticapy as vp
            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                    "c": ['A', 'A', 'A', 'A', 'B', 'B', 'C', 'D'],
                }
            )
            result = data["c"].describe(
                method = "cat_stats",
                numcol = "x",
            )
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDCAgg_describe_stats_table.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDCAgg_describe_stats_table.html

        .. note:: All the calculations are pushed to the database.

        .. seealso::
            | :py:meth:`verticapy.vDataColumn.aggregate` : Aggregations for a specific column.
            | :py:meth:`verticapy.vDataFrame.aggregate` : Aggregations for specific columns.
            | :py:meth:`verticapy.vDataFrame.describe` :
              Summarizes information within the columns.
        """
        assert (method != "cat_stats") or (numcol), ValueError(
            "The parameter 'numcol' must be a vDataFrame column if the method is 'cat_stats'"
        )
        distinct_count, is_numeric, is_date = (
            self.nunique(),
            self.isnum(),
            self.isdate(),
        )
        if (is_date) and method != "categorical":
            result = self.aggregate(["count", "min", "max"])
            index = result.values["index"]
            result = result.values[self._alias]
        elif (method == "cat_stats") and (numcol != ""):
            numcol = self._parent.format_colnames(numcol)
            assert self._parent[numcol].category() in ("float", "int"), TypeError(
                "The column 'numcol' must be numerical"
            )
            cast = "::int" if (self._parent[numcol].isbool()) else ""
            query, cat = [], self.distinct()
            if len(cat) == 1:
                lp, rp = "(", ")"
            else:
                lp, rp = "", ""
            for category in cat:
                tmp_query = f"""
                    SELECT 
                        '{category}' AS 'index', 
                        COUNT({self}) AS count, 
                        100 * COUNT({self}) / {self._parent.shape()[0]} AS percent, 
                        AVG({numcol}{cast}) AS mean, 
                        STDDEV({numcol}{cast}) AS std, 
                        MIN({numcol}{cast}) AS min, 
                        APPROXIMATE_PERCENTILE ({numcol}{cast} 
                            USING PARAMETERS percentile = 0.1) AS 'approx_10%', 
                        APPROXIMATE_PERCENTILE ({numcol}{cast} 
                            USING PARAMETERS percentile = 0.25) AS 'approx_25%', 
                        APPROXIMATE_PERCENTILE ({numcol}{cast} 
                            USING PARAMETERS percentile = 0.5) AS 'approx_50%', 
                        APPROXIMATE_PERCENTILE ({numcol}{cast} 
                            USING PARAMETERS percentile = 0.75) AS 'approx_75%', 
                        APPROXIMATE_PERCENTILE ({numcol}{cast} 
                            USING PARAMETERS percentile = 0.9) AS 'approx_90%', 
                        MAX({numcol}{cast}) AS max 
                   FROM vdf_table"""
                if category in ("None", None):
                    tmp_query += f" WHERE {self} IS NULL"
                else:
                    alias_sql_repr = to_varchar(self.category(), self._alias)
                    tmp_query += f" WHERE {alias_sql_repr} = '{category}'"
                query += [lp + tmp_query + rp]
            values = TableSample.read_sql(
                query=f"""
                    WITH vdf_table AS 
                        (SELECT 
                            * 
                        FROM {self._parent}) 
                        {' UNION ALL '.join(query)}""",
                title=f"Describes the statics of {numcol} partitioned by {self}.",
                sql_push_ext=self._parent._vars["sql_push_ext"],
                symbol=self._parent._vars["symbol"],
            ).values
        elif (
            ((distinct_count < max_cardinality + 1) and (method != "numerical"))
            or not is_numeric
            or (method == "categorical")
        ):
            query = f"""(SELECT 
                            {self} || '', 
                            COUNT(*) 
                        FROM vdf_table 
                        GROUP BY {self} 
                        ORDER BY COUNT(*) DESC 
                        LIMIT {max_cardinality})"""
            if distinct_count > max_cardinality:
                query += f"""
                    UNION ALL 
                    (SELECT 
                        'Others', SUM(count) 
                     FROM 
                        (SELECT 
                            COUNT(*) AS count 
                         FROM vdf_table 
                         WHERE {self} IS NOT NULL 
                         GROUP BY {self} 
                         ORDER BY COUNT(*) DESC 
                         OFFSET {max_cardinality + 1}) VERTICAPY_SUBTABLE) 
                     ORDER BY count DESC"""
            query_result = _executeSQL(
                query=f"""
                    WITH vdf_table AS 
                        (SELECT 
                            /*+LABEL('vDataColumn.describe')*/ * 
                         FROM {self._parent}) {query}""",
                title=f"Computing the descriptive statistics of {self}.",
                method="fetchall",
                sql_push_ext=self._parent._vars["sql_push_ext"],
                symbol=self._parent._vars["symbol"],
            )
            result = [distinct_count, self.count()] + [item[1] for item in query_result]
            index = ["unique", "count"] + [item[0] for item in query_result]
        else:
            result = (
                self._parent.describe(
                    method="numerical", columns=[self._alias], unique=False
                )
                .transpose()
                .values[self._alias]
            )
            result = [distinct_count] + result
            index = [
                "unique",
                "count",
                "mean",
                "std",
                "min",
                "approx_25%",
                "approx_50%",
                "approx_75%",
                "max",
            ]
        if method != "cat_stats":
            values = {
                "index": ["name", "dtype"] + index,
                "value": [self._alias, self.ctype()] + result,
            }
            if ((is_date) and not (method == "categorical")) or (
                method == "is_numeric"
            ):
                self._parent._update_catalog({"index": index, self._alias: result})
        for elem in values:
            for i in range(len(values[elem])):
                if isinstance(values[elem][i], decimal.Decimal):
                    values[elem][i] = float(values[elem][i])
        return TableSample(values)

    # Single Aggregate Functions.

    @save_verticapy_logs
    def aad(self) -> PythonScalar:
        """
        Utilizes the ``aad`` (Average Absolute Deviation) aggregation
        method to analyze the vDataColumn. ``AAD`` measures the average
        absolute deviation of data points from their mean, offering
        valuable insights into data variability and dispersion.
        When we aggregate the vDataColumn using ``aad``, we gain an
        understanding of how data points deviate from the mean on
        average, which is particularly useful for assessing data
        spread and the magnitude of deviations.

        This method is valuable in scenarios where we want to evaluate
        data variability while giving equal weight to all data points,
        regardless of their direction of deviation. Calculating ``aad``
        provides us with information about the overall data consistency
        and can be useful in various analytical and quality assessment
        contexts.

        .. warning::

            To compute aad, VerticaPy needs to execute multiple
            queries. It necessitates, at a minimum, a query that
            includes a subquery to perform this type of aggregation.
            This complexity is the reason why calculating aad
            is typically slower than some other types of aggregations.

        Returns
        -------
        PythonScalar
            aad

        Examples
        --------
        For this example, let's generate a dataset and calculate the
        average absolute deviation of a column:

        .. ipython:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            data["x"].aad()

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataColumn.aad` :
              Standard Deviation for a specific column.
            | :py:meth:`verticapy.vDataFrame.aad` :
              Average Absolute Deviation for particular columns.
        """
        return self.aggregate(["aad"]).values[self._alias][0]

    @save_verticapy_logs
    def avg(self) -> PythonScalar:
        """
        This operation aggregates the vDataFrame using the ``AVG``
        aggregation, which calculates the average value for the
        input column. It provides insights into the central tendency
        of the data and is a fundamental statistical measure often
        used in data analysis and reporting.

        Returns
        -------
        PythonScalar
            average

        Examples
        --------
        For this example, let's generate a dataset and calculate the
        average of a column:

        .. ipython:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            data["x"].avg()

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataFrame.avg` : Aggregations for particular columns.
            | :py:meth:`verticapy.vDataFrame.max` : Maximum for particular columns.
            | :py:meth:`verticapy.vDataFrame.min` : Minimum for particular columns.
        """
        return self.aggregate(["avg"]).values[self._alias][0]

    mean = avg

    @save_verticapy_logs
    def count(self) -> int:
        """
        This operation aggregates the vDataFrame using the
        ``COUNT`` aggregation, providing the count of non-missing
        values for the input column. This is valuable for
        assessing data completeness and quality.

        Returns
        -------
        int
            number of non-Missing elements.

        Examples
        --------
        For this example, let's generate a dataset and calculate the
        count of a column:

        .. ipython:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            data["x"].count()

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataFrame.count` : Count for particular columns.
            | :py:meth:`verticapy.vDataFrame.count_percent` :
            Percentage count for particular columns.
        """
        return self.aggregate(["count"]).values[self._alias][0]

    @save_verticapy_logs
    def kurtosis(self) -> PythonScalar:
        """
        Calculates the kurtosis of the vDataColumn to obtain a measure
        of the data's peakedness or tailness. The kurtosis statistic
        helps us understand the shape of the data distribution.
        It quantifies whether the data has heavy tails or is more peaked
        relative to a normal distribution.

        By aggregating the vDataColumn with kurtosis, we can gain valuable
        insights into the data's distribution characteristics.

        .. warning::

            To compute kurtosis, VerticaPy needs to execute multiple
            queries. It necessitates, at a minimum, a query that
            includes a subquery to perform this type of aggregation.
            This complexity is the reason why calculating kurtosis
            is typically slower than some other types of aggregations.

        Returns
        -------
        PythonScalar
            kurtosis

        Examples
        --------
        For this example, let's generate a dataset and calculate the
        kurtosis of a column:

        .. ipython:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            data["x"].kurtosis()

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataColumn.std` : Standard Deviation for a specific column.
            | :py:meth:`verticapy.vDataFrame.kurtosis` : Kurtosis for particular columns.
        """
        return self.aggregate(["kurtosis"]).values[self._alias][0]

    kurt = kurtosis

    @save_verticapy_logs
    def mad(self) -> PythonScalar:
        """
        Utilizes the ``mad`` (Median Absolute Deviation) aggregation
        method with the vDataFrame. 'MAD' measures the dispersion
        of data points around the median, and it is particularly
        valuable for assessing the robustness of data in the
        presence of outliers. When we aggregate the vDataColumn
        using ``mad``, we gain insights into the variability and
        the degree to which data points deviate from the median.

        This is especially useful for datasets where we want to
        understand the spread of values while being resistant to
        the influence of extreme outliers. Calculating ``mad`` can
        involve robust statistical computations, making it a useful
        tool for outlier-robust analysis and data quality evaluation.

        .. warning::

            To compute mad, VerticaPy needs to execute multiple
            queries. It necessitates, at a minimum, a query that
            includes a subquery to perform this type of aggregation.
            This complexity is the reason why calculating mad
            is typically slower than some other types of aggregations.

        Returns
        -------
        PythonScalar
            mad

        Examples
        --------
        For this example, let's generate a dataset and calculate the
        median absolute deviation of a column:

        .. ipython:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            data["x"].mad()

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataColumn.std` : Standard Deviation for a specific column.
            | :py:meth:`verticapy.vDataFrame.mad` : Mean Absolute Deviation for particular columns.
        """
        return self.aggregate(["mad"]).values[self._alias][0]

    @save_verticapy_logs
    def max(self) -> PythonScalar:
        """
        Aggregates the vDataFrame by applying the 'MAX' aggregation,
        which calculates the maximum value, for the input column.
        This aggregation provides insights into the highest values
        within the dataset, aiding in understanding the data
        distribution.

        Returns
        -------
        PythonScalar
            maximum

        Examples
        --------
        For this example, let's generate a dataset and calculate the
        maximum of a column:

        .. ipython:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            data["x"].max()

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataFrame.max` : Maximum for particular columns.
            | :py:meth:`verticapy.vDataColumn.min` : Minimum for a specific column.
        """
        return self.aggregate(["max"]).values[self._alias][0]

    @save_verticapy_logs
    def median(
        self,
        approx: bool = True,
    ) -> PythonScalar:
        """
        Aggregates the vDataFrame using the ``MEDIAN`` or ``APPROX_MEDIAN``
        aggregation, which calculates the median value for the specified
        columns. The median is a robust measure of central tendency and
        helps in understanding the distribution of data, especially in
        the presence of outliers.

        .. warning::

            When you set `approx` to True, the approximate median is
            computed, which is significantly faster than the exact
            calculation. However, be cautious when setting `approx`
            to False, as it can significantly slow down the performance.

        Parameters
        ----------
        approx: bool, optional
            If set to True, the approximate median is returned.
            By setting this parameter to False, the function's
            performance can drastically decrease.

        Returns
        -------
        PythonScalar
            median

        Examples
        --------
        For this example, let's generate a dataset and calculate the
        median of a column:

        .. ipython:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            data["x"].median(approx = True)

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataColumn.mean` : Mean for a specific column.
            | :py:meth:`verticapy.vDataFrame.median` : Median for particular columns.
        """
        return self.quantile(0.5, approx=approx)

    @save_verticapy_logs
    def min(self) -> PythonScalar:
        """
        Aggregates the vDataFrame by applying the ``MIN`` aggregation,
        which calculates the minimum value, for the input column.
        This aggregation provides insights into the lowest values
        within the dataset, aiding in understanding the data
        distribution.

        Returns
        -------
        PythonScalar
            minimum

        Examples
        --------
        For this example, let's generate a dataset and calculate the
        minimum of a column:

        .. ipython:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            data["x"].min()

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataColumn.max` : Maximum for a specific column.
            | :py:meth:`verticapy.vDataFrame.min` : Minimum for particular columns.
        """
        return self.aggregate(["min"]).values[self._alias][0]

    @save_verticapy_logs
    def product(self) -> PythonScalar:
        """
        Aggregates the vDataColumn by applying the ``product``
        aggregation function. This function computes the
        product of values within the dataset, providing
        insights into the multiplication of data points.

        The ``product`` aggregation can be particularly useful
        when we need to assess cumulative effects or when
        multiplying values is a key aspect of the analysis.
        This operation can be relevant in various domains,
        such as finance, economics, and engineering, where
        understanding the combined impact of values is
        critical for decision-making and modeling.

        .. note::

            Since ``product`` is not a conventional SQL
            aggregation, we employ a unique approach by
            combining the sum of logarithms and the
            exponential function for its computation.
            This non-standard methodology is utilized to
            derive the product of values within the dataset,
            offering a distinctive way to understand the
            multiplicative effects of data points.

        Returns
        -------
        PythonScalar
            product

        Examples
        --------
        For this example, let's generate a dataset and calculate the
        product of a column:

        .. ipython:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            data["x"].product()

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataColumn.aggregate` : Aggregations for a specific column.
            | :py:meth:`verticapy.vDataColumn.quantile` : Quantile Aggregates for a specific column.
        """
        return self.aggregate(func=["prod"]).values[self._alias][0]

    prod = product

    @save_verticapy_logs
    def quantile(self, q: PythonNumber, approx: bool = True) -> PythonScalar:
        """
        Aggregates the vDataColumn using a specified ``quantile``.
        The ``quantile`` function is an indispensable tool for
        comprehending data distribution. By providing a quantile
        value as input, this aggregation method helps us identify
        the data point below which a certain percentage of the data
        falls. This can be pivotal for tasks like analyzing data
        distributions, assessing skewness, and determining essential
        percentiles such as medians or quartiles.

        .. warning::

            It's important to note that the ``quantile`` aggregation
            operates in two distinct modes, allowing flexibility in
            computation. Depending on the ``approx`` parameter, it can
            use either ``APPROXIMATE_QUANTILE`` or ``QUANTILE`` methods
            to derive the final aggregation. The ``APPROXIMATE_QUANTILE``
            method provides faster results by estimating the quantile
            values with an approximation technique, while ``QUANTILE``
            calculates precise quantiles through rigorous computation.
            This choice empowers users to strike a balance between
            computational efficiency and the level of precision
            required for their specific data analysis tasks.

        Parameters
        ----------
        q: PythonNumber
            A float between 0 and  1 that represents the
            quantile.  For  example:  0.25 represents Q1.
        approx: bool, optional
            If set to True,  the approximate quantile is
            returned. By setting this parameter to False,
            the  function's performance can  drastically
            decrease.

        Returns
        -------
        PythonScalar
            quantile (or approximate quantile).

        Examples
        --------
        For this example, let's generate a dataset and calculate the
        approximate median of a column:

        .. ipython:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            data["x"].quantile(q = 0.5, approx = True)

        Let's compute the approximate last decile of a column.

        .. ipython:: python

            data["x"].quantile(q = 0.9, approx = True)

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataColumn.aggregate` : Aggregations for a specific column.
            | :py:meth:`verticapy.vDataFrame.aggregate` : Aggregates for particular columns.
        """
        prefix = "approx_" if approx else ""
        return self.aggregate(func=[f"{prefix}{q * 100}%"]).values[self._alias][0]

    @save_verticapy_logs
    def sem(self) -> PythonScalar:
        """
        Leverages the ``sem`` (Standard Error of the Mean) aggregation
        technique to perform analysis and aggregation on the vDataColumn.
        Standard Error of the Mean is a valuable statistical measure used
        to estimate the precision of the sample mean as an approximation
        of the population mean.

        When we aggregate the vDataColumn using ``sem``, we gain insights
        into the variability or uncertainty associated with the sample
        mean. This measure helps us assess the reliability of the sample
        mean as an estimate of the true population mean.

        It is worth noting that computing the Standard Error of the Mean
        requires statistical calculations and can be particularly useful
        when evaluating the precision of sample statistics or making
        inferences about a larger dataset based on a sample.

        .. warning::

            To compute sem, VerticaPy needs to execute multiple
            queries. It necessitates, at a minimum, a query that
            includes a subquery to perform this type of aggregation.
            This complexity is the reason why calculating sem is
            typically slower than some other types of aggregations.

        Returns
        -------
        PythonScalar
            sem

        Examples
        --------
        For this example, let's generate a dataset and calculate the
        standard error of the mean of a column:

        .. ipython:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            data["x"].sem()

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataColumn.mad` : Mean Absolute Deviation for a specific column.
            | :py:meth:`verticapy.vDataFrame.sem` : Standard Error of Mean for particular columns.
        """
        return self.aggregate(["sem"]).values[self._alias][0]

    @save_verticapy_logs
    def skewness(self) -> PythonScalar:
        """
        Utilizes the ``skewness`` aggregation method to analyze and
        aggregate the vDataColumn. Skewness, a measure of the asymmetry
        in the data's distribution, helps us understand the data's
        deviation from a perfectly symmetrical distribution. When we
        aggregate the vDataFrame using skewness, we gain insights into
        the data's tendency to be skewed to the left or right, or if
        it follows a normal distribution.

        .. warning::

            To compute skewness, VerticaPy needs to execute multiple
            queries. It necessitates, at a minimum, a query that
            includes a subquery to perform this type of aggregation.
            This complexity is the reason why calculating skewness
            is typically slower than some other types of aggregations.

        Returns
        -------
        PythonScalar
            skewness

        Examples
        --------
        For this example, let's generate a dataset and calculate the
        skewness of a column:

        .. ipython:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            data["x"].skewness()

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataColumn.kurtosis` : Kurtosis for a specific column.
            | :py:meth:`verticapy.vDataFrame.skewness` : Skewness for particular columns.
            | :py:meth:`verticapy.vDataFrame.std` : Standard Deviation for particular columns.
        """
        return self.aggregate(["skewness"]).values[self._alias][0]

    skew = skewness

    @save_verticapy_logs
    def std(self) -> PythonScalar:
        """
        Aggregates the vDataFrame using ``STDDEV`` aggregation
        (Standard Deviation), providing insights into the
        spread or variability of data for the input column.
        The standard deviation is a measure of how much individual
        data points deviate from the mean, helping to assess data
        consistency and variation.

        Returns
        -------
        PythonScalar
            std

        Examples
        --------
        For this example, let's generate a dataset and calculate the
        standard deviation of a column:

        .. ipython:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            data["x"].std()

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataColumn.kurtosis` : Kurtosis for a specific column.
            | :py:meth:`verticapy.vDataFrame.skewness` : Skewness for particular columns.
            | :py:meth:`verticapy.vDataFrame.std` : Standard Deviation for particular columns.
        """
        return self.aggregate(["stddev"]).values[self._alias][0]

    stddev = std

    @save_verticapy_logs
    def sum(self) -> PythonScalar:
        """
        Aggregates the vDataFrame using ``SUM`` aggregation, which
        computes the total sum of values for the specified columns,
        providing a cumulative view of numerical data.

        Returns
        -------
        PythonScalar
            sum

        Examples
        --------
        For this example, let's generate a dataset and calculate the
        sum of a column:

        .. ipython:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            data["x"].sum()

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataFrame.sum` : Sum for particular columns.
            | :py:meth:`verticapy.vDataColumn.max` : Maximum for a specific colum.
        """
        return self.aggregate(["sum"]).values[self._alias][0]

    @save_verticapy_logs
    def var(self) -> PythonScalar:
        """
        Aggregates the vDataFrame using ``VAR`` aggregation
        (Variance), providing insights into the spread or
        variability of data for the input column.
        The variance is a measure of how much individual
        data points deviate from the mean, helping to assess
        data consistency and variation.

        Returns
        -------
        PythonScalar
            variance

        Examples
        --------
        For this example, let's generate a dataset and calculate the
        variance of a column:

        .. ipython:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            data["x"].sum()

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataColumn.aggregate` : Aggregations for a specific column.
            | :py:meth:`verticapy.vDataFrame.aggregate` : Aggregates for particular columns.
        """
        return self.aggregate(["variance"]).values[self._alias][0]

    variance = var

    # TOPK.

    @save_verticapy_logs
    def mode(self, dropna: bool = False, n: int = 1) -> PythonScalar:
        """
        This function returns the nth most frequently occurring element
        in the vDataColumn. It's a practical method for identifying the
        element with a specific rank in terms of its occurrence frequency
        within the column. For example, you can use this function to find
        the third, fifth, or any other desired most frequent element.

        .. warning::

            This function first groups the data by a specific column,
            then computes the count of each group, and finally applies
            filtering. It's important to note that this operation can
            be computationally expensive, especially for datasets with
            a large cardinality.

        Parameters
        ----------
        dropna: bool, optional
            If set to True, NULL values are not considered
            during the computation.
        n: int, optional
            Integer  corresponding to the offset. For  example,
            if n = 1, this method returns the mode of the
            vDataColumn.

        Returns
        -------
        PythonScalar
            vDataColumn nth most occurent element.

        Examples
        --------
        For this example, let's generate a dataset and calculate the
        mode of a column:

        .. ipython:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            data["y"].mode()

        Let's now return the second most frequent element:

        .. ipython:: python

            data["y"].mode(n = 2)

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataColumn.mean` : Mean for a specific column.
            | :py:meth:`verticapy.vDataFrame.median` : Median for particular columns.
        """
        if n == 1:
            pre_comp = self._parent._get_catalog_value(self._alias, "top")
            if pre_comp != "VERTICAPY_NOT_PRECOMPUTED":
                if not dropna and (not isinstance(pre_comp, NoneType)):
                    return pre_comp
        assert n >= 1, ValueError("Parameter 'n' must be greater or equal to 1")
        where = f" WHERE {self} IS NOT NULL " if (dropna) else " "
        result = _executeSQL(
            f"""
            SELECT 
                /*+LABEL('vDataColumn.mode')*/ {self} 
            FROM (
                SELECT 
                    {self}, 
                    COUNT(*) AS _verticapy_cnt_ 
                FROM {self._parent}
                {where}GROUP BY {self} 
                ORDER BY _verticapy_cnt_ DESC 
                LIMIT {n}) VERTICAPY_SUBTABLE 
                ORDER BY _verticapy_cnt_ ASC 
                LIMIT 1""",
            title="Computing the mode.",
            method="fetchall",
            sql_push_ext=self._parent._vars["sql_push_ext"],
            symbol=self._parent._vars["symbol"],
        )
        top = None if not result else result[0][0]
        if not dropna:
            n = "" if (n == 1) else str(int(n))
            if isinstance(top, decimal.Decimal):
                top = float(top)
            self._parent._update_catalog({"index": [f"top{n}"], self._alias: [top]})
        return top

    @save_verticapy_logs
    def value_counts(self, k: int = 30) -> TableSample:
        """
        This function returns the k most frequently occurring
        elements in a column, along with information about how
        often they occur. Additionally, it provides various
        statistical details to give you a comprehensive view of
        the data distribution.

        Parameters
        ----------
        k: int, optional
            Number of most occurent elements to return.

        Returns
        -------
        TableSample
            result.

        Examples
        --------
        For this example, we will use the following dataset:

        .. code-block:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )

        Now, let's calculate the values and counts for a specific
        column.

        .. code-block:: python

            data["x"].value_counts(k = 6)

        .. ipython:: python
            :suppress:

            import verticapy as vp
            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            result = data["x"].value_counts(k = 6)
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_value_counts_table.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_value_counts_table.html

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataColumn.nunique` : Cardinality for a specific column.
            | :py:meth:`verticapy.vDataFrame.duplicated` : Duplicated values for particular columns.
        """
        return self.describe(method="categorical", max_cardinality=k)

    @save_verticapy_logs
    def topk(self, k: int = -1, dropna: bool = True) -> TableSample:
        """
        This function returns the k most frequently occurring elements
        in a column, along with their distribution expressed as
        percentages. It's a useful tool for understanding the
        composition of your data and identifying the most prominent
        elements.

        Parameters
        ----------
        k: int, optional
            Number of most occurent elements to return.
        dropna: bool, optional
            If  set to True, NULL  values  are not
            considered during the computation.

        Returns
        -------
        TableSample
            result.

        Examples
        --------
        For this example, we will use the following dataset:

        .. code-block:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )

        Now, let's calculate the top k values for a specific
        column.

        .. code-block:: python

            data["x"].topk()

        .. ipython:: python
            :suppress:

            import verticapy as vp
            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            result = data["x"].topk()
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_topk_table.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFAgg_topk_table.html

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataColumn.nunique` : Cardinality for a specific column.
            | :py:meth:`verticapy.vDataFrame.nunique` : Cardinality for particular columns.
        """
        limit, where, topk_cat = "", "", ""
        if k >= 1:
            limit = f"LIMIT {k}"
            topk_cat = k
        if dropna:
            where = f" WHERE {self} IS NOT NULL"
        alias_sql_repr = to_varchar(self.category(), self._alias)
        result = _executeSQL(
            query=f"""
            SELECT 
                /*+LABEL('vDataColumn.topk')*/
                {alias_sql_repr} AS {self},
                COUNT(*) AS _verticapy_cnt_,
                100 * COUNT(*) / {self._parent.shape()[0]} AS percent
            FROM {self._parent}
            {where} 
            GROUP BY {alias_sql_repr} 
            ORDER BY _verticapy_cnt_ DESC
            {limit}""",
            title=f"Computing the top{topk_cat} categories of {self}.",
            method="fetchall",
            sql_push_ext=self._parent._vars["sql_push_ext"],
            symbol=self._parent._vars["symbol"],
        )
        values = {
            "index": [item[0] for item in result],
            "count": [int(item[1]) for item in result],
            "percent": [float(round(item[2], 3)) for item in result],
        }
        return TableSample(values)

    # Distincts.

    def distinct(self, **kwargs) -> list:
        """
        This function returns the distinct categories or unique values
        within a vDataColumn. It's a valuable method for exploring
        the unique elements within a column, which can be particularly
        useful when working with categorical data.

        Returns
        -------
        list
            Distinct categories of the vDataColumn.

        Examples
        --------
        For this example, let's generate a dataset and compute
        all the distinct elements of a column:

        .. ipython:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            data["y"].distinct()

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the ``aggregate`` method.

        .. seealso::
            | :py:meth:`verticapy.vDataColumn.aggregate` : Aggregations for a specific column.
            | :py:meth:`verticapy.vDataFrame.aggregate` : Aggregates for particular columns.
        """
        alias_sql_repr = to_varchar(self.category(), self._alias)
        if "agg" not in kwargs:
            query = f"""
                SELECT 
                    /*+LABEL('vDataColumn.distinct')*/ 
                    {alias_sql_repr} AS {self} 
                FROM {self._parent} 
                WHERE {self} IS NOT NULL 
                GROUP BY {self} 
                ORDER BY {self}"""
        else:
            query = f"""
                SELECT 
                    /*+LABEL('vDataColumn.distinct')*/ {self} 
                FROM 
                    (SELECT 
                        {alias_sql_repr} AS {self}, 
                        {kwargs['agg']} AS verticapy_agg 
                     FROM {self._parent} 
                     WHERE {self} IS NOT NULL 
                     GROUP BY 1) x 
                ORDER BY verticapy_agg DESC"""
        query_result = _executeSQL(
            query=query,
            title=f"Computing the distinct categories of {self}.",
            method="fetchall",
            sql_push_ext=self._parent._vars["sql_push_ext"],
            symbol=self._parent._vars["symbol"],
        )
        return [item for sublist in query_result for item in sublist]

    @save_verticapy_logs
    def nunique(self, approx: bool = True) -> int:
        """
        When aggregating the vDataFrame using `nunique` (cardinality),
        VerticaPy employs the COUNT DISTINCT function to determine the
        number of unique values in particular columns. It also offers
        the option to use APPROXIMATE_COUNT_DISTINCT, a more efficient
        approximation method for calculating cardinality.

        .. hint::

            This flexibility allows you to optimize the computation
            based on your specific requirements, keeping in mind
            that using APPROXIMATE_COUNT_DISTINCT can significantly
            improve performance when cardinality estimation is sufficient
            for your analysis.

        .. important::

            To calculate the exact cardinality of a column, you should
            set the parameter `approx` to False. This will ensure that
            the cardinality is computed accurately rather than using the
            approximate method.

        Parameters
        ----------
        approx: bool, optional
            If  set  to  True,  the  approximate  cardinality
            is   returned.  By  setting  this  parameter   to
            False, the function's performance can drastically
            decrease.

        Returns
        -------
        int
            vDataColumn cardinality (or approximate cardinality).

        Examples
        --------
        For this example, let's generate a dataset and calculate the
        cardinality of a column:

        .. ipython:: python

            import verticapy as vp

            data = vp.vDataFrame(
                {
                    "x": [1, 2, 4, 9, 10, 15, 20, 22],
                    "y": [1, 2, 1, 2, 1, 1, 2, 1],
                    "z": [10, 12, 2, 1, 9, 8, 1, 3],
                }
            )
            data["y"].nunique()

        .. note:: All the calculations are pushed to the database.

        .. hint:: For more precise control, please refer to the `aggregate` method.

        .. seealso::
            | :py:meth:`verticapy.vDataColumn.aggregate` : Aggregations for a specific column.
            | :py:meth:`verticapy.vDataFrame.aggregate` : Aggregates for particular columns.
        """
        if approx:
            return self.aggregate(func=["approx_unique"]).values[self._alias][0]
        else:
            return self.aggregate(func=["unique"]).values[self._alias][0]
