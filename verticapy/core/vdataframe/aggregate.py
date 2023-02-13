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
#
#
# Modules
#
# Standard Python Modules
import warnings
import multiprocessing
from typing import Union, Literal

# Other modules
from tqdm.auto import tqdm

# VerticaPy Modules
from verticapy.core.tablesample import tablesample
from verticapy._utils._collect import save_verticapy_logs
from verticapy.errors import (
    EmptyParameter,
    FunctionError,
)
from verticapy._version import vertica_version
from verticapy._utils._sql import _executeSQL
from verticapy.sql._utils._format import (
    format_magic,
    quote_ident,
)
from verticapy.core._utils._map import verticapy_agg_name
from verticapy.connect.connect import current_cursor
from verticapy._config.config import OPTIONS
from verticapy.sql.read import to_tablesample


class vDFAGG:
    @save_verticapy_logs
    def groupby(
        self,
        columns: Union[str, list],
        expr: Union[str, list] = [],
        rollup: Union[bool, list] = False,
        having: str = "",
    ):
        """
    Aggregates the vDataFrame by grouping the elements.

    Parameters
    ----------
    columns: str / list
        List of the vColumns used to group the elements or a customized expression. 
        If rollup is set to True, this can be a list of tuples.
    expr: str / list, optional
        List of the different aggregations in pure SQL. Aliases can be used.
        For example, 'SUM(column)' or 'AVG(column) AS my_new_alias' are correct 
        whereas 'AVG' is incorrect. Aliases are recommended to keep the track of 
        the features and to prevent ambiguous names. For example, the MODE 
        function does not exist, but can be replicated by using the 'analytic' 
        method and then grouping the result.
    rollup: bool / list of bools, optional
        If set to True, the rollup operator is used.
        If set to a list of bools, the rollup operator is used on the matching
        indexes and the length of 'rollup' must match the length of 'columns.'
        For example, for columns = ['col1', ('col2', 'col3'), 'col4'] and
        rollup = [False, True, True], the rollup operator is used on the set
        ('col2', 'col3') and on 'col4'.
    having: str, optional
        Expression used to filter the result.

    Returns
    -------
    vDataFrame
        object result of the grouping.

    See Also
    --------
    vDataFrame.append   : Merges the vDataFrame with another relation.
    vDataFrame.analytic : Adds a new vColumn to the vDataFrame by using an advanced 
        analytical function on a specific vColumn.
    vDataFrame.join     : Joins the vDataFrame with another relation.
    vDataFrame.sort     : Sorts the vDataFrame.
        """
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(expr, str):
            expr = [expr]
        assert not (isinstance(rollup, list)) or len(rollup) == len(
            columns
        ), ParameterError(
            "If parameter 'rollup' is of type list, it should have "
            "the same length as the 'columns' parameter."
        )
        columns_to_select = []
        if rollup == True:
            rollup_expr = "ROLLUP(" if rollup == True else ""
        else:
            rollup_expr = ""
        for idx, elem in enumerate(columns):
            if isinstance(elem, tuple) and rollup:
                if rollup == True:
                    rollup_expr += "("
                elif rollup[idx] == True:
                    rollup_expr += "ROLLUP("
                elif not (isinstance(rollup[idx], bool)):
                    raise ParameterError(
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
                    if not (isinstance(rollup, bool)) and (rollup[idx] == True):
                        rollup_expr += "ROLLUP(" + colname + ")"
                    else:
                        rollup_expr += colname
                    columns_to_select += [colname]
                else:
                    if not (isinstance(rollup, bool)) and (rollup[idx] == True):
                        rollup_expr += "ROLLUP(" + str(elem) + ")"
                    else:
                        rollup_expr += str(elem)
                    columns_to_select += [elem]
                rollup_expr += ", "
            else:
                raise ParameterError(
                    "Parameter 'columns' must be a string; list of strings "
                    "or tuples (only when rollup is set to True)."
                )
        rollup_expr = rollup_expr[:-2]
        if rollup == True:
            rollup_expr += ")"
        if having:
            having = f" HAVING {having}"
        columns_str = ", ".join(
            [str(elem) for elem in columns_to_select] + [str(elem) for elem in expr]
        )
        if not (rollup):
            rollup_expr_str = ", ".join(
                [
                    str(i + 1)
                    for i in range(len([str(elem) for elem in columns_to_select]))
                ],
            )
        else:
            rollup_expr_str = rollup_expr
        relation = f"""
            (SELECT 
                {columns_str} 
            FROM {self.__genSQL__()} 
            GROUP BY {rollup_expr_str}{having}) VERTICAPY_SUBTABLE"""
        if not (rollup):
            rollup_expr_str = ", ".join([str(c) for c in columns_to_select])
        else:
            rollup_expr_str = rollup_expr
        return self.__vDataFrameSQL__(
            relation,
            "groupby",
            f"[Groupby]: The columns were grouped by {rollup_expr_str}",
        )

    @save_verticapy_logs
    def duplicated(
        self, columns: Union[str, list] = [], count: bool = False, limit: int = 30
    ):
        """
    Returns the duplicated values.

    Parameters
    ----------
    columns: str / list, optional
        List of the vColumns names. If empty, all vColumns will be selected.
    count: bool, optional
        If set to True, the method will also return the count of each duplicates.
    limit: int, optional
        The limited number of elements to be displayed.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.drop_duplicates : Filters the duplicated values.
        """
        if not (columns):
            columns = self.get_columns()
        elif isinstance(columns, str):
            columns = [columns]
        columns = self.format_colnames(columns)
        columns = ", ".join(columns)
        main_table = f"""
            (SELECT 
                *, 
                ROW_NUMBER() OVER (PARTITION BY {columns}) AS duplicated_index 
             FROM {self.__genSQL__()}) duplicated_index_table 
             WHERE duplicated_index > 1"""
        if count:
            total = _executeSQL(
                query=f"""
                    SELECT 
                        /*+LABEL('vDataframe.duplicated')*/ COUNT(*) 
                    FROM {main_table}""",
                title="Computing the number of duplicates.",
                method="fetchfirstelem",
                sql_push_ext=self._VERTICAPY_VARIABLES_["sql_push_ext"],
                symbol=self._VERTICAPY_VARIABLES_["symbol"],
            )
            return total
        result = to_tablesample(
            query=f"""
                SELECT 
                    {columns},
                    MAX(duplicated_index) AS occurrence 
                FROM {main_table} 
                GROUP BY {columns} 
                ORDER BY occurrence DESC LIMIT {limit}""",
            sql_push_ext=self._VERTICAPY_VARIABLES_["sql_push_ext"],
            symbol=self._VERTICAPY_VARIABLES_["symbol"],
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
            sql_push_ext=self._VERTICAPY_VARIABLES_["sql_push_ext"],
            symbol=self._VERTICAPY_VARIABLES_["symbol"],
        )
        return result

    @save_verticapy_logs
    def aggregate(
        self,
        func: Union[str, list],
        columns: Union[str, list] = [],
        ncols_block: int = 20,
        processes: int = 1,
    ):
        """
    Aggregates the vDataFrame using the input functions.

    Parameters
    ----------
    func: str / list
        List of the different aggregations.
            aad            : average absolute deviation
            approx_median  : approximate median
            approx_q%      : approximate q quantile 
                             (ex: approx_50% for the approximate median)
            approx_unique  : approximative cardinality
            count          : number of non-missing elements
            cvar           : conditional value at risk
            dtype          : virtual column type
            iqr            : interquartile range
            kurtosis       : kurtosis
            jb             : Jarque-Bera index 
            mad            : median absolute deviation
            max            : maximum
            mean           : average
            median         : median
            min            : minimum
            mode           : most occurent element
            percent        : percent of non-missing elements 
            q%             : q quantile (ex: 50% for the median)
                             Use the 'approx_q%' (approximate quantile) 
                             aggregation to get better performances.
            prod           : product
            range          : difference between the max and the min
            sem            : standard error of the mean
            skewness       : skewness
            sum            : sum
            std            : standard deviation
            topk           : kth most occurent element (ex: top1 for the mode)
            topk_percent   : kth most occurent element density
            unique         : cardinality (count distinct)
            var            : variance
                Other aggregations will work if supported by your version of 
                the database.
    columns: str / list, optional
        List of the vColumn's names. If empty, depending on the aggregations,
        all or only numerical vColumns will be used.
    ncols_block: int, optional
        Number of columns used per query. Setting this parameter divides
        what would otherwise be one large query into many smaller queries called
        "blocks." The size of each block is determined by the ncols_block parameter.
    processes: int, optional
        Number of child processes to create. Setting this with the ncols_block parameter
        lets you parallelize a single query into many smaller queries, where each child 
        process creates its own connection to the database and sends one query. This can 
        improve query performance, but consumes more resources. If processes is set to 1, 
        the queries are sent iteratively from a single process.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.analytic : Adds a new vColumn to the vDataFrame by using an advanced 
        analytical function on a specific vColumn.
        """
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(func, str):
            func = [func]
        if not (columns):
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
                if (len(fun.lower()) >= 8) and fun[0:7] == "approx_":
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

            if OPTIONS["tqdm"]:
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
                pre_comp = self.__get_catalog_value__(column, fun)

                if pre_comp != "VERTICAPY_NOT_PRECOMPUTED":
                    nb_precomputed += 1
                    if pre_comp == None or pre_comp != pre_comp:
                        expr = "NULL"
                    elif isinstance(pre_comp, (int, float)):
                        expr = pre_comp
                    else:
                        pre_comp_str = str(pre_comp).replace("'", "''")
                        expr = f"'{pre_comp_str}'"

                elif ("_percent" in fun.lower()) and (fun.lower()[0:3] == "top"):
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

                elif (len(fun.lower()) > 2) and (fun.lower()[0:3] == "top"):
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
                        or (std == None)
                        or (avg == None)
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
                        or (std == None)
                        or (avg == None)
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
                        if (len(fun.lower()) >= 8) and fun[0:7] == "approx_":
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
                    expr = (
                        f"ROUND(COUNT({column}) / { self.shape()[0]} * 100, 3)::float"
                    )

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
                except:
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
                        FROM {self.__genSQL__()} 
                        LIMIT 1""",
                    title="Computing the different aggregations.",
                    method="fetchrow",
                    sql_push_ext=self._VERTICAPY_VARIABLES_["sql_push_ext"],
                    symbol=self._VERTICAPY_VARIABLES_["symbol"],
                )
            result = [item for item in res]
            try:
                result = [float(item) for item in result]
            except:
                pass
            values = {"index": func}
            i = 0
            for column in columns:
                values[column] = result[i : i + len(func)]
                i += len(func)

        except:

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
                         FROM {self.__genSQL__()}) {query}"""
                if nb_precomputed == len(func) * len(columns):
                    result = _executeSQL(query, print_time_sql=False, method="fetchall")
                else:
                    result = _executeSQL(
                        query,
                        title="Computing the different aggregations using UNION ALL.",
                        method="fetchall",
                        sql_push_ext=self._VERTICAPY_VARIABLES_["sql_push_ext"],
                        symbol=self._VERTICAPY_VARIABLES_["symbol"],
                    )

                for idx, elem in enumerate(result):
                    values[columns[idx]] = [item for item in elem]

            except:

                try:

                    for i, elem in enumerate(agg):
                        pre_comp_val = []
                        for fun in func:
                            pre_comp = self.__get_catalog_value__(columns[i], fun)
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
                                        FROM {self.__genSQL__()}""",
                                    title=(
                                        "Computing the different aggregations one "
                                        "vColumn at a time."
                                    ),
                                    sql_push_ext=self._VERTICAPY_VARIABLES_[
                                        "sql_push_ext"
                                    ],
                                    symbol=self._VERTICAPY_VARIABLES_["symbol"],
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
                except:

                    for i, elem in enumerate(agg):
                        values[columns[i]] = []
                        for j, agg_fun in enumerate(elem):
                            pre_comp = self.__get_catalog_value__(columns[i], func[j])
                            if pre_comp == "VERTICAPY_NOT_PRECOMPUTED":
                                result = _executeSQL(
                                    query=f"""
                                        SELECT 
                                            /*+LABEL('vDataframe.aggregate')*/ 
                                            {agg_fun} 
                                        FROM {self.__genSQL__()}""",
                                    title=(
                                        "Computing the different aggregations one "
                                        "vColumn & one agg at a time."
                                    ),
                                    method="fetchfirstelem",
                                    sql_push_ext=self._VERTICAPY_VARIABLES_[
                                        "sql_push_ext"
                                    ],
                                    symbol=self._VERTICAPY_VARIABLES_["symbol"],
                                )
                            else:
                                result = pre_comp
                            values[columns[i]] += [result]

        for elem in values:
            for idx in range(len(values[elem])):
                if isinstance(values[elem][idx], str) and "top" not in elem:
                    try:
                        values[elem][idx] = float(values[elem][idx])
                    except:
                        pass

        self.__update_catalog__(values)
        return tablesample(values=values).decimal_to_float().transpose()

    agg = aggregate

    @save_verticapy_logs
    def aad(
        self, columns: list = [], **agg_kwds,
    ):
        """
    Aggregates the vDataFrame using 'aad' (Average Absolute Deviation).

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.
    **agg_kwds
        Any optional parameter to pass to the Aggregate function.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        return self.aggregate(func=["aad"], columns=columns, **agg_kwds,)

    @save_verticapy_logs
    def all(
        self, columns: list, **agg_kwds,
    ):
        """
    Aggregates the vDataFrame using 'bool_and'.

    Parameters
    ----------
    columns: list
        List of the vColumns names.
    **agg_kwds
        Any optional parameter to pass to the Aggregate function.


    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        return self.aggregate(func=["bool_and"], columns=columns, **agg_kwds,)

    @save_verticapy_logs
    def any(
        self, columns: list, **agg_kwds,
    ):
        """
    Aggregates the vDataFrame using 'bool_or'.

    Parameters
    ----------
    columns: list
        List of the vColumns names.
    **agg_kwds
        Any optional parameter to pass to the Aggregate function.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        return self.aggregate(func=["bool_or"], columns=columns, **agg_kwds,)

    @save_verticapy_logs
    def avg(
        self, columns: list = [], **agg_kwds,
    ):
        """
    Aggregates the vDataFrame using 'avg' (Average).

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.
    **agg_kwds
        Any optional parameter to pass to the Aggregate function.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        return self.aggregate(func=["avg"], columns=columns, **agg_kwds,)

    mean = avg

    @save_verticapy_logs
    def count(
        self, columns: list = [], **agg_kwds,
    ):
        """
    Aggregates the vDataFrame using a list of 'count' (Number of non-missing 
    values).

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all vColumns will be used.
    **agg_kwds
        Any optional parameter to pass to the Aggregate function.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        return self.aggregate(func=["count"], columns=columns, **agg_kwds,)

    @save_verticapy_logs
    def count_percent(
        self,
        columns: Union[str, list] = [],
        sort_result: bool = True,
        desc: bool = True,
        **agg_kwds,
    ):
        """
    Aggregates the vDataFrame using a list of 'count' (the number of non-missing 
    values) and percent (the percent of non-missing values).

    Parameters
    ----------
    columns: str / list, optional
        List of vColumn names. If empty, all vColumns will be used.
    sort_result: bool, optional
        If set to True, the result will be sorted.
    desc: bool, optional
        If set to True and 'sort_result' is set to True, the result will be 
        sorted in descending order.
    **agg_kwds
        Any optional parameter to pass to the Aggregate function.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        result = self.aggregate(func=["count", "percent"], columns=columns, **agg_kwds,)
        if sort_result:
            result.sort("count", desc)
        return result

    @save_verticapy_logs
    def describe(
        self,
        method: Literal[
            "numerical", "categorical", "statistics", "length", "range", "all", "auto",
        ] = "auto",
        columns: Union[str, list] = [],
        unique: bool = False,
        ncols_block: int = 20,
        processes: int = 1,
    ):
        """
    Aggregates the vDataFrame using multiple statistical aggregations: min, 
    max, median, unique... depending on the types of the vColumns.

    Parameters
    ----------
    method: str, optional
        The describe method.
            all         : Aggregates all selected vDataFrame vColumns different 
                methods depending on the vColumn type (numerical dtype: numerical; 
                timestamp dtype: range; categorical dtype: length)
            auto        : Sets the method to 'numerical' if at least one vColumn 
                of the vDataFrame is numerical, 'categorical' otherwise.
            categorical : Uses only categorical aggregations.
            length      : Aggregates the vDataFrame using numerical aggregation 
                on the length of all selected vColumns.
            numerical   : Uses only numerical descriptive statistics which are 
                 computed in a faster way than the 'aggregate' method.
            range       : Aggregates the vDataFrame using multiple statistical
                aggregations - min, max, range...
            statistics  : Aggregates the vDataFrame using multiple statistical 
                aggregations - kurtosis, skewness, min, max...
    columns: str / list, optional
        List of the vColumns names. If empty, the vColumns will be selected
        depending on the parameter 'method'.
    unique: bool, optional
        If set to True, the cardinality of each element will be computed.
    ncols_block: int, optional
        Number of columns used per query. Setting this parameter divides
        what would otherwise be one large query into many smaller queries called
        "blocks." The size of each block is determined by the ncols_block parmeter.
    processes: int, optional
        Number of child processes to create. Setting this with the ncols_block parameter
        lets you parallelize a single query into many smaller queries, where each child 
        process creates its own connection to the database and sends one query. This can 
        improve query performance, but consumes more resources. If processes is set to 1, 
        the queries are sent iteratively from a single process.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        if isinstance(columns, str):
            columns = [columns]
        if method == "auto":
            method = "numerical" if (self.numcol()) else "categorical"
        columns = self.format_colnames(columns)
        for i in range(len(columns)):
            columns[i] = quote_ident(columns[i])
        dtype, percent = {}, {}

        if method == "numerical":

            if not (columns):
                columns = self.numcol()
            else:
                for column in columns:
                    assert self[column].isnum(), TypeError(
                        f"vColumn {column} must be numerical to run describe"
                        " using parameter method = 'numerical'"
                    )
            assert columns, EmptyParameter(
                "No Numerical Columns found to run describe using parameter"
                " method = 'numerical'."
            )
            if ncols_block < len(columns) and processes <= 1:
                if OPTIONS["tqdm"]:
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
                            pre_comp = self.__get_catalog_value__(column, fun)
                            if pre_comp == "VERTICAPY_NOT_PRECOMPUTED":
                                col_to_compute += [column]
                                break
                    elif OPTIONS["print_info"]:
                        warning_message = (
                            f"The vColumn {column} is not numerical, it was ignored."
                            "\nTo get statistical information about all different "
                            "variables, please use the parameter method = 'categorical'."
                        )
                        warnings.warn(warning_message, Warning)
                for column in columns:
                    if column not in col_to_compute:
                        values["index"] += [column.replace('"', "")]
                        for fun in idx[1:]:
                            values[fun] += [self.__get_catalog_value__(column, fun)]
                if col_to_compute:
                    cols_to_compute_str = [
                        col if not (self[col].isbool()) else f"{col}::int"
                        for col in col_to_compute
                    ]
                    cols_to_compute_str = ", ".join(cols_to_compute_str)
                    query_result = _executeSQL(
                        query=f"""
                            SELECT 
                                /*+LABEL('vDataframe.describe')*/ 
                                SUMMARIZE_NUMCOL({cols_to_compute_str}) OVER () 
                            FROM {self.__genSQL__()}""",
                        title=(
                            "Computing the descriptive statistics of all numerical "
                            "columns using SUMMARIZE_NUMCOL."
                        ),
                        method="fetchall",
                    )

                    # Formatting - to have the same columns' order than the input one.
                    for i, key in enumerate(idx):
                        values[key] += [elem[i] for elem in query_result]
                    tb = tablesample(values).transpose()
                    vals = {"index": tb["index"]}
                    for col in columns:
                        vals[col] = tb[col]
                    values = tablesample(vals).transpose().values

            except:

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
                func, columns=columns, ncols_block=ncols_block, processes=processes,
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

            if not (columns):
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

            if not (columns):
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
            if not (columns):
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

        self.__update_catalog__(tablesample(values).transpose().values)
        values["index"] = [quote_ident(elem) for elem in values["index"]]
        result = tablesample(values, percent=percent, dtype=dtype).decimal_to_float()
        if method == "all":
            result = result.transpose()

        return result

    @save_verticapy_logs
    def kurtosis(
        self, columns: list = [], **agg_kwds,
    ):
        """
    Aggregates the vDataFrame using 'kurtosis'.

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.
    **agg_kwds
        Any optional parameter to pass to the Aggregate function.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        return self.aggregate(func=["kurtosis"], columns=columns, **agg_kwds,)

    kurt = kurtosis

    @save_verticapy_logs
    def mad(
        self, columns: list = [], **agg_kwds,
    ):
        """
    Aggregates the vDataFrame using 'mad' (Median Absolute Deviation).

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.
    **agg_kwds
        Any optional parameter to pass to the Aggregate function.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        return self.aggregate(func=["mad"], columns=columns, **agg_kwds,)

    @save_verticapy_logs
    def max(
        self, columns: list = [], **agg_kwds,
    ):
        """
    Aggregates the vDataFrame using 'max' (Maximum).

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.
    **agg_kwds
        Any optional parameter to pass to the Aggregate function.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        return self.aggregate(func=["max"], columns=columns, **agg_kwds,)

    @save_verticapy_logs
    def median(
        self, columns: list = [], approx: bool = True, **agg_kwds,
    ):
        """
    Aggregates the vDataFrame using 'median'.

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.
    approx: bool, optional
        If set to True, the approximate median is returned. By setting this 
        parameter to False, the function's performance can drastically decrease.
    **agg_kwds
        Any optional parameter to pass to the Aggregate function.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        return self.quantile(0.5, columns=columns, approx=approx, **agg_kwds,)

    @save_verticapy_logs
    def min(
        self, columns: list = [], **agg_kwds,
    ):
        """
    Aggregates the vDataFrame using 'min' (Minimum).

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.
    **agg_kwds
        Any optional parameter to pass to the Aggregate function.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        return self.aggregate(func=["min"], columns=columns, **agg_kwds,)

    @save_verticapy_logs
    def nunique(
        self, columns: list = [], approx: bool = True, **agg_kwds,
    ):
        """
    Aggregates the vDataFrame using 'unique' (cardinality).

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all vColumns will be used.
    approx: bool, optional
        If set to True, the approximate cardinality is returned. By setting 
        this parameter to False, the function's performance can drastically 
        decrease.
    **agg_kwds
        Any optional parameter to pass to the Aggregate function.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        func = ["approx_unique"] if approx else ["unique"]
        return self.aggregate(func=func, columns=columns, **agg_kwds,)

    @save_verticapy_logs
    def product(
        self, columns: list = [], **agg_kwds,
    ):
        """
    Aggregates the vDataFrame using 'product'.

    Parameters
    ----------
    columns: list, optional
        List of the vColumn names. If empty, all numerical vColumns will be used.
    **agg_kwds
        Any optional parameter to pass to the Aggregate function.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        return self.aggregate(func=["prod"], columns=columns, **agg_kwds,)

    prod = product

    @save_verticapy_logs
    def quantile(
        self,
        q: Union[int, float, list],
        columns: list = [],
        approx: bool = True,
        **agg_kwds,
    ):
        """
    Aggregates the vDataFrame using a list of 'quantiles'.

    Parameters
    ----------
    q: int / float / list
        List of the different quantiles. They must be numbers between 0 and 1.
        For example [0.25, 0.75] will return Q1 and Q3.
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.
    approx: bool, optional
        If set to True, the approximate quantile is returned. By setting this 
        parameter to False, the function's performance can drastically decrease.
    **agg_kwds
        Any optional parameter to pass to the Aggregate function.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        if isinstance(q, (int, float)):
            q = [q]
        prefix = "approx_" if approx else ""
        return self.aggregate(
            func=[verticapy_agg_name(prefix + f"{float(item) * 100}%") for item in q],
            columns=columns,
            **agg_kwds,
        )

    @save_verticapy_logs
    def sem(
        self, columns: list = [], **agg_kwds,
    ):
        """
    Aggregates the vDataFrame using 'sem' (Standard Error of the Mean).

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.
    **agg_kwds
        Any optional parameter to pass to the Aggregate function.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        return self.aggregate(func=["sem"], columns=columns, **agg_kwds,)

    @save_verticapy_logs
    def skewness(
        self, columns: list = [], **agg_kwds,
    ):
        """
    Aggregates the vDataFrame using 'skewness'.

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.
    **agg_kwds
        Any optional parameter to pass to the Aggregate function.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        return self.aggregate(func=["skewness"], columns=columns, **agg_kwds,)

    skew = skewness

    @save_verticapy_logs
    def std(
        self, columns: list = [], **agg_kwds,
    ):
        """
    Aggregates the vDataFrame using 'std' (Standard Deviation).

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.
    **agg_kwds
        Any optional parameter to pass to the Aggregate function.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        return self.aggregate(func=["stddev"], columns=columns, **agg_kwds,)

    stddev = std

    @save_verticapy_logs
    def sum(
        self, columns: list = [], **agg_kwds,
    ):
        """
    Aggregates the vDataFrame using 'sum'.

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.
    **agg_kwds
        Any optional parameter to pass to the Aggregate function.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        return self.aggregate(func=["sum"], columns=columns, **agg_kwds,)

    @save_verticapy_logs
    def var(
        self, columns: list = [], **agg_kwds,
    ):
        """
    Aggregates the vDataFrame using 'variance'.

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.
    **agg_kwds
        Any optional parameter to pass to the Aggregate function.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        return self.aggregate(func=["variance"], columns=columns, **agg_kwds,)

    variance = var


#
# Multiprocessing
#

#
# Functions used to send multiple queries at the same time.
#

# Aggregate
def aggregate_parallel_block(vdf, func: list, columns: list, ncols_block: int, i: int):
    return vdf.aggregate(
        func=func, columns=columns[i : i + ncols_block], ncols_block=ncols_block
    )


# Describe
def describe_parallel_block(
    vdf, method: str, columns: list, unique: bool, ncols_block: int, i: int,
):
    return vdf.describe(
        method=method,
        columns=columns[i : i + ncols_block],
        unique=unique,
        ncols_block=ncols_block,
    )
