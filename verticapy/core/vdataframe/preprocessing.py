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
# Standard Python Modules
import warnings, datetime, math, re
from itertools import combinations_with_replacement
from typing import Union, Literal

# VerticaPy Modules
from verticapy._utils._collect import save_verticapy_logs
from verticapy.errors import EmptyParameter, ParameterError, QueryError
from verticapy.sql.flex import compute_vmap_keys, isvmap
from verticapy._utils._cast import to_category, to_varchar
from verticapy._version import vertica_version
from verticapy.core.str_sql import str_sql
from verticapy.sql._utils._format import quote_ident
from verticapy.core._utils._merge import gen_coalesce, group_similar_names
from verticapy._config.config import OPTIONS
from verticapy.sql.dtypes import get_data_types
from verticapy.sql.drop import drop
from verticapy._utils._sql import _executeSQL
from verticapy._utils._gen import gen_tmp_name

class vDFPREP:
    def __setattr__(self, attr, val):
        from verticapy.core.vcolumn import vColumn

        if isinstance(val, (str, str_sql, int, float)) and not isinstance(val, vColumn):
            val = str(val)
            if self.is_colname_in(attr):
                self[attr].apply(func=val)
            else:
                self.eval(name=attr, expr=val)
        elif isinstance(val, vColumn) and not (val.init):
            final_trans, n = val.init_transf, len(val.transformations)
            for i in range(1, n):
                final_trans = val.transformations[i][0].replace("{}", final_trans)
            self.eval(name=attr, expr=final_trans)
        else:
            self.__dict__[attr] = val

    def __setitem__(self, index, val):
        setattr(self, index, val)

    @save_verticapy_logs
    def eval(self, name: str, expr: Union[str, str_sql]):
        """
    Evaluates a customized expression.

    Parameters
    ----------
    name: str
        Name of the new vColumn.
    expr: str
        Expression in pure SQL to use to compute the new feature.
        For example, 'CASE WHEN "column" > 3 THEN 2 ELSE NULL END' and
        'POWER("column", 2)' will work.

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame.analytic : Adds a new vColumn to the vDataFrame by using an advanced 
        analytical function on a specific vColumn.
        """
        from verticapy.core.vcolumn import vColumn

        if isinstance(expr, str_sql):
            expr = str(expr)
        name = quote_ident(name.replace('"', "_"))
        if self.is_colname_in(name):
            raise NameError(
                f"A vColumn has already the alias {name}.\n"
                "By changing the parameter 'name', you'll "
                "be able to solve this issue."
            )
        try:
            query = f"SELECT {expr} AS {name} FROM {self.__genSQL__()} LIMIT 0"
            ctype = get_data_types(query, name[1:-1].replace("'", "''"),)
        except:
            raise QueryError(
                f"The expression '{expr}' seems to be incorrect.\nBy "
                "turning on the SQL with the 'set_option' function, "
                "you'll print the SQL code generation and probably "
                "see why the evaluation didn't work."
            )
        if not (ctype):
            ctype = "undefined"
        elif (ctype.lower()[0:12] in ("long varbina", "long varchar")) and (
            self._VERTICAPY_VARIABLES_["isflex"]
            or isvmap(expr=f"({query}) VERTICAPY_SUBTABLE", column=name,)
        ):
            category = "vmap"
            ctype = "VMAP(" + "(".join(ctype.split("(")[1:]) if "(" in ctype else "VMAP"
        else:
            category = to_category(ctype=ctype)
        all_cols, max_floor = self.get_columns(), 0
        for column in all_cols:
            column_str = column.replace('"', "")
            if (quote_ident(column) in expr) or (
                re.search(re.compile(f"\\b{column_str}\\b"), expr)
            ):
                max_floor = max(len(self[column].transformations), max_floor)
        transformations = [
            (
                "___VERTICAPY_UNDEFINED___",
                "___VERTICAPY_UNDEFINED___",
                "___VERTICAPY_UNDEFINED___",
            )
            for i in range(max_floor)
        ] + [(expr, ctype, category)]
        new_vColumn = vColumn(name, parent=self, transformations=transformations)
        setattr(self, name, new_vColumn)
        setattr(self, name.replace('"', ""), new_vColumn)
        new_vColumn.init = False
        new_vColumn.init_transf = name
        self._VERTICAPY_VARIABLES_["columns"] += [name]
        self.__add_to_history__(
            f"[Eval]: A new vColumn {name} was added to the vDataFrame."
        )
        return self

    @save_verticapy_logs
    def add_duplicates(self, weight: Union[int, str], use_gcd: bool = True):
        """
    Duplicates the vDataFrame using the input weight.

    Parameters
    ----------
    weight: str / integer
        vColumn or integer representing the weight.
    use_gcd: bool
        If set to True, uses the GCD (Greatest Common Divisor) to reduce all 
        common weights to avoid unnecessary duplicates.

    Returns
    -------
    vDataFrame
        the output vDataFrame
        """
        if isinstance(weight, str):
            weight = self.format_colnames(weight)
            assert self[weight].category() == "int", TypeError(
                "The weight vColumn category must be "
                f"'integer', found {self[weight].category()}."
            )
            L = sorted(self[weight].distinct())
            gcd, max_value, n = L[0], L[-1], len(L)
            assert gcd >= 0, ValueError(
                "The weight vColumn must only include positive integers."
            )
            if use_gcd:
                if gcd != 1:
                    for i in range(1, n):
                        if gcd != 1:
                            gcd = math.gcd(gcd, L[i])
                        else:
                            break
            else:
                gcd = 1
            columns = self.get_columns(exclude_columns=[weight])
            vdf = self.search(self[weight] != 0, usecols=columns)
            for i in range(2, int(max_value / gcd) + 1):
                vdf = vdf.append(
                    self.search((self[weight] / gcd) >= i, usecols=columns)
                )
        else:
            assert weight >= 2 and isinstance(weight, int), ValueError(
                "The weight must be an integer greater or equal to 2."
            )
            vdf = self.copy()
            for i in range(2, weight + 1):
                vdf = vdf.append(self)
        return vdf

    @save_verticapy_logs
    def append(
        self,
        input_relation: Union[str, str_sql],
        expr1: Union[str, list] = [],
        expr2: Union[str, list] = [],
        union_all: bool = True,
    ):
        """
    Merges the vDataFrame with another one or an input relation and returns 
    a new vDataFrame.

    Parameters
    ----------
    input_relation: str / vDataFrame
        Relation to use to do the merging.
    expr1: str / list, optional
        List of pure-SQL expressions from the current vDataFrame to use during merging.
        For example, 'CASE WHEN "column" > 3 THEN 2 ELSE NULL END' and 'POWER("column", 2)' 
        will work. If empty, all vDataFrame vColumns will be used. Aliases are 
        recommended to avoid auto-naming.
    expr2: str / list, optional
        List of pure-SQL expressions from the input relation to use during the merging.
        For example, 'CASE WHEN "column" > 3 THEN 2 ELSE NULL END' and 'POWER("column", 2)' 
        will work. If empty, all input relation columns will be used. Aliases are 
        recommended to avoid auto-naming.
    union_all: bool, optional
        If set to True, the vDataFrame will be merged with the input relation using an
        'UNION ALL' instead of an 'UNION'.

    Returns
    -------
    vDataFrame
       vDataFrame of the Union

    See Also
    --------
    vDataFrame.groupby : Aggregates the vDataFrame.
    vDataFrame.join    : Joins the vDataFrame with another relation.
    vDataFrame.sort    : Sorts the vDataFrame.
        """
        from verticapy.core.vdataframe.vdataframe import vDataFrame

        if isinstance(expr1, str):
            expr1 = [expr1]
        if isinstance(expr2, str):
            expr2 = [expr2]
        first_relation = self.__genSQL__()
        if isinstance(input_relation, str):
            second_relation = input_relation
        elif isinstance(input_relation, vDataFrame):
            second_relation = input_relation.__genSQL__()
        columns = ", ".join(self.get_columns()) if not (expr1) else ", ".join(expr1)
        columns2 = columns if not (expr2) else ", ".join(expr2)
        union = "UNION" if not (union_all) else "UNION ALL"
        table = f"""
            (SELECT 
                {columns} 
             FROM {first_relation}) 
             {union} 
            (SELECT 
                {columns2} 
             FROM {second_relation})"""
        return self.__vDataFrameSQL__(
            f"({table}) append_table",
            self._VERTICAPY_VARIABLES_["input_relation"],
            "[Append]: Union of two relations",
        )

    @save_verticapy_logs
    def cdt(
        self,
        columns: Union[str, list] = [],
        max_cardinality: int = 20,
        nbins: int = 10,
        tcdt: bool = True,
        drop_transf_cols: bool = True,
    ):
        """
    Returns the complete disjunctive table of the vDataFrame.
    Numerical features are transformed to categorical using
    the 'discretize' method. Applying PCA on TCDT leads to MCA 
    (Multiple correspondence analysis).

    \u26A0 Warning : This method can become computationally expensive when
                     used with categorical variables with many categories.

    Parameters
    ----------
    columns: str / list, optional
        List of the vColumns names.
    max_cardinality: int, optional
        For any categorical variable, keeps the most frequent categories and 
        merges the less frequent categories into a new unique category.
    nbins: int, optional
        Number of bins used for the discretization (must be > 1).
    tcdt: bool, optional
        If set to True, returns the transformed complete disjunctive table 
        (TCDT). 
    drop_transf_cols: bool, optional
        If set to True, drops the columns used during the transformation.

    Returns
    -------
    vDataFrame
        the CDT relation.
        """
        if isinstance(columns, str):
            columns = [columns]
        if columns:
            columns = self.format_colnames(columns)
        else:
            columns = self.get_columns()
        vdf = self.copy()
        columns_to_drop = []
        for elem in columns:
            if vdf[elem].isbool():
                vdf[elem].astype("int")
            elif vdf[elem].isnum():
                vdf[elem].discretize(nbins=nbins)
                columns_to_drop += [elem]
            elif vdf[elem].isdate():
                vdf[elem].drop()
            else:
                vdf[elem].discretize(method="topk", k=max_cardinality)
                columns_to_drop += [elem]
        new_columns = vdf.get_columns()
        vdf.one_hot_encode(
            columns=columns,
            max_cardinality=max(max_cardinality, nbins) + 2,
            drop_first=False,
        )
        new_columns = vdf.get_columns(exclude_columns=new_columns)
        if drop_transf_cols:
            vdf.drop(columns=columns_to_drop)
        if tcdt:
            for elem in new_columns:
                sum_cat = vdf[elem].sum()
                vdf[elem].apply(f"{{}} / {sum_cat} - 1")
        return vdf

    @save_verticapy_logs
    def fillna(self, val: dict = {}, method: dict = {}, numeric_only: bool = False):
        """
    Fills the vColumns missing elements using specific rules.

    Parameters
    ----------
    val: dict, optional
        Dictionary of values. The dictionary must be similar to the following:
        {"column1": val1 ..., "columnk": valk}. Each key of the dictionary must
        be a vColumn. The missing values of the input vColumns will be replaced
        by the input value.
    method: dict, optional
        Method to use to impute the missing values.
            auto    : Mean for the numerical and Mode for the categorical vColumns.
            mean    : Average.
            median  : Median.
            mode    : Mode (most occurent element).
            0ifnull : 0 when the vColumn is null, 1 otherwise.
                More Methods are available on the vDataFrame[].fillna method.
    numeric_only: bool, optional
        If parameters 'val' and 'method' are empty and 'numeric_only' is set
        to True then all numerical vColumns will be imputed by their average.
        If set to False, all categorical vColumns will be also imputed by their
        mode.

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame[].fillna : Fills the vColumn missing values. This method is more 
        complete than the vDataFrame.fillna method by allowing more parameters.
        """
        print_info = OPTIONS["print_info"]
        OPTIONS["print_info"] = False
        try:
            if not (val) and not (method):
                cols = self.get_columns()
                for column in cols:
                    if numeric_only:
                        if self[column].isnum():
                            self[column].fillna(method="auto")
                    else:
                        self[column].fillna(method="auto")
            else:
                for column in val:
                    self[self.format_colnames(column)].fillna(val=val[column])
                for column in method:
                    self[self.format_colnames(column)].fillna(method=method[column],)
            return self
        finally:
            OPTIONS["print_info"] = print_info

    @save_verticapy_logs
    def flat_vmap(
        self,
        vmap_col: Union[str, list] = [],
        limit: int = 100,
        exclude_columns: list = [],
    ):
        """
    Flatten the selected VMap. A new vDataFrame is returned.
    
    \u26A0 Warning : This function might have a long runtime and can make your
                     vDataFrame less performant. It makes many calls to the
                     MAPLOOKUP function, which can be slow if your VMap is
                     large.

    Parameters
    ----------
    vmap_col: str / list, optional
        List of VMap columns to flatten.
    limit: int, optional
        Maximum number of keys to consider for each VMap. Only the most occurent 
        keys are used.
    exclude_columns: list, optional
        List of VMap columns to exclude.

    Returns
    -------
    vDataFrame
        object with the flattened VMaps.
        """
        if not (vmap_col):
            vmap_col = []
            all_cols = self.get_columns()
            for col in all_cols:
                if self[col].isvmap():
                    vmap_col += [col]
        if isinstance(vmap_col, str):
            vmap_col = [vmap_col]
        exclude_columns_final, vmap_col_final = (
            [quote_ident(col).lower() for col in exclude_columns],
            [],
        )
        for col in vmap_col:
            if quote_ident(col).lower() not in exclude_columns_final:
                vmap_col_final += [col]
        if not (vmap_col):
            raise EmptyParameter("No VMAP was detected.")
        maplookup = []
        for vmap in vmap_col_final:
            keys = compute_vmap_keys(expr=self, vmap_col=vmap, limit=limit)
            keys = [k[0] for k in keys]
            for k in keys:
                column = quote_ident(vmap)
                alias = quote_ident(vmap.replace('"', "") + "." + k.replace('"', ""))
                maplookup += [f"MAPLOOKUP({column}, '{k}') AS {alias}"]
        return self.select(self.get_columns() + maplookup)

    @save_verticapy_logs
    def interpolate(
        self,
        ts: str,
        rule: Union[str, datetime.timedelta],
        method: dict = {},
        by: Union[str, list] = [],
    ):
        """
    Computes a regular time interval vDataFrame by interpolating the missing 
    values using different techniques.

    Parameters
    ----------
    ts: str
        TS (Time Series) vColumn to use to order the data. The vColumn type 
        must be date like (date, datetime, timestamp...)
    rule: str / time
        Interval used to create the time slices. The final interpolation is 
        divided by these intervals. For example, specifying '5 minutes' 
        creates records separated by time intervals of '5 minutes' 
    method: dict, optional
        Dictionary, with the following format, of interpolation methods:
        {"column1": "interpolation1" ..., "columnk": "interpolationk"}
        Interpolation methods must be one of the following:
            bfill  : Interpolates with the final value of the time slice.
            ffill  : Interpolates with the first value of the time slice.
            linear : Linear interpolation.
    by: str / list, optional
        vColumns used in the partition.

    Returns
    -------
    vDataFrame
        object result of the interpolation.

    See Also
    --------
    vDataFrame[].fillna  : Fills the vColumn missing values.
    vDataFrame[].slice   : Slices the vColumn.
        """
        if isinstance(by, str):
            by = [by]
        method, ts, by = self.format_colnames(method, ts, by)
        all_elements = []
        for column in method:
            assert method[column] in (
                "bfill",
                "backfill",
                "pad",
                "ffill",
                "linear",
            ), ParameterError(
                "Each element of the 'method' dictionary must be "
                "in bfill|backfill|pad|ffill|linear"
            )
            if method[column] in ("bfill", "backfill"):
                func, interp = "TS_LAST_VALUE", "const"
            elif method[column] in ("pad", "ffill"):
                func, interp = "TS_FIRST_VALUE", "const"
            else:
                func, interp = "TS_FIRST_VALUE", "linear"
            all_elements += [f"{func}({column}, '{interp}') AS {column}"]
        table = f"SELECT {{}} FROM {self.__genSQL__()}"
        tmp_query = [f"slice_time AS {quote_ident(ts)}"]
        tmp_query += [quote_ident(column) for column in by]
        tmp_query += all_elements
        table = table.format(", ".join(tmp_query))
        partition = ""
        if by:
            partition = ", ".join([quote_ident(column) for column in by])
            partition = f"PARTITION BY {partition} "
        table += f""" 
            TIMESERIES slice_time AS '{rule}' 
            OVER ({partition}ORDER BY {quote_ident(ts)}::timestamp)"""
        return self.__vDataFrameSQL__(
            f"({table}) interpolate",
            "interpolate",
            "[interpolate]: The data was resampled",
        )

    asfreq = interpolate

    @save_verticapy_logs
    def join(
        self,
        input_relation,
        on: Union[tuple, dict, list] = {},
        on_interpolate: dict = {},
        how: Literal[
            "left", "right", "cross", "full", "natural", "self", "inner", ""
        ] = "natural",
        expr1: Union[str, list] = ["*"],
        expr2: Union[str, list] = ["*"],
    ):
        """
    Joins the vDataFrame with another one or an input relation.

    \u26A0 Warning : Joins can make the vDataFrame structure heavier. It is 
                     recommended to always check the current structure 
                     using the 'current_relation' method and to save it using the 
                     'to_db' method with the parameters 'inplace = True' and 
                     'relation_type = table'

    Parameters
    ----------
    input_relation: str/vDataFrame
        Relation to use to do the merging.
    on: tuple / dict / list, optional
        If it is a list then:
        List of 3-tuples. Each tuple must include (key1, key2, operator)—where
        key1 is the key of the vDataFrame, key2 is the key of the input relation,
        and operator can be one of the following:
                     '=' : exact match
                     '<' : key1  < key2
                     '>' : key1  > key2
                    '<=' : key1 <= key2
                    '>=' : key1 >= key2
                 'llike' : key1 LIKE '%' || key2 || '%'
                 'rlike' : key2 LIKE '%' || key1 || '%'
           'linterpolate': key1 INTERPOLATE key2
           'rinterpolate': key2 INTERPOLATE key1
        Some operators need 5-tuples: (key1, key2, operator, operator2, x)—where
        operator2 is a simple operator (=, >, <, <=, >=), x is a float or an integer, 
        and operator is one of the following:
                 'jaro' : JARO(key1, key2) operator2 x
                'jarow' : JARO_WINCKLER(key1, key2) operator2 x
                  'lev' : LEVENSHTEIN(key1, key2) operator2 x
        
        If it is a dictionary then:
        This parameter must include all the different keys. It must be similar 
        to the following:
        {"relationA_key1": "relationB_key1" ..., "relationA_keyk": "relationB_keyk"}
        where relationA is the current vDataFrame and relationB is the input relation
        or the input vDataFrame.
    on_interpolate: dict, optional
        Dictionary of all different keys. Used to join two event series together 
        using some ordered attribute, event series joins let you compare values from 
        two series directly, rather than having to normalize the series to the same 
        measurement interval. The dict must be similar to the following:
        {"relationA_key1": "relationB_key1" ..., "relationA_keyk": "relationB_keyk"}
        where relationA is the current vDataFrame and relationB is the input relation
        or the input vDataFrame.
    how: str, optional
        Join Type.
            left    : Left Join.
            right   : Right Join.
            cross   : Cross Join.
            full    : Full Outer Join.
            natural : Natural Join.
            inner   : Inner Join.
    expr1: str / list, optional
        List of the different columns in pure SQL to select from the current 
        vDataFrame, optionally as aliases. Aliases are recommended to avoid 
        ambiguous names. For example: 'column' or 'column AS my_new_alias'. 
    expr2: str / list, optional
        List of the different columns in pure SQL to select from the input 
        relation optionally as aliases. Aliases are recommended to avoid 
        ambiguous names. For example: 'column' or 'column AS my_new_alias'.

    Returns
    -------
    vDataFrame
        object result of the join.

    See Also
    --------
    vDataFrame.append  : Merges the vDataFrame with another relation.
    vDataFrame.groupby : Aggregates the vDataFrame.
    vDataFrame.sort    : Sorts the vDataFrame.
        """
        from verticapy.core.vdataframe.vdataframe import vDataFrame

        if isinstance(expr1, str):
            expr1 = [expr1]
        if isinstance(expr2, str):
            expr2 = [expr2]
        if isinstance(on, tuple):
            on = [on]
        # Giving the right alias to the right relation
        def create_final_relation(relation: str, alias: str):
            if (
                ("SELECT" in relation.upper())
                and ("FROM" in relation.upper())
                and ("(" in relation)
                and (")" in relation)
            ):
                return f"(SELECT * FROM {relation}) AS {alias}"
            else:
                return f"{relation} AS {alias}"

        # List with the operators
        if str(how).lower() == "natural" and (on or on_interpolate):
            raise ParameterError(
                "Natural Joins cannot be computed if any of "
                "the parameters 'on' or 'on_interpolate' are "
                "defined."
            )
        on_list = []
        if isinstance(on, dict):
            on_list += [(key, on[key], "=") for key in on]
        else:
            on_list += [elem for elem in on]
        on_list += [(key, on[key], "linterpolate") for key in on_interpolate]
        # Checks
        self.format_colnames([elem[0] for elem in on_list])
        if isinstance(input_relation, vDataFrame):
            input_relation.format_colnames([elem[1] for elem in on_list])
            relation = input_relation.__genSQL__()
        else:
            relation = input_relation
        # Relations
        first_relation = create_final_relation(self.__genSQL__(), alias="x")
        second_relation = create_final_relation(relation, alias="y")
        # ON
        on_join = []
        all_operators = [
            "=",
            ">",
            ">=",
            "<",
            "<=",
            "llike",
            "rlike",
            "linterpolate",
            "rinterpolate",
            "jaro",
            "jarow",
            "lev",
        ]
        simple_operators = all_operators[0:5]
        for elem in on_list:
            key1, key2, op = quote_ident(elem[0]), quote_ident(elem[1]), elem[2]
            if op not in all_operators:
                raise ValueError(
                    f"Incorrect operator: '{op}'.\nCorrect values: {', '.join(simple_operators)}."
                )
            if op in ("=", ">", ">=", "<", "<="):
                on_join += [f"x.{key1} {op} y.{key2}"]
            elif op == "llike":
                on_join += [f"x.{key1} LIKE '%' || y.{key2} || '%'"]
            elif op == "rlike":
                on_join += [f"y.{key2} LIKE '%' || x.{key1} || '%'"]
            elif op == "linterpolate":
                on_join += [f"x.{key1} INTERPOLATE PREVIOUS VALUE y.{key2}"]
            elif op == "rinterpolate":
                on_join += [f"y.{key2} INTERPOLATE PREVIOUS VALUE x.{key1}"]
            elif op in ("jaro", "jarow", "lev"):
                if op in ("jaro", "jarow"):
                    vertica_version(condition=[12, 0, 2])
                else:
                    vertica_version(condition=[10, 1, 0])
                op2, x = elem[3], elem[4]
                if op2 not in simple_operators:
                    raise ValueError(
                        f"Incorrect operator: '{op2}'.\nCorrect values: {', '.join(simple_operators)}."
                    )
                map_to_fun = {
                    "jaro": "JARO_DISTANCE",
                    "jarow": "JARO_WINKLER_DISTANCE",
                    "lev": "EDIT_DISTANCE",
                }
                fun = map_to_fun[op]
                on_join += [f"{fun}(x.{key1}, y.{key2}) {op2} {x}"]
        # Final
        on_join = " ON " + " AND ".join(on_join) if on_join else ""
        expr = [f"x.{key}" for key in expr1] + [f"y.{key}" for key in expr2]
        expr = "*" if not (expr) else ", ".join(expr)
        if how:
            how = " " + how.upper() + " "
        table = (
            f"SELECT {expr} FROM {first_relation}{how}JOIN {second_relation} {on_join}"
        )
        return self.__vDataFrameSQL__(
            f"({table}) VERTICAPY_SUBTABLE",
            "join",
            "[Join]: Two relations were joined together",
        )

    @save_verticapy_logs
    def merge_similar_names(self, skip_word: Union[str, list]):
        """
    Merges columns with similar names. The function generates a COALESCE 
    statement that merges the columns into a single column that excludes 
    the input words. Note that the order of the variables in the COALESCE 
    statement is based on the order of the 'get_columns' method.
    
    Parameters
    ---------- 
    skip_word: str / list, optional
        List of words to exclude from the provided column names. 
        For example, if two columns are named 'age.information.phone' 
        and 'age.phone' AND skip_word is set to ['.information'], then 
        the two columns will be merged together with the following 
        COALESCE statement:
        COALESCE("age.phone", "age.information.phone") AS "age.phone"

    Returns
    -------
    vDataFrame
        An object containing the merged element.
        """
        if isinstance(skip_word, str):
            skip_word = [skip_word]
        columns = self.get_columns()
        group_dict = group_similar_names(columns, skip_word=skip_word)
        sql = f"""
            (SELECT 
                {gen_coalesce(group_dict)} 
            FROM {self.__genSQL__()}) VERTICAPY_SUBTABLE"""
        return self.__vDataFrameSQL__(
            sql,
            "merge_similar_names",
            "[merge_similar_names]: The columns were merged.",
        )

    @save_verticapy_logs
    def narrow(
        self,
        index: Union[str, list],
        columns: Union[str, list] = [],
        col_name: str = "column",
        val_name: str = "value",
    ):
        """
    Returns the Narrow Table of the vDataFrame using the input vColumns.

    Parameters
    ----------
    index: str / list
        Index(es) used to identify the Row.
    columns: str / list, optional
        List of the vColumns names. If empty, all vColumns except the index(es)
        will be used.
    col_name: str, optional
        Alias of the vColumn representing the different input vColumns names as 
        categories.
    val_name: str, optional
        Alias of the vColumn representing the different input vColumns values.

    Returns
    -------
    vDataFrame
        the narrow table object.

    See Also
    --------
    vDataFrame.pivot : Returns the pivot table of the vDataFrame.
        """
        index, columns = self.format_colnames(index, columns)
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(index, str):
            index = [index]
        if not (columns):
            columns = self.numcol()
        for idx in index:
            if idx in columns:
                columns.remove(idx)
        query = []
        all_are_num, all_are_date = True, True
        for column in columns:
            if not (self[column].isnum()):
                all_are_num = False
            if not (self[column].isdate()):
                all_are_date = False
        for column in columns:
            conv = ""
            if not (all_are_num) and not (all_are_num):
                conv = "::varchar"
            elif self[column].category() == "int":
                conv = "::int"
            column_str = column.replace("'", "''")[1:-1]
            query += [
                f"""
                (SELECT 
                    {', '.join(index)}, 
                    '{column_str}' AS {col_name}, 
                    {column}{conv} AS {val_name} 
                FROM {self.__genSQL__()})"""
            ]
        query = " UNION ALL ".join(query)
        query = f"({query}) VERTICAPY_SUBTABLE"
        return self.__vDataFrameSQL__(
            query, "narrow", f"[Narrow]: Narrow table using index = {index}",
        )

    melt = narrow

    @save_verticapy_logs
    def normalize(
        self,
        columns: Union[str, list] = [],
        method: Literal["zscore", "robust_zscore", "minmax"] = "zscore",
    ):
        """
    Normalizes the input vColumns using the input method.

    Parameters
    ----------
    columns: str / list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.
    method: str, optional
        Method to use to normalize.
            zscore        : Normalization using the Z-Score (avg and std).
                (x - avg) / std
            robust_zscore : Normalization using the Robust Z-Score (median and mad).
                (x - median) / (1.4826 * mad)
            minmax        : Normalization using the MinMax (min and max).
                (x - min) / (max - min)

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame.outliers    : Computes the vDataFrame Global Outliers.
    vDataFrame[].normalize : Normalizes the vColumn. This method is more complete 
        than the vDataFrame.normalize method by allowing more parameters.
        """
        if isinstance(columns, str):
            columns = [columns]
        no_cols = True if not (columns) else False
        columns = self.numcol() if not (columns) else self.format_colnames(columns)
        for column in columns:
            if self[column].isnum() and not (self[column].isbool()):
                self[column].normalize(method=method)
            elif (no_cols) and (self[column].isbool()):
                pass
            elif OPTIONS["print_info"]:
                warning_message = (
                    f"The vColumn {column} was skipped.\n"
                    "Normalize only accept numerical data types."
                )
                warnings.warn(warning_message, Warning)
        return self

    @save_verticapy_logs
    def one_hot_encode(
        self,
        columns: Union[str, list] = [],
        max_cardinality: int = 12,
        prefix_sep: str = "_",
        drop_first: bool = True,
        use_numbers_as_suffix: bool = False,
    ):
        """
    Encodes the vColumns using the One Hot Encoding algorithm.

    Parameters
    ----------
    columns: str / list, optional
        List of the vColumns to use to train the One Hot Encoding model. If empty, 
        only the vColumns having a cardinality lesser than 'max_cardinality' will 
        be used.
    max_cardinality: int, optional
        Cardinality threshold to use to determine if the vColumn will be taken into
        account during the encoding. This parameter is used only if the parameter 
        'columns' is empty.
    prefix_sep: str, optional
        Prefix delimitor of the dummies names.
    drop_first: bool, optional
        Drops the first dummy to avoid the creation of correlated features.
    use_numbers_as_suffix: bool, optional
        Uses numbers as suffix instead of the vColumns categories.

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame[].decode       : Encodes the vColumn using a user defined Encoding.
    vDataFrame[].discretize   : Discretizes the vColumn.
    vDataFrame[].get_dummies  : Computes the vColumns result of One Hot Encoding.
    vDataFrame[].label_encode : Encodes the vColumn using the Label Encoding.
    vDataFrame[].mean_encode  : Encodes the vColumn using the Mean Encoding of a response.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns = self.format_colnames(columns)
        if not (columns):
            columns = self.get_columns()
        cols_hand = True if (columns) else False
        for column in columns:
            if self[column].nunique(True) < max_cardinality:
                self[column].get_dummies(
                    "", prefix_sep, drop_first, use_numbers_as_suffix
                )
            elif cols_hand and OPTIONS["print_info"]:
                warning_message = (
                    f"The vColumn '{column}' was ignored because of "
                    "its high cardinality.\nIncrease the parameter "
                    "'max_cardinality' to solve this issue or use "
                    "directly the vColumn get_dummies method."
                )
                warnings.warn(warning_message, Warning)
        return self

    get_dummies = one_hot_encode

    @save_verticapy_logs
    def pivot(
        self,
        index: str,
        columns: str,
        values: str,
        aggr: str = "sum",
        prefix: str = "",
    ):
        """
    Returns the Pivot of the vDataFrame using the input aggregation.

    Parameters
    ----------
    index: str
        vColumn to use to group the elements.
    columns: str
        The vColumn used to compute the different categories, which then act 
        as the columns in the pivot table.
    values: str
        The vColumn whose values populate the new vDataFrame.
    aggr: str, optional
        Aggregation to use on 'values'. To use complex aggregations, 
        you must use braces: {}. For example, to aggregate using the 
        aggregation: x -> MAX(x) - MIN(x), write "MAX({}) - MIN({})".
    prefix: str, optional
        The prefix for the pivot table's column names.

    Returns
    -------
    vDataFrame
        the pivot table object.

    See Also
    --------
    vDataFrame.narrow      : Returns the Narrow table of the vDataFrame.
    vDataFrame.pivot_table : Draws the pivot table of one or two columns based on an 
        aggregation.
        """
        index, columns, values = self.format_colnames(index, columns, values)
        aggr = aggr.upper()
        if "{}" not in aggr:
            aggr += "({})"
        new_cols = self[columns].distinct()
        new_cols_trans = []
        for elem in new_cols:
            if elem == None:
                new_cols_trans += [
                    aggr.replace(
                        "{}",
                        f"(CASE WHEN {columns} IS NULL THEN {values} ELSE NULL END)",
                    )
                    + f"AS '{prefix}NULL'"
                ]
            else:
                new_cols_trans += [
                    aggr.replace(
                        "{}",
                        f"(CASE WHEN {columns} = '{elem}' THEN {values} ELSE NULL END)",
                    )
                    + f"AS '{prefix}{elem}'"
                ]
        return self.__vDataFrameSQL__(
            f"""
            (SELECT 
                {index},
                {", ".join(new_cols_trans)}
             FROM {self.__genSQL__()}
             GROUP BY 1) VERTICAPY_SUBTABLE""",
            "pivot",
            (
                f"[Pivot]: Pivot table using index = {index} & "
                f"columns = {columns} & values = {values}"
            ),
        )

    @save_verticapy_logs
    def polynomial_comb(self, columns: Union[str, list] = [], r: int = 2):
        """
    Returns a vDataFrame containing different product combination of the 
    input vColumns. This function is ideal for bivariate analysis.

    Parameters
    ----------
    columns: str / list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.
    r: int, optional
        Degree of the polynomial.

    Returns
    -------
    vDataFrame
        the Polynomial object.
        """
        if isinstance(columns, str):
            columns = [columns]
        if not (columns):
            numcol = self.numcol()
        else:
            numcol = self.format_colnames(columns)
        vdf = self.copy()
        all_comb = combinations_with_replacement(numcol, r=r)
        for elem in all_comb:
            name = "_".join(elem)
            vdf.eval(name.replace('"', ""), expr=" * ".join(elem))
        return vdf

    @save_verticapy_logs
    def sessionize(
        self,
        ts: str,
        by: Union[str, list] = [],
        session_threshold: str = "30 minutes",
        name: str = "session_id",
    ):
        """
    Adds a new vColumn to the vDataFrame which will correspond to sessions 
    (user activity during a specific time). A session ends when ts - lag(ts) 
    is greater than a specific threshold.

    Parameters
    ----------
    ts: str
        vColumn used as timeline. It will be to use to order the data. It can be
        a numerical or type date like (date, datetime, timestamp...) vColumn.
    by: str / list, optional
        vColumns used in the partition.
    session_threshold: str, optional
        This parameter is the threshold which will determine the end of the 
        session. For example, if it is set to '10 minutes' the session ends
        after 10 minutes of inactivity.
    name: str, optional
        The session name.

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame.analytic : Adds a new vColumn to the vDataFrame by using an advanced 
        analytical function on a specific vColumn.
        """
        if isinstance(by, str):
            by = [by]
        by, ts = self.format_colnames(by, ts)
        partition = ""
        if by:
            partition = f"PARTITION BY {', '.join(by)}"
        expr = f"""CONDITIONAL_TRUE_EVENT(
                    {ts}::timestamp - LAG({ts}::timestamp) 
                  > '{session_threshold}') 
                  OVER ({partition} ORDER BY {ts})"""
        return self.eval(name=name, expr=expr)

    @save_verticapy_logs
    def sort(self, columns: Union[str, dict, list]):
        """
    Sorts the vDataFrame using the input vColumns.

    Parameters
    ----------
    columns: str / dict / list
        List of the vColumns to use to sort the data using asc order or
        dictionary of all sorting methods. For example, to sort by "column1"
        ASC and "column2" DESC, write {"column1": "asc", "column2": "desc"}

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame.append  : Merges the vDataFrame with another relation.
    vDataFrame.groupby : Aggregates the vDataFrame.
    vDataFrame.join    : Joins the vDataFrame with another relation.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns = self.format_colnames(columns)
        max_pos = 0
        columns_tmp = [elem for elem in self._VERTICAPY_VARIABLES_["columns"]]
        for column in columns_tmp:
            max_pos = max(max_pos, len(self[column].transformations) - 1)
        self._VERTICAPY_VARIABLES_["order_by"][max_pos] = self.__get_sort_syntax__(
            columns
        )
        return self


class vDCPREP:
    def __setattr__(self, attr, val):
        self.__dict__[attr] = val

    @save_verticapy_logs
    def clip(
        self,
        lower: Union[int, float, datetime.datetime, datetime.date] = None,
        upper: Union[int, float, datetime.datetime, datetime.date] = None,
    ):
        """
    Clips the vColumn by transforming the values lesser than the lower bound to 
    the lower bound itself and the values higher than the upper bound to the upper 
    bound itself.

    Parameters
    ----------
    lower: int / float / date, optional
        Lower bound.
    upper: int / float / date, optional
        Upper bound.

    Returns
    -------
    vDataFrame
        self.parent

    See Also
    --------
    vDataFrame[].fill_outliers : Fills the vColumn outliers using the input method.
        """
        assert (lower != None) or (upper != None), ParameterError(
            "At least 'lower' or 'upper' must have a numerical value"
        )
        lower_when = (
            f"WHEN {{}} < {lower} THEN {lower} "
            if (isinstance(lower, (float, int)))
            else ""
        )
        upper_when = (
            f"WHEN {{}} > {upper} THEN {upper} "
            if (isinstance(upper, (float, int)))
            else ""
        )
        func = f"(CASE {lower_when}{upper_when}ELSE {{}} END)"
        self.apply(func=func)
        return self.parent

    @save_verticapy_logs
    def cut(
        self,
        breaks: list,
        labels: list = [],
        include_lowest: bool = True,
        right: bool = True,
    ):
        """
    Discretizes the vColumn using the input list. 

    Parameters
    ----------
    breaks: list
        List of values used to cut the vColumn.
    labels: list, optional
        Labels used to name the new categories. If empty, names will be generated.
    include_lowest: bool, optional
        If set to True, the lowest element of the list will be included.
    right: bool, optional
        How the intervals should be closed. If set to True, the intervals will be
        closed on the right.

    Returns
    -------
    vDataFrame
        self.parent

    See Also
    --------
    vDataFrame[].apply : Applies a function to the input vColumn.
        """
        assert self.isnum() or self.isdate(), TypeError(
            "cut only works on numerical / date-like vColumns."
        )
        assert len(breaks) >= 2, ParameterError(
            "Length of parameter 'breaks' must be greater or equal to 2."
        )
        assert len(breaks) == len(labels) + 1 or not (labels), ParameterError(
            "Length of parameter breaks must be equal to the length of parameter "
            "'labels' + 1 or parameter 'labels' must be empty."
        )
        conditions, column = [], self.alias
        for idx in range(len(breaks) - 1):
            first_elem, second_elem = breaks[idx], breaks[idx + 1]
            if right:
                op1, op2, close_l, close_r = "<", "<=", "]", "]"
            else:
                op1, op2, close_l, close_r = "<=", "<", "[", "["
            if idx == 0 and include_lowest:
                op1, close_l = "<=", "["
            elif idx == 0:
                op1, close_l = "<", "]"
            if labels:
                label = labels[idx]
            else:
                label = f"{close_l}{first_elem};{second_elem}{close_r}"
            conditions += [
                f"'{first_elem}' {op1} {column} AND {column} {op2} '{second_elem}' THEN '{label}'"
            ]
        expr = "CASE WHEN " + " WHEN ".join(conditions) + " END"
        self.apply(func=expr)

    @save_verticapy_logs
    def date_part(self, field: str):
        """
    Extracts a specific TS field from the vColumn (only if the vColumn type is 
    date like). The vColumn will be transformed.

    Parameters
    ----------
    field: str
        The field to extract. It must be one of the following: 
        CENTURY / DAY / DECADE / DOQ / DOW / DOY / EPOCH / HOUR / ISODOW / ISOWEEK /
        ISOYEAR / MICROSECONDS / MILLENNIUM / MILLISECONDS / MINUTE / MONTH / QUARTER / 
        SECOND / TIME ZONE / TIMEZONE_HOUR / TIMEZONE_MINUTE / WEEK / YEAR

    Returns
    -------
    vDataFrame
        self.parent

    See Also
    --------
    vDataFrame[].slice : Slices the vColumn using a time series rule.
        """
        return self.apply(func=f"DATE_PART('{field}', {{}})")

    @save_verticapy_logs
    def discretize(
        self,
        method: Literal["auto", "smart", "same_width", "same_freq", "topk"] = "auto",
        h: Union[int, float] = 0,
        nbins: int = -1,
        k: int = 6,
        new_category: str = "Others",
        RFmodel_params: dict = {},
        response: str = "",
        return_enum_trans: bool = False,
    ):
        """
    Discretizes the vColumn using the input method.

    Parameters
    ----------
    method: str, optional
        The method to use to discretize the vColumn.
            auto       : Uses method 'same_width' for numerical vColumns, cast 
                the other types to varchar.
            same_freq  : Computes bins with the same number of elements.
            same_width : Computes regular width bins.
            smart      : Uses the Random Forest on a response column to find the most 
                relevant interval to use for the discretization.
            topk       : Keeps the topk most frequent categories and merge the other 
                into one unique category.
    h: int / float, optional
        The interval size to convert to use to convert the vColumn. If this parameter 
        is equal to 0, an optimised interval will be computed.
    nbins: int, optional
        Number of bins used for the discretization (must be > 1)
    k: int, optional
        The integer k of the 'topk' method.
    new_category: str, optional
        The name of the merging category when using the 'topk' method.
    RFmodel_params: dict, optional
        Dictionary of the Random Forest model parameters used to compute the best splits 
        when 'method' is set to 'smart'. A RF Regressor will be trained if the response
        is numerical (except ints and bools), a RF Classifier otherwise.
        Example: Write {"n_estimators": 20, "max_depth": 10} to train a Random Forest with
        20 trees and a maximum depth of 10.
    response: str, optional
        Response vColumn when method is set to 'smart'.
    return_enum_trans: bool, optional
        Returns the transformation instead of the vDataFrame parent and do not apply
        it. This parameter is very useful for testing to be able to look at the final 
        transformation.

    Returns
    -------
    vDataFrame
        self.parent

    See Also
    --------
    vDataFrame[].decode       : Encodes the vColumn with user defined Encoding.
    vDataFrame[].get_dummies  : Encodes the vColumn with One-Hot Encoding.
    vDataFrame[].label_encode : Encodes the vColumn with Label Encoding.
    vDataFrame[].mean_encode  : Encodes the vColumn using the mean encoding of a response.
        """
        from verticapy.learn.ensemble import (
            RandomForestRegressor,
            RandomForestClassifier,
        )

        if self.isnum() and method == "smart":
            schema = OPTIONS["temp_schema"]
            if not (schema):
                schema = "public"
            tmp_view_name = gen_tmp_name(schema=schema, name="view")
            tmp_model_name = gen_tmp_name(schema=schema, name="model")
            assert nbins >= 2, ParameterError(
                "Parameter 'nbins' must be greater or equals to 2 in case "
                "of discretization using the method 'smart'."
            )
            assert response, ParameterError(
                "Parameter 'response' can not be empty in case of "
                "discretization using the method 'smart'."
            )
            response = self.parent.format_colnames(response)
            drop(tmp_view_name, method="view")
            self.parent.to_db(tmp_view_name)
            drop(tmp_model_name, method="model")
            if self.parent[response].category() == "float":
                model = RandomForestRegressor(tmp_model_name)
            else:
                model = RandomForestClassifier(tmp_model_name)
            model.set_params({"n_estimators": 20, "max_depth": 8, "nbins": 100})
            model.set_params(RFmodel_params)
            parameters = model.get_params()
            try:
                model.fit(tmp_view_name, [self.alias], response)
                query = [
                    f"""
                    (SELECT 
                        READ_TREE(USING PARAMETERS 
                            model_name = '{tmp_model_name}', 
                            tree_id = {i}, 
                            format = 'tabular'))"""
                    for i in range(parameters["n_estimators"])
                ]
                query = f"""
                    SELECT 
                        /*+LABEL('vColumn.discretize')*/ split_value 
                    FROM 
                        (SELECT 
                            split_value, 
                            MAX(weighted_information_gain) 
                        FROM ({' UNION ALL '.join(query)}) VERTICAPY_SUBTABLE 
                        WHERE split_value IS NOT NULL 
                        GROUP BY 1 ORDER BY 2 DESC LIMIT {nbins - 1}) VERTICAPY_SUBTABLE 
                    ORDER BY split_value::float"""
                result = _executeSQL(
                    query=query,
                    title="Computing the optimized histogram nbins using Random Forest.",
                    method="fetchall",
                    sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
                    symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
                )
                result = [x[0] for x in result]
            finally:
                drop(tmp_view_name, method="view")
                drop(tmp_model_name, method="model")
            result = [self.min()] + result + [self.max()]
        elif method == "topk":
            assert k >= 2, ParameterError(
                "Parameter 'k' must be greater or equals to 2 in "
                "case of discretization using the method 'topk'"
            )
            distinct = self.topk(k).values["index"]
            category_str = to_varchar(self.category())
            X_str = ", ".join([f"""'{str(x).replace("'", "''")}'""" for x in distinct])
            new_category_str = new_category.replace("'", "''")
            trans = (
                f"""(CASE 
                        WHEN {category_str} IN ({X_str})
                        THEN {category_str} || '' 
                        ELSE '{new_category_str}' 
                     END)""",
                "varchar",
                "text",
            )
        elif self.isnum() and method == "same_freq":
            assert nbins >= 2, ParameterError(
                "Parameter 'nbins' must be greater or equals to 2 in case "
                "of discretization using the method 'same_freq'"
            )
            count = self.count()
            nb = int(float(count / int(nbins)))
            assert nb != 0, Exception(
                "Not enough values to compute the Equal Frequency discretization"
            )
            total, query, nth_elems = nb, [], []
            while total < int(float(count / int(nbins))) * int(nbins):
                nth_elems += [str(total)]
                total += nb
            possibilities = ", ".join(["1"] + nth_elems + [str(count)])
            where = f"WHERE _verticapy_row_nb_ IN ({possibilities})"
            query = f"""
                SELECT /*+LABEL('vColumn.discretize')*/ 
                    {self.alias} 
                FROM (SELECT 
                        {self.alias}, 
                        ROW_NUMBER() OVER (ORDER BY {self.alias}) AS _verticapy_row_nb_ 
                      FROM {self.parent.__genSQL__()} 
                      WHERE {self.alias} IS NOT NULL) VERTICAPY_SUBTABLE {where}"""
            result = _executeSQL(
                query=query,
                title="Computing the equal frequency histogram bins.",
                method="fetchall",
                sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
                symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
            )
            result = [elem[0] for elem in result]
        elif self.isnum() and method in ("same_width", "auto"):
            if not (h) or h <= 0:
                if nbins <= 0:
                    h = self.numh()
                else:
                    h = (self.max() - self.min()) * 1.01 / nbins
                if h > 0.01:
                    h = round(h, 2)
                elif h > 0.0001:
                    h = round(h, 4)
                elif h > 0.000001:
                    h = round(h, 6)
                if self.category() == "int":
                    h = int(max(math.floor(h), 1))
            floor_end = -1 if (self.category() == "int") else ""
            if (h > 1) or (self.category() == "float"):
                trans = (
                    f"'[' || FLOOR({{}} / {h}) * {h} || ';' || (FLOOR({{}} / {h}) * {h} + {h}{floor_end}) || ']'",
                    "varchar",
                    "text",
                )
            else:
                trans = ("FLOOR({}) || ''", "varchar", "text")
        else:
            trans = ("{} || ''", "varchar", "text")
        if (self.isnum() and method == "same_freq") or (
            self.isnum() and method == "smart"
        ):
            n = len(result)
            trans = "(CASE "
            for i in range(1, n):
                trans += f"""
                    WHEN {{}} 
                        BETWEEN {result[i - 1]} 
                        AND {result[i]} 
                    THEN '[{result[i - 1]};{result[i]}]' """
            trans += " ELSE NULL END)"
            trans = (trans, "varchar", "text")
        if return_enum_trans:
            return trans
        else:
            self.transformations += [trans]
            sauv = {}
            for elem in self.catalog:
                sauv[elem] = self.catalog[elem]
            self.parent.__update_catalog__(erase=True, columns=[self.alias])
            try:
                if "count" in sauv:
                    self.catalog["count"] = sauv["count"]
                    self.catalog["percent"] = (
                        100 * sauv["count"] / self.parent.shape()[0]
                    )
            except:
                pass
            self.parent.__add_to_history__(
                f"[Discretize]: The vColumn {self.alias} was discretized."
            )
        return self.parent

    @save_verticapy_logs
    def fill_outliers(
        self,
        method: Literal["winsorize", "null", "mean"] = "winsorize",
        threshold: Union[int, float] = 4.0,
        use_threshold: bool = True,
        alpha: Union[int, float] = 0.05,
    ):
        """
    Fills the vColumns outliers using the input method.

    Parameters
        ----------
        method: str, optional
            Method to use to fill the vColumn outliers.
                mean      : Replaces the upper and lower outliers by their respective 
                    average. 
                null      : Replaces the outliers by the NULL value.
                winsorize : Clips the vColumn using as lower bound quantile(alpha) and as 
                    upper bound quantile(1-alpha) if 'use_threshold' is set to False else 
                    the lower and upper ZScores.
        threshold: int / float, optional
            Uses the Gaussian distribution to define the outliers. After normalizing the 
            data (Z-Score), if the absolute value of the record is greater than the 
            threshold it will be considered as an outlier.
        use_threshold: bool, optional
            Uses the threshold instead of the 'alpha' parameter.
        alpha: int / float, optional
            Number representing the outliers threshold. Values lesser than quantile(alpha) 
            or greater than quantile(1-alpha) will be filled.

        Returns
        -------
        vDataFrame
            self.parent

    See Also
    --------
    vDataFrame[].drop_outliers : Drops outliers in the vColumn.
    vDataFrame.outliers      : Adds a new vColumn labeled with 0 and 1 
        (1 meaning global outlier).
        """
        if use_threshold:
            result = self.aggregate(func=["std", "avg"]).transpose().values
            p_alpha, p_1_alpha = (
                -threshold * result["std"][0] + result["avg"][0],
                threshold * result["std"][0] + result["avg"][0],
            )
        else:
            query = f"""
                SELECT /*+LABEL('vColumn.fill_outliers')*/ 
                    PERCENTILE_CONT({alpha}) WITHIN GROUP (ORDER BY {self.alias}) OVER (), 
                    PERCENTILE_CONT(1 - {alpha}) WITHIN GROUP (ORDER BY {self.alias}) OVER () 
                FROM {self.parent.__genSQL__()} LIMIT 1"""
            p_alpha, p_1_alpha = _executeSQL(
                query=query,
                title=f"Computing the quantiles of {self.alias}.",
                method="fetchrow",
                sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
                symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
            )
        if method == "winsorize":
            self.clip(lower=p_alpha, upper=p_1_alpha)
        elif method == "null":
            self.apply(
                func=f"(CASE WHEN ({{}} BETWEEN {p_alpha} AND {p_1_alpha}) THEN {{}} ELSE NULL END)"
            )
        elif method == "mean":
            query = f"""
                WITH vdf_table AS 
                    (SELECT 
                        /*+LABEL('vColumn.fill_outliers')*/ * 
                    FROM {self.parent.__genSQL__()}) 
                    (SELECT 
                        AVG({self.alias}) 
                    FROM vdf_table WHERE {self.alias} < {p_alpha}) 
                    UNION ALL 
                    (SELECT 
                        AVG({self.alias}) 
                    FROM vdf_table WHERE {self.alias} > {p_1_alpha})"""
            mean_alpha, mean_1_alpha = [
                item[0]
                for item in _executeSQL(
                    query=query,
                    title=f"Computing the average of the {self.alias}'s lower and upper outliers.",
                    method="fetchall",
                    sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
                    symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
                )
            ]
            if mean_alpha == None:
                mean_alpha = "NULL"
            if mean_1_alpha == None:
                mean_alpha = "NULL"
            self.apply(
                func=f"""
                    (CASE 
                        WHEN {{}} < {p_alpha} 
                        THEN {mean_alpha} 
                        WHEN {{}} > {p_1_alpha} 
                        THEN {mean_1_alpha} 
                        ELSE {{}} 
                    END)"""
            )
        return self.parent

    @save_verticapy_logs
    def fillna(
        self,
        val: Union[int, float, str, datetime.datetime, datetime.date] = None,
        method: Literal[
            "auto",
            "mode",
            "0ifnull",
            "mean",
            "avg",
            "median",
            "ffill",
            "pad",
            "bfill",
            "backfill",
        ] = "auto",
        expr: Union[str, str_sql] = "",
        by: Union[str, list] = [],
        order_by: Union[str, list] = [],
    ):
        """
    Fills missing elements in the vColumn with a user-specified rule.

    Parameters
    ----------
    val: int / float / str / date, optional
        Value to use to impute the vColumn.
    method: dict, optional
        Method to use to impute the missing values.
            auto    : Mean for the numerical and Mode for the categorical vColumns.
            bfill   : Back Propagation of the next element (Constant Interpolation).
            ffill   : Propagation of the first element (Constant Interpolation).
            mean    : Average.
            median  : median.
            mode    : mode (most occurent element).
            0ifnull : 0 when the vColumn is null, 1 otherwise.
    expr: str, optional
        SQL expression.
    by: str / list, optional
        vColumns used in the partition.
    order_by: str / list, optional
        List of the vColumns to use to sort the data when using TS methods.

    Returns
    -------
    vDataFrame
        self.parent

    See Also
    --------
    vDataFrame[].dropna : Drops the vColumn missing values.
        """
        by, order_by = self.parent.format_colnames(by, order_by)
        if isinstance(by, str):
            by = [by]
        if isinstance(order_by, str):
            order_by = [order_by]
        method = method.lower()
        if method == "auto":
            method = "mean" if (self.isnum() and self.nunique(True) > 6) else "mode"
        total = self.count()
        if (method == "mode") and (val == None):
            val = self.mode(dropna=True)
            if val == None:
                warning_message = (
                    f"The vColumn {self.alias} has no mode "
                    "(only missing values).\nNothing was filled."
                )
                warnings.warn(warning_message, Warning)
                return self.parent
        if isinstance(val, str):
            val = val.replace("'", "''")
        if val != None:
            new_column = f"COALESCE({{}}, '{val}')"
        elif expr:
            new_column = f"COALESCE({{}}, {expr})"
        elif method == "0ifnull":
            new_column = "DECODE({}, NULL, 0, 1)"
        elif method in ("mean", "avg", "median"):
            fun = "MEDIAN" if (method == "median") else "AVG"
            if by == []:
                if fun == "AVG":
                    val = self.avg()
                elif fun == "MEDIAN":
                    val = self.median()
                new_column = f"COALESCE({{}}, {val})"
            elif (len(by) == 1) and (self.parent[by[0]].nunique() < 50):
                try:
                    if fun == "MEDIAN":
                        fun = "APPROXIMATE_MEDIAN"
                    query = f"""
                        SELECT 
                            /*+LABEL('vColumn.fillna')*/ {by[0]}, 
                            {fun}({self.alias})
                        FROM {self.parent.__genSQL__()} 
                        GROUP BY {by[0]};"""
                    result = _executeSQL(
                        query=query,
                        title="Computing the different aggregations.",
                        method="fetchall",
                        sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
                        symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
                    )
                    for idx, x in enumerate(result):
                        if x[0] == None:
                            result[idx][0] = "NULL"
                        else:
                            x0 = str(x[0]).replace("'", "''")
                            result[idx][0] = f"'{x0}'"
                        result[idx][1] = "NULL" if (x[1] == None) else str(x[1])
                    val = ", ".join([f"{x[0]}, {x[1]}" for x in result])
                    new_column = f"COALESCE({{}}, DECODE({by[0]}, {val}, NULL))"
                    _executeSQL(
                        query=f"""
                            SELECT 
                                /*+LABEL('vColumn.fillna')*/ 
                                {new_column.format(self.alias)} 
                            FROM {self.parent.__genSQL__()} 
                            LIMIT 1""",
                        print_time_sql=False,
                        sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
                        symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
                    )
                except:
                    new_column = f"""
                        COALESCE({{}}, {fun}({{}}) 
                            OVER (PARTITION BY {', '.join(by)}))"""
            else:
                new_column = f"""
                    COALESCE({{}}, {fun}({{}}) 
                        OVER (PARTITION BY {', '.join(by)}))"""
        elif method in ("ffill", "pad", "bfill", "backfill"):
            assert order_by, ParameterError(
                "If the method is in ffill|pad|bfill|backfill then 'order_by'"
                " must be a list of at least one element to use to order the data"
            )
            desc = "" if (method in ("ffill", "pad")) else " DESC"
            partition_by = f"PARTITION BY {', '.join(by)}" if (by) else ""
            order_by_ts = ", ".join([quote_ident(column) + desc for column in order_by])
            new_column = f"""
                COALESCE({{}}, LAST_VALUE({{}} IGNORE NULLS) 
                    OVER ({partition_by} 
                    ORDER BY {order_by_ts}))"""
        if method in ("mean", "median") or isinstance(val, float):
            category, ctype = "float", "float"
        elif method == "0ifnull":
            category, ctype = "int", "bool"
        else:
            category, ctype = self.category(), self.ctype()
        copy_trans = [elem for elem in self.transformations]
        total = self.count()
        if method not in ["mode", "0ifnull"]:
            max_floor = 0
            all_partition = by
            if method in ["ffill", "pad", "bfill", "backfill"]:
                all_partition += [elem for elem in order_by]
            for elem in all_partition:
                if len(self.parent[elem].transformations) > max_floor:
                    max_floor = len(self.parent[elem].transformations)
            max_floor -= len(self.transformations)
            for k in range(max_floor):
                self.transformations += [("{}", self.ctype(), self.category())]
        self.transformations += [(new_column, ctype, category)]
        try:
            sauv = {}
            for elem in self.catalog:
                sauv[elem] = self.catalog[elem]
            self.parent.__update_catalog__(erase=True, columns=[self.alias])
            total = abs(self.count() - total)
        except Exception as e:
            self.transformations = [elem for elem in copy_trans]
            raise QueryError(f"{e}\nAn Error happened during the filling.")
        if total > 0:
            try:
                if "count" in sauv:
                    self.catalog["count"] = int(sauv["count"]) + total
                    self.catalog["percent"] = (
                        100 * (int(sauv["count"]) + total) / self.parent.shape()[0]
                    )
            except:
                pass
            total = int(total)
            conj = "s were " if total > 1 else " was "
            if OPTIONS["print_info"]:
                print(f"{total} element{conj}filled.")
            self.parent.__add_to_history__(
                f"[Fillna]: {total} {self.alias} missing value{conj} filled."
            )
        else:
            if OPTIONS["print_info"]:
                print("Nothing was filled.")
            self.transformations = [t for t in copy_trans]
            for s in sauv:
                self.catalog[s] = sauv[s]
        return self.parent

    @save_verticapy_logs
    def one_hot_encode(
        self,
        prefix: str = "",
        prefix_sep: str = "_",
        drop_first: bool = True,
        use_numbers_as_suffix: bool = False,
    ):
        """
    Encodes the vColumn with the One-Hot Encoding algorithm.

    Parameters
    ----------
    prefix: str, optional
        Prefix of the dummies.
    prefix_sep: str, optional
        Prefix delimitor of the dummies.
    drop_first: bool, optional
        Drops the first dummy to avoid the creation of correlated features.
    use_numbers_as_suffix: bool, optional
        Uses numbers as suffix instead of the vColumns categories.

    Returns
    -------
    vDataFrame
        self.parent

    See Also
    --------
    vDataFrame[].decode       : Encodes the vColumn with user defined Encoding.
    vDataFrame[].discretize   : Discretizes the vColumn.
    vDataFrame[].label_encode : Encodes the vColumn with Label Encoding.
    vDataFrame[].mean_encode  : Encodes the vColumn using the mean encoding of a response.
        """
        from verticapy.core.vcolumn import vColumn

        distinct_elements = self.distinct()
        if distinct_elements not in ([0, 1], [1, 0]) or self.isbool():
            all_new_features = []
            if not (prefix):
                prefix = self.alias.replace('"', "") + prefix_sep.replace('"', "_")
            else:
                prefix = prefix.replace('"', "_") + prefix_sep.replace('"', "_")
            n = 1 if drop_first else 0
            for k in range(len(distinct_elements) - n):
                distinct_elements_k = str(distinct_elements[k]).replace('"', "_")
                if use_numbers_as_suffix:
                    name = f'"{prefix}{k}"'
                else:
                    name = f'"{prefix}{distinct_elements_k}"'
                assert not (self.parent.is_colname_in(name)), NameError(
                    "A vColumn has already the alias of one of "
                    f"the dummies ({name}).\nIt can be the result "
                    "of using previously the method on the vColumn "
                    "or simply because of ambiguous columns naming."
                    "\nBy changing one of the parameters ('prefix', "
                    "'prefix_sep'), you'll be able to solve this "
                    "issue."
                )
            for k in range(len(distinct_elements) - n):
                distinct_elements_k = str(distinct_elements[k]).replace("'", "''")
                if use_numbers_as_suffix:
                    name = f'"{prefix}{k}"'
                else:
                    name = f'"{prefix}{distinct_elements_k}"'
                name = (
                    name.replace(" ", "_")
                    .replace("/", "_")
                    .replace(",", "_")
                    .replace("'", "_")
                )
                expr = f"DECODE({{}}, '{distinct_elements_k}', 1, 0)"
                transformations = self.transformations + [(expr, "bool", "int")]
                new_vColumn = vColumn(
                    name,
                    parent=self.parent,
                    transformations=transformations,
                    catalog={
                        "min": 0,
                        "max": 1,
                        "count": self.parent.shape()[0],
                        "percent": 100.0,
                        "unique": 2,
                        "approx_unique": 2,
                        "prod": 0,
                    },
                )
                setattr(self.parent, name, new_vColumn)
                setattr(self.parent, name.replace('"', ""), new_vColumn)
                self.parent._VERTICAPY_VARIABLES_["columns"] += [name]
                all_new_features += [name]
            conj = "s were " if len(all_new_features) > 1 else " was "
            self.parent.__add_to_history__(
                "[Get Dummies]: One hot encoder was applied to the vColumn "
                f"{self.alias}\n{len(all_new_features)} feature{conj}created: "
                f"{', '.join(all_new_features)}."
            )
        return self.parent

    get_dummies = one_hot_encode

    @save_verticapy_logs
    def label_encode(self):
        """
    Encodes the vColumn using a bijection from the different categories to
    [0, n - 1] (n being the vColumn cardinality).

    Returns
    -------
    vDataFrame
        self.parent

    See Also
    --------
    vDataFrame[].decode       : Encodes the vColumn with a user defined Encoding.
    vDataFrame[].discretize   : Discretizes the vColumn.
    vDataFrame[].get_dummies  : Encodes the vColumn with One-Hot Encoding.
    vDataFrame[].mean_encode  : Encodes the vColumn using the mean encoding of a response.
        """
        if self.category() in ["date", "float"]:
            warning_message = (
                "label_encode is only available for categorical variables."
            )
            warnings.warn(warning_message, Warning)
        else:
            distinct_elements = self.distinct()
            expr = ["DECODE({}"]
            text_info = "\n"
            for k in range(len(distinct_elements)):
                distinct_elements_k = str(distinct_elements[k]).replace("'", "''")
                expr += [f"'{distinct_elements_k}', {k}"]
                text_info += f"\t{distinct_elements[k]} => {k}"
            expr = f"{', '.join(expr)}, {len(distinct_elements)})"
            self.transformations += [(expr, "int", "int")]
            self.parent.__update_catalog__(erase=True, columns=[self.alias])
            self.catalog["count"] = self.parent.shape()[0]
            self.catalog["percent"] = 100
            self.parent.__add_to_history__(
                "[Label Encoding]: Label Encoding was applied to the vColumn"
                f" {self.alias} using the following mapping:{text_info}"
            )
        return self.parent

    @save_verticapy_logs
    def mean_encode(self, response: str):
        """
    Encodes the vColumn using the average of the response partitioned by the 
    different vColumn categories.

    Parameters
    ----------
    response: str
        Response vColumn.

    Returns
    -------
    vDataFrame
        self.parent

    See Also
    --------
    vDataFrame[].decode       : Encodes the vColumn using a user-defined encoding.
    vDataFrame[].discretize   : Discretizes the vColumn.
    vDataFrame[].label_encode : Encodes the vColumn with Label Encoding.
    vDataFrame[].get_dummies  : Encodes the vColumn with One-Hot Encoding.
        """
        response = self.parent.format_colnames(response)
        assert self.parent[response].isnum(), TypeError(
            "The response column must be numerical to use a mean encoding"
        )
        max_floor = len(self.parent[response].transformations) - len(
            self.transformations
        )
        for k in range(max_floor):
            self.transformations += [("{}", self.ctype(), self.category())]
        self.transformations += [
            (f"AVG({response}) OVER (PARTITION BY {{}})", "int", "float",)
        ]
        self.parent.__update_catalog__(erase=True, columns=[self.alias])
        self.parent.__add_to_history__(
            f"[Mean Encode]: The vColumn {self.alias} was transformed "
            f"using a mean encoding with {response} as Response Column."
        )
        if OPTIONS["print_info"]:
            print("The mean encoding was successfully done.")
        return self.parent

    @save_verticapy_logs
    def normalize(
        self,
        method: Literal["zscore", "robust_zscore", "minmax"] = "zscore",
        by: Union[str, list] = [],
        return_trans: bool = False,
    ):
        """
    Normalizes the input vColumns using the input method.

    Parameters
    ----------
    method: str, optional
        Method to use to normalize.
            zscore        : Normalization using the Z-Score (avg and std).
                (x - avg) / std
            robust_zscore : Normalization using the Robust Z-Score (median and mad).
                (x - median) / (1.4826 * mad)
            minmax        : Normalization using the MinMax (min and max).
                (x - min) / (max - min)
    by: str / list, optional
        vColumns used in the partition.
    return_trans: bool, optimal
        If set to True, the method will return the transformation used instead of
        the parent vDataFrame. This parameter is used for testing purpose.

    Returns
    -------
    vDataFrame
        self.parent

    See Also
    --------
    vDataFrame.outliers : Computes the vDataFrame Global Outliers.
        """
        if isinstance(by, str):
            by = [by]
        method = method.lower()
        by = self.parent.format_colnames(by)
        nullifzero, n = 1, len(by)
        if self.isbool():

            warning_message = "Normalize doesn't work on booleans"
            warnings.warn(warning_message, Warning)

        elif self.isnum():

            if method == "zscore":

                if n == 0:
                    nullifzero = 0
                    avg, stddev = self.aggregate(["avg", "std"]).values[self.alias]
                    if stddev == 0:
                        warning_message = (
                            f"Can not normalize {self.alias} using a "
                            "Z-Score - The Standard Deviation is null !"
                        )
                        warnings.warn(warning_message, Warning)
                        return self
                elif (n == 1) and (self.parent[by[0]].nunique() < 50):
                    try:
                        result = _executeSQL(
                            query=f"""
                                SELECT 
                                    /*+LABEL('vColumn.normalize')*/ {by[0]}, 
                                    AVG({self.alias}), 
                                    STDDEV({self.alias}) 
                                FROM {self.parent.__genSQL__()} GROUP BY {by[0]}""",
                            title="Computing the different categories to normalize.",
                            method="fetchall",
                            sql_push_ext=self.parent._VERTICAPY_VARIABLES_[
                                "sql_push_ext"
                            ],
                            symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
                        )
                        for i in range(len(result)):
                            if result[i][2] == None:
                                pass
                            elif math.isnan(result[i][2]):
                                result[i][2] = None
                        avg_stddev = []
                        for i in range(1, 3):
                            if x[0] != None:
                                x0 = f"""'{str(x[0]).replace("'", "''")}'"""
                            else:
                                x0 = "NULL"
                            x_tmp = [
                                f"""{x0}, {x[i] if x[i] != None else "NULL"}"""
                                for x in result
                                if x[i] != None
                            ]
                            avg_stddev += [
                                f"""DECODE({by[0]}, {", ".join(x_tmp)}, NULL)"""
                            ]
                        avg, stddev = avg_stddev
                        _executeSQL(
                            query=f"""
                                SELECT 
                                    /*+LABEL('vColumn.normalize')*/ 
                                    {avg},
                                    {stddev} 
                                FROM {self.parent.__genSQL__()} 
                                LIMIT 1""",
                            print_time_sql=False,
                            sql_push_ext=self.parent._VERTICAPY_VARIABLES_[
                                "sql_push_ext"
                            ],
                            symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
                        )
                    except:
                        avg, stddev = (
                            f"AVG({self.alias}) OVER (PARTITION BY {', '.join(by)})",
                            f"STDDEV({self.alias}) OVER (PARTITION BY {', '.join(by)})",
                        )
                else:
                    avg, stddev = (
                        f"AVG({self.alias}) OVER (PARTITION BY {', '.join(by)})",
                        f"STDDEV({self.alias}) OVER (PARTITION BY {', '.join(by)})",
                    )
                nullifzero = "NULLIFZERO" if (nullifzero) else ""
                if return_trans:
                    return f"({self.alias} - {avg}) / {nullifzero}({stddev})"
                else:
                    final_transformation = [
                        (f"({{}} - {avg}) / {nullifzero}({stddev})", "float", "float",)
                    ]

            elif method == "robust_zscore":

                if n > 0:
                    warning_message = (
                        "The method 'robust_zscore' is available only if the "
                        "parameter 'by' is empty\nIf you want to normalize by "
                        "grouping by elements, please use a method in zscore|minmax"
                    )
                    warnings.warn(warning_message, Warning)
                    return self
                mad, med = self.aggregate(["mad", "approx_median"]).values[self.alias]
                mad *= 1.4826
                if mad != 0:
                    if return_trans:
                        return f"({self.alias} - {med}) / ({mad})"
                    else:
                        final_transformation = [
                            (f"({{}} - {med}) / ({mad})", "float", "float",)
                        ]
                else:
                    warning_message = (
                        f"Can not normalize {self.alias} using a "
                        "Robust Z-Score - The MAD is null !"
                    )
                    warnings.warn(warning_message, Warning)
                    return self

            elif method == "minmax":

                if n == 0:
                    nullifzero = 0
                    cmin, cmax = self.aggregate(["min", "max"]).values[self.alias]
                    if cmax - cmin == 0:
                        warning_message = (
                            f"Can not normalize {self.alias} using "
                            "the MIN and the MAX. MAX = MIN !"
                        )
                        warnings.warn(warning_message, Warning)
                        return self
                elif n == 1:
                    try:
                        result = _executeSQL(
                            query=f"""
                                SELECT 
                                    /*+LABEL('vColumn.normalize')*/ {by[0]}, 
                                    MIN({self.alias}), 
                                    MAX({self.alias})
                                FROM {self.parent.__genSQL__()} 
                                GROUP BY {by[0]}""",
                            title=f"Computing the different categories {by[0]} to normalize.",
                            method="fetchall",
                            sql_push_ext=self.parent._VERTICAPY_VARIABLES_[
                                "sql_push_ext"
                            ],
                            symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
                        )
                        cmin_cmax = []
                        for i in range(1, 3):
                            if x[0] != None:
                                x0 = f"""'{str(x[0]).replace("'", "''")}'"""
                            else:
                                x0 = "NULL"
                            x_tmp = [
                                f"""{x0}, {x[i] if x[i] != None else "NULL"}"""
                                for x in result
                                if x[i] != None
                            ]
                            cmin_cmax += [
                                f"""DECODE({by[0]}, {", ".join(x_tmp)}, NULL)"""
                            ]
                        cmin, cmax = cmin_cmax
                        _executeSQL(
                            query=f"""
                                SELECT 
                                    /*+LABEL('vColumn.normalize')*/ 
                                    {cmin_cmax[1]}, 
                                    {cmin_cmax[0]} 
                                FROM {self.parent.__genSQL__()} 
                                LIMIT 1""",
                            print_time_sql=False,
                            sql_push_ext=self.parent._VERTICAPY_VARIABLES_[
                                "sql_push_ext"
                            ],
                            symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
                        )
                    except:
                        cmax, cmin = (
                            f"MAX({self.alias}) OVER (PARTITION BY {', '.join(by)})",
                            f"MIN({self.alias}) OVER (PARTITION BY {', '.join(by)})",
                        )
                else:
                    cmax, cmin = (
                        f"MAX({self.alias}) OVER (PARTITION BY {', '.join(by)})",
                        f"MIN({self.alias}) OVER (PARTITION BY {', '.join(by)})",
                    )
                nullifzero = "NULLIFZERO" if (nullifzero) else ""
                if return_trans:
                    return f"({self.alias} - {cmin}) / {nullifzero}({cmax} - {cmin})"
                else:
                    final_transformation = [
                        (
                            f"({{}} - {cmin}) / {nullifzero}({cmax} - {cmin})",
                            "float",
                            "float",
                        )
                    ]

            if method != "robust_zscore":
                max_floor = 0
                for elem in by:
                    if len(self.parent[elem].transformations) > max_floor:
                        max_floor = len(self.parent[elem].transformations)
                max_floor -= len(self.transformations)
                for k in range(max_floor):
                    self.transformations += [("{}", self.ctype(), self.category())]
            self.transformations += final_transformation
            sauv = {}
            for elem in self.catalog:
                sauv[elem] = self.catalog[elem]
            self.parent.__update_catalog__(erase=True, columns=[self.alias])
            try:

                if "count" in sauv:
                    self.catalog["count"] = sauv["count"]
                    self.catalog["percent"] = (
                        100 * sauv["count"] / self.parent.shape()[0]
                    )

                for elem in sauv:

                    if "top" in elem:

                        if "percent" in elem:
                            self.catalog[elem] = sauv[elem]
                        elif elem == None:
                            self.catalog[elem] = None
                        elif method == "robust_zscore":
                            self.catalog[elem] = (sauv[elem] - sauv["approx_50%"]) / (
                                1.4826 * sauv["mad"]
                            )
                        elif method == "zscore":
                            self.catalog[elem] = (sauv[elem] - sauv["mean"]) / sauv[
                                "std"
                            ]
                        elif method == "minmax":
                            self.catalog[elem] = (sauv[elem] - sauv["min"]) / (
                                sauv["max"] - sauv["min"]
                            )

            except:
                pass
            if method == "robust_zscore":
                self.catalog["median"] = 0
                self.catalog["mad"] = 1 / 1.4826
            elif method == "zscore":
                self.catalog["mean"] = 0
                self.catalog["std"] = 1
            elif method == "minmax":
                self.catalog["min"] = 0
                self.catalog["max"] = 1
            self.parent.__add_to_history__(
                f"[Normalize]: The vColumn '{self.alias}' was "
                f"normalized with the method '{method}'."
            )
        else:
            raise TypeError("The vColumn must be numerical for Normalization")
        return self.parent
