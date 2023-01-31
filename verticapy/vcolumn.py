# (c) Copyright [2018-2023] Micro Focus or one of its affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# |_     |~) _  _| _  /~\    _ |.
# |_)\/  |_)(_|(_||   \_/|_|(_|||
#    /
#              ____________       ______
#             / __        `\     /     /
#            |  \/         /    /     /
#            |______      /    /     /
#                   |____/    /     /
#          _____________     /     /
#          \           /    /     /
#           \         /    /     /
#            \_______/    /     /
#             ______     /     /
#             \    /    /     /
#              \  /    /     /
#               \/    /     /
#                    /     /
#                   /     /
#                   \    /
#                    \  /
#                     \/
#                    _
# \  / _  __|_. _ _ |_)
#  \/ (/_|  | |(_(_|| \/
#                     /
# VerticaPy is a Python library with scikit-like functionality for conducting
# data science projects on data stored in Vertica, taking advantage Vertica’s
# speed and built-in analytics and machine learning features. It supports the
# entire data science life cycle, uses a ‘pipeline’ mechanism to sequentialize
# data transformation operations, and offers beautiful graphical options.
#
# VerticaPy aims to do all of the above. The idea is simple: instead of moving
# data around for processing, VerticaPy brings the logic to the data.
#
#
# Modules
#
# Standard Python Modules
import math, re, decimal, warnings, datetime, sys
from collections.abc import Iterable
from typing import Union

# VerticaPy Modules
import verticapy as vp
import verticapy.stats as st
import verticapy.plot as plt
import verticapy.learn.ensemble as vpy_ensemble
import verticapy.learn.neighbors as vpy_neighbors
from verticapy.decorators import (
    save_verticapy_logs,
    check_dtypes,
    check_minimum_version,
)
from verticapy.utilities import *
from verticapy.toolbox import *
from verticapy.errors import *

# Other modules
import numpy as np

##
#
#   __   ___  ______     ______     __         __  __     __    __     __   __
#  /\ \ /  / /\  ___\   /\  __ \   /\ \       /\ \/\ \   /\ "-./  \   /\ "-.\ \
#  \ \ \' /  \ \ \____  \ \ \/\ \  \ \ \____  \ \ \_\ \  \ \ \-./\ \  \ \ \-.  \
#   \ \__/    \ \_____\  \ \_____\  \ \_____\  \ \_____\  \ \_\ \ \_\  \ \_\\"\_\
#    \/_/      \/_____/   \/_____/   \/_____/   \/_____/   \/_/  \/_/   \/_/ \/_/
#
#
# ---#
class vColumn(str_sql):
    """
----------------------------------------------------------------------------------------
Python object which that stores all user transformations. If the vDataFrame
represents the entire relation, a vColumn can be seen as one column of that
relation. vColumns simplify several processes with its abstractions.

Parameters
----------
alias: str
	vColumn alias.
transformations: list, optional
	List of the different transformations. Each transformation must be similar
	to the following: (function, type, category)  
parent: vDataFrame, optional
	Parent of the vColumn. One vDataFrame can have multiple children vColumns 
	whereas one vColumn can only have one parent.
catalog: dict, optional
	Catalog where each key corresponds to an aggregation. vColumns will memorize
	the already computed aggregations to gain in performance. The catalog will
	be updated when the parent vDataFrame is modified.

Attributes
----------
	alias, str           : vColumn alias.
	catalog, dict        : Catalog of pre-computed aggregations.
	parent, vDataFrame   : Parent of the vColumn.
	transformations, str : List of the different transformations.
	"""

    #
    # Special Methods
    #
    # ---#
    def __init__(
        self, alias: str, transformations: list = [], parent=None, catalog: dict = {},
    ):
        self.parent, self.alias, self.transformations = (
            parent,
            alias,
            [elem for elem in transformations],
        )
        self.catalog = {
            "cov": {},
            "pearson": {},
            "spearman": {},
            "spearmand": {},
            "kendall": {},
            "cramer": {},
            "biserial": {},
            "regr_avgx": {},
            "regr_avgy": {},
            "regr_count": {},
            "regr_intercept": {},
            "regr_r2": {},
            "regr_slope": {},
            "regr_sxx": {},
            "regr_sxy": {},
            "regr_syy": {},
        }
        for elem in catalog:
            self.catalog[elem] = catalog[elem]
        self.init_transf = self.transformations[0][0]
        if self.init_transf == "___VERTICAPY_UNDEFINED___":
            self.init_transf = self.alias
        self.init = True

    # ---#
    def __getitem__(self, index):
        if isinstance(index, slice):
            assert index.step in (1, None), ValueError(
                "vColumn doesn't allow slicing having steps different than 1."
            )
            index_stop = index.stop
            index_start = index.start
            if not (isinstance(index_start, int)):
                index_start = 0
            if self.isarray():
                vertica_version(condition=[10, 0, 0])
                if index_start < 0:
                    index_start_str = f"{index_start} + APPLY_COUNT_ELEMENTS({{}})"
                else:
                    index_start_str = str(index_start)
                if isinstance(index_stop, int):
                    if index_stop < 0:
                        index_stop_str = f"{index_stop} + APPLY_COUNT_ELEMENTS({{}})"
                    else:
                        index_stop_str = str(index_stop)
                else:
                    index_stop_str = "1 + APPLY_COUNT_ELEMENTS({})"
                elem_to_select = f"{self.alias}[{index_start_str}:{index_stop_str}]"
                elem_to_select = elem_to_select.replace("{}", self.alias)
                new_alias = quote_ident(
                    f"{self.alias[1:-1]}.{index_start}:{index_stop}"
                )
                query = f"""
                    (SELECT 
                        {elem_to_select} AS {new_alias} 
                    FROM {self.parent.__genSQL__()}) VERTICAPY_SUBTABLE"""
                vcol = vDataFrameSQL(query)[new_alias]
                vcol.transformations[-1] = (
                    new_alias,
                    self.ctype(),
                    self.category(),
                )
                vcol.init_transf = (
                    f"{self.init_transf}[{index_start_str}:{index_stop_str}]"
                )
                vcol.init_transf = vcol.init_transf.replace("{}", self.init_transf)
                return vcol
            else:
                if index_start < 0:
                    index_start += self.parent.shape()[0]
                if isinstance(index_stop, int):
                    if index_stop < 0:
                        index_stop += self.parent.shape()[0]
                    limit = index_stop - index_start
                    if limit <= 0:
                        limit = 0
                    limit = f" LIMIT {limit}"
                else:
                    limit = ""
                query = f"""
                    (SELECT 
                        {self.alias} 
                    FROM {self.parent.__genSQL__()}
                    {self.parent.__get_last_order_by__()} 
                    OFFSET {index_start}
                    {limit}) VERTICAPY_SUBTABLE"""
                return vDataFrameSQL(query)
        elif isinstance(index, int):
            if self.isarray():
                vertica_version(condition=[9, 3, 0])
                elem_to_select = f"{self.alias}[{index}]"
                new_alias = quote_ident(f"{self.alias[1:-1]}.{index}")
                query = f"""
                    (SELECT 
                        {elem_to_select} AS {new_alias} 
                    FROM {self.parent.__genSQL__()}) VERTICAPY_SUBTABLE"""
                vcol = vDataFrameSQL(query)[new_alias]
                vcol.init_transf = f"{self.init_transf}[{index}]"
                return vcol
            else:
                cast = "::float" if self.category() == "float" else ""
                if index < 0:
                    index += self.parent.shape()[0]
                return executeSQL(
                    query=f"""
                        SELECT 
                            /*+LABEL('vColumn.__getitem__')*/ 
                            {self.alias}{cast} 
                        FROM {self.parent.__genSQL__()}
                        {self.parent.__get_last_order_by__()} 
                        OFFSET {index} 
                        LIMIT 1""",
                    title="Getting the vColumn element.",
                    method="fetchfirstelem",
                    sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
                    symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
                )
        elif isinstance(index, str):
            if self.category() == "vmap":
                index_str = index.replace("'", "''")
                elem_to_select = f"MAPLOOKUP({self.alias}, '{index_str}')"
                init_transf = f"MAPLOOKUP({self.init_transf}, '{index_str}')"
            else:
                vertica_version(condition=[10, 0, 0])
                elem_to_select = f"{self.alias}.{quote_ident(index)}"
                init_transf = f"{self.init_transf}.{quote_ident(index)}"
            query = f"""
                (SELECT 
                    {elem_to_select} AS {quote_ident(index)} 
                FROM {self.parent.__genSQL__()}) VERTICAPY_SUBTABLE"""
            vcol = vDataFrameSQL(query)[index]
            vcol.init_transf = init_transf
            return vcol
        else:
            return getattr(self, index)

    # ---#
    def __len__(self):
        return int(self.count())

    # ---#
    def __nonzero__(self):
        return self.count() > 0

    # ---#
    def __repr__(self):
        return self.head(limit=vp.OPTIONS["max_rows"]).__repr__()

    # ---#
    def _repr_html_(self):
        return self.head(limit=vp.OPTIONS["max_rows"])._repr_html_()

    # ---#
    def __setattr__(self, attr, val):
        self.__dict__[attr] = val

    #
    # Methods
    #
    # ---#
    @save_verticapy_logs
    def aad(self):
        """
    ----------------------------------------------------------------------------------------
    Aggregates the vColumn using 'aad' (Average Absolute Deviation).

    Returns
    -------
    float
        aad

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        return self.aggregate(["aad"]).values[self.alias][0]

    # ---#
    @save_verticapy_logs
    def abs(self):
        """
	----------------------------------------------------------------------------------------
	Applies the absolute value function to the input vColumn. 

 	Returns
 	-------
 	vDataFrame
		self.parent

	See Also
	--------
	vDataFrame[].apply : Applies a function to the input vColumn.
		"""
        return self.apply(func="ABS({})")

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def add(self, x: Union[int, float]):
        """
	----------------------------------------------------------------------------------------
	Adds the input element to the vColumn.

	Parameters
 	----------
 	x: float
 		If the vColumn type is date like (date, datetime ...), the parameter 'x' 
 		will represent the number of seconds, otherwise it will represent a number.

 	Returns
 	-------
 	vDataFrame
		self.parent

	See Also
	--------
	vDataFrame[].apply : Applies a function to the input vColumn.
		"""
        if self.isdate():
            return self.apply(func=f"TIMESTAMPADD(SECOND, {x}, {{}})")
        else:
            return self.apply(func=f"{{}} + ({x})")

    # ---#
    @check_dtypes
    def add_copy(self, name: str):
        """
	----------------------------------------------------------------------------------------
	Adds a copy vColumn to the parent vDataFrame.

	Parameters
 	----------
 	name: str
 		Name of the copy.

 	Returns
 	-------
 	vDataFrame
		self.parent

	See Also
	--------
	vDataFrame.eval : Evaluates a customized expression.
		"""
        name = quote_ident(name.replace('"', "_"))
        assert name.replace('"', ""), EmptyParameter(
            "The parameter 'name' must not be empty"
        )
        assert not (self.parent.is_colname_in(name)), NameError(
            f"A vColumn has already the alias {name}.\nBy changing "
            "the parameter 'name', you'll be able to solve this issue."
        )
        new_vColumn = vColumn(
            name,
            parent=self.parent,
            transformations=[item for item in self.transformations],
            catalog=self.catalog,
        )
        setattr(self.parent, name, new_vColumn)
        setattr(self.parent, name[1:-1], new_vColumn)
        self.parent._VERTICAPY_VARIABLES_["columns"] += [name]
        self.parent.__add_to_history__(
            f"[Add Copy]: A copy of the vColumn {self.alias} "
            f"named {name} was added to the vDataFrame."
        )
        return self.parent

    # ---#
    @save_verticapy_logs
    def aggregate(self, func: list):
        """
	----------------------------------------------------------------------------------------
	Aggregates the vColumn using the input functions.

	Parameters
 	----------
 	func: list
 		List of the different aggregation.
            aad            : average absolute deviation
 			approx_unique  : approximative cardinality
 			count          : number of non-missing elements
			cvar           : conditional value at risk
			dtype          : vColumn type
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
				Other aggregations could work if it is part of 
				the DB version you are using.

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
        return self.parent.aggregate(func=func, columns=[self.alias]).transpose()

    agg = aggregate
    # ---#
    @check_dtypes
    @save_verticapy_logs
    def apply(self, func: Union[str, str_sql], copy_name: str = ""):
        """
	----------------------------------------------------------------------------------------
	Applies a function to the vColumn.

	Parameters
 	----------
 	func: str,
 		Function in pure SQL used to transform the vColumn.
 		The function variable must be composed of two flower brackets {}. For 
 		example to apply the function: x -> x^2 + 2 use "POWER({}, 2) + 2".
 	copy_name: str, optional
 		If not empty, a copy will be created using the input Name.

 	Returns
 	-------
 	vDataFrame
		self.parent

	See Also
	--------
	vDataFrame.apply    : Applies functions to the input vColumns.
	vDataFrame.applymap : Applies a function to all the vColumns.
	vDataFrame.eval     : Evaluates a customized expression.
		"""
        if isinstance(func, str_sql):
            func = str(func)
        func_apply = func.replace("{}", self.alias)
        alias_sql_repr = self.alias.replace('"', "")
        try:
            ctype = get_data_types(
                expr=f"""
                    SELECT 
                        {func_apply} AS apply_test_feature 
                    FROM {self.parent.__genSQL__()} 
                    WHERE {self.alias} IS NOT NULL 
                    LIMIT 0""",
                column="apply_test_feature",
            )
            category = get_category_from_vertica_type(ctype=ctype)
            all_cols, max_floor = self.parent.get_columns(), 0
            for column in all_cols:
                try:
                    if (quote_ident(column) in func) or (
                        re.search(
                            re.compile("\\b{}\\b".format(column.replace('"', ""))),
                            func,
                        )
                    ):
                        max_floor = max(
                            len(self.parent[column].transformations), max_floor
                        )
                except:
                    pass
            max_floor -= len(self.transformations)
            if copy_name:
                copy_name_str = copy_name.replace('"', "")
                self.add_copy(name=copy_name)
                for k in range(max_floor):
                    self.parent[copy_name].transformations += [
                        ("{}", self.ctype(), self.category())
                    ]
                self.parent[copy_name].transformations += [(func, ctype, category)]
                self.parent[copy_name].catalog = self.catalog
            else:
                for k in range(max_floor):
                    self.transformations += [("{}", self.ctype(), self.category())]
                self.transformations += [(func, ctype, category)]
                self.parent.__update_catalog__(erase=True, columns=[self.alias])
            self.parent.__add_to_history__(
                f"[Apply]: The vColumn '{alias_sql_repr}' was "
                f"transformed with the func 'x -> {func_apply}'."
            )
            return self.parent
        except Exception as e:
            raise QueryError(
                f"{e}\nError when applying the func 'x -> {func_apply}' "
                f"to '{alias_sql_repr}'"
            )

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def apply_fun(self, func: str, x: Union[str, int, float] = 2):
        """
	----------------------------------------------------------------------------------------
	Applies a default function to the vColumn.

	Parameters
 	----------
 	func: str
 		Function to use to transform the vColumn.
			abs          : absolute value
			acos         : trigonometric inverse cosine
			asin         : trigonometric inverse sine
			atan         : trigonometric inverse tangent
            avg / mean   : average
			cbrt         : cube root
			ceil         : value up to the next whole number
            contain      : checks if 'x' is in the collection
            count        : number of non-null elements
			cos          : trigonometric cosine
			cosh         : hyperbolic cosine
			cot          : trigonometric cotangent
            dim          : dimension (only for arrays)
			exp          : exponential function
            find         : returns the ordinal position of a specified element 
                           in an array (only for arrays)
			floor        : value down to the next whole number
            len / length : length
			ln           : natural logarithm
			log          : logarithm
			log10        : base 10 logarithm
            max          : maximum
            min          : minimum
			mod          : remainder of a division operation
			pow          : number raised to the power of another number
			round        : rounds a value to a specified number of decimal places
			sign         : arithmetic sign
			sin          : trigonometric sine
			sinh         : hyperbolic sine
			sqrt         : arithmetic square root
            sum          : sum
			tan          : trigonometric tangent
			tanh         : hyperbolic tangent
	x: int / float / str, optional
		If the function has two arguments (example, power or mod), 'x' represents 
		the second argument.

 	Returns
 	-------
 	vDataFrame
		self.parent

	See Also
	--------
	vDataFrame[].apply : Applies a function to the vColumn.
		"""
        raise_error_if_not_in(
            "func",
            func,
            [
                "abs",
                "acos",
                "asin",
                "atan",
                "avg",
                "cbrt",
                "ceil",
                "contain",
                "count",
                "cos",
                "cosh",
                "cot",
                "dim",
                "exp",
                "find",
                "floor",
                "len",
                "length",
                "ln",
                "log",
                "log10",
                "max",
                "mean",
                "mod",
                "min",
                "pow",
                "round",
                "sign",
                "sin",
                "sinh",
                "sum",
                "sqrt",
                "tan",
                "tanh",
            ],
        )
        if func == "mean":
            func = "avg"
        elif func == "length":
            func = "len"
        cat = self.category()
        if func == "len":
            if cat == "vmap":
                func = "MAPSIZE"
            elif cat == "complex":
                func = "APPLY_COUNT_ELEMENTS"
            else:
                func = "LENTGH"
        elif func in ("max", "min", "sum", "avg", "count"):
            func = "APPLY_" + func
        elif func == "dim":
            func = "ARRAY_DIMS"
        if func not in ("log", "mod", "pow", "round", "contain", "find"):
            expr = f"{func.upper()}({{}})"
        elif func in ("log", "mod", "pow", "round"):
            expr = f"{func.upper()}({{}}, {x})"
        elif func in ("contain", "find"):
            if func == "contain":
                if cat == "vmap":
                    f = "MAPCONTAINSVALUE"
                else:
                    f = "CONTAINS"
            elif func == "find":
                f = "ARRAY_FIND"
            if isinstance(x, str):
                x = "'" + str(x).replace("'", "''") + "'"
            expr = f"{f}({{}}, {x})"
        return self.apply(func=expr)

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def astype(self, dtype: Union[str, type]):
        """
	----------------------------------------------------------------------------------------
	Converts the vColumn to the input type.

	Parameters
 	----------
 	dtype: str or Python data type
 		New type. One of the following values:
        'json': Converts to a JSON string.
        'array': Converts to an array.
        'vmap': Converts to a VMap. If converting a delimited string, you can 
        add the header_names as follows: dtype = 'vmap(age,name,date)', where 
        the header_names are age, name, and date.

 	Returns
 	-------
 	vDataFrame
		self.parent

	See Also
	--------
	vDataFrame.astype : Converts the vColumns to the input type.
		"""
        dtype = get_vertica_type(dtype)
        try:
            if (
                dtype == "array" or str(dtype).startswith("vmap")
            ) and self.category() == "text":
                if dtype == "array":
                    vertica_version(condition=[10, 0, 0])
                query = f"""
                    SELECT 
                        {self.alias} 
                    FROM {self.parent.__genSQL__()} 
                    ORDER BY LENGTH({self.alias}) DESC 
                    LIMIT 1"""
                biggest_str = executeSQL(
                    query, title="getting the biggest string", method="fetchfirstelem",
                )
                biggest_str = biggest_str.strip()
                sep = guess_sep(biggest_str)
                if str(dtype).startswith("vmap"):
                    if len(biggest_str) > 2 and (
                        (biggest_str[0] == "{" and biggest_str[-1] == "}")
                    ):
                        transformation_2 = """MAPJSONEXTRACTOR({} 
                                                    USING PARAMETERS flatten_maps=false)"""
                    else:
                        header_names = ""
                        if len(dtype) > 4 and dtype[:5] == "vmap(" and dtype[-1] == ")":
                            header_names = f", header_names='{dtype[5:-1]}'"
                        transformation_2 = f"""MAPDELIMITEDEXTRACTOR({{}} 
                                                            USING PARAMETERS 
                                                            delimiter='{sep}'
                                                            {header_names})"""
                    dtype = "vmap"
                elif dtype == "array":
                    if biggest_str.replace(" ", "").count(sep + sep) > 0:
                        collection_null_element = ", collection_null_element=''"
                    else:
                        collection_null_element = ""
                    if len(biggest_str) > 2 and (
                        (biggest_str[0] == "(" and biggest_str[-1] == ")")
                        or (biggest_str[0] == "{" and biggest_str[-1] == "}")
                    ):
                        collection_open = f", collection_open='{biggest_str[0]}'"
                        collection_close = f", collection_close='{biggest_str[-1]}'"
                    else:
                        collection_open, collection_close = "", ""
                    transformation_2 = f"""
                        STRING_TO_ARRAY({{}} 
                                        USING PARAMETERS 
                                        collection_delimiter='{sep}'
                                        {collection_open}
                                        {collection_close}
                                        {collection_null_element})"""
            elif (
                dtype[0:7] == "varchar" or dtype[0:4] == "char"
            ) and self.category() == "vmap":
                transformation_2 = f"""MAPTOSTRING({{}} 
                                                   USING PARAMETERS 
                                                   canonical_json=false)::{dtype}"""
            elif dtype == "json":
                if self.category() == "vmap":
                    transformation_2 = (
                        "MAPTOSTRING({} USING PARAMETERS canonical_json=true)"
                    )
                else:
                    vertica_version(condition=[10, 1, 0])
                    transformation_2 = "TO_JSON({})"
                dtype = "varchar"
            else:
                transformation_2 = f"{{}}::{dtype}"
            transformation_2 = clean_query(transformation_2)
            transformation = (transformation_2.format(self.alias), transformation_2)
            query = f"""
                SELECT 
                    /*+LABEL('vColumn.astype')*/ 
                    {transformation[0]} AS {self.alias} 
                FROM {self.parent.__genSQL__()} 
                WHERE {self.alias} IS NOT NULL 
                LIMIT 20"""
            executeSQL(
                query,
                title="Testing the Type casting.",
                sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
                symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
            )
            self.transformations += [
                (transformation[1], dtype, get_category_from_vertica_type(ctype=dtype),)
            ]
            self.parent.__add_to_history__(
                f"[AsType]: The vColumn {self.alias} was converted to {dtype}."
            )
            return self.parent
        except Exception as e:
            raise ConversionError(
                f"{e}\nThe vColumn {self.alias} can not be converted to {dtype}"
            )

    # ---#
    @save_verticapy_logs
    def avg(self):
        """
	----------------------------------------------------------------------------------------
	Aggregates the vColumn using 'avg' (Average).

 	Returns
 	-------
 	float
 		average

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
        return self.aggregate(["avg"]).values[self.alias][0]

    mean = avg
    # ---#
    @check_dtypes
    @save_verticapy_logs
    def bar(
        self,
        method: str = "density",
        of: str = "",
        max_cardinality: int = 6,
        nbins: int = 0,
        h: Union[int, float] = 0,
        ax=None,
        **style_kwds,
    ):
        """
	----------------------------------------------------------------------------------------
	Draws the bar chart of the vColumn based on an aggregation.

	Parameters
 	----------
 	method: str, optional
 		The method to use to aggregate the data.
 			count   : Number of elements.
 			density : Percentage of the distribution.
 			mean    : Average of the vColumn 'of'.
 			min     : Minimum of the vColumn 'of'.
 			max     : Maximum of the vColumn 'of'.
 			sum     : Sum of the vColumn 'of'.
 			q%      : q Quantile of the vColumn 'of' (ex: 50% to get the median).
        It can also be a cutomized aggregation (ex: AVG(column1) + 5).
 	of: str, optional
 		The vColumn to use to compute the aggregation.
	max_cardinality: int, optional
 		Maximum number of the vColumn distinct elements to be used as categorical 
 		(No h will be picked or computed)
 	nbins: int, optional
 		Number of nbins. If empty, an optimized number of nbins will be computed.
 	h: int / float, optional
 		Interval width of the bar. If empty, an optimized h will be computed.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object

 	See Also
 	--------
 	vDataFrame[].hist : Draws the histogram of the vColumn based on an aggregation.
		"""
        of = self.parent.format_colnames(of)
        return plt.bar(self, method, of, max_cardinality, nbins, h, ax=ax, **style_kwds)

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def boxplot(
        self,
        by: str = "",
        h: Union[int, float] = 0,
        max_cardinality: int = 8,
        cat_priority: Union[str, int, datetime.datetime, datetime.date, list] = [],
        ax=None,
        **style_kwds,
    ):
        """
	----------------------------------------------------------------------------------------
	Draws the box plot of the vColumn.

	Parameters
 	----------
 	by: str, optional
 		vColumn to use to partition the data.
 	h: int / float, optional
 		Interval width if the vColumn is numerical or of type date like. Optimized 
 		h will be computed if the parameter is empty or invalid.
 	max_cardinality: int, optional
 		Maximum number of vColumn distinct elements to be used as categorical. 
 		The less frequent elements will be gathered together to create a new 
 		category : 'Others'.
 	cat_priority: str / int / date / list, optional
 		List of the different categories to consider when drawing the box plot.
 		The other categories will be filtered.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object

 	See Also
 	--------
 	vDataFrame.boxplot : Draws the Box Plot of the input vColumns. 
		"""
        if isinstance(cat_priority, str) or not (isinstance(cat_priority, Iterable)):
            cat_priority = [cat_priority]
        by = self.parent.format_colnames(by)
        return plt.boxplot(
            self, by, h, max_cardinality, cat_priority, ax=ax, **style_kwds
        )

    # ---#
    def category(self):
        """
	----------------------------------------------------------------------------------------
	Returns the category of the vColumn. The category will be one of the following:
	date / int / float / text / binary / spatial / uuid / undefined

 	Returns
 	-------
 	str
 		vColumn category.

	See Also
	--------
	vDataFrame[].ctype : Returns the vColumn database type.
		"""
        return self.transformations[-1][2]

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def clip(
        self,
        lower: Union[int, float, datetime.datetime, datetime.date] = None,
        upper: Union[int, float, datetime.datetime, datetime.date] = None,
    ):
        """
	----------------------------------------------------------------------------------------
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

    # ---#
    @save_verticapy_logs
    def count(self):
        """
	----------------------------------------------------------------------------------------
	Aggregates the vColumn using 'count' (Number of non-Missing elements).

 	Returns
 	-------
 	int
 		number of non-Missing elements.

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
        return self.aggregate(["count"]).values[self.alias][0]

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def cut(
        self,
        breaks: list,
        labels: list = [],
        include_lowest: bool = True,
        right: bool = True,
    ):
        """
    ----------------------------------------------------------------------------------------
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

    # ---#
    def ctype(self):
        """
	----------------------------------------------------------------------------------------
	Returns the vColumn DB type.

 	Returns
 	-------
 	str
 		vColumn DB type.
		"""
        return self.transformations[-1][1].lower()

    dtype = ctype
    # ---#
    @save_verticapy_logs
    def date_part(self, field: str):
        """
	----------------------------------------------------------------------------------------
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

    # ---#
    @save_verticapy_logs
    def decode(self, *argv):
        """
	----------------------------------------------------------------------------------------
	Encodes the vColumn using a user-defined encoding.

	Parameters
 	----------
 	argv: object
        Any amount of expressions.
        The expression generated will look like:
        even: CASE ... WHEN vColumn = argv[2 * i] THEN argv[2 * i + 1] ... END
        odd : CASE ... WHEN vColumn = argv[2 * i] THEN argv[2 * i + 1] ... ELSE argv[n] END

 	Returns
 	-------
 	vDataFrame
 		self.parent

	See Also
	--------
	vDataFrame.case_when      : Creates a new feature by evaluating some conditions.
	vDataFrame[].discretize   : Discretizes the vColumn.
	vDataFrame[].label_encode : Encodes the vColumn with Label Encoding.
	vDataFrame[].get_dummies  : Encodes the vColumn with One-Hot Encoding.
	vDataFrame[].mean_encode  : Encodes the vColumn using the mean encoding of a response.
		"""
        return self.apply(func=st.decode(str_sql("{}"), *argv))

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def density(
        self,
        by: str = "",
        bandwidth: Union[int, float] = 1.0,
        kernel: str = "gaussian",
        nbins: int = 200,
        xlim: tuple = None,
        ax=None,
        **style_kwds,
    ):
        """
	----------------------------------------------------------------------------------------
	Draws the vColumn Density Plot.

	Parameters
 	----------
    by: str, optional
        vColumn to use to partition the data.
 	bandwidth: int / float, optional
 		The bandwidth of the kernel.
 	kernel: str, optional
 		The method used for the plot.
 			gaussian  : Gaussian kernel.
 			logistic  : Logistic kernel.
 			sigmoid   : Sigmoid kernel.
 			silverman : Silverman kernel.
 	nbins: int, optional
        Maximum number of points to use to evaluate the approximate density function.
        Increasing this parameter will increase the precision but will also increase 
        the time of the learning and scoring phases.
    xlim: tuple, optional
        Set the x limits of the current axes.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object

	See Also
	--------
	vDataFrame[].hist : Draws the histogram of the vColumn based on an aggregation.
		"""
        from matplotlib.lines import Line2D

        raise_error_if_not_in(
            "kernel", kernel, ["gaussian", "logistic", "sigmoid", "silverman"]
        )
        if by:
            by = self.parent.format_colnames(by)
            colors = plt.gen_colors()
            if not xlim:
                xmin = self.min()
                xmax = self.max()
            else:
                xmin, xmax = xlim
            custom_lines = []
            columns = self.parent[by].distinct()
            for idx, column in enumerate(columns):
                param = {"color": colors[idx % len(colors)]}
                ax = self.parent.search(f"{self.parent[by].alias} = '{column}'")[
                    self.alias
                ].density(
                    bandwidth=bandwidth,
                    kernel=kernel,
                    nbins=nbins,
                    xlim=(xmin, xmax),
                    ax=ax,
                    **updated_dict(param, style_kwds, idx),
                )
                custom_lines += [
                    Line2D(
                        [0],
                        [0],
                        color=updated_dict(param, style_kwds, idx)["color"],
                        lw=4,
                    ),
                ]
            ax.set_title("KernelDensity")
            ax.legend(
                custom_lines,
                columns,
                title=by,
                loc="center left",
                bbox_to_anchor=[1, 0.5],
            )
            ax.set_xlabel(self.alias)
            return ax
        kernel = kernel.lower()
        schema = vp.OPTIONS["temp_schema"]
        if not (schema):
            schema = "public"
        name = gen_tmp_name(schema=schema, name="kde")
        if isinstance(xlim, (tuple, list)):
            xlim_tmp = [xlim]
        else:
            xlim_tmp = []
        model = vpy_neighbors.KernelDensity(
            name,
            bandwidth=bandwidth,
            kernel=kernel,
            nbins=nbins,
            xlim=xlim_tmp,
            store=False,
        )
        try:
            result = model.fit(self.parent.__genSQL__(), [self.alias]).plot(
                ax=ax, **style_kwds
            )
            return result
        finally:
            model.drop()

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def describe(
        self, method: str = "auto", max_cardinality: int = 6, numcol: str = ""
    ):
        """
	----------------------------------------------------------------------------------------
	Aggregates the vColumn using multiple statistical aggregations: 
	min, max, median, unique... depending on the input method.

	Parameters
 	----------
 	method: str, optional
 		The describe method.
 			auto 	    : Sets the method to 'numerical' if the vColumn is numerical
 				, 'categorical' otherwise.
			categorical : Uses only categorical aggregations during the computation.
			cat_stats   : Computes statistics of a numerical column for each vColumn
				category. In this case, the parameter 'numcol' must be defined.
 			numerical   : Uses popular numerical aggregations during the computation.
 	max_cardinality: int, optional
 		Cardinality threshold to use to determine if the vColumn will be considered
 		as categorical.
 	numcol: str, optional
 		Numerical vColumn to use when the parameter method is set to 'cat_stats'.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, see
 		utilities.tablesample.

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
        raise_error_if_not_in(
            "method", method, ["auto", "numerical", "categorical", "cat_stats"]
        )
        assert (method != "cat_stats") or (numcol), ParameterError(
            "The parameter 'numcol' must be a vDataFrame column if the method is 'cat_stats'"
        )
        distinct_count, is_numeric, is_date = (
            self.nunique(),
            self.isnum(),
            self.isdate(),
        )
        if (is_date) and not (method == "categorical"):
            result = self.aggregate(["count", "min", "max"])
            index = result.values["index"]
            result = result.values[self.alias]
        elif (method == "cat_stats") and (numcol != ""):
            numcol = self.parent.format_colnames(numcol)
            assert self.parent[numcol].category() in ("float", "int"), TypeError(
                "The column 'numcol' must be numerical"
            )
            cast = "::int" if (self.parent[numcol].isbool()) else ""
            query, cat = [], self.distinct()
            if len(cat) == 1:
                lp, rp = "(", ")"
            else:
                lp, rp = "", ""
            for category in cat:
                tmp_query = f"""
                    SELECT 
                        '{category}' AS 'index', 
                        COUNT({self.alias}) AS count, 
                        100 * COUNT({self.alias}) / {self.parent.shape()[0]} AS percent, 
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
                    tmp_query += f" WHERE {self.alias} IS NULL"
                else:
                    alias_sql_repr = bin_spatial_to_str(self.category(), self.alias)
                    tmp_query += f" WHERE {alias_sql_repr} = '{category}'"
                query += [lp + tmp_query + rp]
            values = to_tablesample(
                query=f"""
                    WITH vdf_table AS 
                        (SELECT 
                            * 
                        FROM {self.parent.__genSQL__()}) 
                        {' UNION ALL '.join(query)}""",
                title=f"Describes the statics of {numcol} partitioned by {self.alias}.",
                sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
                symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
            ).values
        elif (
            ((distinct_count < max_cardinality + 1) and (method != "numerical"))
            or not (is_numeric)
            or (method == "categorical")
        ):
            query = f"""(SELECT 
                            {self.alias} || '', 
                            COUNT(*) 
                        FROM vdf_table 
                        GROUP BY {self.alias} 
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
                         WHERE {self.alias} IS NOT NULL 
                         GROUP BY {self.alias} 
                         ORDER BY COUNT(*) DESC 
                         OFFSET {max_cardinality + 1}) VERTICAPY_SUBTABLE) 
                     ORDER BY count DESC"""
            query_result = executeSQL(
                query=f"""
                    WITH vdf_table AS 
                        (SELECT 
                            /*+LABEL('vColumn.describe')*/ * 
                         FROM {self.parent.__genSQL__()}) {query}""",
                title=f"Computing the descriptive statistics of {self.alias}.",
                method="fetchall",
                sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
                symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
            )
            result = [distinct_count, self.count()] + [item[1] for item in query_result]
            index = ["unique", "count"] + [item[0] for item in query_result]
        else:
            result = (
                self.parent.describe(
                    method="numerical", columns=[self.alias], unique=False
                )
                .transpose()
                .values[self.alias]
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
                "value": [self.alias, self.ctype()] + result,
            }
            if ((is_date) and not (method == "categorical")) or (
                method == "is_numeric"
            ):
                self.parent.__update_catalog__({"index": index, self.alias: result})
        for elem in values:
            for i in range(len(values[elem])):
                if isinstance(values[elem][i], decimal.Decimal):
                    values[elem][i] = float(values[elem][i])
        return tablesample(values)

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def discretize(
        self,
        method: str = "auto",
        h: Union[int, float] = 0,
        nbins: int = -1,
        k: int = 6,
        new_category: str = "Others",
        RFmodel_params: dict = {},
        response: str = "",
        return_enum_trans: bool = False,
    ):
        """
	----------------------------------------------------------------------------------------
	Discretizes the vColumn using the input method.

	Parameters
 	----------
 	method: str, optional
 		The method to use to discretize the vColumn.
 			auto 	   : Uses method 'same_width' for numerical vColumns, cast 
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
        raise_error_if_not_in(
            "method", method, ["auto", "smart", "same_width", "same_freq", "topk"]
        )
        if self.isnum() and method == "smart":
            schema = vp.OPTIONS["temp_schema"]
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
                model = vpy_ensemble.RandomForestRegressor(tmp_model_name)
            else:
                model = vpy_ensemble.RandomForestClassifier(tmp_model_name)
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
                result = executeSQL(
                    query=query,
                    title="Computing the optimized histogram nbins using Random Forest.",
                    method="fetchall",
                    sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
                    symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
                )
                result = [elem[0] for elem in result]
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
            trans = (
                "(CASE WHEN {} IN ({}) THEN {} || '' ELSE '{}' END)".format(
                    bin_spatial_to_str(self.category()),
                    ", ".join(
                        [
                            "'{}'".format(str(elem).replace("'", "''"))
                            for elem in distinct
                        ]
                    ),
                    bin_spatial_to_str(self.category()),
                    new_category.replace("'", "''"),
                ),
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
            result = executeSQL(
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

    # ---#
    def distinct(self, **kwargs):
        """
	----------------------------------------------------------------------------------------
	Returns the distinct categories of the vColumn.

 	Returns
 	-------
 	list
 		Distinct caterogies of the vColumn.

	See Also
	--------
	vDataFrame.topk : Returns the vColumn most occurent elements.
		"""
        alias_sql_repr = bin_spatial_to_str(self.category(), self.alias)
        if "agg" not in kwargs:
            query = f"""
                SELECT 
                    /*+LABEL('vColumn.distinct')*/ 
                    {alias_sql_repr} AS {self.alias} 
                FROM {self.parent.__genSQL__()} 
                WHERE {self.alias} IS NOT NULL 
                GROUP BY {self.alias} 
                ORDER BY {self.alias}"""
        else:
            query = f"""
                SELECT 
                    /*+LABEL('vColumn.distinct')*/ {self.alias} 
                FROM 
                    (SELECT 
                        {alias_sql_repr} AS {self.alias}, 
                        {kwargs['agg']} AS verticapy_agg 
                     FROM {self.parent.__genSQL__()} 
                     WHERE {self.alias} IS NOT NULL 
                     GROUP BY 1) x 
                ORDER BY verticapy_agg DESC"""
        query_result = executeSQL(
            query=query,
            title=f"Computing the distinct categories of {self.alias}.",
            method="fetchall",
            sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
            symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
        )
        return [item for sublist in query_result for item in sublist]

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def div(self, x: Union[int, float]):
        """
	----------------------------------------------------------------------------------------
	Divides the vColumn by the input element.

	Parameters
 	----------
 	x: int / float
 		Input number.

 	Returns
 	-------
 	vDataFrame
		self.parent

	See Also
	--------
	vDataFrame[].apply : Applies a function to the input vColumn.
		"""
        assert x != 0, ValueError("Division by 0 is forbidden !")
        return self.apply(func=f"{{}} / ({x})")

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def drop(self, add_history: bool = True):
        """
	----------------------------------------------------------------------------------------
	Drops the vColumn from the vDataFrame. Dropping a vColumn means simply
    not selecting it in the final generated SQL code.
    
    Note: Dropping a vColumn can make the vDataFrame "heavier" if it is used
    to compute other vColumns.

	Parameters
 	----------
 	add_history: bool, optional
 		If set to True, the information will be stored in the vDataFrame history.

 	Returns
 	-------
 	vDataFrame
		self.parent

	See Also
	--------
	vDataFrame.drop: Drops the input vColumns from the vDataFrame.
		"""
        try:
            parent = self.parent
            force_columns = [
                column for column in self.parent._VERTICAPY_VARIABLES_["columns"]
            ]
            force_columns.remove(self.alias)
            executeSQL(
                query=f"""
                    SELECT 
                        /*+LABEL('vColumn.drop')*/ * 
                    FROM {self.parent.__genSQL__(force_columns=force_columns)} 
                    LIMIT 10""",
                print_time_sql=False,
                sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
                symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
            )
            self.parent._VERTICAPY_VARIABLES_["columns"].remove(self.alias)
            delattr(self.parent, self.alias)
        except:
            self.parent._VERTICAPY_VARIABLES_["exclude_columns"] += [self.alias]
        if add_history:
            self.parent.__add_to_history__(
                f"[Drop]: vColumn {self.alias} was deleted from the vDataFrame."
            )
        return parent

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def drop_outliers(
        self,
        threshold: Union[int, float] = 4.0,
        use_threshold: bool = True,
        alpha: Union[int, float] = 0.05,
    ):
        """
	----------------------------------------------------------------------------------------
	Drops outliers in the vColumn.

	Parameters
 	----------
 	threshold: int / float, optional
 		Uses the Gaussian distribution to identify outliers. After normalizing 
 		the data (Z-Score), if the absolute value of the record is greater than 
 		the threshold, it will be considered as an outlier.
 	use_threshold: bool, optional
 		Uses the threshold instead of the 'alpha' parameter.
 	alpha: int / float, optional
 		Number representing the outliers threshold. Values lesser than 
 		quantile(alpha) or greater than quantile(1-alpha) will be dropped.

 	Returns
 	-------
 	vDataFrame
		self.parent

	See Also
	--------
	vDataFrame.fill_outliers : Fills the outliers in the vColumn.
	vDataFrame.outliers      : Adds a new vColumn labeled with 0 and 1 
		(1 meaning global outlier).
		"""
        if use_threshold:
            result = self.aggregate(func=["std", "avg"]).transpose().values
            self.parent.filter(
                f"""
                    ABS({self.alias} - {result["avg"][0]}) 
                  / {result["std"][0]} < {threshold}"""
            )
        else:
            p_alpha, p_1_alpha = (
                self.parent.quantile([alpha, 1 - alpha], [self.alias])
                .transpose()
                .values[self.alias]
            )
            self.parent.filter(f"({self.alias} BETWEEN {p_alpha} AND {p_1_alpha})")
        return self.parent

    # ---#
    @save_verticapy_logs
    def dropna(self):
        """
	----------------------------------------------------------------------------------------
	Filters the vDataFrame where the vColumn is missing.

 	Returns
 	-------
 	vDataFrame
		self.parent

 	See Also
	--------
	vDataFrame.filter: Filters the data using the input expression.
		"""
        self.parent.filter(f"{self.alias} IS NOT NULL")
        return self.parent

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def fill_outliers(
        self,
        method: str = "winsorize",
        threshold: Union[int, float] = 4.0,
        use_threshold: bool = True,
        alpha: Union[int, float] = 0.05,
    ):
        """
	----------------------------------------------------------------------------------------
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
        raise_error_if_not_in("method", method, ["winsorize", "null", "mean"])
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
            p_alpha, p_1_alpha = executeSQL(
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
                for item in executeSQL(
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

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def fillna(
        self,
        val: Union[int, float, str, datetime.datetime, datetime.date] = None,
        method: str = "auto",
        expr: Union[str, str_sql] = "",
        by: Union[str, list] = [],
        order_by: Union[str, list] = [],
    ):
        """
	----------------------------------------------------------------------------------------
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
        raise_error_if_not_in(
            "method",
            method,
            [
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
            ],
        )
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
                    result = executeSQL(
                        query=query,
                        title="Computing the different aggregations.",
                        method="fetchall",
                        sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
                        symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
                    )
                    for idx, elem in enumerate(result):
                        result[idx][0] = (
                            "NULL"
                            if (elem[0] == None)
                            else "'{}'".format(str(elem[0]).replace("'", "''"))
                        )
                        result[idx][1] = "NULL" if (elem[1] == None) else str(elem[1])
                    new_column = "COALESCE({}, DECODE({}, {}, NULL))".format(
                        "{}",
                        by[0],
                        ", ".join([f"{elem[0]}, {elem[1]}" for elem in result]),
                    )
                    executeSQL(
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
            if vp.OPTIONS["print_info"]:
                print(f"{total} element{conj}filled.")
            self.parent.__add_to_history__(
                f"[Fillna]: {total} {self.alias} missing value{conj} filled."
            )
        else:
            if vp.OPTIONS["print_info"]:
                print("Nothing was filled.")
            self.transformations = [t for t in copy_trans]
            for s in sauv:
                self.catalog[s] = sauv[s]
        return self.parent

    # ---#
    @save_verticapy_logs
    def geo_plot(self, *args, **kwargs):
        """
    ----------------------------------------------------------------------------------------
    Draws the Geospatial object.

    Parameters
    ----------
    *args / **kwargs
        Any optional parameter to pass to the geopandas plot function.
        For more information, see: 
        https://geopandas.readthedocs.io/en/latest/docs/reference/api/
                geopandas.GeoDataFrame.plot.html
    
    Returns
    -------
    ax
        Matplotlib axes object
        """
        columns = [self.alias]
        check = True
        if len(args) > 0:
            column = args[0]
        elif "column" in kwargs:
            column = kwargs["column"]
        else:
            check = False
        if check:
            column = self.parent.format_colnames(column)
            columns += [column]
            if not ("cmap" in kwargs):
                kwargs["cmap"] = plt.gen_cmap()[0]
        else:
            if not ("color" in kwargs):
                kwargs["color"] = plt.gen_colors()[0]
        if not ("legend" in kwargs):
            kwargs["legend"] = True
        if not ("figsize" in kwargs):
            kwargs["figsize"] = (14, 10)
        return self.parent[columns].to_geopandas(self.alias).plot(*args, **kwargs)

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def get_dummies(
        self,
        prefix: str = "",
        prefix_sep: str = "_",
        drop_first: bool = True,
        use_numbers_as_suffix: bool = False,
    ):
        """
	----------------------------------------------------------------------------------------
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

    one_hot_encode = get_dummies

    # ---#
    def get_len(self):
        """
    ----------------------------------------------------------------------------------------
    Returns a new vColumn that represents the length of each element.

    Returns
    -------
    vColumn
        vColumn that includes the length of each element.
        """
        cat = self.category()
        if cat == "vmap":
            fun = "MAPSIZE"
        elif cat == "complex":
            fun = "APPLY_COUNT_ELEMENTS"
        else:
            fun = "LENGTH"
        elem_to_select = f"{fun}({self.alias})"
        init_transf = f"{fun}({self.init_transf})"
        new_alias = quote_ident(self.alias[1:-1] + ".length")
        query = f"""
            (SELECT 
                {elem_to_select} AS {new_alias} 
            FROM {self.parent.__genSQL__()}) VERTICAPY_SUBTABLE"""
        vcol = vDataFrameSQL(query)[new_alias]
        vcol.init_transf = init_transf
        return vcol

    # ---#
    def head(self, limit: int = 5):
        """
	----------------------------------------------------------------------------------------
	Returns the head of the vColumn.

	Parameters
 	----------
 	limit: int, optional
 		Number of elements to display.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, see
 		utilities.tablesample.

	See Also
	--------
	vDataFrame[].tail : Returns the a part of the vColumn.
		"""
        return self.iloc(limit=limit)

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def hist(
        self,
        method: str = "density",
        of: str = "",
        max_cardinality: int = 6,
        nbins: int = 0,
        h: Union[int, float] = 0,
        ax=None,
        **style_kwds,
    ):
        """
	----------------------------------------------------------------------------------------
	Draws the histogram of the vColumn based on an aggregation.

	Parameters
 	----------
 	method: str, optional
 		The method to use to aggregate the data.
 			count   : Number of elements.
 			density : Percentage of the distribution.
 			mean    : Average of the vColumn 'of'.
 			min     : Minimum of the vColumn 'of'.
 			max     : Maximum of the vColumn 'of'.
 			sum     : Sum of the vColumn 'of'.
 			q%      : q Quantile of the vColumn 'of' (ex: 50% to get the median).
        It can also be a cutomized aggregation (ex: AVG(column1) + 5).
 	of: str, optional
 		The vColumn to use to compute the aggregation.
	max_cardinality: int, optional
 		Maximum number of the vColumn distinct elements to be used as categorical 
 		(No h will be picked or computed)
 	nbins: int, optional
 		Number of bins. If empty, an optimized number of bins will be computed.
 	h: int / float, optional
 		Interval width of the bar. If empty, an optimized h will be computed.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object

 	See Also
 	--------
 	vDataFrame[].bar : Draws the Bar Chart of vColumn based on an aggregation.
		"""
        of = self.parent.format_colnames(of)
        return plt.hist(
            self, method, of, max_cardinality, nbins, h, ax=ax, **style_kwds
        )

    # ---#
    @check_dtypes
    def iloc(self, limit: int = 5, offset: int = 0):
        """
    ----------------------------------------------------------------------------------------
    Returns a part of the vColumn (delimited by an offset and a limit).

    Parameters
    ----------
    limit: int, optional
        Number of elements to display.
    offset: int, optional
        Number of elements to skip.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame[].head : Returns the head of the vColumn.
    vDataFrame[].tail : Returns the tail of the vColumn.
        """
        if offset < 0:
            offset = max(0, self.parent.shape()[0] - limit)
        title = f"Reads {self.alias}."
        alias_sql_repr = bin_spatial_to_str(self.category(), self.alias)
        tail = to_tablesample(
            query=f"""
                SELECT 
                    {alias_sql_repr} AS {self.alias} 
                FROM {self.parent.__genSQL__()}
                {self.parent.__get_last_order_by__()} 
                LIMIT {limit} 
                OFFSET {offset}""",
            title=title,
            sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
            symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
        )
        tail.count = self.parent.shape()[0]
        tail.offset = offset
        tail.dtype[self.alias] = self.ctype()
        tail.name = self.alias
        return tail

    # ---#
    def isarray(self):
        """
    ----------------------------------------------------------------------------------------
    Returns True if the vColumn is an array, False otherwise.

    Returns
    -------
    bool
        True if the vColumn is an array.
        """
        return self.ctype()[0:5].lower() == "array"

    # ---#
    def isbool(self):
        """
    ----------------------------------------------------------------------------------------
    Returns True if the vColumn is boolean, False otherwise.

    Returns
    -------
    bool
        True if the vColumn is boolean.

    See Also
    --------
    vDataFrame[].isdate : Returns True if the vColumn category is date.
    vDataFrame[].isnum  : Returns True if the vColumn is numerical.
        """
        return self.ctype()[0:4] == "bool"

    # ---#
    def isdate(self):
        """
	----------------------------------------------------------------------------------------
	Returns True if the vColumn category is date, False otherwise.

 	Returns
 	-------
 	bool
 		True if the vColumn category is date.

	See Also
	--------
    vDataFrame[].isbool : Returns True if the vColumn is boolean.
	vDataFrame[].isnum  : Returns True if the vColumn is numerical.
		"""
        return self.category() == "date"

    # ---#
    @save_verticapy_logs
    def isin(
        self,
        val: Union[str, int, float, datetime.datetime, datetime.date, list],
        *args,
    ):
        """
	----------------------------------------------------------------------------------------
	Looks if some specific records are in the vColumn and it returns the new 
    vDataFrame of the search.

	Parameters
 	----------
 	val: str / int / float / date / list
 		List of the different records. For example, to check if Badr and Fouad  
 		are in the vColumn. You can write the following list: ["Fouad", "Badr"]

 	Returns
 	-------
 	vDataFrame
 		The vDataFrame of the search.

 	See Also
 	--------
 	vDataFrame.isin : Looks if some specific records are in the vDataFrame.
		"""
        if isinstance(val, str) or not (isinstance(val, Iterable)):
            val = [val]
        val += list(args)
        val = {self.alias: val}
        return self.parent.isin(val)

    # ---#
    def isnum(self):
        """
	----------------------------------------------------------------------------------------
	Returns True if the vColumn is numerical, False otherwise.

 	Returns
 	-------
 	bool
 		True if the vColumn is numerical.

	See Also
	--------
    vDataFrame[].isbool : Returns True if the vColumn is boolean.
	vDataFrame[].isdate : Returns True if the vColumn category is date.
		"""
        return self.category() in ("float", "int")

    # ---#
    def isvmap(self):
        """
    ----------------------------------------------------------------------------------------
    Returns True if the vColumn category is VMap, False otherwise.

    Returns
    -------
    bool
        True if the vColumn category is VMap.
        """
        return self.category() == "vmap" or isvmap(
            column=self.alias, expr=self.parent.__genSQL__()
        )

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def iv_woe(self, y: str, nbins: int = 10):
        """
    ----------------------------------------------------------------------------------------
    Computes the Information Value (IV) / Weight Of Evidence (WOE) Table. It tells 
    the predictive power of an independent variable in relation to the dependent 
    variable.

    Parameters
    ----------
    y: str
        Response vColumn.
    nbins: int, optional
        Maximum number of nbins used for the discretization (must be > 1)

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.iv_woe : Computes the Information Value (IV) Table.
        """
        y = self.parent.format_colnames(y)
        assert self.parent[y].nunique() == 2, TypeError(
            f"vColumn {y} must be binary to use iv_woe."
        )
        response_cat = self.parent[y].distinct()
        response_cat.sort()
        assert response_cat == [0, 1], TypeError(
            f"vColumn {y} must be binary to use iv_woe."
        )
        self.parent[y].distinct()
        trans = self.discretize(
            method="same_width" if self.isnum() else "topk",
            nbins=nbins,
            k=nbins,
            new_category="Others",
            return_enum_trans=True,
        )[0].replace("{}", self.alias)
        query = f"""
            SELECT 
                {trans} AS {self.alias}, 
                {self.alias} AS ord, 
                {y}::int AS {y} 
            FROM {self.parent.__genSQL__()}"""
        query = f"""
            SELECT 
                {self.alias}, 
                MIN(ord) AS ord, 
                SUM(1 - {y}) AS non_events, 
                SUM({y}) AS events 
            FROM ({query}) x GROUP BY 1"""
        query = f"""
            SELECT 
                {self.alias}, 
                ord, 
                non_events, 
                events, 
                non_events / NULLIFZERO(SUM(non_events) OVER ()) AS pt_non_events, 
                events / NULLIFZERO(SUM(events) OVER ()) AS pt_events 
            FROM ({query}) x"""
        query = f"""
            SELECT 
                {self.alias} AS index, 
                non_events, 
                events, 
                pt_non_events, 
                pt_events, 
                CASE 
                    WHEN non_events = 0 OR events = 0 THEN 0 
                    ELSE ZEROIFNULL(LN(pt_non_events / NULLIFZERO(pt_events))) 
                END AS woe, 
                CASE 
                    WHEN non_events = 0 OR events = 0 THEN 0 
                    ELSE (pt_non_events - pt_events) 
                        * ZEROIFNULL(LN(pt_non_events 
                        / NULLIFZERO(pt_events))) 
                END AS iv 
            FROM ({query}) x ORDER BY ord"""
        title = f"Computing WOE & IV of {self.alias} (response = {y})."
        result = to_tablesample(
            query,
            title=title,
            sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
            symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
        )
        result.values["index"] += ["total"]
        result.values["non_events"] += [sum(result["non_events"])]
        result.values["events"] += [sum(result["events"])]
        result.values["pt_non_events"] += [""]
        result.values["pt_events"] += [""]
        result.values["woe"] += [""]
        result.values["iv"] += [sum(result["iv"])]
        return result

    # ---#
    @save_verticapy_logs
    def kurtosis(self):
        """
	----------------------------------------------------------------------------------------
	Aggregates the vColumn using 'kurtosis'.

 	Returns
 	-------
 	float
 		kurtosis

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
        return self.aggregate(["kurtosis"]).values[self.alias][0]

    kurt = kurtosis
    # ---#
    @save_verticapy_logs
    def label_encode(self):
        """
	----------------------------------------------------------------------------------------
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

    # ---#
    @save_verticapy_logs
    def mad(self):
        """
	----------------------------------------------------------------------------------------
	Aggregates the vColumn using 'mad' (median absolute deviation).

 	Returns
 	-------
 	float
 		mad

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
        return self.aggregate(["mad"]).values[self.alias][0]

    # ---#
    @save_verticapy_logs
    def max(self):
        """
	----------------------------------------------------------------------------------------
	Aggregates the vColumn using 'max' (Maximum).

 	Returns
 	-------
 	float/str
 		maximum

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
        return self.aggregate(["max"]).values[self.alias][0]

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def mean_encode(self, response: str):
        """
	----------------------------------------------------------------------------------------
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
        if vp.OPTIONS["print_info"]:
            print("The mean encoding was successfully done.")
        return self.parent

    # ---#
    @save_verticapy_logs
    def median(
        self, approx: bool = True,
    ):
        """
	----------------------------------------------------------------------------------------
	Aggregates the vColumn using 'median'.

    Parameters
    ----------
    approx: bool, optional
        If set to True, the approximate median is returned. By setting this 
        parameter to False, the function's performance can drastically decrease.

 	Returns
 	-------
 	float/str
 		median

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
        return self.quantile(0.5, approx=approx)

    # ---#
    @save_verticapy_logs
    def memory_usage(self):
        """
	----------------------------------------------------------------------------------------
	Returns the vColumn memory usage. 

 	Returns
 	-------
 	float
 		vColumn memory usage (byte)

	See Also
	--------
	vDataFrame.memory_usage : Returns the vDataFrame memory usage.
		"""
        total = (
            sys.getsizeof(self)
            + sys.getsizeof(self.alias)
            + sys.getsizeof(self.transformations)
            + sys.getsizeof(self.catalog)
        )
        for elem in self.catalog:
            total += sys.getsizeof(elem)
        return total

    # ---#
    @save_verticapy_logs
    def min(self):
        """
	----------------------------------------------------------------------------------------
	Aggregates the vColumn using 'min' (Minimum).

 	Returns
 	-------
 	float/str
 		minimum

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
        return self.aggregate(["min"]).values[self.alias][0]

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def mode(self, dropna: bool = False, n: int = 1):
        """
	----------------------------------------------------------------------------------------
	Returns the nth most occurent element.

	Parameters
 	----------
 	dropna: bool, optional
 		If set to True, NULL values will not be considered during the computation.
 	n: int, optional
 		Integer corresponding to the offset. For example, if n = 1 then this
 		method will return the mode of the vColumn.

 	Returns
 	-------
 	str/float/int
 		vColumn nth most occurent element.

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
        if n == 1:
            pre_comp = self.parent.__get_catalog_value__(self.alias, "top")
            if pre_comp != "VERTICAPY_NOT_PRECOMPUTED":
                if not (dropna) and (pre_comp != None):
                    return pre_comp
        assert n >= 1, ParameterError("Parameter 'n' must be greater or equal to 1")
        where = f" WHERE {self.alias} IS NOT NULL " if (dropna) else " "
        result = executeSQL(
            f"""
            SELECT 
                /*+LABEL('vColumn.mode')*/ {self.alias} 
            FROM (
                SELECT 
                    {self.alias}, 
                    COUNT(*) AS _verticapy_cnt_ 
                FROM {self.parent.__genSQL__()}
                {where}GROUP BY {self.alias} 
                ORDER BY _verticapy_cnt_ DESC 
                LIMIT {n}) VERTICAPY_SUBTABLE 
                ORDER BY _verticapy_cnt_ ASC 
                LIMIT 1""",
            title="Computing the mode.",
            method="fetchall",
            sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
            symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
        )
        top = None if not (result) else result[0][0]
        if not (dropna):
            n = "" if (n == 1) else str(int(n))
            if isinstance(top, decimal.Decimal):
                top = float(top)
            self.parent.__update_catalog__({"index": [f"top{n}"], self.alias: [top]})
        return top

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def mul(self, x: Union[int, float]):
        """
	----------------------------------------------------------------------------------------
	Multiplies the vColumn by the input element.

	Parameters
 	----------
 	x: int / float
 		Input number.

 	Returns
 	-------
 	vDataFrame
		self.parent

	See Also
	--------
	vDataFrame[].apply : Applies a function to the input vColumn.
		"""
        return self.apply(func=f"{{}} * ({x})")

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def nlargest(self, n: int = 10):
        """
	----------------------------------------------------------------------------------------
	Returns the n largest vColumn elements.

	Parameters
 	----------
 	n: int, optional
 		Offset.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, see
 		utilities.tablesample.

	See Also
	--------
	vDataFrame[].nsmallest : Returns the n smallest elements in the vColumn.
		"""
        query = f"""
            SELECT 
                * 
            FROM {self.parent.__genSQL__()} 
            WHERE {self.alias} IS NOT NULL 
            ORDER BY {self.alias} DESC LIMIT {n}"""
        title = f"Reads {self.alias} {n} largest elements."
        return to_tablesample(
            query,
            title=title,
            sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
            symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
        )

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def normalize(
        self,
        method: str = "zscore",
        by: Union[str, list] = [],
        return_trans: bool = False,
    ):
        """
	----------------------------------------------------------------------------------------
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
        raise_error_if_not_in("method", method, ["zscore", "robust_zscore", "minmax"])
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
                        result = executeSQL(
                            f"""
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
                        avg = "DECODE({}, {}, NULL)".format(
                            by[0],
                            ", ".join(
                                [
                                    "{}, {}".format(
                                        "'{}'".format(str(elem[0]).replace("'", "''"))
                                        if elem[0] != None
                                        else "NULL",
                                        elem[1] if elem[1] != None else "NULL",
                                    )
                                    for elem in result
                                    if elem[1] != None
                                ]
                            ),
                        )
                        stddev = "DECODE({}, {}, NULL)".format(
                            by[0],
                            ", ".join(
                                [
                                    "{}, {}".format(
                                        "'{}'".format(str(elem[0]).replace("'", "''"))
                                        if elem[0] != None
                                        else "NULL",
                                        elem[2] if elem[2] != None else "NULL",
                                    )
                                    for elem in result
                                    if elem[2] != None
                                ]
                            ),
                        )
                        executeSQL(
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
                        result = executeSQL(
                            f"""
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
                        cmin = "DECODE({}, {}, NULL)".format(
                            by[0],
                            ", ".join(
                                [
                                    "{}, {}".format(
                                        "'{}'".format(str(elem[0]).replace("'", "''"))
                                        if elem[0] != None
                                        else "NULL",
                                        elem[1] if elem[1] != None else "NULL",
                                    )
                                    for elem in result
                                    if elem[1] != None
                                ]
                            ),
                        )
                        cmax = "DECODE({}, {}, NULL)".format(
                            by[0],
                            ", ".join(
                                [
                                    "{}, {}".format(
                                        "'{}'".format(str(elem[0]).replace("'", "''"))
                                        if elem[0] != None
                                        else "NULL",
                                        elem[2] if elem[2] != None else "NULL",
                                    )
                                    for elem in result
                                    if elem[2] != None
                                ]
                            ),
                        )
                        executeSQL(
                            query=f"""
                                SELECT 
                                    /*+LABEL('vColumn.normalize')*/ 
                                    {cmax}, 
                                    {cmin} 
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

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def nsmallest(self, n: int = 10):
        """
	----------------------------------------------------------------------------------------
	Returns the n smallest elements in the vColumn.

	Parameters
 	----------
 	n: int, optional
 		Offset.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, see
 		utilities.tablesample.

	See Also
	--------
	vDataFrame[].nlargest : Returns the n largest vColumn elements.
		"""
        return to_tablesample(
            f"""
            SELECT 
                * 
            FROM {self.parent.__genSQL__()} 
            WHERE {self.alias} IS NOT NULL 
            ORDER BY {self.alias} ASC LIMIT {n}""",
            title=f"Reads {n} {self.alias} smallest elements.",
            sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
            symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
        )

    # ---#
    @check_dtypes
    def numh(self, method: str = "auto"):
        """
	----------------------------------------------------------------------------------------
	Computes the optimal vColumn bar width.

	Parameters
 	----------
 	method: str, optional
 		Method to use to compute the optimal h.
 			auto              : Combination of Freedman Diaconis and Sturges.
 			freedman_diaconis : Freedman Diaconis [2 * IQR / n ** (1 / 3)]
 			sturges           : Sturges [CEIL(log2(n)) + 1]

 	Returns
 	-------
 	float
 		optimal bar width.
		"""
        raise_error_if_not_in(
            "method", method, ["sturges", "freedman_diaconis", "fd", "auto"]
        )
        if method == "auto":
            pre_comp = self.parent.__get_catalog_value__(self.alias, "numh")
            if pre_comp != "VERTICAPY_NOT_PRECOMPUTED":
                return pre_comp
        assert self.isnum() or self.isdate(), ParameterError(
            "numh is only available on type numeric|date"
        )
        if self.isnum():
            result = (
                self.parent.describe(
                    method="numerical", columns=[self.alias], unique=False
                )
                .transpose()
                .values[self.alias]
            )
            count, vColumn_min, vColumn_025, vColumn_075, vColumn_max = (
                result[0],
                result[3],
                result[4],
                result[6],
                result[7],
            )
        elif self.isdate():
            result = executeSQL(
                f"""
                SELECT 
                    /*+LABEL('vColumn.numh')*/ COUNT({self.alias}) AS NAs, 
                    MIN({self.alias}) AS min, 
                    APPROXIMATE_PERCENTILE({self.alias} 
                        USING PARAMETERS percentile = 0.25) AS Q1, 
                    APPROXIMATE_PERCENTILE({self.alias} 
                        USING PARAMETERS percentile = 0.75) AS Q3, 
                    MAX({self.alias}) AS max 
                FROM 
                    (SELECT 
                        DATEDIFF('second', 
                                 '{self.min()}'::timestamp, 
                                 {self.alias}) AS {self.alias} 
                    FROM {self.parent.__genSQL__()}) VERTICAPY_OPTIMAL_H_TABLE""",
                title="Different aggregations to compute the optimal h.",
                method="fetchrow",
                sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
                symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
            )
            count, vColumn_min, vColumn_025, vColumn_075, vColumn_max = result
        sturges = max(
            float(vColumn_max - vColumn_min) / int(math.floor(math.log(count, 2) + 2)),
            1e-99,
        )
        fd = max(2.0 * (vColumn_075 - vColumn_025) / (count) ** (1.0 / 3.0), 1e-99)
        if method.lower() == "sturges":
            best_h = sturges
        elif method.lower() in ("freedman_diaconis", "fd"):
            best_h = fd
        else:
            best_h = max(sturges, fd)
            self.parent.__update_catalog__({"index": ["numh"], self.alias: [best_h]})
        if self.category() == "int":
            best_h = max(math.floor(best_h), 1)
        return best_h

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def nunique(self, approx: bool = True):
        """
	----------------------------------------------------------------------------------------
	Aggregates the vColumn using 'unique' (cardinality).

	Parameters
 	----------
 	approx: bool, optional
 		If set to True, the approximate cardinality is returned. By setting 
        this parameter to False, the function's performance can drastically 
        decrease.

 	Returns
 	-------
 	int
 		vColumn cardinality (or approximate cardinality).

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
        if approx:
            return self.aggregate(func=["approx_unique"]).values[self.alias][0]
        else:
            return self.aggregate(func=["unique"]).values[self.alias][0]

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def pie(
        self,
        method: str = "density",
        of: str = "",
        max_cardinality: int = 6,
        h: Union[int, float] = 0,
        pie_type: str = "auto",
        ax=None,
        **style_kwds,
    ):
        """
	----------------------------------------------------------------------------------------
	Draws the pie chart of the vColumn based on an aggregation.

	Parameters
 	----------
 	method: str, optional
 		The method to use to aggregate the data.
 			count   : Number of elements.
 			density : Percentage of the distribution.
 			mean    : Average of the vColumn 'of'.
 			min     : Minimum of the vColumn 'of'.
 			max     : Maximum of the vColumn 'of'.
 			sum     : Sum of the vColumn 'of'.
 			q%      : q Quantile of the vColumn 'of' (ex: 50% to get the median).
        It can also be a cutomized aggregation (ex: AVG(column1) + 5).
 	of: str, optional
 		The vColumn to use to compute the aggregation.
	max_cardinality: int, optional
 		Maximum number of the vColumn distinct elements to be used as categorical 
 		(No h will be picked or computed)
 	h: int / float, optional
 		Interval width of the bar. If empty, an optimized h will be computed.
    pie_type: str, optional
        The type of pie chart.
            auto   : Regular pie chart.
            donut  : Donut chart.
            rose   : Rose chart.
        It can also be a cutomized aggregation (ex: AVG(column1) + 5).
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object

 	See Also
 	--------
 	vDataFrame.donut : Draws the donut chart of the vColumn based on an aggregation.
		"""
        raise_error_if_not_in("pie_type", pie_type, ["auto", "donut", "rose"])
        donut, rose = (pie_type == "donut"), (pie_type == "rose")
        of = self.parent.format_colnames(of)
        return plt.pie(
            self, method, of, max_cardinality, h, donut, rose, ax=None, **style_kwds,
        )

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def plot(
        self,
        ts: str,
        by: str = "",
        start_date: Union[str, int, float, datetime.datetime, datetime.date] = "",
        end_date: Union[str, int, float, datetime.datetime, datetime.date] = "",
        area: bool = False,
        step: bool = False,
        ax=None,
        **style_kwds,
    ):
        """
	----------------------------------------------------------------------------------------
	Draws the Time Series of the vColumn.

	Parameters
 	----------
 	ts: str
 		TS (Time Series) vColumn to use to order the data. The vColumn type must be
 		date like (date, datetime, timestamp...) or numerical.
 	by: str, optional
 		vColumn to use to partition the TS.
 	start_date: str / int / float / date, optional
 		Input Start Date. For example, time = '03-11-1993' will filter the data when 
 		'ts' is lesser than November 1993 the 3rd.
 	end_date: str / int / float / date, optional
 		Input End Date. For example, time = '03-11-1993' will filter the data when 
 		'ts' is greater than November 1993 the 3rd.
 	area: bool, optional
 		If set to True, draw an Area Plot.
    step: bool, optional
        If set to True, draw a Step Plot.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object

	See Also
	--------
	vDataFrame.plot : Draws the time series.
		"""
        ts, by = self.parent.format_colnames(ts, by)
        return plt.ts_plot(
            self, ts, by, start_date, end_date, area, step, ax=ax, **style_kwds,
        )

    # ---#
    @save_verticapy_logs
    def product(self):
        """
	----------------------------------------------------------------------------------------
	Aggregates the vColumn using 'product'.

 	Returns
 	-------
 	float
 		product

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
        return self.aggregate(func=["prod"]).values[self.alias][0]

    prod = product

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def quantile(self, x: Union[int, float], approx: bool = True):
        """
	----------------------------------------------------------------------------------------
	Aggregates the vColumn using an input 'quantile'.

	Parameters
 	----------
 	x: int / float
 		A float between 0 and 1 that represents the quantile.
        For example: 0.25 represents Q1.
    approx: bool, optional
        If set to True, the approximate quantile is returned. By setting this 
        parameter to False, the function's performance can drastically decrease.

 	Returns
 	-------
 	float
 		quantile (or approximate quantile).

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
        prefix = "approx_" if approx else ""
        return self.aggregate(func=[f"{prefix}{x * 100}%"]).values[self.alias][0]

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def range_plot(
        self,
        ts: str,
        q: Union[tuple, list] = (0.25, 0.75),
        start_date: Union[str, int, float, datetime.datetime, datetime.date] = "",
        end_date: Union[str, int, float, datetime.datetime, datetime.date] = "",
        plot_median: bool = False,
        ax=None,
        **style_kwds,
    ):
        """
    ----------------------------------------------------------------------------------------
    Draws the range plot of the vColumn. The aggregations used are the median 
    and two input quantiles.

    Parameters
    ----------
    ts: str
        TS (Time Series) vColumn to use to order the data. The vColumn type must be
        date like (date, datetime, timestamp...) or numerical.
    q: tuple / list, optional
        Tuple including the 2 quantiles used to draw the Plot.
    start_date: str / int / float / date, optional
        Input Start Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is lesser than November 1993 the 3rd.
    end_date: str / int / float / date, optional
        Input End Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is greater than November 1993 the 3rd.
    plot_median: bool, optional
        If set to True, the Median will be drawn.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object

    See Also
    --------
    vDataFrame.plot : Draws the time series.
        """
        ts = self.parent.format_colnames(ts)
        return plt.range_curve_vdf(
            self, ts, q, start_date, end_date, plot_median, ax=ax, **style_kwds,
        )

    # ---#
    @check_dtypes
    def rename(self, new_name: str):
        """
	----------------------------------------------------------------------------------------
	Renames the vColumn by dropping the current vColumn and creating a copy with 
    the specified name.

    \u26A0 Warning : SQL code generation will be slower if the vDataFrame has been 
                     transformed multiple times, so it's better practice to use 
                     this method when first preparing your data.

	Parameters
 	----------
 	new_name: str
 		The new vColumn alias.

 	Returns
 	-------
 	vDataFrame
 		self.parent

	See Also
	--------
	vDataFrame.add_copy : Creates a copy of the vColumn.
		"""
        old_name = quote_ident(self.alias)
        new_name = new_name.replace('"', "")
        assert not (self.parent.is_colname_in(new_name)), NameError(
            f"A vColumn has already the alias {new_name}.\n"
            "By changing the parameter 'new_name', you'll "
            "be able to solve this issue."
        )
        self.add_copy(new_name)
        parent = self.drop(add_history=False)
        parent.__add_to_history__(
            f"[Rename]: The vColumn {old_name} was renamed '{new_name}'."
        )
        return parent

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def round(self, n: int):
        """
	----------------------------------------------------------------------------------------
	Rounds the vColumn by keeping only the input number of digits after the comma.

	Parameters
 	----------
 	n: int
 		Number of digits to keep after the comma.

 	Returns
 	-------
 	vDataFrame
 		self.parent

	See Also
	--------
	vDataFrame[].apply : Applies a function to the input vColumn.
		"""
        return self.apply(func=f"ROUND({{}}, {n})")

    # ---#
    @save_verticapy_logs
    def sem(self):
        """
	----------------------------------------------------------------------------------------
	Aggregates the vColumn using 'sem' (standard error of mean).

 	Returns
 	-------
 	float
 		sem

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
        return self.aggregate(["sem"]).values[self.alias][0]

    # ---#
    @save_verticapy_logs
    def skewness(self):
        """
	----------------------------------------------------------------------------------------
	Aggregates the vColumn using 'skewness'.

 	Returns
 	-------
 	float
 		skewness

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
        return self.aggregate(["skewness"]).values[self.alias][0]

    skew = skewness
    # ---#
    @check_dtypes
    @save_verticapy_logs
    def slice(self, length: int, unit: str = "second", start: bool = True):
        """
	----------------------------------------------------------------------------------------
	Slices and transforms the vColumn using a time series rule.

	Parameters
 	----------
 	length: int
 		Slice size.
 	unit: str, optional
 		Slice size unit. For example, it can be 'minute' 'hour'...
 	start: bool, optional
 		If set to True, the record will be sliced using the floor of the slicing
 		instead of the ceiling.

 	Returns
 	-------
 	vDataFrame
 		self.parent

	See Also
	--------
	vDataFrame[].date_part : Extracts a specific TS field from the vColumn.
		"""
        start_or_end = "START" if (start) else "END"
        unit = unit.upper()
        return self.apply(
            func=f"TIME_SLICE({{}}, {length}, '{unit}', '{start_or_end}')"
        )

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def spider(
        self,
        by: str = "",
        method: str = "density",
        of: str = "",
        max_cardinality: Union[int, tuple, list] = (6, 6),
        h: Union[int, float, tuple, list] = (None, None),
        ax=None,
        **style_kwds,
    ):
        """
    ----------------------------------------------------------------------------------------
    Draws the spider plot of the input vColumn based on an aggregation.

    Parameters
    ----------
    by: str, optional
        vColumn to use to partition the data.
    method: str, optional
        The method to use to aggregate the data.
            count   : Number of elements.
            density : Percentage of the distribution.
            mean    : Average of the vColumn 'of'.
            min     : Minimum of the vColumn 'of'.
            max     : Maximum of the vColumn 'of'.
            sum     : Sum of the vColumn 'of'.
            q%      : q Quantile of the vColumn 'of' (ex: 50% to get the median).
        It can also be a cutomized aggregation (ex: AVG(column1) + 5).
    of: str, optional
        The vColumn to use to compute the aggregation.
    h: int / float / tuple / list, optional
        Interval width of the vColumns 1 and 2 bars. It is only valid if the 
        vColumns are numerical. Optimized h will be computed if the parameter 
        is empty or invalid.
    max_cardinality: int / tuple / list, optional
        Maximum number of distinct elements for vColumns 1 and 2 to be used as 
        categorical (No h will be picked or computed)
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object

    See Also
    --------
    vDataFrame.bar : Draws the Bar Chart of the input vColumns based on an aggregation.
        """
        by, of = self.parent.format_colnames(by, of)
        columns = [self.alias]
        if by:
            columns += [by]
        return plt.spider(
            self.parent, columns, method, of, max_cardinality, h, ax=ax, **style_kwds,
        )

    # ---#
    @save_verticapy_logs
    def std(self):
        """
	----------------------------------------------------------------------------------------
	Aggregates the vColumn using 'std' (Standard Deviation).

 	Returns
 	-------
 	float
 		std

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
        return self.aggregate(["stddev"]).values[self.alias][0]

    stddev = std
    # ---#
    @save_verticapy_logs
    def store_usage(self):
        """
	----------------------------------------------------------------------------------------
	Returns the vColumn expected store usage (unit: b).

 	Returns
 	-------
 	int
 		vColumn expected store usage.

	See Also
	--------
	vDataFrame.expected_store_usage : Returns the vDataFrame expected store usage.
		"""
        pre_comp = self.parent.__get_catalog_value__(self.alias, "store_usage")
        if pre_comp != "VERTICAPY_NOT_PRECOMPUTED":
            return pre_comp
        alias_sql_repr = bin_spatial_to_str(self.category(), self.alias)
        store_usage = executeSQL(
            query=f"""
                SELECT 
                    /*+LABEL('vColumn.storage_usage')*/ 
                    ZEROIFNULL(SUM(LENGTH({alias_sql_repr}::varchar))) 
                FROM {self.parent.__genSQL__()}""",
            title=f"Computing the Store Usage of the vColumn {self.alias}.",
            method="fetchfirstelem",
            sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
            symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
        )
        self.parent.__update_catalog__(
            {"index": ["store_usage"], self.alias: [store_usage]}
        )
        return store_usage

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def str_contains(self, pat: str):
        """
	----------------------------------------------------------------------------------------
	Verifies if the regular expression is in each of the vColumn records. 
	The vColumn will be transformed.

	Parameters
 	----------
 	pat: str
 		Regular expression.

 	Returns
 	-------
 	vDataFrame
 		self.parent

	See Also
	--------
	vDataFrame[].str_count   : Computes the number of matches for the regular expression
        in each record of the vColumn.
	vDataFrame[].extract     : Extracts the regular expression in each record of the 
		vColumn.
	vDataFrame[].str_replace : Replaces the regular expression matches in each of the 
		vColumn records by an input value.
	vDataFrame[].str_slice   : Slices the vColumn.
		"""
        pat = pat.replace("'", "''")
        return self.apply(func=f"REGEXP_COUNT({{}}, '{pat}') > 0")

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def str_count(self, pat: str):
        """
	----------------------------------------------------------------------------------------
	Computes the number of matches for the regular expression in each record of 
    the vColumn. The vColumn will be transformed.

	Parameters
 	----------
 	pat: str
 		regular expression.

 	Returns
 	-------
 	vDataFrame
 		self.parent

	See Also
	--------
	vDataFrame[].str_contains : Verifies if the regular expression is in each of the 
		vColumn records. 
	vDataFrame[].extract      : Extracts the regular expression in each record of the 
		vColumn.
	vDataFrame[].str_replace  : Replaces the regular expression matches in each of the 
		vColumn records by an input value.
	vDataFrame[].str_slice    : Slices the vColumn.
		"""
        pat = pat.replace("'", "''")
        return self.apply(func=f"REGEXP_COUNT({{}}, '{pat}')")

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def str_extract(self, pat: str):
        """
	----------------------------------------------------------------------------------------
	Extracts the regular expression in each record of the vColumn.
	The vColumn will be transformed.

	Parameters
 	----------
 	pat: str
 		regular expression.

 	Returns
 	-------
 	vDataFrame
 		self.parent

 	See Also
 	--------
	vDataFrame[].str_contains : Verifies if the regular expression is in each of the 
		vColumn records. 
	vDataFrame[].str_count    : Computes the number of matches for the regular expression
        in each record of the vColumn.
	vDataFrame[].str_replace  : Replaces the regular expression matches in each of the 
		vColumn records by an input value.
	vDataFrame[].str_slice    : Slices the vColumn.
		"""
        pat = pat.replace("'", "''")
        return self.apply(func=f"REGEXP_SUBSTR({{}}, '{pat}')")

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def str_replace(self, to_replace: str, value: str = ""):
        """
	----------------------------------------------------------------------------------------
	Replaces the regular expression matches in each of the vColumn record by an
	input value. The vColumn will be transformed.

	Parameters
 	----------
 	to_replace: str
 		Regular expression to replace.
 	value: str, optional
 		New value.

 	Returns
 	-------
 	vDataFrame
 		self.parent

	See Also
	--------
	vDataFrame[].str_contains : Verifies if the regular expression is in each of the 
		vColumn records. 
	vDataFrame[].str_count    : Computes the number of matches for the regular expression
        in each record of the vColumn.
	vDataFrame[].extract      : Extracts the regular expression in each record of the 
		vColumn.
	vDataFrame[].str_slice    : Slices the vColumn.
		"""
        to_replace = to_replace.replace("'", "''")
        value = value.replace("'", "''")
        return self.apply(func=f"REGEXP_REPLACE({{}}, '{to_replace}', '{value}')")

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def str_slice(self, start: int, step: int):
        """
	----------------------------------------------------------------------------------------
	Slices the vColumn. The vColumn will be transformed.

	Parameters
 	----------
 	start: int
 		Start of the slicing.
 	step: int
 		Size of the slicing.

 	Returns
 	-------
 	vDataFrame
 		self.parent

	See Also
	--------
	vDataFrame[].str_contains : Verifies if the regular expression is in each of the 
		vColumn records. 
	vDataFrame[].str_count    : Computes the number of matches for the regular expression
        in each record of the vColumn.
	vDataFrame[].extract      : Extracts the regular expression in each record of the 
		vColumn.
	vDataFrame[].str_replace  : Replaces the regular expression matches in each of the 
		vColumn records by an input value.
		"""
        return self.apply(func=f"SUBSTR({{}}, {start}, {step})")

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def sub(self, x: Union[int, float]):
        """
	----------------------------------------------------------------------------------------
	Subtracts the input element from the vColumn.

	Parameters
 	----------
 	x: int / float
 		If the vColumn type is date like (date, datetime ...), the parameter 'x' 
 		will represent the number of seconds, otherwise it will represent a number.

 	Returns
 	-------
 	vDataFrame
		self.parent

	See Also
	--------
	vDataFrame[].apply : Applies a function to the input vColumn.
		"""
        if self.isdate():
            return self.apply(func=f"TIMESTAMPADD(SECOND, -({x}), {{}})")
        else:
            return self.apply(func=f"{{}} - ({x})")

    # ---#
    @save_verticapy_logs
    def sum(self):
        """
	----------------------------------------------------------------------------------------
	Aggregates the vColumn using 'sum'.

 	Returns
 	-------
 	float
 		sum

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
        return self.aggregate(["sum"]).values[self.alias][0]

    # ---#
    def tail(self, limit: int = 5):
        """
	----------------------------------------------------------------------------------------
	Returns the tail of the vColumn.

	Parameters
 	----------
 	limit: int, optional
 		Number of elements to display.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, see
 		utilities.tablesample.

	See Also
	--------
	vDataFrame[].head : Returns the head of the vColumn.
		"""
        return self.iloc(limit=limit, offset=-1)

    # ---#
    @check_dtypes
    @save_verticapy_logs
    def topk(self, k: int = -1, dropna: bool = True):
        """
	----------------------------------------------------------------------------------------
	Returns the k most occurent elements and their distributions as percents.

	Parameters
 	----------
 	k: int, optional
 		Number of most occurent elements to return.
 	dropna: bool, optional
 		If set to True, NULL values will not be considered during the computation.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, see
 		utilities.tablesample.

	See Also
	--------
	vDataFrame[].describe : Computes the vColumn descriptive statistics.
		"""
        limit, where, topk_cat = "", "", ""
        if k >= 1:
            limit = f"LIMIT {k}"
            topk_cat = k
        if dropna:
            where = f" WHERE {self.alias} IS NOT NULL"
        alias_sql_repr = bin_spatial_to_str(self.category(), self.alias)
        result = executeSQL(
            query=f"""
            SELECT 
                /*+LABEL('vColumn.topk')*/
                {alias_sql_repr} AS {self.alias},
                COUNT(*) AS _verticapy_cnt_,
                100 * COUNT(*) / {self.parent.shape()[0]} AS percent
            FROM {self.parent.__genSQL__()}
            {where} 
            GROUP BY {alias_sql_repr} 
            ORDER BY _verticapy_cnt_ DESC
            {limit}""",
            title=f"Computing the top{topk_cat} categories of {self.alias}.",
            method="fetchall",
            sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
            symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
        )
        values = {
            "index": [item[0] for item in result],
            "count": [int(item[1]) for item in result],
            "percent": [float(round(item[2], 3)) for item in result],
        }
        return tablesample(values)

    # ---#
    @save_verticapy_logs
    def value_counts(self, k: int = 30):
        """
	----------------------------------------------------------------------------------------
	Returns the k most occurent elements, how often they occur, and other
	statistical information.

	Parameters
 	----------
 	k: int, optional
 		Number of most occurent elements to return.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, see
 		utilities.tablesample.

	See Also
	--------
	vDataFrame[].describe : Computes the vColumn descriptive statistics.
		"""
        return self.describe(method="categorical", max_cardinality=k)

    # ---#
    @save_verticapy_logs
    def var(self):
        """
	----------------------------------------------------------------------------------------
	Aggregates the vColumn using 'var' (Variance).

 	Returns
 	-------
 	float
 		var

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
        return self.aggregate(["variance"]).values[self.alias][0]

    variance = var
