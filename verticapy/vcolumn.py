# (c) Copyright [2018-2022] Micro Focus or one of its affiliates.
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
import math, re, decimal, warnings, datetime
from collections.abc import Iterable
from typing import Union

# VerticaPy Modules
import verticapy
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
---------------------------------------------------------------------------
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
        self, alias: str, transformations: list = [], parent=None, catalog: dict = {}
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
                version(condition=[10, 0, 0])
                if index_start < 0:
                    index_start_str = str(index_start) + " + APPLY_COUNT_ELEMENTS({})"
                else:
                    index_start_str = str(index_start)
                if isinstance(index_stop, int):
                    if index_stop < 0:
                        index_stop_str = str(index_stop) + " + APPLY_COUNT_ELEMENTS({})"
                    else:
                        index_stop_str = str(index_stop)
                else:
                    index_stop_str = "1 + APPLY_COUNT_ELEMENTS({})"
                elem_to_select = "{0}[{1}:{2}]".format(
                    self.alias, index_start_str, index_stop_str
                ).replace("{}", self.alias)
                new_alias = quote_ident(
                    self.alias[1:-1] + "." + str(index_start) + ":" + str(index_stop)
                )
                query = "(SELECT {0} AS {1} FROM {2}) VERTICAPY_SUBTABLE".format(
                    elem_to_select, new_alias, self.parent.__genSQL__(),
                )
                vcol = vDataFrameSQL(query)[new_alias]
                vcol.transformations[-1] = (new_alias, self.ctype(), self.category())
                vcol.init_transf = "{0}[{1}:{2}]".format(
                    self.init_transf, index_start_str, index_stop_str
                ).replace("{}", self.init_transf)
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
                query = "(SELECT {0} FROM {1}{2} OFFSET {3}{4}) VERTICAPY_SUBTABLE".format(
                    self.alias,
                    self.parent.__genSQL__(),
                    self.parent.__get_last_order_by__(),
                    index_start,
                    limit,
                )
                return vDataFrameSQL(query)
        elif isinstance(index, int):
            if self.isarray():
                version(condition=[9, 3, 0])
                elem_to_select = "{0}[{1}]".format(self.alias, index)
                new_alias = quote_ident(self.alias[1:-1] + "." + str(index))
                query = "(SELECT {0} AS {1} FROM {2}) VERTICAPY_SUBTABLE".format(
                    elem_to_select, new_alias, self.parent.__genSQL__(),
                )
                vcol = vDataFrameSQL(query)[new_alias]
                vcol.init_transf = "{0}[{1}]".format(self.init_transf, index)
                return vcol
            else:
                cast = "::float" if self.category() == "float" else ""
                if index < 0:
                    index += self.parent.shape()[0]
                query = "SELECT /*+LABEL('vColumn.__getitem__')*/ {}{} FROM {}{} OFFSET {} LIMIT 1".format(
                    self.alias,
                    cast,
                    self.parent.__genSQL__(),
                    self.parent.__get_last_order_by__(),
                    index,
                )
                return executeSQL(
                    query=query,
                    title="Getting the vColumn element.",
                    method="fetchfirstelem",
                    sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
                    symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
                )
        elif isinstance(index, str):
            if self.category() == "vmap":
                elem_to_select = "MAPLOOKUP({0}, '{1}')".format(
                    self.alias, index.replace("'", "''")
                )
                init_transf = "MAPLOOKUP({0}, '{1}')".format(
                    self.init_transf, index.replace("'", "''")
                )
            else:
                version(condition=[10, 0, 0])
                elem_to_select = self.alias + "." + quote_ident(index)
                init_transf = self.init_transf + "." + quote_ident(index)
            query = "(SELECT {0} AS {1} FROM {2}) VERTICAPY_SUBTABLE".format(
                elem_to_select, quote_ident(index), self.parent.__genSQL__(),
            )
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
        return self.head(limit=verticapy.options["max_rows"]).__repr__()

    # ---#
    def _repr_html_(self):
        return self.head(limit=verticapy.options["max_rows"])._repr_html_()

    # ---#
    def __setattr__(self, attr, val):
        self.__dict__[attr] = val

    #
    # Methods
    #
    # ---#
    def aad(self):
        """
    ---------------------------------------------------------------------------
    Aggregates the vColumn using 'aad' (Average Absolute Deviation).

    Returns
    -------
    float
        aad

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        # Saving information to the query profile table
        save_to_query_profile(
            name="aad", path="vcolumn.vColumn", json_dict={},
        )
        # -#
        return self.aggregate(["aad"]).values[self.alias][0]

    # ---#
    def abs(self):
        """
	---------------------------------------------------------------------------
	Applies the absolute value function to the input vColumn. 

 	Returns
 	-------
 	vDataFrame
		self.parent

	See Also
	--------
	vDataFrame[].apply : Applies a function to the input vColumn.
		"""
        # Saving information to the query profile table
        save_to_query_profile(
            name="abs", path="vcolumn.vColumn", json_dict={},
        )
        # -#
        return self.apply(func="ABS({})")

    # ---#
    def add(self, x: float):
        """
	---------------------------------------------------------------------------
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="add", path="vcolumn.vColumn", json_dict={"x": x,},
        )
        # -#
        check_types([("x", x, [int, float])])
        if self.isdate():
            return self.apply(func="TIMESTAMPADD(SECOND, {}, {})".format(x, "{}"))
        else:
            return self.apply(func="{} + ({})".format("{}", x))

    # ---#
    def add_copy(self, name: str):
        """
	---------------------------------------------------------------------------
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
        # -#
        check_types([("name", name, [str])])
        name = quote_ident(name.replace('"', "_"))
        assert name.replace('"', ""), EmptyParameter(
            "The parameter 'name' must not be empty"
        )
        assert not (self.parent.is_colname_in(name)), NameError(
            f"A vColumn has already the alias {name}.\nBy changing the parameter 'name', you'll be able to solve this issue."
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
            "[Add Copy]: A copy of the vColumn {} named {} was added to the vDataFrame.".format(
                self.alias, name
            )
        )
        return self.parent

    # ---#
    def aggregate(self, func: list):
        """
	---------------------------------------------------------------------------
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="aggregate", path="vcolumn.vColumn", json_dict={"func": func,},
        )
        # -#
        return self.parent.aggregate(func=func, columns=[self.alias]).transpose()

    agg = aggregate
    # ---#
    def apply(self, func: str, copy_name: str = ""):
        """
	---------------------------------------------------------------------------
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="apply",
            path="vcolumn.vColumn",
            json_dict={"func": func, "copy_name": copy_name,},
        )
        # -#
        if isinstance(func, str_sql):
            func = str(func)
        check_types([("func", func, [str]), ("copy_name", copy_name, [str])])
        try:
            try:
                ctype = get_data_types(
                    "SELECT {} AS apply_test_feature FROM {} WHERE {} IS NOT NULL LIMIT 0".format(
                        func.replace("{}", self.alias),
                        self.parent.__genSQL__(),
                        self.alias,
                    ),
                    "apply_test_feature",
                )
            except:
                ctype = get_data_types(
                    "SELECT {} AS apply_test_feature FROM {} WHERE {} IS NOT NULL LIMIT 0".format(
                        func.replace("{}", self.alias),
                        self.parent.__genSQL__(),
                        self.alias,
                    ),
                    "apply_test_feature",
                )
            category = get_category_from_vertica_type(ctype=ctype)
            all_cols, max_floor = self.parent.get_columns(), 0
            for column in all_cols:
                try:
                    if (quote_ident(column) in func) or (
                        re.search(
                            re.compile("\\b{}\\b".format(column.replace('"', ""))), func
                        )
                    ):
                        max_floor = max(
                            len(self.parent[column].transformations), max_floor
                        )
                except:
                    pass
            max_floor -= len(self.transformations)
            if copy_name:
                self.add_copy(name=copy_name)
                for k in range(max_floor):
                    self.parent[copy_name].transformations += [
                        ("{}", self.ctype(), self.category())
                    ]
                self.parent[copy_name].transformations += [(func, ctype, category)]
                self.parent[copy_name].catalog = self.catalog
                self.parent.__add_to_history__(
                    "[Apply]: The vColumn '{}' was transformed with the func 'x -> {}'.".format(
                        copy_name.replace('"', ""), func.replace("{}", "x"),
                    )
                )
            else:
                for k in range(max_floor):
                    self.transformations += [("{}", self.ctype(), self.category())]
                self.transformations += [(func, ctype, category)]
                self.parent.__update_catalog__(erase=True, columns=[self.alias])
                self.parent.__add_to_history__(
                    "[Apply]: The vColumn '{}' was transformed with the func 'x -> {}'.".format(
                        self.alias.replace('"', ""), func.replace("{}", "x"),
                    )
                )
            return self.parent
        except Exception as e:
            raise QueryError(
                "{}\nError when applying the func 'x -> {}' to '{}'".format(
                    e, func.replace("{}", "x"), self.alias.replace('"', "")
                )
            )

    # ---#
    def apply_fun(self, func: str, x: float = 2):
        """
	---------------------------------------------------------------------------
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="apply_fun", path="vcolumn.vColumn", json_dict={"func": func, "x": x,},
        )
        if isinstance(func, str):
            func = func.lower()
        if func == "mean":
            func = "avg"
        elif func == "length":
            func = "len"
        # -#
        check_types(
            [
                (
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
                        "ln",
                        "log",
                        "log10",
                        "max",
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
                ),
                ("x", x, [int, float, str]),
            ]
        )
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
            expr = "{}({})".format(func.upper(), "{}")
        elif func in ("log", "mod", "pow", "round"):
            expr = "{}({}, {})".format(func.upper(), "{}", x)
        elif func in ("contain", "find"):
            if func == "contain":
                if cat == "vmap":
                    f = "MAPCONTAINSVALUE"
                else:
                    f = "CONTAINS"
            elif func == "find":
                f = "ARRAY_FIND"
            if isinstance(x, (str,)):
                x = "'" + str(x).replace("'", "''") + "'"
            expr = "{}({}, {})".format(f, "{}", x)
        return self.apply(func=expr)

    # ---#
    def astype(self, dtype):
        """
	---------------------------------------------------------------------------
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="astype", path="vcolumn.vColumn", json_dict={"dtype": dtype,},
        )
        dtype = get_vertica_type(dtype)
        # -#
        check_types([("dtype", dtype, [str])])
        try:
            if (
                dtype == "array" or str(dtype).startswith("vmap")
            ) and self.category() == "text":
                if dtype == "array":
                    version(condition=[10, 0, 0])
                query = "SELECT {0} FROM {1} ORDER BY LENGTH({0}) DESC LIMIT 1".format(
                    self.alias, self.parent.__genSQL__()
                )
                biggest_str = executeSQL(
                    query, title="getting the biggest string", method="fetchfirstelem"
                )
                biggest_str = biggest_str.strip()
                sep = guess_sep(biggest_str)
                if str(dtype).startswith("vmap"):
                    if len(biggest_str) > 2 and (
                        (biggest_str[0] == "{" and biggest_str[-1] == "}")
                    ):
                        transformation = (
                            "MAPJSONEXTRACTOR({0} USING PARAMETERS flatten_maps=false)".format(
                                self.alias,
                            ),
                            "MAPJSONEXTRACTOR({} USING PARAMETERS flatten_maps=false)",
                        )
                    else:
                        header_names = ""
                        if len(dtype) > 4 and dtype[:5] == "vmap(" and dtype[-1] == ")":
                            header_names = ", header_names='{0}'".format(dtype[5:-1])
                        transformation = (
                            "MAPDELIMITEDEXTRACTOR({0} USING PARAMETERS delimiter='{1}'{2})".format(
                                self.alias, sep, header_names,
                            ),
                            "MAPDELIMITEDEXTRACTOR({0} USING PARAMETERS delimiter='{1}'{2})".format(
                                "{}", sep, header_names,
                            ),
                        )
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
                        collection_open = ", collection_open='{0}'".format(
                            biggest_str[0]
                        )
                        collection_close = ", collection_close='{0}'".format(
                            biggest_str[-1]
                        )
                    else:
                        collection_open, collection_close = "", ""
                    transformation = (
                        "STRING_TO_ARRAY({0} USING PARAMETERS collection_delimiter='{1}'{2}{3}{4})".format(
                            self.alias,
                            sep,
                            collection_open,
                            collection_close,
                            collection_null_element,
                        ),
                        "STRING_TO_ARRAY({0} USING PARAMETERS collection_delimiter='{1}'{2}{3}{4})".format(
                            "{}",
                            sep,
                            collection_open,
                            collection_close,
                            collection_null_element,
                        ),
                    )
            elif (
                dtype[0:7] == "varchar" or dtype[0:4] == "char"
            ) and self.category() == "vmap":
                transformation = (
                    "MAPTOSTRING({0} USING PARAMETERS canonical_json=false)::{1}".format(
                        self.alias, dtype
                    ),
                    "MAPTOSTRING({} USING PARAMETERS canonical_json=false)::" + dtype,
                )
            elif dtype == "json":
                if self.category() == "vmap":
                    transformation = (
                        "MAPTOSTRING({0} USING PARAMETERS canonical_json=true)".format(
                            self.alias
                        ),
                        "MAPTOSTRING({} USING PARAMETERS canonical_json=true)",
                    )
                else:
                    version(condition=[10, 1, 0])
                    transformation = "TO_JSON({0})".format(self.alias), "TO_JSON({})"
                dtype = "varchar"
            else:
                transformation = "{0}::{1}".format(self.alias, dtype), "{}::" + dtype
            query = "SELECT /*+LABEL('vColumn.astype')*/ {0} AS {1} FROM {2} WHERE {1} IS NOT NULL LIMIT 20".format(
                transformation[0], self.alias, self.parent.__genSQL__()
            )
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
                "[AsType]: The vColumn {} was converted to {}.".format(
                    self.alias, dtype
                )
            )
            return self.parent
        except Exception as e:
            raise ConversionError(
                "{0}\nThe vColumn {1} can not be converted to {2}".format(
                    e, self.alias, dtype
                )
            )

    # ---#
    def avg(self):
        """
	---------------------------------------------------------------------------
	Aggregates the vColumn using 'avg' (Average).

 	Returns
 	-------
 	float
 		average

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
        # Saving information to the query profile table
        save_to_query_profile(
            name="avg", path="vcolumn.vColumn", json_dict={},
        )
        # -#
        return self.aggregate(["avg"]).values[self.alias][0]

    mean = avg
    # ---#
    def bar(
        self,
        method: str = "density",
        of: str = "",
        max_cardinality: int = 6,
        nbins: int = 0,
        h: float = 0,
        ax=None,
        **style_kwds,
    ):
        """
	---------------------------------------------------------------------------
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
 	h: float, optional
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="bar",
            path="vcolumn.vColumn",
            json_dict={
                **{
                    "method": method,
                    "of": of,
                    "max_cardinality": max_cardinality,
                    "nbins": nbins,
                    "h": h,
                },
                **style_kwds,
            },
        )
        # -#
        check_types(
            [
                ("method", method, [str]),
                ("of", of, [str]),
                ("max_cardinality", max_cardinality, [int, float]),
                ("nbins", nbins, [int, float]),
                ("h", h, [int, float]),
            ]
        )
        if of:
            self.parent.are_namecols_in(of)
            of = self.parent.format_colnames(of)
        from verticapy.plot import bar

        return bar(self, method, of, max_cardinality, nbins, h, ax=ax, **style_kwds)

    # ---#
    def boxplot(
        self,
        by: str = "",
        h: float = 0,
        max_cardinality: int = 8,
        cat_priority: list = [],
        ax=None,
        **style_kwds,
    ):
        """
	---------------------------------------------------------------------------
	Draws the box plot of the vColumn.

	Parameters
 	----------
 	by: str, optional
 		vColumn to use to partition the data.
 	h: float, optional
 		Interval width if the vColumn is numerical or of type date like. Optimized 
 		h will be computed if the parameter is empty or invalid.
 	max_cardinality: int, optional
 		Maximum number of vColumn distinct elements to be used as categorical. 
 		The less frequent elements will be gathered together to create a new 
 		category : 'Others'.
 	cat_priority: list, optional
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="boxplot",
            path="vcolumn.vColumn",
            json_dict={
                **{
                    "by": by,
                    "h": h,
                    "max_cardinality": max_cardinality,
                    "cat_priority": cat_priority,
                },
                **style_kwds,
            },
        )
        # -#
        if isinstance(cat_priority, str) or not (isinstance(cat_priority, Iterable)):
            cat_priority = [cat_priority]
        check_types(
            [
                ("by", by, [str]),
                ("max_cardinality", max_cardinality, [int, float]),
                ("h", h, [int, float]),
                ("cat_priority", cat_priority, [list]),
            ]
        )
        if by:
            self.parent.are_namecols_in(by)
            by = self.parent.format_colnames(by)
        from verticapy.plot import boxplot

        return boxplot(self, by, h, max_cardinality, cat_priority, ax=ax, **style_kwds)

    # ---#
    def category(self):
        """
	---------------------------------------------------------------------------
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
    def clip(self, lower=None, upper=None):
        """
	---------------------------------------------------------------------------
	Clips the vColumn by transforming the values lesser than the lower bound to 
	the lower bound itself and the values higher than the upper bound to the upper 
	bound itself.

	Parameters
 	----------
 	lower: float, optional
 		Lower bound.
 	upper: float, optional
 		Upper bound.

 	Returns
 	-------
 	vDataFrame
		self.parent

 	See Also
 	--------
 	vDataFrame[].fill_outliers : Fills the vColumn outliers using the input method.
		"""
        # Saving information to the query profile table
        save_to_query_profile(
            name="clip",
            path="vcolumn.vColumn",
            json_dict={"lower": lower, "upper": upper,},
        )
        # -#
        check_types([("lower", lower, [float, int]), ("upper", upper, [float, int])])
        assert (lower != None) or (upper != None), ParameterError(
            "At least 'lower' or 'upper' must have a numerical value"
        )
        lower_when = (
            "WHEN {} < {} THEN {} ".format("{}", lower, lower)
            if (isinstance(lower, (float, int)))
            else ""
        )
        upper_when = (
            "WHEN {} > {} THEN {} ".format("{}", upper, upper)
            if (isinstance(upper, (float, int)))
            else ""
        )
        func = "(CASE {}{}ELSE {} END)".format(lower_when, upper_when, "{}")
        self.apply(func=func)
        return self.parent

    # ---#
    def count(self):
        """
	---------------------------------------------------------------------------
	Aggregates the vColumn using 'count' (Number of non-Missing elements).

 	Returns
 	-------
 	int
 		number of non-Missing elements.

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
        # Saving information to the query profile table
        save_to_query_profile(
            name="count", path="vcolumn.vColumn", json_dict={},
        )
        # -#
        return self.aggregate(["count"]).values[self.alias][0]

    # ---#
    def cut(
        self,
        breaks: list,
        labels: list = [],
        include_lowest: bool = True,
        right: bool = True,
    ):
        """
    ---------------------------------------------------------------------------
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="cut",
            path="vcolumn.vColumn",
            json_dict={
                "breaks": breaks,
                "labels": labels,
                "include_lowest": include_lowest,
                "right": right,
            },
        )
        # -#
        check_types(
            [
                ("breaks", breaks, [list]),
                ("labels", labels, [list]),
                ("include_lowest", include_lowest, [bool]),
                ("right", right, [bool]),
            ]
        )
        assert self.isnum() or self.isdate(), TypeError(
            "cut only works on numerical / date-like vColumns."
        )
        assert len(breaks) >= 2, ParameterError(
            "Length of parameter 'breaks' must be greater or equal to 2."
        )
        assert len(breaks) == len(labels) + 1 or not (labels), ParameterError(
            "Length of parameter breaks must be equal to the length of parameter 'labels' + 1 or parameter 'labels' must be empty."
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
	---------------------------------------------------------------------------
	Returns the vColumn DB type.

 	Returns
 	-------
 	str
 		vColumn DB type.
		"""
        return self.transformations[-1][1].lower()

    dtype = ctype
    # ---#
    def date_part(self, field: str):
        """
	---------------------------------------------------------------------------
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="date_part", path="vcolumn.vColumn", json_dict={"field": field,},
        )
        # -#
        return self.apply(func="DATE_PART('{}', {})".format(field, "{}"))

    # ---#
    def decode(self, *argv):
        """
	---------------------------------------------------------------------------
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="decode", path="vcolumn.vColumn", json_dict={"argv": argv},
        )
        # -#
        import verticapy.stats as st

        return self.apply(func=st.decode(str_sql("{}"), *argv))

    # ---#
    def density(
        self,
        by: str = "",
        bandwidth: float = 1.0,
        kernel: str = "gaussian",
        nbins: int = 200,
        xlim: tuple = None,
        ax=None,
        **style_kwds,
    ):
        """
	---------------------------------------------------------------------------
	Draws the vColumn Density Plot.

	Parameters
 	----------
    by: str, optional
        vColumn to use to partition the data.
 	bandwidth: float, optional
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="density",
            path="vcolumn.vColumn",
            json_dict={
                **{"by": by, "kernel": kernel, "bandwidth": bandwidth, "nbins": nbins,},
                **style_kwds,
            },
        )
        # -#
        check_types(
            [
                ("by", by, [str]),
                ("kernel", kernel, ["gaussian", "logistic", "sigmoid", "silverman"]),
                ("bandwidth", bandwidth, [int, float]),
                ("nbins", nbins, [float, int]),
            ]
        )
        if by:
            self.parent.are_namecols_in(by)
            by = self.parent.format_colnames(by)
            from verticapy.plot import gen_colors
            from matplotlib.lines import Line2D

            colors = gen_colors()
            if not xlim:
                xmin = self.min()
                xmax = self.max()
            else:
                xmin, xmax = xlim
            custom_lines = []
            columns = self.parent[by].distinct()
            for idx, column in enumerate(columns):
                param = {"color": colors[idx % len(colors)]}
                ax = self.parent.search(
                    "{} = '{}'".format(self.parent[by].alias, column)
                )[self.alias].density(
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
        from verticapy.learn.neighbors import KernelDensity

        schema = verticapy.options["temp_schema"]
        if not (schema):
            schema = "public"
        name = gen_tmp_name(schema=schema, name="kde")
        if isinstance(xlim, (tuple, list)):
            xlim_tmp = [xlim]
        else:
            xlim_tmp = []
        model = KernelDensity(
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
            model.drop()
            return result
        except:
            model.drop()
            raise

    # ---#
    def describe(
        self, method: str = "auto", max_cardinality: int = 6, numcol: str = ""
    ):
        """
	---------------------------------------------------------------------------
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="describe",
            path="vcolumn.vColumn",
            json_dict={
                "method": method,
                "max_cardinality": max_cardinality,
                "numcol": numcol,
            },
        )
        # -#
        check_types(
            [
                ("method", method, ["auto", "numerical", "categorical", "cat_stats"]),
                ("max_cardinality", max_cardinality, [int, float]),
                ("numcol", numcol, [str]),
            ]
        )

        method = method.lower()
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
                tmp_query = """SELECT 
                                    '{0}' AS 'index', 
                                    COUNT({1}) AS count, 
                                    100 * COUNT({1}) / {2} AS percent, 
                                    AVG({3}{4}) AS mean, 
                                    STDDEV({3}{4}) AS std, 
                                    MIN({3}{4}) AS min, 
                                    APPROXIMATE_PERCENTILE ({3}{4} 
                                        USING PARAMETERS percentile = 0.1) AS 'approx_10%', 
                                    APPROXIMATE_PERCENTILE ({3}{4} 
                                        USING PARAMETERS percentile = 0.25) AS 'approx_25%', 
                                    APPROXIMATE_PERCENTILE ({3}{4} 
                                        USING PARAMETERS percentile = 0.5) AS 'approx_50%', 
                                    APPROXIMATE_PERCENTILE ({3}{4} 
                                        USING PARAMETERS percentile = 0.75) AS 'approx_75%', 
                                    APPROXIMATE_PERCENTILE ({3}{4} 
                                        USING PARAMETERS percentile = 0.9) AS 'approx_90%', 
                                    MAX({3}{4}) AS max 
                               FROM vdf_table""".format(
                    category, self.alias, self.parent.shape()[0], numcol, cast,
                )
                tmp_query += (
                    " WHERE {} IS NULL".format(self.alias)
                    if (category in ("None", None))
                    else " WHERE {} = '{}'".format(
                        bin_spatial_to_str(self.category(), self.alias), category,
                    )
                )
                query += [lp + tmp_query + rp]
            query = "WITH vdf_table AS (SELECT * FROM {}) {}".format(
                self.parent.__genSQL__(), " UNION ALL ".join(query)
            )
            title = "Describes the statics of {} partitioned by {}.".format(
                numcol, self.alias
            )
            values = to_tablesample(
                query,
                title=title,
                sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
                symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
            ).values
        elif (
            ((distinct_count < max_cardinality + 1) and (method != "numerical"))
            or not (is_numeric)
            or (method == "categorical")
        ):
            query = """(SELECT 
                            {0} || '', 
                            COUNT(*) 
                        FROM vdf_table 
                        GROUP BY {0} 
                        ORDER BY COUNT(*) DESC 
                        LIMIT {1})""".format(
                self.alias, max_cardinality
            )
            if distinct_count > max_cardinality:
                query += (
                    "UNION ALL (SELECT 'Others', SUM(count) FROM (SELECT COUNT(*) AS count"
                    " FROM vdf_table WHERE {0} IS NOT NULL GROUP BY {0} ORDER BY COUNT(*)"
                    " DESC OFFSET {1}) VERTICAPY_SUBTABLE) ORDER BY count DESC"
                ).format(self.alias, max_cardinality + 1)
            query = "WITH vdf_table AS (SELECT /*+LABEL('vColumn.describe')*/ * FROM {}) {}".format(
                self.parent.__genSQL__(), query
            )
            query_result = executeSQL(
                query=query,
                title="Computing the descriptive statistics of {}.".format(self.alias),
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
    def discretize(
        self,
        method: str = "auto",
        h: float = 0,
        nbins: int = -1,
        k: int = 6,
        new_category: str = "Others",
        RFmodel_params: dict = {},
        response: str = "",
        return_enum_trans: bool = False,
    ):
        """
	---------------------------------------------------------------------------
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
 	h: float, optional
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="discretize",
            path="vcolumn.vColumn",
            json_dict={
                "RFmodel_params": RFmodel_params,
                "return_enum_trans": return_enum_trans,
                "h": h,
                "response": response,
                "nbins": nbins,
                "method": method,
                "return_enum_trans": return_enum_trans,
            },
        )
        # -#
        check_types(
            [
                ("RFmodel_params", RFmodel_params, [dict]),
                ("return_enum_trans", return_enum_trans, [bool]),
                ("h", h, [int, float]),
                ("response", response, [str]),
                ("nbins", nbins, [int, float]),
                (
                    "method",
                    method,
                    ["auto", "smart", "same_width", "same_freq", "topk"],
                ),
                ("return_enum_trans", return_enum_trans, [bool]),
            ]
        )
        method = method.lower()
        if self.isnum() and method == "smart":
            schema = verticapy.options["temp_schema"]
            if not (schema):
                schema = "public"
            tmp_view_name = gen_tmp_name(schema=schema, name="view")
            tmp_model_name = gen_tmp_name(schema=schema, name="model")
            assert nbins >= 2, ParameterError(
                "Parameter 'nbins' must be greater or equals to 2 in case of discretization using the method 'smart'."
            )
            assert response, ParameterError(
                "Parameter 'response' can not be empty in case of discretization using the method 'smart'."
            )
            self.parent.are_namecols_in(response)
            response = self.parent.format_colnames(response)
            drop(tmp_view_name, method="view")
            self.parent.to_db(tmp_view_name)
            from verticapy.learn.ensemble import (
                RandomForestClassifier,
                RandomForestRegressor,
            )

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
                    "(SELECT READ_TREE(USING PARAMETERS model_name = '{}', tree_id = {}, format = 'tabular'))".format(
                        tmp_model_name, i
                    )
                    for i in range(parameters["n_estimators"])
                ]
                query = "SELECT /*+LABEL('vColumn.discretize')*/ split_value FROM (SELECT split_value, MAX(weighted_information_gain) FROM ({}) VERTICAPY_SUBTABLE WHERE split_value IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT {}) VERTICAPY_SUBTABLE ORDER BY split_value::float".format(
                    " UNION ALL ".join(query), nbins - 1
                )
                result = executeSQL(
                    query=query,
                    title="Computing the optimized histogram nbins using Random Forest.",
                    method="fetchall",
                    sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
                    symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
                )
                result = [elem[0] for elem in result]
            except:
                drop(tmp_view_name, method="view")
                drop(tmp_model_name, method="model")
                raise
            drop(tmp_view_name, method="view")
            drop(tmp_model_name, method="model")
            result = [self.min()] + result + [self.max()]
        elif method == "topk":
            assert k >= 2, ParameterError(
                "Parameter 'k' must be greater or equals to 2 in case of discretization using the method 'topk'"
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
                "Parameter 'nbins' must be greater or equals to 2 in case of discretization using the method 'same_freq'"
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
            where = "WHERE _verticapy_row_nb_ IN ({})".format(
                ", ".join(["1"] + nth_elems + [str(count)])
            )
            query = "SELECT /*+LABEL('vColumn.discretize')*/ {0} FROM (SELECT {0}, ROW_NUMBER() OVER (ORDER BY {0}) AS _verticapy_row_nb_ FROM {1} WHERE {0} IS NOT NULL) VERTICAPY_SUBTABLE {2}".format(
                self.alias, self.parent.__genSQL__(), where,
            )
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
                    "'[' || FLOOR({} / {}) * {} || ';' || (FLOOR({} / {}) * {} + {}{}) || ']'".format(
                        "{}", h, h, "{}", h, h, h, floor_end
                    ),
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
                trans += "WHEN {} BETWEEN {} AND {} THEN '[{};{}]' ".format(
                    "{}", result[i - 1], result[i], result[i - 1], result[i]
                )
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
                "[Discretize]: The vColumn {} was discretized.".format(self.alias)
            )
        return self.parent

    # ---#
    def distinct(self, **kwargs):
        """
	---------------------------------------------------------------------------
	Returns the distinct categories of the vColumn.

 	Returns
 	-------
 	list
 		Distinct caterogies of the vColumn.

	See Also
	--------
	vDataFrame.topk : Returns the vColumn most occurent elements.
		"""
        if "agg" not in kwargs:
            query = "SELECT /*+LABEL('vColumn.distinct')*/ {0} AS {1} FROM {2} WHERE {1} IS NOT NULL GROUP BY {1} ORDER BY {1}".format(
                bin_spatial_to_str(self.category(), self.alias),
                self.alias,
                self.parent.__genSQL__(),
            )
        else:
            query = "SELECT /*+LABEL('vColumn.distinct')*/ {0} FROM (SELECT {1} AS {0}, {2} AS verticapy_agg FROM {3} WHERE {0} IS NOT NULL GROUP BY 1) x ORDER BY verticapy_agg DESC".format(
                self.alias,
                bin_spatial_to_str(self.category(), self.alias),
                kwargs["agg"],
                self.parent.__genSQL__(),
            )
        query_result = executeSQL(
            query=query,
            title="Computing the distinct categories of {}.".format(self.alias),
            method="fetchall",
            sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
            symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
        )
        return [item for sublist in query_result for item in sublist]

    # ---#
    def div(self, x: float):
        """
	---------------------------------------------------------------------------
	Divides the vColumn by the input element.

	Parameters
 	----------
 	x: float
 		Input number.

 	Returns
 	-------
 	vDataFrame
		self.parent

	See Also
	--------
	vDataFrame[].apply : Applies a function to the input vColumn.
		"""
        # Saving information to the query profile table
        save_to_query_profile(
            name="div", path="vcolumn.vColumn", json_dict={"x": x,},
        )
        # -#
        check_types([("x", x, [int, float])])
        assert x != 0, ValueError("Division by 0 is forbidden !")
        return self.apply(func="{} / ({})".format("{}", x))

    # ---#
    def drop(self, add_history: bool = True):
        """
	---------------------------------------------------------------------------
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="drop",
            path="vcolumn.vColumn",
            json_dict={"add_history": add_history,},
        )
        # -#
        check_types([("add_history", add_history, [bool])])
        try:
            parent = self.parent
            force_columns = [
                column for column in self.parent._VERTICAPY_VARIABLES_["columns"]
            ]
            force_columns.remove(self.alias)
            executeSQL(
                "SELECT /*+LABEL('vColumn.drop')*/ * FROM {} LIMIT 10".format(
                    self.parent.__genSQL__(force_columns=force_columns)
                ),
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
                "[Drop]: vColumn {} was deleted from the vDataFrame.".format(self.alias)
            )
        return parent

    # ---#
    def drop_outliers(
        self, threshold: float = 4.0, use_threshold: bool = True, alpha: float = 0.05
    ):
        """
	---------------------------------------------------------------------------
	Drops outliers in the vColumn.

	Parameters
 	----------
 	threshold: float, optional
 		Uses the Gaussian distribution to identify outliers. After normalizing 
 		the data (Z-Score), if the absolute value of the record is greater than 
 		the threshold, it will be considered as an outlier.
 	use_threshold: bool, optional
 		Uses the threshold instead of the 'alpha' parameter.
 	alpha: float, optional
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="drop_outliers",
            path="vcolumn.vColumn",
            json_dict={
                "threshold": threshold,
                "use_threshold": use_threshold,
                "alpha": alpha,
            },
        )
        # -#
        check_types(
            [
                ("alpha", alpha, [int, float]),
                ("use_threshold", use_threshold, [bool]),
                ("threshold", threshold, [int, float]),
            ]
        )
        if use_threshold:
            result = self.aggregate(func=["std", "avg"]).transpose().values
            self.parent.filter(
                "ABS({} - {}) / {} < {}".format(
                    self.alias, result["avg"][0], result["std"][0], threshold
                )
            )
        else:
            p_alpha, p_1_alpha = (
                self.parent.quantile([alpha, 1 - alpha], [self.alias])
                .transpose()
                .values[self.alias]
            )
            self.parent.filter(
                "({} BETWEEN {} AND {})".format(self.alias, p_alpha, p_1_alpha)
            )
        return self.parent

    # ---#
    def dropna(self):
        """
	---------------------------------------------------------------------------
	Filters the vDataFrame where the vColumn is missing.

 	Returns
 	-------
 	vDataFrame
		self.parent

 	See Also
	--------
	vDataFrame.filter: Filters the data using the input expression.
		"""
        # Saving information to the query profile table
        save_to_query_profile(
            name="dropna", path="vcolumn.vColumn", json_dict={},
        )
        # -#
        self.parent.filter("{} IS NOT NULL".format(self.alias))
        return self.parent

    # ---#
    def fill_outliers(
        self,
        method: str = "winsorize",
        threshold: float = 4.0,
        use_threshold: bool = True,
        alpha: float = 0.05,
    ):
        """
	---------------------------------------------------------------------------
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
		threshold: float, optional
			Uses the Gaussian distribution to define the outliers. After normalizing the 
			data (Z-Score), if the absolute value of the record is greater than the 
			threshold it will be considered as an outlier.
		use_threshold: bool, optional
			Uses the threshold instead of the 'alpha' parameter.
		alpha: float, optional
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="fill_outliers",
            path="vcolumn.vColumn",
            json_dict={
                "method": method,
                "alpha": alpha,
                "use_threshold": use_threshold,
                "threshold": threshold,
            },
        )
        # -#
        if isinstance(method, str):
            method = method.lower()
        check_types(
            [
                ("method", method, ["winsorize", "null", "mean"]),
                ("alpha", alpha, [int, float]),
                ("use_threshold", use_threshold, [bool]),
                ("threshold", threshold, [int, float]),
            ]
        )
        if use_threshold:
            result = self.aggregate(func=["std", "avg"]).transpose().values
            p_alpha, p_1_alpha = (
                -threshold * result["std"][0] + result["avg"][0],
                threshold * result["std"][0] + result["avg"][0],
            )
        else:
            query = "SELECT /*+LABEL('vColumn.fill_outliers')*/ PERCENTILE_CONT({0}) WITHIN GROUP (ORDER BY {1}) OVER (), PERCENTILE_CONT(1 - {0}) WITHIN GROUP (ORDER BY {1}) OVER () FROM {2} LIMIT 1".format(
                alpha, self.alias, self.parent.__genSQL__()
            )
            p_alpha, p_1_alpha = executeSQL(
                query=query,
                title="Computing the quantiles of {0}.".format(self.alias),
                method="fetchrow",
                sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
                symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
            )
        if method == "winsorize":
            self.clip(lower=p_alpha, upper=p_1_alpha)
        elif method == "null":
            self.apply(
                func="(CASE WHEN ({0} BETWEEN {1} AND {2}) THEN {0} ELSE NULL END)".format(
                    "{}", p_alpha, p_1_alpha
                )
            )
        elif method == "mean":
            query = "WITH vdf_table AS (SELECT /*+LABEL('vColumn.fill_outliers')*/ * FROM {0}) (SELECT AVG({1}) FROM vdf_table WHERE {1} < {2}) UNION ALL (SELECT AVG({1}) FROM vdf_table WHERE {1} > {3})".format(
                self.parent.__genSQL__(), self.alias, p_alpha, p_1_alpha,
            )
            mean_alpha, mean_1_alpha = [
                item[0]
                for item in executeSQL(
                    query=query,
                    title="Computing the average of the {}'s lower and upper outliers.".format(
                        self.alias
                    ),
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
                func="(CASE WHEN {} < {} THEN {} WHEN {} > {} THEN {} ELSE {} END)".format(
                    "{}", p_alpha, mean_alpha, "{}", p_1_alpha, mean_1_alpha, "{}"
                )
            )
        return self.parent

    # ---#
    def fillna(
        self,
        val=None,
        method: str = "auto",
        expr: str = "",
        by: list = [],
        order_by: list = [],
    ):
        """
	---------------------------------------------------------------------------
	Fills missing elements in the vColumn with a user-specified rule.

	Parameters
 	----------
 	val: int/float/str, optional
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
	by: list, optional
 		vColumns used in the partition.
 	order_by: list, optional
 		List of the vColumns to use to sort the data when using TS methods.

 	Returns
 	-------
 	vDataFrame
		self.parent

	See Also
	--------
	vDataFrame[].dropna : Drops the vColumn missing values.
		"""
        # Saving information to the query profile table
        save_to_query_profile(
            name="fillna",
            path="vcolumn.vColumn",
            json_dict={
                "val": val,
                "method": method,
                "expr": expr,
                "by": by,
                "order_by": order_by,
            },
        )
        # -#
        if isinstance(by, str):
            by = [by]
        if isinstance(order_by, str):
            order_by = [order_by]
        check_types(
            [
                (
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
                ),
                ("expr", expr, [str]),
                ("by", by, [list]),
                ("order_by", order_by, [list]),
            ]
        )
        method = method.lower()
        self.parent.are_namecols_in([elem for elem in order_by] + by)
        by = self.parent.format_colnames(by)
        if method == "auto":
            method = "mean" if (self.isnum() and self.nunique(True) > 6) else "mode"
        total = self.count()
        if (method == "mode") and (val == None):
            val = self.mode(dropna=True)
            if val == None:
                warning_message = "The vColumn {} has no mode (only missing values).\nNothing was filled.".format(
                    self.alias
                )
                warnings.warn(warning_message, Warning)
                return self.parent
        if isinstance(val, str):
            val = val.replace("'", "''")
        if val != None:
            new_column = "COALESCE({}, '{}')".format("{}", val)
        elif expr:
            new_column = "COALESCE({}, {})".format("{}", expr)
        elif method == "0ifnull":
            new_column = "DECODE({}, NULL, 0, 1)"
        elif method in ("mean", "avg", "median"):
            fun = "MEDIAN" if (method == "median") else "AVG"
            if by == []:
                if fun == "AVG":
                    val = self.avg()
                elif fun == "MEDIAN":
                    val = self.median()
                new_column = "COALESCE({}, {})".format("{}", val)
            elif (len(by) == 1) and (self.parent[by[0]].nunique() < 50):
                try:
                    if fun == "MEDIAN":
                        fun = "APPROXIMATE_MEDIAN"
                    query = "SELECT /*+LABEL('vColumn.fillna')*/ {0}, {1}({2}) FROM {3} GROUP BY {0};".format(
                        by[0], fun, self.alias, self.parent.__genSQL__()
                    )
                    result = executeSQL(
                        query,
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
                        ", ".join(
                            ["{}, {}".format(elem[0], elem[1]) for elem in result]
                        ),
                    )
                    executeSQL(
                        "SELECT /*+LABEL('vColumn.fillna')*/ {} FROM {} LIMIT 1".format(
                            new_column.format(self.alias), self.parent.__genSQL__()
                        ),
                        print_time_sql=False,
                        sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
                        symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
                    )
                except:
                    new_column = "COALESCE({}, {}({}) OVER (PARTITION BY {}))".format(
                        "{}", fun, "{}", ", ".join(by)
                    )
            else:
                new_column = "COALESCE({}, {}({}) OVER (PARTITION BY {}))".format(
                    "{}", fun, "{}", ", ".join(by)
                )
        elif method in ("ffill", "pad", "bfill", "backfill"):
            assert order_by, ParameterError(
                "If the method is in ffill|pad|bfill|backfill then 'order_by' must be a list of at least one element to use to order the data"
            )
            desc = "" if (method in ("ffill", "pad")) else " DESC"
            partition_by = (
                "PARTITION BY {}".format(
                    ", ".join([quote_ident(column) for column in by])
                )
                if (by)
                else ""
            )
            order_by_ts = ", ".join([quote_ident(column) + desc for column in order_by])
            new_column = "COALESCE({}, LAST_VALUE({} IGNORE NULLS) OVER ({} ORDER BY {}))".format(
                "{}", "{}", partition_by, order_by_ts
            )
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
            raise QueryError("{}\nAn Error happened during the filling.".format(e))
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
            if verticapy.options["print_info"]:
                print("{} element{}filled.".format(total, conj))
            self.parent.__add_to_history__(
                "[Fillna]: {} {} missing value{} filled.".format(
                    total, self.alias, conj,
                )
            )
        else:
            if verticapy.options["print_info"]:
                print("Nothing was filled.")
            self.transformations = [elem for elem in copy_trans]
            for elem in sauv:
                self.catalog[elem] = sauv[elem]
        return self.parent

    # ---#
    def geo_plot(self, *args, **kwargs):
        """
    ---------------------------------------------------------------------------
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="geo_plot", path="vcolumn.vColumn", json_dict=kwargs,
        )
        # -#
        columns = [self.alias]
        check = True
        if len(args) > 0:
            column = args[0]
        elif "column" in kwargs:
            column = kwargs["column"]
        else:
            check = False
        if check:
            self.parent.are_namecols_in(column)
            column = self.parent.format_colnames(column)
            columns += [column]
            if not ("cmap" in kwargs):
                from verticapy.plot import gen_cmap

                kwargs["cmap"] = gen_cmap()[0]
        else:
            if not ("color" in kwargs):
                from verticapy.plot import gen_colors

                kwargs["color"] = gen_colors()[0]
        if not ("legend" in kwargs):
            kwargs["legend"] = True
        if not ("figsize" in kwargs):
            kwargs["figsize"] = (14, 10)
        return self.parent[columns].to_geopandas(self.alias).plot(*args, **kwargs)

    # ---#
    def get_dummies(
        self,
        prefix: str = "",
        prefix_sep: str = "_",
        drop_first: bool = True,
        use_numbers_as_suffix: bool = False,
    ):
        """
	---------------------------------------------------------------------------
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="one_hot_encode",
            path="vcolumn.vColumn",
            json_dict={
                "prefix": prefix,
                "prefix_sep": prefix_sep,
                "drop_first": drop_first,
                "use_numbers_as_suffix": use_numbers_as_suffix,
            },
        )
        # -#
        check_types(
            [
                ("prefix", prefix, [str]),
                ("prefix_sep", prefix_sep, [str]),
                ("drop_first", drop_first, [bool]),
                ("use_numbers_as_suffix", use_numbers_as_suffix, [bool]),
            ]
        )
        distinct_elements = self.distinct()
        if distinct_elements not in ([0, 1], [1, 0]) or self.isbool():
            all_new_features = []
            prefix = (
                self.alias.replace('"', "") + prefix_sep.replace('"', "_")
                if not (prefix)
                else prefix.replace('"', "_") + prefix_sep.replace('"', "_")
            )
            n = 1 if drop_first else 0
            for k in range(len(distinct_elements) - n):
                name = (
                    '"{}{}"'.format(prefix, k)
                    if (use_numbers_as_suffix)
                    else '"{}{}"'.format(
                        prefix, str(distinct_elements[k]).replace('"', "_")
                    )
                )
                assert not (self.parent.is_colname_in(name)), NameError(
                    f"A vColumn has already the alias of one of the dummies ({name}).\n"
                    "It can be the result of using previously the method on the vColumn "
                    "or simply because of ambiguous columns naming.\nBy changing one of "
                    "the parameters ('prefix', 'prefix_sep'), you'll be able to solve this "
                    "issue."
                )
            for k in range(len(distinct_elements) - n):
                name = (
                    '"{}{}"'.format(prefix, k)
                    if (use_numbers_as_suffix)
                    else '"{}{}"'.format(
                        prefix, str(distinct_elements[k]).replace('"', "_")
                    )
                )
                name = (
                    name.replace(" ", "_")
                    .replace("/", "_")
                    .replace(",", "_")
                    .replace("'", "_")
                )
                expr = "DECODE({}, '{}', 1, 0)".format(
                    "{}", str(distinct_elements[k]).replace("'", "''")
                )
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
                "[Get Dummies]: One hot encoder was applied to the vColumn {}\n{} feature{}created: {}".format(
                    self.alias, len(all_new_features), conj, ", ".join(all_new_features)
                )
                + "."
            )
        return self.parent

    one_hot_encode = get_dummies

    # ---#
    def get_len(self):
        """
    ---------------------------------------------------------------------------
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
        elem_to_select = "{0}({1})".format(fun, self.alias,)
        init_transf = "{0}({1})".format(fun, self.init_transf,)
        new_alias = quote_ident(self.alias[1:-1] + ".length")
        query = "(SELECT {0} AS {1} FROM {2}) VERTICAPY_SUBTABLE".format(
            elem_to_select, new_alias, self.parent.__genSQL__(),
        )
        vcol = vDataFrameSQL(query)[new_alias]
        vcol.init_transf = init_transf
        return vcol

    # ---#
    def head(self, limit: int = 5):
        """
	---------------------------------------------------------------------------
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
    def hist(
        self,
        method: str = "density",
        of: str = "",
        max_cardinality: int = 6,
        nbins: int = 0,
        h: float = 0,
        ax=None,
        **style_kwds,
    ):
        """
	---------------------------------------------------------------------------
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
 	h: float, optional
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="hist",
            path="vcolumn.vColumn",
            json_dict={
                **{
                    "method": method,
                    "of": of,
                    "max_cardinality": max_cardinality,
                    "h": h,
                    "nbins": nbins,
                },
                **style_kwds,
            },
        )
        # -#
        check_types(
            [
                ("method", method, [str]),
                ("of", of, [str]),
                ("max_cardinality", max_cardinality, [int, float]),
                ("h", h, [int, float]),
                ("nbins", nbins, [int, float]),
            ]
        )
        if of:
            self.parent.are_namecols_in(of)
            of = self.parent.format_colnames(of)
        from verticapy.plot import hist

        return hist(self, method, of, max_cardinality, nbins, h, ax=ax, **style_kwds)

    # ---#
    def iloc(self, limit: int = 5, offset: int = 0):
        """
    ---------------------------------------------------------------------------
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
        # -#
        check_types([("limit", limit, [int, float]), ("offset", offset, [int, float])])
        if offset < 0:
            offset = max(0, self.parent.shape()[0] - limit)
        title = "Reads {}.".format(self.alias)
        tail = to_tablesample(
            "SELECT {} AS {} FROM {}{} LIMIT {} OFFSET {}".format(
                bin_spatial_to_str(self.category(), self.alias),
                self.alias,
                self.parent.__genSQL__(),
                self.parent.__get_last_order_by__(),
                limit,
                offset,
            ),
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
    ---------------------------------------------------------------------------
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
    ---------------------------------------------------------------------------
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
	---------------------------------------------------------------------------
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
    def isin(self, val: list, *args):
        """
	---------------------------------------------------------------------------
	Looks if some specific records are in the vColumn and it returns the new 
    vDataFrame of the search.

	Parameters
 	----------
 	val: list
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="isin", path="vcolumn.vColumn", json_dict={"val": val,},
        )
        # -#
        if isinstance(val, str) or not (isinstance(val, Iterable)):
            val = [val]
        val += list(args)
        check_types([("val", val, [list])])
        val = {self.alias: val}
        return self.parent.isin(val)

    # ---#
    def isnum(self):
        """
	---------------------------------------------------------------------------
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
    ---------------------------------------------------------------------------
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
    def iv_woe(self, y: str, nbins: int = 10):
        """
    ---------------------------------------------------------------------------
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="iv_woe", path="vcolumn.vColumn", json_dict={"y": y, "nbins": nbins,},
        )
        # -#
        check_types([("y", y, [str]), ("nbins", nbins, [int])])
        self.parent.are_namecols_in(y)
        y = self.parent.format_colnames(y)
        assert self.parent[y].nunique() == 2, TypeError(
            "vColumn {} must be binary to use iv_woe.".format(y)
        )
        response_cat = self.parent[y].distinct()
        response_cat.sort()
        assert response_cat == [0, 1], TypeError(
            "vColumn {} must be binary to use iv_woe.".format(y)
        )
        self.parent[y].distinct()
        trans = self.discretize(
            method="same_width" if self.isnum() else "topk",
            nbins=nbins,
            k=nbins,
            new_category="Others",
            return_enum_trans=True,
        )[0].replace("{}", self.alias)
        query = "SELECT {} AS {}, {} AS ord, {}::int AS {} FROM {}".format(
            trans, self.alias, self.alias, y, y, self.parent.__genSQL__(),
        )
        query = "SELECT {}, MIN(ord) AS ord, SUM(1 - {}) AS non_events, SUM({}) AS events FROM ({}) x GROUP BY 1".format(
            self.alias, y, y, query,
        )
        query = "SELECT {}, ord, non_events, events, non_events / NULLIFZERO(SUM(non_events) OVER ()) AS pt_non_events, events / NULLIFZERO(SUM(events) OVER ()) AS pt_events FROM ({}) x".format(
            self.alias, query,
        )
        query = "SELECT {} AS index, non_events, events, pt_non_events, pt_events, CASE WHEN non_events = 0 OR events = 0 THEN 0 ELSE ZEROIFNULL(LN(pt_non_events / NULLIFZERO(pt_events))) END AS woe, CASE WHEN non_events = 0 OR events = 0 THEN 0 ELSE (pt_non_events - pt_events) * ZEROIFNULL(LN(pt_non_events / NULLIFZERO(pt_events))) END AS iv FROM ({}) x ORDER BY ord".format(
            self.alias, query,
        )
        title = "Computing WOE & IV of {} (response = {}).".format(self.alias, y)
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
    def kurtosis(self):
        """
	---------------------------------------------------------------------------
	Aggregates the vColumn using 'kurtosis'.

 	Returns
 	-------
 	float
 		kurtosis

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
        # Saving information to the query profile table
        save_to_query_profile(
            name="kurtosis", path="vcolumn.vColumn", json_dict={},
        )
        # -#
        return self.aggregate(["kurtosis"]).values[self.alias][0]

    kurt = kurtosis
    # ---#
    def label_encode(self):
        """
	---------------------------------------------------------------------------
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="label_encode", path="vcolumn.vColumn", json_dict={},
        )
        # -#
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
                expr += [
                    "'{}', {}".format(str(distinct_elements[k]).replace("'", "''"), k)
                ]
                text_info += "\t{} => {}".format(distinct_elements[k], k)
            expr = ", ".join(expr) + ", {})".format(len(distinct_elements))
            self.transformations += [(expr, "int", "int")]
            self.parent.__update_catalog__(erase=True, columns=[self.alias])
            self.catalog["count"] = self.parent.shape()[0]
            self.catalog["percent"] = 100
            self.parent.__add_to_history__(
                "[Label Encoding]: Label Encoding was applied to the vColumn {} using the following mapping:{}".format(
                    self.alias, text_info
                )
            )
        return self.parent

    # ---#
    def mad(self):
        """
	---------------------------------------------------------------------------
	Aggregates the vColumn using 'mad' (median absolute deviation).

 	Returns
 	-------
 	float
 		mad

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
        # Saving information to the query profile table
        save_to_query_profile(
            name="mad", path="vcolumn.vColumn", json_dict={},
        )
        # -#
        return self.aggregate(["mad"]).values[self.alias][0]

    # ---#
    def max(self):
        """
	---------------------------------------------------------------------------
	Aggregates the vColumn using 'max' (Maximum).

 	Returns
 	-------
 	float/str
 		maximum

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
        # Saving information to the query profile table
        save_to_query_profile(
            name="max", path="vcolumn.vColumn", json_dict={},
        )
        # -#
        return self.aggregate(["max"]).values[self.alias][0]

    # ---#
    def mean_encode(self, response: str):
        """
	---------------------------------------------------------------------------
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="mean_encode",
            path="vcolumn.vColumn",
            json_dict={"response": response,},
        )
        # -#
        check_types([("response", response, [str])])
        self.parent.are_namecols_in(response)
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
            ("AVG({}) OVER (PARTITION BY {})".format(response, "{}"), "int", "float")
        ]
        self.parent.__update_catalog__(erase=True, columns=[self.alias])
        self.parent.__add_to_history__(
            "[Mean Encode]: The vColumn {} was transformed using a mean encoding with {} as Response Column.".format(
                self.alias, response
            )
        )
        if verticapy.options["print_info"]:
            print("The mean encoding was successfully done.")
        return self.parent

    # ---#
    def median(
        self, approx: bool = True,
    ):
        """
	---------------------------------------------------------------------------
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="median", path="vcolumn.vColumn", json_dict={"approx": approx,},
        )
        # -#
        return self.quantile(0.5, approx=approx)

    # ---#
    def memory_usage(self):
        """
	---------------------------------------------------------------------------
	Returns the vColumn memory usage. 

 	Returns
 	-------
 	float
 		vColumn memory usage (byte)

	See Also
	--------
	vDataFrame.memory_usage : Returns the vDataFrame memory usage.
		"""
        # Saving information to the query profile table
        save_to_query_profile(
            name="memory_usage", path="vcolumn.vColumn", json_dict={},
        )
        # -#
        import sys

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
    def min(self):
        """
	---------------------------------------------------------------------------
	Aggregates the vColumn using 'min' (Minimum).

 	Returns
 	-------
 	float/str
 		minimum

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
        # Saving information to the query profile table
        save_to_query_profile(
            name="min", path="vcolumn.vColumn", json_dict={},
        )
        # -#
        return self.aggregate(["min"]).values[self.alias][0]

    # ---#
    def mode(self, dropna: bool = False, n: int = 1):
        """
	---------------------------------------------------------------------------
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="mode", path="vcolumn.vColumn", json_dict={"dropna": dropna, "n": n,},
        )
        # -#
        check_types([("dropna", dropna, [bool]), ("n", n, [int, float])])
        if n == 1:
            pre_comp = self.parent.__get_catalog_value__(self.alias, "top")
            if pre_comp != "VERTICAPY_NOT_PRECOMPUTED":
                if not (dropna) and (pre_comp != None):
                    return pre_comp
        assert n >= 1, ParameterError("Parameter 'n' must be greater or equal to 1")
        where = " WHERE {} IS NOT NULL ".format(self.alias) if (dropna) else " "
        result = executeSQL(
            "SELECT /*+LABEL('vColumn.mode')*/ {0} FROM (SELECT {0}, COUNT(*) AS _verticapy_cnt_ FROM {1}{2}GROUP BY {0} ORDER BY _verticapy_cnt_ DESC LIMIT {3}) VERTICAPY_SUBTABLE ORDER BY _verticapy_cnt_ ASC LIMIT 1".format(
                self.alias, self.parent.__genSQL__(), where, n
            ),
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
            self.parent.__update_catalog__(
                {"index": ["top{}".format(n)], self.alias: [top]}
            )
        return top

    # ---#
    def mul(self, x: float):
        """
	---------------------------------------------------------------------------
	Multiplies the vColumn by the input element.

	Parameters
 	----------
 	x: float
 		Input number.

 	Returns
 	-------
 	vDataFrame
		self.parent

	See Also
	--------
	vDataFrame[].apply : Applies a function to the input vColumn.
		"""
        # Saving information to the query profile table
        save_to_query_profile(
            name="mul", path="vcolumn.vColumn", json_dict={"x": x,},
        )
        # -#
        check_types([("x", x, [int, float])])
        return self.apply(func="{} * ({})".format("{}", x))

    # ---#
    def nlargest(self, n: int = 10):
        """
	---------------------------------------------------------------------------
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="nlargest", path="vcolumn.vColumn", json_dict={"n": n,},
        )
        # -#
        check_types([("n", n, [int, float])])
        query = "SELECT * FROM {0} WHERE {1} IS NOT NULL ORDER BY {1} DESC LIMIT {2}".format(
            self.parent.__genSQL__(), self.alias, n
        )
        title = "Reads {} {} largest elements.".format(self.alias, n)
        return to_tablesample(
            query,
            title=title,
            sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
            symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
        )

    # ---#
    def normalize(
        self, method: str = "zscore", by: list = [], return_trans: bool = False
    ):
        """
	---------------------------------------------------------------------------
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
	by: list, optional
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="normalize",
            path="vcolumn.vColumn",
            json_dict={"method": method, "by": by, "return_trans": return_trans,},
        )
        # -#
        if isinstance(by, str):
            by = [by]
        check_types(
            [
                ("method", method, ["zscore", "robust_zscore", "minmax"]),
                ("by", by, [list]),
                ("return_trans", return_trans, [bool]),
            ]
        )
        method = method.lower()
        self.parent.are_namecols_in(by)
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
                        warning_message = "Can not normalize {} using a Z-Score - The Standard Deviation is null !".format(
                            self.alias
                        )
                        warnings.warn(warning_message, Warning)
                        return self
                elif (n == 1) and (self.parent[by[0]].nunique() < 50):
                    try:
                        result = executeSQL(
                            "SELECT /*+LABEL('vColumn.normalize')*/ {0}, AVG({1}), STDDEV({1}) FROM {2} GROUP BY {0}".format(
                                by[0], self.alias, self.parent.__genSQL__(),
                            ),
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
                            "SELECT /*+LABEL('vColumn.normalize')*/ {}, {} FROM {} LIMIT 1".format(
                                avg, stddev, self.parent.__genSQL__()
                            ),
                            print_time_sql=False,
                            sql_push_ext=self.parent._VERTICAPY_VARIABLES_[
                                "sql_push_ext"
                            ],
                            symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
                        )
                    except:
                        avg, stddev = (
                            "AVG({}) OVER (PARTITION BY {})".format(
                                self.alias, ", ".join(by)
                            ),
                            "STDDEV({}) OVER (PARTITION BY {})".format(
                                self.alias, ", ".join(by)
                            ),
                        )
                else:
                    avg, stddev = (
                        "AVG({}) OVER (PARTITION BY {})".format(
                            self.alias, ", ".join(by)
                        ),
                        "STDDEV({}) OVER (PARTITION BY {})".format(
                            self.alias, ", ".join(by)
                        ),
                    )
                if return_trans:
                    return "({} - {}) / {}({})".format(
                        self.alias, avg, "NULLIFZERO" if (nullifzero) else "", stddev
                    )
                else:
                    final_transformation = [
                        (
                            "({} - {}) / {}({})".format(
                                "{}", avg, "NULLIFZERO" if (nullifzero) else "", stddev
                            ),
                            "float",
                            "float",
                        )
                    ]

            elif method == "robust_zscore":

                if n > 0:
                    warning_message = "The method 'robust_zscore' is available only if the parameter 'by' is empty\nIf you want to normalize by grouping by elements, please use a method in zscore|minmax"
                    warnings.warn(warning_message, Warning)
                    return self
                mad, med = self.aggregate(["mad", "approx_median"]).values[self.alias]
                mad *= 1.4826
                if mad != 0:
                    if return_trans:
                        return "({} - {}) / ({})".format(self.alias, med, mad)
                    else:
                        final_transformation = [
                            (
                                "({} - {}) / ({})".format("{}", med, mad),
                                "float",
                                "float",
                            )
                        ]
                else:
                    warning_message = "Can not normalize {} using a Robust Z-Score - The MAD is null !".format(
                        self.alias
                    )
                    warnings.warn(warning_message, Warning)
                    return self

            elif method == "minmax":

                if n == 0:
                    nullifzero = 0
                    cmin, cmax = self.aggregate(["min", "max"]).values[self.alias]
                    if cmax - cmin == 0:
                        warning_message = "Can not normalize {} using the MIN and the MAX. MAX = MIN !".format(
                            self.alias
                        )
                        warnings.warn(warning_message, Warning)
                        return self
                elif n == 1:
                    try:
                        result = executeSQL(
                            "SELECT /*+LABEL('vColumn.normalize')*/ {0}, MIN({1}), MAX({1}) FROM {2} GROUP BY {0}".format(
                                by[0], self.alias, self.parent.__genSQL__(),
                            ),
                            title="Computing the different categories {} to normalize.".format(
                                by[0]
                            ),
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
                            "SELECT /*+LABEL('vColumn.normalize')*/ {}, {} FROM {} LIMIT 1".format(
                                cmax, cmin, self.parent.__genSQL__()
                            ),
                            print_time_sql=False,
                            sql_push_ext=self.parent._VERTICAPY_VARIABLES_[
                                "sql_push_ext"
                            ],
                            symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
                        )
                    except:
                        cmax, cmin = (
                            "MAX({}) OVER (PARTITION BY {})".format(
                                self.alias, ", ".join(by)
                            ),
                            "MIN({}) OVER (PARTITION BY {})".format(
                                self.alias, ", ".join(by)
                            ),
                        )
                else:
                    cmax, cmin = (
                        "MAX({}) OVER (PARTITION BY {})".format(
                            self.alias, ", ".join(by)
                        ),
                        "MIN({}) OVER (PARTITION BY {})".format(
                            self.alias, ", ".join(by)
                        ),
                    )
                if return_trans:
                    return "({0} - {1}) / {2}({3} - {1})".format(
                        self.alias, cmin, "NULLIFZERO" if (nullifzero) else "", cmax,
                    )
                else:
                    final_transformation = [
                        (
                            "({0} - {1}) / {2}({3} - {1})".format(
                                "{}", cmin, "NULLIFZERO" if (nullifzero) else "", cmax,
                            ),
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
                "[Normalize]: The vColumn '{}' was normalized with the method '{}'.".format(
                    self.alias, method
                )
            )
        else:
            raise TypeError("The vColumn must be numerical for Normalization")
        return self.parent

    # ---#
    def nsmallest(self, n: int = 10):
        """
	---------------------------------------------------------------------------
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="nsmallest", path="vcolumn.vColumn", json_dict={"n": n,},
        )
        # -#
        check_types([("n", n, [int, float])])
        query = "SELECT * FROM {0} WHERE {1} IS NOT NULL ORDER BY {1} ASC LIMIT {2}".format(
            self.parent.__genSQL__(), self.alias, n
        )
        title = "Reads {} {} smallest elements.".format(n, self.alias)
        return to_tablesample(
            query,
            title=title,
            sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
            symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
        )

    # ---#
    def numh(self, method: str = "auto"):
        """
	---------------------------------------------------------------------------
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
        # -#
        check_types(
            [("method", method, ["sturges", "freedman_diaconis", "fd", "auto"])]
        )
        method = method.lower()
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
            min_date = self.min()
            table = "(SELECT DATEDIFF('second', '{0}'::timestamp, {1}) AS {1} FROM {2}) VERTICAPY_OPTIMAL_H_TABLE".format(
                min_date, self.alias, self.parent.__genSQL__()
            )
            query = "SELECT /*+LABEL('vColumn.numh')*/ COUNT({0}) AS NAs, MIN({0}) AS min, APPROXIMATE_PERCENTILE({0} USING PARAMETERS percentile = 0.25) AS Q1, APPROXIMATE_PERCENTILE({0} USING PARAMETERS percentile = 0.75) AS Q3, MAX({0}) AS max FROM {1}".format(
                self.alias, table
            )
            result = executeSQL(
                query,
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
    def nunique(self, approx: bool = True):
        """
	---------------------------------------------------------------------------
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="nunique", path="vcolumn.vColumn", json_dict={"approx": approx,},
        )
        # -#
        check_types([("approx", approx, [bool])])
        if approx:
            return self.aggregate(func=["approx_unique"]).values[self.alias][0]
        else:
            return self.aggregate(func=["unique"]).values[self.alias][0]

    # ---#
    def pie(
        self,
        method: str = "density",
        of: str = "",
        max_cardinality: int = 6,
        h: float = 0,
        pie_type: str = "auto",
        ax=None,
        **style_kwds,
    ):
        """
	---------------------------------------------------------------------------
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
 	h: float, optional
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="pie",
            path="vcolumn.vColumn",
            json_dict={
                **{
                    "method": method,
                    "of": of,
                    "max_cardinality": max_cardinality,
                    "h": h,
                    "pie_type": pie_type,
                },
                **style_kwds,
            },
        )
        # -#
        if isinstance(pie_type, str):
            pie_type = pie_type.lower()
        check_types(
            [
                ("method", method, [str]),
                ("of", of, [str]),
                ("max_cardinality", max_cardinality, [int, float]),
                ("h", h, [int, float]),
                ("pie_type", pie_type, ["auto", "donut", "rose"]),
            ]
        )
        donut = True if pie_type == "donut" else False
        rose = True if pie_type == "rose" else False
        if of:
            self.parent.are_namecols_in(of)
            of = self.parent.format_colnames(of)
        from verticapy.plot import pie

        return pie(
            self, method, of, max_cardinality, h, donut, rose, ax=None, **style_kwds,
        )

    # ---#
    def plot(
        self,
        ts: str,
        by: str = "",
        start_date: Union[str, datetime.datetime, datetime.date] = "",
        end_date: Union[str, datetime.datetime, datetime.date] = "",
        area: bool = False,
        step: bool = False,
        ax=None,
        **style_kwds,
    ):
        """
	---------------------------------------------------------------------------
	Draws the Time Series of the vColumn.

	Parameters
 	----------
 	ts: str
 		TS (Time Series) vColumn to use to order the data. The vColumn type must be
 		date like (date, datetime, timestamp...) or numerical.
 	by: str, optional
 		vColumn to use to partition the TS.
 	start_date: str / date, optional
 		Input Start Date. For example, time = '03-11-1993' will filter the data when 
 		'ts' is lesser than November 1993 the 3rd.
 	end_date: str / date, optional
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="plot",
            path="vcolumn.vColumn",
            json_dict={
                **{
                    "ts": ts,
                    "by": by,
                    "start_date": start_date,
                    "end_date": end_date,
                    "area": area,
                    "step": step,
                },
                **style_kwds,
            },
        )
        # -#
        check_types(
            [
                ("ts", ts, [str]),
                ("by", by, [str]),
                ("start_date", start_date, [str, datetime.datetime, datetime.date]),
                ("end_date", end_date, [str, datetime.datetime, datetime.date]),
                ("area", area, [bool]),
                ("step", step, [bool]),
            ]
        )
        self.parent.are_namecols_in(ts)
        ts = self.parent.format_colnames(ts)
        if by:
            self.parent.are_namecols_in(by)
            by = self.parent.format_colnames(by)
        from verticapy.plot import ts_plot

        return ts_plot(
            self, ts, by, start_date, end_date, area, step, ax=ax, **style_kwds,
        )

    # ---#
    def product(self):
        """
	---------------------------------------------------------------------------
	Aggregates the vColumn using 'product'.

 	Returns
 	-------
 	float
 		product

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
        # Saving information to the query profile table
        save_to_query_profile(
            name="product", path="vcolumn.vColumn", json_dict={},
        )
        # -#
        return self.aggregate(func=["prod"]).values[self.alias][0]

    prod = product

    # ---#
    def quantile(self, x: float, approx: bool = True):
        """
	---------------------------------------------------------------------------
	Aggregates the vColumn using an input 'quantile'.

	Parameters
 	----------
 	x: float
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="quantile",
            path="vcolumn.vColumn",
            json_dict={"x": x, "approx": approx,},
        )
        # -#
        check_types([("x", x, [int, float], ("approx", approx, [bool]))])
        prefix = "approx_" if approx else ""
        return self.aggregate(func=[prefix + "{}%".format(x * 100)]).values[self.alias][
            0
        ]

    # ---#
    def range_plot(
        self,
        ts: str,
        q: tuple = (0.25, 0.75),
        start_date: Union[str, datetime.datetime, datetime.date] = "",
        end_date: Union[str, datetime.datetime, datetime.date] = "",
        plot_median: bool = False,
        ax=None,
        **style_kwds,
    ):
        """
    ---------------------------------------------------------------------------
    Draws the range plot of the vColumn. The aggregations used are the median 
    and two input quantiles.

    Parameters
    ----------
    ts: str
        TS (Time Series) vColumn to use to order the data. The vColumn type must be
        date like (date, datetime, timestamp...) or numerical.
    q: tuple, optional
        Tuple including the 2 quantiles used to draw the Plot.
    start_date: str / date, optional
        Input Start Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is lesser than November 1993 the 3rd.
    end_date: str / date, optional
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="range_plot",
            path="vcolumn.vColumn",
            json_dict={
                **{
                    "ts": ts,
                    "q": q,
                    "start_date": start_date,
                    "end_date": end_date,
                    "plot_median": plot_median,
                },
                **style_kwds,
            },
        )
        # -#
        check_types(
            [
                ("ts", ts, [str]),
                ("q", q, [tuple]),
                (
                    "start_date",
                    start_date,
                    [str, datetime.datetime, datetime.date, int, float],
                ),
                (
                    "end_date",
                    end_date,
                    [str, datetime.datetime, datetime.date, int, float],
                ),
                ("plot_median", plot_median, [bool]),
            ]
        )
        self.parent.are_namecols_in(ts)
        ts = self.parent.format_colnames(ts)
        from verticapy.plot import range_curve_vdf

        return range_curve_vdf(
            self, ts, q, start_date, end_date, plot_median, ax=ax, **style_kwds,
        )

    # ---#
    def rename(self, new_name: str):
        """
	---------------------------------------------------------------------------
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
        check_types([("new_name", new_name, [str])])
        old_name = quote_ident(self.alias)
        new_name = new_name.replace('"', "")
        assert not (self.parent.is_colname_in(new_name)), NameError(
            f"A vColumn has already the alias {new_name}.\nBy changing the parameter 'new_name', you'll be able to solve this issue."
        )
        self.add_copy(new_name)
        parent = self.drop(add_history=False)
        parent.__add_to_history__(
            "[Rename]: The vColumn {} was renamed '{}'.".format(old_name, new_name)
        )
        return parent

    # ---#
    def round(self, n: int):
        """
	---------------------------------------------------------------------------
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="round", path="vcolumn.vColumn", json_dict={"n": n,},
        )
        # -#
        check_types([("n", n, [int, float])])
        return self.apply(func="ROUND({}, {})".format("{}", n))

    # ---#
    def sem(self):
        """
	---------------------------------------------------------------------------
	Aggregates the vColumn using 'sem' (standard error of mean).

 	Returns
 	-------
 	float
 		sem

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
        # Saving information to the query profile table
        save_to_query_profile(
            name="sem", path="vcolumn.vColumn", json_dict={},
        )
        # -#
        return self.aggregate(["sem"]).values[self.alias][0]

    # ---#
    def skewness(self):
        """
	---------------------------------------------------------------------------
	Aggregates the vColumn using 'skewness'.

 	Returns
 	-------
 	float
 		skewness

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
        # Saving information to the query profile table
        save_to_query_profile(
            name="skewness", path="vcolumn.vColumn", json_dict={},
        )
        # -#
        return self.aggregate(["skewness"]).values[self.alias][0]

    skew = skewness
    # ---#
    def slice(self, length: int, unit: str = "second", start: bool = True):
        """
	---------------------------------------------------------------------------
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="slice",
            path="vcolumn.vColumn",
            json_dict={"length": length, "unit": unit, "start": start,},
        )
        # -#
        check_types(
            [
                ("length", length, [int, float]),
                ("unit", unit, [str]),
                ("start", start, [bool]),
            ]
        )
        start_or_end = "START" if (start) else "END"
        return self.apply(
            func="TIME_SLICE({}, {}, '{}', '{}')".format(
                "{}", length, unit.upper(), start_or_end
            )
        )

    # ---#
    def spider(
        self,
        by: str = "",
        method: str = "density",
        of: str = "",
        max_cardinality: Union[int, tuple] = (6, 6),
        h: Union[int, float, tuple] = (None, None),
        ax=None,
        **style_kwds,
    ):
        """
    ---------------------------------------------------------------------------
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
    h: int/float/tuple, optional
        Interval width of the vColumns 1 and 2 bars. It is only valid if the 
        vColumns are numerical. Optimized h will be computed if the parameter 
        is empty or invalid.
    max_cardinality: int/tuple, optional
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="spider",
            path="vcolumn.vColumn",
            json_dict={
                **{
                    "by": by,
                    "method": method,
                    "of": of,
                    "max_cardinality": max_cardinality,
                    "h": h,
                },
                **style_kwds,
            },
        )
        # -#
        check_types(
            [
                ("by", by, [str]),
                ("method", method, [str]),
                ("of", of, [str]),
                ("max_cardinality", max_cardinality, [list]),
                ("h", h, [list, float, int]),
            ]
        )
        if by:
            self.parent.are_namecols_in(by)
            by = self.parent.format_colnames(by)
            columns = [self.alias, by]
        else:
            columns = [self.alias]
        if of:
            self.parent.are_namecols_in(of)
            of = self.parent.format_colnames(of)
        from verticapy.plot import spider as spider_plot

        return spider_plot(
            self.parent, columns, method, of, max_cardinality, h, ax=ax, **style_kwds,
        )

    # ---#
    def std(self):
        """
	---------------------------------------------------------------------------
	Aggregates the vColumn using 'std' (Standard Deviation).

 	Returns
 	-------
 	float
 		std

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
        # Saving information to the query profile table
        save_to_query_profile(
            name="std", path="vcolumn.vColumn", json_dict={},
        )
        # -#
        return self.aggregate(["stddev"]).values[self.alias][0]

    stddev = std
    # ---#
    def store_usage(self):
        """
	---------------------------------------------------------------------------
	Returns the vColumn expected store usage (unit: b).

 	Returns
 	-------
 	int
 		vColumn expected store usage.

	See Also
	--------
	vDataFrame.expected_store_usage : Returns the vDataFrame expected store usage.
		"""
        # Saving information to the query profile table
        save_to_query_profile(
            name="store_usage", path="vcolumn.vColumn", json_dict={},
        )
        # -#
        pre_comp = self.parent.__get_catalog_value__(self.alias, "store_usage")
        if pre_comp != "VERTICAPY_NOT_PRECOMPUTED":
            return pre_comp
        store_usage = executeSQL(
            "SELECT /*+LABEL('vColumn.storage_usage')*/ ZEROIFNULL(SUM(LENGTH({}::varchar))) FROM {}".format(
                bin_spatial_to_str(self.category(), self.alias),
                self.parent.__genSQL__(),
            ),
            title="Computing the Store Usage of the vColumn {}.".format(self.alias),
            method="fetchfirstelem",
            sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
            symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
        )
        self.parent.__update_catalog__(
            {"index": ["store_usage"], self.alias: [store_usage]}
        )
        return store_usage

    # ---#
    def str_contains(self, pat: str):
        """
	---------------------------------------------------------------------------
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="str_contains", path="vcolumn.vColumn", json_dict={"pat": pat,},
        )
        # -#
        check_types([("pat", pat, [str])])
        return self.apply(
            func="REGEXP_COUNT({}, '{}') > 0".format("{}", pat.replace("'", "''"))
        )

    # ---#
    def str_count(self, pat: str):
        """
	---------------------------------------------------------------------------
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="str_count", path="vcolumn.vColumn", json_dict={"pat": pat,},
        )
        # -#
        check_types([("pat", pat, [str])])
        return self.apply(
            func="REGEXP_COUNT({}, '{}')".format("{}", pat.replace("'", "''"))
        )

    # ---#
    def str_extract(self, pat: str):
        """
	---------------------------------------------------------------------------
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="str_extract", path="vcolumn.vColumn", json_dict={"pat": pat,},
        )
        # -#
        check_types([("pat", pat, [str])])
        return self.apply(
            func="REGEXP_SUBSTR({}, '{}')".format("{}", pat.replace("'", "''"))
        )

    # ---#
    def str_replace(self, to_replace: str, value: str = ""):
        """
	---------------------------------------------------------------------------
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="str_replace",
            path="vcolumn.vColumn",
            json_dict={"to_replace": to_replace, "value": value,},
        )
        # -#
        check_types([("to_replace", to_replace, [str]), ("value", value, [str])])
        return self.apply(
            func="REGEXP_REPLACE({}, '{}', '{}')".format(
                "{}", to_replace.replace("'", "''"), value.replace("'", "''")
            )
        )

    # ---#
    def str_slice(self, start: int, step: int):
        """
	---------------------------------------------------------------------------
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="str_slice",
            path="vcolumn.vColumn",
            json_dict={"start": start, "step": step,},
        )
        # -#
        check_types([("start", start, [int, float]), ("step", step, [int, float])])
        return self.apply(func="SUBSTR({}, {}, {})".format("{}", start, step))

    # ---#
    def sub(self, x: float):
        """
	---------------------------------------------------------------------------
	Subtracts the input element from the vColumn.

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
        # Saving information to the query profile table
        save_to_query_profile(
            name="sub", path="vcolumn.vColumn", json_dict={"x": x,},
        )
        # -#
        check_types([("x", x, [int, float])])
        if self.isdate():
            return self.apply(func="TIMESTAMPADD(SECOND, -({}), {})".format(x, "{}"))
        else:
            return self.apply(func="{} - ({})".format("{}", x))

    # ---#
    def sum(self):
        """
	---------------------------------------------------------------------------
	Aggregates the vColumn using 'sum'.

 	Returns
 	-------
 	float
 		sum

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
        # Saving information to the query profile table
        save_to_query_profile(
            name="sum", path="vcolumn.vColumn", json_dict={},
        )
        # -#
        return self.aggregate(["sum"]).values[self.alias][0]

    # ---#
    def tail(self, limit: int = 5):
        """
	---------------------------------------------------------------------------
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
    def topk(self, k: int = -1, dropna: bool = True):
        """
	---------------------------------------------------------------------------
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="topk", path="vcolumn.vColumn", json_dict={"k": k, "dropna": dropna,},
        )
        # -#
        check_types([("k", k, [int, float]), ("dropna", dropna, [bool])])
        topk = "" if (k < 1) else "LIMIT {}".format(k)
        dropna = " WHERE {} IS NOT NULL".format(self.alias) if (dropna) else ""
        query = "SELECT /*+LABEL('vColumn.topk')*/ {0} AS {1}, COUNT(*) AS _verticapy_cnt_, 100 * COUNT(*) / {2} AS percent FROM {3}{4} GROUP BY {0} ORDER BY _verticapy_cnt_ DESC {5}".format(
            bin_spatial_to_str(self.category(), self.alias),
            self.alias,
            self.parent.shape()[0],
            self.parent.__genSQL__(),
            dropna,
            topk,
        )
        result = executeSQL(
            query,
            title="Computing the top{} categories of {}.".format(
                k if k > 0 else "", self.alias
            ),
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
    def value_counts(self, k: int = 30):
        """
	---------------------------------------------------------------------------
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
        # Saving information to the query profile table
        save_to_query_profile(
            name="value_counts", path="vcolumn.vColumn", json_dict={"k": k,},
        )
        # -#
        return self.describe(method="categorical", max_cardinality=k)

    # ---#
    def var(self):
        """
	---------------------------------------------------------------------------
	Aggregates the vColumn using 'var' (Variance).

 	Returns
 	-------
 	float
 		var

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
        # Saving information to the query profile table
        save_to_query_profile(
            name="var", path="vcolumn.vColumn", json_dict={},
        )
        # -#
        return self.aggregate(["variance"]).values[self.alias][0]

    variance = var
