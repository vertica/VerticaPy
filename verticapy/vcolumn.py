# (c) Copyright [2018-2021] Micro Focus or one of its affiliates.
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
# VerticaPy is a Python library with scikit-like functionality to use to conduct
# data science projects on data stored in Vertica, taking advantage Vertica’s
# speed and built-in analytics and machine learning features. It supports the
# entire data science life cycle, uses a ‘pipeline’ mechanism to sequentialize
# data transformation operations, and offers beautiful graphical options.
#
# VerticaPy aims to solve all of these problems. The idea is simple: instead
# of moving data around for processing, VerticaPy brings the logic to the data.
#
#
# Modules
#
# Standard Python Modules
import math, re, decimal, warnings, datetime
from collections.abc import Iterable

# VerticaPy Modules
import verticapy
from verticapy.utilities import *
from verticapy.toolbox import *
from verticapy.errors import *

##
#
#   __   __   ______     ______     __         __  __     __    __     __   __
#  /\ \ / /  /\  ___\   /\  __ \   /\ \       /\ \/\ \   /\ "-./  \   /\ "-.\ \
#  \ \ \'/   \ \ \____  \ \ \/\ \  \ \ \____  \ \ \_\ \  \ \ \-./\ \  \ \ \-.  \
#   \ \__|    \ \_____\  \ \_____\  \ \_____\  \ \_____\  \ \_\ \ \_\  \ \_\\"\_\
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

    # ---#
    def __getitem__(self, index):
        if isinstance(index, slice):
            assert index.step in (1, None), ValueError("vColumn doesn't allow slicing having steps different than 1.")
            index_stop = index.stop
            index_start = index.start
            if not (isinstance(index_start, int)):
                index_start = 0
            if index_start < 0:
                index_start += self.parent.shape()[0]
            if isinstance(index_stop, int):
                if index_stop < 0:
                    index_stop += self.parent.shape()[0]
                limit = index_stop - index_start
                if limit <= 0:
                    limit = 0
                limit = " LIMIT {}".format(limit)
            else:
                limit = ""
            query = "(SELECT {} FROM {}{} OFFSET {}{}) VERTICAPY_SUBTABLE".format(
                self.alias,
                self.parent.__genSQL__(),
                last_order_by(self.parent),
                index_start,
                limit,
            )
            return vdf_from_relation(
                query, cursor=self.parent._VERTICAPY_VARIABLES_["cursor"]
            )
        elif isinstance(index, int):
            cast = "::float" if self.category() == "float" else ""
            if index < 0:
                index += self.parent.shape()[0]
            query = "SELECT {}{} FROM {}{} OFFSET {} LIMIT 1".format(
                self.alias,
                cast,
                self.parent.__genSQL__(),
                last_order_by(self.parent),
                index,
            )
            self.parent.__executeSQL__(query=query, title="Gets the vColumn element.")
            return self.parent._VERTICAPY_VARIABLES_["cursor"].fetchone()[0]
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

    # ---#
    def __executeSQL__(self, query: str, title: str = ""):
        """
		Same as parent.__executeSQL__
		It is to use to simplify the execution of some methods.
		"""
        return self.parent.__executeSQL__(query=query, title=title)

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
        check_types([("x", x, [int, float],)])
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
        check_types([("name", name, [str],)])
        name = str_column(name.replace('"', "_"))
        assert name.replace('"', ""), EmptyParameter("The parameter 'name' must not be empty")
        assert not(column_check_ambiguous(name, self.parent.get_columns())), NameError(f"A vColumn has already the alias {name}.\nBy changing the parameter 'name', you'll be able to solve this issue.")
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
        if isinstance(func, str_sql):
            func = str(func)
        check_types(
            [("func", func, [str],), ("copy_name", copy_name, [str],),]
        )
        try:
            try:
                ctype = get_data_types(
                    "SELECT {} AS apply_test_feature FROM {} WHERE {} IS NOT NULL LIMIT 0".format(
                        func.replace("{}", self.alias),
                        self.parent.__genSQL__(),
                        self.alias,
                    ),
                    self.parent._VERTICAPY_VARIABLES_["cursor"],
                    "apply_test_feature",
                    self.parent._VERTICAPY_VARIABLES_["schema_writing"],
                )
            except:
                ctype = get_data_types(
                    "SELECT {} AS apply_test_feature FROM {} WHERE {} IS NOT NULL LIMIT 0".format(
                        func.replace("{}", self.alias),
                        self.parent.__genSQL__(),
                        self.alias,
                    ),
                    self.parent._VERTICAPY_VARIABLES_["cursor"],
                    "apply_test_feature",
                )
            category = category_from_type(ctype=ctype)
            all_cols, max_floor = self.parent.get_columns(), 0
            for column in all_cols:
                try:
                    if (str_column(column) in func) or (
                        re.search(
                            re.compile("\\b{}\\b".format(column.replace('"', ""))), func
                        )
                    ):
                        max_floor = max(len(self.parent[column].transformations), max_floor)
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
			abs     : absolute value
			acos    : trigonometric inverse cosine
			asin    : trigonometric inverse sine
			atan    : trigonometric inverse tangent
			cbrt    : cube root
			ceil    : value up to the next whole number
			cos     : trigonometric cosine
			cosh    : hyperbolic cosine
			cot     : trigonometric cotangent
			exp     : exponential function
			floor   : value down to the next whole number
			ln      : natural logarithm
			log     : logarithm
			log10   : base 10 logarithm
			mod     : remainder of a division operation
			pow     : number raised to the power of another number
			round   : rounds a value to a specified number of decimal places
			sign    : arithmetic sign
			sin     : trigonometric sine
			sinh    : hyperbolic sine
			sqrt    : arithmetic square root
			tan     : trigonometric tangent
			tanh    : hyperbolic tangent
	x: int/float, optional
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
                        "cbrt",
                        "ceil",
                        "cos",
                        "cosh",
                        "cot",
                        "exp",
                        "floor",
                        "ln",
                        "log",
                        "log10",
                        "mod",
                        "pow",
                        "round",
                        "sign",
                        "sin",
                        "sinh",
                        "sqrt",
                        "tan",
                        "tanh",
                    ],
                ),
                ("x", x, [int, float],),
            ]
        )
        if func not in ("log", "mod", "pow", "round"):
            expr = "{}({})".format(func.upper(), "{}")
        else:
            expr = "{}({}, {})".format(func.upper(), "{}", x)
        return self.apply(func=expr)

    # ---#
    def astype(self, dtype: str):
        """
	---------------------------------------------------------------------------
	Converts the vColumn to the input type.

	Parameters
 	----------
 	dtype: str
 		New type.

 	Returns
 	-------
 	vDataFrame
		self.parent

	See Also
	--------
	vDataFrame.astype : Converts the vColumns to the input type.
		"""
        check_types([("dtype", dtype, [str],)])
        try:
            query = "SELECT {}::{} AS {} FROM {} WHERE {} IS NOT NULL LIMIT 20".format(
                self.alias, dtype, self.alias, self.parent.__genSQL__(), self.alias
            )
            self.parent._VERTICAPY_VARIABLES_["cursor"].execute(query)
            self.transformations += [
                ("{}::{}".format("{}", dtype), dtype, category_from_type(ctype=dtype))
            ]
            self.parent.__add_to_history__(
                "[AsType]: The vColumn {} was converted to {}.".format(
                    self.alias, dtype
                )
            )
            return self.parent
        except Exception as e:
            raise ConversionError(
                "{}\nThe vColumn {} can not be converted to {}".format(
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
        return self.aggregate(["avg"]).values[self.alias][0]

    mean = avg
    # ---#
    def bar(
        self,
        method: str = "density",
        of: str = "",
        max_cardinality: int = 6,
        bins: int = 0,
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
 	bins: int, optional
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
 	vDataFrame[].hist : Draws the histogram of the vColumn based on an aggregation.
		"""
        check_types(
            [
                ("method", method, [str],),
                ("of", of, [str],),
                ("max_cardinality", max_cardinality, [int, float],),
                ("bins", bins, [int, float],),
                ("h", h, [int, float],),
            ]
        )
        if of:
            columns_check([of], self.parent)
            of = vdf_columns_names([of], self.parent)[0]
        from verticapy.plot import bar

        return bar(self, method, of, max_cardinality, bins, h, ax=ax, **style_kwds,)

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
        if isinstance(cat_priority, str) or not (isinstance(cat_priority, Iterable)):
            cat_priority = [cat_priority]
        check_types(
            [
                ("by", by, [str],),
                ("max_cardinality", max_cardinality, [int, float],),
                ("h", h, [int, float],),
                ("cat_priority", cat_priority, [list],),
            ]
        )
        if by:
            columns_check([by], self.parent)
            by = vdf_columns_names([by], self.parent)[0]
        from verticapy.plot import boxplot

        return boxplot(self, by, h, max_cardinality, cat_priority, ax=ax, **style_kwds,)

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
        check_types(
            [("lower", lower, [float, int,],), ("upper", upper, [float, int,],),]
        )
        assert (lower != None) or (upper != None), ParameterError("At least 'lower' or 'upper' must have a numerical value")
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
        return self.aggregate(["count"]).values[self.alias][0]

    # ---#
    def cut(self, 
            breaks: list, 
            labels: list = [], 
            include_lowest: bool = True,
            right: bool = True):
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
        check_types([("breaks", breaks, [list],),
                     ("labels", labels, [list],),
                     ("include_lowest", include_lowest, [bool],),
                     ("right", right, [bool],),])
        assert self.isnum() or self.isdate(), TypeError("cut only works on numerical / date-like vColumns.")
        assert len(breaks) >= 2, ParameterError("Length of parameter 'breaks' must be greater or equal to 2.")
        assert len(breaks) == len(labels) + 1 or not(labels), ParameterError("Length of parameter breaks must be equal to the length of parameter 'labels' + 1 or parameter 'labels' must be empty.")
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
            conditions += [f"'{first_elem}' {op1} {column} AND {column} {op2} '{second_elem}' THEN '{label}'"]
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
        return self.apply(func="DATE_PART('{}', {})".format(field, "{}"))

    # ---#
    def decode(self, *argv):
        """
	---------------------------------------------------------------------------
	Encodes the vColumn using a user-defined encoding.

	Parameters
 	----------
 	argv: object
        Infinite Number of Expressions.
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
        check_types(
            [
                ("by", by, [str],),
                ("kernel", kernel, ["gaussian", "logistic", "sigmoid", "silverman"],),
                ("bandwidth", bandwidth, [int, float],),
                ("nbins", nbins, [float, int],),
            ]
        )
        if by:
            columns_check([by], self.parent)
            by = vdf_columns_names([by], self.parent)[0]
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
        schema = self.parent._VERTICAPY_VARIABLES_["schema_writing"]
        if not (schema):
            schema = "public"
        name = "{}.VERTICAPY_TEMP_MODEL_KDE_{}".format(
            schema, get_session(self.parent._VERTICAPY_VARIABLES_["cursor"])
        )
        if isinstance(xlim, (tuple, list)):
            xlim_tmp = [xlim]
        else:
            xlim_tmp = []
        model = KernelDensity(
            name,
            cursor=self.parent._VERTICAPY_VARIABLES_["cursor"],
            bandwidth=bandwidth,
            kernel=kernel,
            nbins=nbins,
            xlim=xlim_tmp,
            store=False,
        )
        try:
            result = model.fit(self.parent.__genSQL__(), [self.alias]).plot(
                ax=ax, **style_kwds,
            )
            model.drop()
            return result
        except:
            model.drop()
            raise

    # ---#
    def describe(
        self, method: str = "auto", max_cardinality: int = 6, numcol: str = "",
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
        check_types(
            [
                ("method", method, ["auto", "numerical", "categorical", "cat_stats"],),
                ("max_cardinality", max_cardinality, [int, float],),
                ("numcol", numcol, [str],),
            ]
        )

        method = method.lower()
        assert (method != "cat_stats") or (numcol), ParameterError("The parameter 'numcol' must be a vDataFrame column if the method is 'cat_stats'")
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
            numcol = vdf_columns_names([numcol], self.parent)[0]
            assert self.parent[numcol].category() in ("float", "int"), TypeError("The column 'numcol' must be numerical")
            cast = "::int" if (self.parent[numcol].isbool()) else ""
            query, cat = [], self.distinct()
            if len(cat) == 1:
                lp, rp = "(", ")"
            else:
                lp, rp = "", ""
            for category in cat:
                tmp_query = "SELECT '{}' AS 'index', COUNT({}) AS count, 100 * COUNT({}) / {} AS percent, AVG({}{}) AS mean, STDDEV({}{}) AS std, MIN({}{}) AS min, APPROXIMATE_PERCENTILE ({}{} USING PARAMETERS percentile = 0.1) AS '10%', APPROXIMATE_PERCENTILE ({}{} USING PARAMETERS percentile = 0.25) AS '25%', APPROXIMATE_PERCENTILE ({}{} USING PARAMETERS percentile = 0.5) AS '50%', APPROXIMATE_PERCENTILE ({}{} USING PARAMETERS percentile = 0.75) AS '75%', APPROXIMATE_PERCENTILE ({}{} USING PARAMETERS percentile = 0.9) AS '90%', MAX({}{}) AS max FROM vdf_table"
                tmp_query = tmp_query.format(
                    category,
                    self.alias,
                    self.alias,
                    self.parent.shape()[0],
                    numcol,
                    cast,
                    numcol,
                    cast,
                    numcol,
                    cast,
                    numcol,
                    cast,
                    numcol,
                    cast,
                    numcol,
                    cast,
                    numcol,
                    cast,
                    numcol,
                    cast,
                    numcol,
                    cast,
                )
                tmp_query += (
                    " WHERE {} IS NULL".format(self.alias)
                    if (category in ("None", None))
                    else " WHERE {} = '{}'".format(
                        convert_special_type(self.category(), False, self.alias),
                        category,
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
                query, self.parent._VERTICAPY_VARIABLES_["cursor"], title=title,
            ).values
        elif (
            ((distinct_count < max_cardinality + 1) and (method != "numerical"))
            or not (is_numeric)
            or (method == "categorical")
        ):
            query = "(SELECT {} || '', COUNT(*) FROM vdf_table GROUP BY {} ORDER BY COUNT(*) DESC LIMIT {})".format(
                self.alias, self.alias, max_cardinality
            )
            if distinct_count > max_cardinality:
                query += (
                    "UNION ALL (SELECT 'Others', SUM(count) FROM (SELECT COUNT(*) AS count FROM vdf_table WHERE {} IS NOT NULL GROUP BY {} ORDER BY COUNT(*) DESC OFFSET {}) VERTICAPY_SUBTABLE) ORDER BY count DESC"
                ).format(self.alias, self.alias, max_cardinality + 1)
            query = "WITH vdf_table AS (SELECT * FROM {}) {}".format(
                self.parent.__genSQL__(), query
            )
            self.parent.__executeSQL__(
                query=query,
                title="Computes the descriptive statistics of {}.".format(self.alias),
            )
            query_result = self.parent._VERTICAPY_VARIABLES_["cursor"].fetchall()
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
                "25%",
                "50%",
                "75%",
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
        bins: int = -1,
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
 	bins: int, optional
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
        check_types(
            [
                ("RFmodel_params", RFmodel_params, [dict],),
                ("return_enum_trans", return_enum_trans, [bool],),
                ("h", h, [int, float],),
                ("response", response, [str],),
                ("bins", bins, [int, float],),
                (
                    "method",
                    method,
                    ["auto", "smart", "same_width", "same_freq", "topk"],
                ),
                ("return_enum_trans", return_enum_trans, [bool],),
            ]
        )
        method = method.lower()
        if self.isnum() and method == "smart":
            schema = self.parent._VERTICAPY_VARIABLES_["schema_writing"]
            if not (schema):
                schema = "public"
            temp_information = (
                "{}.VERTICAPY_TEMP_VIEW_{}".format(
                    schema, get_session(self.parent._VERTICAPY_VARIABLES_["cursor"])
                ),
                "{}.VERTICAPY_TEMP_MODEL_{}".format(
                    schema, get_session(self.parent._VERTICAPY_VARIABLES_["cursor"])
                ),
            )
            assert bins >= 2, ParameterError(
                "Parameter 'bins' must be greater or equals to 2 in case of discretization using the method 'smart'."
            )
            assert response, ParameterError(
                "Parameter 'response' can not be empty in case of discretization using the method 'smart'."
            )
            columns_check([response], self.parent)
            response = vdf_columns_names([response], self.parent)[0]

            def drop_temp_elem(self, temp_information):
                with warnings.catch_warnings(record=True) as w:
                    try:
                        drop(
                            temp_information[1],
                            cursor=self.parent._VERTICAPY_VARIABLES_["cursor"],
                            method="model",
                        )
                    except:
                        pass
                    try:
                        drop(
                            temp_information[0],
                            cursor=self.parent._VERTICAPY_VARIABLES_["cursor"],
                            method="view",
                        )
                    except:
                        pass

            drop_temp_elem(self, temp_information)
            self.parent.to_db(temp_information[0])
            from verticapy.learn.ensemble import (
                RandomForestClassifier,
                RandomForestRegressor,
            )

            if self.parent[response].category() == "float":
                model = RandomForestRegressor(
                    temp_information[1], self.parent._VERTICAPY_VARIABLES_["cursor"],
                )
            else:
                model = RandomForestClassifier(
                    temp_information[1], self.parent._VERTICAPY_VARIABLES_["cursor"],
                )
            model.set_params({"n_estimators": 20, "max_depth": 8, "nbins": 100})
            model.set_params(RFmodel_params)
            parameters = model.get_params()
            try:
                model.fit(temp_information[0], [self.alias], response)
                query = [
                    "(SELECT READ_TREE(USING PARAMETERS model_name = '{}', tree_id = {}, format = 'tabular'))".format(
                        temp_information[1], i
                    )
                    for i in range(parameters["n_estimators"])
                ]
                query = "SELECT split_value FROM (SELECT split_value, MAX(weighted_information_gain) FROM ({}) VERTICAPY_SUBTABLE WHERE split_value IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT {}) VERTICAPY_SUBTABLE ORDER BY split_value::float".format(
                    " UNION ALL ".join(query), bins - 1
                )
                self.parent.__executeSQL__(
                    query=query,
                    title="Computes the optimized histogram bins using Random Forest.",
                )
                result = self.parent._VERTICAPY_VARIABLES_["cursor"].fetchall()
                result = [elem[0] for elem in result]
            except:
                drop_temp_elem(self, temp_information)
                raise
            drop_temp_elem(self, temp_information)
            result = [self.min()] + result + [self.max()]
        elif method == "topk":
            assert k >= 2, ParameterError("Parameter 'k' must be greater or equals to 2 in case of discretization using the method 'topk'")
            distinct = self.topk(k).values["index"]
            trans = (
                "(CASE WHEN {} IN ({}) THEN {} || '' ELSE '{}' END)".format(
                    convert_special_type(self.category(), False),
                    ", ".join(
                        [
                            "'{}'".format(str(elem).replace("'", "''"))
                            for elem in distinct
                        ]
                    ),
                    convert_special_type(self.category(), False),
                    new_category.replace("'", "''"),
                ),
                "varchar",
                "text",
            )
        elif self.isnum() and method == "same_freq":
            assert bins >= 2, ParameterError("Parameter 'bins' must be greater or equals to 2 in case of discretization using the method 'same_freq'")
            count = self.count()
            nb = int(float(count / int(bins - 1)))
            assert nb != 0, Exception("Not enough values to compute the Equal Frequency discretization")
            total, query, nth_elems = nb, [], []
            while total < count - 1:
                nth_elems += [str(total)]
                total += nb
            where = "WHERE _verticapy_row_nb_ IN ({})".format(
                ", ".join(["1"] + nth_elems + [str(count)])
            )
            query = "SELECT {} FROM (SELECT {}, ROW_NUMBER() OVER (ORDER BY {}) AS _verticapy_row_nb_ FROM {} WHERE {} IS NOT NULL) VERTICAPY_SUBTABLE {}".format(
                self.alias,
                self.alias,
                self.alias,
                self.parent.__genSQL__(),
                self.alias,
                where,
            )
            self.parent.__executeSQL__(
                query=query, title="Computes the equal frequency histogram bins."
            )
            result = self.parent._VERTICAPY_VARIABLES_["cursor"].fetchall()
            result = [elem[0] for elem in result]
        elif self.isnum() and method in ("same_width", "auto"):
            if not (h) or h <= 0:
                if bins <= 0:
                    h = self.numh()
                else:
                    h = (self.max() - self.min()) * 1.01 / bins
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
            query = "SELECT {} AS {} FROM {} WHERE {} IS NOT NULL GROUP BY {} ORDER BY {}".format(
                convert_special_type(self.category(), True, self.alias),
                self.alias,
                self.parent.__genSQL__(),
                self.alias,
                self.alias,
                self.alias,
            )
        else:
            query = "SELECT {} FROM (SELECT {} AS {}, {} AS verticapy_agg FROM {} WHERE {} IS NOT NULL GROUP BY 1) x ORDER BY verticapy_agg DESC".format(
                self.alias,
                convert_special_type(self.category(), True, self.alias),
                self.alias,
                kwargs["agg"],
                self.parent.__genSQL__(),
                self.alias,
            )
        self.parent.__executeSQL__(
            query=query,
            title="Computes the distinct categories of {}.".format(self.alias),
        )
        query_result = self.parent._VERTICAPY_VARIABLES_["cursor"].fetchall()
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
        check_types([("x", x, [int, float],)])
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
        check_types([("add_history", add_history, [bool],)])
        try:
            parent = self.parent
            force_columns = [
                column for column in self.parent._VERTICAPY_VARIABLES_["columns"]
            ]
            force_columns.remove(self.alias)
            self.parent._VERTICAPY_VARIABLES_["cursor"].execute(
                "SELECT * FROM {} LIMIT 10".format(
                    self.parent.__genSQL__(force_columns=force_columns)
                )
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
        self, threshold: float = 4.0, use_threshold: bool = True, alpha: float = 0.05,
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
        check_types(
            [
                ("alpha", alpha, [int, float],),
                ("use_threshold", use_threshold, [bool],),
                ("threshold", threshold, [int, float],),
            ]
        )
        if use_threshold:
            result = self.aggregate(func=["std", "avg"]).transpose().values
            self.parent.filter(
                "ABS({} - {}) / {} < {}".format(
                    self.alias, result["avg"][0], result["std"][0], threshold
                ),
            )
        else:
            p_alpha, p_1_alpha = (
                self.parent.quantile([alpha, 1 - alpha], [self.alias])
                .transpose()
                .values[self.alias]
            )
            self.parent.filter(
                "({} BETWEEN {} AND {})".format(self.alias, p_alpha, p_1_alpha),
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
        if isinstance(method, str):
            method = method.lower()
        check_types(
            [
                ("method", method, ["winsorize", "null", "mean"],),
                ("alpha", alpha, [int, float],),
                ("use_threshold", use_threshold, [bool],),
                ("threshold", threshold, [int, float],),
            ]
        )
        if use_threshold:
            result = self.aggregate(func=["std", "avg"]).transpose().values
            p_alpha, p_1_alpha = (
                -threshold * result["std"][0] + result["avg"][0],
                threshold * result["std"][0] + result["avg"][0],
            )
        else:
            query = "SELECT PERCENTILE_CONT({}) WITHIN GROUP (ORDER BY {}) OVER (), PERCENTILE_CONT(1 - {}) WITHIN GROUP (ORDER BY {}) OVER () FROM {} LIMIT 1".format(
                alpha, self.alias, alpha, self.alias, self.parent.__genSQL__()
            )
            self.parent.__executeSQL__(
                query=query,
                title="Computes the quantiles of {}.".format(self.alias),
            )
            p_alpha, p_1_alpha = self.parent._VERTICAPY_VARIABLES_[
                "cursor"
            ].fetchone()
        if method == "winsorize":
            self.clip(lower=p_alpha, upper=p_1_alpha)
        elif method == "null":
            self.apply(
                func="(CASE WHEN ({} BETWEEN {} AND {}) THEN {} ELSE NULL END)".format(
                    "{}", p_alpha, p_1_alpha, "{}"
                )
            )
        elif method == "mean":
            query = "WITH vdf_table AS (SELECT * FROM {}) (SELECT AVG({}) FROM vdf_table WHERE {} < {}) UNION ALL (SELECT AVG({}) FROM vdf_table WHERE {} > {})".format(
                self.parent.__genSQL__(),
                self.alias,
                self.alias,
                p_alpha,
                self.alias,
                self.alias,
                p_1_alpha,
            )
            self.parent.__executeSQL__(
                query=query,
                title="Computes the average of the {}'s lower and upper outliers.".format(
                    self.alias
                ),
            )
            mean_alpha, mean_1_alpha = [
                item[0]
                for item in self.parent._VERTICAPY_VARIABLES_["cursor"].fetchall()
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
                ("expr", expr, [str],),
                ("by", by, [list],),
                ("order_by", order_by, [list],),
            ]
        )
        method = method.lower()
        columns_check([elem for elem in order_by] + by, self.parent)
        by = vdf_columns_names(by, self.parent)
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
                    query = "SELECT {}, {}({}) FROM {} GROUP BY {};".format(
                        by[0], fun, self.alias, self.parent.__genSQL__(), by[0]
                    )
                    self.parent.__executeSQL__(
                        query, title="Computes the different aggregations."
                    )
                    result = self.parent._VERTICAPY_VARIABLES_["cursor"].fetchall()
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
                    self.parent._VERTICAPY_VARIABLES_["cursor"].execute(
                        "SELECT {} FROM {} LIMIT 1".format(
                            new_column.format(self.alias), self.parent.__genSQL__()
                        )
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
            assert order_by, ParameterError("If the method is in ffill|pad|bfill|backfill then 'order_by' must be a list of at least one element to use to order the data")
            desc = " DESC" if (method in ("ffill", "pad")) else ""
            partition_by = (
                "PARTITION BY {}".format(
                    ", ".join([str_column(column) for column in by])
                )
                if (by)
                else ""
            )
            order_by_ts = ", ".join([str_column(column) + desc for column in order_by])
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
    def geo_plot(
        self, *args, **kwargs,
    ):
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
        columns = [self.alias]
        check = True
        if len(args) > 0:
            column = args[0]
        elif "column" in kwargs:
            column = kwargs["column"]
        else:
            check = False
        if check:
            columns_check([column], self.parent)
            column = vdf_columns_names([column], self.parent)[0]
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
        return self.parent[columns].to_geopandas(self.alias).plot(*args, **kwargs,)

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
        check_types(
            [
                ("prefix", prefix, [str],),
                ("prefix_sep", prefix_sep, [str],),
                ("drop_first", drop_first, [bool],),
                ("use_numbers_as_suffix", use_numbers_as_suffix, [bool],),
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
            columns = self.parent.get_columns()
            for k in range(len(distinct_elements) - n):
                name = (
                    '"{}{}"'.format(prefix, k)
                    if (use_numbers_as_suffix)
                    else '"{}{}"'.format(
                        prefix, str(distinct_elements[k]).replace('"', "_")
                    )
                )
                assert not(column_check_ambiguous(name, columns)), NameError(f"A vColumn has already the alias of one of the dummies ({name}).\nIt can be the result of using previously the method on the vColumn or simply because of ambiguous columns naming.\nBy changing one of the parameters ('prefix', 'prefix_sep'), you'll be able to solve this issue.")
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
        bins: int = 0,
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
 	bins: int, optional
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
        check_types(
            [
                ("method", method, [str],),
                ("of", of, [str],),
                ("max_cardinality", max_cardinality, [int, float],),
                ("h", h, [int, float],),
                ("bins", bins, [int, float],),
            ]
        )
        if of:
            columns_check([of], self.parent)
            of = vdf_columns_names([of], self.parent)[0]
        from verticapy.plot import hist

        return hist(self, method, of, max_cardinality, bins, h, ax=ax, **style_kwds,)

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
        check_types(
            [("limit", limit, [int, float],), ("offset", offset, [int, float],),]
        )
        if offset < 0:
            offset = max(0, self.parent.shape()[0] - limit)
        title = "Reads {}.".format(self.alias)
        tail = to_tablesample(
            "SELECT {} AS {} FROM {}{} LIMIT {} OFFSET {}".format(
                convert_special_type(self.category(), False, self.alias),
                self.alias,
                self.parent.__genSQL__(),
                last_order_by(self.parent),
                limit,
                offset,
            ),
            self.parent._VERTICAPY_VARIABLES_["cursor"],
            title=title,
        )
        tail.count = self.parent.shape()[0]
        tail.offset = offset
        tail.dtype[self.alias] = self.ctype()
        tail.name = self.alias
        return tail

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
        return self.ctype().lower() in ("bool", "boolean")

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
    def isin(
        self, val: list, *args,
    ):
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
        if isinstance(val, str) or not (isinstance(val, Iterable)):
            val = [val]
        val += list(args)
        check_types([("val", val, [list],)])
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
    def iv_woe(
        self, y: str, bins: int = 10,
    ):
        """
    ---------------------------------------------------------------------------
    Computes the Information Value (IV) / Weight Of Evidence (WOE) Table. It tells 
    the predictive power of an independent variable in relation to the dependent 
    variable.

    Parameters
    ----------
    y: str
        Response vColumn.
    bins: int, optional
        Maximum number of bins used for the discretization (must be > 1)

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.iv_woe : Computes the Information Value (IV) Table.
        """
        check_types([("y", y, [str],), ("bins", bins, [int],)])
        columns_check([y], self.parent)
        y = vdf_columns_names([y], self.parent)[0]
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
            bins=bins,
            k=bins,
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
        query = "SELECT {} AS index, non_events, events, pt_non_events, pt_events, CASE WHEN non_events = 0 OR events = 0 THEN 0 ELSE ZEROIFNULL(LOG(pt_non_events / NULLIFZERO(pt_events))) END AS woe, CASE WHEN non_events = 0 OR events = 0 THEN 0 ELSE (pt_non_events - pt_events) * ZEROIFNULL(LOG(pt_non_events / NULLIFZERO(pt_events))) END AS iv FROM ({}) x ORDER BY ord".format(
            self.alias, query,
        )
        title = "Computing WOE & IV of {} (response = {}).".format(self.alias, y)
        result = to_tablesample(
            query, self.parent._VERTICAPY_VARIABLES_["cursor"], title=title,
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
        return self.aggregate(["max"]).values[self.alias][0]

    # ---#
    def mean_encode(self, response_column: str):
        """
	---------------------------------------------------------------------------
	Encodes the vColumn using the average of the response partitioned by the 
	different vColumn categories.

	Parameters
 	----------
 	response_column: str
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
        check_types([("response_column", response_column, [str],)])
        columns_check([response_column], self.parent)
        response_column = vdf_columns_names([response_column], self.parent)[0]
        assert self.parent[response_column].isnum(), TypeError("The response column must be numerical to use a mean encoding")
        max_floor = len(self.parent[response_column].transformations) - len(
            self.transformations
        )
        for k in range(max_floor):
            self.transformations += [("{}", self.ctype(), self.category())]
        self.transformations += [
            (
                "AVG({}) OVER (PARTITION BY {})".format(response_column, "{}"),
                "int",
                "float",
            )
        ]
        self.parent.__update_catalog__(erase=True, columns=[self.alias])
        self.parent.__add_to_history__(
            "[Mean Encode]: The vColumn {} was transformed using a mean encoding with {} as Response Column.".format(
                self.alias, response_column
            )
        )
        if verticapy.options["print_info"]:
            print("The mean encoding was successfully done.")
        return self.parent

    # ---#
    def median(self):
        """
	---------------------------------------------------------------------------
	Aggregates the vColumn using 'median'.

 	Returns
 	-------
 	float/str
 		median

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
        return self.quantile(0.5)

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
        check_types([("dropna", dropna, [bool],), ("n", n, [int, float],)])
        if n == 1:
            pre_comp = self.parent.__get_catalog_value__(self.alias, "top")
            if pre_comp != "VERTICAPY_NOT_PRECOMPUTED":
                if not (dropna) and (pre_comp != None):
                    return pre_comp
        assert n >= 1, ParameterError("Parameter 'n' must be greater or equal to 1")
        where = " WHERE {} IS NOT NULL ".format(self.alias) if (dropna) else " "
        self.parent.__executeSQL__(
            "SELECT {} FROM (SELECT {}, COUNT(*) AS _verticapy_cnt_ FROM {}{}GROUP BY {} ORDER BY _verticapy_cnt_ DESC LIMIT {}) VERTICAPY_SUBTABLE ORDER BY _verticapy_cnt_ ASC LIMIT 1".format(
                self.alias, self.alias, self.parent.__genSQL__(), where, self.alias, n
            )
        )
        try:
            top = self.parent._VERTICAPY_VARIABLES_["cursor"].fetchone()[0]
        except:
            top = None
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
        check_types([("x", x, [int, float],)])
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
        check_types([("n", n, [int, float],)])
        query = "SELECT * FROM {} WHERE {} IS NOT NULL ORDER BY {} DESC LIMIT {}".format(
            self.parent.__genSQL__(), self.alias, self.alias, n
        )
        title = "Reads {} {} largest elements.".format(self.alias, n)
        return to_tablesample(
            query, self.parent._VERTICAPY_VARIABLES_["cursor"], title=title,
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
        if isinstance(by, str):
            by = [by]
        check_types(
            [
                ("method", method, ["zscore", "robust_zscore", "minmax"],),
                ("by", by, [list],),
                ("return_trans", return_trans, [bool],),
            ]
        )
        method = method.lower()
        columns_check(by, self.parent)
        by = vdf_columns_names(by, self.parent)
        nullifzero, n = 1, len(by)
        if self.isbool():
            warning_message = "Normalize doesn't work on booleans".format(self.alias)
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
                        self.parent.__executeSQL__(
                            "SELECT {}, AVG({}), STDDEV({}) FROM {} GROUP BY {}".format(
                                by[0],
                                self.alias,
                                self.alias,
                                self.parent.__genSQL__(),
                                by[0],
                            ),
                            title="Computes the different categories to normalize.",
                        )
                        result = self.parent._VERTICAPY_VARIABLES_["cursor"].fetchall()
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
                        self.parent._VERTICAPY_VARIABLES_["cursor"].execute(
                            "SELECT {}, {} FROM {} LIMIT 1".format(
                                avg, stddev, self.parent.__genSQL__()
                            )
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
                mad, med = self.aggregate(["mad", "median"]).values[self.alias]
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
                        self.parent.__executeSQL__(
                            "SELECT {}, MIN({}), MAX({}) FROM {} GROUP BY {}".format(
                                by[0],
                                self.alias,
                                self.alias,
                                self.parent.__genSQL__(),
                                by[0],
                            ),
                            title="Computes the different categories {} to normalize.".format(
                                by[0]
                            ),
                        )
                        result = self.parent._VERTICAPY_VARIABLES_["cursor"].fetchall()
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
                        self.parent._VERTICAPY_VARIABLES_["cursor"].execute(
                            "SELECT {}, {} FROM {} LIMIT 1".format(
                                cmax, cmin, self.parent.__genSQL__()
                            )
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
                    return "({} - {}) / {}({} - {})".format(
                        self.alias,
                        cmin,
                        "NULLIFZERO" if (nullifzero) else "",
                        cmax,
                        cmin,
                    )
                else:
                    final_transformation = [
                        (
                            "({} - {}) / {}({} - {})".format(
                                "{}",
                                cmin,
                                "NULLIFZERO" if (nullifzero) else "",
                                cmax,
                                cmin,
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
                            self.catalog[elem] = (sauv[elem] - sauv["50%"]) / (
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
        check_types([("n", n, [int, float],)])
        query = "SELECT * FROM {} WHERE {} IS NOT NULL ORDER BY {} ASC LIMIT {}".format(
            self.parent.__genSQL__(), self.alias, self.alias, n
        )
        title = "Reads {} {} smallest elements.".format(n, self.alias)
        return to_tablesample(
            query, self.parent._VERTICAPY_VARIABLES_["cursor"], title=title,
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
        check_types(
            [("method", method, ["sturges", "freedman_diaconis", "fd", "auto"],)]
        )
        method = method.lower()
        if method == "auto":
            pre_comp = self.parent.__get_catalog_value__(self.alias, "numh")
            if pre_comp != "VERTICAPY_NOT_PRECOMPUTED":
                return pre_comp
        assert self.isnum() or self.isdate(), ParameterError("numh is only available on type numeric|date")
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
            table = "(SELECT DATEDIFF('second', '{}'::timestamp, {}) AS {} FROM {}) VERTICAPY_OPTIMAL_H_TABLE".format(
                min_date, self.alias, self.alias, self.parent.__genSQL__()
            )
            query = "SELECT COUNT({}) AS NAs, MIN({}) AS min, APPROXIMATE_PERCENTILE({} USING PARAMETERS percentile = 0.25) AS Q1, APPROXIMATE_PERCENTILE({} USING PARAMETERS percentile = 0.75) AS Q3, MAX({}) AS max FROM {}".format(
                self.alias, self.alias, self.alias, self.alias, self.alias, table
            )
            self.parent.__executeSQL__(
                query, title="Different aggregations to compute the optimal h."
            )
            result = self.parent._VERTICAPY_VARIABLES_["cursor"].fetchone()
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
    def nunique(self, approx: bool = False):
        """
	---------------------------------------------------------------------------
	Aggregates the vColumn using 'unique' (cardinality).

	Parameters
 	----------
 	approx: bool, optional
 		If set to True, the method will compute the approximate count distinct 
 		instead of the exact one.

 	Returns
 	-------
 	int
 		vColumn cardinality (or approximate cardinality).

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
        check_types([("approx", approx, [bool],)])
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
        if isinstance(pie_type, str):
            pie_type = pie_type.lower()
        check_types(
            [
                ("method", method, [str],),
                ("of", of, [str],),
                ("max_cardinality", max_cardinality, [int, float],),
                ("h", h, [int, float],),
                ("pie_type", pie_type, ["auto", "donut", "rose"]),
            ]
        )
        donut = True if pie_type == "donut" else False
        rose = True if pie_type == "rose" else False
        if of:
            columns_check([of], self.parent)
            of = vdf_columns_names([of], self.parent)[0]
        from verticapy.plot import pie

        return pie(
            self, method, of, max_cardinality, h, donut, rose, ax=None, **style_kwds,
        )

    # ---#
    def plot(
        self,
        ts: str,
        by: str = "",
        start_date: (str, datetime.datetime, datetime.date,) = "",
        end_date: (str, datetime.datetime, datetime.date,) = "",
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
        check_types(
            [
                ("ts", ts, [str],),
                ("by", by, [str],),
                ("start_date", start_date, [str, datetime.datetime, datetime.date,],),
                ("end_date", end_date, [str, datetime.datetime, datetime.date,],),
                ("area", area, [bool],),
                ("step", step, [bool],),
            ]
        )
        columns_check([ts], self.parent)
        ts = vdf_columns_names([ts], self.parent)[0]
        if by:
            columns_check([by], self.parent)
            by = vdf_columns_names([by], self.parent)[0]
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
        return self.aggregate(func=["prod"]).values[self.alias][0]

    prod = product

    # ---#
    def quantile(self, x: float):
        """
	---------------------------------------------------------------------------
	Aggregates the vColumn using an input 'quantile'.

	Parameters
 	----------
 	x: float
 		A float between 0 and 1 that represents the quantile.
        For example: 0.25 represents Q1.

 	Returns
 	-------
 	float
 		quantile

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
        check_types([("x", x, [int, float],)])
        return self.aggregate(func=["{}%".format(x * 100)]).values[self.alias][0]

    # ---#
    def range_plot(
        self,
        ts: str,
        q: tuple = (0.25, 0.75),
        start_date: (str, datetime.datetime, datetime.date,) = "",
        end_date: (str, datetime.datetime, datetime.date,) = "",
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
        check_types(
            [
                ("ts", ts, [str],),
                ("q", q, [tuple],),
                (
                    "start_date",
                    start_date,
                    [str, datetime.datetime, datetime.date, int, float,],
                ),
                (
                    "end_date",
                    end_date,
                    [str, datetime.datetime, datetime.date, int, float,],
                ),
                ("plot_median", plot_median, [bool],),
            ]
        )
        columns_check([ts], self.parent)
        ts = vdf_columns_names([ts], self.parent)[0]
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
        check_types([("new_name", new_name, [str],)])
        old_name = str_column(self.alias)
        new_name = new_name.replace('"', "")
        assert not(column_check_ambiguous(new_name, self.parent.get_columns())), NameError(f"A vColumn has already the alias {new_name}.\nBy changing the parameter 'new_name', you'll be able to solve this issue.")
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
        check_types([("n", n, [int, float],)])
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
        check_types(
            [
                ("length", length, [int, float],),
                ("unit", unit, [str],),
                ("start", start, [bool],),
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
        max_cardinality: (int, tuple) = (6, 6),
        h: (int, float, tuple) = (None, None),
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
        check_types(
            [
                ("by", by, [str],),
                ("method", method, [str],),
                ("of", of, [str],),
                ("max_cardinality", max_cardinality, [list],),
                ("h", h, [list, float, int],),
            ]
        )
        if by:
            columns_check([by], self.parent)
            by = vdf_columns_names([by], self.parent)[0]
            columns = [self.alias, by]
        else:
            columns = [self.alias]
        if of:
            columns_check([of], self)
            of = vdf_columns_names([of], self.parent)[0]
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
        pre_comp = self.parent.__get_catalog_value__(self.alias, "store_usage")
        if pre_comp != "VERTICAPY_NOT_PRECOMPUTED":
            return pre_comp
        self.parent.__executeSQL__(
            "SELECT ZEROIFNULL(SUM(LENGTH({}::varchar))) FROM {}".format(
                convert_special_type(self.category(), False, self.alias),
                self.parent.__genSQL__(),
            ),
            title="Computes the Store Usage of the vColumn {}.".format(self.alias),
        )
        store_usage = self.parent._VERTICAPY_VARIABLES_["cursor"].fetchone()[0]
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
        check_types([("pat", pat, [str],)])
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
        check_types([("pat", pat, [str],)])
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
        check_types([("pat", pat, [str],)])
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
        check_types([("to_replace", to_replace, [str],), ("value", value, [str],)])
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
        check_types([("start", start, [int, float],), ("step", step, [int, float],)])
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
        check_types([("x", x, [int, float],)])
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
        check_types([("k", k, [int, float],), ("dropna", dropna, [bool],)])
        try:
            version(
                cursor=self.parent._VERTICAPY_VARIABLES_["cursor"], condition=[9, 0, 1]
            )
            topk = "" if (k < 1) else "TOPK = {},".format(k)
            query = "SELECT SUMMARIZE_CATCOL({}::varchar USING PARAMETERS {} WITH_TOTALCOUNT = False) OVER () FROM {}".format(
                self.alias, topk, self.parent.__genSQL__()
            )
            if dropna:
                query += " WHERE {} IS NOT NULL".format(self.alias)
            self.parent.__executeSQL__(
                query,
                title="Computes the top{} categories of {}.".format(
                    k if k > 0 else "", self.alias
                ),
            )
        except:
            topk = "" if (k < 1) else "LIMIT {}".format(k)
            query = "SELECT {} AS {}, COUNT(*) AS _verticapy_cnt_, 100 * COUNT(*) / {} AS percent FROM {} GROUP BY {} ORDER BY _verticapy_cnt_ DESC {}".format(
                convert_special_type(self.category(), True, self.alias),
                self.alias,
                self.parent.shape()[0],
                self.parent.__genSQL__(),
                self.alias,
                topk,
            )
            self.parent.__executeSQL__(
                query,
                title="Computes the top{} categories of {}.".format(
                    k if k > 0 else "", self.alias
                ),
            )
        result = self.parent._VERTICAPY_VARIABLES_["cursor"].fetchall()
        values = {
            "index": [item[0] for item in result],
            "count": [item[1] for item in result],
            "percent": [round(item[2], 3) for item in result],
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
        return self.aggregate(["variance"]).values[self.alias][0]

    variance = var
