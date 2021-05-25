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
import random

# VerticaPy Modules
from verticapy.utilities import *
from verticapy.toolbox import *
from verticapy import vDataFrame
from verticapy.learn.vmodel import *

# ---#
def Balance(
    name: str,
    input_relation: str,
    y: str,
    cursor=None,
    method: str = "hybrid",
    ratio: float = 0.5,
):
    """
---------------------------------------------------------------------------
Creates a view with an equal distribution of the input data based on the 
response_column.
 
Parameters
----------
name: str
	Name of the the view.
input_relation: str
	Relation to use to create the new relation.
y: str
	Response column.
cursor: DBcursor, optional
	Vertica database cursor.
method: str, optional
	Method to use to do the balancing.
		hybrid : Performs over-sampling and under-sampling on different 
			classes so each class is equally represented.
		over   : Over-samples on all classes, with the exception of the 
			most majority class, towards the most majority class's cardinality. 
		under  : Under-samples on all classes, with the exception of the most 
			minority class, towards the most minority class's cardinality.
ratio: float, optional
	The desired ratio between the majority class and the minority class. This 
	value has no effect when used with balance method 'hybrid'.

Returns
-------
vDataFrame
	vDataFrame of the created view
	"""
    check_types(
        [
            ("name", name, [str],),
            ("input_relation", input_relation, [str],),
            ("y", y, [str],),
            ("method", method, ["hybrid", "over", "under"],),
            ("ratio", ratio, [float],),
        ]
    )
    cursor = check_cursor(cursor)[0]
    version(cursor=cursor, condition=[8, 1, 1])
    method = method.lower()
    sql = "SELECT BALANCE('{}', '{}', '{}', '{}_sampling' USING PARAMETERS sampling_ratio = {})".format(
        name, input_relation, y, method, ratio
    )
    executeSQL(cursor, sql, "Computing the Balanced Relation.")
    return vDataFrame(name, cursor)


# ---#
class CountVectorizer(vModel):
    """
---------------------------------------------------------------------------
Creates a Text Index which will count the occurences of each word in the 
data.
 
Parameters
----------
name: str
	Name of the the model.
cursor: DBcursor, optional
	Vertica database cursor.
lowercase: bool, optional
	Converts all the elements to lowercase before processing.
max_df: float, optional
	Keeps the words which represent less than this float in the total dictionary 
	distribution.
min_df: float, optional
	Keeps the words which represent more than this float in the total dictionary 
	distribution.
max_features: int, optional
	Keeps only the top words of the dictionary.
ignore_special: bool, optional
	Ignores all the special characters to build the dictionary.
max_text_size: int, optional
	The maximum size of the column which is the concatenation of all the text 
	columns during the fitting.
	"""

    def __init__(
        self,
        name: str,
        cursor=None,
        lowercase: bool = True,
        max_df: float = 1.0,
        min_df: float = 0.0,
        max_features: int = -1,
        ignore_special: bool = True,
        max_text_size: int = 2000,
    ):
        check_types([("name", name, [str],)])
        self.type, self.name = "CountVectorizer", name
        self.set_params(
            {
                "lowercase": lowercase,
                "max_df": max_df,
                "min_df": min_df,
                "max_features": max_features,
                "ignore_special": ignore_special,
                "max_text_size": max_text_size,
            }
        )
        cursor = check_cursor(cursor)[0]
        self.cursor = cursor

    # ---#
    def deploySQL(self):
        """
    ---------------------------------------------------------------------------
    Returns the SQL code needed to deploy the model.

    Returns
    -------
    str/list
        the SQL code needed to deploy the model.
        """
        sql = "SELECT * FROM (SELECT token, cnt / SUM(cnt) OVER () AS df, cnt, rnk FROM (SELECT token, COUNT(*) AS cnt, RANK() OVER (ORDER BY COUNT(*) DESC) AS rnk FROM {} GROUP BY 1) VERTICAPY_SUBTABLE) VERTICAPY_SUBTABLE WHERE (df BETWEEN {} AND {})".format(
            self.name, self.parameters["min_df"], self.parameters["max_df"]
        )
        if self.parameters["max_features"] > 0:
            sql += " AND (rnk <= {})".format(self.parameters["max_features"])
        return sql.format(", ".join(self.X), self.name)

    # ---#
    def fit(self, input_relation: (str, vDataFrame), X: list = []):
        """
	---------------------------------------------------------------------------
	Trains the model.

	Parameters
	----------
	input_relation: str/vDataFrame
		Training relation.
	X: list
		List of the predictors. If empty, all the columns will be used.

	Returns
	-------
	object
 		self
		"""
        if isinstance(X, str):
            X = [X]
        check_types(
            [("input_relation", input_relation, [str, vDataFrame],), ("X", X, [list],)]
        )
        self.cursor = check_cursor(self.cursor, input_relation, True)[0]
        if isinstance(input_relation, vDataFrame):
            if not (X):
                X = input_relation.get_columns()
            self.input_relation = input_relation.__genSQL__()
        else:
            if not (X):
                X = vDataFrame(input_relation, self.cursor).get_columns()
            self.input_relation = input_relation
        self.X = [str_column(elem) for elem in X]
        schema, relation = schema_relation(self.name)
        schema = str_column(schema)
        tmp_name = "{}.VERTICAPY_COUNT_VECTORIZER_{}_{}".format(
            schema, get_session(self.cursor), random.randint(0, 10000000),
        )
        try:
            self.drop()
        except:
            pass
        sql = "CREATE TABLE {}(id identity(2000) primary key, text varchar({})) ORDER BY id SEGMENTED BY HASH(id) ALL NODES KSAFE;"
        executeSQL(
            self.cursor,
            sql.format(tmp_name, self.parameters["max_text_size"]),
            "Computing the CountVectorizer - STEP 0.",
        )
        text = (
            " || ".join(self.X)
            if not (self.parameters["lowercase"])
            else "LOWER({})".format(" || ".join(self.X))
        )
        if self.parameters["ignore_special"]:
            text = "REGEXP_REPLACE({}, '[^a-zA-Z0-9\\s]+', '')".format(text)
        sql = "INSERT INTO {}(text) SELECT {} FROM {}".format(
            tmp_name, text, self.input_relation
        )
        executeSQL(self.cursor, sql, "Computing the CountVectorizer - STEP 1.")
        sql = "CREATE TEXT INDEX {} ON {}(id, text) stemmer NONE;".format(
            self.name, tmp_name
        )
        executeSQL(self.cursor, sql, "Computing the CountVectorizer - STEP 2.")
        stop_words = "SELECT token FROM (SELECT token, cnt / SUM(cnt) OVER () AS df, rnk FROM (SELECT token, COUNT(*) AS cnt, RANK() OVER (ORDER BY COUNT(*) DESC) AS rnk FROM {} GROUP BY 1) VERTICAPY_SUBTABLE) VERTICAPY_SUBTABLE WHERE not(df BETWEEN {} AND {})".format(
            self.name, self.parameters["min_df"], self.parameters["max_df"]
        )
        if self.parameters["max_features"] > 0:
            stop_words += " OR (rnk > {})".format(self.parameters["max_features"])
        self.cursor.execute(stop_words)
        self.stop_words_ = [item[0] for item in self.cursor.fetchall()]
        self.cursor.execute(self.deploySQL())
        self.vocabulary_ = [item[0] for item in self.cursor.fetchall()]
        self.countvectorizer_table = tmp_name
        model_save = {
            "type": "CountVectorizer",
            "input_relation": self.input_relation,
            "X": self.X,
            "countvectorizer_table": tmp_name,
            "lowercase": self.parameters["lowercase"],
            "max_df": self.parameters["max_df"],
            "min_df": self.parameters["min_df"],
            "max_features": self.parameters["max_features"],
            "ignore_special": self.parameters["ignore_special"],
            "max_text_size": self.parameters["max_text_size"],
            "vocabulary": self.vocabulary_,
            "stop_words": self.stop_words_,
        }
        insert_verticapy_schema(
            model_name=self.name,
            model_type="CountVectorizer",
            model_save=model_save,
            cursor=self.cursor,
        )
        return self

    # ---#
    def transform(self):
        """
	---------------------------------------------------------------------------
	Creates a vDataFrame of the model.

	Returns
	-------
	vDataFrame
 		object result of the model transformation.
		"""
        return vdf_from_relation(
            "({}) VERTICAPY_SUBTABLE".format(self.deploySQL()), self.name, self.cursor
        )


# ---#
class Normalizer(Preprocessing):
    """
---------------------------------------------------------------------------
Creates a Vertica Normalizer object.
 
Parameters
----------
name: str
	Name of the the model.
cursor: DBcursor, optional
	Vertica database cursor.
method: str, optional
	Method to use to normalize.
		zscore        : Normalization using the Z-Score (avg and std).
		(x - avg) / std
		robust_zscore : Normalization using the Robust Z-Score (median and mad).
		(x - median) / (1.4826 * mad)
		minmax        : Normalization using the MinMax (min and max).
		(x - min) / (max - min)
	"""

    def __init__(self, name: str, cursor=None, method: str = "zscore"):
        check_types([("name", name, [str],)])
        self.type, self.name = "Normalizer", name
        self.set_params({"method": method})
        cursor = check_cursor(cursor)[0]
        self.cursor = cursor
        version(cursor=cursor, condition=[8, 1, 0])


# ---#
class StandardScaler(Normalizer):
    """i.e. Normalizer with param method = 'zscore'"""

    def __init__(
        self, name: str, cursor=None,
    ):
        super().__init__(name, cursor, "zscore")


# ---#
class RobustScaler(Normalizer):
    """i.e. Normalizer with param method = 'robust_zscore'"""

    def __init__(
        self, name: str, cursor=None,
    ):
        super().__init__(name, cursor, "robust_zscore")


# ---#
class MinMaxScaler(Normalizer):
    """i.e. Normalizer with param method = 'minmax'"""

    def __init__(
        self, name: str, cursor=None,
    ):
        super().__init__(name, cursor, "minmax")


# ---#
class OneHotEncoder(Preprocessing):
    """
---------------------------------------------------------------------------
Creates a Vertica One Hot Encoder object.
 
Parameters
----------
name: str
	Name of the the model.
cursor: DBcursor, optional
	Vertica database cursor.
extra_levels: dict, optional
	Additional levels in each category that are not in the input relation.
drop_first: bool, optional
    If set to True, it treats the first level of the categorical variable 
    as the reference level. Otherwise, every level of the categorical variable 
    has a corresponding column in the output view.
ignore_null: bool, optional
    If set to True, Null values set all corresponding one-hot binary columns to null. 
    Otherwise, null values in the input columns are treated as a categorical level.
separator: str, optional
    The character that separates the input variable name and the indicator variable 
    level in the output table.To avoid using any separator, set this parameter to 
    null value.
column_naming: str, optional
    Appends categorical levels to column names according to the specified method:
        indices                : Uses integer indices to represent categorical levels.
        values/values_relaxed  : Both methods use categorical level names. If duplicate 
                                 column names occur, the function attempts to disambiguate 
                                 them by appending _n, where n is a zero-based integer 
                                 index (_0, _1,…).
null_column_name: str, optional
    The string used in naming the indicator column for null values, used only if 
    ignore_null is set to false and column_naming is set to values or values_relaxed.
	"""

    def __init__(
        self,
        name: str,
        cursor=None,
        extra_levels: dict = {},
        drop_first: bool = True,
        ignore_null: bool = True,
        separator: str = "_",
        column_naming: str = "indices",
        null_column_name: str = "null",
    ):
        check_types(
            [
                ("name", name, [str],),
                ("extra_levels", extra_levels, [dict],),
                ("drop_first", drop_first, [bool],),
                ("ignore_null", ignore_null, [bool],),
                ("separator", separator, [str],),
                (
                    "column_naming",
                    column_naming,
                    ["indices", "values", "values_relaxed"],
                ),
                ("null_column_name", null_column_name, [str],),
            ]
        )
        self.type, self.name = "OneHotEncoder", name
        self.set_params(
            {
                "extra_levels": extra_levels,
                "drop_first": drop_first,
                "ignore_null": ignore_null,
                "separator": separator,
                "column_naming": column_naming,
                "null_column_name": null_column_name,
            }
        )
        cursor = check_cursor(cursor)[0]
        self.cursor = cursor
        version(cursor=cursor, condition=[9, 0, 0])
