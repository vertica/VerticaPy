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
import random
from typing import Union

# VerticaPy Modules
from verticapy.decorators import (
    save_verticapy_logs,
    check_dtypes,
    check_minimum_version,
)
from verticapy.utilities import *
from verticapy.toolbox import *
from verticapy import vDataFrame
from verticapy.learn.vmodel import *

# ---#
@check_minimum_version
@check_dtypes
@save_verticapy_logs
def Balance(
    name: str, input_relation: str, y: str, method: str = "hybrid", ratio: float = 0.5,
):
    """
----------------------------------------------------------------------------------------
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
    raise_error_if_not_in("method", method, ["hybrid", "over", "under"])
    sql = f"SELECT /*+LABEL('learn.preprocessing.Balance')*/ BALANCE('{name}', '{input_relation}', '{y}', '{method}_sampling' USING PARAMETERS sampling_ratio = {ratio})"
    executeSQL(sql, "Computing the Balanced Relation.")
    return vDataFrame(name)


# ---#
class CountVectorizer(vModel):
    """
----------------------------------------------------------------------------------------
Creates a Text Index which will count the occurences of each word in the 
data.
 
Parameters
----------
name: str
	Name of the the model.
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

    @check_dtypes
    @save_verticapy_logs
    def __init__(
        self,
        name: str,
        lowercase: bool = True,
        max_df: float = 1.0,
        min_df: float = 0.0,
        max_features: int = -1,
        ignore_special: bool = True,
        max_text_size: int = 2000,
    ):
        self.type, self.name = "CountVectorizer", name
        self.MODEL_TYPE = "UNSUPERVISED"
        self.MODEL_SUBTYPE = "PREPROCESSING"
        self.parameters = {
            "lowercase": lowercase,
            "max_df": max_df,
            "min_df": min_df,
            "max_features": max_features,
            "ignore_special": ignore_special,
            "max_text_size": max_text_size,
        }

    # ---#
    def compute_stop_words(self):
        """
    ----------------------------------------------------------------------------------------
    Computes the CountVectorizer Stop Words. It will affect the result to the
    stop_words_ attribute.
        """
        stop_words = """SELECT /*+LABEL('learn.preprocessing.CountVectorizer.fit')*/
                            token 
                        FROM 
                            (SELECT 
                                token, 
                                cnt / SUM(cnt) OVER () AS df, 
                                rnk 
                            FROM 
                                (SELECT 
                                    token, 
                                    COUNT(*) AS cnt, 
                                    RANK() OVER (ORDER BY COUNT(*) DESC) AS rnk 
                                 FROM {0} GROUP BY 1) VERTICAPY_SUBTABLE) VERTICAPY_SUBTABLE 
                                 WHERE not(df BETWEEN {1} AND {2})""".format(
            self.name, self.parameters["min_df"], self.parameters["max_df"]
        )
        if self.parameters["max_features"] > 0:
            stop_words += " OR (rnk > {})".format(self.parameters["max_features"])
        res = executeSQL(stop_words, print_time_sql=False, method="fetchall")
        self.stop_words_ = [item[0] for item in res]

    # ---#
    def compute_vocabulary(self):
        """
    ----------------------------------------------------------------------------------------
    Computes the CountVectorizer Vocabulary. It will affect the result to the
    vocabulary_ attribute.
        """
        res = executeSQL(self.deploySQL(), print_time_sql=False, method="fetchall")
        self.vocabulary_ = [item[0] for item in res]

    # ---#
    def deploySQL(self):
        """
    ----------------------------------------------------------------------------------------
    Returns the SQL code needed to deploy the model.

    Returns
    -------
    str/list
        the SQL code needed to deploy the model.
        """
        sql = """SELECT 
                    * 
                 FROM (SELECT 
                          token, 
                          cnt / SUM(cnt) OVER () AS df, 
                          cnt, 
                          rnk 
                 FROM (SELECT 
                          token, 
                          COUNT(*) AS cnt, 
                          RANK() OVER (ORDER BY COUNT(*) DESC) AS rnk 
                       FROM {0} GROUP BY 1) VERTICAPY_SUBTABLE) VERTICAPY_SUBTABLE 
                       WHERE (df BETWEEN {1} AND {2})""".format(
            self.name, self.parameters["min_df"], self.parameters["max_df"]
        )
        if self.parameters["max_features"] > 0:
            sql += " AND (rnk <= {})".format(self.parameters["max_features"])
        return sql.format(", ".join(self.X), self.name)

    # ---#
    @check_dtypes
    def fit(self, input_relation: Union[str, vDataFrame], X: Union[str, list] = []):
        """
	----------------------------------------------------------------------------------------
	Trains the model.

	Parameters
	----------
	input_relation: str / vDataFrame
		Training relation.
	X: str / list
		List of the predictors. If empty, all the columns will be used.

	Returns
	-------
	object
 		self
		"""
        if isinstance(X, str):
            X = [X]
        if verticapy.OPTIONS["overwrite_model"]:
            self.drop()
        else:
            does_model_exist(name=self.name, raise_error=True)
        if isinstance(input_relation, vDataFrame):
            if not (X):
                X = input_relation.get_columns()
            self.input_relation = input_relation.__genSQL__()
        else:
            if not (X):
                X = vDataFrame(input_relation).get_columns()
            self.input_relation = input_relation
        self.X = [quote_ident(elem) for elem in X]
        schema, relation = schema_relation(self.name)
        schema = quote_ident(schema)
        tmp_name = gen_tmp_name(schema=schema, name="countvectorizer")
        try:
            self.drop()
        except:
            pass
        sql = """CREATE TABLE {0}(id identity(2000) primary key, 
                                  text varchar({1})) 
                 ORDER BY id SEGMENTED BY HASH(id) ALL NODES KSAFE;"""
        executeSQL(
            sql.format(tmp_name, self.parameters["max_text_size"]),
            title="Computing the CountVectorizer [Step 0].",
        )
        text = (
            " || ".join(self.X)
            if not (self.parameters["lowercase"])
            else "LOWER({})".format(" || ".join(self.X))
        )
        if self.parameters["ignore_special"]:
            text = f"REGEXP_REPLACE({text}, '[^a-zA-Z0-9\\s]+', '')"
        sql = "INSERT /*+LABEL('learn.preprocessing.CountVectorizer.fit')*/ INTO {}(text) SELECT {} FROM {}".format(
            tmp_name, text, self.input_relation
        )
        executeSQL(sql, "Computing the CountVectorizer [Step 1].")
        sql = "CREATE TEXT INDEX {} ON {}(id, text) stemmer NONE;".format(
            self.name, tmp_name
        )
        executeSQL(sql, "Computing the CountVectorizer [Step 2].")
        self.compute_stop_words()
        self.compute_vocabulary()
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
        }
        insert_verticapy_schema(
            model_name=self.name, model_type="CountVectorizer", model_save=model_save,
        )
        return self

    # ---#
    def transform(self):
        """
	----------------------------------------------------------------------------------------
	Creates a vDataFrame of the model.

	Returns
	-------
	vDataFrame
 		object result of the model transformation.
		"""
        return vDataFrameSQL(
            "({}) VERTICAPY_SUBTABLE".format(self.deploySQL()), self.name,
        )


# ---#
class Normalizer(Preprocessing):
    """
----------------------------------------------------------------------------------------
Creates a Vertica Normalizer object.
 
Parameters
----------
name: str
	Name of the the model.
method: str, optional
	Method to use to normalize.
		zscore        : Normalization using the Z-Score (avg and std).
		(x - avg) / std
		robust_zscore : Normalization using the Robust Z-Score (median and mad).
		(x - median) / (1.4826 * mad)
		minmax        : Normalization using the MinMax (min and max).
		(x - min) / (max - min)
	"""

    @check_minimum_version
    @check_dtypes
    @save_verticapy_logs
    def __init__(self, name: str, method: str = "zscore"):
        raise_error_if_not_in(
            "method", str(method).lower(), ["zscore", "robust_zscore", "minmax"]
        )
        self.type, self.name = "Normalizer", name
        self.VERTICA_FIT_FUNCTION_SQL = "NORMALIZE_FIT"
        self.VERTICA_TRANSFORM_FUNCTION_SQL = "APPLY_NORMALIZE"
        self.VERTICA_INVERSE_TRANSFORM_FUNCTION_SQL = "REVERSE_NORMALIZE"
        self.MODEL_TYPE = "UNSUPERVISED"
        self.MODEL_SUBTYPE = "PREPROCESSING"
        self.parameters = {"method": str(method).lower()}


# ---#
class StandardScaler(Normalizer):
    """i.e. Normalizer with param method = 'zscore'"""

    def __init__(self, name: str):
        super().__init__(name, "zscore")


# ---#
class RobustScaler(Normalizer):
    """i.e. Normalizer with param method = 'robust_zscore'"""

    def __init__(self, name: str):
        super().__init__(name, "robust_zscore")


# ---#
class MinMaxScaler(Normalizer):
    """i.e. Normalizer with param method = 'minmax'"""

    def __init__(self, name: str):
        super().__init__(name, "minmax")


# ---#
class OneHotEncoder(Preprocessing):
    """
----------------------------------------------------------------------------------------
Creates a Vertica One Hot Encoder object.
 
Parameters
----------
name: str
	Name of the the model.
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

    @check_minimum_version
    @check_dtypes
    @save_verticapy_logs
    def __init__(
        self,
        name: str,
        extra_levels: dict = {},
        drop_first: bool = True,
        ignore_null: bool = True,
        separator: str = "_",
        column_naming: str = "indices",
        null_column_name: str = "null",
    ):
        raise_error_if_not_in(
            "column_naming",
            str(column_naming).lower(),
            ["indices", "values", "values_relaxed"],
        )
        self.type, self.name = "OneHotEncoder", name
        self.VERTICA_FIT_FUNCTION_SQL = "ONE_HOT_ENCODER_FIT"
        self.VERTICA_TRANSFORM_FUNCTION_SQL = "APPLY_ONE_HOT_ENCODER"
        self.VERTICA_INVERSE_TRANSFORM_FUNCTION_SQL = ""
        self.MODEL_TYPE = "UNSUPERVISED"
        self.MODEL_SUBTYPE = "PREPROCESSING"
        self.parameters = {
            "extra_levels": extra_levels,
            "drop_first": drop_first,
            "ignore_null": ignore_null,
            "separator": separator,
            "column_naming": str(column_naming).lower(),
            "null_column_name": null_column_name,
        }
