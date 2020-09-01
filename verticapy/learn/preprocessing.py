# (c) Copyright [2018-2020] Micro Focus or one of its affiliates.
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
# VerticaPy Modules
from verticapy.utilities import *
from verticapy.toolbox import *
from verticapy import vDataFrame
from verticapy.connections.connect import read_auto_connect
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
	Vertica DB cursor.
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
            ("name", name, [str], False),
            ("input_relation", input_relation, [str], False),
            ("y", y, [str], False),
            ("method", method, ["hybrid", "over", "under"], True),
            ("ratio", ratio, [float], False),
        ]
    )
    if not (cursor):
        cursor = read_auto_connect().cursor()
    else:
        check_cursor(cursor)
    version(cursor=cursor, condition=[8, 1, 1])
    method = method.lower()
    sql = "SELECT BALANCE('{}', '{}', '{}', '{}_sampling' USING PARAMETERS sampling_ratio = {})".format(
        name, input_relation, y, method, ratio
    )
    cursor.execute(sql)
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
	Vertica DB cursor.
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
        check_types([("name", name, [str], False)])
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
        if not (cursor):
            cursor = read_auto_connect().cursor()
        else:
            check_cursor(cursor)
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
        sql = "SELECT * FROM (SELECT token, cnt / SUM(cnt) OVER () AS df, cnt, rnk FROM (SELECT token, COUNT(*) AS cnt, RANK() OVER (ORDER BY COUNT(*) DESC) AS rnk FROM {} GROUP BY 1) x) y WHERE (df BETWEEN {} AND {})".format(
            self.name, self.parameters["min_df"], self.parameters["max_df"]
        )
        if self.parameters["max_features"] > 0:
            sql += " AND (rnk <= {})".format(self.parameters["max_features"])
        return sql.format(", ".join(self.X), self.name)

    # ---#
    def fit(self, input_relation: str, X: list):
        """
	---------------------------------------------------------------------------
	Trains the model.

	Parameters
	----------
	input_relation: str
		Train relation.
	X: list
		List of the predictors.

	Returns
	-------
	object
 		self
		"""
        check_types(
            [("input_relation", input_relation, [str], False), ("X", X, [list], False)]
        )
        self.input_relation = input_relation
        self.X = [str_column(elem) for elem in X]
        schema, relation = schema_relation(input_relation)
        schema = str_column(schema)
        relation_alpha = "".join(ch for ch in relation if ch.isalnum())
        try:
            self.cursor.execute(
                "DROP TABLE IF EXISTS {}.VERTICAPY_COUNT_VECTORIZER_{} CASCADE".format(
                    schema, relation_alpha
                )
            )
        except:
            pass
        sql = "CREATE TABLE {}.VERTICAPY_COUNT_VECTORIZER_{}(id identity(2000) primary key, text varchar({})) ORDER BY id SEGMENTED BY HASH(id) ALL NODES KSAFE;"
        self.cursor.execute(
            sql.format(schema, relation_alpha, self.parameters["max_text_size"])
        )
        text = (
            " || ".join(self.X)
            if not (self.parameters["lowercase"])
            else "LOWER({})".format(" || ".join(self.X))
        )
        if self.parameters["ignore_special"]:
            text = "REGEXP_REPLACE({}, '[^a-zA-Z0-9\\s]+', '')".format(text)
        sql = "INSERT INTO {}.VERTICAPY_COUNT_VECTORIZER_{}(text) SELECT {} FROM {}".format(
            schema, relation_alpha, text, input_relation
        )
        self.cursor.execute(sql)
        sql = "CREATE TEXT INDEX {} ON {}.VERTICAPY_COUNT_VECTORIZER_{}(id, text) stemmer NONE;".format(
            self.name, schema, relation_alpha
        )
        self.cursor.execute(sql)
        stop_words = "SELECT token FROM (SELECT token, cnt / SUM(cnt) OVER () AS df, rnk FROM (SELECT token, COUNT(*) AS cnt, RANK() OVER (ORDER BY COUNT(*) DESC) AS rnk FROM {} GROUP BY 1) x) y WHERE not(df BETWEEN {} AND {})".format(
            self.name, self.parameters["min_df"], self.parameters["max_df"]
        )
        if self.parameters["max_features"] > 0:
            stop_words += " OR (rnk > {})".format(self.parameters["max_features"])
        self.cursor.execute(stop_words)
        self.stop_words_ = [item[0] for item in self.cursor.fetchall()]
        self.cursor.execute(self.deploySQL())
        self.vocabulary_ = [item[0] for item in self.cursor.fetchall()]
        model_save = {
            "type": "CountVectorizer",
            "input_relation": self.input_relation,
            "X": self.X,
            "lowercase": self.parameters["lowercase"],
            "max_df": self.parameters["max_df"],
            "min_df": self.parameters["min_df"],
            "max_features": self.parameters["max_features"],
            "ignore_special": self.parameters["ignore_special"],
            "max_text_size": self.parameters["max_text_size"],
            "vocabulary": self.vocabulary_,
            "stop_words": self.stop_words_,
        }
        path = os.path.dirname(
            verticapy.__file__
        ) + "/learn/models/{}.verticapy".format(self.name)
        file = open(path, "x")
        file.write("model_save = " + str(model_save))
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
            "({}) x".format(self.deploySQL()), self.name, self.cursor
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
	Vertica DB cursor.
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
        check_types([("name", name, [str], False)])
        self.type, self.name = "Normalizer", name
        self.set_params({"method": method})
        if not (cursor):
            cursor = read_auto_connect().cursor()
        else:
            check_cursor(cursor)
        self.cursor = cursor
        version(cursor=cursor, condition=[8, 1, 0])

    # ---#
    def deployInverseSQL(self, X: list = []):
        """
    ---------------------------------------------------------------------------
    Returns the SQL code needed to deploy the inverse model. 

    Parameters
    ----------
    X: list, optional
        List of the columns used to deploy the self. If empty, the model
        predictors will be used.

    Returns
    -------
    str
        the SQL code needed to deploy the inverse self.
        """
        check_types([("X", X, [list], False)])
        X = [str_column(elem) for elem in X]
        fun = self.get_model_fun()[2]
        sql = "{}({} USING PARAMETERS model_name = '{}', match_by_pos = 'true')"
        return sql.format(fun, ", ".join(self.X if not (X) else X), self.name)

    # ---#
    def inverse_transform_preprocessing(self, vdf=None, X: list = []):
        """
    ---------------------------------------------------------------------------
    Creates a vDataFrame of the model.

    Parameters
    ----------
    vdf: vDataFrame, optional
        input vDataFrame.
    X: list, optional
        List of the input vcolumns.

    Returns
    -------
    vDataFrame
        object result of the model transformation.
        """
        check_types([("X", X, [list], False)])
        if vdf:
            check_types(vdf=["vdf", vdf])
            X = vdf_columns_names(X, vdf)
            relation = vdf.__genSQL__()
        else:
            relation = self.input_relation
            X = [str_column(elem) for elem in X]
        return vdf_from_relation(
            "(SELECT {} FROM {}) x".format(
                self.deployInverseSQL(self.X if not (X) else X), relation
            ),
            self.name,
            self.cursor,
        )


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
	Vertica DB cursor.
extra_levels: dict, optional
	Additional levels in each category that are not in the input relation.

Attributes
----------
After the object creation, all the parameters become attributes. 
The model will also create extra attributes when fitting the model:

param: tablesample
	The One Hot Encoder parameters.
input_relation: str
	Train relation.
X: list
	List of the predictors.
	"""

    def __init__(self, name: str, cursor=None, extra_levels: dict = {}):
        check_types([("name", name, [str], False)])
        self.type, self.name = "OneHotEncoder", name
        self.set_params({"extra_levels": extra_levels})
        if not (cursor):
            cursor = read_auto_connect().cursor()
        else:
            check_cursor(cursor)
        self.cursor = cursor
        version(cursor=cursor, condition=[9, 0, 0])
