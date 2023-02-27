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
from typing import Literal, Union

import verticapy._config.config as conf
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._gen import gen_tmp_name
from verticapy._utils._sql._format import quote_ident, schema_relation, clean_query
from verticapy._utils._sql._sys import _executeSQL
from verticapy._utils._sql._vertica_version import check_minimum_version


from verticapy.core.tablesample.base import TableSample
from verticapy.core.vdataframe.base import vDataFrame

import verticapy.machine_learning.memmodel as mm
from verticapy.machine_learning.vertica.base import Preprocessing, vModel

from verticapy.sql.insert import insert_verticapy_schema


@check_minimum_version
@save_verticapy_logs
def Balance(
    name: str,
    input_relation: str,
    y: str,
    method: Literal["hybrid", "over", "under"] = "hybrid",
    ratio: float = 0.5,
):
    """
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
    _executeSQL(
        query=f"""
            SELECT 
                /*+LABEL('learn.preprocessing.Balance')*/ 
                BALANCE('{name}', 
                        '{input_relation}', 
                        '{y}', 
                        '{method}_sampling' 
                        USING PARAMETERS 
                        sampling_ratio = {ratio})""",
        title="Computing the Balanced Relation.",
    )
    return vDataFrame(name)


class CountVectorizer(vModel):
    """
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

    @property
    def _vertica_fit_sql(self) -> Literal[""]:
        return ""

    @property
    def _vertica_transform_sql(self) -> Literal[""]:
        return ""

    @property
    def _vertica_inverse_transform_sql(self) -> Literal[""]:
        return ""

    @property
    def _model_category(self) -> Literal["UNSUPERVISED"]:
        return "UNSUPERVISED"

    @property
    def _model_subcategory(self) -> Literal["PREPROCESSING"]:
        return "PREPROCESSING"

    @property
    def _model_type(self) -> Literal["CountVectorizer"]:
        return "CountVectorizer"

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
        self.model_name = name
        self.parameters = {
            "lowercase": lowercase,
            "max_df": max_df,
            "min_df": min_df,
            "max_features": max_features,
            "ignore_special": ignore_special,
            "max_text_size": max_text_size,
        }

    def compute_stop_words(self):
        """
    Computes the CountVectorizer Stop Words. It will affect the result to the
    stop_words_ attribute.
        """
        query = self.deploySQL(return_main_table=True)
        query = query.format(
            "/*+LABEL('learn.preprocessing.CountVectorizer.compute_stop_words')*/ token",
            "not",
        )
        if self.parameters["max_features"] > 0:
            query += f" OR (rnk > {self.parameters['max_features']})"
        res = _executeSQL(query=query, print_time_sql=False, method="fetchall")
        self.stop_words_ = [item[0] for item in res]

    def compute_vocabulary(self):
        """
    Computes the CountVectorizer Vocabulary. It will affect the result to the
    vocabulary_ attribute.
        """
        res = _executeSQL(self.deploySQL(), print_time_sql=False, method="fetchall")
        self.vocabulary_ = [item[0] for item in res]

    def deploySQL(self, return_main_table: bool = False):
        """
    Returns the SQL code needed to deploy the model.

    Returns
    -------
    str/list
        the SQL code needed to deploy the model.
        """
        query = f"""
            SELECT 
                {{}} 
            FROM (SELECT 
                      token, 
                      cnt / SUM(cnt) OVER () AS df, 
                      cnt, 
                      rnk 
                  FROM (SELECT 
                            token, 
                            COUNT(*) AS cnt, 
                            RANK() OVER (ORDER BY COUNT(*) DESC) AS rnk 
                        FROM {self.model_name} GROUP BY 1) VERTICAPY_SUBTABLE) VERTICAPY_SUBTABLE 
                        WHERE {{}}(df BETWEEN {self.parameters['min_df']} 
                                   AND {self.parameters['max_df']})"""
        if return_main_table:
            return query
        if self.parameters["max_features"] > 0:
            query += f" AND (rnk <= {self.parameters['max_features']})"

        return clean_query(query.format("*", ""))

    def fit(self, input_relation: Union[str, vDataFrame], X: Union[str, list] = []):
        """
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
        if conf.get_option("overwrite_model"):
            self.drop()
        else:
            does_model_exist(name=self.model_name, raise_error=True)
        if isinstance(input_relation, vDataFrame):
            if not (X):
                X = input_relation.get_columns()
            self.input_relation = input_relation._genSQL()
        else:
            if not (X):
                X = vDataFrame(input_relation).get_columns()
            self.input_relation = input_relation
        self.X = [quote_ident(elem) for elem in X]
        schema, relation = schema_relation(self.model_name)
        schema = quote_ident(schema)
        tmp_name = gen_tmp_name(schema=schema, name="countvectorizer")
        try:
            self.drop()
        except:
            pass
        _executeSQL(
            query=f"""
                CREATE TABLE {tmp_name}
                (id identity(2000) primary key, 
                 text varchar({self.parameters['max_text_size']})) 
                ORDER BY id SEGMENTED BY HASH(id) ALL NODES KSAFE;""",
            title="Computing the CountVectorizer [Step 0].",
        )
        if not (self.parameters["lowercase"]):
            text = " || ".join(self.X)
        else:
            text = f"LOWER({' || '.join(self.X)})"
        if self.parameters["ignore_special"]:
            text = f"REGEXP_REPLACE({text}, '[^a-zA-Z0-9\\s]+', '')"
        _executeSQL(
            query=f"""
                INSERT 
                    /*+LABEL('learn.preprocessing.CountVectorizer.fit')*/ 
                INTO {tmp_name}(text) 
                SELECT {text} FROM {self.input_relation}""",
            title="Computing the CountVectorizer [Step 1].",
        )
        _executeSQL(
            query=f"""
                CREATE TEXT INDEX {self.model_name} 
                ON {tmp_name}(id, text) stemmer NONE;""",
            title="Computing the CountVectorizer [Step 2].",
        )
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
            model_name=self.model_name,
            model_type="CountVectorizer",
            model_save=model_save,
        )
        return self

    def transform(self):
        """
	Creates a vDataFrame of the model.

	Returns
	-------
	vDataFrame
 		object result of the model transformation.
		"""
        return vDataFrame(self.deploySQL())


class Scaler(Preprocessing):
    """
Creates a Vertica Scaler object.
 
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

    @property
    def _vertica_fit_sql(self) -> Literal["NORMALIZE_FIT"]:
        return "NORMALIZE_FIT"

    @property
    def _vertica_transform_sql(self) -> Literal["APPLY_NORMALIZE"]:
        return "APPLY_NORMALIZE"

    @property
    def _vertica_inverse_transform_sql(self) -> Literal["REVERSE_NORMALIZE"]:
        return "REVERSE_NORMALIZE"

    @property
    def _model_category(self) -> Literal["UNSUPERVISED"]:
        return "UNSUPERVISED"

    @property
    def _model_subcategory(self) -> Literal["PREPROCESSING"]:
        return "PREPROCESSING"

    @property
    def _model_type(self) -> Literal["Scaler"]:
        return "Scaler"

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self, name: str, method: Literal["zscore", "robust_zscore", "minmax"] = "zscore"
    ):
        self.model_name = name
        self.parameters = {"method": str(method).lower()}

    def _compute_attributes(self) -> None:
        """
        Computes the model's attributes.
        """
        values = self.get_attr("details").to_numpy()[:, 1:].astype(float)
        if self.parameters["method"] == "minmax":
            self.min_ = values[:, 0]
            self.max_ = values[:, 1]
        else:
            self.mean_ = values[:, 0]
            self.std_ = values[:, 1]
        return None

    def to_memmodel(self) -> mm.Scaler:
        """
        Converts the model to an InMemory object which
        can be used to do different types of predictions.
        """
        if self.parameters["method"] == "minmax":
            return mm.MinMaxScaler(self.min_, self.max_)
        else:
            return mm.StandardScaler(self.mean_, self.std_)


class StandardScaler(Scaler):
    """i.e. Scaler with param method = 'zscore'"""

    def __init__(self, name: str):
        super().__init__(name, "zscore")


class RobustScaler(Scaler):
    """i.e. Scaler with param method = 'robust_zscore'"""

    def __init__(self, name: str):
        super().__init__(name, "robust_zscore")


class MinMaxScaler(Scaler):
    """i.e. Scaler with param method = 'minmax'"""

    def __init__(self, name: str):
        super().__init__(name, "minmax")


class OneHotEncoder(Preprocessing):
    """
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

    @property
    def _vertica_fit_sql(self) -> Literal["ONE_HOT_ENCODER_FIT"]:
        return "ONE_HOT_ENCODER_FIT"

    @property
    def _vertica_transform_sql(self) -> Literal["APPLY_ONE_HOT_ENCODER"]:
        return "APPLY_ONE_HOT_ENCODER"

    @property
    def _vertica_inverse_transform_sql(self) -> Literal[""]:
        return ""

    @property
    def _model_category(self) -> Literal["UNSUPERVISED"]:
        return "UNSUPERVISED"

    @property
    def _model_subcategory(self) -> Literal["PREPROCESSING"]:
        return "PREPROCESSING"

    @property
    def _model_type(self) -> Literal["OneHotEncoder"]:
        return "OneHotEncoder"

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str,
        extra_levels: dict = {},
        drop_first: bool = True,
        ignore_null: bool = True,
        separator: str = "_",
        column_naming: Literal["indices", "values", "values_relaxed"] = "indices",
        null_column_name: str = "null",
    ):
        self.model_name = name
        self.parameters = {
            "extra_levels": extra_levels,
            "drop_first": drop_first,
            "ignore_null": ignore_null,
            "separator": separator,
            "column_naming": str(column_naming).lower(),
            "null_column_name": null_column_name,
        }

    def _compute_attributes(self) -> None:
        """
        Computes the model's attributes.
        """
        query = f"""SELECT 
                        category_name, 
                        category_level::varchar, 
                        category_level_index 
                    FROM (SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS 
                                    model_name = '{self.model_name}', 
                                    attr_name = 'integer_categories')) 
                                    VERTICAPY_SUBTABLE"""
        try:
            self.cat_ = TableSample.read_sql(
                query=f"""{query}
                          UNION ALL 
                          SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS 
                                    model_name = '{self.model_name}', 
                                    attr_name = 'varchar_categories')""",
                title="Getting Model Attributes.",
            )
        except:
            try:
                self.cat_ = TableSample.read_sql(
                    query=query, title="Getting Model Attributes.",
                )
            except:
                self.cat_ = self.get_attr("varchar_categories")
        self.cat_ = np.array(self.cat_)
        cat = self._compute_ohe_array(self.cat_[:, 0:2])
        cat_list_idx = []
        for i, x1 in enumerate(cat[0]):
            for j, x2 in enumerate(self.X):
                if x2.lower()[1:-1] == x1:
                    cat_list_idx += [j]
        categories = []
        for i in cat_list_idx:
            categories += [cat[1][i]]
        self.categories_ = categories
        self.column_naming_ = self.parameters["column_naming"]
        self.drop_first_ = self.parameters["drop_first"]
        return None

    @staticmethod
    def _compute_ohe_array(categories: list):
        # Allows to split the One Hot Encoder Array by features categories
        cat, tmp_cat = [], []
        init_cat, X = categories[0][0], [categories[0][0]]
        for c in categories:
            if c[0] != init_cat:
                init_cat = c[0]
                X += [c[0]]
                cat += [tmp_cat]
                tmp_cat = [c[1]]
            else:
                tmp_cat += [c[1]]
        cat += [tmp_cat]
        return [X, cat]

    def to_memmodel(self) -> mm.OneHotEncoder:
        """
        Converts the model to an InMemory object which
        can be used to do different types of predictions.
        """
        return mm.OneHotEncoder(self.categories_, self.column_naming_, self.drop_first_)
