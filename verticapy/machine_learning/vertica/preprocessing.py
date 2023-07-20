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
from abc import abstractmethod
from typing import Literal, Optional
import numpy as np

from vertica_python.errors import QueryError

import verticapy._config.config as conf
from verticapy._typing import NoneType, SQLColumns, SQLRelation
from verticapy._utils._gen import gen_tmp_name
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import (
    clean_query,
    format_type,
    quote_ident,
    schema_relation,
)
from verticapy._utils._sql._sys import _executeSQL
from verticapy._utils._sql._vertica_version import check_minimum_version


from verticapy.core.tablesample.base import TableSample
from verticapy.core.vdataframe.base import vDataFrame

import verticapy.machine_learning.memmodel as mm
from verticapy.machine_learning.vertica.base import Unsupervised, VerticaModel

from verticapy.sql.drop import drop

"""
General Functions.
"""


@check_minimum_version
@save_verticapy_logs
def Balance(
    name: str,
    input_relation: str,
    y: str,
    method: Literal["hybrid", "over", "under"] = "hybrid",
    ratio: float = 0.5,
) -> vDataFrame:
    """
    Creates a view with an equal distribution of the
    input data based on the response_column.

    Parameters
    ----------
    name: str
        Name of the view.
    input_relation: str
        Relation used to create the new relation.
    y: str
        Response column.
    method: str, optional
        Method used to do the balancing.
            hybrid : Performs  over-sampling   and
                     under-sampling  on  different
                     classes so that each class is
                     equally represented.
            over   : Over-samples on  all classes,
                     except the most represented
                     class, towards the  most
                     represented class's cardinality.
            under  : Under-samples on  all classes,
                     except the least represented
                     class,  towards  the least
                     represented class's cardinality.
    ratio: float, optional
        The desired ratio between the majority class
        and the minority class. This value has no
        effect when used with the 'hybrid' balance
        method.

    Returns
    -------
    vDataFrame
        vDataFrame of the created view.
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


"""
General Classes.
"""


class Preprocessing(Unsupervised):
    # Properties.

    @property
    @abstractmethod
    def _vertica_transform_sql(self) -> str:
        """Must be overridden in child class"""
        raise NotImplementedError

    @property
    @abstractmethod
    def _vertica_inverse_transform_sql(self) -> str:
        """Must be overridden in child class"""
        raise NotImplementedError

    # I/O Methods.

    def _get_names(
        self, inverse: bool = False, X: Optional[SQLColumns] = None
    ) -> SQLColumns:
        """
        Returns the Transformation output names.

        Parameters
        ----------
        inverse: bool, optional
            If set to True, returns the inverse transform
            output names.
        X: list, optional
            List of the columns used to get the model output
            names. If empty, the model predictors names are
            used.

        Returns
        -------
        list
            names.
        """
        X = format_type(X, dtype=list)
        X = quote_ident(X)
        if not X:
            X = self.X
        if self._model_type in ("PCA", "SVD", "MCA") and not inverse:
            if self._model_type in ("PCA", "SVD"):
                n = self.parameters["n_components"]
                if not n:
                    n = len(self.X)
            else:
                n = len(self.X)
            return [f"col{i}" for i in range(1, n + 1)]
        elif self._model_type == "OneHotEncoder" and not inverse:
            names = []
            for column in self.X:
                k = 0
                for i in range(len(self.cat_["category_name"])):
                    if quote_ident(self.cat_["category_name"][i]) == quote_ident(
                        column
                    ):
                        if (k != 0 or not self.parameters["drop_first"]) and (
                            not self.parameters["ignore_null"]
                            or not (
                                isinstance(self.cat_["category_level"][i], NoneType)
                            )
                        ):
                            if self.parameters["column_naming"] == "indices":
                                name = f'"{quote_ident(column)[1:-1]}{self.parameters["separator"]}'
                                name += f'{self.cat_["category_level_index"][i]}"'
                                names += [name]
                            else:
                                if not (
                                    isinstance(self.cat_["category_level"][i], NoneType)
                                ):
                                    category_level = self.cat_["category_level"][
                                        i
                                    ].lower()
                                else:
                                    category_level = self.parameters["null_column_name"]
                                name = f'"{quote_ident(column)[1:-1]}{self.parameters["separator"]}'
                                name += f'{category_level}"'
                                names += [name]
                        k += 1
            return names
        else:
            return X

    def deploySQL(
        self,
        X: Optional[SQLColumns] = None,
        key_columns: Optional[SQLColumns] = None,
        exclude_columns: Optional[SQLColumns] = None,
    ) -> str:
        """
        Returns the SQL code needed to deploy the model.

        Parameters
        ----------
        X: SQLColumns, optional
            List of the columns used to deploy the model.
            If empty,  the model predictors are used.
        key_columns: SQLColumns, optional
            Predictors   used   during   the   algorithm
            computation which will  be deployed with the
            principal components.
        exclude_columns: SQLColumns, optional
            Columns to exclude from the prediction.

        Returns
        -------
        str
            the SQL code needed to deploy the model.
        """
        key_columns, exclude_columns = format_type(
            key_columns, exclude_columns, dtype=list
        )
        X = format_type(X, dtype=list, na_out=self.X)
        X = quote_ident(X)
        if key_columns:
            key_columns = ", ".join(quote_ident(key_columns))
        if exclude_columns:
            exclude_columns = ", ".join(quote_ident(exclude_columns))
        sql = f"""
            {self._vertica_transform_sql}({', '.join(X)} 
               USING PARAMETERS 
               model_name = '{self.model_name}',
               match_by_pos = 'true'"""
        if key_columns:
            sql += f", key_columns = '{key_columns}'"
        if exclude_columns:
            sql += f", exclude_columns = '{exclude_columns}'"
        if self._model_type == "OneHotEncoder":
            if isinstance(self.parameters["separator"], NoneType):
                separator = "null"
            else:
                separator = self.parameters["separator"].lower()
            sql += f""", 
                drop_first = '{str(self.parameters['drop_first']).lower()}',
                ignore_null = '{str(self.parameters['ignore_null']).lower()}',
                separator = '{separator}',
                column_naming = '{self.parameters['column_naming']}'"""
            if self.parameters["column_naming"].lower() in (
                "values",
                "values_relaxed",
            ):
                if isinstance(self.parameters["null_column_name"], NoneType):
                    null_column_name = "null"
                else:
                    null_column_name = self.parameters["null_column_name"].lower()
                sql += f", null_column_name = '{null_column_name}'"
        sql += ")"
        return clean_query(sql)

    def deployInverseSQL(
        self,
        key_columns: Optional[SQLColumns] = None,
        exclude_columns: Optional[SQLColumns] = None,
        X: Optional[SQLColumns] = None,
    ) -> str:
        """
        Returns  the SQL code needed to deploy the  inverse
        model.

        Parameters
        ----------
        key_columns: SQLColumns, optional
            Predictors used during the algorithm computation
            which  will  be  deployed   with  the  principal
            components.
        exclude_columns: SQLColumns, optional
            Columns to exclude from the prediction.
        X: SQLColumns, optional
            List  of the columns used to deploy the  inverse
            model. If empty, the model predictors are used.

        Returns
        -------
        str
            the SQL code needed to deploy the inverse model.
        """
        if isinstance(X, NoneType):
            X = self.X
        else:
            X = quote_ident(X)
        X, key_columns, exclude_columns = format_type(
            X, key_columns, exclude_columns, dtype=list
        )
        if self._model_type == "OneHotEncoder":
            raise AttributeError(
                "method 'deployInverseSQL' is not supported for OneHotEncoder models."
            )
        sql = f"""
            {self._vertica_inverse_transform_sql}({', '.join(X)} 
                                                          USING PARAMETERS 
                                                          model_name = '{self.model_name}',
                                                          match_by_pos = 'true'"""
        if key_columns:
            key_columns = ", ".join([quote_ident(kcol) for kcol in key_columns])
            sql += f", key_columns = '{key_columns}'"
        if exclude_columns:
            exclude_columns = ", ".join([quote_ident(ecol) for ecol in exclude_columns])
            sql += f", exclude_columns = '{exclude_columns}'"
        sql += ")"
        return clean_query(sql)

    # Prediction / Transformation Methods.

    def transform(
        self, vdf: SQLRelation = None, X: Optional[SQLColumns] = None
    ) -> vDataFrame:
        """
        Applies the model on a vDataFrame.

        Parameters
        ----------
        vdf: SQLRelation, optional
            Input vDataFrame. You can also specify a customized
            relation,  but you must  enclose it with an  alias.
            For  example:  "(SELECT 1) x"  is  valid  whereas
            "(SELECT 1)" and "SELECT 1" are invalid.
        X: SQLColumns, optional
            List of the input vDataColumns.

        Returns
        -------
        vDataFrame
            object result of the model transformation.
        """
        if isinstance(X, NoneType):
            X = self.X
        X = format_type(X, dtype=list)
        if not vdf:
            vdf = self.input_relation
        if isinstance(vdf, str):
            vdf = vDataFrame(vdf)
        X = vdf.format_colnames(X)
        exclude_columns = vdf.get_columns(exclude_columns=X)
        all_columns = vdf.get_columns()
        columns = self.deploySQL(all_columns, exclude_columns, exclude_columns)
        main_relation = f"(SELECT {columns} FROM {vdf}) VERTICAPY_SUBTABLE"
        return vDataFrame(main_relation)

    def inverse_transform(
        self, vdf: SQLRelation, X: Optional[SQLColumns] = None
    ) -> vDataFrame:
        """
        Applies the Inverse Model on a vDataFrame.

        Parameters
        ----------
        vdf: SQLRelation
            Input vDataFrame. You can also specify a customized
            relation,  but you must  enclose it with an  alias.
            For  example:  "(SELECT 1) x"  is  valid  whereas
            "(SELECT 1)" and "SELECT 1" are invalid.
        X: SQLColumns, optional
            List of the input vDataColumns.

        Returns
        -------
        vDataFrame
            object result of the model transformation.
        """
        X = format_type(X, dtype=list)
        if self._model_type == "OneHotEncoder":
            raise AttributeError(
                "method 'inverse_transform' is not supported for OneHotEncoder models."
            )
        if not vdf:
            vdf = self.input_relation
        if not X:
            X = self._get_names()
        if isinstance(vdf, str):
            vdf = vDataFrame(vdf)
        X = vdf.format_colnames(X)
        exclude_columns = vdf.get_columns(exclude_columns=X)
        all_columns = vdf.get_columns()
        inverse_sql = self.deployInverseSQL(
            exclude_columns, exclude_columns, all_columns
        )
        main_relation = f"(SELECT {inverse_sql} FROM {vdf}) VERTICAPY_SUBTABLE"
        return vDataFrame(main_relation)


"""
Algorithms used for text analytics.
"""


class CountVectorizer(VerticaModel):
    """
    Creates a Text Index that counts the occurences
    of each word in the data.

    Parameters
    ----------
    name: str, optional
        Name of the model.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    lowercase: bool, optional
        Converts  all  the elements to lowercase  before
        processing.
    max_df: float, optional
        Keeps  the words that represent less than  this
        float in the total dictionary distribution.
    min_df: float, optional
        Keeps  the words that represent more than  this
        float in the total dictionary distribution.
    max_features: int, optional
        Keeps only the top words of the dictionary.
    ignore_special: bool, optional
        Ignores all the special characters when building
        the dictionary.
    max_text_size: int, optional
        The maximum size of the column that concatenates
        all of the  text columns during fitting.
    """

    # Properties.

    @property
    def _is_native(self) -> Literal[False]:
        return False

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

    @property
    def _attributes(self) -> list[str]:
        return ["stop_words_", "vocabulary_", "n_errors_"]

    # System & Special Methods.

    @save_verticapy_logs
    def __init__(
        self,
        name: str = None,
        overwrite_model: bool = False,
        lowercase: bool = True,
        max_df: float = 1.0,
        min_df: float = 0.0,
        max_features: int = -1,
        ignore_special: bool = True,
        max_text_size: int = 2000,
    ) -> None:
        super().__init__(name, overwrite_model)
        self.parameters = {
            "lowercase": lowercase,
            "max_df": max_df,
            "min_df": min_df,
            "max_features": max_features,
            "ignore_special": ignore_special,
            "max_text_size": max_text_size,
        }

    def drop(self) -> bool:
        """
        Drops the model from the Vertica database.
        """
        return drop(self.model_name, method="text")

    # Attributes Methods.

    def _compute_attributes(self) -> None:
        """
        Computes the model's attributes.
        """
        self.stop_words_ = self._compute_stop_words()
        self.vocabulary_ = self._compute_vocabulary()

    def _compute_stop_words(self) -> np.ndarray:
        """
        Computes the CountVectorizer Stop Words. It will
        affect the result to  the stop_words_ attribute.
        """
        query = self.deploySQL(_return_main_table=True)
        query = query.format(
            "/*+LABEL('learn.preprocessing.CountVectorizer.compute_stop_words')*/ token",
            "not",
        )
        if self.parameters["max_features"] > 0:
            query += f" OR (rnk > {self.parameters['max_features']})"
        res = _executeSQL(query=query, print_time_sql=False, method="fetchall")
        return np.array([w[0] for w in res])

    def _compute_vocabulary(self) -> np.ndarray:
        """
        Computes the CountVectorizer Vocabulary. It will
        affect the result to  the vocabulary_ attribute.
        """
        res = _executeSQL(self.deploySQL(), print_time_sql=False, method="fetchall")
        return np.array([w[0] for w in res])

    # I/O Methods.

    def deploySQL(self, _return_main_table: bool = False) -> str:
        """
        Returns the SQL code needed to deploy the model.

        Returns
        -------
        SQLExpression
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

        if _return_main_table:
            return query

        if self.parameters["max_features"] > 0:
            query += f" AND (rnk <= {self.parameters['max_features']})"

        return clean_query(query.format("*", ""))

    # Model Fitting Method.

    def fit(self, input_relation: SQLRelation, X: Optional[SQLColumns] = None) -> None:
        """
        Trains the model.

        Parameters
        ----------
        input_relation: SQLRelation
                Training relation.
        X: SQLColumns
                List of the predictors. If empty, all the
            columns are used.
        """
        X = format_type(X, dtype=list)
        if self.overwrite_model:
            self.drop()
        else:
            self._is_already_stored(raise_error=True)
        if isinstance(input_relation, vDataFrame):
            if isinstance(X, NoneType):
                X = input_relation.get_columns()
            self.input_relation = input_relation.current_relation()
        else:
            if isinstance(X, NoneType):
                X = vDataFrame(input_relation).get_columns()
            self.input_relation = input_relation
        self.X = quote_ident(X)
        schema = schema_relation(self.model_name)[0]
        schema = quote_ident(schema)
        tmp_name = gen_tmp_name(schema=schema, name="countvectorizer")
        _executeSQL(
            query=f"""
                CREATE TABLE {tmp_name}
                (id identity(2000) primary key, 
                 text varchar({self.parameters['max_text_size']})) 
                ORDER BY id SEGMENTED BY HASH(id) ALL NODES KSAFE;""",
            title="Computing the CountVectorizer [Step 0].",
        )
        if not self.parameters["lowercase"]:
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
        self._compute_attributes()

    # Prediction / Transformation Methods.

    def transform(self) -> vDataFrame:
        """
        Creates a vDataFrame of the model.

        Returns
        -------
        vDataFrame
                object result of the model transformation.
        """
        return vDataFrame(self.deploySQL())


"""
Algorithms used for scaling.
"""


class Scaler(Preprocessing):
    """
    Creates a Vertica Scaler object.

    Parameters
    ----------
    name: str, optional
        Name of the model.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    method: str, optional
        Method used to normalize.
                zscore        : Normalization   using   the   Z-Score.
                            (x - avg) / std
                robust_zscore : Normalization using the Robust Z-Score.
                                (x - median) / (1.4826 * mad)
                minmax        : Normalization  using  the  Min  &  Max.
                                (x - min) / (max - min)
    """

    # Properties.

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
    def _model_subcategory(self) -> Literal["PREPROCESSING"]:
        return "PREPROCESSING"

    @property
    def _model_type(self) -> Literal["Scaler"]:
        return "Scaler"

    @property
    def _attributes(self) -> list[str]:
        if self.parameters["method"] == "minmax":
            return ["min_", "max_"]
        elif self.parameters["method"] == "robust_zscore":
            return ["median_", "mad_"]
        else:
            return ["mean_", "std_"]

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str = None,
        overwrite_model: bool = False,
        method: Literal["zscore", "robust_zscore", "minmax"] = "zscore",
    ) -> None:
        super().__init__(name, overwrite_model)
        self.parameters = {"method": str(method).lower()}

    # Attributes Methods.

    def _compute_attributes(self) -> None:
        """
        Computes the model's attributes.
        """
        values = self.get_vertica_attributes("details").to_numpy()[:, 1:].astype(float)
        if self.parameters["method"] == "minmax":
            self.min_ = values[:, 0]
            self.max_ = values[:, 1]
        elif self.parameters["method"] == "robust_zscore":
            self.median_ = values[:, 0]
            self.mad_ = values[:, 1]
        else:
            self.mean_ = values[:, 0]
            self.std_ = values[:, 1]

    # I/O Methods.

    def to_memmodel(self) -> mm.Scaler:
        """
        Converts the model to an InMemory object that can
        be used for different types of predictions.
        """
        if self.parameters["method"] == "minmax":
            return mm.MinMaxScaler(self.min_, self.max_)
        elif self.parameters["method"] == "robust_zscore":
            return mm.StandardScaler(self.median_, self.mad_)
        else:
            return mm.StandardScaler(self.mean_, self.std_)


class StandardScaler(Scaler):
    """i.e. Scaler with param method = 'zscore'"""

    @property
    def _attributes(self) -> list[str]:
        return ["mean_", "std_"]

    def __init__(self, name: str = None, overwrite_model: bool = False) -> None:
        super().__init__(name, overwrite_model, "zscore")


class RobustScaler(Scaler):
    """i.e. Scaler with param method = 'robust_zscore'"""

    @property
    def _attributes(self) -> list[str]:
        return ["median_", "mad_"]

    def __init__(self, name: str = None, overwrite_model: bool = False) -> None:
        super().__init__(name, overwrite_model, "robust_zscore")


class MinMaxScaler(Scaler):
    """i.e. Scaler with param method = 'minmax'"""

    @property
    def _attributes(self) -> list[str]:
        return ["min_", "max_"]

    def __init__(self, name: str = None, overwrite_model: bool = False) -> None:
        super().__init__(name, overwrite_model, "minmax")


"""
Algorithms used for encoding.
"""


class OneHotEncoder(Preprocessing):
    """
    Creates a Vertica OneHotEncoder object.

    Parameters
    ----------
    name: str, optional
        Name of the model.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    extra_levels: dict, optional
        Additional levels in each  category that are not
        in the input relation.
    drop_first: bool, optional
        If set to True,  treats the first level of the
        categorical variable as the reference level.
        Otherwise, every level of the categorical variable
        has a corresponding column in the output view.
    ignore_null: bool, optional
        If set to True, Null values set all corresponding
        one-hot binary columns to null.  Otherwise,  null
        values in the input columns are treated as a
        categorical level.
    separator: str, optional
        The  character that separates the input  variable
        name  and  the indicator variable  level  in  the
        output table.  To avoid using any separator,  set
        this parameter to null value.
    column_naming: str, optional
        Appends   categorical  levels  to  column   names
        according to the specified method:
            indices : Uses  integer indices to  represent
                      categorical levels.
            values  : Uses  categorical  level names.  If
                      duplicate  column names occur,  the
                      function attempts  to  disambiguate
                      them  by appending _n,  where n  is
                      a zero-based integer index (_0, _1,
                      ..., _n).
    null_column_name: str, optional
        The  string used in  naming the indicator  column
        for null values,  used only if ignore_null is set
        to false and column_naming is set to 'values'.
    """

    # Properties.

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
    def _model_subcategory(self) -> Literal["PREPROCESSING"]:
        return "PREPROCESSING"

    @property
    def _model_type(self) -> Literal["OneHotEncoder"]:
        return "OneHotEncoder"

    @property
    def _attributes(self) -> list[str]:
        return ["categories_", "column_naming_", "drop_first_"]

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str = None,
        overwrite_model: bool = False,
        extra_levels: Optional[dict] = None,
        drop_first: bool = True,
        ignore_null: bool = True,
        separator: str = "_",
        column_naming: Literal["indices", "values", "values_relaxed"] = "indices",
        null_column_name: str = "null",
    ) -> None:
        super().__init__(name, overwrite_model)
        self.parameters = {
            "extra_levels": format_type(extra_levels, dtype=dict),
            "drop_first": drop_first,
            "ignore_null": ignore_null,
            "separator": separator,
            "column_naming": str(column_naming).lower(),
            "null_column_name": null_column_name,
        }

    # Attributes Methods.

    @staticmethod
    def _compute_ohe_list(categories: list) -> list:
        """
        Allows to split the One Hot Encoder Array by
        features categories.
        """
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

    def _compute_attributes(self) -> None:
        """
        Computes the model's attributes.
        """
        query = f"""
            SELECT 
                category_name, 
                category_level::varchar, 
                category_level_index 
            FROM (SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS 
                            model_name = '{self.model_name}', 
                            attr_name = 'integer_categories')) 
                            VERTICAPY_SUBTABLE"""
        try:
            self.cat_ = TableSample.read_sql(
                query=f"""
                    {query}
                    UNION ALL 
                    SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS 
                            model_name = '{self.model_name}', 
                            attr_name = 'varchar_categories')""",
                title="Getting Model Attributes.",
            )
        except QueryError:
            try:
                self.cat_ = TableSample.read_sql(
                    query=query,
                    title="Getting Model Attributes.",
                )
            except QueryError:
                self.cat_ = self.get_vertica_attributes("varchar_categories")
        self.cat_ = self.cat_.to_list()
        cat = self._compute_ohe_list([c[0:2] for c in self.cat_])
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

    # I/O Methods.

    def to_memmodel(self) -> mm.OneHotEncoder:
        """
        Converts the model to an InMemory object that
        can be used for different types of predictions.
        """
        return mm.OneHotEncoder(self.categories_, self.column_naming_, self.drop_first_)
