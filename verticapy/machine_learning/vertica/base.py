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
import copy, warnings
from abc import abstractmethod
from typing import Any, Callable, Literal, Optional, Union, get_type_hints
from collections.abc import Iterable
import numpy as np

import verticapy._config.config as conf
from verticapy._typing import (
    ArrayLike,
    PlottingObject,
    PythonNumber,
    PythonScalar,
    SQLColumns,
    SQLRelation,
    SQLExpression,
)
from verticapy._utils._gen import gen_name, gen_tmp_name
from verticapy._utils._sql._format import clean_query, quote_ident, schema_relation
from verticapy._utils._sql._sys import _executeSQL
from verticapy._utils._sql._vertica_version import (
    check_minimum_version,
    vertica_version,
)
from verticapy.errors import (
    ConversionError,
    FunctionError,
    ParameterError,
    ModelError,
    VersionError,
)

from verticapy.core.tablesample.base import TableSample
from verticapy.core.vdataframe.base import vDataFrame

import verticapy.machine_learning.metrics as mt

from verticapy.plotting._utils import PlottingUtils

from verticapy.sql.drop import drop

if conf._get_import_success("graphviz"):
    from graphviz import Source

##
#  ___      ___  ___      ___     ______    ________    _______  ___
# |"  \    /"  ||"  \    /"  |   /    " \  |"      "\  /"     "||"  |
#  \   \  //  /  \   \  //   |  // ____  \ (.  ___  :)(: ______)||  |
#   \\  \/. ./   /\\  \/.    | /  /    ) :)|: \   ) || \/    |  |:  |
#    \.    //   |: \.        |(: (____/ // (| (___\ || // ___)_  \  |___
#     \\   /    |.  \    /:  | \        /  |:       :)(:      "|( \_|:  \
#      \__/     |___|\__/|___|  \"_____/   (________/  \_______) \_______)
#
##


class VerticaModel(PlottingUtils):
    """
    Base Class for Vertica Models.
	"""

    # Properties.

    @property
    def _is_native(self) -> Literal[True]:
        return True

    @property
    def _is_using_native(self) -> Literal[False]:
        return False

    @property
    def _object_type(self) -> Literal["VerticaModel"]:
        return "VerticaModel"

    @property
    @abstractmethod
    def _vertica_fit_sql(self) -> str:
        """Must be overridden in child class"""
        raise NotImplementedError

    @property
    @abstractmethod
    def _model_category(self) -> str:
        """Must be overridden in child class"""
        raise NotImplementedError

    @property
    @abstractmethod
    def _model_subcategory(self) -> str:
        """Must be overridden in child class"""
        raise NotImplementedError

    @property
    @abstractmethod
    def _model_type(self) -> str:
        """Must be overridden in child class"""
        raise NotImplementedError

    # Formatting Methods.

    @staticmethod
    def _array_to_int(object_: np.ndarray) -> np.ndarray:
        """
        Converts the input numpy.array values to integer
        if it is possible.
        """
        res = copy.deepcopy(object_)
        try:
            return res.astype(int)
        except ValueError:
            return res

    @staticmethod
    def _format_vector(X: ArrayLike, vector: list[tuple]) -> np.ndarray:
        """
        Format the 2D vector to match with the input columns'
        names.
        """
        res = []
        for x in X:
            for y in vector:
                if quote_ident(y[0]).lower() == x.lower():
                    res += [y[1]]
        return np.array(res)

    @staticmethod
    def _get_match_index(x: str, col_list: list, str_check: bool = True) -> None:
        """
        Returns the matching index.
        """
        for idx, col in enumerate(col_list):
            if (str_check and quote_ident(x.lower()) == quote_ident(col.lower())) or (
                x == col
            ):
                return idx
        return None

    # System & Special Methods.

    def __repr__(self) -> str:
        """
        Returns the model Representation.
        """
        return f"<{self._model_type}>"

    def drop(self) -> bool:
        """
        Drops the model from the Vertica database.
        """
        return drop(self.model_name, method="model")

    def _is_already_stored(
        self, raise_error: bool = False, return_model_type: bool = False,
    ) -> Union[bool, str]:
        """
        Checks if the model is stored in the Vertica DB.

        Parameters
        ----------
        raise_error: bool, optional
            If set to True and an error occurs, it 
            raises the error.
        return_model_type: bool, optional
            If set to True, returns the model type.

        Returns
        -------
        bool
            True if the model is stored in the
            Vertica DB.
        """
        return self.does_model_exists(
            name=self.model_name,
            raise_error=raise_error,
            return_model_type=return_model_type,
        )

    @staticmethod
    def does_model_exists(
        name: str, raise_error: bool = False, return_model_type: bool = False,
    ) -> Union[bool, str]:
        """
        Checks if the model is stored in the Vertica DB.

        Parameters
        ----------
        name: str, optional
            Model's name.
        raise_error: bool, optional
            If set to True and an error occurs, it 
            raises the error.
        return_model_type: bool, optional
            If set to True, returns the model type.

        Returns
        -------
        bool
            True if the model is stored in the
            Vertica DB.
        """
        model_type = None
        if name == None:
            name = self.model_name
        schema, model_name = schema_relation(name)
        schema, model_name = schema[1:-1], model_name[1:-1]
        res = _executeSQL(
            query=f"""
                SELECT 
                    /*+LABEL('learn.tools._is_already_stored')*/ 
                    model_type 
                FROM MODELS 
                WHERE LOWER(model_name) = LOWER('{model_name}') 
                  AND LOWER(schema_name) = LOWER('{schema}') 
                LIMIT 1""",
            method="fetchrow",
            print_time_sql=False,
        )
        if res:
            model_type = res[0]
            res = True
        else:
            res = False
        if raise_error and res:
            raise NameError(f"The model '{model_name}' already exists !")
        if return_model_type:
            return model_type
        return res

    # Attributes Methods.

    def get_attributes(self, attr_name: str = "") -> Any:
        """
        Returns the model attributes.

        Parameters
        ----------
        attr_name: str, optional
            Attribute Name.

        Returns
        -------
        Any
            model attribute.
        """
        if not (attr_name):
            return self._attributes
        elif attr_name in self._attributes:
            if hasattr(self, attr_name):
                return copy.deepcopy(getattr(self, attr_name))
            else:
                return AttributeError("The attribute is not yet computed.")
        elif attr_name + "_" in self._attributes:
            return self.get_attributes(attr_name + "_")
        else:
            raise AttributeError(
                "Method 'get_vertica_attributes' is not available for "
                "non-native models.\nUse 'get_attributes' method instead."
            )

    def get_vertica_attributes(self, attr_name: str = "") -> TableSample:
        """
        Returns the model vertica attributes. Those are stored
        in Vertica.

        Parameters
        ----------
        attr_name: str, optional
            Attribute Name.

        Returns
        -------
        TableSample
            model attributes.
        """
        if self._is_native or self._is_using_native:
            vertica_version(condition=[8, 1, 1])
            if attr_name:
                attr_name_str = f", attr_name = '{attr_name}'"
            else:
                attr_name_str = ""
            return TableSample.read_sql(
                query=f"""
                    SELECT 
                        GET_MODEL_ATTRIBUTE(
                            USING PARAMETERS 
                            model_name = '{self.model_name}'{attr_name_str})""",
                title="Getting Model Attributes.",
            )
        else:
            raise AttributeError(
                "Method 'get_vertica_attributes' is not available for "
                "non-native models.\nUse 'get_attributes' method instead."
            )

    def _is_binary_classifier(self) -> Literal[False]:
        """
        Returns True if the model is a Binary Classifier.
        """
        return False

    # Parameters Methods.

    @staticmethod
    def _map_to_vertica_param_dict() -> dict[str, str]:
        """
        Returns a dictionary used to map VerticaPy parameters 
        names to the Vertica ones.
        """
        return {
            "class_weights": "class_weight",
            "solver": "optimizer",
            "tol": "epsilon",
            "max_iter": "max_iterations",
            "penalty": "regularization",
            "c": "lambda",
            "l1_ratio": "alpha",
            "n_estimators": "ntree",
            "max_features": "mtry",
            "sample": "sampling_size",
            "max_leaf_nodes": "max_breadth",
            "min_samples_leaf": "min_leaf_size",
            "n_components": "num_components",
            "init": "init_method",
        }

    def _map_to_vertica_param_name(self, param: str) -> str:
        """
        Maps the input VerticaPy parameter name to the 
        Vertica one.
        """
        options = self._map_to_vertica_param_dict()
        param = param.lower()
        if param in options:
            return options[param]
        return param

    def _get_vertica_param_dict(self) -> dict[str, str]:
        """
        Returns the Vertica parameters dict to use when fitting 
        the model. As some model's parameters names are not the 
        same in Vertica. It is important to map them.

        Returns
        -------
        dict
            Vertica parameters.
        """
        parameters = {}

        for param in self.parameters:

            if param == "class_weight":
                if isinstance(self.parameters[param], (list, np.ndarray)):
                    parameters[
                        "class_weights"
                    ] = f"'{', '.join([str(p) for p in self.parameters[param]])}'"
                else:
                    parameters["class_weights"] = f"'{self.parameters[param]}'"

            elif isinstance(self.parameters[param], (str, dict)):
                parameters[
                    self._map_to_vertica_param_name(param)
                ] = f"'{self.parameters[param]}'"

            else:
                parameters[self._map_to_vertica_param_name(param)] = self.parameters[
                    param
                ]

        return parameters

    def _map_to_verticapy_param_name(self, param: str) -> str:
        """
        Maps the Vertica parameter name to the VerticaPy one.
        """
        options = self._map_to_vertica_param_dict()
        for key in options:
            if options[key] == param:
                return key
        return param

    def _get_verticapy_param_dict(self, options: dict = {}, **kwargs) -> dict:
        """
        Takes as input a dictionary of Vertica options and 
        returns  the  associated  dictionary of  VerticaPy
        options.
        """
        parameters = {}
        map_dict = {**options, **kwargs}
        for param in map_dict:
            parameters[self._map_to_verticapy_param_name(param)] = map_dict[param]
        return parameters

    def get_params(self) -> dict:
        """
        Returns the parameters of the model.

        Returns
        -------
        dict
            model parameters.
        """
        all_init_params = list(get_type_hints(self.__init__).keys())
        parameters = copy.deepcopy(self.parameters)
        parameters_keys = list(parameters.keys())
        for p in parameters_keys:
            if p not in all_init_params:
                del parameters[p]
        return parameters

    def set_params(self, parameters: dict = {}, **kwargs) -> None:
        """
        Sets the parameters of the model.

        Parameters
        ----------
        parameters: dict, optional
            New parameters.
        **kwargs
            New  parameters can  also be passed as  arguments
            Example: set_params(param1 = val1, param2 = val2)
        """
        all_init_params = list(get_type_hints(self.__init__).keys())
        new_parameters = copy.deepcopy({**self.parameters, **kwargs})
        new_parameters_keys = list(new_parameters.keys())
        for p in new_parameters_keys:
            if p not in all_init_params:
                del new_parameters[p]
        for p in parameters:
            if p not in all_init_params:
                warning_message = (
                    f"parameter 'parameters' got an unexpected keyword argument '{p}'"
                )
                warnings.warn(warning_message, Warning)
            new_parameters[p] = parameters[p]
        self.__init__(name=self.model_name, **new_parameters)
        return None

    # Model's Summary.

    def summarize(self) -> str:
        """
        Summarizes the model.
        """
        if self._is_native:
            try:
                vertica_version(condition=[9, 0, 0])
                func = f"""
                    GET_MODEL_SUMMARY(USING PARAMETERS 
                    model_name = '{self.model_name}')"""
            except VersionError:
                func = f"SUMMARIZE_MODEL('{self.model_name}')"
            return _executeSQL(
                f"SELECT /*+LABEL('learn.VerticaModel.__repr__')*/ {func}",
                title="Summarizing the model.",
                method="fetchfirstelem",
            )
        else:
            raise AttributeError(
                "Method 'summarize' is not available for non-native models."
            )

    # I/O Methods.

    def deploySQL(self, X: Optional[SQLColumns] = None) -> str:
        """
        Returns the SQL code needed to deploy the model. 

        Parameters
        ----------
        X: SQLColumns, optional
            List of the columns used to deploy the model. 
            If empty,  the model predictors will be used.

        Returns
        -------
        str
            the SQL code needed to deploy the model.
        """
        if self._vertica_predict_sql:
            if isinstance(X, str):
                X = [X]
            X = (
                self.X
                if isinstance(X, type(None))
                else [quote_ident(predictor) for predictor in X]
            )
            sql = f"""
                {self._vertica_predict_sql}({', '.join(X)} 
                                            USING PARAMETERS 
                                            model_name = '{self.model_name}',
                                            match_by_pos = 'true')"""
            return clean_query(sql)
        else:
            raise AttributeError(
                f"Method 'deploySQL' does not exist for {self._model_type} models."
            )

    def to_python(
        self, return_proba: bool = False, return_distance_clusters: bool = False,
    ) -> Callable:
        """
        Returns the Python function needed to do in-memory 
        scoring  without using built-in Vertica functions.

        Parameters
        ----------
        return_proba: bool, optional
            If  set to True  and  the  model is a  classifier,
            the  function  returns  the  model  probabilities.
        return_distance_clusters: bool, optional
            If  set to  True and the  model is  cluster-based, 
            the function returns the model clusters distances. 
            If the model is KPrototypes, the  function returns 
            the dissimilarity function.

        Returns
        -------
        Callable
            Python function.
        """
        model = self.to_memmodel()
        if return_proba:
            return model.predict_proba
        elif hasattr(model, "predict") and not (return_distance_clusters):
            return model.predict
        else:
            return model.transform

    def to_sql(
        self,
        X: list = [],
        return_proba: bool = False,
        return_distance_clusters: bool = False,
    ) -> SQLExpression:
        """
        Returns  the SQL  code  needed  to deploy the  model 
        without using built-in Vertica functions.

        Parameters
        ----------
        X: list, optional
            input predictors name.
        return_proba: bool, optional
            If  set to  True and  the  model is a  classifier,
            the function will return  the class probabilities.
        return_distance_clusters: bool, optional
            If  set to  True and the  model is  cluster-based, 
            the function returns the model clusters distances. 
            If the model is  KPrototypes, the function returns 
            the dissimilarity function.

        Returns
        -------
        SQLExpression
            SQL code.
        """
        if not X:
            X = self.X
        model = self.to_memmodel()
        if return_proba:
            return model.predict_proba_sql(X)
        elif hasattr(model, "predict") and not (return_distance_clusters):
            return model.predict_sql(X)
        else:
            return model.transform_sql(X)

    # Plotting Methods.

    def _get_plot_args(self, method: Optional[str] = None) -> list:
        """
        Returns the args used by plotting methods.
        """
        if method == "contour":
            args = [self.X, self.deploySQL(X=self.X)]
        else:
            raise NotImplementedError
        return args

    def _get_plot_kwargs(
        self,
        nbins: int = 30,
        chart: Optional[PlottingObject] = None,
        method: Optional[str] = None,
    ) -> dict:
        """
        Returns the kwargs used by plotting methods.
        """
        res = {"nbins": nbins, "chart": chart}
        if method == "contour":
            if self._model_subcategory == "CLASSIFIER":
                res["func_name"] = f"p({self.y} = 1)"
            else:
                res["func_name"] = self.y
        else:
            raise NotImplementedError
        return res

    def contour(
        self, nbins: int = 100, chart: Optional[PlottingObject] = None, **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the model's contour plot.

        Parameters
        ----------
        nbins: int, optional
            Number of bins used to discretize the 
            two predictors.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any optional parameter to pass to the 
            Plotting functions.

        Returns
        -------
        obj
            Plotting Object.
        """
        return vDataFrame(self.input_relation).contour(
            *self._get_plot_args(method="contour"),
            **self._get_plot_kwargs(nbins=nbins, chart=chart, method="contour"),
            **style_kwargs,
        )


class Supervised(VerticaModel):
    @property
    @abstractmethod
    def _vertica_predict_sql(self) -> str:
        """Must be overridden in child class"""
        raise NotImplementedError

    # Model Fitting Method.

    def fit(
        self,
        input_relation: SQLRelation,
        X: SQLColumns,
        y: str,
        test_relation: SQLRelation = "",
    ) -> Optional[str]:
        """
    	Trains the model.

    	Parameters
    	----------
    	input_relation: SQLRelation
    		Training relation.
    	X: SQLColumns
    		List of the predictors.
    	y: str
    		Response column.
    	test_relation: SQLRelation, optional
    		Relation used to test the model.

        Returns
        -------
        str
            model's summary.
		"""
        if isinstance(X, str):
            X = [X]
        if conf.get_option("overwrite_model"):
            self.drop()
        else:
            self._is_already_stored(raise_error=True)
        self.X = [quote_ident(column) for column in X]
        self.y = quote_ident(y)
        id_column, id_column_name = "", gen_tmp_name(name="id_column")
        if self._is_native:
            nb_lookup_table = {
                "bernoulli": "bool",
                "categorical": "varchar",
                "multinomial": "int",
                "gaussian": "float",
            }
            if (self._model_type == "NaiveBayes") and (
                self.parameters["nbtype"] in nb_lookup_table
            ):
                new_types = {}
                for x in X:
                    new_types[x] = nb_lookup_table[self.parameters["nbtype"]]
                if not (isinstance(input_relation, vDataFrame)):
                    input_relation = vDataFrame(input_relation)
                else:
                    input_relation.copy()
                input_relation.astype(new_types)
            if self._model_type in (
                "RandomForestClassifier",
                "RandomForestRegressor",
                "XGBClassifier",
                "XGBRegressor",
            ) and isinstance(conf.get_option("random_state"), int):
                id_column = f""", 
                    ROW_NUMBER() OVER 
                    (ORDER BY {', '.join(X)}) 
                    AS {id_column_name}"""
            tmp_view = False
        if isinstance(input_relation, vDataFrame) or (id_column):
            tmp_view = True
            if isinstance(input_relation, vDataFrame):
                self.input_relation = input_relation._genSQL()
            else:
                self.input_relation = input_relation
            if self._is_native:
                relation = gen_tmp_name(
                    schema=schema_relation(self.model_name)[0], name="view"
                )
                drop(relation, method="view")
                _executeSQL(
                    query=f"""
                        CREATE VIEW {relation} AS 
                            SELECT 
                                /*+LABEL('learn.VerticaModel.fit')*/ 
                                *{id_column} 
                            FROM {self.input_relation}""",
                    title="Creating a temporary view to fit the model.",
                )
        else:
            self.input_relation = input_relation
            relation = input_relation
        if isinstance(test_relation, vDataFrame):
            self.test_relation = test_relation._genSQL()
        elif test_relation:
            self.test_relation = test_relation
        else:
            self.test_relation = self.input_relation
        if self._is_native:
            parameters = self._get_vertica_param_dict()
            if (
                "regularization" in parameters
                and parameters["regularization"].lower() == "'enet'"
            ):
                alpha = parameters["alpha"]
                del parameters["alpha"]
            else:
                alpha = None
            if "mtry" in parameters:
                if parameters["mtry"] == "'auto'":
                    parameters["mtry"] = int(len(self.X) / 3 + 1)
                elif parameters["mtry"] == "'max'":
                    parameters["mtry"] = len(self.X)
            for param in ("nbtype",):
                if param in parameters:
                    del parameters[param]
            fun = self._vertica_fit_sql
            query = f"""
                SELECT 
                    /*+LABEL('learn.VerticaModel.fit')*/ 
                    {self._vertica_fit_sql}
                    ('{self.model_name}', 
                     '{relation}',
                     '{self.y}',
                     '{', '.join(self.X)}' 
                     USING PARAMETERS 
                     {', '.join([f"{p} = {parameters[p]}" for p in parameters])}"""
            if alpha != None:
                query += f", alpha = {alpha}"
            if self._model_type in (
                "RandomForestClassifier",
                "RandomForestRegressor",
                "XGBClassifier",
                "XGBRegressor",
            ) and isinstance(conf.get_option("random_state"), int):
                query += f""", 
                    seed={conf.get_option('random_state')}, 
                    id_column='{id_column_name}'"""
            query += ")"
            try:
                _executeSQL(query, title="Fitting the model.")
            finally:
                if tmp_view:
                    drop(relation, method="view")
        self._compute_attributes()
        if self._is_native:
            return self.summarize()
        else:
            return None


class Tree:
    # System & Special Methods.

    def _compute_trees_arrays(
        self, tree: TableSample, X: list, return_probability: bool = False
    ) -> list[list]:
        """
        Takes as input a tree which is represented by a 
        TableSample.  It returns a list of arrays. Each 
        index of the arrays represents a node value.
        """
        tree_list = []
        for i in range(len(tree["tree_id"])):
            tree.values["left_child_id"] = [
                i if node_id == tree.values["node_id"][i] else node_id
                for node_id in tree.values["left_child_id"]
            ]
            tree.values["right_child_id"] = [
                i if node_id == tree.values["node_id"][i] else node_id
                for node_id in tree.values["right_child_id"]
            ]
            tree.values["node_id"][i] = i

            for j, xj in enumerate(X):
                if (
                    quote_ident(tree["split_predictor"][i]).lower()
                    == quote_ident(xj).lower()
                ):
                    tree["split_predictor"][i] = j

            if self._model_type == "XGBClassifier" and isinstance(
                tree["log_odds"][i], str
            ):
                val, all_val = tree["log_odds"][i].split(","), {}
                for v in val:
                    all_val[v.split(":")[0]] = float(v.split(":")[1])
                tree.values["log_odds"][i] = all_val
        if self._model_type == "IsolationForest":
            tree.values["prediction"], n = [], len(tree.values["leaf_path_length"])
            for i in range(n):
                if tree.values["leaf_path_length"][i] != None:
                    tree.values["prediction"] += [
                        [
                            int(float(tree.values["leaf_path_length"][i])),
                            int(float(tree.values["training_row_count"][i])),
                        ]
                    ]
                else:
                    tree.values["prediction"] += [None]
        trees_arrays = [
            tree["left_child_id"],
            tree["right_child_id"],
            tree["split_predictor"],
            tree["split_value"],
            tree["prediction"],
            tree["is_categorical_split"],
        ]
        if self._model_type == "XGBClassifier":
            trees_arrays += [tree["log_odds"]]
        if return_probability:
            trees_arrays += [tree["probability/variance"]]
        return trees_arrays

    # Features Importance Methods.

    def _compute_features_importance(self, tree_id: Optional[int] = None) -> None:
        """
        Computes the model's features importance.
        """
        vertica_version(condition=[9, 1, 1])
        tree_id_str = "" if tree_id is None else f", tree_id={tree_id}"
        query = f"""
        SELECT /*+LABEL('learn.VerticaModel.features_importance')*/
            predictor_name AS predictor, 
            SIGN({self._model_importance_feature})::int * 
            ROUND(100 * ABS({self._model_importance_feature}) / 
            SUM(ABS({self._model_importance_feature}))
            OVER (), 2)::float AS importance
        FROM 
            (SELECT {self._model_importance_function} ( 
                    USING PARAMETERS model_name = '{self.model_name}'{tree_id_str})) 
                    VERTICAPY_SUBTABLE 
        ORDER BY 2 DESC;"""
        importance = _executeSQL(
            query=query, title="Computing Features Importance.", method="fetchall"
        )
        importance = self._format_vector(self.X, importance)
        if isinstance(tree_id, int) and (0 <= tree_id < self.n_estimators_):
            if hasattr(self, "features_importance_trees_"):
                self.features_importance_trees_[tree_id] = importance
            else:
                self.features_importance_trees_ = {tree_id: importance}
        elif tree_id == None:
            self.features_importance_ = importance
        return None

    def _get_features_importance(self, tree_id: Optional[int] = None) -> np.ndarray:
        """
        Returns model's features importances.
        """
        if tree_id == None and hasattr(self, "features_importance_"):
            return copy.deepcopy(self.features_importance_)
        elif (
            isinstance(tree_id, int)
            and (0 <= tree_id < self.n_estimators_)
            and hasattr(self, "features_importance_trees_")
            and (tree_id in self.features_importance_trees_)
        ):
            return copy.deepcopy(self.features_importance_trees_[tree_id])
        else:
            self._compute_features_importance(tree_id=tree_id)
            return self._get_features_importance(tree_id=tree_id)

    def features_importance(
        self,
        tree_id: Optional[int] = None,
        show: bool = True,
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> TableSample:
        """
        Computes the model's features importance.

        Parameters
        ----------
        tree_id: int
            Tree ID.
        show: bool
            If  set to True,  draw the features  importance.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any optional parameter to pass to the Plotting 
            functions.

        Returns
        -------
        TableSample
            features importance.
        """
        fi = self._get_features_importance(tree_id=tree_id)
        if show:
            data = {
                "importance": fi,
            }
            layout = {"columns": copy.deepcopy(self.X)}
            vpy_plt, kwargs = self._get_plotting_lib(
                class_name="ImportanceBarChart", chart=chart, style_kwargs=style_kwargs,
            )
            return vpy_plt.ImportanceBarChart(data=data, layout=layout).draw(**kwargs)
        importances = {
            "index": [quote_ident(x)[1:-1].lower() for x in self.X],
            "importance": list(abs(fi)),
            "sign": list(np.sign(fi)),
        }
        return TableSample(values=importances).sort(column="importance", desc=True)

    def get_score(self, tree_id: int = None,) -> TableSample:
        """
        Returns the feature importance metrics for the input 
        tree.

        Parameters
        ----------
        tree_id: int, optional
            Unique  tree identifier, an integer in the range 
            [0, n_estimators - 1]. If tree_id is  undefined, 
            all  the trees in the model are used to  compute 
            the metrics.

        Returns
        -------
        TableSample
            model's score.
        """
        tree_id = "" if tree_id == None else f", tree_id={tree_id}"
        query = f"""
            SELECT {self._model_importance_function} 
            (USING PARAMETERS model_name = '{self.model_name}'{tree_id})"""
        return TableSample.read_sql(query=query, title="Reading Tree.")

    # Plotting Methods.

    def plot(
        self,
        max_nb_points: int = 100,
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the model.

        Parameters
        ----------
        max_nb_points: int
            Maximum  number of points to display.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any optional parameter to pass to the 
            Plotting functions.

        Returns
        -------
        obj
            Plotting Object.
        """
        if self._model_subcategory == "REGRESSOR":
            vdf = vDataFrame(self.input_relation)
            vdf["_prediction"] = self.deploySQL()
            vpy_plt, kwargs = self._get_plotting_lib(
                class_name="RegressionTreePlot", chart=chart, style_kwargs=style_kwargs,
            )
            return vpy_plt.RegressionTreePlot(
                vdf=vdf,
                columns=self.X + [self.y] + ["_prediction"],
                max_nb_points=max_nb_points,
            ).draw(**kwargs)
        else:
            raise NotImplementedError

    # Trees Representation Methods.

    @check_minimum_version
    def get_tree(self, tree_id: int = 0) -> TableSample:
        """
        Returns a table with all the input tree information.

        Parameters
        ----------
        tree_id: int, optional
            Unique tree  identifier, an integer in the range 
            [0, n_estimators - 1].

        Returns
        -------
        TableSample
            tree.
        """
        query = f"""
            SELECT * FROM (SELECT READ_TREE (
                             USING PARAMETERS 
                             model_name = '{self.model_name}', 
                             tree_id = {tree_id}, 
                             format = 'tabular')) x ORDER BY node_id;"""
        return TableSample.read_sql(query=query, title="Reading Tree.")

    def to_graphviz(
        self,
        tree_id: int = 0,
        classes_color: list = [],
        round_pred: int = 2,
        percent: bool = False,
        vertical: bool = True,
        node_style: dict = {"shape": "box", "style": "filled"},
        arrow_style: dict = {},
        leaf_style: dict = {},
    ) -> str:
        """
        Returns the code for a Graphviz tree.

        Parameters
        ----------
        tree_id: int, optional
            Unique  tree identifier,  an integer in the  range 
            [0, n_estimators - 1].
        classes_color: ArrayLike, optional
            Colors that represent the different classes.
        round_pred: int, optional
            The number of decimals to round the prediction to. 
            0 rounds to an integer.
        percent: bool, optional
            If set to True, the probabilities are returned as 
            percents.
        vertical: bool, optional
            If set to True, the function generates a vertical 
            tree.
        node_style: dict, optional
            Dictionary  of options to customize each node  of 
            the tree. For a list of options, see the Graphviz 
            API: https://graphviz.org/doc/info/attrs.html
        arrow_style: dict, optional
            Dictionary of options to customize each arrow  of 
            the tree. For a list of options, see the Graphviz 
            API: https://graphviz.org/doc/info/attrs.html
        leaf_style: dict, optional
            Dictionary  of options to customize each leaf  of 
            the tree. For a list of options, see the Graphviz 
            API: https://graphviz.org/doc/info/attrs.html

        Returns
        -------
        str
            Graphviz code.
        """
        return self.trees_[tree_id].to_graphviz(
            feature_names=self.X,
            classes_color=classes_color,
            round_pred=round_pred,
            percent=percent,
            vertical=vertical,
            node_style=node_style,
            arrow_style=arrow_style,
            leaf_style=leaf_style,
        )

    def plot_tree(
        self, tree_id: int = 0, pic_path: str = "", *args, **kwargs,
    ) -> "Source":
        """
        Draws the input tree. Requires the graphviz module.

        Parameters
        ----------
        tree_id: int, optional
            Unique tree identifier, an integer in the range 
            [0, n_estimators - 1].
        pic_path: str, optional
            Absolute  path to save  the image of the  tree.
        *args, **kwargs: Any, optional
            Arguments to pass to the 'to_graphviz'  method.

        Returns
        -------
        graphviz.Source
            graphviz object.
        """
        return self.trees_[tree_id].plot_tree(
            pic_path=pic_path, feature_names=self.X, *args, **kwargs,
        )


class Classifier(Supervised):
    pass


class BinaryClassifier(Classifier):

    # Properties.

    @property
    def classes_(self) -> np.ndarray:
        return np.array([0, 1])

    # Attributes Methods.

    def _is_binary_classifier(self) -> Literal[True]:
        """
        Returns True if the model is a Binary Classifier.
        """
        return True

    # I/O Methods.

    def deploySQL(
        self, X: Optional[SQLColumns] = None, cutoff: Optional[PythonNumber] = None
    ) -> str:
        """
    	Returns  the  SQL code  needed to deploy  the  model. 

    	Parameters
    	----------
        X: SQLColumns, optional
            List of the  columns used to deploy the model. If 
            empty, the model predictors will be used.
    	cutoff: PythonNumber, optional
    		Probability cutoff. If this number is not between 
            0 and 1,  the method will return the  probability 
            to be of class 1.

    	Returns
    	-------
    	str
    		the SQL code needed to deploy the model.
		"""
        if isinstance(X, type(None)):
            X = self.X
        elif isinstance(X, str):
            X = [X]
        else:
            X = [quote_ident(elem) for elem in X]
        sql = f"""
        {self._vertica_predict_sql}({', '.join(X)} 
            USING PARAMETERS
            model_name = '{self.model_name}',
            type = 'probability',
            match_by_pos = 'true')"""
        if not (isinstance(cutoff, type(None))) and (0 <= cutoff <= 1):
            sql = f"""
                (CASE 
                    WHEN {sql} >= {cutoff} 
                        THEN 1 
                    WHEN {sql} IS NULL 
                        THEN NULL 
                    ELSE 0 
                END)"""
        return clean_query(sql)

    # Model Evaluation Methods.

    def classification_report(
        self,
        metrics: Union[
            None, str, list[Literal[tuple(mt.FUNCTIONS_CLASSIFICATION_DICTIONNARY)]]
        ] = None,
        cutoff: PythonNumber = 0.5,
        nbins: int = 10000,
    ) -> Union[float, TableSample]:
        """
        Computes a classification report using multiple metrics 
        to evaluate the model  (AUC, accuracy, PRC AUC, F1...). 

        Parameters
        ----------
        metrics: list, optional
            List  of the  metrics  to use to compute the  final 
            report.
                accuracy    : Accuracy
                aic         : Akaike’s  Information  Criterion
                auc         : Area Under the Curve (ROC)
                ba          : Balanced Accuracy
                              = (tpr + tnr) / 2
                best_cutoff : Cutoff  which optimised the  ROC 
                              Curve prediction.
                bic         : Bayesian  Information  Criterion
                bm          : Informedness = tpr + tnr - 1
                csi         : Critical Success Index 
                              = tp / (tp + fn + fp)
                f1          : F1 Score
                fdr         : False Discovery Rate = 1 - ppv
                fm          : Fowlkes–Mallows index
                              = sqrt(ppv * tpr)
                fnr         : False Negative Rate 
                              = fn / (fn + tp)
                for         : False Omission Rate = 1 - npv
                fpr         : False Positive Rate 
                              = fp / (fp + tn)
                logloss     : Log Loss
                lr+         : Positive Likelihood Ratio
                              = tpr / fpr
                lr-         : Negative Likelihood Ratio
                              = fnr / tnr
                dor         : Diagnostic Odds Ratio
                mcc         : Matthews Correlation Coefficient 
                mk          : Markedness = ppv + npv - 1
                npv         : Negative Predictive Value 
                              = tn / (tn + fn)
                prc_auc     : Area Under the Curve (PRC)
                precision   : Precision = tp / (tp + fp)
                pt          : Prevalence Threshold
                              = sqrt(fpr) / (sqrt(tpr) + sqrt(fpr))
                recall      : Recall = tp / (tp + fn)
                specificity : Specificity = tn / (tn + fp)
        cutoff: PythonNumber, optional
            Probability cutoff.
        nbins: int, optional
            [Used to compute ROC AUC, PRC AUC and the best cutoff]
            An  integer  value  that   determines  the  number  of 
            decision  boundaries. Decision  boundaries are set  at 
            equally  spaced intervals between 0 and 1,  inclusive. 
            Greater values for nbins give more precise estimations 
            of   the   metrics,  but   can  potentially   decrease 
            performance. The maximum value is 999,999. If negative, 
            the maximum value is used.

        Returns
        -------
        TableSample
            report.
        """
        return mt.classification_report(
            self.y,
            [self.deploySQL(), self.deploySQL(cutoff=cutoff)],
            self.test_relation,
            metrics=metrics,
            cutoff=cutoff,
            nbins=nbins,
        )

    report = classification_report

    def confusion_matrix(self, cutoff: PythonNumber = 0.5) -> TableSample:
        """
        Computes the model confusion matrix.

        Parameters
        ----------
        cutoff: PythonNumber, optional
            Probability cutoff.

        Returns
        -------
        TableSample
            confusion matrix.
        """
        return mt.confusion_matrix(
            self.y, self.deploySQL(cutoff=cutoff), self.test_relation,
        )

    def score(
        self,
        metric: Literal[tuple(mt.FUNCTIONS_CLASSIFICATION_DICTIONNARY)] = "accuracy",
        cutoff: PythonNumber = 0.5,
        nbins: int = 10000,
    ) -> float:
        """
        Computes the model score.

        Parameters
        ----------
        metric: str, optional
            The metric to use to compute the score.
                accuracy    : Accuracy
                aic         : Akaike’s  Information  Criterion
                auc         : Area Under the Curve (ROC)
                ba          : Balanced Accuracy
                              = (tpr + tnr) / 2
                best_cutoff : Cutoff  which optimised the  ROC 
                              Curve prediction.
                bic         : Bayesian  Information  Criterion
                bm          : Informedness = tpr + tnr - 1
                csi         : Critical Success Index 
                              = tp / (tp + fn + fp)
                f1          : F1 Score
                fdr         : False Discovery Rate = 1 - ppv
                fm          : Fowlkes–Mallows index
                              = sqrt(ppv * tpr)
                fnr         : False Negative Rate 
                              = fn / (fn + tp)
                for         : False Omission Rate = 1 - npv
                fpr         : False Positive Rate 
                              = fp / (fp + tn)
                logloss     : Log Loss
                lr+         : Positive Likelihood Ratio
                              = tpr / fpr
                lr-         : Negative Likelihood Ratio
                              = fnr / tnr
                dor         : Diagnostic Odds Ratio
                mcc         : Matthews Correlation Coefficient 
                mk          : Markedness = ppv + npv - 1
                npv         : Negative Predictive Value 
                              = tn / (tn + fn)
                prc_auc     : Area Under the Curve (PRC)
                precision   : Precision = tp / (tp + fp)
                pt          : Prevalence Threshold
                              = sqrt(fpr) / (sqrt(tpr) + sqrt(fpr))
                recall      : Recall = tp / (tp + fn)
                specificity : Specificity = tn / (tn + fp)
        cutoff: PythonNumber, optional
            Cutoff for which the tested category will be
            accepted as a prediction.
        nbins: int, optional
            [Only  when  method  is set  to  auc|prc_auc|best_cutoff] 
            An  integer value that determines the number of  decision
            boundaries. Decision boundaries are set at equally spaced 
            intervals between 0 and 1,  inclusive. Greater values for 
            nbins give more precise  estimations of the AUC,  but can 
            potentially  decrease performance.  The maximum value  is 
            999,999. If negative, the maximum value is used.

        Returns
        -------
        float
            score
        """
        fun = mt.FUNCTIONS_CLASSIFICATION_DICTIONNARY[metric]
        if metric in (
            "log_loss",
            "logloss",
            "aic",
            "bic",
            "prc_auc",
            "auc",
            "best_cutoff",
            "best_threshold",
        ):
            args2 = self.deploySQL()
        else:
            args2 = self.deploySQL(cutoff=cutoff)
        args = [self.y, args2, self.test_relation]
        kwargs = {}
        if metric in mt.FUNCTIONS_CLASSIFICATION_DICTIONNARY and (
            metric not in ("aic", "bic")
        ):
            kwargs["pos_label"] = 1
        if metric in ("aic", "bic"):
            args += [len(self.X)]
        elif metric in ("prc_auc", "auc", "best_cutoff", "best_threshold"):
            kwargs["nbins"] = nbins
        return fun(*args, **kwargs)

    # Prediction / Transformation Methods.

    def predict(
        self,
        vdf: SQLRelation,
        X: Optional[SQLColumns] = None,
        name: str = "",
        cutoff: PythonNumber = 0.5,
        inplace: bool = True,
    ) -> vDataFrame:
        """
        Predicts using the input relation.

        Parameters
        ----------
        vdf: SQLRelation
            Object  to use to run  the prediction.  You can 
            also  specify a  customized  relation,  but you 
            must  enclose  it with an alias.  For  example, 
            "(SELECT 1) x" is correct, whereas "(SELECT 1)" 
            and "SELECT 1" are incorrect.
        X: SQLColumns, optional
            List of the columns  used to deploy the models. 
            If empty, the model predictors will be used.
        name: str, optional
            Name of the added vDataColumn. If empty, a name 
            will be generated.
        cutoff: float, optional
            Probability cutoff.
        inplace: bool, optional
            If set to True, the prediction will be added to 
            the vDataFrame.

        Returns
        -------
        vDataFrame
            the input object.
        """
        # Inititalization
        if isinstance(X, type(None)):
            X = self.X
        elif isinstance(X, str):
            X = [X]
        if not (0 <= cutoff <= 1):
            raise ValueError(
                "Incorrect parameter 'cutoff'.\nThe cutoff "
                "must be between 0 and 1, inclusive."
            )
        if isinstance(vdf, str):
            vdf = vDataFrame(vdf)
        X = [quote_ident(elem) for elem in X]
        if not (name):
            name = gen_name([self._model_type, self.model_name])

        # In Place
        vdf_return = vdf if inplace else vdf.copy()

        # Result
        return vdf_return.eval(name, self.deploySQL(X=X, cutoff=cutoff))

    def predict_proba(
        self,
        vdf: SQLRelation,
        X: Optional[SQLColumns] = None,
        name: str = "",
        pos_label: Optional[PythonScalar] = None,
        inplace: bool = True,
    ) -> vDataFrame:
        """
        Returns the model's  probabilities  using the input 
        relation.

        Parameters
        ----------
        vdf: SQLRelation
            Object  to use to run  the prediction.  You can 
            also  specify a  customized  relation,  but you 
            must  enclose  it with an alias.  For  example, 
            "(SELECT 1) x" is correct, whereas "(SELECT 1)" 
            and "SELECT 1" are incorrect.
        X: SQLColumns, optional
            List of the columns  used to deploy the models. 
            If empty, the model predictors will be used.
        name: str, optional
            Name of the added vDataColumn. If empty, a name 
            will be generated.
        pos_label: PythonScalar, optional
            Class  label.  For binary classification,  this 
            can be either 1 or 0.
        inplace: bool, optional
            If set to True, the prediction will be added to 
            the vDataFrame.

        Returns
        -------
        vDataFrame
            the input object.
        """
        # Inititalization
        if isinstance(X, type(None)):
            X = self.X
        elif isinstance(X, str):
            X = [X]
        if pos_label not in [1, 0, "0", "1", None]:
            raise ValueError(
                "Incorrect parameter 'pos_label'.\nThe class label "
                "can only be 1 or 0 in case of Binary Classification."
            )
        if isinstance(vdf, str):
            vdf = vDataFrame(vdf)
        X = [quote_ident(elem) for elem in X]
        if not (name):
            name = gen_name([self._model_type, self.model_name])

        # In Place
        vdf_return = vdf if inplace else vdf.copy()

        # Result
        name_tmp = name
        if pos_label in [0, "0", None]:
            if pos_label == None:
                name_tmp = f"{name}_0"
            vdf_return.eval(name_tmp, f"1 - {self.deploySQL(X=X)}")
        if pos_label in [1, "1", None]:
            if pos_label == None:
                name_tmp = f"{name}_1"
            vdf_return.eval(name_tmp, self.deploySQL(X=X))

        return vdf_return

    # Plotting Methods.

    def cutoff_curve(
        self,
        nbins: int = 30,
        show: bool = True,
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> TableSample:
        """
        Draws the model Cutoff curve.

        Parameters
        ----------
        nbins: int, optional
            The number of bins.
        show: bool, optional
            If set to True,  the  Plotting 
            object  will be returned.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any optional parameter to pass 
            to the Plotting functions.

        Returns
        -------
        TableSample
            cutoff curve data points.
        """
        return mt.roc_curve(
            self.y,
            self.deploySQL(),
            self.test_relation,
            nbins=nbins,
            cutoff_curve=True,
            show=show,
            chart=chart,
            **style_kwargs,
        )

    def lift_chart(
        self,
        nbins: int = 1000,
        show: bool = True,
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> TableSample:
        """
    	Draws the model Lift Chart.

        Parameters
        ----------
        nbins: int, optional
            The number of bins.
        show: bool, optional
            If set to True,  the  Plotting 
            object  will be returned.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any optional parameter to pass 
            to the Plotting functions.

    	Returns
    	-------
    	TableSample
    		lift chart data points.
		"""
        return mt.lift_chart(
            self.y,
            self.deploySQL(),
            self.test_relation,
            nbins=nbins,
            show=show,
            chart=chart,
            **style_kwargs,
        )

    def prc_curve(
        self,
        nbins: int = 30,
        show: bool = True,
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> TableSample:
        """
    	Draws the model PRC curve.

        Parameters
        ----------
        nbins: int, optional
            The number of bins.
        show: bool, optional
            If set to True,  the  Plotting 
            object  will be returned.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any optional parameter to pass 
            to the Plotting functions.

    	Returns
    	-------
    	TableSample
    		PRC curve data points.
		"""
        return mt.prc_curve(
            self.y,
            self.deploySQL(),
            self.test_relation,
            nbins=nbins,
            show=show,
            chart=chart,
            **style_kwargs,
        )

    def roc_curve(
        self,
        nbins: int = 30,
        show: bool = True,
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> TableSample:
        """
        Draws the model ROC curve.

        Parameters
        ----------
        nbins: int, optional
            The number of bins.
        show: bool, optional
            If set to True,  the  Plotting 
            object  will be returned.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any optional parameter to pass 
            to the Plotting functions.

        Returns
        -------
        TableSample
            ROC curve data points.
        """
        return mt.roc_curve(
            self.y,
            self.deploySQL(),
            self.test_relation,
            nbins=nbins,
            show=show,
            chart=chart,
            **style_kwargs,
        )


class MulticlassClassifier(Classifier):

    # System & Special Methods.

    def _check_pos_label(self, pos_label: PythonScalar) -> PythonScalar:
        """
        Checks if the pos_label is correct.
        """
        if pos_label == None and self._is_binary_classifier():
            return 1
        elif pos_label == None:
            return None
        elif str(pos_label) not in [str(c) for c in self.classes_]:
            raise ParameterError(
                "Parameter 'pos_label' must be one of the response column classes."
            )
        return pos_label

    # Attributes Methods.

    def _get_classes(self) -> np.ndarray:
        """
        Returns the model's classes.
        """
        classes = _executeSQL(
            query=f"""
                SELECT 
                    /*+LABEL('learn.VerticaModel.fit')*/ 
                    DISTINCT {self.y} 
                FROM {self.input_relation} 
                WHERE {self.y} IS NOT NULL 
                ORDER BY 1""",
            method="fetchall",
            print_time_sql=False,
        )
        classes = np.array([c[0] for c in classes])
        return self._array_to_int(classes)

    def _is_binary_classifier(self) -> bool:
        """
        Returns True if the model is a Binary Classifier.
        """
        if len(self.classes_) == 2 and self.classes_[0] == 0 and self.classes_[1] == 1:
            return True
        return False

    # I/O Methods.

    def deploySQL(
        self,
        X: Optional[SQLColumns] = None,
        pos_label: Optional[PythonScalar] = None,
        cutoff: Optional[PythonNumber] = None,
        allSQL: bool = False,
    ) -> SQLExpression:
        """
        Returns the SQL code needed to deploy the model. 

        Parameters
        ----------
        X: SQLColumns, optional
            List of the columns used to deploy the model. 
            If empty, the model predictors will be used.
        pos_label: PythonScalar, optional
            Label to consider as positive. All the other 
            classes  will be  merged and  considered  as 
            negative for multiclass classification.
        cutoff: PythonNumber, optional
            Cutoff for which the tested category will be 
            accepted  as a prediction. If the cutoff  is 
            not  between 0 and 1,  a probability will be 
            returned.
        allSQL: bool, optional
            If set to True, the output will be a list of
            the different SQL codes needed to deploy the 
            different categories score.

        Returns
        -------
        SQLExpression
            the SQL code needed to deploy the model.
        """
        if isinstance(X, type(None)):
            X = self.X
        elif isinstance(X, str):
            X = [X]
        X = [quote_ident(x) for x in X]

        if not (self._is_native):
            sql = self.to_memmodel().predict_proba_sql(X)
        else:
            sql = [
                f"""
                {self._vertica_predict_sql}({', '.join(X)} 
                    USING PARAMETERS 
                    model_name = '{self.model_name}',
                    class = '{{}}',
                    type = 'probability',
                    match_by_pos = 'true')""",
                f"""
                {self._vertica_predict_sql}({', '.join(X)} 
                    USING PARAMETERS 
                    model_name = '{self.model_name}',
                    match_by_pos = 'true')""",
            ]
        if not (allSQL):
            if pos_label in list(self.classes_):
                if not (self._is_native):
                    sql = sql[self._get_match_index(pos_label, self.classes_, False)]
                else:
                    sql = sql[0].format(pos_label)
                if isinstance(cutoff, (int, float)) and 0.0 <= cutoff <= 1.0:
                    sql = f"""
                        (CASE 
                            WHEN {sql} >= {cutoff} 
                                THEN '{pos_label}' 
                            WHEN {sql} IS NULL 
                                THEN NULL 
                            ELSE '{{}}' 
                        END)"""
                    if len(self.classes_) > 2:
                        sql = sql.format(f"Non-{pos_label}")
                    else:
                        if self.classes_[0] != pos_label:
                            non_pos_label = self.classes_[0]
                        else:
                            non_pos_label = self.classes_[1]
                        sql = sql.format(non_pos_label)
            else:
                if not (self._is_native):
                    sql = self.to_memmodel().predict_sql(X)
                else:
                    sql = sql[1]
        return clean_query(sql)

    # Model Evaluation Methods.

    def _get_final_relation(self, pos_label: Optional[PythonScalar] = None,) -> str:
        """
        Returns  the  final  relation  used to do  the 
        predictions.
        """
        return self.test_relation

    def _get_y_proba(self, pos_label: Optional[PythonScalar] = None,) -> str:
        """
        Returns the input which represents the  model's 
        probabilities.
        """
        return self.deploySQL(allSQL=True)[0].format(pos_label)

    def _get_y_score(
        self,
        pos_label: Optional[PythonScalar] = None,
        cutoff: Optional[PythonNumber] = None,
        allSQL: bool = False,
    ) -> str:
        """
        Returns  the input which represents the model's 
        scoring.
        """
        return self.deploySQL(pos_label=pos_label, cutoff=cutoff, allSQL=allSQL)

    def classification_report(
        self,
        metrics: Union[
            None, str, list[Literal[tuple(mt.FUNCTIONS_CLASSIFICATION_DICTIONNARY)]]
        ] = None,
        cutoff: PythonNumber = None,
        labels: Union[None, str, list[str]] = None,
        nbins: int = 10000,
    ) -> Union[float, TableSample]:
        """
        Computes a classification report using multiple metrics
        to evaluate the model  (AUC, accuracy, PRC AUC, F1...). 
        For  multiclass classification,  it will consider  each 
        category as positive and switch to the next one  during 
        the computation.

        Parameters
        ----------
        metrics: list, optional
            List of  the metrics  to use to compute the  final 
            report.
                accuracy    : Accuracy
                aic         : Akaike’s  Information  Criterion
                auc         : Area Under the Curve (ROC)
                ba          : Balanced Accuracy
                              = (tpr + tnr) / 2
                best_cutoff : Cutoff  which optimised the  ROC 
                              Curve prediction.
                bic         : Bayesian  Information  Criterion
                bm          : Informedness = tpr + tnr - 1
                csi         : Critical Success Index 
                              = tp / (tp + fn + fp)
                f1          : F1 Score
                fdr         : False Discovery Rate = 1 - ppv
                fm          : Fowlkes–Mallows index
                              = sqrt(ppv * tpr)
                fnr         : False Negative Rate 
                              = fn / (fn + tp)
                for         : False Omission Rate = 1 - npv
                fpr         : False Positive Rate 
                              = fp / (fp + tn)
                logloss     : Log Loss
                lr+         : Positive Likelihood Ratio
                              = tpr / fpr
                lr-         : Negative Likelihood Ratio
                              = fnr / tnr
                dor         : Diagnostic Odds Ratio
                mcc         : Matthews Correlation Coefficient 
                mk          : Markedness = ppv + npv - 1
                npv         : Negative Predictive Value 
                              = tn / (tn + fn)
                prc_auc     : Area Under the Curve (PRC)
                precision   : Precision = tp / (tp + fp)
                pt          : Prevalence Threshold
                              = sqrt(fpr) / (sqrt(tpr) + sqrt(fpr))
                recall      : Recall = tp / (tp + fn)
                specificity : Specificity = tn / (tn + fp)
        cutoff: PythonNumber, optional
            Cutoff for which the tested category will be  accepted
            as a prediction.  For multiclass  classification, each 
            tested category becomes  the positives  and the others 
            are  merged  into  the   negatives.  The  cutoff  will 
            represent  the  classes  threshold.  If  it  is  empty, 
            the  regular  cutoff (1 / number of classes)  will  be 
            used.
        labels: str / list, optional
            List  of the  different  labels to be used during  the 
            computation.
        nbins: int, optional
            [Used to compute ROC AUC, PRC AUC and the best cutoff]
            An  integer  value  that   determines  the  number  of 
            decision  boundaries.  Decision boundaries are set  at 
            equally spaced intervals  between 0 and 1,  inclusive. 
            Greater values for nbins give more precise estimations 
            of   the  metrics,   but   can  potentially   decrease 
            performance. 
            The maximum value is 999,999. If negative, the maximum 
            value is used.

        Returns
        -------
        TableSample
            report.
        """
        if isinstance(labels, type(None)):
            labels = self.classes_
        elif isinstance(labels, str):
            labels = [labels]
        return mt.classification_report(
            estimator=self, metrics=metrics, labels=labels, cutoff=cutoff, nbins=nbins,
        )

    report = classification_report

    def confusion_matrix(
        self,
        pos_label: Optional[PythonScalar] = None,
        cutoff: Optional[PythonNumber] = None,
    ) -> TableSample:
        """
        Computes the model confusion matrix.

        Parameters
        ----------
        pos_label: PythonScalar, optional
            Label  to consider  as positive.  All the other  classes 
            will be merged and considered as negative for multiclass 
            classification.  If  the 'pos_label' is not defined, the 
            entire confusion matrix will be drawn.
        cutoff: PythonNumber, optional
            Cutoff for which the tested category will be accepted as 
            a prediction. It is only used if 'pos_label' is defined.

        Returns
        -------
        TableSample
            confusion matrix.
        """
        if hasattr(self, "_confusion_matrix"):
            return self._confusion_matrix(pos_label=pos_label, cutoff=cutoff,)
        elif pos_label == None:
            return mt.multilabel_confusion_matrix(
                self.y, self.deploySQL(), self.test_relation, self.classes_
            )
        else:
            pos_label = self._check_pos_label(pos_label=pos_label)
            if cutoff == None:
                cutoff = 1.0 / len(self.classes_)
            return mt.confusion_matrix(
                self.y,
                self.deploySQL(pos_label=pos_label, cutoff=cutoff),
                self.test_relation,
                pos_label=pos_label,
            )

    def score(
        self,
        metric: Literal[tuple(mt.FUNCTIONS_CLASSIFICATION_DICTIONNARY)] = "accuracy",
        average: Literal["micro", "macro", "weighted", "scores"] = "weighted",
        pos_label: Optional[PythonScalar] = None,
        cutoff: PythonNumber = 0.5,
        nbins: int = 10000,
    ) -> Union[float, list[float]]:
        """
        Computes the model score.

        Parameters
        ----------
        metric: str, optional
            The metric to use to compute the score.
                accuracy    : Accuracy
                aic         : Akaike’s  Information  Criterion
                auc         : Area Under the Curve (ROC)
                ba          : Balanced Accuracy
                              = (tpr + tnr) / 2
                best_cutoff : Cutoff  which optimised the  ROC 
                              Curve prediction.
                bic         : Bayesian  Information  Criterion
                bm          : Informedness = tpr + tnr - 1
                csi         : Critical Success Index 
                              = tp / (tp + fn + fp)
                f1          : F1 Score
                fdr         : False Discovery Rate = 1 - ppv
                fm          : Fowlkes–Mallows index
                              = sqrt(ppv * tpr)
                fnr         : False Negative Rate 
                              = fn / (fn + tp)
                for         : False Omission Rate = 1 - npv
                fpr         : False Positive Rate 
                              = fp / (fp + tn)
                logloss     : Log Loss
                lr+         : Positive Likelihood Ratio
                              = tpr / fpr
                lr-         : Negative Likelihood Ratio
                              = fnr / tnr
                dor         : Diagnostic Odds Ratio
                mcc         : Matthews Correlation Coefficient 
                mk          : Markedness = ppv + npv - 1
                npv         : Negative Predictive Value 
                              = tn / (tn + fn)
                prc_auc     : Area Under the Curve (PRC)
                precision   : Precision = tp / (tp + fp)
                pt          : Prevalence Threshold
                              = sqrt(fpr) / (sqrt(tpr) + sqrt(fpr))
                recall      : Recall = tp / (tp + fn)
                specificity : Specificity = tn / (tn + fp)
        average: str, optional
            The method used to  compute the final score for
            multiclass-classification.
                micro    : positive  and   negative  values 
                           globally.
                macro    : average  of  the  score of  each 
                           class.
                weighted : weighted average of the score of 
                           each class.
                scores   : scores  for   all  the  classes.
        pos_label: PythonScalar, optional
            Label  to  consider   as  positive.  All the 
            other classes will be  merged and considered 
            as negative  for multiclass  classification.
        cutoff: PythonNumber, optional
            Cutoff for which the tested category will be
            accepted as a prediction.
        nbins: int, optional
            [Only  when  method  is set  to  auc|prc_auc|best_cutoff] 
            An  integer value that determines the number of  decision
            boundaries. Decision boundaries are set at equally spaced 
            intervals between 0 and 1,  inclusive. Greater values for 
            nbins give more precise  estimations of the AUC,  but can 
            potentially  decrease performance.  The maximum value  is 
            999,999. If negative, the maximum value is used.

        Returns
        -------
        float
            score.
        """
        fun = mt.FUNCTIONS_CLASSIFICATION_DICTIONNARY[metric]
        pos_label = self._check_pos_label(pos_label=pos_label)
        y_proba = self._get_y_proba(pos_label=pos_label)
        if metric in (
            "auc",
            "prc_auc",
            "best_cutoff",
            "best_threshold",
            "logloss",
            "log_loss",
        ):
            y_score = self._get_y_score(allSQL=True)
        else:
            y_score = self._get_y_score(pos_label=pos_label, cutoff=cutoff)
        final_relation = self._get_final_relation(pos_label=pos_label)
        args = [self.y, y_score, final_relation]
        kwargs = {}
        if metric not in ("aic", "bic"):
            kwargs = {
                "average": average,
                "labels": self.classes_,
                "pos_label": pos_label,
            }
        if metric in ("aic", "bic"):
            args += [len(self.X)]
        elif metric in ("auc", "prc_auc", "best_cutoff", "best_threshold"):
            kwargs["nbins"] = nbins
        return fun(*args, **kwargs)

    # Prediction / Transformation Methods.

    def predict(
        self,
        vdf: SQLRelation,
        X: Optional[SQLColumns] = None,
        name: str = "",
        cutoff: Optional[PythonNumber] = None,
        inplace: bool = True,
    ) -> vDataFrame:
        """
        Predicts using the input relation.

        Parameters
        ----------
        vdf: SQLRelation
            Object  to use to run  the prediction.  You can 
            also  specify a  customized  relation,  but you 
            must  enclose  it with an alias.  For  example, 
            "(SELECT 1) x" is correct, whereas "(SELECT 1)" 
            and "SELECT 1" are incorrect.
        X: SQLColumns, optional
            List of the columns  used to deploy the models. 
            If empty, the model predictors will be used.
        name: str, optional
            Name of the added vDataColumn. If empty, a name 
            will be generated.
        cutoff: PythonNumber, optional
            Cutoff  for which  the tested category will  be 
            accepted  as a  prediction.  This parameter  is 
            only used for binary classification.
        inplace: bool, optional
            If set to True, the prediction will be added to 
            the vDataFrame.

        Returns
        -------
        vDataFrame
            the input object.
        """
        # Using special method in case of non-native models
        if hasattr(self, "_predict"):
            return self._predict(
                vdf=vdf, X=X, name=name, cutoff=cutoff, inplace=inplace
            )

        # Inititalization
        if isinstance(X, type(None)):
            X = self.X
        elif isinstance(X, str):
            X = [X]
        else:
            X = [quote_ident(elem) for elem in X]
        if not (name):
            name = gen_name([self._model_type, self.model_name])
        if cutoff == None:
            cutoff = 1.0 / len(self.classes_)
        elif not (0 <= cutoff <= 1):
            raise ValueError(
                "Incorrect parameter 'cutoff'.\nThe cutoff "
                "must be between 0 and 1, inclusive."
            )
        if isinstance(vdf, str):
            vdf = vDataFrame(vdf)

        # In Place
        vdf_return = vdf if inplace else vdf.copy()

        # Check if it is a Binary Classifier
        pos_label = None
        if (
            len(self.classes_) == 2
            and self.classes_[0] in [0, "0"]
            and self.classes_[1] in [1, "1"]
        ):
            pos_label = 1

        # Result
        return vdf_return.eval(
            name, self.deploySQL(X=X, pos_label=pos_label, cutoff=cutoff)
        )

    def predict_proba(
        self,
        vdf: SQLRelation,
        X: Optional[SQLColumns] = None,
        name: str = "",
        pos_label: Optional[PythonScalar] = None,
        inplace: bool = True,
    ) -> vDataFrame:
        """
        Returns the model's probabilities using the input 
        relation.

        Parameters
        ----------
        vdf: SQLRelation
            Object  to use to run  the prediction.  You can 
            also  specify a  customized  relation,  but you 
            must  enclose  it with an alias.  For  example, 
            "(SELECT 1) x" is correct, whereas "(SELECT 1)" 
            and "SELECT 1" are incorrect.
        X: SQLColumns, optional
            List of the columns  used to deploy the models. 
            If empty, the model predictors will be used.
        name: str, optional
            Name of the added vDataColumn. If empty, a name 
            will be generated.
        pos_label: PythonScalar, optional
            Class  label.  For binary classification,  this 
            can be either 1 or 0.
        inplace: bool, optional
            If set to True, the prediction will be added to 
            the vDataFrame.

        Returns
        -------
        vDataFrame
            the input object.
        """
        if hasattr(self, "_predict_proba"):
            return self._predict_proba(
                vdf=vdf, X=X, name=name, pos_label=pos_label, inplace=inplace,
            )
        # Inititalization
        if isinstance(X, type(None)):
            X = self.X
        elif isinstance(X, str):
            X = [X]
        else:
            X = [quote_ident(elem) for elem in X]
        assert pos_label is None or pos_label in self.classes_, ParameterError(
            "Incorrect parameter 'pos_label'.\nThe class label "
            f"must be in [{'|'.join([str(c) for c in self.classes_])}]. "
            f"Found '{pos_label}'."
        )
        if isinstance(vdf, str):
            vdf = vDataFrame(vdf)
        if not (name):
            name = gen_name([self._model_type, self.model_name])

        # In Place
        vdf_return = vdf if inplace else vdf.copy()

        # Result
        if pos_label == None:
            for c in self.classes_:
                name_tmp = gen_name([name, c])
                vdf_return.eval(name_tmp, self.deploySQL(pos_label=c, cutoff=None, X=X))
        else:
            vdf_return.eval(name, self.deploySQL(pos_label=pos_label, cutoff=None, X=X))

        return vdf_return

    # Plotting Methods.

    def _get_sql_plot(self, pos_label: PythonScalar) -> str:
        """
        Returns the SQL needed to draw the plot.
        """
        pos_label = self._check_pos_label(pos_label)
        if not (self._is_native):
            return self.deploySQL(allSQL=True)[
                self._get_match_index(pos_label, self.classes_, False)
            ]
        else:
            return self.deploySQL(allSQL=True)[0].format(pos_label)

    def _get_plot_args(
        self, pos_label: Optional[PythonScalar] = None, method: Optional[str] = None
    ) -> list:
        """
        Returns the args used by plotting methods.
        """
        pos_label = self._check_pos_label(pos_label)
        if method == "contour":
            args = [
                self.X,
                self.deploySQL(X=self.X, pos_label=pos_label),
            ]
        else:
            args = [
                self.y,
                self._get_sql_plot(pos_label),
                self.test_relation,
                pos_label,
            ]
        return args

    def _get_plot_kwargs(
        self,
        pos_label: Optional[PythonScalar] = None,
        nbins: int = 30,
        chart: Optional[PlottingObject] = None,
        method: Optional[str] = None,
    ) -> dict:
        """
        Returns the kwargs used by plotting methods.
        """
        pos_label = self._check_pos_label(pos_label)
        res = {"nbins": nbins, "chart": chart}
        if method == "contour":
            res["func_name"] = f"p({self.y} = '{pos_label}')"
        elif method == "cutoff":
            res["cutoff_curve"] = True
        return res

    def contour(
        self,
        pos_label: Optional[PythonScalar] = None,
        nbins: int = 100,
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the model's contour plot.

        Parameters
        ----------
        pos_label: PythonScalar, optional
            Label  to  consider  as positive. All  the other 
            classes  will  be  merged   and   considered  as 
            negative for multiclass classification.
        nbins: int, optional
             Number  of  bins  used to  discretize  the  two 
             predictors.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any optional parameter to pass to the Plotting 
            functions.

        Returns
        -------
        obj
            Plotting Object.
        """
        pos_label = self._check_pos_label(pos_label=pos_label)
        return vDataFrame(self.input_relation).contour(
            *self._get_plot_args(pos_label=pos_label, method="contour"),
            **self._get_plot_kwargs(
                pos_label=pos_label, nbins=nbins, chart=chart, method="contour"
            ),
            **style_kwargs,
        )

    def cutoff_curve(
        self,
        pos_label: Optional[PythonScalar] = None,
        nbins: int = 30,
        show: bool = True,
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> TableSample:
        """
        Draws the model Cutoff curve.

        Parameters
        ----------
        pos_label: PythonScalar, optional
            To  draw the Cutoff curve, one of the response  column 
            classes  must  be  the  positive  one.  The  parameter 
            'pos_label' represents this class.
        nbins: int, optional
            An integer value that determines the number of decision   
            boundaries.  Decision  boundaries  are   set at equally
            -spaced intervals between 0 and 1, inclusive.
        show: bool, optional
            If set to True,  the  Plotting object will be returned.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any  optional  parameter  to  pass  to  the  Plotting 
            functions.

        Returns
        -------
        TableSample
            cutoff curve data points.
        """
        return mt.roc_curve(
            *self._get_plot_args(pos_label=pos_label, method="cutoff"),
            show=show,
            **self._get_plot_kwargs(nbins=nbins, chart=chart, method="cutoff"),
            **style_kwargs,
        )

    def lift_chart(
        self,
        pos_label: Optional[PythonScalar] = None,
        nbins: int = 1000,
        show: bool = True,
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> TableSample:
        """
    	Draws the model Lift Chart.

    	Parameters
    	----------
    	pos_label: PythonScalar, optional
            To  draw  the Lift Chart, one of the  response  column 
            classes  must  be  the  positive  one.  The  parameter 
            'pos_label' represents this class.
        nbins: int, optional
            An integer value that determines the number of decision   
            boundaries.  Decision  boundaries  are   set at equally
            -spaced intervals between 0 and 1, inclusive.
        show: bool, optional
            If set to True,  the  Plotting object will be returned.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any  optional  parameter  to  pass  to  the  Plotting 
            functions.

    	Returns
    	-------
    	TableSample
    		lift chart data points.
		"""
        return mt.lift_chart(
            *self._get_plot_args(pos_label=pos_label),
            show=show,
            **self._get_plot_kwargs(nbins=nbins, chart=chart),
            **style_kwargs,
        )

    def prc_curve(
        self,
        pos_label: Optional[PythonScalar] = None,
        nbins: int = 30,
        show: bool = True,
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> TableSample:
        """
    	Draws the model PRC curve.

    	Parameters
    	----------
    	pos_label: PythonScalar, optional
            To  draw  the PRC curve,  one of the  response  column 
            classes  must  be  the  positive  one.  The  parameter 
            'pos_label' represents this class.
        nbins: int, optional
            An integer value that determines the number of decision   
            boundaries.  Decision  boundaries  are   set at equally
            -spaced intervals between 0 and 1, inclusive.
        show: bool, optional
            If set to True,  the  Plotting object will be returned.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any  optional  parameter  to  pass  to  the  Plotting 
            functions.

    	Returns
    	-------
    	TableSample
    		PRC curve data points.
		"""
        return mt.prc_curve(
            *self._get_plot_args(pos_label=pos_label),
            show=show,
            **self._get_plot_kwargs(nbins=nbins, chart=chart),
            **style_kwargs,
        )

    def roc_curve(
        self,
        pos_label: Optional[PythonScalar] = None,
        nbins: int = 30,
        show: bool = True,
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> TableSample:
        """
    	Draws the model ROC curve.

    	Parameters
    	----------
    	pos_label: PythonScalar, optional
            To  draw  the ROC curve,  one of the  response  column 
            classes  must  be  the  positive  one.  The  parameter 
            'pos_label' represents this class.
        nbins: int, optional
            An integer value that determines the number of decision   
            boundaries.  Decision  boundaries  are   set at equally
            -spaced intervals between 0 and 1, inclusive.
        show: bool, optional
            If set to True,  the  Plotting object will be returned.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any  optional  parameter  to  pass  to  the  Plotting 
            functions.

    	Returns
    	-------
    	TableSample
    		roc curve data points.
		"""
        return mt.roc_curve(
            *self._get_plot_args(pos_label=pos_label),
            show=show,
            **self._get_plot_kwargs(nbins=nbins, chart=chart),
            **style_kwargs,
        )


class Regressor(Supervised):

    # Model Evaluation Methods.

    def regression_report(
        self,
        metrics: Union[
            str,
            Literal[None, "anova", "details"],
            list[Literal[tuple(mt.FUNCTIONS_REGRESSION_DICTIONNARY)]],
        ] = None,
    ) -> Union[float, TableSample]:
        """
        Computes a regression report using multiple metrics to 
        evaluate the model (r2, mse, max error...). 

        Parameters
        ----------
        metrics: str, optional
            The metrics to use to compute the regression report.
                None    : Computes the model different metrics.
                anova   : Computes the model ANOVA table.
                details : Computes the model details.
            It can also be a list of different metrics among the
            following:
                aic    : Akaike’s Information Criterion
                bic    : Bayesian Information Criterion
                max    : Max Error
                mae    : Mean Absolute Error
                median : Median Absolute Error
                mse    : Mean Squared Error
                msle   : Mean Squared Log Error
                qe     : quantile  error,  the quantile must be
                         included in the name. Example:
                         qe50.1% will return the quantile error 
                         using q=0.501.
                r2     : R squared coefficient
                r2a    : R2 adjusted
                rmse   : Root Mean Squared Error
                var    : Explained Variance 

        Returns
        -------
        TableSample
            report.
        """
        prediction = self.deploySQL()
        if self._model_type == "KNeighborsRegressor":
            test_relation = self.deploySQL()
            prediction = "predict_neighbors"
        elif self._model_type == "KernelDensity":
            test_relation = self.map
        else:
            test_relation = self.test_relation
        if metrics == "anova":
            return mt.anova_table(self.y, prediction, test_relation, len(self.X))
        elif metrics == "details":
            vdf = vDataFrame(f"SELECT {self.y} FROM {self.input_relation}")
            n = vdf[self.y].count()
            kurt = vdf[self.y].kurt()
            skew = vdf[self.y].skew()
            jb = vdf[self.y].agg(["jb"])[self.y][0]
            R2 = self.score(metric="r2")
            R2_adj = 1 - ((1 - R2) * (n - 1) / (n - len(self.X) - 1))
            anova_T = mt.anova_table(self.y, prediction, test_relation, len(self.X))
            F = anova_T["F"][0]
            p_F = anova_T["p_value"][0]
            return TableSample(
                {
                    "index": [
                        "Dep. Variable",
                        "Model",
                        "No. Observations",
                        "No. Predictors",
                        "R-squared",
                        "Adj. R-squared",
                        "F-statistic",
                        "Prob (F-statistic)",
                        "Kurtosis",
                        "Skewness",
                        "Jarque-Bera (JB)",
                    ],
                    "value": [
                        self.y,
                        self._model_type,
                        n,
                        len(self.X),
                        R2,
                        R2_adj,
                        F,
                        p_F,
                        kurt,
                        skew,
                        jb,
                    ],
                }
            )
        elif isinstance(metrics, type(None)) or isinstance(
            metrics, (str, list, np.ndarray)
        ):
            return mt.regression_report(
                self.y, prediction, test_relation, metrics=metrics, k=len(self.X)
            )

    report = regression_report

    def score(
        self,
        metric: Literal[
            tuple(mt.FUNCTIONS_REGRESSION_DICTIONNARY)
            + ("r2a", "r2_adj", "rsquared_adj", "r2adj", "r2adjusted", "rmse")
        ] = "r2",
    ) -> float:
        """
        Computes the model score.

        Parameters
        ----------
        metric: str, optional
            The metric to use to compute the score.
                aic    : Akaike’s Information Criterion
                bic    : Bayesian Information Criterion
                max    : Max Error
                mae    : Mean Absolute Error
                median : Median Absolute Error
                mse    : Mean Squared Error
                msle   : Mean Squared Log Error
                r2     : R squared coefficient
                r2a    : R2 adjusted
                rmse   : Root Mean Squared Error
                var    : Explained Variance 

        Returns
        -------
        float
            score.
        """
        # Initialization
        metric = str(metric).lower()
        if metric in ["r2adj", "r2adjusted"]:
            metric = "r2a"
        adj, root = False, False
        if metric in ("r2a", "r2adj", "r2adjusted", "r2_adj", "rsquared_adj"):
            metric, adj = "r2", True
        elif metric == "rmse":
            metric, root = "mse", True
        fun = mt.FUNCTIONS_REGRESSION_DICTIONNARY[metric]

        # Scoring
        if self._model_type == "KNeighborsRegressor":
            test_relation, prediction = self.deploySQL(), "predict_neighbors"
        elif self._model_type == "KernelDensity":
            test_relation, prediction = self.map, self.deploySQL()
        else:
            test_relation, prediction = self.test_relation, self.deploySQL()
        arg = [self.y, prediction, test_relation]
        if metric in ("aic", "bic") or adj:
            arg += [len(self.X)]
        if root or adj:
            arg += [True]
        return fun(*arg)

    # Prediction / Transformation Methods.

    def predict(
        self,
        vdf: SQLRelation,
        X: Optional[SQLColumns] = None,
        name: str = "",
        inplace: bool = True,
    ) -> vDataFrame:
        """
    	Predicts using the input relation.

    	Parameters
    	----------
    	vdf: SQLRelation
            Object  to use to run  the prediction.  You can 
            also  specify a  customized  relation,  but you 
            must  enclose  it with an alias.  For  example, 
            "(SELECT 1) x" is correct, whereas "(SELECT 1)" 
            and "SELECT 1" are incorrect.
        X: SQLColumns, optional
            List of the columns  used to deploy the models. 
            If empty, the model predictors will be used.
        name: str, optional
            Name of the added vDataColumn. If empty, a name 
            will be generated.
    	inplace: bool, optional
    		If set to True, the prediction will be added to 
            the vDataFrame.

    	Returns
    	-------
    	vDataFrame
    		the input object.
		"""
        if hasattr(self, "_predict"):
            return self._predict(vdf=vdf, X=X, name=name, inplace=inplace)
        if isinstance(X, type(None)):
            X = self.X
        if isinstance(X, str):
            X = [X]
        else:
            X = [quote_ident(elem) for elem in X]
        if isinstance(vdf, str):
            vdf = vDataFrame(vdf)
        if not (name):
            name = f"{self._model_type}_" + "".join(
                ch for ch in self.model_name if ch.isalnum()
            )
        if inplace:
            return vdf.eval(name, self.deploySQL(X=X))
        else:
            return vdf.copy().eval(name, self.deploySQL(X=X))


class Unsupervised(VerticaModel):

    # Model Fitting Method.

    def fit(
        self, input_relation: SQLRelation, X: Optional[SQLColumns] = None
    ) -> Optional[str]:
        """
    	Trains the model.

    	Parameters
    	----------
    	input_relation: SQLRelation
    		Training relation.
    	X: SQLColumns, optional
    		List of the predictors. If empty, all the 
            numerical columns will be used.

    	Returns
    	-------
    	str
    		model's summary.
		"""
        if isinstance(X, str):
            X = [X]
        if conf.get_option("overwrite_model"):
            self.drop()
        else:
            self._is_already_stored(raise_error=True)
        id_column, id_column_name = "", gen_tmp_name(name="id_column")
        if self._model_type in ("BisectingKMeans", "IsolationForest") and isinstance(
            conf.get_option("random_state"), int
        ):
            X_str = ", ".join([quote_ident(x) for x in X])
            id_column = f", ROW_NUMBER() OVER (ORDER BY {X_str}) AS {id_column_name}"
        if isinstance(input_relation, str) and self._model_type == "MCA":
            input_relation = vDataFrame(input_relation)
        tmp_view = False
        if isinstance(input_relation, vDataFrame) or (id_column):
            tmp_view = True
            if isinstance(input_relation, vDataFrame):
                self.input_relation = input_relation._genSQL()
            else:
                self.input_relation = input_relation
            if self._model_type == "MCA":
                result = input_relation.sum(columns=X)
                if isinstance(result, (int, float)):
                    result = [result]
                else:
                    result = result["sum"]
                result = sum(result) + (input_relation.shape()[0] - 1) * len(result)
                assert abs(result) < 0.01, ConversionError(
                    "MCA can only work on a transformed complete disjunctive table. "
                    "You should transform your relation first.\nTips: Use the "
                    "vDataFrame.cdt method to transform the relation."
                )
            relation = gen_tmp_name(
                schema=schema_relation(self.model_name)[0], name="view"
            )
            drop(relation, method="view")
            _executeSQL(
                query=f"""
                    CREATE VIEW {relation} AS 
                        SELECT 
                            /*+LABEL('learn.VerticaModel.fit')*/ *
                            {id_column} 
                        FROM {self.input_relation}""",
                title="Creating a temporary view to fit the model.",
            )
            if isinstance(X, type(None)) and (self._model_type == "KPrototypes"):
                X = input_relation.get_columns()
            elif isinstance(X, type(None)):
                X = input_relation.numcol()
        else:
            self.input_relation = input_relation
            relation = input_relation
            if isinstance(X, type(None)):
                X = vDataFrame(input_relation).numcol()
        self.X = [quote_ident(column) for column in X]
        parameters = self._get_vertica_param_dict()
        if "num_components" in parameters and not (parameters["num_components"]):
            del parameters["num_components"]
        fun = self._vertica_fit_sql if self._model_type != "MCA" else "PCA"
        query = f"""
            SELECT 
                /*+LABEL('learn.VerticaModel.fit')*/ 
                {fun}('{self.model_name}', '{relation}', '{', '.join(self.X)}'"""
        if self._model_type in ("BisectingKMeans", "KMeans", "KPrototypes",):
            query += f", {parameters['n_cluster']}"
        elif self._model_type == "Scaler":
            query += f", {parameters['method']}"
            del parameters["method"]
        if self._model_type not in ("Scaler", "MCA"):
            query += " USING PARAMETERS "
        for param in (
            "n_cluster",
            "separator",
            "null_column_name",
            "column_naming",
            "ignore_null",
            "drop_first",
        ):
            if param in parameters:
                del parameters[param]
        if (
            "init_method" in parameters
            and not (isinstance(parameters["init_method"], str))
            and self._model_type in ("KMeans", "BisectingKMeans", "KPrototypes",)
        ):
            name_init = gen_tmp_name(
                schema=schema_relation(self.model_name)[0],
                name=f"{self._model_type.lower()}_init",
            )
            del parameters["init_method"]
            drop(name_init, method="table")
            if len(self.parameters["init"]) != self.parameters["n_cluster"]:
                raise ParameterError(
                    f"'init' must be a list of 'n_cluster' = {self.parameters['n_cluster']} points"
                )
            else:
                for item in self.parameters["init"]:
                    if len(X) != len(item):
                        raise ParameterError(
                            f"Each points of 'init' must be of size len(X) = {len(self.X)}"
                        )
                query0 = []
                for i in range(len(self.parameters["init"])):
                    line = []
                    for j in range(len(self.parameters["init"][0])):
                        val = self.parameters["init"][i][j]
                        if isinstance(val, str):
                            val = "'" + val.replace("'", "''") + "'"
                        line += [str(val) + " AS " + X[j]]
                    line = ",".join(line)
                    if i == 0:
                        query0 += [
                            "SELECT /*+LABEL('learn.VerticaModel.fit')*/ " + line
                        ]
                    else:
                        query0 += ["SELECT " + line]
                query0 = " UNION ".join(query0)
                query0 = f"CREATE TABLE {name_init} AS {query0}"
                _executeSQL(query0, print_time_sql=False)
                query += f"initial_centers_table = '{name_init}', "
        elif "init_method" in parameters:
            del parameters["init_method"]
            query += f"init_method = '{self.parameters['init']}', "
        query += ", ".join([f"{p} = {parameters[p]}" for p in parameters])
        if self._model_type == "BisectingKMeans" and isinstance(
            conf.get_option("random_state"), int
        ):
            query += f", kmeans_seed={conf.get_option('random_state')}"
            query += f", id_column='{id_column_name}'"
        elif self._model_type == "IsolationForest" and isinstance(
            conf.get_option("random_state"), int
        ):
            query += f", seed={conf.get_option('random_state')}"
            query += f", id_column='{id_column_name}'"
        query += ")"
        try:
            _executeSQL(query, "Fitting the model.")
        except:
            if (
                "init_method" in parameters
                and not (isinstance(parameters["init_method"], str))
                and self._model_type in ("KMeans", "BisectingKMeans", "KPrototypes",)
            ):
                drop(name_init, method="table")
            raise
        finally:
            if tmp_view:
                drop(relation, method="view")
        if self._model_type in ("KMeans", "BisectingKMeans", "KPrototypes",):
            if "init_method" in parameters and not (
                isinstance(parameters["init_method"], str)
            ):
                drop(name_init, method="table")
        self._compute_attributes()
        if self._is_native:
            return self.summarize()
        else:
            return None
