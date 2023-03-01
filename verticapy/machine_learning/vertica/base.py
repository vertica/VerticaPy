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
from typing import Any, Literal, Optional, Union, get_type_hints
from abc import abstractmethod
from collections.abc import Iterable
import numpy as np

import verticapy._config.config as conf
from verticapy._utils._gen import gen_name, gen_tmp_name
from verticapy._utils._sql._format import clean_query, quote_ident, schema_relation
from verticapy._utils._sql._sys import _executeSQL
from verticapy._utils._sql._vertica_version import (
    check_minimum_version,
    vertica_version,
)
from verticapy._typing import ArrayLike, PythonScalar
from verticapy.errors import (
    ConversionError,
    FunctionError,
    ParameterError,
    ModelError,
    VersionError,
)

from verticapy.core.tablesample.base import TableSample
from verticapy.core.vdataframe.base import vDataFrame

from verticapy.plotting._matplotlib.mlplot import (
    plot_importance,
    regression_tree_plot,
    plot_pca_circle,
)

from verticapy.machine_learning._utils import get_match_index
import verticapy.machine_learning.metrics as mt
from verticapy.machine_learning.model_management.read import does_model_exist
import verticapy.machine_learning.model_selection as ms

from verticapy.sql.drop import drop

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


class VerticaModel:
    """
Base Class for Vertica Models.
	"""

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

    def __repr__(self):
        """
	Returns the model Representation.
		"""
        return f"<{self._model_type}>"

    def summarize(self):
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

    def contour(
        self, nbins: int = 100, ax=None, **style_kwds,
    ):
        """
        Draws the model's contour plot.

        Parameters
        ----------
        nbins: int, optional
             Number of bins used to discretize the two predictors.
        ax: Matplotlib axes object, optional
            The axes to plot on.
        **style_kwds
            Any optional parameter to pass to the Matplotlib 
            functions.

        Returns
        -------
        ax
            Matplotlib axes object
        """
        return vDataFrame(self.input_relation).contour(
            self.X,
            self.deploySQL(X=self.X),
            cbar_title=self.y,
            nbins=nbins,
            ax=ax,
            **style_kwds,
        )

    def deploySQL(self, X: Union[str, list] = []):
        """
	Returns the SQL code needed to deploy the model. 

	Parameters
	----------
	X: str / list, optional
		List of the columns used to deploy the model. If empty, the model
		predictors will be used.

	Returns
	-------
	str
		the SQL code needed to deploy the model.
		"""
        if isinstance(X, str):
            X = [X]
        if self._model_type == "AutoML":
            return self.best_model_.deploySQL(X)
        if self._model_type not in ("DBSCAN", "LocalOutlierFactor"):
            X = self.X if not (X) else [quote_ident(predictor) for predictor in X]
            sql = f"""
                {self._vertica_predict_sql}({', '.join(X)} 
                                                    USING PARAMETERS 
                                                    model_name = '{self.model_name}',
                                                    match_by_pos = 'true')"""
            return clean_query(sql)
        else:
            raise FunctionError(
                f"Method 'deploySQL' for '{self._model_type}' doesn't exist."
            )

    def drop(self):
        """
	Drops the model from the Vertica database.
		"""
        drop(self.model_name, method="model")

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

    def get_vertica_attributes(self, attr_name: str = ""):
        """
	Returns the model vertica attributes. Those are stored in Vertica.

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

    def get_params(self):
        """
	Returns the parameters of the model.

	Returns
	-------
	dict
		model parameters
		"""
        all_init_params = list(get_type_hints(self.__init__).keys())
        parameters = copy.deepcopy(self.parameters)
        parameters_keys = list(parameters.keys())
        for p in parameters_keys:
            if p not in all_init_params:
                del parameters[p]
        return parameters

    def _get_vertica_param_dict(self):
        """
    Returns the Vertica parameters dict to use when fitting the
    model. As some model's parameters names are not the same in
    Vertica. It is important to map them.

    Returns
    -------
    dict
        vertica parameters
        """

        def map_to_vertica_param_name(param: str):
            param = param.lower()
            options = {
                "class_weights": "class_weight",
                "solver": "optimizer",
                "tol": "epsilon",
                "max_iter": "max_iterations",
                "penalty": "regularization",
                "C": "lambda",
                "l1_ratio": "alpha",
                "n_estimators": "ntree",
                "max_features": "mtry",
                "sample": "sampling_size",
                "max_leaf_nodes": "max_breadth",
                "min_samples_leaf": "min_leaf_size",
                "n_components": "num_components",
                "init": "init_method",
            }
            if param in options:
                return options[param]
            return param

        parameters = {}

        for param in self.parameters:

            if self._model_type in ("LinearSVC", "LinearSVR") and param == "C":
                parameters[param] = self.parameters[param]

            elif (
                self._model_type in ("LinearRegression", "LogisticRegression")
                and param == "C"
            ):
                parameters["lambda"] = self.parameters[param]

            elif self._model_type == "BisectingKMeans" and param in (
                "init",
                "max_iter",
                "tol",
            ):
                if param == "init":
                    parameters["kmeans_center_init_method"] = (
                        "'" + self.parameters[param] + "'"
                    )
                elif param == "max_iter":
                    parameters["kmeans_max_iterations"] = self.parameters[param]
                elif param == "tol":
                    parameters["kmeans_epsilon"] = self.parameters[param]

            elif param == "max_leaf_nodes":
                parameters[map_to_vertica_param_name(param)] = int(
                    self.parameters[param]
                )

            elif param == "class_weight":
                if isinstance(self.parameters[param], Iterable):
                    parameters[
                        "class_weights"
                    ] = f"'{', '.join([str(p) for p in self.parameters[param]])}'"
                else:
                    parameters["class_weights"] = f"'{self.parameters[param]}'"

            elif isinstance(self.parameters[param], (str, dict)):
                parameters[
                    map_to_vertica_param_name(param)
                ] = f"'{self.parameters[param]}'"

            else:
                parameters[map_to_vertica_param_name(param)] = self.parameters[param]

        return parameters

    def set_params(self, parameters: dict = {}):
        """
	Sets the parameters of the model.

	Parameters
	----------
	parameters: dict, optional
		New parameters.
		"""
        all_init_params = list(get_type_hints(self.__init__).keys())
        new_parameters = copy.deepcopy(self.parameters)
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

    def to_python(
        self, return_proba: bool = False, return_distance_clusters: bool = False,
    ):
        """
    Returns the Python function needed to do in-memory scoring 
    without using built-in Vertica functions.

    Parameters
    ----------
    return_proba: bool, optional
        If set to True and the model is a classifier, the function
        returns the model probabilities.
    return_distance_clusters: bool, optional
        If set to True and the model is cluster-based, 
        the function returns the model clusters distances. If the model
        is KPrototypes, the function returns the dissimilarity function.


    Returns
    -------
    str / func
        Python function
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
    ):
        """
    Returns the SQL code needed to deploy the model without using built-in 
    Vertica functions.

    Parameters
    ----------
    X: list, optional
        input predictors name.
    return_proba: bool, optional
        If set to True and the model is a classifier, the function will return 
        the class probabilities.
    return_distance_clusters: bool, optional
        If set to True and the model is cluster-based, 
        the function returns the model clusters distances. If the model
        is KPrototypes, the function returns the dissimilarity function.

    Returns
    -------
    str / list
        SQL code
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


class Supervised(VerticaModel):
    @property
    @abstractmethod
    def _vertica_predict_sql(self) -> str:
        """Must be overridden in child class"""
        raise NotImplementedError

    def fit(
        self,
        input_relation: Union[str, vDataFrame],
        X: Union[str, list],
        y: str,
        test_relation: Union[str, vDataFrame] = "",
    ):
        """
	Trains the model.

	Parameters
	----------
	input_relation: str / vDataFrame
		Training relation.
	X: str / list
		List of the predictors.
	y: str
		Response column.
	test_relation: str / vDataFrame, optional
		Relation used to test the model.

	Returns
	-------
	object
		model
		"""
        if isinstance(X, str):
            X = [X]
        if conf.get_option("overwrite_model"):
            self.drop()
        else:
            does_model_exist(name=self.model_name, raise_error=True)
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
        return self


class Tree:
    def _compute_features_importance(self, tree_id: Optional[int] = None) -> None:
        """
        Computes the features importance.
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

    def _compute_trees_arrays(
        self, tree: TableSample, X: list, return_probability: bool = False
    ):
        """
        Takes as input a tree which is represented by a TableSample
        It returns a list of arrays. Each index of the arrays represents
        a node value.
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

    def plot(self, max_nb_points: int = 100, ax=None, **style_kwds):
        """
    Draws the model.

    Parameters
    ----------
    max_nb_points: int
        Maximum number of points to display.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the 
        Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object
        """
        if self._model_subcategory == "REGRESSOR":
            return regression_tree_plot(
                self.X + [self.deploySQL()],
                self.y,
                self.input_relation,
                max_nb_points,
                ax=ax,
                **style_kwds,
            )
        else:
            raise NotImplementedError

    def features_importance(
        self, tree_id: Optional[int] = None, show: bool = True, ax=None, **style_kwds
    ) -> TableSample:
        """
        Computes the model's features importance.

        Parameters
        ----------
        tree_id: int
            Tree ID.
        show: bool
            If set to True, draw the features importance.
        ax: Matplotlib axes object, optional
            The axes to plot on.
        **style_kwds
            Any optional parameter to pass to the Matplotlib 
            functions.

        Returns
        -------
        TableSample
            An object containing the result. For more information,
            see utilities.TableSample.
        """
        fi = self._get_features_importance(tree_id=tree_id)
        if show:
            plot_importance(
                self.X, fi, print_legend=False, ax=ax, **style_kwds,
            )
        importances = {
            "index": [quote_ident(x)[1:-1].lower() for x in self.X],
            "importance": list(abs(fi)),
            "sign": list(np.sign(fi)),
        }
        return TableSample(values=importances).sort(column="importance", desc=True)

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
    ):
        """
        Returns the code for a Graphviz tree.

        Parameters
        ----------
        tree_id: int, optional
            Unique tree identifier, an integer in the range 
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
            Dictionary of options to customize each node of 
            the tree. For a list of options, see the Graphviz 
            API: https://graphviz.org/doc/info/attrs.html
        arrow_style: dict, optional
            Dictionary of options to customize each arrow of 
            the tree. For a list of options, see the Graphviz 
            API: https://graphviz.org/doc/info/attrs.html
        leaf_style: dict, optional
            Dictionary of options to customize each leaf of 
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

    @check_minimum_version
    def get_tree(self, tree_id: int = 0):
        """
	Returns a table with all the input tree information.

	Parameters
	----------
	tree_id: int, optional
        Unique tree identifier, an integer in the range [0, n_estimators - 1].

	Returns
	-------
	TableSample
		An object containing the result. For more information, see
		utilities.TableSample.
		"""
        query = f"""
            SELECT * FROM (SELECT READ_TREE (
                             USING PARAMETERS 
                             model_name = '{self.model_name}', 
                             tree_id = {tree_id}, 
                             format = 'tabular')) x ORDER BY node_id;"""
        return TableSample.read_sql(query=query, title="Reading Tree.")

    def plot_tree(
        self, tree_id: int = 0, pic_path: str = "", *argv, **kwds,
    ):
        """
        Draws the input tree. Requires the graphviz module.

        Parameters
        ----------
        tree_id: int, optional
            Unique tree identifier, an integer in the range 
            [0, n_estimators - 1].
        pic_path: str, optional
            Absolute path to save the image of the tree.
        *argv, **kwds: Any, optional
            Arguments to pass to the 'to_graphviz' method.

        Returns
        -------
        graphviz.Source
            graphviz object.
        """
        return self.trees_[tree_id].plot_tree(
            pic_path=pic_path, feature_names=self.X, *argv, **kwds,
        )

    def get_score(
        self, tree_id: int = None,
    ):
        """
        Returns the feature importance metrics for the input tree.

        Parameters
        ----------
        tree_id: int, optional
            Unique tree identifier, an integer in the range 
            [0, n_estimators - 1]. If tree_id is undefined, 
            all the trees in the model are used to compute 
            the metrics.

        Returns
        -------
        TableSample
            An object containing the result. For more information, 
            see utilities.TableSample.
        """
        tree_id = "" if tree_id == None else f", tree_id={tree_id}"
        query = f"""
            SELECT {self._model_importance_function} 
            (USING PARAMETERS model_name = '{self.model_name}'{tree_id})"""
        return TableSample.read_sql(query=query, title="Reading Tree.")


class Classifier(Supervised):
    pass


class BinaryClassifier(Classifier):

    classes_ = np.array([0, 1])

    def classification_report(
        self, cutoff: Union[int, float] = 0.5, nbins: int = 10000,
    ):
        """
	Computes a classification report using multiple metrics to evaluate the model
	(AUC, accuracy, PRC AUC, F1...). 

	Parameters
	----------
	cutoff: int / float, optional
		Probability cutoff.
    nbins: int, optional
        [Used to compute ROC AUC, PRC AUC and the best cutoff]
        An integer value that determines the number of decision boundaries. 
        Decision boundaries are set at equally spaced intervals between 0 and 1, 
        inclusive. Greater values for nbins give more precise estimations of the 
        metrics, but can potentially decrease performance. The maximum value 
        is 999,999. If negative, the maximum value is used.

	Returns
	-------
	TableSample
		An object containing the result. For more information, see
		utilities.TableSample.
		"""
        if cutoff > 1 or cutoff < 0:
            cutoff = self.score(method="best_cutoff")
        return mt.classification_report(
            self.y,
            [self.deploySQL(), self.deploySQL(cutoff)],
            self.test_relation,
            cutoff=cutoff,
            nbins=nbins,
        )

    report = classification_report

    def confusion_matrix(self, cutoff: Union[int, float] = 0.5):
        """
	Computes the model confusion matrix.

	Parameters
	----------
	cutoff: int / float, optional
		Probability cutoff.

	Returns
	-------
	TableSample
		An object containing the result. For more information, see
		utilities.TableSample.
		"""
        return mt.confusion_matrix(self.y, self.deploySQL(cutoff), self.test_relation,)

    def deploySQL(self, cutoff: Union[int, float] = -1, X: Union[str, list] = []):
        """
	Returns the SQL code needed to deploy the model. 

	Parameters
	----------
	cutoff: int / float, optional
		Probability cutoff. If this number is not between 0 and 1, the method 
		will return the probability to be of class 1.
	X: str / list, optional
		List of the columns used to deploy the model. If empty, the model
		predictors will be used.

	Returns
	-------
	str
		the SQL code needed to deploy the model.
		"""
        if not (X):
            X = self.X
        elif isinstance(X, str):
            X = [X]
        else:
            X = [quote_ident(elem) for elem in X]
        sql = f"""
            {self._vertica_predict_sql}
            ({', '.join(X)} USING PARAMETERS
                            model_name = '{self.model_name}',
                            type = 'probability',
                            match_by_pos = 'true')"""
        if cutoff <= 1 and cutoff >= 0:
            sql = f"""
                (CASE 
                    WHEN {sql} >= {cutoff} 
                        THEN 1 
                    WHEN {sql} IS NULL 
                        THEN NULL 
                    ELSE 0 
                END)"""
        return clean_query(sql)

    def lift_chart(self, ax=None, nbins: int = 1000, **style_kwds):
        """
	Draws the model Lift Chart.

    Parameters
    ----------
    ax: Matplotlib axes object, optional
        The axes to plot on.
    nbins: int, optional
        The number of bins.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

	Returns
	-------
	TableSample
		An object containing the result. For more information, see
		utilities.TableSample.
		"""
        return ms.lift_chart(
            self.y,
            self.deploySQL(),
            self.test_relation,
            ax=ax,
            nbins=nbins,
            **style_kwds,
        )

    def prc_curve(self, ax=None, nbins: int = 30, **style_kwds):
        """
	Draws the model PRC curve.

    Parameters
    ----------
    ax: Matplotlib axes object, optional
        The axes to plot on.
    nbins: int, optional
        The number of bins.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

	Returns
	-------
	TableSample
		An object containing the result. For more information, see
		utilities.TableSample.
		"""
        return ms.prc_curve(
            self.y,
            self.deploySQL(),
            self.test_relation,
            ax=ax,
            nbins=nbins,
            **style_kwds,
        )

    def predict(
        self,
        vdf: Union[str, vDataFrame],
        X: Union[str, list] = [],
        name: str = "",
        cutoff: Union[int, float] = 0.5,
        inplace: bool = True,
    ):
        """
	Predicts using the input relation.

	Parameters
	----------
	vdf: str / vDataFrame
		Object to use to run the prediction. You can also specify a customized 
        relation, but you must enclose it with an alias. For example, 
        "(SELECT 1) x" is correct, whereas "(SELECT 1)" and "SELECT 1" are 
        incorrect.
	X: str / list, optional
		List of the columns used to deploy the models. If empty, the model
		predictors will be used.
	name: str, optional
		Name of the added vcolumn. If empty, a name will be generated.
	cutoff: float, optional
		Probability cutoff.
	inplace: bool, optional
		If set to True, the prediction will be added to the vDataFrame.

	Returns
	-------
	vDataFrame
		the input object.
		"""
        # Inititalization
        if isinstance(X, str):
            X = [X]
        assert 0 <= cutoff <= 1, ParameterError(
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
        return vdf_return.eval(name, self.deploySQL(cutoff=cutoff, X=X))

    def predict_proba(
        self,
        vdf: Union[str, vDataFrame],
        X: Union[str, list] = [],
        name: str = "",
        pos_label: Union[str, int, float] = None,
        inplace: bool = True,
    ):
        """
    Returns the model's probabilities using the input relation.

    Parameters
    ----------
    vdf: str / vDataFrame
        Object to use to run the prediction. You can also specify a customized 
        relation, but you must enclose it with an alias. For example, 
        "(SELECT 1) x" is correct whereas, "(SELECT 1)" and "SELECT 1" are 
        incorrect.
    X: str / list, optional
        List of the columns used to deploy the models. If empty, the model
        predictors will be used.
    name: str, optional
        Name of the added vcolumn. If empty, a name will be generated.
    pos_label: int / float / str, optional
        Class label. For binary classification, this can be either 1 or 0.
    inplace: bool, optional
        If set to True, the prediction will be added to the vDataFrame.

    Returns
    -------
    vDataFrame
        the input object.
        """
        # Inititalization
        if isinstance(X, str):
            X = [X]
        assert pos_label in [1, 0, "0", "1", None], ParameterError(
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

    def cutoff_curve(self, ax=None, nbins: int = 30, **style_kwds):
        """
    Draws the model Cutoff curve.

    Parameters
    ----------
    ax: Matplotlib axes object, optional
        The axes to plot on.
    nbins: int, optional
        The number of bins.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    TableSample
        An object containing the result. For more information, see
        utilities.TableSample.
        """
        return ms.roc_curve(
            self.y,
            self.deploySQL(),
            self.test_relation,
            ax=ax,
            cutoff_curve=True,
            nbins=nbins,
            **style_kwds,
        )

    def roc_curve(self, ax=None, nbins: int = 30, **style_kwds):
        """
	Draws the model ROC curve.

    Parameters
    ----------
    ax: Matplotlib axes object, optional
        The axes to plot on.
    nbins: int, optional
        The number of bins.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

	Returns
	-------
	TableSample
		An object containing the result. For more information, see
		utilities.TableSample.
		"""
        return ms.roc_curve(
            self.y,
            self.deploySQL(),
            self.test_relation,
            ax=ax,
            nbins=nbins,
            **style_kwds,
        )

    def score(
        self,
        method: Literal[tuple(mt.FUNCTIONS_CLASSIFICATION_DICTIONNARY)] = "accuracy",
        cutoff: Union[int, float] = 0.5,
        nbins: int = 10000,
    ):
        """
	Computes the model score.

	Parameters
	----------
	method: str, optional
		The method to use to compute the score.
			accuracy	: Accuracy
            aic         : Akaike’s Information Criterion
			auc		    : Area Under the Curve (ROC)
			best_cutoff : Cutoff which optimised the ROC Curve prediction.
            bic         : Bayesian Information Criterion
			bm		    : Informedness = tpr + tnr - 1
			csi		    : Critical Success Index = tp / (tp + fn + fp)
			f1		    : F1 Score 
			logloss	    : Log Loss
			mcc		    : Matthews Correlation Coefficient 
			mk		    : Markedness = ppv + npv - 1
			npv		    : Negative Predictive Value = tn / (tn + fn)
			prc_auc	    : Area Under the Curve (PRC)
			precision   : Precision = tp / (tp + fp)
			recall	    : Recall = tp / (tp + fn)
			specificity : Specificity = tn / (tn + fp)
	cutoff: int / float, optional
		Cutoff for which the tested category will be accepted as a prediction.
    nbins: int, optional
        [Only when method is set to auc|prc_auc|best_cutoff] 
        An integer value that determines the number of decision boundaries. 
        Decision boundaries are set at equally spaced intervals between 0 and 1, 
        inclusive. Greater values for nbins give more precise estimations of the AUC, 
        but can potentially decrease performance. The maximum value is 999,999. 
        If negative, the maximum value is used.

	Returns
	-------
	float
		score
		"""
        fun = mt.FUNCTIONS_CLASSIFICATION_DICTIONNARY[method]
        if method in (
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
            args2 = self.deploySQL(cutoff)
        args = [self.y, args2, self.test_relation]
        kwds = {}
        if method in ("accuracy", "acc"):
            kwds["pos_label"] = 1
        elif method in ("aic", "bic"):
            args += [len(self.X)]
        elif method in ("prc_auc", "auc", "best_cutoff", "best_threshold"):
            kwds["nbins"] = nbins
            if method in ("best_cutoff", "best_threshold"):
                kwds["best_threshold"] = True
        return fun(*args, **kwds)


class MulticlassClassifier(Classifier):
    @staticmethod
    def _array_to_int(object_: np.ndarray):
        res = copy.deepcopy(object_)
        try:
            return res.astype(int)
        except ValueError:
            return res

    def _is_binary_classifier(self):
        if len(self.classes_) == 2 and self.classes_[0] == 0 and self.classes_[1] == 1:
            return True
        return False

    def _get_classes(self):
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

    def contour(
        self, pos_label: PythonScalar = None, nbins: int = 100, ax=None, **style_kwds,
    ):
        """
        Draws the model's contour plot.

        Parameters
        ----------
        pos_label: PythonScalar, optional
            Label to consider as positive. All the other 
            classes will be merged and considered as negative 
            for multiclass classification.
        nbins: int, optional
             Number of bins used to discretize the two predictors.
        ax: Matplotlib axes object, optional
            The axes to plot on.
        **style_kwds
            Any optional parameter to pass to the Matplotlib 
            functions.

        Returns
        -------
        ax
            Matplotlib axes object
        """
        if not (pos_label):
            pos_label = sorted(self.classes_)[-1]
        return vDataFrame(self.input_relation).contour(
            self.X,
            self.deploySQL(X=self.X, pos_label=pos_label),
            cbar_title=self.y,
            nbins=nbins,
            ax=ax,
            **style_kwds,
        )

    def classification_report(
        self,
        cutoff: Union[int, float, list] = [],
        labels: Union[str, list] = [],
        nbins: int = 10000,
    ):
        """
	Computes a classification report using multiple metrics to evaluate the model
	(AUC, accuracy, PRC AUC, F1...). For multiclass classification, it will consider 
    each category as positive and switch to the next one during the computation.

	Parameters
	----------
	cutoff: int / float / list, optional
		Cutoff for which the tested category will be accepted as a prediction.
		For multiclass classification, each tested category becomes 
		the positives and the others are merged into the negatives. The list will 
		represent the classes threshold. If it is empty, the best cutoff will be used.
	labels: str / list, optional
		List of the different labels to be used during the computation.
    nbins: int, optional
        [Used to compute ROC AUC, PRC AUC and the best cutoff]
        An integer value that determines the number of decision boundaries. 
        Decision boundaries are set at equally spaced intervals between 0 and 1, 
        inclusive. Greater values for nbins give more precise estimations of the 
        metrics, but can potentially decrease performance. The maximum value 
        is 999,999. If negative, the maximum value is used.

	Returns
	-------
	TableSample
		An object containing the result. For more information, see
		utilities.TableSample.
		"""
        if not (labels):
            labels = self.classes_
        elif isinstance(labels, str):
            labels = [labels]
        return mt.classification_report(
            cutoff=cutoff, estimator=self, labels=labels, nbins=nbins,
        )

    report = classification_report

    def confusion_matrix(
        self, pos_label: PythonScalar = None, cutoff: Union[int, float] = -1,
    ):
        """
	Computes the model confusion matrix.

	Parameters
	----------
	pos_label: int / float / str, optional
		Label to consider as positive. All the other classes will be merged and
		considered as negative for multiclass classification.
	cutoff: int / float, optional
		Cutoff for which the tested category will be accepted as a prediction.If the 
		cutoff is not between 0 and 1, the entire confusion matrix will be drawn.

	Returns
	-------
	TableSample
		An object containing the result. For more information, see
		utilities.TableSample.
		"""
        if pos_label == None and len(self.classes_) == 2:
            pos_label = self.classes_[1]
        elif pos_label:
            return mt.confusion_matrix(
                self.y,
                self.deploySQL(pos_label, cutoff),
                self.test_relation,
                pos_label=pos_label,
            )
        else:
            return mt.multilabel_confusion_matrix(
                self.y, self.deploySQL(), self.test_relation, self.classes_
            )

    def cutoff_curve(
        self, pos_label: PythonScalar = None, ax=None, nbins: int = 30, **style_kwds,
    ):
        """
    Draws the model Cutoff curve.

    Parameters
    ----------
    pos_label: PythonScalar, optional
        To draw the ROC curve, one of the response column classes must be the 
        positive one. The parameter 'pos_label' represents this class.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    nbins: int, optional
        An integer value that determines the number of decision boundaries. Decision 
        boundaries are set at equally-spaced intervals between 0 and 1, inclusive.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    TableSample
        An object containing the result. For more information, see
        utilities.TableSample.
        """
        if pos_label == None and len(self.classes_) == 2:
            pos_label = self.classes_[1]
        elif pos_label not in self.classes_:
            raise ParameterError(
                "'pos_label' must be one of the response column classes"
            )
        if self._model_type == "NearestCentroid":
            deploySQL_str = self.deploySQL(allSQL=True)[
                get_match_index(pos_label, self.classes_, False)
            ]
        else:
            deploySQL_str = self.deploySQL(allSQL=True)[0].format(pos_label)
        return ms.roc_curve(
            self.y,
            deploySQL_str,
            self.test_relation,
            pos_label,
            ax=ax,
            cutoff_curve=True,
            nbins=nbins,
            **style_kwds,
        )

    def deploySQL(
        self,
        pos_label: PythonScalar = None,
        cutoff: Union[int, float] = -1,
        allSQL: bool = False,
        X: Union[str, list] = [],
    ):
        """
	Returns the SQL code needed to deploy the model. 

	Parameters
	----------
	pos_label: int / float / str, optional
		Label to consider as positive. All the other classes will be merged and
		considered as negative for multiclass classification.
	cutoff: int / float, optional
		Cutoff for which the tested category will be accepted as a prediction.If 
		the cutoff is not between 0 and 1, a probability will be returned.
	allSQL: bool, optional
		If set to True, the output will be a list of the different SQL codes 
		needed to deploy the different categories score.
	X: str / list, optional
		List of the columns used to deploy the model. If empty, the model
		predictors will be used.

	Returns
	-------
	str / list
		the SQL code needed to deploy the self.
		"""
        if not (X):
            X = self.X
        elif isinstance(X, str):
            X = [X]
        else:
            X = [quote_ident(x) for x in X]
        fun = self._vertica_predict_sql

        if self._model_type == "NearestCentroid":
            sql = self.to_memmodel().predict_proba_sql(X)
        else:
            sql = [
                f"""
                {fun}({', '.join(X)} 
                      USING PARAMETERS 
                      model_name = '{self.model_name}',
                      class = '{{}}',
                      type = 'probability',
                      match_by_pos = 'true')""",
                f"""
                    {fun}({', '.join(X)} 
                          USING PARAMETERS 
                          model_name = '{self.model_name}',
                          match_by_pos = 'true')""",
            ]
        if not (allSQL):
            if pos_label in self.classes_:
                if self._model_type == "NearestCentroid":
                    sql = sql[get_match_index(pos_label, self.classes_, False)]
                else:
                    sql = sql[0].format(pos_label)
            if pos_label in self.classes_ and cutoff <= 1 and cutoff >= 0:
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
            elif pos_label not in self.classes_:
                if self._model_type == "NearestCentroid":
                    sql = self.to_memmodel().predict_sql(X)
                else:
                    sql = sql[1]
        if isinstance(sql, str):
            sql = clean_query(sql)
        else:
            sql = [clean_query(q) for q in sql]
        return sql

    def lift_chart(
        self, pos_label: PythonScalar = None, ax=None, nbins: int = 1000, **style_kwds,
    ):
        """
	Draws the model Lift Chart.

	Parameters
	----------
	pos_label: PythonScalar, optional
		To draw a lift chart, one of the response column classes must be the
		positive one. The parameter 'pos_label' represents this class.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    nbins: int, optional
        An integer value that determines the number of decision boundaries. Decision 
        boundaries are set at equally-spaced intervals between 0 and 1, inclusive.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

	Returns
	-------
	TableSample
		An object containing the result. For more information, see
		utilities.TableSample.
		"""
        if pos_label == None and len(self.classes_) == 2:
            pos_label = self.classes_[1]
        if pos_label not in self.classes_:
            raise ParameterError(
                "'pos_label' must be one of the response column classes"
            )
        if self._model_type == "NearestCentroid":
            deploySQL_str = self.deploySQL(allSQL=True)[
                get_match_index(pos_label, self.classes_, False)
            ]
        else:
            deploySQL_str = self.deploySQL(allSQL=True)[0].format(pos_label)
        return ms.lift_chart(
            self.y,
            deploySQL_str,
            self.test_relation,
            pos_label,
            ax=ax,
            nbins=nbins,
            **style_kwds,
        )

    def prc_curve(
        self, pos_label: PythonScalar = None, ax=None, nbins: int = 30, **style_kwds,
    ):
        """
	Draws the model PRC curve.

	Parameters
	----------
	pos_label: PythonScalar, optional
		To draw the PRC curve, one of the response column classes must be the 
		positive one. The parameter 'pos_label' represents this class.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    nbins: int, optional
        An integer value that determines the number of decision boundaries. Decision 
        boundaries are set at equally-spaced intervals between 0 and 1, inclusive.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

	Returns
	-------
	TableSample
		An object containing the result. For more information, see
		utilities.TableSample.
		"""
        if pos_label == None and len(self.classes_) == 2:
            pos_label = self.classes_[1]
        if pos_label not in self.classes_:
            raise ParameterError(
                "'pos_label' must be one of the response column classes"
            )
        if self._model_type == "NearestCentroid":
            deploySQL_str = self.deploySQL(allSQL=True)[
                get_match_index(pos_label, self.classes_, False)
            ]
        else:
            deploySQL_str = self.deploySQL(allSQL=True)[0].format(pos_label)
        return ms.prc_curve(
            self.y,
            deploySQL_str,
            self.test_relation,
            pos_label,
            ax=ax,
            nbins=nbins,
            **style_kwds,
        )

    def predict(
        self,
        vdf: Union[str, vDataFrame],
        X: Union[str, list] = [],
        name: str = "",
        cutoff: Union[int, float] = 0.5,
        inplace: bool = True,
    ):
        """
	Predicts using the input relation.

	Parameters
	----------
	vdf: str / vDataFrame
		Object to use to run the prediction. You can also specify a customized 
        relation, but you must enclose it with an alias. For example, 
        "(SELECT 1) x" is correct, whereas "(SELECT 1)" and "SELECT 1" are 
        incorrect.
	X: str / list, optional
		List of the columns used to deploy the models. If empty, the model
		predictors will be used.
	name: str, optional
		Name of the added vcolumn. If empty, a name will be generated.
	cutoff: int / float, optional
		Cutoff for which the tested category will be accepted as a prediction.
		This parameter is only used for binary classification.
	inplace: bool, optional
		If set to True, the prediction will be added to the vDataFrame.

	Returns
	-------
	vDataFrame
		the input object.
		"""
        # Inititalization
        if not (X):
            X = self.X
        elif isinstance(X, str):
            X = [X]
        else:
            X = [quote_ident(elem) for elem in X]
        if not (name):
            name = gen_name([self._model_type, self.model_name])
        assert 0 <= cutoff <= 1, ParameterError(
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
            name, self.deploySQL(pos_label=pos_label, cutoff=cutoff, X=X)
        )

    def predict_proba(
        self,
        vdf: Union[str, vDataFrame],
        X: Union[str, list] = [],
        name: str = "",
        pos_label: Union[int, str, float] = None,
        inplace: bool = True,
    ):
        """
    Returns the model's probabilities using the input relation.

    Parameters
    ----------
    vdf: str / vDataFrame
        Object to use to run the prediction. You can also specify a customized 
        relation, but you must enclose it with an alias. For example, 
        "(SELECT 1) x" is correct, whereas "(SELECT 1)" and "SELECT 1" are 
        incorrect.
    X: str / list, optional
        List of the columns used to deploy the models. If empty, the model
        predictors will be used.
    name: str, optional
        Name of the additional prediction vDataColumn. If unspecified, a name is 
	    generated based on the model and class names.
    pos_label: int / float / str, optional
        Class label, the class for which the probability is calculated. 
	    If name is specified and pos_label is unspecified, the probability 
        column names use the following format: name_class1, name_class2, etc.
    inplace: bool, optional
        If set to True, the prediction will be added to the vDataFrame.

    Returns
    -------
    vDataFrame
        the input object.
        """
        # Inititalization
        if not (X):
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
                vdf_return.eval(name_tmp, self.deploySQL(pos_label=c, cutoff=-1, X=X))
        else:
            vdf_return.eval(name, self.deploySQL(pos_label=pos_label, cutoff=-1, X=X))

        return vdf_return

    def roc_curve(
        self, pos_label: PythonScalar = None, ax=None, nbins: int = 30, **style_kwds,
    ):
        """
	Draws the model ROC curve.

	Parameters
	----------
	pos_label: PythonScalar, optional
		To draw the ROC curve, one of the response column classes must be the
		positive one. The parameter 'pos_label' represents this class.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    nbins: int, optional
        An integer value that determines the number of decision boundaries. Decision 
        boundaries are set at equally-spaced intervals between 0 and 1, inclusive.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

	Returns
	-------
	TableSample
		An object containing the result. For more information, see
		utilities.TableSample.
		"""
        if pos_label == None and len(self.classes_) == 2:
            pos_label = self.classes_[1]
        if pos_label not in self.classes_:
            raise ParameterError(
                "'pos_label' must be one of the response column classes"
            )
        if self._model_type == "NearestCentroid":
            deploySQL_str = self.deploySQL(allSQL=True)[
                get_match_index(pos_label, self.classes_, False)
            ]
        else:
            deploySQL_str = self.deploySQL(allSQL=True)[0].format(pos_label)
        return ms.roc_curve(
            self.y,
            deploySQL_str,
            self.test_relation,
            pos_label,
            ax=ax,
            nbins=nbins,
            **style_kwds,
        )

    def score(
        self,
        method: Literal[tuple(mt.FUNCTIONS_CLASSIFICATION_DICTIONNARY)] = "accuracy",
        pos_label: PythonScalar = None,
        cutoff: Union[int, float] = 0.5,
        nbins: int = 10000,
    ):
        """
	Computes the model score.

	Parameters
	----------
	pos_label: int / float / str, optional
		Label to consider as positive. All the other classes will be merged and
		considered as negative for multiclass classification.
	cutoff: int / float, optional
		Cutoff for which the tested category will be accepted as a prediction.
	method: str, optional
		The method to use to compute the score.
			accuracy	: Accuracy
			auc		    : Area Under the Curve (ROC)
			best_cutoff : Cutoff which optimised the ROC Curve prediction.
			bm		    : Informedness = tpr + tnr - 1
			csi		    : Critical Success Index = tp / (tp + fn + fp)
			f1		    : F1 Score 
			logloss	    : Log Loss
			mcc		    : Matthews Correlation Coefficient 
			mk		    : Markedness = ppv + npv - 1
			npv		    : Negative Predictive Value = tn / (tn + fn)
			prc_auc	    : Area Under the Curve (PRC)
			precision   : Precision = tp / (tp + fp)
			recall	    : Recall = tp / (tp + fn)
			specificity : Specificity = tn / (tn + fp)
    nbins: int, optional
        [Only used when the method is set to 'auc,' 'prc_auc,' or 'best_cutoff']
        An integer value that determines the number of decision boundaries. Decision 
        boundaries are set at equally-spaced intervals between 0 and 1, inclusive.
        The greater number of decision boundaries, the greater precision, but  
        the greater decrease in performance. Maximum value: 999,999. If negative, the
        maximum value is used.

	Returns
	-------
	float
		score
		"""
        fun = mt.FUNCTIONS_CLASSIFICATION_DICTIONNARY[method]
        if pos_label == None and len(self.classes_) == 2:
            pos_label = self.classes_[1]
        if (pos_label not in self.classes_) and (method != "accuracy"):
            raise ParameterError(
                "'pos_label' must be one of the response column classes"
            )
        if self._model_type == "NearestCentroid":
            deploySQL_str = self.deploySQL(allSQL=True)[
                get_match_index(pos_label, self.classes_, False)
            ]
        else:
            deploySQL_str = self.deploySQL(allSQL=True)[0].format(pos_label)
        args = [self.y, self.deploySQL(pos_label, cutoff), self.test_relation]
        kwds = {}
        if method in ("accuracy", "acc"):
            args += [pos_label]
        elif method in ("aic", "bic"):
            args += [len(self.X)]
        elif method in (
            "auc",
            "prc_auc",
            "best_cutoff",
            "best_threshold",
            "log_loss",
            "logloss",
        ):
            args = [
                f"DECODE({self.y}, '{pos_label}', 1, 0)",
                deploySQL_str,
                self.test_relation,
            ]
            if method in ("auc", "prc_auc", "best_cutoff", "best_threshold"):
                kwds["nbins"] = nbins
            if method in ("best_cutoff", "best_threshold"):
                kwds["best_threshold"] = True
        return fun(*args, **kwds)


class Regressor(Supervised):
    def predict(
        self,
        vdf: Union[str, vDataFrame],
        X: Union[str, list] = [],
        name: str = "",
        inplace: bool = True,
    ):
        """
	Predicts using the input relation.

	Parameters
	----------
	vdf: str / vDataFrame
		Object to use to run the prediction. You can also specify a customized 
        relation, but you must enclose it with an alias. For example "(SELECT 1) x" 
        is correct whereas "(SELECT 1)" and "SELECT 1" are incorrect.
	X: str / list, optional
		List of the columns used to deploy the models. If empty, the model
		predictors will be used.
	name: str, optional
		Name of the added vcolumn. If empty, a name will be generated.
	inplace: bool, optional
		If set to True, the prediction will be added to the vDataFrame.

	Returns
	-------
	vDataFrame
		the input object.
		"""
        if not (X):
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

    def regression_report(
        self, method: Literal["anova", "metrics", "details"] = "metrics"
    ):
        """
	Computes a regression report using multiple metrics to evaluate the model
	(r2, mse, max error...). 

    Parameters
    ----------
    method: str, optional
        The method to use to compute the regression report.
            anova   : Computes the model ANOVA table.
            details : Computes the model details.
            metrics : Computes the model different metrics.

	Returns
	-------
	TableSample
		An object containing the result. For more information, see
		utilities.TableSample.
		"""
        if method in ("anova", "details") and self._model_type in ("KernelDensity",):
            raise ModelError(
                f"'{method}' method is not available for {self._model_type} models."
            )
        prediction = self.deploySQL()
        if self._model_type == "KNeighborsRegressor":
            test_relation = self.deploySQL()
            prediction = "predict_neighbors"
        elif self._model_type == "KernelDensity":
            test_relation = self.map
        else:
            test_relation = self.test_relation
        if method == "metrics":
            return mt.regression_report(self.y, prediction, test_relation, len(self.X))
        elif method == "anova":
            return mt.anova_table(self.y, prediction, test_relation, len(self.X))
        elif method == "details":
            vdf = vDataFrame(f"SELECT {self.y} FROM {self.input_relation}")
            n = vdf[self.y].count()
            kurt = vdf[self.y].kurt()
            skew = vdf[self.y].skew()
            jb = vdf[self.y].agg(["jb"])[self.y][0]
            R2 = self.score()
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

    report = regression_report

    def score(
        self,
        method: Literal[
            tuple(mt.FUNCTIONS_REGRESSION_DICTIONNARY)
            + ("r2a", "r2adj", "r2adjusted", "rmse")
        ] = "r2",
    ):
        """
	Computes the model score.

	Parameters
	----------
	method: str, optional
		The method to use to compute the score.
            aic    : Akaike’s Information Criterion
            bic    : Bayesian Information Criterion
			max	   : Max Error
			mae	   : Mean Absolute Error
			median : Median Absolute Error
			mse	   : Mean Squared Error
			msle   : Mean Squared Log Error
			r2	   : R squared coefficient
            r2a    : R2 adjusted
            rmse   : Root Mean Squared Error
			var	   : Explained Variance 

	Returns
	-------
	float
		score
		"""
        # Initialization
        method = str(method).lower()
        if method in ["r2adj", "r2adjusted"]:
            method = "r2a"
        adj, root = False, False
        if method in ("r2a", "r2adj", "r2adjusted"):
            method, adj = "r2", True
        elif method == "rmse":
            method, root = "mse", True
        fun = mt.FUNCTIONS_REGRESSION_DICTIONNARY[method]

        # Scoring
        if self._model_type == "KNeighborsRegressor":
            test_relation, prediction = self.deploySQL(), "predict_neighbors"
        elif self._model_type == "KernelDensity":
            test_relation, prediction = self.map, self.deploySQL()
        else:
            test_relation, prediction = self.test_relation, self.deploySQL()
        arg = [self.y, prediction, test_relation]
        if method in ("aic", "bic") or adj:
            arg += [len(self.X)]
        if root or adj:
            arg += [True]
        return fun(*arg)


class Unsupervised(VerticaModel):
    def fit(self, input_relation: Union[str, vDataFrame], X: Union[str, list] = []):
        """
	Trains the model.

	Parameters
	----------
	input_relation: str / vDataFrame
		Training relation.
	X: str / list, optional
		List of the predictors. If empty, all the numerical columns will be used.

	Returns
	-------
	object
		model
		"""
        if isinstance(X, str):
            X = [X]
        if conf.get_option("overwrite_model"):
            self.drop()
        else:
            does_model_exist(name=self.model_name, raise_error=True)
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
            if not (X) and (self._model_type == "KPrototypes"):
                X = input_relation.get_columns()
            elif not (X):
                X = input_relation.numcol()
        else:
            self.input_relation = input_relation
            relation = input_relation
            if not (X):
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
                        query0 += ["SELECT /*+LABEL('learn.VerticaModel.fit')*/ " + line]
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
        return self


class Preprocessing(Unsupervised):
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

    def deploySQL(
        self,
        key_columns: Union[str, list] = [],
        exclude_columns: Union[str, list] = [],
        X: Union[str, list] = [],
    ):
        """
    Returns the SQL code needed to deploy the model. 

    Parameters
    ----------
    key_columns: str / list, optional
        Predictors used during the algorithm computation which will be deployed
        with the principal components.
    exclude_columns: str / list, optional
        Columns to exclude from the prediction.
    X: str / list, optional
        List of the columns used to deploy the self. If empty, the model
        predictors will be used.

    Returns
    -------
    str
        the SQL code needed to deploy the model.
        """
        if isinstance(key_columns, str):
            key_columns = [key_columns]
        if isinstance(exclude_columns, str):
            exclude_columns = [exclude_columns]
        if isinstance(X, str):
            X = [X]
        if not (X):
            X = self.X
        else:
            X = [quote_ident(elem) for elem in X]
        if key_columns:
            key_columns = ", ".join([quote_ident(col) for col in key_columns])
        if exclude_columns:
            exclude_columns = ", ".join([quote_ident(col) for col in exclude_columns])
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
            if self.parameters["separator"] == None:
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
                if self.parameters["null_column_name"] == None:
                    null_column_name = "null"
                else:
                    null_column_name = self.parameters["null_column_name"].lower()
                sql += f", null_column_name = '{null_column_name}'"
        sql += ")"
        return clean_query(sql)

    def deployInverseSQL(
        self,
        key_columns: Union[str, list] = [],
        exclude_columns: Union[str, list] = [],
        X: Union[str, list] = [],
    ) -> str:
        """
    Returns the SQL code needed to deploy the inverse model. 

    Parameters
    ----------
    key_columns: str / list, optional
        Predictors used during the algorithm computation which will be deployed
        with the principal components.
    exclude_columns: str / list, optional
        Columns to exclude from the prediction.
    X: str / list, optional
        List of the columns used to deploy the inverse model. If empty, the model
        predictors will be used.

    Returns
    -------
    str
        the SQL code needed to deploy the inverse model.
        """
        if isinstance(key_columns, str):
            key_columns = [key_columns]
        if isinstance(exclude_columns, str):
            exclude_columns = [exclude_columns]
        if not (X):
            X = self.X
        elif isinstance(X, str):
            X = [X]
        else:
            X = [quote_ident(x) for x in X]
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

    def get_names(self, inverse: bool = False, X: list = []):
        """
    Returns the Transformation output names.

    Parameters
    ----------
    inverse: bool, optional
        If set to True, it returns the inverse transform output names.
    X: list, optional
        List of the columns used to get the model output names. If empty, 
        the model predictors names will be used.

    Returns
    -------
    list
        Python list.
        """
        if isinstance(X, str):
            X = [X]
        X = [quote_ident(elem) for elem in X]
        if not (X):
            X = self.X
        if self._model_type in ("PCA", "SVD", "MCA") and not (inverse):
            if self._model_type in ("PCA", "SVD"):
                n = self.parameters["n_components"]
                if not (n):
                    n = len(self.X)
            else:
                n = len(self.X)
            return [f"col{i}" for i in range(1, n + 1)]
        elif self._model_type == "OneHotEncoder" and not (inverse):
            names = []
            for column in self.X:
                k = 0
                for i in range(len(self.cat_["category_name"])):
                    if quote_ident(self.cat_["category_name"][i]) == quote_ident(
                        column
                    ):
                        if (k != 0 or not (self.parameters["drop_first"])) and (
                            not (self.parameters["ignore_null"])
                            or self.cat_["category_level"][i] != None
                        ):
                            if self.parameters["column_naming"] == "indices":
                                name = f'"{quote_ident(column)[1:-1]}{self.parameters["separator"]}'
                                name += f'{self.cat_["category_level_index"][i]}"'
                                names += [name]
                            else:
                                if self.cat_["category_level"][i] != None:
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

    def inverse_transform(self, vdf: Union[str, vDataFrame], X: Union[str, list] = []):
        """
    Applies the Inverse Model on a vDataFrame.

    Parameters
    ----------
    vdf: str / vDataFrame
        input vDataFrame. You can also specify a customized relation, 
        but you must enclose it with an alias. For example "(SELECT 1) x" is 
        correct whereas "(SELECT 1)" and "SELECT 1" are incorrect.
    X: str / list, optional
        List of the input vcolumns.

    Returns
    -------
    vDataFrame
        object result of the model transformation.
        """
        if isinstance(X, str):
            X = [X]
        if self._model_type == "OneHotEncoder":
            raise AttributeError(
                "method 'inverse_transform' is not supported for OneHotEncoder models."
            )
        if not (vdf):
            vdf = self.input_relation
        if not (X):
            X = self.get_names()
        if isinstance(vdf, str):
            vdf = vDataFrame(vdf)
        X = vdf._format_colnames(X)
        relation = vdf._genSQL()
        exclude_columns = vdf.get_columns(exclude_columns=X)
        all_columns = vdf.get_columns()
        inverse_sql = self.deployInverseSQL(
            exclude_columns, exclude_columns, all_columns
        )
        main_relation = f"(SELECT {inverse_sql} FROM {relation}) VERTICAPY_SUBTABLE"
        return vDataFrame(main_relation)

    def transform(self, vdf: Union[str, vDataFrame] = None, X: Union[str, list] = []):
        """
    Applies the model on a vDataFrame.

    Parameters
    ----------
    vdf: str / vDataFrame, optional
        Input vDataFrame. You can also specify a customized relation, 
        but you must enclose it with an alias. For example "(SELECT 1) x" is 
        correct whereas "(SELECT 1)" and "SELECT 1" are incorrect.
    X: str / list, optional
        List of the input vcolumns.

    Returns
    -------
    vDataFrame
        object result of the model transformation.
        """
        if isinstance(X, str):
            X = [X]
        if not (vdf):
            vdf = self.input_relation
        if not (X):
            X = self.X
        if isinstance(vdf, str):
            vdf = vDataFrame(vdf)
        X = vdf._format_colnames(X)
        relation = vdf._genSQL()
        exclude_columns = vdf.get_columns(exclude_columns=X)
        all_columns = vdf.get_columns()
        columns = self.deploySQL(exclude_columns, exclude_columns, all_columns)
        main_relation = f"(SELECT {columns} FROM {relation}) VERTICAPY_SUBTABLE"
        return vDataFrame(main_relation)


class Decomposition(Preprocessing):
    def deploySQL(
        self,
        n_components: int = 0,
        cutoff: Union[int, float] = 1,
        key_columns: Union[str, list] = [],
        exclude_columns: Union[str, list] = [],
        X: Union[str, list] = [],
    ):
        """
    Returns the SQL code needed to deploy the model. 

    Parameters
    ----------
    n_components: int, optional
        Number of components to return. If set to 0, all the components will be
        deployed.
    cutoff: int / float, optional
        Specifies the minimum accumulated explained variance. Components are taken 
        until the accumulated explained variance reaches this value.
    key_columns: str / list, optional
        Predictors used during the algorithm computation which will be deployed
        with the principal components.
    exclude_columns: str / list, optional
        Columns to exclude from the prediction.
    X: str / list, optional
        List of the columns used to deploy the self. If empty, the model
        predictors will be used.

    Returns
    -------
    str
        the SQL code needed to deploy the model.
        """
        if isinstance(key_columns, str):
            key_columns = [key_columns]
        if isinstance(exclude_columns, str):
            exclude_columns = [exclude_columns]
        if isinstance(X, str):
            X = [X]
        if not (X):
            X = self.X
        else:
            X = [quote_ident(elem) for elem in X]
        fun = self._vertica_transform_sql
        sql = f"""{self._vertica_transform_sql}({', '.join(X)} 
                                            USING PARAMETERS
                                            model_name = '{self.model_name}',
                                            match_by_pos = 'true'"""
        if key_columns:
            key_columns = ", ".join([quote_ident(col) for col in key_columns])
            sql += f", key_columns = '{key_columns}'"
        if exclude_columns:
            exclude_columns = ", ".join([quote_ident(col) for col in exclude_columns])
            sql += f", exclude_columns = '{exclude_columns}'"
        if n_components:
            sql += f", num_components = {n_components}"
        else:
            sql += f", cutoff = {cutoff}"
        sql += ")"
        return clean_query(sql)

    def plot(self, dimensions: tuple = (1, 2), ax=None, **style_kwds):
        """
    Draws a decomposition scatter plot.
    Parameters
    ----------
    dimensions: tuple, optional
        Tuple of two elements representing the IDs of the model's components.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.
    Returns
    -------
    ax
        Matplotlib axes object
        """
        vdf = vDataFrame(self.input_relation)
        ax = self.transform(vdf).scatter(
            columns=[f"col{dimensions[0]}", f"col{dimensions[1]}"],
            max_nb_points=100000,
            ax=ax,
            **style_kwds,
        )
        explained_variance = self.get_vertica_attributes("singular_values")[
            "explained_variance"
        ]
        if not (explained_variance[dimensions[0] - 1]):
            dimensions_1 = ""
        else:
            dimensions_1 = f"({round(explained_variance[dimensions[0] - 1] * 100, 1)}%)"
        ax.set_xlabel(f"Dim{dimensions[0]} {dimensions_1}")
        ax.set_ylabel(f"Dim{dimensions[0]} {dimensions_1}")
        return ax

    def plot_circle(self, dimensions: tuple = (1, 2), ax=None, **style_kwds):
        """
    Draws a decomposition circle.

    Parameters
    ----------
    dimensions: tuple, optional
        Tuple of two elements representing the IDs of the model's components.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object
        """
        if self._model_type == "SVD":
            x = self.get_vertica_attributes("right_singular_vectors")[
                f"vector{dimensions[0]}"
            ]
            y = self.get_vertica_attributes("right_singular_vectors")[
                f"vector{dimensions[1]}"
            ]
        else:
            x = self.get_vertica_attributes("principal_components")[
                f"PC{dimensions[0]}"
            ]
            y = self.get_vertica_attributes("principal_components")[
                f"PC{dimensions[1]}"
            ]
        explained_variance = self.get_vertica_attributes("singular_values")[
            "explained_variance"
        ]
        return plot_pca_circle(
            x,
            y,
            self.X,
            (
                explained_variance[dimensions[0] - 1],
                explained_variance[dimensions[1] - 1],
            ),
            dimensions,
            ax,
            **style_kwds,
        )

    def plot_scree(self, ax=None, **style_kwds):
        """
    Draws a decomposition scree plot.

    Parameters
    ----------
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object
        """
        explained_variance = self.get_vertica_attributes("singular_values")[
            "explained_variance"
        ]
        explained_variance, n = (
            [100 * elem for elem in explained_variance],
            len(explained_variance),
        )
        information = TableSample(
            {
                "dimensions": [i + 1 for i in range(n)],
                "percentage_explained_variance": explained_variance,
            }
        ).to_vdf()
        information["dimensions_center"] = information["dimensions"] + 0.5
        ax = information["dimensions"].hist(
            method="avg",
            of="percentage_explained_variance",
            h=1,
            max_cardinality=1,
            ax=ax,
            **style_kwds,
        )
        ax = information["percentage_explained_variance"].plot(
            ts="dimensions_center", ax=ax, color="black"
        )
        ax.set_xlim(1, n + 1)
        ax.set_xticks([i + 1.5 for i in range(n)])
        ax.set_xticklabels([i + 1 for i in range(n)])
        ax.set_ylabel('"percentage_explained_variance"')
        ax.set_xlabel('"dimensions"')
        for i in range(n):
            text_str = f"{round(explained_variance[i], 1)}%"
            ax.text(
                i + 1.5, explained_variance[i] + 1, text_str,
            )
        return ax

    def score(
        self,
        X: Union[str, list] = [],
        input_relation: str = "",
        method: Literal["avg", "median"] = "avg",
        p: int = 2,
    ):
        """
    Returns the decomposition score on a dataset for each transformed column. It
    is the average / median of the p-distance between the real column and its 
    result after applying the decomposition model and its inverse.  

    Parameters
    ----------
    X: str / list, optional
        List of the columns used to deploy the self. If empty, the model
        predictors will be used.
    input_relation: str, optional
        Input Relation. If empty, the model input relation will be used.
    method: str, optional
        Distance Method used to do the scoring.
            avg    : The average is used as aggregation.
            median : The median is used as aggregation.
    p: int, optional
        The p of the p-distance.

    Returns
    -------
    TableSample
        An object containing the result. For more information, see
        utilities.TableSample.
        """
        if isinstance(X, str):
            X = [X]
        if not (X):
            X = self.X
        if not (input_relation):
            input_relation = self.input_relation
        method = str(method).upper()
        if method == "MEDIAN":
            method = "APPROXIMATE_MEDIAN"
        if self._model_type in ("PCA", "SVD"):
            n_components = self.parameters["n_components"]
            if not (n_components):
                n_components = len(X)
        else:
            n_components = len(X)
        col_init_1 = [f"{X[idx]} AS col_init{idx}" for idx in range(len(X))]
        col_init_2 = [f"col_init{idx}" for idx in range(len(X))]
        cols = [f"col{idx + 1}" for idx in range(n_components)]
        query = f"""SELECT 
                        {self._vertica_transform_sql}
                        ({', '.join(self.X)} 
                            USING PARAMETERS 
                            model_name = '{self.model_name}', 
                            key_columns = '{', '.join(self.X)}', 
                            num_components = {n_components}) OVER () 
                    FROM {input_relation}"""
        query = f"""
            SELECT 
                {', '.join(col_init_1 + cols)} 
            FROM ({query}) VERTICAPY_SUBTABLE"""
        query = f"""
            SELECT 
                {self._vertica_inverse_transform_sql}
                ({', '.join(col_init_2 + cols)} 
                    USING PARAMETERS 
                    model_name = '{self.model_name}', 
                    key_columns = '{', '.join(col_init_2)}', 
                    exclude_columns = '{', '.join(col_init_2)}', 
                    num_components = {n_components}) OVER () 
            FROM ({query}) y"""
        p_distances = [
            f"""{method}(POWER(ABS(POWER({X[idx]}, {p}) 
                         - POWER(col_init{idx}, {p})), {1 / p})) 
                         AS {X[idx]}"""
            for idx in range(len(X))
        ]
        query = f"""
            SELECT 
                'Score' AS 'index', 
                {', '.join(p_distances)} 
            FROM ({query}) z"""
        return TableSample.read_sql(query, title="Getting Model Score.").transpose()

    def transform(
        self,
        vdf: Union[str, vDataFrame] = None,
        X: Union[str, list] = [],
        n_components: int = 0,
        cutoff: Union[int, float] = 1,
    ):
        """
    Applies the model on a vDataFrame.

    Parameters
    ----------
    vdf: str / vDataFrame, optional
        Input vDataFrame. You can also specify a customized relation, 
        but you must enclose it with an alias. For example "(SELECT 1) x" is 
        correct whereas "(SELECT 1)" and "SELECT 1" are incorrect.
    X: str / list, optional
        List of the input vcolumns.
    n_components: int, optional
        Number of components to return. If set to 0, all the components will 
        be deployed.
    cutoff: int / float, optional
        Specifies the minimum accumulated explained variance. Components are 
        taken until the accumulated explained variance reaches this value.

    Returns
    -------
    vDataFrame
        object result of the model transformation.
        """
        if isinstance(X, str):
            X = [X]
        if not (vdf):
            vdf = self.input_relation
        if not (X):
            X = self.X
        if isinstance(vdf, str):
            vdf = vDataFrame(vdf)
        X = vdf._format_colnames(X)
        relation = vdf._genSQL()
        exclude_columns = vdf.get_columns(exclude_columns=X)
        all_columns = vdf.get_columns()
        columns = self.deploySQL(
            n_components, cutoff, exclude_columns, exclude_columns, all_columns
        )
        main_relation = f"(SELECT {columns} FROM {relation}) VERTICAPY_SUBTABLE"
        return vDataFrame(main_relation)
