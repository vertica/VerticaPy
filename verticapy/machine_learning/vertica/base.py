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
from typing import Literal, Union, get_type_hints
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
from verticapy.errors import ConversionError, FunctionError, ParameterError, ModelError

from verticapy.core.tablesample.base import TableSample
from verticapy.core.vdataframe.base import vDataFrame

from verticapy.plotting._matplotlib.mlplot import (
    plot_importance,
    regression_tree_plot,
    lof_plot,
    plot_pca_circle,
    logit_plot,
    svm_classifier_plot,
    regression_plot,
)

from verticapy.machine_learning._utils import get_match_index
import verticapy.machine_learning.metrics as mt
from verticapy.machine_learning.model_management.read import does_model_exist
import verticapy.machine_learning.model_selection as ms

from verticapy.sql.drop import drop
from verticapy.sql.read import to_tablesample

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


class vModel:
    """
Base Class for Vertica Models.
	"""

    @property
    def _object_type(self) -> Literal["vModel"]:
        return "vModel"

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

    def __repr__(self):
        """
	Returns the model Representation.
		"""
        try:
            rep = ""
            if self._model_type not in (
                "DBSCAN",
                "NearestCentroid",
                "VAR",
                "SARIMAX",
                "LocalOutlierFactor",
                "KNeighborsRegressor",
                "KNeighborsClassifier",
                "CountVectorizer",
                "AutoML",
            ):
                name = (
                    self.tree_name
                    if self._model_type == "KernelDensity"
                    else self.model_name
                )
                try:
                    vertica_version(condition=[9, 0, 0])
                    func = f"GET_MODEL_SUMMARY(USING PARAMETERS model_name = '{name}')"
                except:
                    func = f"SUMMARIZE_MODEL('{name}')"
                res = _executeSQL(
                    f"SELECT /*+LABEL('learn.vModel.__repr__')*/ {func}",
                    title="Summarizing the model.",
                    method="fetchfirstelem",
                )
                return res
            elif self._model_type == "AutoML":
                rep = self.best_model_.__repr__()
            elif self._model_type == "AutoDataPrep":
                rep = self.final_relation_.__repr__()
            elif self._model_type == "DBSCAN":
                rep = f"=======\ndetails\n=======\nNumber of Clusters: {self.n_cluster_}\nNumber of Outliers: {self.n_noise_}"
            elif self._model_type == "LocalOutlierFactor":
                rep = f"=======\ndetails\n=======\nNumber of Errors: {self.n_errors_}"
            elif self._model_type == "NearestCentroid":
                rep = "=======\ndetails\n=======\n" + self.centroids_.__repr__()
            elif self._model_type == "VAR":
                rep = "=======\ndetails\n======="
                for idx, elem in enumerate(self.X):
                    rep += "\n\n # " + str(elem) + "\n\n" + self.coef_[idx].__repr__()
                rep += "\n\n===============\nAdditional Info\n==============="
                rep += f"\nInput Relation : {self.input_relation}"
                rep += f"\nX : {', '.join(self.X)}"
                rep += f"\nts : {self.ts}"
            elif self._model_type == "SARIMAX":
                rep = "=======\ndetails\n======="
                rep += "\n\n# Coefficients\n\n" + self.coef_.__repr__()
                if self.ma_piq_:
                    rep += "\n\n# MA PIQ\n\n" + self.ma_piq_.__repr__()
                rep += "\n\n===============\nAdditional Info\n==============="
                rep += f"\nInput Relation : {self.input_relation}"
                rep += f"\ny : {self.y}"
                rep += f"\nts : {self.ts}"
                if self.exogenous:
                    rep += f"\nExogenous Variables : {', '.join(self.exogenous)}"
                if self.ma_avg_:
                    rep += f"\nMA AVG : {self.ma_avg_}"
            elif self._model_type == "CountVectorizer":
                rep = "=======\ndetails\n======="
                if self.vocabulary_:
                    voc = [str(elem) for elem in self.vocabulary_]
                    if len(voc) > 100:
                        voc = voc[0:100] + [f"... ({len(self.vocabulary_) - 100} more)"]
                    rep += "\n\n# Vocabulary\n\n" + ", ".join(voc)
                if self.stop_words_:
                    rep += "\n\n# Stop Words\n\n" + ", ".join(
                        [str(elem) for elem in self.stop_words_]
                    )
                rep += "\n\n===============\nAdditional Info\n==============="
                rep += f"\nInput Relation : {self.input_relation}"
                rep += f"\nX : {', '.join(self.X)}"
            if self._model_type in (
                "DBSCAN",
                "NearestCentroid",
                "LocalOutlierFactor",
                "KNeighborsRegressor",
                "KNeighborsClassifier",
            ):
                rep += "\n\n===============\nAdditional Info\n==============="
                rep += f"\nInput Relation : {self.input_relation}"
                rep += f"\nX : {', '.join(self.X)}"
            if self._model_type in (
                "NearestCentroid",
                "KNeighborsRegressor",
                "KNeighborsClassifier",
            ):
                rep += f"\ny : {self.y}"
            return rep
        except:
            return f"<{self._model_type}>"

    def contour(
        self,
        nbins: int = 100,
        pos_label: Union[int, float, str] = None,
        ax=None,
        **style_kwds,
    ):
        """
    Draws the model's contour plot. Only available for regressors, binary 
    classifiers, and for models of exactly two predictors.

    Parameters
    ----------
    nbins: int, optional
        Number of bins used to discretize the two input numerical vcolumns.
    pos_label: int/float/str, optional
        Label to consider as positive. All the other classes will be merged and
        considered as negative for multiclass classification.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object
        """
        if self._model_type in (
            "RandomForestClassifier",
            "XGBoostClassifier",
            "NaiveBayes",
            "NearestCentroid",
            "KNeighborsClassifier",
        ):
            if not (pos_label):
                pos_label = sorted(self.classes_)[-1]
            if self._model_type in (
                "RandomForestClassifier",
                "XGBoostClassifier",
                "NaiveBayes",
                "NearestCentroid",
            ):
                return vDataFrame(self.input_relation).contour(
                    self.X,
                    self.deploySQL(X=self.X, pos_label=pos_label),
                    cbar_title=self.y,
                    nbins=nbins,
                    ax=ax,
                    **style_kwds,
                )
            else:
                return vDataFrame(self.input_relation).contour(
                    self.X,
                    self,
                    pos_label=pos_label,
                    cbar_title=self.y,
                    nbins=nbins,
                    ax=ax,
                    **style_kwds,
                )
        elif self._model_type == "KNeighborsRegressor":
            return vDataFrame(self.input_relation).contour(
                self.X, self, cbar_title=self.y, nbins=nbins, ax=ax, **style_kwds
            )
        elif self._model_type in (
            "KMeans",
            "BisectingKMeans",
            "KPrototypes",
            "IsolationForest",
        ):
            cbar_title = "cluster"
            if self._model_type == "IsolationForest":
                cbar_title = "anomaly_score"
            return vDataFrame(self.input_relation).contour(
                self.X, self, cbar_title=cbar_title, nbins=nbins, ax=ax, **style_kwds,
            )
        else:
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
            name = (
                self.tree_name
                if self._model_type == "KernelDensity"
                else self.model_name
            )
            X = self.X if not (X) else [quote_ident(predictor) for predictor in X]
            sql = f"""
                {self._vertica_predict_sql}({', '.join(X)} 
                                                    USING PARAMETERS 
                                                    model_name = '{name}',
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
        drop(self.model_name, method="model", model_type=self._model_type)

    def features_importance(
        self, ax=None, tree_id: int = None, show: bool = True, **style_kwds
    ):
        """
		Computes the model's features importance.

        Parameters
        ----------
        ax: Matplotlib axes object, optional
            The axes to plot on.
        tree_id: int
            Tree ID in case of Tree Based models.
        show: bool
            If set to True, draw the features importance.
        **style_kwds
            Any optional parameter to pass to the Matplotlib functions.

		Returns
		-------
		TableSample
			An object containing the result. For more information, see
			utilities.TableSample.
		"""
        if self._model_type == "AutoML":
            if self.stepwise_:
                coeff_importances = {}
                for idx in range(len(self.stepwise_["importance"])):
                    if self.stepwise_["variable"][idx] != None:
                        coeff_importances[
                            self.stepwise_["variable"][idx]
                        ] = self.stepwise_["importance"][idx]
                return plot_importance(
                    coeff_importances, print_legend=False, ax=ax, **style_kwds
                )
            return self.best_model_.features_importance(ax, tree_id, show, **style_kwds)
        if self._model_type in (
            "RandomForestClassifier",
            "RandomForestRegressor",
            "KernelDensity",
            "XGBoostClassifier",
            "XGBoostRegressor",
        ):
            name = (
                self.tree_name
                if self._model_type == "KernelDensity"
                else self.model_name
            )
            if self._model_type in ("XGBoostClassifier", "XGBoostRegressor",):
                vertica_version(condition=[12, 0, 3])
                fname = "XGB_PREDICTOR_IMPORTANCE"
                var = "avg_gain"
            else:
                vertica_version(condition=[9, 1, 1])
                fname = "RF_PREDICTOR_IMPORTANCE"
                var = "importance_value"
            tree_id = "" if tree_id is None else f", tree_id={tree_id}"
            query = f"""SELECT /*+LABEL('learn.vModel.features_importance')*/
                            predictor_name AS predictor, 
                            ROUND(100 * ABS({var}) / SUM(ABS({var}))
                                OVER (), 2)::float AS importance, 
                            SIGN({var})::int AS sign 
                        FROM 
                            (SELECT {fname} ( 
                                    USING PARAMETERS model_name = '{name}'{tree_id})) 
                                    VERTICAPY_SUBTABLE 
                        ORDER BY 2 DESC;"""
            print_legend = False
        elif self._model_type in (
            "LinearRegression",
            "LogisticRegression",
            "LinearSVC",
            "LinearSVR",
        ):
            relation = self.input_relation
            vertica_version(condition=[8, 1, 1])
            query = f"""
                SELECT /*+LABEL('learn.vModel.features_importance')*/
                    predictor, 
                    ROUND(100 * importance / SUM(importance) OVER(), 2) AS importance, 
                    sign 
                FROM (SELECT 
                        stat.predictor AS predictor, 
                        ABS(coefficient * (max - min))::float AS importance, 
                        SIGN(coefficient)::int AS sign 
                      FROM (SELECT 
                                LOWER("column") AS predictor, 
                                min, 
                                max 
                            FROM (SELECT 
                                    SUMMARIZE_NUMCOL({', '.join(self.X)}) OVER() 
                                  FROM {relation}) VERTICAPY_SUBTABLE) stat 
                                  NATURAL JOIN ({self.coef_.to_sql()}) coeff) importance_t 
                                  ORDER BY 2 DESC;"""
            print_legend = True
        else:
            raise FunctionError(
                f"Method 'features_importance' for '{self._model_type}' doesn't exist."
            )
        result = _executeSQL(
            query, title="Computing Features Importance.", method="fetchall"
        )
        coeff_importances, coeff_sign = {}, {}
        for elem in result:
            coeff_importances[elem[0]] = elem[1]
            coeff_sign[elem[0]] = elem[2]
        if show:
            plot_importance(
                coeff_importances,
                coeff_sign,
                print_legend=print_legend,
                ax=ax,
                **style_kwds,
            )
        importances = {"index": ["importance", "sign"]}
        for elem in coeff_importances:
            importances[elem] = [coeff_importances[elem], coeff_sign[elem]]
        return TableSample(values=importances).transpose()

    def get_attr(self, attr_name: str = ""):
        """
	Returns the model attribute.

	Parameters
	----------
	attr_name: str, optional
		Attribute Name.

	Returns
	-------
	TableSample
		model attribute
		"""
        if self._model_type == "AutoML":
            return self.best_model_.get_attr(attr_name)
        if self._model_type not in (
            "DBSCAN",
            "LocalOutlierFactor",
            "VAR",
            "SARIMAX",
            "KNeighborsClassifier",
            "KNeighborsRegressor",
            "NearestCentroid",
            "CountVectorizer",
        ):
            name = (
                self.tree_name
                if self._model_type == "KernelDensity"
                else self.model_name
            )
            vertica_version(condition=[8, 1, 1])
            if attr_name:
                attr_name_str = f", attr_name = '{attr_name}'"
            else:
                attr_name_str = ""
            result = to_tablesample(
                query=f"""
                    SELECT 
                        GET_MODEL_ATTRIBUTE(USING PARAMETERS 
                                            model_name = '{name}'{attr_name_str})""",
                title="Getting Model Attributes.",
            )
            return result
        elif self._model_type == "DBSCAN":
            if attr_name == "n_cluster":
                return self.n_cluster_
            elif attr_name == "n_noise":
                return self.n_noise_
            elif not (attr_name):
                result = TableSample(
                    values={
                        "attr_name": ["n_cluster", "n_noise"],
                        "value": [self.n_cluster_, self.n_noise_],
                    },
                )
                return result
            else:
                raise ParameterError(f"Attribute '{attr_name}' doesn't exist.")
        elif self._model_type == "CountVectorizer":
            if attr_name == "lowercase":
                return self.parameters["lowercase"]
            elif attr_name == "max_df":
                return self.parameters["max_df"]
            elif attr_name == "min_df":
                return self.parameters["min_df"]
            elif attr_name == "max_features":
                return self.parameters["max_features"]
            elif attr_name == "ignore_special":
                return self.parameters["ignore_special"]
            elif attr_name == "max_text_size":
                return self.parameters["max_text_size"]
            elif attr_name == "vocabulary":
                return self.parameters["vocabulary"]
            elif attr_name == "stop_words":
                return self.parameters["stop_words"]
            elif not (attr_name):
                result = TableSample(
                    values={
                        "attr_name": [
                            "lowercase",
                            "max_df",
                            "min_df",
                            "max_features",
                            "ignore_special",
                            "max_text_size",
                            "vocabulary",
                            "stop_words",
                        ],
                    },
                )
                return result
            else:
                raise ParameterError(f"Attribute '{attr_name}' doesn't exist.")
        elif self._model_type == "NearestCentroid":
            if attr_name == "p":
                return self.parameters["p"]
            elif attr_name == "centroids":
                return self.centroids_
            elif attr_name == "classes":
                return self.classes_
            elif not (attr_name):
                result = TableSample(
                    values={"attr_name": ["centroids", "classes", "p"],},
                )
                return result
            else:
                raise ParameterError(f"Attribute '{attr_name}' doesn't exist.")
        elif self._model_type == "KNeighborsClassifier":
            if attr_name == "p":
                return self.parameters["p"]
            elif attr_name == "n_neighbors":
                return self.parameters["n_neighbors"]
            elif attr_name == "classes":
                return self.classes_
            elif not (attr_name):
                result = TableSample(
                    values={"attr_name": ["n_neighbors", "p", "classes"],},
                )
                return result
            else:
                raise ParameterError(f"Attribute '{attr_name}' doesn't exist.")
        elif self._model_type == "KNeighborsRegressor":
            if attr_name == "p":
                return self.parameters["p"]
            elif attr_name == "n_neighbors":
                return self.parameters["n_neighbors"]
            elif not (attr_name):
                result = TableSample(values={"attr_name": ["n_neighbors", "p"],},)
                return result
            else:
                raise ParameterError(f"Attribute '{attr_name}' doesn't exist.")
        elif self._model_type == "LocalOutlierFactor":
            if attr_name == "n_errors":
                return self.n_errors_
            elif not (attr_name):
                result = TableSample(
                    values={"attr_name": ["n_errors"], "value": [self.n_errors_]},
                )
                return result
            else:
                raise ParameterError(f"Attribute '{attr_name}' doesn't exist.")
        elif self._model_type == "SARIMAX":
            if attr_name == "coefficients":
                return self.coef_
            elif attr_name == "ma_avg":
                return self.ma_avg_
            elif attr_name == "ma_piq":
                return self.ma_piq_
            elif not (attr_name):
                result = TableSample(
                    values={"attr_name": ["coefficients", "ma_avg", "ma_piq"]},
                )
                return result
            else:
                raise ParameterError(f"Attribute '{attr_name}' doesn't exist.")
        elif self._model_type == "VAR":
            if attr_name == "coefficients":
                return self.coef_
            elif not (attr_name):
                result = TableSample(values={"attr_name": ["coefficients"]})
                return result
            else:
                raise ParameterError(f"Attribute '{attr_name}' doesn't exist.")
        elif self._model_type == "KernelDensity":
            if attr_name == "map":
                return self.map_
            elif not (attr_name):
                result = TableSample(values={"attr_name": ["map"]})
                return result
            else:
                raise ParameterError(f"Attribute '{attr_name}' doesn't exist.")
        else:
            raise FunctionError(
                f"Method 'get_attr' for '{self._model_type}' doesn't exist."
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

    def get_vertica_param_dict(self):
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
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object
		"""
        if self._model_type in (
            "LinearRegression",
            "LogisticRegression",
            "LinearSVC",
            "LinearSVR",
        ):
            coefficients = self.coef_.values["coefficient"]
            if self._model_type == "LogisticRegression":
                return logit_plot(
                    self.X,
                    self.y,
                    self.input_relation,
                    coefficients,
                    max_nb_points,
                    ax=ax,
                    **style_kwds,
                )
            elif self._model_type == "LinearSVC":
                return svm_classifier_plot(
                    self.X,
                    self.y,
                    self.input_relation,
                    coefficients,
                    max_nb_points,
                    ax=ax,
                    **style_kwds,
                )
            else:
                return regression_plot(
                    self.X,
                    self.y,
                    self.input_relation,
                    coefficients,
                    max_nb_points,
                    ax=ax,
                    **style_kwds,
                )
        elif self._model_type in (
            "KMeans",
            "BisectingKMeans",
            "KPrototypes",
            "DBSCAN",
            "IsolationForest",
        ):
            if self._model_type in ("KMeans", "BisectingKMeans", "KPrototypes",):
                if self._model_type == "KPrototypes":
                    if any(
                        [
                            ("char" in self.cluster_centers_.dtype[key].lower())
                            for key in self.cluster_centers_.dtype
                        ]
                    ):
                        raise TypeError(
                            "KPrototypes' plots with categorical inputs is not yet supported."
                        )
                vdf = vDataFrame(self.input_relation)
                catcol = f"{self._model_type.lower()}_cluster"
                self.predict(vdf, name=catcol)
            elif self._model_type == "DBSCAN":
                vdf = vDataFrame(self.model_name)
                catcol = "dbscan_cluster"
            elif self._model_type == "IsolationForest":
                vdf = vDataFrame(self.input_relation)
                self.predict(vdf, name="anomaly_score")
                return vdf.bubble(
                    columns=self.X,
                    cmap_col="anomaly_score",
                    max_nb_points=max_nb_points,
                    ax=ax,
                    **style_kwds,
                )
            return vdf.scatter(
                columns=self.X,
                catcol=catcol,
                max_cardinality=100,
                max_nb_points=max_nb_points,
                ax=ax,
                **style_kwds,
            )
        elif self._model_type == "LocalOutlierFactor":
            cnt = _executeSQL(
                query=f"SELECT /*+LABEL('learn.vModel.plot')*/ COUNT(*) FROM {self.model_name}",
                method="fetchfirstelem",
                print_time_sql=False,
            )
            TableSample = 100 * min(float(max_nb_points / cnt), 1)
            return lof_plot(
                self.model_name, self.X, "lof_score", 100, ax=ax, **style_kwds
            )
        elif self._model_type in ("RandomForestRegressor", "XGBoostRegressor"):
            return regression_tree_plot(
                self.X + [self.deploySQL()],
                self.y,
                self.input_relation,
                max_nb_points,
                ax=ax,
                **style_kwds,
            )
        else:
            raise FunctionError(
                f"Method 'plot' for '{self._model_type}' doesn't exist."
            )

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

    def to_memmodel(self, **kwds):
        """
    Converts a specified Vertica model to a memModel model.

    Returns
    -------
    object
        memModel model.
        """
        from verticapy.machine_learning.memmodel.base import memModel
        from verticapy.machine_learning.vertica.tree import get_tree_list_of_arrays

        if self._model_type == "AutoML":
            return self.best_model_.to_memmodel()
        elif self._model_type in (
            "LinearRegression",
            "LogisticRegression",
            "LinearSVC",
            "LinearSVR",
        ):
            attributes = {
                "coefficients": self.coef_["coefficient"][1:],
                "intercept": self.coef_["coefficient"][0],
            }
        elif self._model_type == "BisectingKMeans":
            attributes = {
                "clusters": self.cluster_centers_.to_numpy()[:, 1 : len(self.X) + 1],
                "left_child": self.cluster_centers_["left_child"],
                "right_child": self.cluster_centers_["right_child"],
                "cluster_size": self.cluster_centers_["cluster_size"],
                "cluster_score": [
                    self.cluster_centers_["withinss"][i]
                    / self.cluster_centers_["totWithinss"][i]
                    for i in range(len(self.cluster_centers_["withinss"]))
                ],
                "p": 2,
            }
        elif self._model_type in ("KMeans", "KPrototypes"):
            attributes = {"clusters": self.cluster_centers_.to_numpy(), "p": 2}
            if self._model_type == "KPrototypes":
                attributes["gamma"] = self.parameters["gamma"]
                attributes["is_categorical"] = [
                    ("char" in self.cluster_centers_.dtype[key].lower())
                    for key in self.cluster_centers_.dtype
                ]
        elif self._model_type == "NearestCentroid":
            attributes = {
                "clusters": self.centroids_.to_numpy()[:, 0:-1],
                "p": self.parameters["p"],
                "classes": self.classes_,
            }
        elif self._model_type == "NaiveBayes":
            attributes = {
                "attributes": self.get_var_info(),
                "prior": self.get_attr("prior")["probability"],
                "classes": self.classes_,
            }
        elif self._model_type == "OneHotEncoder":

            def get_one_hot_encode_X_cat(L: list):
                # Allows to split the One Hot Encoder Array by features categories
                cat, tmp_cat, init_cat, X = [], [], L[0][0], [L[0][0]]
                for c in L:
                    if c[0] != init_cat:
                        init_cat = c[0]
                        X += [c[0]]
                        cat += [tmp_cat]
                        tmp_cat = [c[1]]
                    else:
                        tmp_cat += [c[1]]
                cat += [tmp_cat]
                return X, cat

            cat = list(
                get_one_hot_encode_X_cat([l[0:2] for l in self.param_.to_list()])
            )
            cat_list_idx = []
            for i, x1 in enumerate(cat[0]):
                for j, x2 in enumerate(self.X):
                    if x2.lower()[1:-1] == x1:
                        cat_list_idx += [j]
            categories = []
            for i in cat_list_idx:
                categories += [cat[1][i]]
            attributes = {
                "categories": categories,
                "drop_first": self.parameters["drop_first"],
                "column_naming": self.parameters["column_naming"],
            }
        elif self._model_type == "PCA":
            attributes = {
                "principal_components": self.get_attr(
                    "principal_components"
                ).to_numpy(),
                "mean": self.get_attr("columns")["mean"],
            }
        elif self._model_type == "SVD":
            attributes = {
                "vectors": self.get_attr("right_singular_vectors").to_numpy(),
                "values": self.get_attr("singular_values")["value"],
            }
        elif self._model_type == "Normalizer":
            attributes = {
                "values": self.get_attr("details").to_numpy()[:, 1:].astype(float),
                "method": self.parameters["method"],
            }
        elif self._model_type in (
            "XGBoostClassifier",
            "XGBoostRegressor",
            "RandomForestClassifier",
            "RandomForestRegressor",
            "IsolationForest",
        ):
            if self._model_type in (
                "RandomForestClassifier",
                "RandomForestRegressor",
                "IsolationForest",
            ):
                n = self.parameters["n_estimators"]
            else:
                n = self.get_attr("tree_count")["tree_count"][0]
            trees = []
            if self._model_type in ("XGBoostRegressor", "RandomForestRegressor"):
                tree_type = "BinaryTreeRegressor"
            elif self._model_type in ("XGBoostClassifier", "RandomForestClassifier"):
                tree_type = "BinaryTreeClassifier"
            else:
                tree_type = "BinaryTreeAnomaly"
            return_prob_rf = self._model_type == "RandomForestClassifier"
            is_iforest = self._model_type == "IsolationForest"
            for i in range(n):
                if ("return_tree" in kwds and kwds["return_tree"] == i) or (
                    "return_tree" not in kwds
                ):
                    tree = self.get_tree(i)
                    tree = get_tree_list_of_arrays(
                        tree, self.X, self._model_type, return_prob_rf
                    )
                    tree_attributes = {
                        "children_left": tree[0],
                        "children_right": tree[1],
                        "feature": tree[2],
                        "threshold": tree[3],
                        "value": tree[4],
                    }
                    for idx in range(len(tree[5])):
                        if not (tree[5][idx]) and isinstance(
                            tree_attributes["threshold"][idx], str
                        ):
                            tree_attributes["threshold"][idx] = float(
                                tree_attributes["threshold"][idx]
                            )
                    if tree_type == "BinaryTreeClassifier":
                        tree_attributes["classes"] = self.classes_
                    elif tree_type == "BinaryTreeRegressor":
                        tree_attributes["value"] = [
                            float(val) if isinstance(val, str) else val
                            for val in tree_attributes["value"]
                        ]
                    if self._model_type == "XGBoostClassifier":
                        tree_attributes["value"] = tree[6]
                        for idx in range(len(tree[6])):
                            if tree[6][idx] != None:
                                all_classes_logodss = []
                                for c in self.classes_:
                                    all_classes_logodss += [tree[6][idx][str(c)]]
                                tree_attributes["value"][idx] = all_classes_logodss
                    elif self._model_type == "RandomForestClassifier":
                        for idx in range(len(tree_attributes["value"])):
                            if tree_attributes["value"][idx] != None:
                                prob = [0.0 for i in range(len(self.classes_))]
                                for idx2, c in enumerate(self.classes_):
                                    if str(c) == str(tree_attributes["value"][idx]):
                                        prob[idx2] = tree[6][idx]
                                        break
                                other_proba = (1 - tree[6][idx]) / (
                                    len(self.classes_) - 1
                                )
                                for idx2, p in enumerate(prob):
                                    if p == 0.0:
                                        prob[idx2] = other_proba
                                tree_attributes["value"][idx] = prob
                    if tree_type == "BinaryTreeAnomaly":
                        tree_attributes["psy"] = int(
                            self.parameters["sample"]
                            * int(
                                self.get_attr("accepted_row_count")[
                                    "accepted_row_count"
                                ][0]
                            )
                        )
                    model = memModel(model_type=tree_type, attributes=tree_attributes)
                    if "return_tree" in kwds and kwds["return_tree"] == i:
                        return model
                    trees += [model]
            attributes = {"trees": trees}
            if self._model_type in ("XGBoostRegressor", "XGBoostClassifier"):
                attributes["learning_rate"] = self.parameters["learning_rate"]
                if self._model_type == "XGBoostRegressor":
                    attributes["mean"] = self.prior_
                elif not (isinstance(self.prior_, list)):
                    attributes["logodds"] = [
                        np.log((1 - self.prior_) / self.prior_),
                        np.log(self.prior_ / (1 - self.prior_)),
                    ]
                else:
                    attributes["logodds"] = self.prior_.copy()
        else:
            raise ModelError(
                f"Model type '{self._model_type}' can not be converted to memModel."
            )
        return memModel(model_type=self._model_type, attributes=attributes)

    def to_python(
        self,
        name: str = "predict",
        return_proba: bool = False,
        return_distance_clusters: bool = False,
        return_str: bool = False,
    ):
        """
    Returns the Python code needed to deploy the model without using built-in
    Vertica functions.

    Parameters
    ----------
    name: str, optional
        Function Name.
    return_proba: bool, optional
        If set to True and the model is a classifier, the function
        returns the model probabilities.
    return_distance_clusters: bool, optional
        If set to True and the model type is KMeans or NearestCentroid, 
        the function returns the model clusters distances. If the model
        is KPrototypes, the function returns the dissimilarity function.
    return_str: bool, optional
        If set to True, the function str will be returned.


    Returns
    -------
    str / func
        Python function
        """
        from verticapy.machine_learning.vertica.tree import get_tree_list_of_arrays

        if not (return_str):
            func = self.to_python(
                name=name,
                return_proba=return_proba,
                return_distance_clusters=return_distance_clusters,
                return_str=True,
            )
            all_vars = {}
            exec(func, {}, all_vars)
            return all_vars[name]
        func = f"def {name}(X):\n\timport numpy as np\n\t"
        if self._model_type in (
            "LinearRegression",
            "LinearSVR",
            "LogisticRegression",
            "LinearSVC",
        ):
            a, b = self.coef_["coefficient"][0], self.coef_["coefficient"][1:]
            x = f"{a} + np.sum(np.array({b}) * np.array(X), axis=1)"
            if self._model_type in ("LogisticRegression", "LinearSVC"):
                func += f"result = 1 / (1 + np.exp(- ({x})))"
            else:
                func += f"result =  {x}"
            if return_proba and self._model_type in ("LogisticRegression", "LinearSVC"):
                func += "\n\treturn np.column_stack((1 - result, result))"
            elif not (return_proba) and self._model_type in (
                "LogisticRegression",
                "LinearSVC",
            ):
                func += "\n\treturn np.where(result > 0.5, 1, 0)"
            else:
                func += "\n\treturn result"
            return func
        elif self._model_type == "BisectingKMeans":
            bktree = self.get_attr("BKTree")
            cluster = [elem[1:-7] for elem in bktree.to_list()]
            func += f"centroids = np.array({cluster})\n"
            func += f"\tright_child = {bktree['right_child']}\n"
            func += f"\tleft_child = {bktree['left_child']}\n"
            func += "\tdef predict_tree(right_child, left_child, row, node_id, centroids):\n"
            func += "\t\tif left_child[node_id] == right_child[node_id] == None:\n"
            func += "\t\t\treturn int(node_id)\n"
            func += "\t\telse:\n"
            func += "\t\t\tright_node = int(right_child[node_id])\n"
            func += "\t\t\tleft_node = int(left_child[node_id])\n"
            func += "\t\t\tif np.sum((row - centroids[left_node]) ** 2) < "
            func += "np.sum((row - centroids[right_node]) ** 2):\n"
            func += "\t\t\t\treturn predict_tree(right_child, left_child, row, left_node, centroids)\n"
            func += "\t\t\telse:\n"
            func += "\t\t\t\treturn predict_tree(right_child, left_child, row, right_node, centroids)\n"
            func += "\tdef predict_tree_final(row):\n"
            func += (
                "\t\treturn predict_tree(right_child, left_child, row, 0, centroids)\n"
            )
            func += "\treturn np.apply_along_axis(predict_tree_final, 1, X)\n"
            return func
        elif self._model_type in ("KPrototypes",):
            centroids = self.cluster_centers_.to_list()
            func += f"centroids = np.array({centroids})\n\n"
            func += "\tdef compute_distance_row(X):\n"
            func += "\t\tresult = []\n"
            func += "\t\tfor centroid in centroids:\n"
            func += "\t\t\tdistance_num, distance_cat = 0, 0\n"
            func += "\t\t\tfor idx in range(len(X)):\n"
            func += "\t\t\t\tval, centroid_val = X[idx], centroid[idx]\n"
            func += "\t\t\t\ttry:\n"
            func += "\t\t\t\t\tval = float(val)\n"
            func += "\t\t\t\t\tcentroid_val = float(centroid_val)\n"
            func += "\t\t\t\texcept:\n"
            func += "\t\t\t\t\tpass\n"
            func += (
                "\t\t\t\tif isinstance(centroid_val, str) or centroid_val == None:\n"
            )
            func += "\t\t\t\t\tdistance_cat += abs(int(val == centroid_val) - 1)\n"
            func += "\t\t\t\telse:\n"
            func += "\t\t\t\t\tdistance_num += (val - centroid_val) ** 2\n"
            gamma = self.parameters["gamma"]
            func += f"\t\t\tdistance_final = distance_num + {gamma} * distance_cat\n"
            func += "\t\t\tresult += [distance_final]\n"
            func += "\t\treturn result\n\n"
            func += "\tresult = np.apply_along_axis(compute_distance_row, 1, X)\n\n"
            if return_proba:
                func += "\treturn 1 / (result + 1e-99) / np.sum(1 / (result + 1e-99), axis=1)[:, None]\n"
            elif not (return_distance_clusters):
                func += "\treturn np.argmin(result, axis=1)\n"
            else:
                func += "\treturn result\n"
            return func
        elif self._model_type in ("NearestCentroid", "KMeans",):
            centroids = (
                self.centroids_.to_list()
                if self._model_type == "NearestCentroid"
                else self.cluster_centers_.to_list()
            )
            if self._model_type == "NearestCentroid":
                for center in centroids:
                    del center[-1]
            func += f"centroids = np.array({centroids})\n"
            if self._model_type == "NearestCentroid":
                func += f"\tclasses = np.array({self.classes_})\n"
            func += "\tresult = []\n"
            func += "\tfor centroid in centroids:\n"
            if self._model_type == "NearestCentroid":
                p = self.parameters["p"]
            else:
                p = 2
            func += f"\t\tresult += [np.sum((np.array(centroid) - X) ** {p}, axis=1) ** (1 / {p})]\n"
            func += "\tresult = np.column_stack(result)\n"
            if (
                self._model_type == "NearestCentroid"
                and return_proba
                and not (return_distance_clusters)
            ):
                func += "\tresult = 1 / (result + 1e-99) / np.sum(1 / (result + 1e-99), axis=1)[:,None]\n"
            elif not (return_distance_clusters):
                func += "\tresult = np.argmin(result, axis=1)\n"
                if self._model_type == "NearestCentroid" and self.classes_ != [
                    i for i in range(len(self.classes_))
                ]:
                    func += "\tclass_is_str = isinstance(classes[0], str)\n"
                    func += "\tfor idx, c in enumerate(classes):\n"
                    func += (
                        "\t\ttmp_idx = str(idx) if class_is_str and idx > 0 else idx\n"
                    )
                    func += "\t\tresult = np.where(result == tmp_idx, c, result)\n"
            func += "\treturn result\n"
            return func
        elif self._model_type in ("PCA", "MCA"):
            avg = self.get_attr("columns")["mean"]
            pca = []
            attr = self.get_attr("principal_components")
            n = len(attr["PC1"])
            for i in range(1, n + 1):
                pca += [attr[f"PC{i}"]]
            func += f"avg_values = np.array({avg})\n"
            func += f"\tpca_values = np.array({pca})\n"
            func += "\tresult = (X - avg_values)\n"
            func += "\tL = []\n"
            func += f"\tfor i in range({n}):\n"
            func += "\t\tL += [np.sum(result * pca_values[i], axis=1)]\n"
            func += "\tresult = np.column_stack(L)\n"
            func += "\treturn result\n"
            return func
        elif self._model_type == "Normalizer":
            details = self.get_attr("details")
            sql = []
            if "min" in details.values:
                func += f"min_values = np.array({details['min']})\n"
                func += f"\tmax_values = np.array({details['max']})\n"
                func += "\treturn (X - min_values) / (max_values - min_values)\n"
            elif "median" in details.values:
                func += f"median_values = np.array({details['median']})\n"
                func += f"\tmad_values = np.array({details['mad']})\n"
                func += "\treturn (X - median_values) / mad_values\n"
            else:
                func += f"avg_values = np.array({details['avg']})\n"
                func += f"\tstd_values = np.array({details['std_dev']})\n"
                func += "\treturn (X - avg_values) / std_values\n"
            return func
        elif self._model_type == "SVD":
            sv = []
            attr = self.get_attr("right_singular_vectors")
            n = len(attr["vector1"])
            for i in range(1, n + 1):
                sv += [attr[f"vector{i}"]]
            value = self.get_attr("singular_values")["value"]
            func += f"singular_values = np.array({value})\n"
            func += f"\tright_singular_vectors = np.array({sv})\n"
            func += "\tL = []\n"
            n = len(sv[0])
            func += f"\tfor i in range({n}):\n"
            func += "\t\tL += [np.sum(X * right_singular_vectors[i] / "
            func += "singular_values[i], axis=1)]\n"
            func += "\tresult = np.column_stack(L)\n"
            func += "\treturn result\n"
            return func
        elif self._model_type == "NaiveBayes":
            var_info_simplified = self.get_var_info()
            prior = self.get_attr("prior").values["probability"]
            func += f"var_info_simplified = {var_info_simplified}\n"
            func += f"\tprior = np.array({prior})\n"
            func += f"\tclasses = {self.classes_}\n"
            func += "\tdef naive_bayes_score_row(X):\n"
            func += "\t\tresult = []\n"
            func += "\t\tfor c in classes:\n"
            func += "\t\t\tsub_result = []\n"
            func += "\t\t\tfor idx, elem in enumerate(X):\n"
            func += "\t\t\t\tprob = var_info_simplified[idx]\n"
            func += "\t\t\t\tif prob['type'] == 'multinomial':\n"
            func += "\t\t\t\t\tprob = prob[c] ** float(X[idx])\n"
            func += "\t\t\t\telif prob['type'] == 'bernoulli':\n"
            func += "\t\t\t\t\tprob = prob[c] if X[idx] else 1 - prob[c]\n"
            func += "\t\t\t\telif prob['type'] == 'categorical':\n"
            func += "\t\t\t\t\tprob = prob[str(c)][X[idx]]\n"
            func += "\t\t\t\telse:\n"
            func += "\t\t\t\t\tprob = 1 / np.sqrt(2 * np.pi * prob[c]['sigma_sq'])"
            func += " * np.exp(- (float(X[idx]) - prob[c]['mu']) ** 2 / "
            func += "(2 * prob[c]['sigma_sq']))\n"
            func += "\t\t\t\tsub_result += [prob]\n"
            func += "\t\t\tresult += [sub_result]\n"
            func += "\t\tresult = np.array(result).prod(axis=1) * prior\n"
            if return_proba:
                func += "\t\treturn result / result.sum()\n"
            else:
                func += "\t\treturn classes[np.argmax(result)]\n"
            func += "\treturn np.apply_along_axis(naive_bayes_score_row, 1, X)\n"
            return func
        elif self._model_type == "OneHotEncoder":
            predictors = self.X
            details = self.param_.values
            n, m = len(predictors), len(details["category_name"])
            positions = {}
            for i in range(m):
                val = quote_ident(details["category_name"][i])
                if val not in positions:
                    positions[val] = [i]
                else:
                    positions[val] += [i]
            category_level = []
            for p in predictors:
                pos = positions[p]
                category_level += [details["category_level"][pos[0] : pos[-1] + 1]]
            if self.parameters["drop_first"]:
                category_level = [elem[1:] for elem in category_level]
            func += f"category_level = {category_level}\n\t"
            func += "def ooe_row(X):\n\t"
            func += "\tresult = []\n\t"
            func += "\tfor idx, elem in enumerate(X):\n\t\t"
            func += "\tfor item in category_level[idx]:\n\t\t\t"
            func += "\tif str(elem) == str(item):\n\t\t\t\t"
            func += "\tresult += [1]\n\t\t\t"
            func += "\telse:\n\t\t\t\t"
            func += "\tresult += [0]\n\t"
            func += "\treturn result\n"
            func += "\treturn np.apply_along_axis(ooe_row, 1, X)\n"
            return func
        elif self._model_type in (
            "RandomForestClassifier",
            "RandomForestRegressor",
            "XGBoostRegressor",
            "XGBoostClassifier",
            "IsolationForest",
        ):
            result = []
            if self._model_type in (
                "RandomForestClassifier",
                "RandomForestRegressor",
                "IsolationForest",
            ):
                n = self.parameters["n_estimators"]
            else:
                n = self.get_attr("tree_count")["tree_count"][0]
            func += f"n = {n}\n"
            if self._model_type in ("XGBoostClassifier", "RandomForestClassifier"):
                classes_str = [str(c) for c in self.classes_]
                func += f"\tclasses = np.array({classes_str})\n"
            func += "\ttree_list = []\n"
            for i in range(n):
                tree = self.get_tree(i)
                tree_list = get_tree_list_of_arrays(tree, self.X, self._model_type)
                func += f"\ttree_list += [{tree_list}]\n"
            if self._model_type == "IsolationForest":
                func += "\tdef heuristic_length(i):\n"
                func += "\t\tGAMMA = 0.5772156649\n"
                func += "\t\tif i == 2:\n"
                func += "\t\t\treturn 1\n"
                func += "\t\telif i > 2:\n"
                func += "\t\t\treturn 2 * (np.log(i - 1) + GAMMA) - 2 * (i - 1) / i\n"
                func += "\t\telse:\n"
                func += "\t\t\treturn 0\n"
            func += "\tdef predict_tree(tree, node_id, X):\n"
            func += "\t\tif tree[0][node_id] == tree[1][node_id]:\n"
            if self._model_type in ("RandomForestRegressor", "XGBoostRegressor",):
                func += "\t\t\treturn float(tree[4][node_id])\n"
            elif self._model_type == "IsolationForest":
                psy = int(
                    self.parameters["sample"]
                    * int(self.get_attr("accepted_row_count")["accepted_row_count"][0])
                )
                func += f"\t\t\treturn (tree[4][node_id][0] + heuristic_length(tree[4]"
                func += f"[node_id][1])) / heuristic_length({psy})\n"
            elif self._model_type == "RandomForestClassifier":
                func += "\t\t\treturn tree[4][node_id]\n"
            else:
                func += "\t\t\treturn tree[6][node_id]\n"
            func += "\t\telse:\n"
            func += "\t\t\tidx, right_node, left_node = tree[2]"
            func += "[node_id], tree[1][node_id], tree[0][node_id]\n"
            func += "\t\t\tif (tree[5][node_id] and str(X[idx]) == tree[3][node_id]) "
            func += "or (not(tree[5][node_id]) and float(X[idx]) < float(tree[3][node_id])):\n"
            func += "\t\t\t\treturn predict_tree(tree, left_node, X)\n"
            func += "\t\t\telse:\n"
            func += "\t\t\t\treturn predict_tree(tree, right_node, X)\n"
            func += "\tdef predict_tree_final(X):\n"
            func += "\t\tresult = [predict_tree(tree, 0, X) for tree in tree_list]\n"
            if self._model_type in ("XGBoostClassifier", "RandomForestClassifier"):
                func += "\t\tall_classes_score = {}\n"
                func += "\t\tfor elem in classes:\n"
                func += "\t\t\tall_classes_score[elem] = 0\n"
            if self._model_type in ("XGBoostRegressor", "XGBoostClassifier"):
                if self._model_type == "XGBoostRegressor":
                    avg = self.prior_
                    func += f"\t\treturn {avg} + {self.parameters['learning_rate']} "
                    func += "* np.sum(result)\n"
                else:
                    if not (isinstance(self.prior_, list)):
                        a = np.log((1 - self.prior_) / self.prior_)
                        b = np.log(self.prior_ / (1 - self.prior_))
                        func += f"\t\tlogodds = np.array([{a}, {b}])\n"
                    else:
                        func += f"\t\tlogodds = np.array({self.prior_})\n"
                    func += "\t\tfor idx, elem in enumerate(all_classes_score):\n"
                    func += "\t\t\tfor val in result:\n"
                    func += "\t\t\t\tall_classes_score[elem] += val[elem]\n"
                    func += "\t\t\tall_classes_score[elem] = 1 / (1 + np.exp( - "
                    func += f"(logodds[idx] + {self.parameters['learning_rate']} * "
                    func += "all_classes_score[elem])))\n"
                    func += "\t\tresult = [all_classes_score[elem] for elem in "
                    func += "all_classes_score]\n"
            elif self._model_type == "RandomForestRegressor":
                func += "\t\treturn np.mean(result)\n"
            elif self._model_type == "IsolationForest":
                func += "\t\treturn 2 ** ( - np.mean(result))\n"
            else:
                func += "\t\tfor elem in result:\n"
                func += "\t\t\tif elem in all_classes_score:\n"
                func += "\t\t\t\tall_classes_score[elem] += 1\n"
                func += "\t\tresult = []\n"
                func += "\t\tfor elem in all_classes_score:\n"
                func += "\t\t\tresult += [all_classes_score[elem]]\n"
            if self._model_type in ("XGBoostClassifier", "RandomForestClassifier"):
                if return_proba:
                    func += "\t\treturn np.array(result) / np.sum(result)\n"
                else:
                    if isinstance(self.classes_[0], int):
                        func += "\t\treturn int(classes[np.argmax(np.array(result))])\n"
                    else:
                        func += "\t\treturn classes[np.argmax(np.array(result))]\n"
            func += "\treturn np.apply_along_axis(predict_tree_final, 1, X)\n"
            return func
        else:
            raise ModelError(
                f"Function to_python not yet available for model type '{self._model_type}'."
            )

    def to_sql(self, X: list = [], return_proba: bool = False):
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

    Returns
    -------
    str / list
        SQL code
        """
        if not X:
            X = self.X
        model = self.to_memmodel()
        if self._model_type in ("PCA", "SVD", "Normalizer", "MCA", "OneHotEncoder"):
            return model.transform_sql(X)
        else:
            if return_proba:
                return model.predict_proba_sql(X)
            else:
                return model.predict_sql(X)


class Supervised(vModel):
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
        does_model_exist(name=self.model_name, raise_error=True)
        id_column, id_column_name = "", gen_tmp_name(name="id_column")
        if self._model_type in (
            "RandomForestClassifier",
            "RandomForestRegressor",
            "XGBoostClassifier",
            "XGBoostRegressor",
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
            relation = gen_tmp_name(
                schema=schema_relation(self.model_name)[0], name="view"
            )
            drop(relation, method="view")
            _executeSQL(
                query=f"""
                    CREATE VIEW {relation} AS 
                        SELECT 
                            /*+LABEL('learn.vModel.fit')*/ 
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
        parameters = self.get_vertica_param_dict()
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
                /*+LABEL('learn.vModel.fit')*/ 
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
            "XGBoostClassifier",
            "XGBoostRegressor",
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
        if self._model_type in (
            "LinearSVC",
            "LinearSVR",
            "LogisticRegression",
            "LinearRegression",
            "SARIMAX",
        ):
            self.coef_ = self.get_attr("details")
        elif self._model_type in (
            "RandomForestClassifier",
            "NaiveBayes",
            "XGBoostClassifier",
        ):
            if not (isinstance(input_relation, vDataFrame)):
                classes = _executeSQL(
                    query=f"""
                        SELECT 
                            /*+LABEL('learn.vModel.fit')*/ 
                            DISTINCT {self.y} 
                        FROM {input_relation} 
                        WHERE {self.y} IS NOT NULL 
                        ORDER BY 1""",
                    method="fetchall",
                    print_time_sql=False,
                )
                self.classes_ = [c[0] for c in classes]
            else:
                self.classes_ = input_relation[self.y].distinct()
        if self._model_type in ("XGBoostClassifier", "XGBoostRegressor"):
            self.prior_ = self.get_prior()
        return self


class Tree:
    def to_graphviz(
        self,
        tree_id: int = 0,
        classes_color: list = [],
        round_pred: int = 2,
        percent: bool = False,
        vertical: bool = True,
        node_style: dict = {},
        arrow_style: dict = {},
        leaf_style: dict = {},
    ):
        """
        Returns the code for a Graphviz tree.

        Parameters
        ----------
        tree_id: int, optional
            Unique tree identifier, an integer in the range [0, n_estimators - 1].
        classes_color: list, optional
            Colors that represent the different classes.
        round_pred: int, optional
            The number of decimals to round the prediction to. 0 rounds to an integer.
        percent: bool, optional
            If set to True, the probabilities are returned as percents.
        vertical: bool, optional
            If set to True, the function generates a vertical tree.
        node_style: dict, optional
            Dictionary of options to customize each node of the tree. For a list of options, see
            the Graphviz API: https://graphviz.org/doc/info/attrs.html
        arrow_style: dict, optional
            Dictionary of options to customize each arrow of the tree. For a list of options, see
            the Graphviz API: https://graphviz.org/doc/info/attrs.html
        leaf_style: dict, optional
            Dictionary of options to customize each leaf of the tree. For a list of options, see
            the Graphviz API: https://graphviz.org/doc/info/attrs.html

        Returns
        -------
        str
            Graphviz code.
        """
        return self.to_memmodel(return_tree=tree_id).to_graphviz(
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
        name = (
            self.tree_name if self._model_type == "KernelDensity" else self.model_name
        )
        query = f"""SELECT * FROM (SELECT READ_TREE (USING PARAMETERS 
                                                     model_name = '{name}', 
                                                     tree_id = {tree_id}, 
                                                     format = 'tabular')) x ORDER BY node_id;"""
        result = to_tablesample(query=query, title="Reading Tree.")
        return result

    def plot_tree(
        self,
        pic_path: str = "",
        tree_id: int = 0,
        classes_color: list = [],
        round_pred: int = 2,
        percent: bool = False,
        vertical: bool = True,
        node_style: dict = {},
        arrow_style: dict = {},
        leaf_style: dict = {},
    ):
        """
        Draws the input tree. Requires the graphviz module.

        Parameters
        ----------
        pic_path: str, optional
            Absolute path to which the function saves the image of the tree.
        tree_id: int, optional
            Unique tree identifier, an integer in the range [0, n_estimators - 1].
        classes_color: list, optional
            Colors that represent the different classes.
        round_pred: int, optional
            The number of decimals to round the prediction to. 0 rounds to an integer.
        percent: bool, optional
            If set to True, the probabilities are returned as percents.
        vertical: bool, optional
            If set to True, the function generates a vertical tree.
        node_style: dict, optional
            Dictionary of options to customize each node of the tree. For a list of options, see
            the Graphviz API: https://graphviz.org/doc/info/attrs.html
        arrow_style: dict, optional
            Dictionary of options to customize each arrow of the tree. For a list of options, see
            the Graphviz API: https://graphviz.org/doc/info/attrs.html
        leaf_style: dict, optional
            Dictionary of options to customize each leaf of the tree. For a list of options, see
            the Graphviz API: https://graphviz.org/doc/info/attrs.html

        Returns
        -------
        graphviz.Source
            graphviz object.
        """
        return self.to_memmodel(return_tree=tree_id).plot_tree(
            feature_names=self.X,
            classes_color=classes_color,
            round_pred=round_pred,
            percent=percent,
            vertical=vertical,
            node_style=node_style,
            arrow_style=arrow_style,
            leaf_style=leaf_style,
            pic_path=pic_path,
        )

    def get_score(
        self, tree_id: int = None,
    ):
        """
        Returns the feature importance metrics for the input tree.

        Parameters
        ----------
        tree_id: int, optional
            Unique tree identifier, an integer in the range [0, n_estimators - 1].
            If tree_id is undefined, all the trees in the model are used to compute 
            the metrics.

        Returns
        -------
        TableSample
            An object containing the result. For more information, see
            utilities.TableSample.
        """
        name = (
            self.tree_name if self._model_type == "KernelDensity" else self.model_name
        )
        if self._model_type in ("XGBoostClassifier", "XGBoostRegressor",):
            vertica_version(condition=[12, 0, 3])
            fname = "XGB_PREDICTOR_IMPORTANCE"
        else:
            vertica_version(condition=[9, 1, 1])
            fname = "RF_PREDICTOR_IMPORTANCE"
        tree_id = "" if tree_id is None else f", tree_id={tree_id}"
        query = f"SELECT {fname} (USING PARAMETERS model_name = '{name}'{tree_id})"
        result = to_tablesample(query=query, title="Reading Tree.")
        return result


class Classifier(Supervised):
    pass


class BinaryClassifier(Classifier):

    classes_ = [0, 1]

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
        self, pos_label: Union[int, float, str] = None, cutoff: Union[int, float] = -1,
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
        self,
        pos_label: Union[int, float, str] = None,
        ax=None,
        nbins: int = 30,
        **style_kwds,
    ):
        """
    Draws the model Cutoff curve.

    Parameters
    ----------
    pos_label: int/float/str, optional
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
        pos_label: Union[int, float, str] = None,
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
        self,
        pos_label: Union[int, float, str] = None,
        ax=None,
        nbins: int = 1000,
        **style_kwds,
    ):
        """
	Draws the model Lift Chart.

	Parameters
	----------
	pos_label: int/float/str, optional
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
        self,
        pos_label: Union[int, float, str] = None,
        ax=None,
        nbins: int = 30,
        **style_kwds,
    ):
        """
	Draws the model PRC curve.

	Parameters
	----------
	pos_label: int/float/str, optional
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
        self,
        pos_label: Union[int, float, str] = None,
        ax=None,
        nbins: int = 30,
        **style_kwds,
    ):
        """
	Draws the model ROC curve.

	Parameters
	----------
	pos_label: int/float/str, optional
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
        pos_label: Union[int, float, str] = None,
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
        if method in ("anova", "details") and self._model_type in (
            "SARIMAX",
            "VAR",
            "KernelDensity",
        ):
            raise ModelError(
                f"'{method}' method is not available for {self._model_type} models."
            )
        prediction = self.deploySQL()
        if self._model_type == "SARIMAX":
            test_relation = self.transform_relation
            test_relation = f"""
                (SELECT 
                    {self.deploySQL()} AS prediction, 
                    VerticaPy_y_copy AS {self.y} 
                FROM {test_relation}) VERTICAPY_SUBTABLE"""
            test_relation = (
                test_relation.format(self.test_relation)
                .replace("[VerticaPy_ts]", self.ts)
                .replace("[VerticaPy_y]", self.y)
                .replace(
                    "[VerticaPy_key_columns]",
                    ", " + ", ".join([self.ts] + self.exogenous),
                )
            )
            for idx, elem in enumerate(self.exogenous):
                test_relation = test_relation.replace(f"[X{idx}]", elem)
            prediction = "prediction"
        elif self._model_type == "KNeighborsRegressor":
            test_relation = self.deploySQL()
            prediction = "predict_neighbors"
        elif self._model_type == "VAR":
            relation = self.transform_relation.replace(
                "[VerticaPy_ts]", self.ts
            ).format(self.test_relation)
            for idx, elem in enumerate(self.X):
                relation = relation.replace(f"[X{idx}]", elem)
            values = {
                "index": [
                    "explained_variance",
                    "max_error",
                    "median_absolute_error",
                    "mean_absolute_error",
                    "mean_squared_error",
                    "root_mean_squared_error",
                    "r2",
                    "r2_adj",
                    "aic",
                    "bic",
                ]
            }
            result = TableSample(values)
            for idx, y in enumerate(self.X):
                result.values[y] = mt.regression_report(
                    y,
                    self.deploySQL()[idx],
                    relation,
                    len(self.X) * self.parameters["p"],
                ).values["value"]
            return result
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


class Unsupervised(vModel):
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
                            /*+LABEL('learn.vModel.fit')*/ *
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
        parameters = self.get_vertica_param_dict()
        if "num_components" in parameters and not (parameters["num_components"]):
            del parameters["num_components"]
        fun = self._vertica_fit_sql if self._model_type != "MCA" else "PCA"
        query = f"""
            SELECT 
                /*+LABEL('learn.vModel.fit')*/ 
                {fun}('{self.model_name}', '{relation}', '{', '.join(self.X)}'"""
        if self._model_type in ("BisectingKMeans", "KMeans", "KPrototypes",):
            query += f", {parameters['n_cluster']}"
        elif self._model_type == "Normalizer":
            query += f", {parameters['method']}"
            del parameters["method"]
        if self._model_type not in ("Normalizer", "MCA"):
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
                        query0 += ["SELECT /*+LABEL('learn.vModel.fit')*/ " + line]
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
            if self._model_type in ("KMeans", "KPrototypes",):
                self.cluster_centers_ = self.get_attr("centers")
                result = self.get_attr("metrics").values["metrics"][0]
                values = {
                    "index": [
                        "Between-Cluster Sum of Squares",
                        "Total Sum of Squares",
                        "Total Within-Cluster Sum of Squares",
                        "Between-Cluster SS / Total SS",
                        "converged",
                    ]
                }
                values["value"] = [
                    float(
                        result.split("Between-Cluster Sum of Squares: ")[1].split("\n")[
                            0
                        ]
                    ),
                    float(result.split("Total Sum of Squares: ")[1].split("\n")[0]),
                    float(
                        result.split("Total Within-Cluster Sum of Squares: ")[1].split(
                            "\n"
                        )[0]
                    ),
                    float(
                        result.split("Between-Cluster Sum of Squares: ")[1].split("\n")[
                            0
                        ]
                    )
                    / float(result.split("Total Sum of Squares: ")[1].split("\n")[0]),
                    result.split("Converged: ")[1].split("\n")[0] == "True",
                ]
                self.metrics_ = TableSample(values)
            elif self._model_type == "BisectingKMeans":
                self.metrics_ = self.get_attr("Metrics")
                self.cluster_centers_ = self.get_attr("BKTree")
        elif self._model_type in ("PCA", "MCA"):
            self.components_ = self.get_attr("principal_components")
            if self._model_type == "MCA":
                self.cos2_ = self.components_.to_list()
                for i in range(len(self.cos2_)):
                    self.cos2_[i] = [elem ** 2 for elem in self.cos2_[i]]
                    total = sum(self.cos2_[i])
                    self.cos2_[i] = [elem / total for elem in self.cos2_[i]]
                values = {"index": self.X}
                for idx, elem in enumerate(self.components_.values):
                    if elem != "index":
                        values[elem] = [item[idx - 1] for item in self.cos2_]
                self.cos2_ = TableSample(values)
            self.explained_variance_ = self.get_attr("singular_values")
            self.mean_ = self.get_attr("columns")
        elif self._model_type == "SVD":
            self.singular_values_ = self.get_attr("right_singular_vectors")
            self.explained_variance_ = self.get_attr("singular_values")
        elif self._model_type == "Normalizer":
            self.param_ = self.get_attr("details")
        elif self._model_type == "OneHotEncoder":
            query = f"""SELECT 
                            category_name, 
                            category_level::varchar, 
                            category_level_index 
                        FROM (SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS 
                                        model_name = '{self.model_name}', 
                                        attr_name = 'integer_categories')) 
                                        VERTICAPY_SUBTABLE"""
            try:
                self.param_ = to_tablesample(
                    query=f"""{query}
                              UNION ALL 
                              SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS 
                                        model_name = '{self.model_name}', 
                                        attr_name = 'varchar_categories')""",
                    title="Getting Model Attributes.",
                )
            except:
                try:
                    self.param_ = to_tablesample(
                        query=query, title="Getting Model Attributes.",
                    )
                except:
                    self.param_ = self.get_attr("varchar_categories")
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
            raise ModelError(
                "method 'inverse_transform' is not supported for OneHotEncoder models."
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
                for i in range(len(self.param_["category_name"])):
                    if quote_ident(self.param_["category_name"][i]) == quote_ident(
                        column
                    ):
                        if (k != 0 or not (self.parameters["drop_first"])) and (
                            not (self.parameters["ignore_null"])
                            or self.param_["category_level"][i] != None
                        ):
                            if self.parameters["column_naming"] == "indices":
                                name = f'"{quote_ident(column)[1:-1]}{self.parameters["separator"]}'
                                name += f'{self.param_["category_level_index"][i]}"'
                                names += [name]
                            else:
                                if self.param_["category_level"][i] != None:
                                    category_level = self.param_["category_level"][
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
            raise ModelError(
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
        explained_variance = self.explained_variance_["explained_variance"]
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
            x = self.singular_values_[f"vector{dimensions[0]}"]
            y = self.singular_values_[f"vector{dimensions[1]}"]
        else:
            x = self.components_[f"PC{dimensions[0]}"]
            y = self.components_[f"PC{dimensions[1]}"]
        explained_variance = self.explained_variance_["explained_variance"]
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
        explained_variance = self.explained_variance_["explained_variance"]
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
        return to_tablesample(query, title="Getting Model Score.").transpose()

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


class Clustering(Unsupervised):
    @property
    @abstractmethod
    def _vertica_predict_sql(self) -> str:
        """Must be overridden in child class"""
        raise NotImplementedError

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
        if isinstance(X, str):
            X = [X]
        if isinstance(vdf, str):
            vdf = vDataFrame(vdf)
        X = [quote_ident(elem) for elem in X]
        if not (name):
            name = (
                self._model_type
                + "_"
                + "".join(ch for ch in self.model_name if ch.isalnum())
            )
        if inplace:
            return vdf.eval(name, self.deploySQL(X=X))
        else:
            return vdf.copy().eval(name, self.deploySQL(X=X))
