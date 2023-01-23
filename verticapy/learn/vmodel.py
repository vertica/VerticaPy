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
import os, warnings, typing, copy
import numpy as np
from typing import Union

# VerticaPy Modules
import verticapy
from verticapy.decorators import (
    save_verticapy_logs,
    check_dtypes,
    check_minimum_version,
)
from verticapy import vDataFrame
from verticapy.learn.mlplot import *
from verticapy.learn.model_selection import *
from verticapy.utilities import *
from verticapy.toolbox import *
from verticapy.errors import *
from verticapy.learn.metrics import *
from verticapy.learn.tools import *
from verticapy.learn.memmodel import *

##
#  ___      ___  ___      ___     ______    ________    _______  ___
# |"  \    /"  ||"  \    /"  |   /    " \  |"      "\  /"     "||"  |
#  \   \  //  /  \   \  //   |  // ____  \ (.  ___  :)(: ______)||  |
#   \\  \/. ./   /\\  \/.    | /  /    ) :)|: \   ) || \/    |  |:  |
#    \.    //   |: \.        |(: (____/ // (| (___\ || // ___)_  \  |___
#     \\   /    |.  \    /:  | \        /  |:       :)(:      "|( \_|:  \
#      \__/     |___|\__/|___|  \"_____/   (________/  \_______) \_______)
#
#
# ---#
class vModel:
    """
---------------------------------------------------------------------------
Main Class for Vertica Model
	"""

    # ---#
    def __repr__(self):
        """
	---------------------------------------------------------------------------
	Returns the model Representation.
		"""
        try:
            rep = ""
            if self.type not in (
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
                name = self.tree_name if self.type == "KernelDensity" else self.name
                try:
                    vertica_version(condition=[9, 0, 0])
                    res = executeSQL(
                        f"SELECT /*+LABEL('learn.vModel.__repr__')*/ GET_MODEL_SUMMARY(USING PARAMETERS model_name = '{name}')",
                        title="Summarizing the model.",
                        method="fetchfirstelem",
                    )
                except:
                    res = executeSQL(
                        f"SELECT /*+LABEL('learn.vModel.__repr__')*/ SUMMARIZE_MODEL('{name}')",
                        title="Summarizing the model.",
                        method="fetchfirstelem",
                    )
                return res
            elif self.type == "AutoML":
                rep = self.best_model_.__repr__()
            elif self.type == "AutoDataPrep":
                rep = self.final_relation_.__repr__()
            elif self.type == "DBSCAN":
                rep = f"=======\ndetails\n=======\nNumber of Clusters: {self.n_cluster_}\nNumber of Outliers: {self.n_noise_}"
            elif self.type == "LocalOutlierFactor":
                rep = f"=======\ndetails\n=======\nNumber of Errors: {self.n_errors_}"
            elif self.type == "NearestCentroid":
                rep = "=======\ndetails\n=======\n" + self.centroids_.__repr__()
            elif self.type == "VAR":
                rep = "=======\ndetails\n======="
                for idx, elem in enumerate(self.X):
                    rep += "\n\n # " + str(elem) + "\n\n" + self.coef_[idx].__repr__()
                rep += "\n\n===============\nAdditional Info\n==============="
                rep += f"\nInput Relation : {self.input_relation}"
                rep += f"\nX : {', '.join(self.X)}"
                rep += f"\nts : {self.ts}"
            elif self.type == "SARIMAX":
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
            elif self.type == "CountVectorizer":
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
            if self.type in (
                "DBSCAN",
                "NearestCentroid",
                "LocalOutlierFactor",
                "KNeighborsRegressor",
                "KNeighborsClassifier",
            ):
                rep += "\n\n===============\nAdditional Info\n==============="
                rep += f"\nInput Relation : {self.input_relation}"
                rep += f"\nX : {', '.join(self.X)}"
            if self.type in (
                "NearestCentroid",
                "KNeighborsRegressor",
                "KNeighborsClassifier",
            ):
                rep += f"\ny : {self.y}"
            return rep
        except:
            return f"<{self.type}>"

    # ---#
    def contour(
        self,
        nbins: int = 100,
        pos_label: Union[int, float, str] = None,
        ax=None,
        **style_kwds,
    ):
        """
    ---------------------------------------------------------------------------
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
        if self.type in (
            "RandomForestClassifier",
            "XGBoostClassifier",
            "NaiveBayes",
            "NearestCentroid",
            "KNeighborsClassifier",
        ):
            if not (pos_label):
                pos_label = sorted(self.classes_)[-1]
            if self.type in (
                "RandomForestClassifier",
                "XGBoostClassifier",
                "NaiveBayes",
                "NearestCentroid",
            ):
                return vDataFrameSQL(self.input_relation).contour(
                    self.X,
                    self.deploySQL(X=self.X, pos_label=pos_label),
                    cbar_title=self.y,
                    nbins=nbins,
                    ax=ax,
                    **style_kwds,
                )
            else:
                return vDataFrameSQL(self.input_relation).contour(
                    self.X,
                    self,
                    pos_label=pos_label,
                    cbar_title=self.y,
                    nbins=nbins,
                    ax=ax,
                    **style_kwds,
                )
        elif self.type == "KNeighborsRegressor":
            return vDataFrameSQL(self.input_relation).contour(
                self.X, self, cbar_title=self.y, nbins=nbins, ax=ax, **style_kwds
            )
        elif self.type in (
            "KMeans",
            "BisectingKMeans",
            "KPrototypes",
            "IsolationForest",
        ):
            cbar_title = "cluster"
            if self.type == "IsolationForest":
                cbar_title = "anomaly_score"
            return vDataFrameSQL(self.input_relation).contour(
                self.X, self, cbar_title=cbar_title, nbins=nbins, ax=ax, **style_kwds
            )
        else:
            return vDataFrameSQL(self.input_relation).contour(
                self.X,
                self.deploySQL(X=self.X),
                cbar_title=self.y,
                nbins=nbins,
                ax=ax,
                **style_kwds,
            )

    # ---#
    @check_dtypes
    def deploySQL(self, X: Union[str, list] = []):
        """
	---------------------------------------------------------------------------
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
        if self.type == "AutoML":
            return self.best_model_.deploySQL(X)
        if self.type not in ("DBSCAN", "LocalOutlierFactor"):
            name = self.tree_name if self.type == "KernelDensity" else self.name
            X = self.X if not (X) else [quote_ident(predictor) for predictor in X]
            sql = f"{self.VERTICA_PREDICT_FUNCTION_SQL}({', '.join(X)} USING PARAMETERS model_name = '{name}', match_by_pos = 'true')"
            return sql
        else:
            raise FunctionError(f"Method 'deploySQL' for '{self.type}' doesn't exist.")

    # ---#
    def drop(self):
        """
	---------------------------------------------------------------------------
	Drops the model from the Vertica database.
		"""
        drop(self.name, method="model", model_type=self.type)

    # ---#
    @check_dtypes
    def features_importance(
        self, ax=None, tree_id: int = None, show: bool = True, **style_kwds
    ):
        """
		---------------------------------------------------------------------------
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
		tablesample
			An object containing the result. For more information, see
			utilities.tablesample.
		"""
        if self.type == "AutoML":
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
        if self.type in (
            "RandomForestClassifier",
            "RandomForestRegressor",
            "KernelDensity",
            "XGBoostClassifier",
            "XGBoostRegressor",
        ):
            name = self.tree_name if self.type == "KernelDensity" else self.name
            if self.type in ("XGBoostClassifier", "XGBoostRegressor",):
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
        elif self.type in (
            "LinearRegression",
            "LogisticRegression",
            "LinearSVC",
            "LinearSVR",
        ):
            relation = self.input_relation
            vertica_version(condition=[8, 1, 1])
            query = """SELECT /*+LABEL('learn.vModel.features_importance')*/
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
                                            SUMMARIZE_NUMCOL({0}) OVER() 
                                          FROM {1}) VERTICAPY_SUBTABLE) stat 
                                          NATURAL JOIN ({2}) coeff) importance_t 
                                          ORDER BY 2 DESC;""".format(
                ", ".join(self.X), relation, self.coef_.to_sql()
            )
            print_legend = True
        else:
            raise FunctionError(
                "Method 'features_importance' for '{0}' doesn't exist.".format(
                    self.type
                )
            )
        result = executeSQL(
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
        return tablesample(values=importances).transpose()

    # ---#
    def get_attr(self, attr_name: str = ""):
        """
	---------------------------------------------------------------------------
	Returns the model attribute.

	Parameters
	----------
	attr_name: str, optional
		Attribute Name.

	Returns
	-------
	tablesample
		model attribute
		"""
        if self.type == "AutoML":
            return self.best_model_.get_attr(attr_name)
        if self.type not in (
            "DBSCAN",
            "LocalOutlierFactor",
            "VAR",
            "SARIMAX",
            "KNeighborsClassifier",
            "KNeighborsRegressor",
            "NearestCentroid",
            "CountVectorizer",
        ):
            name = self.tree_name if self.type == "KernelDensity" else self.name
            vertica_version(condition=[8, 1, 1])
            result = to_tablesample(
                query=(
                    "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS "
                    "model_name = '{0}'{1})"
                ).format(
                    name, ", attr_name = '{}'".format(attr_name) if attr_name else "",
                ),
                title="Getting Model Attributes.",
            )
            return result
        elif self.type == "DBSCAN":
            if attr_name == "n_cluster":
                return self.n_cluster_
            elif attr_name == "n_noise":
                return self.n_noise_
            elif not (attr_name):
                result = tablesample(
                    values={
                        "attr_name": ["n_cluster", "n_noise"],
                        "value": [self.n_cluster_, self.n_noise_],
                    },
                )
                return result
            else:
                raise ParameterError(f"Attribute '{attr_name}' doesn't exist.")
        elif self.type == "CountVectorizer":
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
                result = tablesample(
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
        elif self.type == "NearestCentroid":
            if attr_name == "p":
                return self.parameters["p"]
            elif attr_name == "centroids":
                return self.centroids_
            elif attr_name == "classes":
                return self.classes_
            elif not (attr_name):
                result = tablesample(
                    values={"attr_name": ["centroids", "classes", "p"],},
                )
                return result
            else:
                raise ParameterError(f"Attribute '{attr_name}' doesn't exist.")
        elif self.type == "KNeighborsClassifier":
            if attr_name == "p":
                return self.parameters["p"]
            elif attr_name == "n_neighbors":
                return self.parameters["n_neighbors"]
            elif attr_name == "classes":
                return self.classes_
            elif not (attr_name):
                result = tablesample(
                    values={"attr_name": ["n_neighbors", "p", "classes"],},
                )
                return result
            else:
                raise ParameterError(f"Attribute '{attr_name}' doesn't exist.")
        elif self.type == "KNeighborsRegressor":
            if attr_name == "p":
                return self.parameters["p"]
            elif attr_name == "n_neighbors":
                return self.parameters["n_neighbors"]
            elif not (attr_name):
                result = tablesample(values={"attr_name": ["n_neighbors", "p"],},)
                return result
            else:
                raise ParameterError(f"Attribute '{attr_name}' doesn't exist.")
        elif self.type == "LocalOutlierFactor":
            if attr_name == "n_errors":
                return self.n_errors_
            elif not (attr_name):
                result = tablesample(
                    values={"attr_name": ["n_errors"], "value": [self.n_errors_]},
                )
                return result
            else:
                raise ParameterError(f"Attribute '{attr_name}' doesn't exist.")
        elif self.type == "SARIMAX":
            if attr_name == "coefficients":
                return self.coef_
            elif attr_name == "ma_avg":
                return self.ma_avg_
            elif attr_name == "ma_piq":
                return self.ma_piq_
            elif not (attr_name):
                result = tablesample(
                    values={"attr_name": ["coefficients", "ma_avg", "ma_piq"]},
                )
                return result
            else:
                raise ParameterError(f"Attribute '{attr_name}' doesn't exist.")
        elif self.type == "VAR":
            if attr_name == "coefficients":
                return self.coef_
            elif not (attr_name):
                result = tablesample(values={"attr_name": ["coefficients"]})
                return result
            else:
                raise ParameterError(f"Attribute '{attr_name}' doesn't exist.")
        elif self.type == "KernelDensity":
            if attr_name == "map":
                return self.map_
            elif not (attr_name):
                result = tablesample(values={"attr_name": ["map"]})
                return result
            else:
                raise ParameterError(f"Attribute '{attr_name}' doesn't exist.")
        else:
            raise FunctionError(
                "Method 'get_attr' for '{0}' doesn't exist.".format(self.type)
            )

    # ---#
    def get_params(self):
        """
	---------------------------------------------------------------------------
	Returns the parameters of the model.

	Returns
	-------
	dict
		model parameters
		"""
        all_init_params = list(typing.get_type_hints(self.__init__).keys())
        parameters = copy.deepcopy(self.parameters)
        parameters_keys = list(parameters.keys())
        for p in parameters_keys:
            if p not in all_init_params:
                del parameters[p]
        return parameters

    # ---#
    def get_vertica_param_dict(self):
        """
    ---------------------------------------------------------------------------
    Returns the Vertica parameters dict to use when fitting the
    model. As some model's parameters names are not the same in
    Vertica. It is important to map them.

    Returns
    -------
    dict
        vertica parameters
        """

        def map_to_vertica_param_name(param: str):

            if param.lower() == "class_weights":
                return "class_weight"

            elif param.lower() == "solver":
                return "optimizer"

            elif param.lower() == "tol":
                return "epsilon"

            elif param.lower() == "max_iter":
                return "max_iterations"

            elif param.lower() == "penalty":
                return "regularization"

            elif param.lower() == "C":
                return "lambda"

            elif param.lower() == "l1_ratio":
                return "alpha"

            elif param.lower() == "n_estimators":
                return "ntree"

            elif param.lower() == "max_features":
                return "mtry"

            elif param.lower() == "sample":
                return "sampling_size"

            elif param.lower() == "max_leaf_nodes":
                return "max_breadth"

            elif param.lower() == "min_samples_leaf":
                return "min_leaf_size"

            elif param.lower() == "n_components":
                return "num_components"

            elif param.lower() == "init":
                return "init_method"

            else:
                return param

        parameters = {}

        for param in self.parameters:

            if self.type in ("LinearSVC", "LinearSVR") and param == "C":
                parameters[param] = self.parameters[param]

            elif (
                self.type in ("LinearRegression", "LogisticRegression") and param == "C"
            ):
                parameters["lambda"] = self.parameters[param]

            elif self.type == "BisectingKMeans" and param in (
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
                    parameters["class_weights"] = "'{}'".format(
                        ", ".join([str(item) for item in self.parameters[param]])
                    )
                else:
                    parameters["class_weights"] = "'{}'".format(self.parameters[param])

            elif isinstance(self.parameters[param], (str, dict)):
                parameters[map_to_vertica_param_name(param)] = "'{}'".format(
                    self.parameters[param]
                )

            else:
                parameters[map_to_vertica_param_name(param)] = self.parameters[param]

        return parameters

    # ---#
    @check_dtypes
    def plot(self, max_nb_points: int = 100, ax=None, **style_kwds):
        """
	---------------------------------------------------------------------------
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
        if self.type in (
            "LinearRegression",
            "LogisticRegression",
            "LinearSVC",
            "LinearSVR",
        ):
            coefficients = self.coef_.values["coefficient"]
            if self.type == "LogisticRegression":
                return logit_plot(
                    self.X,
                    self.y,
                    self.input_relation,
                    coefficients,
                    max_nb_points,
                    ax=ax,
                    **style_kwds,
                )
            elif self.type == "LinearSVC":
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
        elif self.type in (
            "KMeans",
            "BisectingKMeans",
            "KPrototypes",
            "DBSCAN",
            "IsolationForest",
        ):
            if self.type in ("KMeans", "BisectingKMeans", "KPrototypes",):
                if self.type == "KPrototypes":
                    if any(
                        [
                            ("char" in self.cluster_centers_.dtype[key].lower())
                            for key in self.cluster_centers_.dtype
                        ]
                    ):
                        raise TypeError(
                            "KPrototypes' plots with categorical inputs is not yet supported."
                        )
                vdf = vDataFrameSQL(self.input_relation)
                catcol = "{0}_cluster".format(self.type.lower())
                self.predict(vdf, name=catcol)
            elif self.type == "DBSCAN":
                vdf = vDataFrameSQL(self.name)
                catcol = "dbscan_cluster"
            elif self.type == "IsolationForest":
                vdf = vDataFrameSQL(self.input_relation)
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
        elif self.type == "LocalOutlierFactor":
            query = "SELECT /*+LABEL('learn.vModel.plot')*/ COUNT(*) FROM {}".format(
                self.name
            )
            tablesample = 100 * min(
                float(
                    max_nb_points
                    / executeSQL(query, method="fetchfirstelem", print_time_sql=False)
                ),
                1,
            )
            return lof_plot(self.name, self.X, "lof_score", 100, ax=ax, **style_kwds)
        elif self.type in ("RandomForestRegressor", "XGBoostRegressor"):
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
                "Method 'plot' for '{}' doesn't exist.".format(self.type)
            )

    # ---#
    def set_params(self, parameters: dict = {}):
        """
	---------------------------------------------------------------------------
	Sets the parameters of the model.

	Parameters
	----------
	parameters: dict, optional
		New parameters.
		"""
        all_init_params = list(typing.get_type_hints(self.__init__).keys())
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
        self.__init__(name=self.name, **new_parameters)

    # ---#
    def to_memmodel(self, **kwds):
        """
    ---------------------------------------------------------------------------
    Converts a specified Vertica model to a memModel model.

    Returns
    -------
    object
        memModel model.
        """
        from verticapy.learn.memmodel import memModel
        from verticapy.learn.tree import get_tree_list_of_arrays

        if self.type == "AutoML":
            return self.best_model_.to_memmodel()
        elif self.type in (
            "LinearRegression",
            "LogisticRegression",
            "LinearSVC",
            "LinearSVR",
        ):
            attributes = {
                "coefficients": self.coef_["coefficient"][1:],
                "intercept": self.coef_["coefficient"][0],
            }
        elif self.type == "BisectingKMeans":
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
        elif self.type in ("KMeans", "KPrototypes"):
            attributes = {"clusters": self.cluster_centers_.to_numpy(), "p": 2}
            if self.type == "KPrototypes":
                attributes["gamma"] = self.parameters["gamma"]
                attributes["is_categorical"] = [
                    ("char" in self.cluster_centers_.dtype[key].lower())
                    for key in self.cluster_centers_.dtype
                ]
        elif self.type == "NearestCentroid":
            attributes = {
                "clusters": self.centroids_.to_numpy()[:, 0:-1],
                "p": self.parameters["p"],
                "classes": self.classes_,
            }
        elif self.type == "NaiveBayes":
            attributes = {
                "attributes": self.get_var_info(),
                "prior": self.get_attr("prior")["probability"],
                "classes": self.classes_,
            }
        elif self.type == "OneHotEncoder":

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
        elif self.type == "PCA":
            attributes = {
                "principal_components": self.get_attr(
                    "principal_components"
                ).to_numpy(),
                "mean": self.get_attr("columns")["mean"],
            }
        elif self.type == "SVD":
            attributes = {
                "vectors": self.get_attr("right_singular_vectors").to_numpy(),
                "values": self.get_attr("singular_values")["value"],
            }
        elif self.type == "Normalizer":
            attributes = {
                "values": self.get_attr("details").to_numpy()[:, 1:].astype(float),
                "method": self.parameters["method"],
            }
        elif self.type in (
            "XGBoostClassifier",
            "XGBoostRegressor",
            "RandomForestClassifier",
            "RandomForestRegressor",
            "IsolationForest",
        ):
            if self.type in (
                "RandomForestClassifier",
                "RandomForestRegressor",
                "IsolationForest",
            ):
                n = self.parameters["n_estimators"]
            else:
                n = self.get_attr("tree_count")["tree_count"][0]
            trees = []
            if self.type in ("XGBoostRegressor", "RandomForestRegressor"):
                tree_type = "BinaryTreeRegressor"
            elif self.type in ("XGBoostClassifier", "RandomForestClassifier"):
                tree_type = "BinaryTreeClassifier"
            else:
                tree_type = "BinaryTreeAnomaly"
            return_prob_rf = self.type == "RandomForestClassifier"
            is_iforest = self.type == "IsolationForest"
            for i in range(n):
                if ("return_tree" in kwds and kwds["return_tree"] == i) or (
                    "return_tree" not in kwds
                ):
                    tree = self.get_tree(i)
                    tree = get_tree_list_of_arrays(
                        tree, self.X, self.type, return_prob_rf
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
                    if self.type == "XGBoostClassifier":
                        tree_attributes["value"] = tree[6]
                        for idx in range(len(tree[6])):
                            if tree[6][idx] != None:
                                all_classes_logodss = []
                                for c in self.classes_:
                                    all_classes_logodss += [tree[6][idx][str(c)]]
                                tree_attributes["value"][idx] = all_classes_logodss
                    elif self.type == "RandomForestClassifier":
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
            if self.type in ("XGBoostRegressor", "XGBoostClassifier"):
                attributes["learning_rate"] = self.parameters["learning_rate"]
                if self.type == "XGBoostRegressor":
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
                f"Model type '{self.type}' can not be converted to memModel."
            )
        return memModel(model_type=self.type, attributes=attributes)

    # ---#
    def to_python(
        self,
        name: str = "predict",
        return_proba: bool = False,
        return_distance_clusters: bool = False,
        return_str: bool = False,
    ):
        """
    ---------------------------------------------------------------------------
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
        from verticapy.learn.tree import get_tree_list_of_arrays

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
        if self.type in (
            "LinearRegression",
            "LinearSVR",
            "LogisticRegression",
            "LinearSVC",
        ):
            result = "{} + np.sum(np.array({}) * np.array(X), axis=1)".format(
                self.coef_["coefficient"][0], self.coef_["coefficient"][1:]
            )
            if self.type in ("LogisticRegression", "LinearSVC"):
                func += f"result = 1 / (1 + np.exp(- ({result})))"
            else:
                func += "result =  " + result
            if return_proba and self.type in ("LogisticRegression", "LinearSVC"):
                func += "\n\treturn np.column_stack((1 - result, result))"
            elif not (return_proba) and self.type in (
                "LogisticRegression",
                "LinearSVC",
            ):
                func += "\n\treturn np.where(result > 0.5, 1, 0)"
            else:
                func += "\n\treturn result"
            return func
        elif self.type == "BisectingKMeans":
            bktree = self.get_attr("BKTree")
            cluster = [elem[1:-7] for elem in bktree.to_list()]
            func += "centroids = np.array({})\n".format(cluster)
            func += "\tright_child = {}\n".format(bktree["right_child"])
            func += "\tleft_child = {}\n".format(bktree["left_child"])
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
        elif self.type in ("KPrototypes",):
            centroids = self.cluster_centers_.to_list()
            func += "centroids = np.array({})\n\n".format(centroids)
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
            func += "\t\t\tdistance_final = distance_num + {0} * distance_cat\n".format(
                self.parameters["gamma"]
            )
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
        elif self.type in ("NearestCentroid", "KMeans",):
            centroids = (
                self.centroids_.to_list()
                if self.type == "NearestCentroid"
                else self.cluster_centers_.to_list()
            )
            if self.type == "NearestCentroid":
                for center in centroids:
                    del center[-1]
            func += "centroids = np.array({})\n".format(centroids)
            if self.type == "NearestCentroid":
                func += "\tclasses = np.array({})\n".format(self.classes_)
            func += "\tresult = []\n"
            func += "\tfor centroid in centroids:\n"
            func += "\t\tresult += [np.sum((np.array(centroid) - X) ** {0}, axis=1) ** (1 / {0})]\n".format(
                self.parameters["p"] if self.type == "NearestCentroid" else 2,
            )
            func += "\tresult = np.column_stack(result)\n"
            if (
                self.type == "NearestCentroid"
                and return_proba
                and not (return_distance_clusters)
            ):
                func += "\tresult = 1 / (result + 1e-99) / np.sum(1 / (result + 1e-99), axis=1)[:,None]\n"
            elif not (return_distance_clusters):
                func += "\tresult = np.argmin(result, axis=1)\n"
                if self.type == "NearestCentroid" and self.classes_ != [
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
        elif self.type in ("PCA", "MCA"):
            avg = self.get_attr("columns")["mean"]
            pca = []
            attr = self.get_attr("principal_components")
            n = len(attr["PC1"])
            for i in range(1, n + 1):
                pca += [attr["PC{}".format(i)]]
            func += f"avg_values = np.array({avg})\n"
            func += f"\tpca_values = np.array({pca})\n"
            func += "\tresult = (X - avg_values)\n"
            func += "\tL = []\n"
            func += f"\tfor i in range({n}):\n"
            func += "\t\tL += [np.sum(result * pca_values[i], axis=1)]\n"
            func += "\tresult = np.column_stack(L)\n"
            func += "\treturn result\n"
            return func
        elif self.type == "Normalizer":
            details = self.get_attr("details")
            sql = []
            if "min" in details.values:
                func += "min_values = np.array({})\n".format(details["min"])
                func += "\tmax_values = np.array({})\n".format(details["max"])
                func += "\treturn (X - min_values) / (max_values - min_values)\n"
            elif "median" in details.values:
                func += "median_values = np.array({})\n".format(details["median"])
                func += "\tmad_values = np.array({})\n".format(details["mad"])
                func += "\treturn (X - median_values) / mad_values\n"
            else:
                func += "avg_values = np.array({})\n".format(details["avg"])
                func += "\tstd_values = np.array({})\n".format(details["std_dev"])
                func += "\treturn (X - avg_values) / std_values\n"
            return func
        elif self.type == "SVD":
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
            func += "\t\tL += [np.sum(X * right_singular_vectors[i] / singular_values[i], axis=1)]\n"
            func += "\tresult = np.column_stack(L)\n"
            func += "\treturn result\n"
            return func
        elif self.type == "NaiveBayes":
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
        elif self.type == "OneHotEncoder":
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
            func += "category_level = {}\n\t".format(category_level)
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
        elif self.type in (
            "RandomForestClassifier",
            "RandomForestRegressor",
            "XGBoostRegressor",
            "XGBoostClassifier",
            "IsolationForest",
        ):
            result = []
            if self.type in (
                "RandomForestClassifier",
                "RandomForestRegressor",
                "IsolationForest",
            ):
                n = self.parameters["n_estimators"]
            else:
                n = self.get_attr("tree_count")["tree_count"][0]
            func += "n = {}\n".format(n)
            if self.type in ("XGBoostClassifier", "RandomForestClassifier"):
                func += "\tclasses = np.array({})\n".format(
                    [str(elem) for elem in self.classes_]
                )
            func += "\ttree_list = []\n"
            for i in range(n):
                tree = self.get_tree(i)
                func += "\ttree_list += [{}]\n".format(
                    get_tree_list_of_arrays(tree, self.X, self.type)
                )
            if self.type == "IsolationForest":
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
            if self.type in ("RandomForestRegressor", "XGBoostRegressor",):
                func += "\t\t\treturn float(tree[4][node_id])\n"
            elif self.type == "IsolationForest":
                psy = int(
                    self.parameters["sample"]
                    * int(self.get_attr("accepted_row_count")["accepted_row_count"][0])
                )
                func += "\t\t\treturn (tree[4][node_id][0] + heuristic_length(tree[4][node_id][1])) / heuristic_length({})\n".format(
                    psy
                )
            elif self.type == "RandomForestClassifier":
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
            if self.type in ("XGBoostClassifier", "RandomForestClassifier"):
                func += "\t\tall_classes_score = {}\n"
                func += "\t\tfor elem in classes:\n"
                func += "\t\t\tall_classes_score[elem] = 0\n"
            if self.type in ("XGBoostRegressor", "XGBoostClassifier"):
                if self.type == "XGBoostRegressor":
                    avg = self.prior_
                    func += "\t\treturn {} + {} * np.sum(result)\n".format(
                        avg, self.parameters["learning_rate"]
                    )
                else:
                    if not (isinstance(self.prior_, list)):
                        func += "\t\tlogodds = np.array([{}, {}])\n".format(
                            np.log((1 - self.prior_) / self.prior_),
                            np.log(self.prior_ / (1 - self.prior_)),
                        )
                    else:
                        func += "\t\tlogodds = np.array({})\n".format(self.prior_)
                    func += "\t\tfor idx, elem in enumerate(all_classes_score):\n"
                    func += "\t\t\tfor val in result:\n"
                    func += "\t\t\t\tall_classes_score[elem] += val[elem]\n"
                    func += "\t\t\tall_classes_score[elem] = 1 / (1 + np.exp( - "
                    func += "(logodds[idx] + {} * all_classes_score[elem])))\n".format(
                        self.parameters["learning_rate"]
                    )
                    func += "\t\tresult = [all_classes_score[elem] for elem in "
                    func += "all_classes_score]\n"
            elif self.type == "RandomForestRegressor":
                func += "\t\treturn np.mean(result)\n"
            elif self.type == "IsolationForest":
                func += "\t\treturn 2 ** ( - np.mean(result))\n"
            else:
                func += "\t\tfor elem in result:\n"
                func += "\t\t\tif elem in all_classes_score:\n"
                func += "\t\t\t\tall_classes_score[elem] += 1\n"
                func += "\t\tresult = []\n"
                func += "\t\tfor elem in all_classes_score:\n"
                func += "\t\t\tresult += [all_classes_score[elem]]\n"
            if self.type in ("XGBoostClassifier", "RandomForestClassifier"):
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
                "Function to_python not yet available for model type '{}'.".format(
                    self.type
                )
            )

    # ---#
    def to_sql(self, X: list = [], return_proba: bool = False):
        """
    ---------------------------------------------------------------------------
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
        if self.type in ("PCA", "SVD", "Normalizer", "MCA", "OneHotEncoder"):
            return self.to_memmodel().transform_sql(self.X if not X else X)
        else:
            if return_proba:
                return self.to_memmodel().predict_proba_sql(self.X if not X else X)
            else:
                return self.to_memmodel().predict_sql(self.X if not X else X)


# ---#
class Supervised(vModel):

    # ---#
    @check_dtypes
    def fit(
        self,
        input_relation: Union[str, vDataFrame],
        X: Union[str, list],
        y: str,
        test_relation: Union[str, vDataFrame] = "",
    ):
        """
	---------------------------------------------------------------------------
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
        if verticapy.OPTIONS["overwrite_model"]:
            self.drop()
        else:
            does_model_exist(name=self.name, raise_error=True)
        self.X = [quote_ident(column) for column in X]
        self.y = quote_ident(y)
        if (self.type == "NaiveBayes") and (
            self.parameters["nbtype"]
            in ("bernoulli", "categorical", "multinomial", "gaussian")
        ):
            new_types = {}
            for elem in X:
                if self.parameters["nbtype"] == "bernoulli":
                    new_types[elem] = "bool"
                elif self.parameters["nbtype"] == "categorical":
                    new_types[elem] = "varchar"
                elif self.parameters["nbtype"] == "multinomial":
                    new_types[elem] = "int"
                elif self.parameters["nbtype"] == "gaussian":
                    new_types[elem] = "float"
            if not (isinstance(input_relation, vDataFrame)):
                input_relation = vDataFrameSQL(input_relation)
            else:
                input_relation.copy()
            input_relation.astype(new_types)
        does_model_exist(name=self.name, raise_error=True)
        id_column, id_column_name = "", gen_tmp_name(name="id_column")
        if self.type in (
            "RandomForestClassifier",
            "RandomForestRegressor",
            "XGBoostClassifier",
            "XGBoostRegressor",
        ) and isinstance(verticapy.OPTIONS["random_state"], int):
            id_column = ", ROW_NUMBER() OVER (ORDER BY {0}) AS {1}".format(
                ", ".join(X), id_column_name
            )
        tmp_view = False
        if isinstance(input_relation, vDataFrame) or (id_column):
            tmp_view = True
            if isinstance(input_relation, vDataFrame):
                self.input_relation = input_relation.__genSQL__()
            else:
                self.input_relation = input_relation
            relation = gen_tmp_name(schema=schema_relation(self.name)[0], name="view")
            drop(relation, method="view")
            executeSQL(
                "CREATE VIEW {0} AS SELECT /*+LABEL('learn.vModel.fit')*/ *{1} FROM {2}".format(
                    relation, id_column, self.input_relation
                ),
                title="Creating a temporary view to fit the model.",
            )
        else:
            self.input_relation = input_relation
            relation = input_relation
        if isinstance(test_relation, vDataFrame):
            self.test_relation = test_relation.__genSQL__()
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
        fun = self.VERTICA_FIT_FUNCTION_SQL
        query = "SELECT /*+LABEL('learn.vModel.fit')*/ {}('{}', '{}', '{}', '{}' USING PARAMETERS "
        query = query.format(fun, self.name, relation, self.y, ", ".join(self.X))
        query += ", ".join(
            ["{} = {}".format(elem, parameters[elem]) for elem in parameters]
        )
        if alpha != None:
            query += f", alpha = {alpha}"
        if self.type in (
            "RandomForestClassifier",
            "RandomForestRegressor",
            "XGBoostClassifier",
            "XGBoostRegressor",
        ) and isinstance(verticapy.OPTIONS["random_state"], int):
            query += ", seed={}, id_column='{}'".format(
                verticapy.OPTIONS["random_state"], id_column_name
            )
        query += ")"
        try:
            executeSQL(query, title="Fitting the model.")
            if tmp_view:
                drop(relation, method="view")
        except:
            if tmp_view:
                drop(relation, method="view")
            raise
        if self.type in (
            "LinearSVC",
            "LinearSVR",
            "LogisticRegression",
            "LinearRegression",
            "SARIMAX",
        ):
            self.coef_ = self.get_attr("details")
        elif self.type in ("RandomForestClassifier", "NaiveBayes", "XGBoostClassifier"):
            if not (isinstance(input_relation, vDataFrame)):
                classes = executeSQL(
                    "SELECT /*+LABEL('learn.vModel.fit')*/ DISTINCT {} FROM {} WHERE {} IS NOT NULL ORDER BY 1".format(
                        self.y, input_relation, self.y
                    ),
                    method="fetchall",
                    print_time_sql=False,
                )
                self.classes_ = [item[0] for item in classes]
            else:
                self.classes_ = input_relation[self.y].distinct()
        if self.type in ("XGBoostClassifier", "XGBoostRegressor"):
            self.prior_ = self.get_prior()
        return self


# ---#
class Tree:

    # ---#
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
        ---------------------------------------------------------------------------
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

    # ---#
    @check_minimum_version
    @check_dtypes
    def get_tree(self, tree_id: int = 0):
        """
	---------------------------------------------------------------------------
	Returns a table with all the input tree information.

	Parameters
	----------
	tree_id: int, optional
        Unique tree identifier, an integer in the range [0, n_estimators - 1].

	Returns
	-------
	tablesample
		An object containing the result. For more information, see
		utilities.tablesample.
		"""
        name = self.tree_name if self.type == "KernelDensity" else self.name
        query = """SELECT * FROM (SELECT READ_TREE ( USING PARAMETERS 
                                        model_name = '{0}', 
                                        tree_id = {1}, 
                                        format = 'tabular')) x ORDER BY node_id;""".format(
            name, tree_id
        )
        result = to_tablesample(query=query, title="Reading Tree.")
        return result

    # ---#
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
        ---------------------------------------------------------------------------
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

    # ---#
    @check_dtypes
    def get_score(
        self, tree_id: int = None,
    ):
        """
        ---------------------------------------------------------------------------
        Returns the feature importance metrics for the input tree.

        Parameters
        ----------
        tree_id: int, optional
            Unique tree identifier, an integer in the range [0, n_estimators - 1].
            If tree_id is undefined, all the trees in the model are used to compute 
            the metrics.

        Returns
        -------
        tablesample
            An object containing the result. For more information, see
            utilities.tablesample.
        """
        name = self.tree_name if self.type == "KernelDensity" else self.name
        if self.type in ("XGBoostClassifier", "XGBoostRegressor",):
            vertica_version(condition=[12, 0, 3])
            fname = "XGB_PREDICTOR_IMPORTANCE"
        else:
            vertica_version(condition=[9, 1, 1])
            fname = "RF_PREDICTOR_IMPORTANCE"
        tree_id = "" if tree_id is None else f", tree_id={tree_id}"
        query = f"SELECT {fname} (USING PARAMETERS model_name = '{name}'{tree_id})"
        result = to_tablesample(query=query, title="Reading Tree.")
        return result


# ---#
class Classifier(Supervised):
    pass


# ---#
class BinaryClassifier(Classifier):

    classes_ = [0, 1]

    # ---#
    @check_dtypes
    def classification_report(
        self, cutoff: Union[int, float] = 0.5, nbins: int = 10000,
    ):
        """
	---------------------------------------------------------------------------
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
	tablesample
		An object containing the result. For more information, see
		utilities.tablesample.
		"""
        if cutoff > 1 or cutoff < 0:
            cutoff = self.score(method="best_cutoff")
        return classification_report(
            self.y,
            [self.deploySQL(), self.deploySQL(cutoff)],
            self.test_relation,
            cutoff=cutoff,
            nbins=nbins,
        )

    report = classification_report

    # ---#
    @check_dtypes
    def confusion_matrix(self, cutoff: Union[int, float] = 0.5):
        """
	---------------------------------------------------------------------------
	Computes the model confusion matrix.

	Parameters
	----------
	cutoff: int / float, optional
		Probability cutoff.

	Returns
	-------
	tablesample
		An object containing the result. For more information, see
		utilities.tablesample.
		"""
        return confusion_matrix(self.y, self.deploySQL(cutoff), self.test_relation,)

    # ---#
    @check_dtypes
    def deploySQL(self, cutoff: Union[int, float] = -1, X: Union[str, list] = []):
        """
	---------------------------------------------------------------------------
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
        if isinstance(X, str):
            X = [X]
        X = self.X if not (X) else [quote_ident(elem) for elem in X]
        sql = f"{self.VERTICA_PREDICT_FUNCTION_SQL}({', '.join(X)} USING PARAMETERS model_name = '{self.name}', type = 'probability', match_by_pos = 'true')"
        if cutoff <= 1 and cutoff >= 0:
            sql = f"(CASE WHEN {sql} >= {cutoff} THEN 1 WHEN {sql} IS NULL THEN NULL ELSE 0 END)"
        return sql

    # ---#
    def lift_chart(self, ax=None, nbins: int = 1000, **style_kwds):
        """
	---------------------------------------------------------------------------
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
	tablesample
		An object containing the result. For more information, see
		utilities.tablesample.
		"""
        return lift_chart(
            self.y,
            self.deploySQL(),
            self.test_relation,
            ax=ax,
            nbins=nbins,
            **style_kwds,
        )

    # ---#
    def prc_curve(self, ax=None, nbins: int = 30, **style_kwds):
        """
	---------------------------------------------------------------------------
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
	tablesample
		An object containing the result. For more information, see
		utilities.tablesample.
		"""
        return prc_curve(
            self.y,
            self.deploySQL(),
            self.test_relation,
            ax=ax,
            nbins=nbins,
            **style_kwds,
        )

    # ---#
    def predict(
        self,
        vdf: Union[str, vDataFrame],
        X: Union[str, list] = [],
        name: str = "",
        cutoff: Union[int, float] = 0.5,
        inplace: bool = True,
    ):
        """
	---------------------------------------------------------------------------
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
            vdf = vDataFrameSQL(relation=vdf)
        X = [quote_ident(elem) for elem in X]
        if not (name):
            name = gen_name([self.type, self.name])

        # In Place
        vdf_return = vdf if inplace else vdf.copy()

        # Result
        return vdf_return.eval(name, self.deploySQL(cutoff=cutoff, X=X))

    # ---#
    @check_dtypes
    def predict_proba(
        self,
        vdf: Union[str, vDataFrame],
        X: Union[str, list] = [],
        name: str = "",
        pos_label: Union[str, int, float] = None,
        inplace: bool = True,
    ):
        """
    ---------------------------------------------------------------------------
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
            vdf = vDataFrameSQL(relation=vdf)
        X = [quote_ident(elem) for elem in X]
        if not (name):
            name = gen_name([self.type, self.name])

        # In Place
        vdf_return = vdf if inplace else vdf.copy()

        # Result
        name_tmp = name
        if pos_label in [0, "0", None]:
            if pos_label == None:
                name_tmp = f"{name}_0"
            vdf_return.eval(name_tmp, "1 - {0}".format(self.deploySQL(X=X)))
        if pos_label in [1, "1", None]:
            if pos_label == None:
                name_tmp = f"{name}_1"
            vdf_return.eval(name_tmp, self.deploySQL(X=X))

        return vdf_return

    # ---#
    def cutoff_curve(self, ax=None, nbins: int = 30, **style_kwds):
        """
    ---------------------------------------------------------------------------
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
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.
        """
        return roc_curve(
            self.y,
            self.deploySQL(),
            self.test_relation,
            ax=ax,
            cutoff_curve=True,
            nbins=nbins,
            **style_kwds,
        )

    # ---#
    def roc_curve(self, ax=None, nbins: int = 30, **style_kwds):
        """
	---------------------------------------------------------------------------
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
	tablesample
		An object containing the result. For more information, see
		utilities.tablesample.
		"""
        return roc_curve(
            self.y,
            self.deploySQL(),
            self.test_relation,
            ax=ax,
            nbins=nbins,
            **style_kwds,
        )

    # ---#
    @check_dtypes
    def score(
        self,
        method: str = "accuracy",
        cutoff: Union[int, float] = 0.5,
        nbins: int = 10000,
    ):
        """
	---------------------------------------------------------------------------
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
        if method in ("accuracy", "acc"):
            return accuracy_score(
                self.y, self.deploySQL(cutoff), self.test_relation, pos_label=1
            )
        elif method == "aic":
            return aic_bic(self.y, self.deploySQL(), self.test_relation, len(self.X))[0]
        elif method == "bic":
            return aic_bic(self.y, self.deploySQL(), self.test_relation, len(self.X))[1]
        elif method == "prc_auc":
            return prc_curve(
                self.y, self.deploySQL(), self.test_relation, auc_prc=True, nbins=nbins
            )
        elif method == "auc":
            return roc_curve(
                self.y, self.deploySQL(), self.test_relation, auc_roc=True, nbins=nbins
            )
        elif method in ("best_cutoff", "best_threshold"):
            return roc_curve(
                self.y,
                self.deploySQL(),
                self.test_relation,
                best_threshold=True,
                nbins=nbins,
            )
        elif method in ("recall", "tpr"):
            return recall_score(self.y, self.deploySQL(cutoff), self.test_relation)
        elif method in ("precision", "ppv"):
            return precision_score(self.y, self.deploySQL(cutoff), self.test_relation)
        elif method in ("specificity", "tnr"):
            return specificity_score(self.y, self.deploySQL(cutoff), self.test_relation)
        elif method in ("negative_predictive_value", "npv"):
            return precision_score(self.y, self.deploySQL(cutoff), self.test_relation)
        elif method in ("log_loss", "logloss"):
            return log_loss(self.y, self.deploySQL(), self.test_relation)
        elif method == "f1":
            return f1_score(self.y, self.deploySQL(cutoff), self.test_relation)
        elif method == "mcc":
            return matthews_corrcoef(self.y, self.deploySQL(cutoff), self.test_relation)
        elif method in ("bm", "informedness"):
            return informedness(self.y, self.deploySQL(cutoff), self.test_relation)
        elif method in ("mk", "markedness"):
            return markedness(self.y, self.deploySQL(cutoff), self.test_relation)
        elif method in ("csi", "critical_success_index"):
            return critical_success_index(
                self.y, self.deploySQL(cutoff), self.test_relation
            )
        else:
            raise ParameterError(
                "The parameter 'method' must be in accuracy|auc|prc_auc|best_cutoff|recall"
                "|precision|log_loss|negative_predictive_value|specificity|mcc|informedness"
                "|markedness|critical_success_index|aic|bic"
            )


# ---#
class MulticlassClassifier(Classifier):

    # ---#
    @check_dtypes
    def classification_report(
        self,
        cutoff: Union[int, float, list] = [],
        labels: Union[str, list] = [],
        nbins: int = 10000,
    ):
        """
	---------------------------------------------------------------------------
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
	tablesample
		An object containing the result. For more information, see
		utilities.tablesample.
		"""
        if isinstance(labels, str):
            labels = [labels]
        if not (labels):
            labels = self.classes_
        return classification_report(
            cutoff=cutoff, estimator=self, labels=labels, nbins=nbins,
        )

    report = classification_report

    # ---#
    @check_dtypes
    def confusion_matrix(
        self, pos_label: Union[int, float, str] = None, cutoff: Union[int, float] = -1
    ):
        """
	---------------------------------------------------------------------------
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
	tablesample
		An object containing the result. For more information, see
		utilities.tablesample.
		"""
        pos_label = (
            self.classes_[1]
            if (pos_label == None and len(self.classes_) == 2)
            else pos_label
        )
        if pos_label:
            return confusion_matrix(
                self.y,
                self.deploySQL(pos_label, cutoff),
                self.test_relation,
                pos_label=pos_label,
            )
        else:
            return multilabel_confusion_matrix(
                self.y, self.deploySQL(), self.test_relation, self.classes_
            )

    # ---#
    def cutoff_curve(
        self,
        pos_label: Union[int, float, str] = None,
        ax=None,
        nbins: int = 30,
        **style_kwds,
    ):
        """
    ---------------------------------------------------------------------------
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
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.
        """
        pos_label = (
            self.classes_[1]
            if (pos_label == None and len(self.classes_) == 2)
            else pos_label
        )
        if pos_label not in self.classes_:
            raise ParameterError(
                "'pos_label' must be one of the response column classes"
            )
        if self.type == "NearestCentroid":
            deploySQL_str = self.deploySQL(allSQL=True)[
                get_match_index(pos_label, self.classes_, False)
            ]
        else:
            deploySQL_str = self.deploySQL(allSQL=True)[0].format(pos_label)
        return roc_curve(
            self.y,
            deploySQL_str,
            self.test_relation,
            pos_label,
            ax=ax,
            cutoff_curve=True,
            nbins=nbins,
            **style_kwds,
        )

    # ---#
    @check_dtypes
    def deploySQL(
        self,
        pos_label: Union[int, float, str] = None,
        cutoff: Union[int, float] = -1,
        allSQL: bool = False,
        X: Union[str, list] = [],
    ):
        """
	---------------------------------------------------------------------------
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
        if isinstance(X, str):
            X = [X]
        X = [quote_ident(elem) for elem in X]
        fun = self.VERTICA_PREDICT_FUNCTION_SQL
        if allSQL:
            if self.type == "NearestCentroid":
                sql = self.to_memmodel().predict_proba_sql(self.X if not (X) else X)
            else:
                sql = (
                    "{0}({1} USING PARAMETERS model_name = '{2}', class = '{3}', "
                    "type = 'probability', match_by_pos = 'true')"
                ).format(fun, ", ".join(self.X if not (X) else X), self.name, "{}")
                sql = [
                    sql,
                    "{0}({1} USING PARAMETERS model_name = '{2}', match_by_pos = 'true')".format(
                        fun, ", ".join(self.X if not (X) else X), self.name
                    ),
                ]
        else:
            if pos_label in self.classes_:
                if self.type == "NearestCentroid":
                    sql = self.to_memmodel().predict_proba_sql(
                        self.X if not (X) else X
                    )[get_match_index(pos_label, self.classes_, False)]
                else:
                    sql = (
                        "{0}({1} USING PARAMETERS model_name = '{2}', class = '{3}', "
                        "type = 'probability', match_by_pos = 'true')"
                    ).format(
                        fun, ", ".join(self.X if not (X) else X), self.name, pos_label
                    )
            if pos_label in self.classes_ and cutoff <= 1 and cutoff >= 0:
                if len(self.classes_) > 2:
                    sql = (
                        "(CASE WHEN {0} >= {1} THEN '{2}' WHEN {0} IS NULL THEN NULL "
                        "ELSE 'Non-{2}' END)"
                    ).format(sql, cutoff, pos_label)
                else:
                    non_pos_label = (
                        self.classes_[0]
                        if (self.classes_[0] != pos_label)
                        else self.classes_[1]
                    )
                    sql = (
                        "(CASE WHEN {0} >= {1} THEN '{2}' WHEN {0} IS NULL THEN NULL "
                        "ELSE '{3}' END)"
                    ).format(sql, cutoff, pos_label, non_pos_label)
            elif pos_label not in self.classes_:
                if self.type == "NearestCentroid":
                    sql = self.to_memmodel().predict_sql(self.X if not (X) else X)
                else:
                    sql = (
                        "{0}({1} USING PARAMETERS model_name = '{2}', "
                        "match_by_pos = 'true')"
                    ).format(fun, ", ".join(self.X if not (X) else X), self.name)
        return sql

    # ---#
    def lift_chart(
        self,
        pos_label: Union[int, float, str] = None,
        ax=None,
        nbins: int = 1000,
        **style_kwds,
    ):
        """
	---------------------------------------------------------------------------
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
	tablesample
		An object containing the result. For more information, see
		utilities.tablesample.
		"""
        pos_label = (
            self.classes_[1]
            if (pos_label == None and len(self.classes_) == 2)
            else pos_label
        )
        if pos_label not in self.classes_:
            raise ParameterError(
                "'pos_label' must be one of the response column classes"
            )
        if self.type == "NearestCentroid":
            deploySQL_str = self.deploySQL(allSQL=True)[
                get_match_index(pos_label, self.classes_, False)
            ]
        else:
            deploySQL_str = self.deploySQL(allSQL=True)[0].format(pos_label)
        return lift_chart(
            self.y,
            deploySQL_str,
            self.test_relation,
            pos_label,
            ax=ax,
            nbins=nbins,
            **style_kwds,
        )

    # ---#
    def prc_curve(
        self,
        pos_label: Union[int, float, str] = None,
        ax=None,
        nbins: int = 30,
        **style_kwds,
    ):
        """
	---------------------------------------------------------------------------
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
	tablesample
		An object containing the result. For more information, see
		utilities.tablesample.
		"""
        pos_label = (
            self.classes_[1]
            if (pos_label == None and len(self.classes_) == 2)
            else pos_label
        )
        if pos_label not in self.classes_:
            raise ParameterError(
                "'pos_label' must be one of the response column classes"
            )
        if self.type == "NearestCentroid":
            deploySQL_str = self.deploySQL(allSQL=True)[
                get_match_index(pos_label, self.classes_, False)
            ]
        else:
            deploySQL_str = self.deploySQL(allSQL=True)[0].format(pos_label)
        return prc_curve(
            self.y,
            deploySQL_str,
            self.test_relation,
            pos_label,
            ax=ax,
            nbins=nbins,
            **style_kwds,
        )

    # ---#
    @check_dtypes
    def predict(
        self,
        vdf: Union[str, vDataFrame],
        X: Union[str, list] = [],
        name: str = "",
        cutoff: Union[int, float] = 0.5,
        inplace: bool = True,
    ):
        """
	---------------------------------------------------------------------------
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
        if isinstance(X, str):
            X = [X]
        assert 0 <= cutoff <= 1, ParameterError(
            "Incorrect parameter 'cutoff'.\nThe cutoff "
            "must be between 0 and 1, inclusive."
        )
        if isinstance(vdf, str):
            vdf = vDataFrameSQL(relation=vdf)
        X = [quote_ident(elem) for elem in X]
        if not (name):
            name = gen_name([self.type, self.name])

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

    # ---#
    @check_dtypes
    def predict_proba(
        self,
        vdf: Union[str, vDataFrame],
        X: Union[str, list] = [],
        name: str = "",
        pos_label: Union[int, str, float] = None,
        inplace: bool = True,
    ):
        """
    ---------------------------------------------------------------------------
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
        Name of the additional prediction vColumn. If unspecified, a name is 
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
        if isinstance(X, str):
            X = [X]
        assert pos_label is None or pos_label in self.classes_, ParameterError(
            (
                "Incorrect parameter 'pos_label'.\nThe class label "
                "must be in [{0}]. Found '{1}'."
            ).format("|".join(["{}".format(c) for c in self.classes_]), pos_label)
        )
        if isinstance(vdf, str):
            vdf = vDataFrameSQL(relation=vdf)
        X = [quote_ident(elem) for elem in X]
        if not (name):
            name = gen_name([self.type, self.name])

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

    # ---#
    def roc_curve(
        self,
        pos_label: Union[int, float, str] = None,
        ax=None,
        nbins: int = 30,
        **style_kwds,
    ):
        """
	---------------------------------------------------------------------------
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
	tablesample
		An object containing the result. For more information, see
		utilities.tablesample.
		"""
        pos_label = (
            self.classes_[1]
            if (pos_label == None and len(self.classes_) == 2)
            else pos_label
        )
        if pos_label not in self.classes_:
            raise ParameterError(
                "'pos_label' must be one of the response column classes"
            )
        if self.type == "NearestCentroid":
            deploySQL_str = self.deploySQL(allSQL=True)[
                get_match_index(pos_label, self.classes_, False)
            ]
        else:
            deploySQL_str = self.deploySQL(allSQL=True)[0].format(pos_label)
        return roc_curve(
            self.y,
            deploySQL_str,
            self.test_relation,
            pos_label,
            ax=ax,
            nbins=nbins,
            **style_kwds,
        )

    # ---#
    @check_dtypes
    def score(
        self,
        method: str = "accuracy",
        pos_label: Union[int, float, str] = None,
        cutoff: Union[int, float] = 0.5,
        nbins: int = 10000,
    ):
        """
	---------------------------------------------------------------------------
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
        pos_label = (
            self.classes_[1]
            if (pos_label == None and len(self.classes_) == 2)
            else pos_label
        )
        if (pos_label not in self.classes_) and (method != "accuracy"):
            raise ParameterError(
                "'pos_label' must be one of the response column classes"
            )
        if self.type == "NearestCentroid":
            deploySQL_str = self.deploySQL(allSQL=True)[
                get_match_index(pos_label, self.classes_, False)
            ]
        else:
            deploySQL_str = self.deploySQL(allSQL=True)[0].format(pos_label)
        if method in ("accuracy", "acc"):
            return accuracy_score(
                self.y, self.deploySQL(pos_label, cutoff), self.test_relation, pos_label
            )
        elif method == "auc":
            return auc(
                "DECODE({}, '{}', 1, 0)".format(self.y, pos_label),
                deploySQL_str,
                self.test_relation,
                nbins=nbins,
            )
        elif method == "aic":
            return aic_bic(
                "DECODE({}, '{}', 1, 0)".format(self.y, pos_label),
                deploySQL_str,
                self.test_relation,
                len(self.X),
            )[0]
        elif method == "bic":
            return aic_bic(
                "DECODE({}, '{}', 1, 0)".format(self.y, pos_label),
                deploySQL_str,
                self.test_relation,
                len(self.X),
            )[1]
        elif method == "prc_auc":
            return prc_auc(
                "DECODE({}, '{}', 1, 0)".format(self.y, pos_label),
                deploySQL_str,
                self.test_relation,
                nbins=nbins,
            )
        elif method in ("best_cutoff", "best_threshold"):
            return roc_curve(
                "DECODE({}, '{}', 1, 0)".format(self.y, pos_label),
                deploySQL_str,
                self.test_relation,
                best_threshold=True,
                nbins=nbins,
            )
        elif method in ("recall", "tpr"):
            return recall_score(
                self.y, self.deploySQL(pos_label, cutoff), self.test_relation
            )
        elif method in ("precision", "ppv"):
            return precision_score(
                self.y, self.deploySQL(pos_label, cutoff), self.test_relation
            )
        elif method in ("specificity", "tnr"):
            return specificity_score(
                self.y, self.deploySQL(pos_label, cutoff), self.test_relation
            )
        elif method in ("negative_predictive_value", "npv"):
            return precision_score(
                self.y, self.deploySQL(pos_label, cutoff), self.test_relation
            )
        elif method in ("log_loss", "logloss"):
            return log_loss(
                "DECODE({}, '{}', 1, 0)".format(self.y, pos_label),
                deploySQL_str,
                self.test_relation,
            )
        elif method == "f1":
            return f1_score(
                self.y, self.deploySQL(pos_label, cutoff), self.test_relation
            )
        elif method == "mcc":
            return matthews_corrcoef(
                self.y, self.deploySQL(pos_label, cutoff), self.test_relation
            )
        elif method in ("bm", "informedness"):
            return informedness(
                self.y, self.deploySQL(pos_label, cutoff), self.test_relation
            )
        elif method in ("mk", "markedness"):
            return markedness(
                self.y, self.deploySQL(pos_label, cutoff), self.test_relation
            )
        elif method in ("csi", "critical_success_index"):
            return critical_success_index(
                self.y, self.deploySQL(pos_label, cutoff), self.test_relation
            )
        else:
            raise ParameterError(
                "The parameter 'method' must be in accuracy|auc|prc_auc"
                "|best_cutoff|recall|precision|log_loss|negative_predictive_value"
                "|specificity|mcc|informedness|markedness|critical_success_index"
                "|aic|bic"
            )


# ---#
class Regressor(Supervised):

    # ---#
    @check_dtypes
    def predict(
        self,
        vdf: Union[str, vDataFrame],
        X: Union[str, list] = [],
        name: str = "",
        inplace: bool = True,
    ):
        """
	---------------------------------------------------------------------------
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
            vdf = vDataFrameSQL(relation=vdf)
        X = [quote_ident(elem) for elem in X]
        name = (
            "{}_".format(self.type) + "".join(ch for ch in self.name if ch.isalnum())
            if not (name)
            else name
        )
        if inplace:
            return vdf.eval(name, self.deploySQL(X=X))
        else:
            return vdf.copy().eval(name, self.deploySQL(X=X))

    # ---#
    @check_dtypes
    def regression_report(self, method: str = "metrics"):
        """
	---------------------------------------------------------------------------
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
	tablesample
		An object containing the result. For more information, see
		utilities.tablesample.
		"""
        raise_error_if_not_in("method", method, ["anova", "metrics", "details"])
        if method in ("anova", "details") and self.type in (
            "SARIMAX",
            "VAR",
            "KernelDensity",
        ):
            raise ModelError(
                "'{}' method is not available for {} models.".format(method, self.type)
            )
        prediction = self.deploySQL()
        if self.type == "SARIMAX":
            test_relation = self.transform_relation
            test_relation = "(SELECT {} AS prediction, {} FROM {}) VERTICAPY_SUBTABLE".format(
                self.deploySQL(), "VerticaPy_y_copy AS {}".format(self.y), test_relation
            )
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
                test_relation = test_relation.replace("[X{}]".format(idx), elem)
            prediction = "prediction"
        elif self.type == "KNeighborsRegressor":
            test_relation = self.deploySQL()
            prediction = "predict_neighbors"
        elif self.type == "VAR":
            relation = self.transform_relation.replace(
                "[VerticaPy_ts]", self.ts
            ).format(self.test_relation)
            for idx, elem in enumerate(self.X):
                relation = relation.replace("[X{}]".format(idx), elem)
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
            result = tablesample(values)
            for idx, y in enumerate(self.X):
                result.values[y] = regression_report(
                    y,
                    self.deploySQL()[idx],
                    relation,
                    len(self.X) * self.parameters["p"],
                ).values["value"]
            return result
        elif self.type == "KernelDensity":
            test_relation = self.map
        else:
            test_relation = self.test_relation
        if method == "metrics":
            return regression_report(self.y, prediction, test_relation, len(self.X))
        elif method == "anova":
            return anova_table(self.y, prediction, test_relation, len(self.X))
        elif method == "details":
            vdf = vDataFrameSQL(
                "(SELECT {} FROM ".format(self.y)
                + self.input_relation
                + ") VERTICAPY_SUBTABLE"
            )
            n = vdf[self.y].count()
            kurt = vdf[self.y].kurt()
            skew = vdf[self.y].skew()
            jb = vdf[self.y].agg(["jb"])[self.y][0]
            R2 = self.score()
            R2_adj = 1 - ((1 - R2) * (n - 1) / (n - len(self.X) - 1))
            anova_T = anova_table(self.y, prediction, test_relation, len(self.X))
            F = anova_T["F"][0]
            p_F = anova_T["p_value"][0]
            return tablesample(
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
                        self.type,
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

    # ---#
    @check_dtypes
    def score(self, method: str = "r2"):
        """
	---------------------------------------------------------------------------
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
        method = method.lower()
        if method in ("r2a", "r2adj", "r2adjusted"):
            method = "r2"
            adj = True
        else:
            adj = False
        if method == "rmse":
            method = "mse"
            root = True
        else:
            root = False
        if self.type == "SARIMAX":
            test_relation = self.transform_relation
            test_relation = "(SELECT {} AS prediction, {} FROM {}) VERTICAPY_SUBTABLE".format(
                self.deploySQL(), "VerticaPy_y_copy AS {}".format(self.y), test_relation
            )
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
        elif self.type == "VAR":
            relation = self.transform_relation.replace(
                "[VerticaPy_ts]", self.ts
            ).format(self.test_relation)
            for idx, elem in enumerate(self.X):
                relation = relation.replace(f"[X{idx}]", elem)
            if method == "mse" and root:
                index = "rmse"
            elif method == "r2" and adj:
                index = "r2a"
            else:
                index = method
            result = tablesample({"index": [index]})
        elif self.type == "KNeighborsRegressor":
            test_relation = self.deploySQL()
            prediction = "predict_neighbors"
        elif self.type == "KernelDensity":
            test_relation = self.map
            prediction = self.deploySQL()
        else:
            test_relation = self.test_relation
            prediction = self.deploySQL()
        if method == "aic":
            if self.type == "VAR":
                for idx, y in enumerate(self.X):
                    result.values[y] = [
                        aic_bic(y, self.deploySQL()[idx], relation, len(self.X),)[0]
                    ]
            else:
                return aic_bic(self.y, prediction, test_relation, len(self.X))[0]
        elif method == "bic":
            if self.type == "VAR":
                for idx, y in enumerate(self.X):
                    result.values[y] = [
                        aic_bic(y, self.deploySQL()[idx], relation, len(self.X))[1]
                    ]
            else:
                return aic_bic(self.y, prediction, test_relation, len(self.X))[1]
        elif method in ("r2", "rsquared"):
            if self.type == "VAR":
                for idx, y in enumerate(self.X):
                    result.values[y] = [
                        r2_score(
                            y,
                            self.deploySQL()[idx],
                            relation,
                            len(self.X) * self.parameters["p"],
                            adj,
                        )
                    ]
            else:
                return r2_score(self.y, prediction, test_relation, len(self.X), adj)
        elif method in ("mae", "mean_absolute_error"):
            if self.type == "VAR":
                for idx, y in enumerate(self.X):
                    result.values[y] = [
                        mean_absolute_error(y, self.deploySQL()[idx], relation,)
                    ]
            else:
                return mean_absolute_error(self.y, prediction, test_relation,)
        elif method in ("mse", "mean_squared_error"):
            if self.type == "VAR":
                for idx, y in enumerate(self.X):
                    result.values[y] = [
                        mean_squared_error(y, self.deploySQL()[idx], relation, root)
                    ]
            else:
                return mean_squared_error(self.y, prediction, test_relation, root)
        elif method in ("msle", "mean_squared_log_error"):
            if self.type == "VAR":
                for idx, y in enumerate(self.X):
                    result.values[y] = [
                        mean_squared_log_error(y, self.deploySQL()[idx], relation)
                    ]
            else:
                return mean_squared_log_error(self.y, prediction, test_relation)
        elif method in ("max", "max_error"):
            if self.type == "VAR":
                for idx, y in enumerate(self.X):
                    result.values[y] = [max_error(y, self.deploySQL()[idx], relation)]
            else:
                return max_error(self.y, prediction, test_relation)
        elif method in ("median", "median_absolute_error"):
            if self.type == "VAR":
                for idx, y in enumerate(self.X):
                    result.values[y] = [
                        median_absolute_error(y, self.deploySQL()[idx], relation)
                    ]
            else:
                return median_absolute_error(self.y, prediction, test_relation)
        elif method in ("var", "explained_variance"):
            if self.type == "VAR":
                for idx, y in enumerate(self.X):
                    result.values[y] = [
                        explained_variance(y, self.deploySQL()[idx], relation)
                    ]
            else:
                return explained_variance(self.y, prediction, test_relation)
        else:
            raise ParameterError(
                "The parameter 'method' must be in r2|mae|mse|msle|max|median|var"
            )
        return result.transpose()


# ---#
class Unsupervised(vModel):

    # ---#
    @check_dtypes
    def fit(self, input_relation: Union[str, vDataFrame], X: Union[str, list] = []):
        """
	---------------------------------------------------------------------------
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
        if verticapy.OPTIONS["overwrite_model"]:
            self.drop()
        else:
            does_model_exist(name=self.name, raise_error=True)
        id_column, id_column_name = "", gen_tmp_name(name="id_column")
        if self.type in ("BisectingKMeans", "IsolationForest") and isinstance(
            verticapy.OPTIONS["random_state"], int
        ):
            id_column = ", ROW_NUMBER() OVER (ORDER BY {0}) AS {1}".format(
                ", ".join([quote_ident(column) for column in X]), id_column_name
            )
        if isinstance(input_relation, str) and self.type == "MCA":
            input_relation = vDataFrameSQL(input_relation)
        tmp_view = False
        if isinstance(input_relation, vDataFrame) or (id_column):
            tmp_view = True
            if isinstance(input_relation, vDataFrame):
                self.input_relation = input_relation.__genSQL__()
            else:
                self.input_relation = input_relation
            if self.type == "MCA":
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
            relation = gen_tmp_name(schema=schema_relation(self.name)[0], name="view")
            drop(relation, method="view")
            executeSQL(
                "CREATE VIEW {0} AS SELECT /*+LABEL('learn.vModel.fit')*/ *{1} FROM {2}".format(
                    relation, id_column, self.input_relation
                ),
                title="Creating a temporary view to fit the model.",
            )
            if not (X) and (self.type == "KPrototypes"):
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
        fun = self.VERTICA_FIT_FUNCTION_SQL if self.type != "MCA" else "PCA"
        query = "SELECT /*+LABEL('learn.vModel.fit')*/ {}('{}', '{}', '{}'".format(
            fun, self.name, relation, ", ".join(self.X)
        )
        if self.type in ("BisectingKMeans", "KMeans", "KPrototypes",):
            query += ", {0}".format(parameters["n_cluster"])
        elif self.type == "Normalizer":
            query += ", {0}".format(parameters["method"])
            del parameters["method"]
        if self.type not in ("Normalizer", "MCA"):
            query += " USING PARAMETERS "
        if (
            "init_method" in parameters
            and not (isinstance(parameters["init_method"], str))
            and self.type in ("KMeans", "BisectingKMeans", "KPrototypes",)
        ):
            name_init = gen_tmp_name(
                schema=schema_relation(self.name)[0],
                name="{0}_init".format(self.type.lower()),
            )
            del parameters["init_method"]
            drop(name_init, method="table")
            if len(self.parameters["init"]) != self.parameters["n_cluster"]:
                raise ParameterError(
                    "'init' must be a list of 'n_cluster' = {0} points".format(
                        self.parameters["n_cluster"]
                    )
                )
            else:
                for item in self.parameters["init"]:
                    if len(X) != len(item):
                        raise ParameterError(
                            "Each points of 'init' must be of size len(X) = {0}".format(
                                len(self.X)
                            )
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
                executeSQL(query0, print_time_sql=False)
                query += f"initial_centers_table = '{name_init}', "
        elif "init_method" in parameters:
            del parameters["init_method"]
            query += "init_method = '{0}', ".format(self.parameters["init"])
        query += ", ".join(
            ["{0} = {1}".format(elem, parameters[elem]) for elem in parameters]
        )
        if self.type == "BisectingKMeans" and isinstance(
            verticapy.OPTIONS["random_state"], int
        ):
            query += ", kmeans_seed={0}, id_column='{1}'".format(
                verticapy.OPTIONS["random_state"], id_column_name
            )
        elif self.type == "IsolationForest" and isinstance(
            verticapy.OPTIONS["random_state"], int
        ):
            query += ", seed={0}, id_column='{1}'".format(
                verticapy.OPTIONS["random_state"], id_column_name
            )
        query += ")"
        try:
            executeSQL(query, "Fitting the model.")
            if tmp_view:
                drop(relation, method="view")
        except:
            if tmp_view:
                drop(relation, method="view")
            if (
                "init_method" in parameters
                and not (isinstance(parameters["init_method"], str))
                and self.type in ("KMeans", "BisectingKMeans", "KPrototypes",)
            ):
                drop(name_init, method="table")
            raise
        if self.type in ("KMeans", "BisectingKMeans", "KPrototypes",):
            if "init_method" in parameters and not (
                isinstance(parameters["init_method"], str)
            ):
                drop(name_init, method="table")
            if self.type in ("KMeans", "KPrototypes",):
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
                self.metrics_ = tablesample(values)
            elif self.type == "BisectingKMeans":
                self.metrics_ = self.get_attr("Metrics")
                self.cluster_centers_ = self.get_attr("BKTree")
        elif self.type in ("PCA", "MCA"):
            self.components_ = self.get_attr("principal_components")
            if self.type == "MCA":
                self.cos2_ = self.components_.to_list()
                for i in range(len(self.cos2_)):
                    self.cos2_[i] = [elem ** 2 for elem in self.cos2_[i]]
                    total = sum(self.cos2_[i])
                    self.cos2_[i] = [elem / total for elem in self.cos2_[i]]
                values = {"index": self.X}
                for idx, elem in enumerate(self.components_.values):
                    if elem != "index":
                        values[elem] = [item[idx - 1] for item in self.cos2_]
                self.cos2_ = tablesample(values)
            self.explained_variance_ = self.get_attr("singular_values")
            self.mean_ = self.get_attr("columns")
        elif self.type == "SVD":
            self.singular_values_ = self.get_attr("right_singular_vectors")
            self.explained_variance_ = self.get_attr("singular_values")
        elif self.type == "Normalizer":
            self.param_ = self.get_attr("details")
        elif self.type == "OneHotEncoder":
            try:
                self.param_ = to_tablesample(
                    query="""SELECT 
                                category_name, 
                                category_level::varchar, 
                                category_level_index 
                             FROM (SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS 
                                                model_name = '{0}', 
                                                attr_name = 'integer_categories')) 
                                                VERTICAPY_SUBTABLE 
                             UNION ALL 
                             SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS 
                                        model_name = '{0}', 
                                        attr_name = 'varchar_categories')""".format(
                        self.name,
                    ),
                    title="Getting Model Attributes.",
                )
            except:
                try:
                    self.param_ = to_tablesample(
                        query="""SELECT 
                                    category_name, 
                                    category_level::varchar, 
                                    category_level_index 
                                 FROM (SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS 
                                                model_name = '{0}', 
                                                attr_name = 'integer_categories')) 
                                                VERTICAPY_SUBTABLE""".format(
                            self.name
                        ),
                        title="Getting Model Attributes.",
                    )
                except:
                    self.param_ = self.get_attr("varchar_categories")
        return self


# ---#
class Preprocessing(Unsupervised):

    # ---#
    @check_dtypes
    def deploySQL(
        self,
        key_columns: Union[str, list] = [],
        exclude_columns: Union[str, list] = [],
        X: Union[str, list] = [],
    ):
        """
    ---------------------------------------------------------------------------
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
        X = [quote_ident(elem) for elem in X]
        fun = self.VERTICA_TRANSFORM_FUNCTION_SQL
        sql = "{}({} USING PARAMETERS model_name = '{}', match_by_pos = 'true'"
        if key_columns:
            sql += ", key_columns = '{}'".format(
                ", ".join([quote_ident(item) for item in key_columns])
            )
        if exclude_columns:
            sql += ", exclude_columns = '{}'".format(
                ", ".join([quote_ident(item) for item in exclude_columns])
            )
        if self.type == "OneHotEncoder":
            separator = (
                "NULL"
                if self.parameters["separator"] == None
                else "'{}'".format(self.parameters["separator"])
            )
            null_column_name = (
                "NULL"
                if self.parameters["null_column_name"] == None
                else "'{}'".format(self.parameters["null_column_name"])
            )
            sql += (
                ", drop_first = {0}, ignore_null = {1}, separator = {2}, "
                "column_naming = '{3}'"
            ).format(
                self.parameters["drop_first"],
                self.parameters["ignore_null"],
                separator,
                self.parameters["column_naming"],
            )
            if self.parameters["column_naming"].lower() in ("values", "values_relaxed"):
                sql += ", null_column_name = {}".format(null_column_name)
        sql += ")"
        return sql.format(fun, ", ".join(self.X if not (X) else X), self.name)

    # ---#
    @check_dtypes
    def deployInverseSQL(
        self,
        key_columns: Union[str, list] = [],
        exclude_columns: Union[str, list] = [],
        X: Union[str, list] = [],
    ) -> str:
        """
    ---------------------------------------------------------------------------
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
        if isinstance(X, str):
            X = [X]
        if self.type == "OneHotEncoder":
            raise ModelError(
                "method 'inverse_transform' is not supported for OneHotEncoder models."
            )
        X = [quote_ident(elem) for elem in X]
        fun = self.VERTICA_INVERSE_TRANSFORM_FUNCTION_SQL
        sql = "{}({} USING PARAMETERS model_name = '{}', match_by_pos = 'true'"
        if key_columns:
            sql += ", key_columns = '{}'".format(
                ", ".join([quote_ident(item) for item in key_columns])
            )
        if exclude_columns:
            sql += ", exclude_columns = '{}'".format(
                ", ".join([quote_ident(item) for item in exclude_columns])
            )
        sql += ")"
        return sql.format(fun, ", ".join(self.X if not (X) else X), self.name)

    # ---#
    def get_names(self, inverse: bool = False, X: list = []):
        """
    ---------------------------------------------------------------------------
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
        if self.type in ("PCA", "SVD", "MCA") and not (inverse):
            if self.type in ("PCA", "SVD"):
                n = self.parameters["n_components"]
                if not (n):
                    n = len(self.X)
            else:
                n = len(self.X)
            return [f"col{i}" for i in range(1, n + 1)]
        elif self.type == "OneHotEncoder" and not (inverse):
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
                                names += [
                                    '"'
                                    + quote_ident(column)[1:-1]
                                    + "{}{}".format(
                                        self.parameters["separator"],
                                        self.param_["category_level_index"][i],
                                    )
                                    + '"'
                                ]
                            else:
                                names += [
                                    '"'
                                    + quote_ident(column)[1:-1]
                                    + "{}{}".format(
                                        self.parameters["separator"],
                                        self.param_["category_level"][i].lower()
                                        if self.param_["category_level"][i] != None
                                        else self.parameters["null_column_name"],
                                    )
                                    + '"'
                                ]
                        k += 1
            return names
        else:
            return X

    # ---#
    @check_dtypes
    def inverse_transform(self, vdf: Union[str, vDataFrame], X: Union[str, list] = []):
        """
    ---------------------------------------------------------------------------
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
        if self.type == "OneHotEncoder":
            raise ModelError(
                "method 'inverse_transform' is not supported for OneHotEncoder models."
            )
        if not (vdf):
            vdf = self.input_relation
        if not (X):
            X = self.get_names()
        if isinstance(vdf, str):
            vdf = vDataFrameSQL(relation=vdf)
        X = vdf.format_colnames(X)
        relation = vdf.__genSQL__()
        exclude_columns = vdf.get_columns(exclude_columns=X)
        all_columns = vdf.get_columns()
        main_relation = "(SELECT {} FROM {}) VERTICAPY_SUBTABLE".format(
            self.deployInverseSQL(exclude_columns, exclude_columns, all_columns),
            relation,
        )
        return vDataFrameSQL(main_relation, "Inverse Transformation")

    # ---#
    @check_dtypes
    def transform(self, vdf: Union[str, vDataFrame] = None, X: Union[str, list] = []):
        """
    ---------------------------------------------------------------------------
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
            vdf = vDataFrameSQL(relation=vdf)
        vdf.are_namecols_in(X)
        X = vdf.format_colnames(X)
        relation = vdf.__genSQL__()
        exclude_columns = vdf.get_columns(exclude_columns=X)
        all_columns = vdf.get_columns()
        main_relation = "(SELECT {} FROM {}) VERTICAPY_SUBTABLE".format(
            self.deploySQL(exclude_columns, exclude_columns, all_columns), relation
        )
        return vDataFrameSQL(main_relation, "Inverse Transformation")


# ---#
class Decomposition(Preprocessing):

    # ---#
    @check_dtypes
    def deploySQL(
        self,
        n_components: int = 0,
        cutoff: Union[int, float] = 1,
        key_columns: Union[str, list] = [],
        exclude_columns: Union[str, list] = [],
        X: Union[str, list] = [],
    ):
        """
    ---------------------------------------------------------------------------
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
        X = [quote_ident(elem) for elem in X]
        fun = self.VERTICA_TRANSFORM_FUNCTION_SQL
        sql = "{}({} USING PARAMETERS model_name = '{}', match_by_pos = 'true'"
        if key_columns:
            sql += ", key_columns = '{}'".format(
                ", ".join([quote_ident(item) for item in key_columns])
            )
        if exclude_columns:
            sql += ", exclude_columns = '{}'".format(
                ", ".join([quote_ident(item) for item in exclude_columns])
            )
        if n_components:
            sql += ", num_components = {}".format(n_components)
        else:
            sql += ", cutoff = {}".format(cutoff)
        sql += ")"
        return sql.format(fun, ", ".join(self.X if not (X) else X), self.name)

    # ---#
    @check_dtypes
    def plot(self, dimensions: tuple = (1, 2), ax=None, **style_kwds):
        """
    ---------------------------------------------------------------------------
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
        vdf = vDataFrameSQL(self.input_relation)
        ax = self.transform(vdf).scatter(
            columns=["col{}".format(dimensions[0]), "col{}".format(dimensions[1])],
            max_nb_points=100000,
            ax=ax,
            **style_kwds,
        )
        explained_variance = self.explained_variance_["explained_variance"]
        ax.set_xlabel(
            "Dim{} {}".format(
                dimensions[0],
                ""
                if not (explained_variance[dimensions[0] - 1])
                else "({}%)".format(
                    round(explained_variance[dimensions[0] - 1] * 100, 1)
                ),
            )
        )
        ax.set_ylabel(
            "Dim{} {}".format(
                dimensions[1],
                ""
                if not (explained_variance[dimensions[1] - 1])
                else "({}%)".format(
                    round(explained_variance[dimensions[1] - 1] * 100, 1)
                ),
            )
        )
        return ax

    # ---#
    @check_dtypes
    def plot_circle(self, dimensions: tuple = (1, 2), ax=None, **style_kwds):
        """
    ---------------------------------------------------------------------------
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
        if self.type == "SVD":
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

    # ---#
    def plot_scree(self, ax=None, **style_kwds):
        """
    ---------------------------------------------------------------------------
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
        information = tablesample(
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
            ax.text(
                i + 1.5,
                explained_variance[i] + 1,
                "{}%".format(round(explained_variance[i], 1)),
            )
        return ax

    # ---#
    @check_dtypes
    def score(
        self,
        X: Union[str, list] = [],
        input_relation: str = "",
        method: str = "avg",
        p: int = 2,
    ):
        """
    ---------------------------------------------------------------------------
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
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.
        """
        raise_error_if_not_in("method", str(method).lower(), ["avg", "median"])
        if isinstance(X, str):
            X = [X]
        if not (X):
            X = self.X
        if not (input_relation):
            input_relation = self.input_relation
        method = str(method).upper()
        if method == "MEDIAN":
            method = "APPROXIMATE_MEDIAN"
        if self.type in ("PCA", "SVD"):
            n_components = self.parameters["n_components"]
            if not (n_components):
                n_components = len(X)
        else:
            n_components = len(X)
        col_init_1 = [f"{X[idx]} AS col_init{idx}" for idx in range(len(X))]
        col_init_2 = [f"col_init{idx}" for idx in range(len(X))]
        cols = ["col{}".format(idx + 1) for idx in range(n_components)]
        query = """SELECT 
                        {0}({1} USING PARAMETERS 
                            model_name = '{2}', 
                            key_columns = '{1}', 
                            num_components = {3}) OVER () 
                    FROM {4}""".format(
            self.VERTICA_TRANSFORM_FUNCTION_SQL,
            ", ".join(self.X),
            self.name,
            n_components,
            input_relation,
        )
        query = (
            f"SELECT {', '.join(col_init_1 + cols)} FROM ({query}) VERTICAPY_SUBTABLE"
        )
        query = """SELECT 
                        {0}({1} USING PARAMETERS 
                            model_name = '{2}', 
                            key_columns = '{3}', 
                            exclude_columns = '{3}', 
                            num_components = {4}) OVER () 
                   FROM ({5}) y""".format(
            self.VERTICA_INVERSE_TRANSFORM_FUNCTION_SQL,
            ", ".join(col_init_2 + cols),
            self.name,
            ", ".join(col_init_2),
            n_components,
            query,
        )
        p_distances = [
            f"{method}(POWER(ABS(POWER({X[idx]}, {p}) - POWER(col_init{idx}, {p})), {1 / p})) AS {X[idx]}"
            for idx in range(len(X))
        ]
        query = f"SELECT 'Score' AS 'index', {', '.join(p_distances)} FROM ({query}) z"
        return to_tablesample(query, title="Getting Model Score.").transpose()

    # ---#
    @check_dtypes
    def transform(
        self,
        vdf: Union[str, vDataFrame] = None,
        X: Union[str, list] = [],
        n_components: int = 0,
        cutoff: Union[int, float] = 1,
    ):
        """
    ---------------------------------------------------------------------------
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
            vdf = vDataFrameSQL(relation=vdf)
        vdf.are_namecols_in(X)
        X = vdf.format_colnames(X)
        relation = vdf.__genSQL__()
        exclude_columns = vdf.get_columns(exclude_columns=X)
        all_columns = vdf.get_columns()
        main_relation = "(SELECT {} FROM {}) VERTICAPY_SUBTABLE".format(
            self.deploySQL(
                n_components, cutoff, exclude_columns, exclude_columns, all_columns
            ),
            relation,
        )
        return vDataFrameSQL(main_relation, "Inverse Transformation")


# ---#
class Clustering(Unsupervised):

    # ---#
    @check_dtypes
    def predict(
        self,
        vdf: Union[str, vDataFrame],
        X: Union[str, list] = [],
        name: str = "",
        inplace: bool = True,
    ):
        """
	---------------------------------------------------------------------------
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
            vdf = vDataFrameSQL(relation=vdf)
        X = [quote_ident(elem) for elem in X]
        name = (
            "{}_".format(self.type) + "".join(ch for ch in self.name if ch.isalnum())
            if not (name)
            else name
        )
        if inplace:
            return vdf.eval(name, self.deploySQL(X=X))
        else:
            return vdf.copy().eval(name, self.deploySQL(X=X))
