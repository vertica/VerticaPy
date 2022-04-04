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
import os, warnings
import numpy as np
from typing import Union

# VerticaPy Modules
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
                name = self.tree_name if self.type in ("KernelDensity") else self.name
                try:
                    version(cursor=self.cursor, condition=[9, 0, 0])
                    executeSQL(
                        self.cursor,
                        "SELECT GET_MODEL_SUMMARY(USING PARAMETERS model_name = '{}')".format(
                            name
                        ),
                        "Summarizing the model.",
                    )
                except:
                    executeSQL(
                        self.cursor,
                        "SELECT SUMMARIZE_MODEL('{}')".format(name),
                        "Summarizing the model.",
                    )
                return self.cursor.fetchone()[0]
            elif self.type == "AutoML":
                rep = self.best_model_.__repr__()
            elif self.type == "AutoDataPrep":
                rep = self.final_relation_.__repr__()
            elif self.type == "DBSCAN":
                rep = "=======\ndetails\n=======\nNumber of Clusters: {}\nNumber of Outliers: {}".format(
                    self.n_cluster_, self.n_noise_
                )
            elif self.type == "LocalOutlierFactor":
                rep = "=======\ndetails\n=======\nNumber of Errors: {}".format(
                    self.n_errors_
                )
            elif self.type == "NearestCentroid":
                rep = "=======\ndetails\n=======\n" + self.centroids_.__repr__()
            elif self.type == "VAR":
                rep = "=======\ndetails\n======="
                for idx, elem in enumerate(self.X):
                    rep += "\n\n # " + str(elem) + "\n\n" + self.coef_[idx].__repr__()
                rep += "\n\n===============\nAdditional Info\n==============="
                rep += "\nInput Relation : {}".format(self.input_relation)
                rep += "\nX : {}".format(", ".join(self.X))
                rep += "\nts : {}".format(self.ts)
            elif self.type == "SARIMAX":
                rep = "=======\ndetails\n======="
                rep += "\n\n# Coefficients\n\n" + self.coef_.__repr__()
                if self.ma_piq_:
                    rep += "\n\n# MA PIQ\n\n" + self.ma_piq_.__repr__()
                rep += "\n\n===============\nAdditional Info\n==============="
                rep += "\nInput Relation : {}".format(self.input_relation)
                rep += "\ny : {}".format(self.y)
                rep += "\nts : {}".format(self.ts)
                if self.exogenous:
                    rep += "\nExogenous Variables : {}".format(
                        ", ".join(self.exogenous)
                    )
                if self.ma_avg_:
                    rep += "\nMA AVG : {}".format(self.ma_avg_)
            elif self.type == "CountVectorizer":
                rep = "=======\ndetails\n======="
                if self.vocabulary_:
                    voc = [str(elem) for elem in self.vocabulary_]
                    if len(voc) > 100:
                        voc = voc[0:100] + [
                            "... ({} more)".format(len(self.vocabulary_) - 100)
                        ]
                    rep += "\n\n# Vocabulary\n\n" + ", ".join(voc)
                if self.stop_words_:
                    rep += "\n\n# Stop Words\n\n" + ", ".join(
                        [str(elem) for elem in self.stop_words_]
                    )
                rep += "\n\n===============\nAdditional Info\n==============="
                rep += "\nInput Relation : {}".format(self.input_relation)
                rep += "\nX : {}".format(", ".join(self.X))
            if self.type in (
                "DBSCAN",
                "NearestCentroid",
                "LocalOutlierFactor",
                "KNeighborsRegressor",
                "KNeighborsClassifier",
            ):
                rep += "\n\n===============\nAdditional Info\n==============="
                rep += "\nInput Relation : {}".format(self.input_relation)
                rep += "\nX : {}".format(", ".join(self.X))
            if self.type in (
                "NearestCentroid",
                "KNeighborsRegressor",
                "KNeighborsClassifier",
            ):
                rep += "\ny : {}".format(self.y)
            return rep
        except:
            return "<{}>".format(self.type)

    # ---#
    def contour(
        self, nbins: int = 100, pos_label: Union[int, float, str] = None, ax=None, **style_kwds,
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
        if self.type in ("RandomForestClassifier", "XGBoostClassifier", "NaiveBayes", "NearestCentroid", "KNeighborsClassifier",):
            if not(pos_label):
                pos_label = sorted(self.classes_)[-1]
            if self.type in ("RandomForestClassifier", "XGBoostClassifier", "NaiveBayes", "NearestCentroid",):
                return vdf_from_relation(self.input_relation, cursor=self.cursor,).contour(self.X, self.deploySQL(X = self.X, pos_label=pos_label), cbar_title=self.y, nbins=nbins, ax=ax, **style_kwds,)
            else:
                return vdf_from_relation(self.input_relation, cursor=self.cursor,).contour(self.X, self, pos_label=pos_label, cbar_title=self.y, nbins=nbins, ax=ax, **style_kwds,)
        elif self.type in ("KNeighborsRegressor",):
            return vdf_from_relation(self.input_relation, cursor=self.cursor,).contour(self.X, self, cbar_title=self.y, nbins=nbins, ax=ax, **style_kwds,)
        elif self.type in ("KMeans", "BisectingKMeans",):
            return vdf_from_relation(self.input_relation, cursor=self.cursor,).contour(self.X, self, cbar_title="cluster", nbins=nbins, ax=ax, **style_kwds,)
        else:
            return vdf_from_relation(self.input_relation, cursor=self.cursor,).contour(self.X, self.deploySQL(X = self.X), cbar_title=self.y, nbins=nbins, ax=ax, **style_kwds,)

    # ---#
    def deploySQL(self, X: list = []):
        """
	---------------------------------------------------------------------------
	Returns the SQL code needed to deploy the model. 

	Parameters
	----------
	X: list, optional
		List of the columns used to deploy the model. If empty, the model
		predictors will be used.

	Returns
	-------
	str
		the SQL code needed to deploy the model.
		"""
        if self.type == "AutoML":
            return self.best_model_.deploySQL(X)
        if isinstance(X, str):
            X = [X]
        if self.type not in ("DBSCAN", "LocalOutlierFactor"):
            name = self.tree_name if self.type in ("KernelDensity") else self.name
            check_types([("X", X, [list],)])
            X = [str_column(elem) for elem in X]
            fun = self.get_model_fun()[1]
            sql = "{}({} USING PARAMETERS model_name = '{}', match_by_pos = 'true')"
            return sql.format(fun, ", ".join(self.X if not (X) else X), name)
        else:
            raise FunctionError(
                "Method 'deploySQL' for '{}' doesn't exist.".format(self.type)
            )

    # ---#
    def drop(self):
        """
	---------------------------------------------------------------------------
	Drops the model from the Vertica database.
		"""
        if self.type == "AutoDataPrep":
            with warnings.catch_warnings(record=True) as w:
                drop(
                    self.name, self.cursor, method="table",
                )
        else:
            with warnings.catch_warnings(record=True) as w:
                drop(
                    self.name, self.cursor, method="model",
                )

    # ---#
    def features_importance(
        self, ax=None, tree_id: int = None, show: bool = True, **style_kwds,
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
                        coeff_importances[self.stepwise_["variable"][idx]] = self.stepwise_["importance"][idx]
                return plot_importance(coeff_importances, print_legend=False, ax=ax, **style_kwds,)
            return self.best_model_.features_importance(ax, tree_id, show, **style_kwds,)
        if self.type in (
            "RandomForestClassifier",
            "RandomForestRegressor",
            "KernelDensity",
        ):
            check_types([("tree_id", tree_id, [int])])
            name = self.tree_name if self.type in ("KernelDensity") else self.name
            version(cursor=self.cursor, condition=[9, 1, 1])
            tree_id = "" if not (tree_id) else ", tree_id={}".format(tree_id)
            query = "SELECT predictor_name AS predictor, ROUND(100 * importance_value / SUM(importance_value) OVER (), 2)::float AS importance, SIGN(importance_value)::int AS sign FROM (SELECT RF_PREDICTOR_IMPORTANCE ( USING PARAMETERS model_name = '{}'{})) VERTICAPY_SUBTABLE ORDER BY 2 DESC;".format(
                name, tree_id,
            )
            print_legend = False
        elif self.type in (
            "LinearRegression",
            "LogisticRegression",
            "LinearSVC",
            "LinearSVR",
        ):
            relation = self.input_relation
            version(cursor=self.cursor, condition=[8, 1, 1])
            query = "SELECT predictor, ROUND(100 * importance / SUM(importance) OVER(), 2) AS importance, sign FROM "
            query += "(SELECT stat.predictor AS predictor, ABS(coefficient * (max - min))::float AS importance, SIGN(coefficient)::int AS sign FROM "
            query += '(SELECT LOWER("column") AS predictor, min, max FROM (SELECT SUMMARIZE_NUMCOL({}) OVER() '.format(
                ", ".join(self.X)
            )
            query += " FROM {}) VERTICAPY_SUBTABLE) stat NATURAL JOIN ({})".format(
                relation, self.coef_.to_sql()
            )
            query += " coeff) importance_t ORDER BY 2 DESC;"
            print_legend = True
        else:
            raise FunctionError(
                "Method 'features_importance' for '{}' doesn't exist.".format(self.type)
            )
        executeSQL(self.cursor, query, "Computing Features Importance.")
        result = self.cursor.fetchall()
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
    def get_attr(self, attr_name: str = "",):
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
        if self.type not in ("DBSCAN", "LocalOutlierFactor", "VAR", "SARIMAX", "KNeighborsClassifier", "KNeighborsRegressor", "NearestCentroid", "CountVectorizer",):
            name = self.tree_name if self.type in ("KernelDensity") else self.name
            version(cursor=self.cursor, condition=[8, 1, 1])
            result = to_tablesample(
                query="SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}'{})".format(
                    name, ", attr_name = '{}'".format(attr_name) if attr_name else "",
                ),
                cursor=self.cursor,
                title="Getting Model Attributes.",
            )
            return result
        elif self.type in ("DBSCAN",):
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
                raise ParameterError("Attribute '{}' doesn't exist.".format(attr_name))
        elif self.type in ("CountVectorizer",):
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
                        "attr_name": ["lowercase", "max_df", "min_df", "max_features", "ignore_special", "max_text_size", "vocabulary", "stop_words",],
                    },
                )
                return result
            else:
                raise ParameterError("Attribute '{}' doesn't exist.".format(attr_name))
        elif self.type in ("NearestCentroid",):
            if attr_name == "p":
                return self.parameters["p"]
            elif attr_name == "centroids":
                return self.centroids_
            elif attr_name == "classes":
                return self.classes_
            elif not (attr_name):
                result = tablesample(
                    values={
                        "attr_name": ["centroids", "classes", "p",],
                    },
                )
                return result
            else:
                raise ParameterError("Attribute '{}' doesn't exist.".format(attr_name))
        elif self.type in ("KNeighborsClassifier",):
            if attr_name == "p":
                return self.parameters["p"]
            elif attr_name == "n_neighbors":
                return self.parameters["n_neighbors"]
            elif attr_name == "classes":
                return self.classes_
            elif not (attr_name):
                result = tablesample(
                    values={
                        "attr_name": ["n_neighbors", "p", "classes",],
                    },
                )
                return result
            else:
                raise ParameterError("Attribute '{}' doesn't exist.".format(attr_name))
        elif self.type in ("KNeighborsRegressor",):
            if attr_name == "p":
                return self.parameters["p"]
            elif attr_name == "n_neighbors":
                return self.parameters["n_neighbors"]
            elif not (attr_name):
                result = tablesample(
                    values={
                        "attr_name": ["n_neighbors", "p",],
                    },
                )
                return result
            else:
                raise ParameterError("Attribute '{}' doesn't exist.".format(attr_name))
        elif self.type in ("LocalOutlierFactor",):
            if attr_name == "n_errors":
                return self.n_errors_
            elif not (attr_name):
                result = tablesample(
                    values={"attr_name": ["n_errors"], "value": [self.n_errors_]},
                )
                return result
            else:
                raise ParameterError("Attribute '{}' doesn't exist.".format(attr_name))
        elif self.type in ("SARIMAX",):
            if attr_name == "coefficients":
                return self.coef_
            elif attr_name == "ma_avg":
                return self.ma_avg_
            elif attr_name == "ma_piq":
                return self.ma_piq_
            elif not (attr_name):
                result = tablesample(
                    values={"attr_name": ["coefficients", "ma_avg", "ma_piq",]},
                )
                return result
            else:
                raise ParameterError("Attribute '{}' doesn't exist.".format(attr_name))
        elif self.type in ("VAR",):
            if attr_name == "coefficients":
                return self.coef_
            elif not (attr_name):
                result = tablesample(values={"attr_name": ["coefficients",]},)
                return result
            else:
                raise ParameterError("Attribute '{}' doesn't exist.".format(attr_name))
        elif self.type in ("KernelDensity",):
            if attr_name == "map":
                return self.map_
            elif not (attr_name):
                result = tablesample(values={"attr_name": ["map"]},)
                return result
            else:
                raise ParameterError("Attribute '{}' doesn't exist.".format(attr_name))
        else:
            raise FunctionError(
                "Method 'get_attr' for '{}' doesn't exist.".format(self.type)
            )

    # ---#
    def get_model_fun(self):
        """
	---------------------------------------------------------------------------
	Returns the Vertica functions associated with the model.

	Returns
	-------
	tuple
		(FIT, PREDICT, INVERSE)
		"""
        if self.type == "AutoML":
            return self.best_model_.get_model_fun()
        if self.type in ("LinearRegression", "SARIMAX"):
            return ("LINEAR_REG", "PREDICT_LINEAR_REG", "")
        elif self.type == "LogisticRegression":
            return ("LOGISTIC_REG", "PREDICT_LOGISTIC_REG", "")
        elif self.type == "LinearSVC":
            return ("SVM_CLASSIFIER", "PREDICT_SVM_CLASSIFIER", "")
        elif self.type == "LinearSVR":
            return ("SVM_REGRESSOR", "PREDICT_SVM_REGRESSOR", "")
        elif self.type in ("RandomForestRegressor", "KernelDensity"):
            return ("RF_REGRESSOR", "PREDICT_RF_REGRESSOR", "")
        elif self.type == "RandomForestClassifier":
            return ("RF_CLASSIFIER", "PREDICT_RF_CLASSIFIER", "")
        elif self.type in ("XGBoostRegressor",):
            return ("XGB_REGRESSOR", "PREDICT_XGB_REGRESSOR", "")
        elif self.type == "XGBoostClassifier":
            return ("XGB_CLASSIFIER", "PREDICT_XGB_CLASSIFIER", "")
        elif self.type == "NaiveBayes":
            return ("NAIVE_BAYES", "PREDICT_NAIVE_BAYES", "")
        elif self.type == "KMeans":
            return ("KMEANS", "APPLY_KMEANS", "")
        elif self.type == "BisectingKMeans":
            return ("BISECTING_KMEANS", "APPLY_BISECTING_KMEANS", "")
        elif self.type in ("PCA", "MCA",):
            return ("PCA", "APPLY_PCA", "APPLY_INVERSE_PCA")
        elif self.type == "SVD":
            return ("SVD", "APPLY_SVD", "APPLY_INVERSE_SVD")
        elif self.type == "Normalizer":
            return ("NORMALIZE_FIT", "APPLY_NORMALIZE", "REVERSE_NORMALIZE")
        elif self.type == "OneHotEncoder":
            return ("ONE_HOT_ENCODER_FIT", "APPLY_ONE_HOT_ENCODER", "")
        else:
            return ("", "", "")

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
        return self.parameters

    # ---#
    def plot(
        self, max_nb_points: int = 100, ax=None, **style_kwds,
    ):
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
        check_types([("max_nb_points", max_nb_points, [int, float],)])
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
                    self.cursor,
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
                    self.cursor,
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
                    self.cursor,
                    max_nb_points,
                    ax=ax,
                    **style_kwds,
                )
        elif self.type in ("KMeans", "BisectingKMeans", "DBSCAN"):
            if self.type != "DBSCAN":
                vdf = vdf_from_relation(self.input_relation, cursor=self.cursor)
                self.predict(vdf, name="kmeans_cluster")
                catcol = "kmeans_cluster"
            else:
                vdf = vdf_from_relation(self.name, cursor=self.cursor)
                catcol = "dbscan_cluster"
            return vdf.scatter(
                columns=self.X,
                catcol=catcol,
                max_cardinality=100,
                max_nb_points=max_nb_points,
                ax=ax,
                **style_kwds,
            )
        elif self.type in ("LocalOutlierFactor"):
            query = "SELECT COUNT(*) FROM {}".format(self.name)
            tablesample = 100 * min(
                float(max_nb_points / self.cursor.execute(query).fetchone()[0]), 1
            )
            return lof_plot(
                self.name, self.X, "lof_score", self.cursor, 100, ax=ax, **style_kwds,
            )
        elif self.type in ("RandomForestRegressor", "XGBoostRegressor",):
            return regression_tree_plot(
                self.X + [self.deploySQL()],
                self.y,
                self.input_relation,
                self.cursor,
                max_nb_points,
                ax=ax,
                **style_kwds,
            )
        else:
            raise FunctionError(
                "Method 'plot' for '{}' doesn't exist.".format(self.type)
            )

    # ---#
    def set_cursor(self, cursor):
        """
	---------------------------------------------------------------------------
	Sets a new database cursor.

	Parameters
	----------
	cursor: DBcursor
		New cursor.

	Returns
	-------
	model
		self
		"""
        check_cursor(cursor)
        cursor.execute("SELECT 1;")
        self.cursor = cursor
        return self

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
        try:
            self.parameters
        except:
            self.parameters = {}
        model_parameters = {}
        default_parameters = default_model_parameters(self.type)
        if self.type in ("LinearRegression", "LogisticRegression", "SARIMAX", "VAR"):
            if "solver" in parameters:
                check_types([("solver", parameters["solver"], [str],)])
                assert str(parameters["solver"]).lower() in [
                    "newton",
                    "bfgs",
                    "cgd",
                ], ParameterError(
                    "Incorrect parameter 'solver'.\nThe optimizer must be in (Newton | BFGS | CGD), found '{}'.".format(
                        parameters["solver"]
                    )
                )
                model_parameters["solver"] = parameters["solver"]
            elif "solver" not in self.parameters:
                model_parameters["solver"] = default_parameters["solver"]
            else:
                model_parameters["solver"] = self.parameters["solver"]
            if "penalty" in parameters and self.type in (
                "LinearRegression",
                "LogisticRegression",
            ):
                check_types([("penalty", parameters["penalty"], [str],)])
                assert str(parameters["penalty"]).lower() in [
                    "none",
                    "l1",
                    "l2",
                    "enet",
                ], ParameterError(
                    "Incorrect parameter 'penalty'.\nThe regularization must be in (None | L1 | L2 | ENet), found '{}'.".format(
                        parameters["penalty"]
                    )
                )
                model_parameters["penalty"] = parameters["penalty"]
            elif (
                self.type in ("LinearRegression", "LogisticRegression")
                and "penalty" not in self.parameters
            ):
                model_parameters["penalty"] = default_parameters["penalty"]
            elif self.type in ("LinearRegression", "LogisticRegression"):
                model_parameters["penalty"] = self.parameters["penalty"]
            if "max_iter" in parameters:
                check_types([("max_iter", parameters["max_iter"], [int, float],)])
                assert 0 <= parameters["max_iter"], ParameterError(
                    "Incorrect parameter 'max_iter'.\nThe maximum number of iterations must be positive."
                )
                model_parameters["max_iter"] = parameters["max_iter"]
            elif "max_iter" not in self.parameters:
                model_parameters["max_iter"] = default_parameters["max_iter"]
            else:
                model_parameters["max_iter"] = self.parameters["max_iter"]
            if "l1_ratio" in parameters and self.type in (
                "LinearRegression",
                "LogisticRegression",
            ):
                check_types([("l1_ratio", parameters["l1_ratio"], [int, float],)])
                assert 0 <= parameters["l1_ratio"] <= 1, ParameterError(
                    "Incorrect parameter 'l1_ratio'.\nThe ENet Mixture must be between 0 and 1."
                )
                model_parameters["l1_ratio"] = parameters["l1_ratio"]
            elif (
                self.type in ("LinearRegression", "LogisticRegression")
                and "l1_ratio" not in self.parameters
            ):
                model_parameters["l1_ratio"] = default_parameters["l1_ratio"]
            elif self.type in ("LinearRegression", "LogisticRegression"):
                model_parameters["l1_ratio"] = self.parameters["l1_ratio"]
            if "C" in parameters and self.type in (
                "LinearRegression",
                "LogisticRegression",
            ):
                check_types([("C", parameters["C"], [int, float],)])
                assert 0 <= parameters["C"], ParameterError(
                    "Incorrect parameter 'C'.\nThe regularization parameter value must be positive."
                )
                model_parameters["C"] = parameters["C"]
            elif (
                self.type in ("LinearRegression", "LogisticRegression")
                and "C" not in self.parameters
            ):
                model_parameters["C"] = default_parameters["C"]
            elif self.type in ("LinearRegression", "LogisticRegression"):
                model_parameters["C"] = self.parameters["C"]
            if "tol" in parameters:
                check_types([("tol", parameters["tol"], [int, float],)])
                assert 0 <= parameters["tol"], ParameterError(
                    "Incorrect parameter 'tol'.\nThe tolerance parameter value must be positive."
                )
                model_parameters["tol"] = parameters["tol"]
            elif "tol" not in self.parameters:
                model_parameters["tol"] = default_parameters["tol"]
            else:
                model_parameters["tol"] = self.parameters["tol"]
            if "p" in parameters and self.type in ("SARIMAX", "VAR"):
                check_types([("p", parameters["p"], [int, float],)])
                assert 0 <= parameters["p"], ParameterError(
                    "Incorrect parameter 'p'.\nThe order of the AR part must be positive."
                )
                model_parameters["p"] = parameters["p"]
            elif self.type in ("SARIMAX", "VAR") and "p" not in self.parameters:
                model_parameters["p"] = default_parameters["p"]
            elif self.type in ("SARIMAX", "VAR"):
                model_parameters["p"] = self.parameters["p"]
            if "q" in parameters and self.type == "SARIMAX":
                check_types([("q", parameters["q"], [int, float],)])
                assert 0 <= parameters["q"], ParameterError(
                    "Incorrect parameter 'q'.\nThe order of the MA part must be positive."
                )
                model_parameters["q"] = parameters["q"]
            elif self.type == "SARIMAX" and "q" not in self.parameters:
                model_parameters["q"] = default_parameters["q"]
            elif self.type == "SARIMAX":
                model_parameters["q"] = self.parameters["q"]
            if "d" in parameters and self.type == "SARIMAX":
                check_types([("d", parameters["d"], [int, float],)])
                assert 0 <= parameters["d"], ParameterError(
                    "Incorrect parameter 'd'.\nThe order of the I part must be positive."
                )
                model_parameters["d"] = parameters["d"]
            elif self.type == "SARIMAX" and "d" not in self.parameters:
                model_parameters["d"] = default_parameters["d"]
            elif self.type == "SARIMAX":
                model_parameters["d"] = self.parameters["d"]
            if "P" in parameters and self.type == "SARIMAX":
                check_types([("P", parameters["P"], [int, float],)])
                assert 0 <= parameters["P"], ParameterError(
                    "Incorrect parameter 'P'.\nThe seasonal order of the AR part must be positive."
                )
                model_parameters["P"] = parameters["P"]
            elif self.type == "SARIMAX" and "P" not in self.parameters:
                model_parameters["P"] = default_parameters["P"]
            elif self.type == "SARIMAX":
                model_parameters["P"] = self.parameters["P"]
            if "Q" in parameters and self.type == "SARIMAX":
                check_types([("Q", parameters["Q"], [int, float],)])
                assert 0 <= parameters["Q"], ParameterError(
                    "Incorrect parameter 'Q'.\nThe seasonal order of the MA part must be positive."
                )
                model_parameters["Q"] = parameters["Q"]
            elif self.type == "SARIMAX" and "Q" not in self.parameters:
                model_parameters["Q"] = default_parameters["Q"]
            elif self.type == "SARIMAX":
                model_parameters["Q"] = self.parameters["Q"]
            if "D" in parameters and self.type == "SARIMAX":
                check_types([("D", parameters["D"], [int, float],)])
                assert 0 <= parameters["D"], ParameterError(
                    "Incorrect parameter 'D'.\nThe seasonal order of the I part must be positive."
                )
                model_parameters["D"] = parameters["D"]
            elif self.type == "SARIMAX" and "D" not in self.parameters:
                model_parameters["D"] = default_parameters["D"]
            elif self.type == "SARIMAX":
                model_parameters["D"] = self.parameters["D"]
            if "s" in parameters and self.type == "SARIMAX":
                check_types([("s", parameters["s"], [int, float],)])
                assert 0 <= parameters["s"], ParameterError(
                    "Incorrect parameter 's'.\nThe Span of the seasonality must be positive."
                )
                model_parameters["s"] = parameters["s"]
            elif self.type == "SARIMAX" and "s" not in self.parameters:
                model_parameters["s"] = default_parameters["s"]
            elif self.type == "SARIMAX":
                model_parameters["s"] = self.parameters["s"]
            if "max_pik" in parameters and self.type == "SARIMAX":
                check_types([("max_pik", parameters["max_pik"], [int, float],)])
                assert 0 <= parameters["max_pik"], ParameterError(
                    "Incorrect parameter 'max_pik'.\nThe Maximum number of inverse MA coefficients took during the computation must be positive."
                )
                model_parameters["max_pik"] = parameters["max_pik"]
            elif self.type == "SARIMAX" and "max_pik" not in self.parameters:
                model_parameters["max_pik"] = default_parameters["max_pik"]
            elif self.type == "SARIMAX":
                model_parameters["max_pik"] = self.parameters["max_pik"]
            if "papprox_ma" in parameters and self.type == "SARIMAX":
                check_types([("papprox_ma", parameters["papprox_ma"], [int, float],)])
                assert 0 <= parameters["papprox_ma"], ParameterError(
                    "Incorrect parameter 'papprox_ma'.\nThe Maximum number of AR(P) used to approximate the MA during the computation must be positive."
                )
                model_parameters["papprox_ma"] = parameters["papprox_ma"]
            elif self.type == "SARIMAX" and "papprox_ma" not in self.parameters:
                model_parameters["papprox_ma"] = default_parameters["papprox_ma"]
            elif self.type == "SARIMAX":
                model_parameters["papprox_ma"] = self.parameters["papprox_ma"]
        elif self.type in ("KernelDensity"):
            if "bandwidth" in parameters:
                check_types([("bandwidth", parameters["bandwidth"], [int, float],)])
                assert 0 <= parameters["bandwidth"], ParameterError(
                    "Incorrect parameter 'bandwidth'.\nThe bandwidth must be positive."
                )
                model_parameters["bandwidth"] = parameters["bandwidth"]
            elif "bandwidth" not in self.parameters:
                model_parameters["bandwidth"] = default_parameters["bandwidth"]
            else:
                model_parameters["bandwidth"] = self.parameters["bandwidth"]
            if "kernel" in parameters:
                check_types(
                    [
                        (
                            "kernel",
                            parameters["kernel"],
                            ["gaussian", "logistic", "sigmoid", "silverman"],
                        )
                    ]
                )
                assert parameters["kernel"] in [
                    "gaussian",
                    "logistic",
                    "sigmoid",
                    "silverman",
                ], ParameterError(
                    "Incorrect parameter 'kernel'.\nThe parameter 'kernel' must be in [gaussian|logistic|sigmoid|silverman], found '{}'.".format(
                        kernel
                    )
                )
                model_parameters["kernel"] = parameters["kernel"]
            elif "kernel" not in self.parameters:
                model_parameters["kernel"] = default_parameters["kernel"]
            else:
                model_parameters["kernel"] = self.parameters["kernel"]
            if "max_leaf_nodes" in parameters:
                check_types(
                    [
                        (
                            "max_leaf_nodes",
                            parameters["max_leaf_nodes"],
                            [int, float],
                            False,
                        )
                    ]
                )
                assert 1 <= parameters["max_leaf_nodes"] <= 1e9, ParameterError(
                    "Incorrect parameter 'max_leaf_nodes'.\nThe maximum number of leaf nodes must be between 1 and 1e9, inclusive."
                )
                model_parameters["max_leaf_nodes"] = parameters["max_leaf_nodes"]
            elif "max_leaf_nodes" not in self.parameters:
                model_parameters["max_leaf_nodes"] = default_parameters[
                    "max_leaf_nodes"
                ]
            else:
                model_parameters["max_leaf_nodes"] = self.parameters["max_leaf_nodes"]
            if "max_depth" in parameters:
                check_types([("max_depth", parameters["max_depth"], [int],)])
                assert 1 <= parameters["max_depth"] <= 100, ParameterError(
                    "Incorrect parameter 'max_depth'.\nThe maximum depth for growing each tree must be between 1 and 100, inclusive."
                )
                model_parameters["max_depth"] = parameters["max_depth"]
            elif "max_depth" not in self.parameters:
                model_parameters["max_depth"] = default_parameters["max_depth"]
            else:
                model_parameters["max_depth"] = self.parameters["max_depth"]
            if "min_samples_leaf" in parameters:
                check_types(
                    [
                        (
                            "min_samples_leaf",
                            parameters["min_samples_leaf"],
                            [int, float],
                            False,
                        )
                    ]
                )
                assert 1 <= parameters["min_samples_leaf"] <= 1e6, ParameterError(
                    "Incorrect parameter 'min_samples_leaf'.\nThe minimum number of samples each branch must have after splitting a node must be between 1 and 1e6, inclusive."
                )
                model_parameters["min_samples_leaf"] = parameters["min_samples_leaf"]
            elif "min_samples_leaf" not in self.parameters:
                model_parameters["min_samples_leaf"] = default_parameters[
                    "min_samples_leaf"
                ]
            else:
                model_parameters["min_samples_leaf"] = self.parameters[
                    "min_samples_leaf"
                ]
            if "nbins" in parameters:
                check_types([("nbins", parameters["nbins"], [int, float],)])
                assert 2 <= parameters["nbins"], ParameterError(
                    "Incorrect parameter 'nbins'.\nThe number of bins to use for continuous features must be greater than 2."
                )
                model_parameters["nbins"] = parameters["nbins"]
            elif "nbins" not in self.parameters:
                model_parameters["nbins"] = default_parameters["nbins"]
            else:
                model_parameters["nbins"] = self.parameters["nbins"]
            if "p" in parameters:
                check_types([("p", parameters["p"], [int, float],)])
                assert 0 < parameters["p"], ParameterError(
                    "Incorrect parameter 'p'.\nThe p of the p-distance must be strictly positive."
                )
                model_parameters["p"] = parameters["p"]
            elif "p" not in self.parameters:
                model_parameters["p"] = default_parameters["p"]
            else:
                model_parameters["p"] = self.parameters["p"]
            if "xlim" in parameters:
                check_types([("xlim", parameters["xlim"], [list],)])
                model_parameters["xlim"] = parameters["xlim"]
            elif "xlim" not in self.parameters:
                model_parameters["xlim"] = default_parameters["xlim"]
            else:
                model_parameters["xlim"] = self.parameters["xlim"]
        elif self.type in ("RandomForestClassifier", "RandomForestRegressor"):
            if "n_estimators" in parameters:
                check_types([("n_estimators", parameters["n_estimators"], [int],)])
                assert 0 <= parameters["n_estimators"] <= 1000, ParameterError(
                    "Incorrect parameter 'n_estimators'.\nThe number of trees must be lesser than 1000."
                )
                model_parameters["n_estimators"] = parameters["n_estimators"]
            elif "n_estimators" not in self.parameters:
                model_parameters["n_estimators"] = default_parameters["n_estimators"]
            else:
                model_parameters["n_estimators"] = self.parameters["n_estimators"]
            if "max_features" in parameters:
                check_types(
                    [
                        (
                            "max_features",
                            parameters["max_features"],
                            [int, float, str],
                            False,
                        )
                    ]
                )
                if isinstance(parameters["max_features"], str):
                    assert str(parameters["max_features"]).lower() in [
                        "max",
                        "auto",
                    ], ParameterError(
                        "Incorrect parameter 'init'.\nThe maximum number of features to test must be in (max | auto) or an integer, found '{}'.".format(
                            parameters["max_features"]
                        )
                    )
                model_parameters["max_features"] = parameters["max_features"]
            elif "max_features" not in self.parameters:
                model_parameters["max_features"] = default_parameters["max_features"]
            else:
                model_parameters["max_features"] = self.parameters["max_features"]
            if "max_leaf_nodes" in parameters:
                check_types(
                    [
                        (
                            "max_leaf_nodes",
                            parameters["max_leaf_nodes"],
                            [int, float],
                            False,
                        )
                    ]
                )
                assert 1 <= parameters["max_leaf_nodes"] <= 1e9, ParameterError(
                    "Incorrect parameter 'max_leaf_nodes'.\nThe maximum number of leaf nodes must be between 1 and 1e9, inclusive."
                )
                model_parameters["max_leaf_nodes"] = parameters["max_leaf_nodes"]
            elif "max_leaf_nodes" not in self.parameters:
                model_parameters["max_leaf_nodes"] = default_parameters[
                    "max_leaf_nodes"
                ]
            else:
                model_parameters["max_leaf_nodes"] = self.parameters["max_leaf_nodes"]
            if "sample" in parameters:
                check_types([("sample", parameters["sample"], [int, float],)])
                assert 0 <= parameters["sample"] <= 1, ParameterError(
                    "Incorrect parameter 'sample'.\nThe portion of the input data set that is randomly picked for training each tree must be between 0.0 and 1.0, inclusive."
                )
                model_parameters["sample"] = parameters["sample"]
            elif "sample" not in self.parameters:
                model_parameters["sample"] = default_parameters["sample"]
            else:
                model_parameters["sample"] = self.parameters["sample"]
            if "max_depth" in parameters:
                check_types([("max_depth", parameters["max_depth"], [int],)])
                assert 1 <= parameters["max_depth"] <= 100, ParameterError(
                    "Incorrect parameter 'max_depth'.\nThe maximum depth for growing each tree must be between 1 and 100, inclusive."
                )
                model_parameters["max_depth"] = parameters["max_depth"]
            elif "max_depth" not in self.parameters:
                model_parameters["max_depth"] = default_parameters["max_depth"]
            else:
                model_parameters["max_depth"] = self.parameters["max_depth"]
            if "min_samples_leaf" in parameters:
                check_types(
                    [
                        (
                            "min_samples_leaf",
                            parameters["min_samples_leaf"],
                            [int, float],
                            False,
                        )
                    ]
                )
                assert 1 <= parameters["min_samples_leaf"] <= 1e6, ParameterError(
                    "Incorrect parameter 'min_samples_leaf'.\nThe minimum number of samples each branch must have after splitting a node must be between 1 and 1e6, inclusive."
                )
                model_parameters["min_samples_leaf"] = parameters["min_samples_leaf"]
            elif "min_samples_leaf" not in self.parameters:
                model_parameters["min_samples_leaf"] = default_parameters[
                    "min_samples_leaf"
                ]
            else:
                model_parameters["min_samples_leaf"] = self.parameters[
                    "min_samples_leaf"
                ]
            if "min_info_gain" in parameters:
                check_types(
                    [
                        (
                            "min_info_gain",
                            parameters["min_info_gain"],
                            [int, float],
                            False,
                        )
                    ]
                )
                assert 0 <= parameters["min_info_gain"] <= 1, ParameterError(
                    "Incorrect parameter 'min_info_gain'.\nThe minimum threshold for including a split must be between 0.0 and 1.0, inclusive."
                )
                model_parameters["min_info_gain"] = parameters["min_info_gain"]
            elif "min_info_gain" not in self.parameters:
                model_parameters["min_info_gain"] = default_parameters["min_info_gain"]
            else:
                model_parameters["min_info_gain"] = self.parameters["min_info_gain"]
            if "nbins" in parameters:
                check_types([("nbins", parameters["nbins"], [int, float],)])
                assert 2 <= parameters["nbins"] <= 1000, ParameterError(
                    "Incorrect parameter 'nbins'.\nThe number of bins to use for continuous features must be between 2 and 1000, inclusive."
                )
                model_parameters["nbins"] = parameters["nbins"]
            elif "nbins" not in self.parameters:
                model_parameters["nbins"] = default_parameters["nbins"]
            else:
                model_parameters["nbins"] = self.parameters["nbins"]
        elif self.type in ("XGBoostClassifier", "XGBoostRegressor"):
            if "max_ntree" in parameters:
                check_types([("max_ntree", parameters["max_ntree"], [int],)])
                assert 0 <= parameters["max_ntree"] <= 1000, ParameterError(
                    "Incorrect parameter 'max_ntree'.\nThe maximum number of trees must be lesser than 1000."
                )
                model_parameters["max_ntree"] = parameters["max_ntree"]
            elif "max_ntree" not in self.parameters:
                model_parameters["max_ntree"] = default_parameters["max_ntree"]
            else:
                model_parameters["max_ntree"] = self.parameters["max_ntree"]
            if "split_proposal_method" in parameters:
                assert str(parameters["split_proposal_method"]).lower() in [
                    "global",
                ], ParameterError(
                    "Incorrect parameter 'split_proposal_method'.\nThe Split Proposal Method must be in (global,), found '{}'.".format(
                        parameters["split_proposal_method"]
                    )
                )
                model_parameters["split_proposal_method"] = parameters[
                    "split_proposal_method"
                ]
            elif "split_proposal_method" not in self.parameters:
                model_parameters["split_proposal_method"] = default_parameters[
                    "split_proposal_method"
                ]
            else:
                model_parameters["split_proposal_method"] = self.parameters[
                    "split_proposal_method"
                ]
            if "tol" in parameters:
                check_types([("tol", parameters["tol"], [int, float],)])
                assert 0 < parameters["tol"] <= 1, ParameterError(
                    "Incorrect parameter 'tol'.\nThe tolerance must be between 0 and 1."
                )
                model_parameters["tol"] = parameters["tol"]
            elif "tol" not in self.parameters:
                model_parameters["tol"] = default_parameters["tol"]
            else:
                model_parameters["tol"] = self.parameters["tol"]
            if "learning_rate" in parameters:
                check_types(
                    [("learning_rate", parameters["learning_rate"], [int, float],)]
                )
                assert 0 < parameters["learning_rate"] <= 1, ParameterError(
                    "Incorrect parameter 'learning_rate'.\nThe Learning Rate must be between 0 and 1."
                )
                model_parameters["learning_rate"] = parameters["learning_rate"]
            elif "learning_rate" not in self.parameters:
                model_parameters["learning_rate"] = default_parameters["learning_rate"]
            else:
                model_parameters["learning_rate"] = self.parameters["learning_rate"]
            if "min_split_loss" in parameters:
                check_types(
                    [("min_split_loss", parameters["min_split_loss"], [int, float],)]
                )
                assert 0 <= parameters["min_split_loss"] <= 1000, ParameterError(
                    "Incorrect parameter 'min_split_loss'.\nThe Minimum Split Loss must be must be lesser than 1000."
                )
                model_parameters["min_split_loss"] = parameters["min_split_loss"]
            elif "min_split_loss" not in self.parameters:
                model_parameters["min_split_loss"] = default_parameters[
                    "min_split_loss"
                ]
            else:
                model_parameters["min_split_loss"] = self.parameters["min_split_loss"]
            if "weight_reg" in parameters:
                check_types([("weight_reg", parameters["weight_reg"], [int, float],)])
                assert 0 <= parameters["weight_reg"] <= 1000, ParameterError(
                    "Incorrect parameter 'weight_reg'.\nThe Weight must be lesser than 1000."
                )
                model_parameters["weight_reg"] = parameters["weight_reg"]
            elif "weight_reg" not in self.parameters:
                model_parameters["weight_reg"] = default_parameters["weight_reg"]
            else:
                model_parameters["weight_reg"] = self.parameters["weight_reg"]
            if "sample" in parameters:
                check_types([("sample", parameters["sample"], [int, float],)])
                assert 0 < parameters["sample"] <= 1, ParameterError(
                    "Incorrect parameter 'sample'.\nThe portion of the input data set that is randomly picked for training each tree must be between 0.0 and 1.0."
                )
                model_parameters["sample"] = parameters["sample"]
            elif "sample" not in self.parameters:
                model_parameters["sample"] = default_parameters["sample"]
            else:
                model_parameters["sample"] = self.parameters["sample"]
            v = version(cursor = self.cursor)
            v = (v[0] > 11 or (v[0] == 11 and (v[1] >= 1 or v[2] >= 1)))
            if v:
                if "col_sample_by_tree" in parameters:
                    check_types([("col_sample_by_tree", parameters["col_sample_by_tree"], [int, float],)])
                    assert 0 < parameters["col_sample_by_tree"] <= 1, ParameterError(
                        "Incorrect parameter 'col_sample_by_tree'.\nThe parameter 'col_sample_by_tree' must be between 0.0 and 1.0."
                    )
                    model_parameters["col_sample_by_tree"] = parameters["col_sample_by_tree"]
                elif "col_sample_by_tree" not in self.parameters:
                    model_parameters["col_sample_by_tree"] = default_parameters["col_sample_by_tree"]
                else:
                    model_parameters["col_sample_by_tree"] = self.parameters["col_sample_by_tree"]
                if "col_sample_by_node" in parameters:
                    check_types([("col_sample_by_node", parameters["col_sample_by_node"], [int, float],)])
                    assert 0 < parameters["col_sample_by_node"] <= 1, ParameterError(
                        "Incorrect parameter 'col_sample_by_node'.\nThe parameter 'col_sample_by_node' must be between 0.0 and 1.0."
                    )
                    model_parameters["col_sample_by_node"] = parameters["col_sample_by_node"]
                elif "col_sample_by_tree" not in self.parameters:
                    model_parameters["col_sample_by_node"] = default_parameters["col_sample_by_node"]
                else:
                    model_parameters["col_sample_by_node"] = self.parameters["col_sample_by_node"]
            if "max_depth" in parameters:
                check_types([("max_depth", parameters["max_depth"], [int],)])
                assert 1 <= parameters["max_depth"] <= 20, ParameterError(
                    "Incorrect parameter 'max_depth'.\nThe maximum depth for growing each tree must be between 1 and 20, inclusive."
                )
                model_parameters["max_depth"] = parameters["max_depth"]
            elif "max_depth" not in self.parameters:
                model_parameters["max_depth"] = default_parameters["max_depth"]
            else:
                model_parameters["max_depth"] = self.parameters["max_depth"]
            if "nbins" in parameters:
                check_types([("nbins", parameters["nbins"], [int, float],)])
                assert 2 <= parameters["nbins"] <= 1000, ParameterError(
                    "Incorrect parameter 'nbins'.\nThe number of bins to use for continuous features must be between 2 and 1000, inclusive."
                )
                model_parameters["nbins"] = parameters["nbins"]
            elif "nbins" not in self.parameters:
                model_parameters["nbins"] = default_parameters["nbins"]
            else:
                model_parameters["nbins"] = self.parameters["nbins"]
        elif self.type in ("NaiveBayes",):
            if "alpha" in parameters:
                check_types([("alpha", parameters["alpha"], [int, float],)])
                assert 0 <= parameters["alpha"], ParameterError(
                    "Incorrect parameter 'alpha'.\nThe smoothing factor must be positive."
                )
                model_parameters["alpha"] = parameters["alpha"]
            elif "alpha" not in self.parameters:
                model_parameters["alpha"] = default_parameters["alpha"]
            else:
                model_parameters["alpha"] = self.parameters["alpha"]
            if "nbtype" in parameters:
                check_types([("nbtype", parameters["nbtype"], [str],)])
                if isinstance(parameters["nbtype"], str):
                    assert str(parameters["nbtype"]).lower() in [
                        "bernoulli",
                        "categorical",
                        "multinomial",
                        "gaussian",
                        "auto",
                    ], ParameterError(
                        "Incorrect parameter 'nbtype'.\nThe Naive Bayes type must be in (bernoulli | categorical | multinomial | gaussian | auto), found '{}'.".format(
                            parameters["init"]
                        )
                    )
                model_parameters["nbtype"] = parameters["nbtype"]
            elif "nbtype" not in self.parameters:
                model_parameters["nbtype"] = default_parameters["nbtype"]
            else:
                model_parameters["nbtype"] = self.parameters["nbtype"]
        elif self.type in ("KMeans", "BisectingKMeans"):
            if "max_iter" in parameters:
                check_types([("max_iter", parameters["max_iter"], [int, float],)])
                assert 0 <= parameters["max_iter"], ParameterError(
                    "Incorrect parameter 'max_iter'.\nThe maximum number of iterations must be positive."
                )
                model_parameters["max_iter"] = parameters["max_iter"]
            elif "max_iter" not in self.parameters:
                model_parameters["max_iter"] = default_parameters["max_iter"]
            else:
                model_parameters["max_iter"] = self.parameters["max_iter"]
            if "tol" in parameters:
                check_types([("tol", parameters["tol"], [int, float],)])
                assert 0 <= parameters["tol"], ParameterError(
                    "Incorrect parameter 'tol'.\nThe tolerance parameter value must be positive."
                )
                model_parameters["tol"] = parameters["tol"]
            elif "tol" not in self.parameters:
                model_parameters["tol"] = default_parameters["tol"]
            else:
                model_parameters["tol"] = self.parameters["tol"]
            if "n_cluster" in parameters:
                check_types([("n_cluster", parameters["n_cluster"], [int, float],)])
                assert 1 <= parameters["n_cluster"] <= 10000, ParameterError(
                    "Incorrect parameter 'n_cluster'.\nThe number of clusters must be between 1 and 10000, inclusive."
                )
                model_parameters["n_cluster"] = parameters["n_cluster"]
            elif "n_cluster" not in self.parameters:
                model_parameters["n_cluster"] = default_parameters["n_cluster"]
            else:
                model_parameters["n_cluster"] = self.parameters["n_cluster"]
            if "init" in parameters:
                check_types([("init", parameters["init"], [str, list],)])
                if isinstance(parameters["init"], str):
                    if self.type in ("BisectingKMeans",):
                        assert str(parameters["init"]).lower() in [
                            "random",
                            "kmeanspp",
                            "pseudo",
                        ], ParameterError(
                            "Incorrect parameter 'init'.\nThe initialization method of the clusters must be in (random | kmeanspp | pseudo) or a list of the initial clusters position, found '{}'.".format(
                                parameters["init"]
                            )
                        )
                    else:
                        assert str(parameters["init"]).lower() in [
                            "random",
                            "kmeanspp",
                        ], ParameterError(
                            "Incorrect parameter 'init'.\nThe initialization method of the clusters must be in (random | kmeanspp) or a list of the initial clusters position, found '{}'.".format(
                                parameters["init"]
                            )
                        )
                model_parameters["init"] = parameters["init"]
            elif "init" not in self.parameters:
                model_parameters["init"] = default_parameters["init"]
            else:
                model_parameters["init"] = self.parameters["init"]
            if "bisection_iterations" in parameters:
                check_types(
                    [
                        (
                            "bisection_iterations",
                            parameters["bisection_iterations"],
                            [int, float],
                            False,
                        )
                    ]
                )
                assert (
                    1 <= parameters["bisection_iterations"] <= 1000000
                ), ParameterError(
                    "Incorrect parameter 'bisection_iterations'.\nThe number of iterations the bisecting k-means algorithm performs for each bisection step must be between 1 and 1e6, inclusive."
                )
                model_parameters["bisection_iterations"] = parameters[
                    "bisection_iterations"
                ]
            elif (
                self.type == "BisectingKMeans"
                and "bisection_iterations" not in self.parameters
            ):
                model_parameters["bisection_iterations"] = default_parameters[
                    "bisection_iterations"
                ]
            elif self.type == "BisectingKMeans":
                model_parameters["bisection_iterationss"] = self.parameters[
                    "bisection_iterations"
                ]
            if "split_method" in parameters:
                check_types([("split_method", parameters["split_method"], [str],)])
                assert str(parameters["split_method"]).lower() in [
                    "size",
                    "sum_squares",
                ], ParameterError(
                    "Incorrect parameter 'split_method'.\nThe split method must be in (size | sum_squares), found '{}'.".format(
                        parameters["split_method"]
                    )
                )
                model_parameters["split_method"] = parameters["split_method"]
            elif (
                self.type == "BisectingKMeans" and "split_method" not in self.parameters
            ):
                model_parameters["split_method"] = default_parameters["split_method"]
            elif self.type == "BisectingKMeans":
                model_parameters["split_method"] = self.parameters["split_method"]
            if "min_divisible_cluster_size" in parameters:
                check_types(
                    [
                        (
                            "min_divisible_cluster_size",
                            parameters["min_divisible_cluster_size"],
                            [int, float],
                            False,
                        )
                    ]
                )
                assert 2 <= parameters["min_divisible_cluster_size"], ParameterError(
                    "Incorrect parameter 'min_divisible_cluster_size'.\nThe minimum number of points of a divisible cluster must be greater than or equal to 2."
                )
                model_parameters["min_divisible_cluster_size"] = parameters[
                    "min_divisible_cluster_size"
                ]
            elif (
                self.type == "BisectingKMeans"
                and "min_divisible_cluster_size" not in self.parameters
            ):
                model_parameters["min_divisible_cluster_size"] = default_parameters[
                    "min_divisible_cluster_size"
                ]
            elif self.type == "BisectingKMeans":
                model_parameters["min_divisible_cluster_size"] = self.parameters[
                    "min_divisible_cluster_size"
                ]
            if "distance_method" in parameters:
                check_types(
                    [("distance_method", parameters["distance_method"], [str],)]
                )
                assert str(parameters["distance_method"]).lower() in [
                    "euclidean"
                ], ParameterError(
                    "Incorrect parameter 'distance_method'.\nThe distance method must be in (euclidean), found '{}'.".format(
                        parameters["distance_method"]
                    )
                )
                model_parameters["distance_method"] = parameters["distance_method"]
            elif (
                self.type == "BisectingKMeans"
                and "distance_method" not in self.parameters
            ):
                model_parameters["distance_method"] = default_parameters[
                    "distance_method"
                ]
            elif self.type == "BisectingKMeans":
                model_parameters["distance_method"] = self.parameters["distance_method"]
        elif self.type in ("LinearSVC", "LinearSVR"):
            if "tol" in parameters:
                check_types([("tol", parameters["tol"], [int, float],)])
                assert 0 <= parameters["tol"], ParameterError(
                    "Incorrect parameter 'tol'.\nThe tolerance parameter value must be positive."
                )
                model_parameters["tol"] = parameters["tol"]
            elif "tol" not in self.parameters:
                model_parameters["tol"] = default_parameters["tol"]
            else:
                model_parameters["tol"] = self.parameters["tol"]
            if "C" in parameters:
                check_types([("C", parameters["C"], [int, float],)])
                assert 0 <= parameters["C"], ParameterError(
                    "Incorrect parameter 'C'.\nThe weight for misclassification cost must be positive."
                )
                model_parameters["C"] = parameters["C"]
            elif "C" not in self.parameters:
                model_parameters["C"] = default_parameters["C"]
            else:
                model_parameters["C"] = self.parameters["C"]
            if "max_iter" in parameters:
                check_types([("max_iter", parameters["max_iter"], [int, float],)])
                assert 0 <= parameters["max_iter"], ParameterError(
                    "Incorrect parameter 'max_iter'.\nThe maximum number of iterations must be positive."
                )
                model_parameters["max_iter"] = parameters["max_iter"]
            elif "max_iter" not in self.parameters:
                model_parameters["max_iter"] = default_parameters["max_iter"]
            else:
                model_parameters["max_iter"] = self.parameters["max_iter"]
            if "fit_intercept" in parameters:
                check_types([("fit_intercept", parameters["fit_intercept"], [bool],)])
                model_parameters["fit_intercept"] = parameters["fit_intercept"]
            elif "fit_intercept" not in self.parameters:
                model_parameters["fit_intercept"] = default_parameters["fit_intercept"]
            else:
                model_parameters["fit_intercept"] = self.parameters["fit_intercept"]
            if "intercept_scaling" in parameters:
                check_types(
                    [
                        (
                            "intercept_scaling",
                            parameters["intercept_scaling"],
                            [float],
                            False,
                        )
                    ]
                )
                assert 0 <= parameters["intercept_scaling"], ParameterError(
                    "Incorrect parameter 'intercept_scaling'.\nThe Intercept Scaling parameter value must be positive."
                )
                model_parameters["intercept_scaling"] = parameters["intercept_scaling"]
            elif "intercept_scaling" not in self.parameters:
                model_parameters["intercept_scaling"] = default_parameters[
                    "intercept_scaling"
                ]
            else:
                model_parameters["intercept_scaling"] = self.parameters[
                    "intercept_scaling"
                ]
            if "intercept_mode" in parameters:
                check_types([("intercept_mode", parameters["intercept_mode"], [str],)])
                assert str(parameters["intercept_mode"]).lower() in [
                    "regularized",
                    "unregularized",
                ], ParameterError(
                    "Incorrect parameter 'intercept_mode'.\nThe Intercept Mode must be in (size | sum_squares), found '{}'.".format(
                        parameters["intercept_mode"]
                    )
                )
                model_parameters["intercept_mode"] = parameters["intercept_mode"]
            elif "intercept_mode" not in self.parameters:
                model_parameters["intercept_mode"] = default_parameters[
                    "intercept_mode"
                ]
            else:
                model_parameters["intercept_mode"] = self.parameters["intercept_mode"]
            if ("class_weight" in parameters) and self.type in ("LinearSVC"):
                check_types(
                    [("class_weight", parameters["class_weight"], [list, tuple],)]
                )
                model_parameters["class_weight"] = parameters["class_weight"]
            elif self.type in ("LinearSVC",) and "class_weight" not in self.parameters:
                model_parameters["class_weight"] = default_parameters["class_weight"]
            elif self.type in ("LinearSVC",):
                model_parameters["class_weight"] = self.parameters["class_weight"]
            if ("acceptable_error_margin" in parameters) and self.type in ("LinearSVR"):
                check_types(
                    [
                        (
                            "acceptable_error_margin",
                            parameters["acceptable_error_margin"],
                            [int, float],
                            False,
                        )
                    ]
                )
                assert 0 <= parameters["acceptable_error_margin"], ParameterError(
                    "Incorrect parameter 'acceptable_error_margin'.\nThe Acceptable Error Margin parameter value must be positive."
                )
                model_parameters["acceptable_error_margin"] = parameters[
                    "acceptable_error_margin"
                ]
            elif (
                self.type in ("LinearSVR",)
                and "acceptable_error_margin" not in self.parameters
            ):
                model_parameters["acceptable_error_margin"] = default_parameters[
                    "acceptable_error_margin"
                ]
            elif self.type in ("LinearSVR",):
                model_parameters["acceptable_error_margin"] = self.parameters[
                    "acceptable_error_margin"
                ]
        elif self.type in ("PCA", "SVD", "MCA",):
            if ("scale" in parameters) and self.type in ("PCA"):
                check_types([("scale", parameters["scale"], [bool],)])
                model_parameters["scale"] = parameters["scale"]
            elif self.type in ("PCA",) and "scale" not in self.parameters:
                model_parameters["scale"] = default_parameters["scale"]
            elif self.type in ("PCA",):
                model_parameters["scale"] = self.parameters["scale"]
            if self.type in ("PCA", "SVD",):
                if "method" in parameters:
                    check_types([("method", parameters["method"], [str],)])
                    assert str(parameters["method"]).lower() in ["lapack"], ParameterError(
                        "Incorrect parameter 'method'.\nThe decomposition method must be in (lapack), found '{}'.".format(
                            parameters["method"]
                        )
                    )
                    model_parameters["method"] = parameters["method"]
                elif "method" not in self.parameters:
                    model_parameters["method"] = default_parameters["method"]
                else:
                    model_parameters["method"] = self.parameters["method"]
                if "n_components" in parameters:
                    check_types(
                        [("n_components", parameters["n_components"], [int, float],)]
                    )
                    assert 0 <= parameters["n_components"], ParameterError(
                        "Incorrect parameter 'n_components'.\nThe number of components must be positive. If it is equal to 0, all the components will be considered."
                    )
                    model_parameters["n_components"] = parameters["n_components"]
                elif "n_components" not in self.parameters:
                    model_parameters["n_components"] = default_parameters["n_components"]
                else:
                    model_parameters["n_components"] = self.parameters["n_components"]
        elif self.type in ("OneHotEncoder",):
            if "extra_levels" in parameters:
                check_types([("extra_levels", parameters["extra_levels"], [dict],)])
                model_parameters["extra_levels"] = parameters["extra_levels"]
            elif "extra_levels" not in self.parameters:
                model_parameters["extra_levels"] = default_parameters["extra_levels"]
            else:
                model_parameters["extra_levels"] = self.parameters["extra_levels"]
            if "drop_first" in parameters:
                check_types([("drop_first", parameters["drop_first"], [bool],)])
                model_parameters["drop_first"] = parameters["drop_first"]
            elif "drop_first" not in self.parameters:
                model_parameters["drop_first"] = default_parameters["drop_first"]
            else:
                model_parameters["drop_first"] = self.parameters["drop_first"]
            if "ignore_null" in parameters:
                check_types([("ignore_null", parameters["ignore_null"], [bool],)])
                model_parameters["ignore_null"] = parameters["ignore_null"]
            elif "ignore_null" not in self.parameters:
                model_parameters["ignore_null"] = default_parameters["ignore_null"]
            else:
                model_parameters["ignore_null"] = self.parameters["ignore_null"]
            if "separator" in parameters:
                check_types([("separator", parameters["separator"], [str],)])
                model_parameters["separator"] = parameters["separator"]
            elif "separator" not in self.parameters:
                model_parameters["separator"] = default_parameters["separator"]
            else:
                model_parameters["separator"] = self.parameters["separator"]
            if "null_column_name" in parameters:
                check_types(
                    [("null_column_name", parameters["null_column_name"], [str],)]
                )
                model_parameters["null_column_name"] = parameters["null_column_name"]
            elif "null_column_name" not in self.parameters:
                model_parameters["null_column_name"] = default_parameters[
                    "null_column_name"
                ]
            else:
                model_parameters["null_column_name"] = self.parameters[
                    "null_column_name"
                ]
            if "column_naming" in parameters:
                check_types([("column_naming", parameters["column_naming"], [str],)])
                assert str(parameters["column_naming"]).lower() in [
                    "indices",
                    "values",
                    "values_relaxed",
                ], ParameterError(
                    "Incorrect parameter 'column_naming'.\nThe column_naming method must be in (indices | values | values_relaxed), found '{}'.".format(
                        parameters["column_naming"]
                    )
                )
                model_parameters["column_naming"] = parameters["column_naming"]
            elif "column_naming" not in self.parameters:
                model_parameters["column_naming"] = default_parameters["column_naming"]
            else:
                model_parameters["column_naming"] = self.parameters["column_naming"]
        elif self.type in ("Normalizer",):
            if "method" in parameters:
                check_types([("method", parameters["method"], [str],)])
                assert str(parameters["method"]).lower() in [
                    "zscore",
                    "robust_zscore",
                    "minmax",
                ], ParameterError(
                    "Incorrect parameter 'method'.\nThe normalization method must be in (zscore | robust_zscore | minmax), found '{}'.".format(
                        parameters["method"]
                    )
                )
                model_parameters["method"] = parameters["method"]
            elif "method" not in self.parameters:
                model_parameters["method"] = default_parameters["method"]
            else:
                model_parameters["method"] = self.parameters["method"]
        elif self.type in ("DBSCAN",):
            if "eps" in parameters:
                check_types([("eps", parameters["eps"], [int, float],)])
                assert 0 < parameters["eps"], ParameterError(
                    "Incorrect parameter 'eps'.\nThe radius of a neighborhood must be strictly positive."
                )
                model_parameters["eps"] = parameters["eps"]
            elif "eps" not in self.parameters:
                model_parameters["eps"] = default_parameters["eps"]
            else:
                model_parameters["eps"] = self.parameters["eps"]
            if "p" in parameters:
                check_types([("p", parameters["p"], [int, float],)])
                assert 0 < parameters["p"], ParameterError(
                    "Incorrect parameter 'p'.\nThe p of the p-distance must be strictly positive."
                )
                model_parameters["p"] = parameters["p"]
            elif "p" not in self.parameters:
                model_parameters["p"] = default_parameters["p"]
            else:
                model_parameters["p"] = self.parameters["p"]
            if "min_samples" in parameters:
                check_types([("min_samples", parameters["min_samples"], [int, float],)])
                assert 0 < parameters["min_samples"], ParameterError(
                    "Incorrect parameter 'min_samples'.\nThe minimum number of points required to form a dense region must be strictly positive."
                )
                model_parameters["min_samples"] = parameters["min_samples"]
            elif "min_samples" not in self.parameters:
                model_parameters["min_samples"] = default_parameters["min_samples"]
            else:
                model_parameters["min_samples"] = self.parameters["min_samples"]
        elif self.type in (
            "NearestCentroid",
            "KNeighborsClassifier",
            "KNeighborsRegressor",
            "LocalOutlierFactor",
        ):
            if "p" in parameters:
                check_types([("p", parameters["p"], [int, float],)])
                assert 0 < parameters["p"], ParameterError(
                    "Incorrect parameter 'p'.\nThe p of the p-distance must be strictly positive."
                )
                model_parameters["p"] = parameters["p"]
            elif "p" not in self.parameters:
                model_parameters["p"] = default_parameters["p"]
            else:
                model_parameters["p"] = self.parameters["p"]
            if ("n_neighbors" in parameters) and (self.type != "NearestCentroid"):
                check_types([("n_neighbors", parameters["n_neighbors"], [int, float],)])
                assert 0 < parameters["n_neighbors"], ParameterError(
                    "Incorrect parameter 'n_neighbors'.\nThe number of neighbors must be strictly positive."
                )
                model_parameters["n_neighbors"] = parameters["n_neighbors"]
            elif (
                self.type != "NearestCentroid" and "n_neighbors" not in self.parameters
            ):
                model_parameters["n_neighbors"] = default_parameters["n_neighbors"]
            elif self.type != "NearestCentroid":
                model_parameters["n_neighbors"] = self.parameters["n_neighbors"]
        elif self.type in ("CountVectorizer",):
            if "max_df" in parameters:
                check_types([("max_df", parameters["max_df"], [int, float],)])
                assert 0 <= parameters["max_df"] <= 1, ParameterError(
                    "Incorrect parameter 'max_df'.\nIt must be between 0 and 1, inclusive."
                )
                model_parameters["max_df"] = parameters["max_df"]
            elif "max_df" not in self.parameters:
                model_parameters["max_df"] = default_parameters["max_df"]
            else:
                model_parameters["max_df"] = self.parameters["max_df"]
            if "min_df" in parameters:
                check_types([("min_df", parameters["min_df"], [int, float],)])
                assert 0 <= parameters["min_df"] <= 1, ParameterError(
                    "Incorrect parameter 'min_df'.\nIt must be between 0 and 1, inclusive."
                )
                model_parameters["min_df"] = parameters["min_df"]
            elif "min_df" not in self.parameters:
                model_parameters["min_df"] = default_parameters["min_df"]
            else:
                model_parameters["min_df"] = self.parameters["min_df"]
            if "lowercase" in parameters:
                check_types([("lowercase", parameters["lowercase"], [bool],)])
                model_parameters["lowercase"] = parameters["lowercase"]
            elif "lowercase" not in self.parameters:
                model_parameters["lowercase"] = default_parameters["lowercase"]
            else:
                model_parameters["lowercase"] = self.parameters["lowercase"]
            if "ignore_special" in parameters:
                check_types([("ignore_special", parameters["ignore_special"], [bool],)])
                model_parameters["ignore_special"] = parameters["ignore_special"]
            elif "ignore_special" not in self.parameters:
                model_parameters["ignore_special"] = default_parameters[
                    "ignore_special"
                ]
            else:
                model_parameters["ignore_special"] = self.parameters["ignore_special"]
            if "max_text_size" in parameters:
                check_types(
                    [
                        (
                            "max_text_size",
                            parameters["max_text_size"],
                            [int, float],
                            False,
                        )
                    ]
                )
                assert 0 < parameters["max_text_size"], ParameterError(
                    "Incorrect parameter 'max_text_size'.\nThe maximum text size must be positive."
                )
                model_parameters["max_text_size"] = parameters["max_text_size"]
            elif "max_text_size" not in self.parameters:
                model_parameters["max_text_size"] = default_parameters["max_text_size"]
            else:
                model_parameters["max_text_size"] = self.parameters["max_text_size"]
            if "max_features" in parameters:
                check_types(
                    [("max_features", parameters["max_features"], [int, float],)]
                )
                model_parameters["max_features"] = parameters["max_features"]
            elif "max_features" not in self.parameters:
                model_parameters["max_features"] = default_parameters["max_features"]
            else:
                model_parameters["max_features"] = self.parameters["max_features"]
        from verticapy.learn.linear_model import Lasso, Ridge, LinearRegression, LogisticRegression
        from verticapy.learn.tree import (
            DecisionTreeClassifier,
            DecisionTreeRegressor,
            DummyTreeClassifier,
            DummyTreeRegressor,
        )

        if isinstance(self, LogisticRegression):
            if model_parameters["penalty"] in ("none", "l1", "l2"):
                if "l1_ratio" in model_parameters:
                    del model_parameters["l1_ratio"]
            if model_parameters["penalty"] in ("none",):
                if "C" in model_parameters:
                    del model_parameters["C"]
        elif isinstance(self, Lasso):
            model_parameters["penalty"] = "l1"
            if "l1_ratio" in model_parameters:
                del model_parameters["l1_ratio"]
        elif isinstance(self, Ridge):
            model_parameters["penalty"] = "l2"
            if "l1_ratio" in model_parameters:
                del model_parameters["l1_ratio"]
        elif isinstance(self, LinearRegression):
            model_parameters["penalty"] = "none"
            if "l1_ratio" in model_parameters:
                del model_parameters["l1_ratio"]
            if "C" in model_parameters:
                del model_parameters["C"]
        elif isinstance(
            self,
            (
                DecisionTreeClassifier,
                DecisionTreeRegressor,
                DummyTreeClassifier,
                DummyTreeRegressor,
            ),
        ):
            model_parameters["n_estimators"] = 1
            model_parameters["sample"] = 1.0
            if isinstance(self, (DummyTreeClassifier, DummyTreeRegressor)):
                model_parameters["max_features"] = "max"
                model_parameters["max_leaf_nodes"] = 1e9
                model_parameters["max_depth"] = 100
                model_parameters["min_samples_leaf"] = 1
                model_parameters["min_info_gain"] = 0.0
        self.parameters = model_parameters

    # ---#
    def shapExplainer(self):
        """
    ---------------------------------------------------------------------------
    Creates the Model shapExplainer. Only linear models are supported.

    Returns
    -------
    shap.Explainer
        the shap Explainer.
        """
        if self.type == "AutoML":
            return self.best_model_.shapExplainer()
        try:
            import shap
        except:
            raise ImportError(
                "The shap module seems to not be installed in your environment.\nTo be able to use this method, you'll have to install it.\n[Tips] Run: 'pip3 install shap' in your terminal to install the module."
            )
        if self.type in (
            "LinearRegression",
            "LogisticRegression",
            "LinearSVC",
            "LinearSVR",
        ):
            vdf = vdf_from_relation(self.input_relation, cursor=self.cursor)
            cov_matrix = vdf.cov(self.X, show=False)
            if len(self.X) == 1:
                cov_matrix = np.array([[1]])
            elif len(self.X) == 2:
                cov_matrix = np.array([[1, cov_matrix], [cov_matrix, 1]])
            else:
                cov_matrix = cov_matrix.to_numpy()
            data = (vdf.avg(self.X).to_numpy(), cov_matrix)
            model = self.to_sklearn()
            with warnings.catch_warnings(record=True) as w:
                return shap.LinearExplainer(
                    model, data, feature_perturbation="correlation_dependent"
                )
        else:
            raise FunctionError(
                "The method 'shapExplainer' is not available for model type '{}'.".format(
                    self.type
                )
            )

    # ---#
    def to_memmodel(self, **kwds,):
        """
    ---------------------------------------------------------------------------
    Converts a specified Vertica model to a memModel model.

    Returns
    -------
    object
        memModel model.
        """
        from verticapy.learn.memmodel import memModel

        if self.type == "AutoML":
            return self.best_model_.to_memmodel()
        elif self.type in (
            "LinearRegression",
            "LogisticRegression",
            "LinearSVC",
            "LinearSVR",
        ):
            attributes = {"coefficients": self.coef_["coefficient"][1:],
                          "intercept": self.coef_["coefficient"][0]}
        elif self.type in (
            "BisectingKMeans",
        ):
            attributes = {"clusters": self.cluster_centers_.to_numpy()[:,1:len(self.X)+1],
                          "left_child": self.cluster_centers_["left_child"],
                          "right_child": self.cluster_centers_["right_child"],
                          "cluster_size": self.cluster_centers_["cluster_size"],
                          "cluster_score": [self.cluster_centers_["withinss"][i] / self.cluster_centers_["totWithinss"][i] for i in range(len(self.cluster_centers_["withinss"]))],
                          "p": 2,}
        elif self.type in (
            "KMeans",
        ):
            attributes = {"clusters": self.cluster_centers_.to_numpy(),
                          "p": 2,}
        elif self.type in (
            "NearestCentroid",
        ):
            attributes = {"clusters": self.centroids_.to_numpy()[:,0:-1],
                          "p": self.parameters["p"],
                          "classes": self.classes_,}
        elif self.type in (
            "NaiveBayes",
        ):
            attributes = {"attributes": nb_var_info(self),
                          "prior": self.get_attr("prior")["probability"],
                          "classes": self.classes_,}
        elif self.type in (
            "OneHotEncoder",
        ):
            cat = list(ooe_details_transform([l[0:2] for l in self.param_.to_list()]))
            cat_list_idx = []
            for i, x1 in enumerate(cat[0]):
                for j, x2 in enumerate(self.X):
                    if x2.lower()[1:-1] == x1:
                        cat_list_idx += [j]
            categories = []
            for i in cat_list_idx:
                categories += [cat[1][i]]
            attributes = {"categories": categories,
                          "drop_first": self.parameters["drop_first"],
                          "column_naming": self.parameters["column_naming"],}
        elif self.type in (
            "PCA",
        ):
            attributes = {"principal_components": self.get_attr("principal_components").to_numpy(),
                          "mean": self.get_attr("columns")["mean"],}
        elif self.type in (
            "SVD",
        ):
            attributes = {"vectors": self.get_attr("right_singular_vectors").to_numpy(),
                          "values": self.get_attr("singular_values")["value"],}
        elif self.type in (
            "Normalizer",
        ):
            attributes = {"values": self.get_attr("details").to_numpy()[:,1:].astype(float),
                          "method": self.parameters["method"],}
        elif self.type in (
            "XGBoostClassifier",
            "XGBoostRegressor",
            "RandomForestClassifier",
            "RandomForestRegressor",
        ):
            if self.type in ("RandomForestClassifier", "RandomForestRegressor"):
                n = self.parameters["n_estimators"]
            else:
                n = self.get_attr("tree_count")["tree_count"][0]
            trees = []
            tree_type = "BinaryTreeRegressor" if self.type in ("XGBoostRegressor", "RandomForestRegressor",) else "BinaryTreeClassifier"
            return_prob_rf = (self.type in ("RandomForestClassifier",))
            for i in range(n):
                if ("return_tree" in kwds and kwds["return_tree"] == i) or ("return_tree" not in kwds):
                    tree = self.get_tree(i)
                    tree = tree_attributes_list(tree, self.X, self.type, return_prob_rf)
                    tree_attributes = {"children_left": tree[0],
                                       "children_right": tree[1],
                                       "feature": tree[2],
                                       "threshold": tree[3],
                                       "value": tree[4],}
                    for idx in range(len(tree[5])):
                        if not(tree[5][idx]) and isinstance(tree_attributes["threshold"][idx], str):
                            tree_attributes["threshold"][idx] = float(tree_attributes["threshold"][idx])
                    if tree_type == "BinaryTreeClassifier":
                        tree_attributes["classes"] = self.classes_
                    else:
                        tree_attributes["value"] = [float(val) if isinstance(val, str) else val for val in tree_attributes["value"]]
                    if self.type in ("XGBoostClassifier",):
                        tree_attributes["value"] = tree[6]
                        for idx in range(len(tree[6])):
                            if tree[6][idx] != None:
                                all_classes_logodss = []
                                for c in self.classes_:
                                    all_classes_logodss += [tree[6][idx][str(c)]]
                                tree_attributes["value"][idx] = all_classes_logodss
                    elif self.type in ("RandomForestClassifier",):
                        for idx in range(len(tree_attributes["value"])):
                            if tree_attributes["value"][idx] != None:
                                prob = [0.0 for i in range(len(self.classes_))]
                                for idx2, c in enumerate(self.classes_):
                                    if c == tree_attributes["value"][idx]:
                                        prob[idx2] = tree[6][idx]
                                        break
                                other_proba = (1 - tree[6][idx]) / (len(self.classes_) - 1)
                                for idx2, p in enumerate(prob):
                                    if p == 0.0:
                                        prob[idx2] = other_proba
                                tree_attributes["value"][idx] = prob
                    model = memModel(model_type = tree_type, attributes = tree_attributes)
                    if ("return_tree" in kwds and kwds["return_tree"] == i):
                        return model
                    trees += [model]
            attributes = {"trees": trees}
            if self.type in ("XGBoostRegressor", "XGBoostClassifier",):
                attributes["learning_rate"] = self.parameters["learning_rate"]
                if self.type in ("XGBoostRegressor",):
                    attributes["mean"] = self.prior_
                elif not(isinstance(self.prior_, list)):
                    attributes["logodds"] = [np.log((1 - self.prior_) / self.prior_), np.log(self.prior_ / (1 - self.prior_))]
                else:
                    attributes["logodds"] = self.prior_.copy()
        else:
            raise ModelError("Model type '{}' can not be converted to memModel.".format(self.type))
        return memModel(model_type = self.type, attributes = attributes)

    # ---#
    def to_sklearn(self):
        """
    ---------------------------------------------------------------------------
    Converts the Vertica model to an sklearn model.

    Returns
    -------
    object
        sklearn model.
        """
        warning_message = "'to_sklearn' method is DEPRECATED and it will be removed in VerticaPy 0.9.0. Use 'to_memmodel' instead."
        warnings.warn(warning_message, Warning)
        if self.type == "AutoML":
            return self.best_model_.to_sklearn()

        import verticapy.learn.linear_model as lm
        import verticapy.learn.svm as svm
        import verticapy.learn.naive_bayes as vnb
        import verticapy.learn.cluster as vcl
        import verticapy.learn.ensemble as vens
        import verticapy.learn.neighbors as vng
        import verticapy.learn.preprocessing as vpp
        import verticapy.learn.decomposition as vdcp

        try:
            import sklearn
        except:
            raise ImportError(
                "The scikit-learn module seems to not be installed in your environment.\nTo be able to use this method, you'll have to install it.\n[Tips] Run: 'pip3 install scikit-learn' in your terminal to install the module."
            )
        params = self.get_params()
        if self.type in (
            "LinearRegression",
            "LogisticRegression",
            "LinearSVC",
            "LinearSVR",
        ):
            import sklearn.linear_model as sklm
            import sklearn.svm as sksvm

            if isinstance(self, lm.LinearRegression):
                model = sklm.LinearRegression()
            elif isinstance(self, lm.ElasticNet):
                model = sklm.ElasticNet(
                    alpha=params["C"],
                    l1_ratio=params["l1_ratio"],
                    max_iter=params["max_iter"],
                    tol=params["tol"],
                )
            elif isinstance(self, lm.Lasso):
                model = sklm.Lasso(max_iter=params["max_iter"], tol=params["tol"],)
            elif isinstance(self, lm.Ridge):
                model = sklm.Ridge(max_iter=params["max_iter"], tol=params["tol"],)
            elif isinstance(self, lm.LogisticRegression):
                if "C" not in params:
                    params["C"] = 1.0
                if "l1_ratio" not in params:
                    params["l1_ratio"] = None
                model = sklm.LogisticRegression(
                    penalty=params["penalty"].lower(),
                    C=float(1 / params["C"]),
                    l1_ratio=params["l1_ratio"],
                    max_iter=params["max_iter"],
                    tol=params["tol"],
                )
            elif isinstance(self, svm.LinearSVC):
                if params["intercept_mode"] == "regularized":
                    params["penalty"] = "l2"
                else:
                    params["penalty"] = "l1"
                model = sksvm.LinearSVC(
                    penalty=params["penalty"],
                    C=params["C"],
                    fit_intercept=params["fit_intercept"],
                    intercept_scaling=params["intercept_scaling"],
                    max_iter=params["max_iter"],
                    tol=params["tol"],
                )
            elif isinstance(self, svm.LinearSVR):
                if params["intercept_mode"] == "regularized":
                    params["loss"] = "epsilon_insensitive"
                else:
                    params["loss"] = "squared_epsilon_insensitive"
                model = sksvm.LinearSVR(
                    loss=params["loss"],
                    C=params["C"],
                    fit_intercept=params["fit_intercept"],
                    intercept_scaling=params["intercept_scaling"],
                    max_iter=params["max_iter"],
                    tol=params["tol"],
                )
            if isinstance(self, (lm.LogisticRegression, svm.LinearSVC)):
                model.classes_ = np.array([0, 1])
            model.coef_ = np.array([self.coef_["coefficient"][1:]])
            model.intercept_ = self.coef_["coefficient"][0]
            try:
                model.n_iter_ = self.get_attr("iteration_count")["iteration_count"][0]
            except:
                model.n_iter_ = 1
        elif self.type in ("Normalizer", "OneHotEncoder"):
            import sklearn.preprocessing as skpp

            if isinstance(self, (vpp.Normalizer,)):
                attr = self.get_attr("details")
                if "avg" in attr.values:
                    model = skpp.StandardScaler()
                    model.mean_ = np.array(attr["avg"])
                    model.scale_ = np.array(attr["std_dev"])
                    model.var_ = model.scale_ ** 2
                    model.n_features_in_ = len(self.X)
                    model.n_samples_seen_ = np.array(
                        vdf_from_relation(
                            self.input_relation, cursor=self.cursor
                        ).count(columns=self.X)["count"]
                    )
                elif "median" in attr.values:
                    model = skpp.RobustScaler()
                    model.center_ = np.array(attr["median"])
                    model.scale_ = np.array(attr["mad"])
                    model.n_features_in_ = len(self.X)
                elif "min" in attr.values:
                    model = skpp.MinMaxScaler()
                    model.data_min_ = np.array(attr["min"])
                    model.data_max_ = np.array(attr["max"])
                    model.data_range_ = np.array(attr["max"]) - np.array(attr["min"])
                    model.scale_ = 1 / model.data_range_
                    model.min_ = 0 - model.data_min_ * model.scale_
                    model.n_features_in_ = len(self.X)
                    self.cursor.execute(
                        "SELECT COUNT(*) FROM {} WHERE {}".format(
                            self.input_relation,
                            " AND ".join(
                                ["{} IS NOT NULL".format(elem) for elem in self.X]
                            ),
                        )
                    )
                    model.n_samples_seen_ = self.cursor.fetchone()[0]
            elif isinstance(self, (vpp.OneHotEncoder,)):
                drop = None
                model = skpp.OneHotEncoder()
                model.drop_idx_ = None
                if self.parameters["drop_first"]:
                    model.drop_idx_ = np.array([0 for elem in range(len(self.X))])
                params = self.param_
                vdf = vdf_from_relation(self.input_relation, cursor=self.cursor)
                categories = []
                for column in self.X:
                    idx = []
                    for i in range(len(params["category_name"])):
                        if str_column(params["category_name"][i]) == str_column(
                            column
                        ) and (
                            not (self.parameters["ignore_null"])
                            or params["category_level"][i] != None
                        ):
                            idx += [i]
                    cat_tmp = []
                    for j, i in enumerate(idx):
                        elem = params["category_level"][i]
                        if vdf[column].dtype() == "int":
                            try:
                                elem = int(elem)
                            except:
                                pass
                        cat_tmp += [elem]
                    categories += [np.array(cat_tmp)]
                model.categories_ = categories
        elif self.type in ("PCA", "SVD", "MCA",):
            import sklearn.decomposition as skdcp

            if isinstance(self, (vdcp.PCA,)):
                model = skdcp.PCA(n_components=params["n_components"])
                model.components_ = []
                all_pc = self.get_attr("principal_components")
                for idx, elem in enumerate(all_pc.values):
                    if idx > 0:
                        model.components_ += [np.array(all_pc.values[elem])]
                model.components_ = np.array(model.components_)
                model.explained_variance_ratio_ = np.array(
                    self.get_attr("singular_values")["explained_variance"]
                )
                model.explained_variance_ = np.array(
                    self.get_attr("singular_values")["explained_variance"]
                )
                model.singular_values_ = np.array(
                    self.get_attr("singular_values")["value"]
                )
                model.mean_ = np.array(self.get_attr("columns")["mean"])
                model.n_components_ = params["n_components"]
                model.n_features_ = len(self.X)
                model.n_samples_ = self.get_attr("counters")["counter_value"][0]
                model.noise_variance_ = 0.0
            elif isinstance(self, (vdcp.SVD,)):
                model = skdcp.TruncatedSVD(n_components=params["n_components"])
                model.components_ = []
                all_pc = self.get_attr("right_singular_vectors")
                for idx, elem in enumerate(all_pc.values):
                    if idx > 0:
                        model.components_ += [np.array(all_pc.values[elem])]
                model.components_ = np.array(model.components_)
                model.explained_variance_ratio_ = np.array(
                    self.get_attr("singular_values")["explained_variance"]
                )
                model.explained_variance_ = np.array(
                    self.get_attr("singular_values")["explained_variance"]
                )
                model.singular_values_ = np.array(
                    self.get_attr("singular_values")["value"]
                )
                for i in range(len(model.components_)):
                    for j in range(len(model.components_[0])):
                        model.components_[i][j] /= model.singular_values_[i]
            elif isinstance(self, (vdcp.MCA,)):
                raise ModelError(
                    "Models of type 'MCA' are not supported by Scikit Learn."
                )
        elif self.type in ("NaiveBayes",):
            import sklearn.naive_bayes as sknb

            if isinstance(self, (vnb.NaiveBayes,)):
                all_attr = self.get_attr()
                current_type = None
                for elem in all_attr["attr_name"][6:]:
                    if current_type is None:
                        if "gaussian" in elem.lower():
                            current_type = "gaussian"
                        elif "multinomial" in elem.lower():
                            current_type = "multinomial"
                        elif "bernoulli" in elem.lower():
                            current_type = "bernoulli"
                        else:
                            current_type = "categorical"
                    elif current_type not in elem.lower():
                        raise ModelError(
                            "Naive Bayes Models using different variables types (multinomial, categorical, gaussian...) is not supported by Scikit Learn."
                        )
                self.cursor.execute(
                    "SELECT COUNT(*) FROM {} WHERE {}".format(
                        self.input_relation,
                        " AND ".join(
                            ["{} IS NOT NULL".format(elem) for elem in self.X]
                        ),
                    )
                )
                total_count = self.cursor.fetchone()[0]
                classes = np.array(self.get_attr("prior")["class"])
                class_prior = np.array(self.get_attr("prior")["probability"])
                if current_type == "gaussian":
                    model = sknb.GaussianNB()
                    model.epsilon_ = 0.0
                    model.sigma_, model.theta_ = [], []
                    for elem in classes:
                        model.sigma_ += [
                            self.get_attr("gaussian.{}".format(elem))["sigma_sq"]
                        ]
                        model.theta_ += [
                            self.get_attr("gaussian.{}".format(elem))["mu"]
                        ]
                    model.sigma_, model.theta_ = (
                        np.array(model.sigma_),
                        np.array(model.theta_),
                    )
                    model.class_prior_ = class_prior
                elif current_type in ("multinomial", "bernoulli"):
                    if current_type == "multinomial":
                        model = sknb.MultinomialNB(alpha=params["alpha"])
                    else:
                        model = sknb.BernoulliNB(alpha=params["alpha"])
                    model.class_log_prior_ = np.log(class_prior)
                    model.n_features_ = len(self.X)
                    model.feature_count_, model.feature_log_prob_ = [], []
                    for elem in classes:
                        model.feature_count_ += [
                            self.get_attr("{}.{}".format(current_type, elem))[
                                "probability"
                            ]
                        ]
                        model.feature_log_prob_ += [
                            self.get_attr("{}.{}".format(current_type, elem))[
                                "probability"
                            ]
                        ]
                    model.feature_count_, model.feature_log_prob_ = (
                        (total_count * np.array(model.feature_count_)).astype(int),
                        np.log(np.array(model.feature_log_prob_)),
                    )
                elif current_type == "categorical":
                    model = sknb.CategoricalNB(alpha=params["alpha"])
                    model.class_log_prior_ = np.log(class_prior)
                    model.n_features_ = len(self.X)
                    model.feature_log_prob_, model.category_count_ = [], []
                    for elem in self.get_attr("details")["predictor"]:
                        if str_column(elem) != str_column(self.y):
                            column_class = []
                            categorical = self.get_attr("categorical.{}".format(elem))
                            for idx in classes:
                                column_class += [categorical[idx]]
                            model.feature_log_prob_ += [np.log(np.array(column_class))]
                            model.category_count_ += [np.array(column_class)]
                    for idx in range(len(model.category_count_)):
                        for i in range(len(model.category_count_[idx])):
                            for j in range(len(model.category_count_[idx][i])):
                                model.category_count_[idx][i][j] = int(
                                    model.category_count_[idx][i][j]
                                    * class_prior[i]
                                    * total_count
                                )
                model.classes_ = classes
                model.class_count_ = (total_count * class_prior).astype(int)
        elif self.type in ("NearestCentroid",):
            import sklearn.neighbors as skng

            if isinstance(self, (vng.NearestCentroid,)):
                if params["p"] == 1:
                    metric = "manhattan"
                elif params["p"] == 2:
                    metric = "euclidean"
                else:
                    raise ModelError(
                        "Model Conversion failed. NearestCentroid using parameter 'p' > 2 is not supported."
                    )
                model = skng.NearestCentroid(metric=metric,)
                model.classes_ = np.array(self.classes_)
                model.centroids_ = []
                for i in range(len(self.classes_)):
                    raw = []
                    for idx, elem in enumerate(self.X):
                        raw += [self.centroids_[elem][i]]
                    model.centroids_ += [raw]
                model.centroids_ = np.array(model.centroids_)
        elif self.type in ("KMeans"):
            import sklearn.cluster as skcl

            if isinstance(self, (vcl.KMeans,)):
                if params["init"] == "kmeanspp":
                    params["init"] = "k-means++"
                model = skcl.KMeans(
                    n_clusters=params["n_cluster"],
                    init=params["init"],
                    max_iter=params["max_iter"],
                    tol=params["tol"],
                )
                centers_attribute = self.get_attr("centers").values
                model.cluster_centers_ = []
                for i in range(params["n_cluster"]):
                    model.cluster_centers_ += [
                        [centers_attribute[elem][i] for elem in centers_attribute]
                    ]
                model.cluster_centers_ = np.array(model.cluster_centers_)
                model.inertia_ = self.metrics_["value"][2]
                model.n_iter_ = int(
                    self.get_attr("metrics")["metrics"][0]
                    .split("Number of iterations performed: ")[1]
                    .split("\n")[0]
                )
                model._n_threads = None
        elif self.type in ("RandomForestClassifier", "RandomForestRegressor"):
            if isinstance(self, (vens.RandomForestClassifier,)) or self.type in (
                "RandomForestClassifier",
                "RandomForestRegressor",
            ):
                raise ModelError(
                    "Model Conversion failed. Tree Based Models are not yet supported."
                )
            import sklearn.tree._tree as sktree
            import sklearn.tree as skdtree
            import sklearn.ensemble as skens

            features = {}
            parameters = {
                "max_depth": params["max_depth"],
                "min_samples_leaf": params["min_samples_leaf"],
                "max_features": params["max_features"],
                "min_impurity_split": params["min_info_gain"],
                "max_leaf_nodes": params["max_leaf_nodes"],
            }
            for i in range(len(self.X)):
                features[str_column(self.X[i]).lower()] = i
            if (
                isinstance(self, (vens.RandomForestRegressor,))
                or self.type == "RandomForestRegressor"
            ):
                model = skens.RandomForestRegressor(
                    n_estimators=params["n_estimators"], **parameters
                )
                model.base_estimator_ = skdtree.DecisionTreeRegressor(**parameters)
            elif (
                isinstance(self, (vens.RandomForestClassifier,))
                or self.type == "RandomForestClassifier"
            ):
                model = skens.RandomForestClassifier(
                    n_estimators=params["n_estimators"], **parameters
                )
                model.base_estimator_ = skdtree.DecisionTreeClassifier(**parameters)
                model.classes_ = np.array(self.classes_)
                model.n_classes_ = len(self.classes_)
            model.n_features_ = len(self.X)
            model.n_outputs_ = 1
            model.features_importance_ = np.array(
                [
                    elem / 100
                    for elem in self.features_importance(show=False,)["importance"]
                ]
            )
            model.estimators_ = []
            for i in range(params["n_estimators"]):
                vtree = self.get_tree(i)
                ti = sktree.Tree(
                    model.n_features_,
                    np.array([1] * model.n_outputs_, dtype=np.intp),
                    model.n_outputs_,
                )
                ti.capacity = len(vtree["node_id"])
                d = {}
                d["max_depth"] = max(vtree["node_depth"])
                d["node_count"] = len(vtree["node_id"])
                d["nodes"] = []
                left_child = np.array(
                    [
                        elem - 1 if elem is not None else -1
                        for elem in vtree["left_child_id"]
                    ]
                )
                right_child = np.array(
                    [
                        elem - 1 if elem is not None else -1
                        for elem in vtree["right_child_id"]
                    ]
                )
                feature = np.array(
                    [
                        features[str_column(elem).lower()] if elem is not None else -2
                        for elem in vtree["split_predictor"]
                    ]
                )
                impurity = np.array(vtree["weighted_information_gain"])
                threshold = np.array(
                    [elem if elem is not None else -2 for elem in vtree["split_value"]]
                )
                n_node_samples = np.array([100 for elem in vtree["right_child_id"]])
                weighted_n_node_samples = np.array(
                    [100.0 for elem in vtree["right_child_id"]]
                )
                for k in range(len(left_child)):
                    d["nodes"] += [
                        (
                            left_child[k],
                            right_child[k],
                            feature[k],
                            threshold[k],
                            impurity[k],
                            n_node_samples[k],
                            weighted_n_node_samples[k],
                        )
                    ]
                dtype = [
                    ("left_child", "<i8"),
                    ("right_child", "<i8"),
                    ("feature", "<i8"),
                    ("threshold", "<f8"),
                    ("impurity", "<f8"),
                    ("n_node_samples", "<i8"),
                    ("weighted_n_node_samples", "<f8"),
                ]
                d["nodes"] = np.array(d["nodes"], dtype=dtype)
                if isinstance(self, (vens.RandomForestClassifier,)):
                    dtree = skdtree.DecisionTreeClassifier(**parameters)
                    dtree.classes_ = np.array(self.classes_)
                    dtree.n_classes_ = len(self.classes_)
                    ti.max_n_classes = len(self.classes_)
                    d["values"] = [
                        [[None for id0 in range(len(self.classes_))]]
                        for id1 in range(len(left_child))
                    ]
                    for k in range(len(left_child)):
                        if left_child[k] == right_child[k] == -1:
                            proba = vtree["probability/variance"][k]
                            for j in range(len(self.classes_)):
                                if int(self.classes_[j]) == int(vtree["prediction"][k]):
                                    d["values"][k][0][j] = (
                                        int((len(self.classes_) - 1) / (1 - proba))
                                        if 1 - proba != 0
                                        else 0
                                    )
                                else:
                                    d["values"][k][0][j] = (
                                        int(1 / proba) if proba != 0 else 0
                                    )
                    d["values"] = np.array(d["values"], dtype=np.float64)
                elif isinstance(self, (vens.RandomForestRegressor,)):
                    dtree = skdtree.DecisionTreeRegressor(**parameters)
                    d["values"] = np.array(
                        [
                            [[vtree["prediction"][id1]]]
                            for id1 in range(len(left_child))
                        ],
                        dtype=np.float64,
                    )
                ti.__setstate__(d)
                dtree.features_importance_ = np.array(
                    [
                        elem / 100
                        for elem in self.features_importance(show=False, tree_id=i)[
                            "importance"
                        ]
                    ]
                )
                if isinstance(parameters["max_features"], str):
                    if parameters["max_features"].lower() == "max":
                        dtree.max_features_ = len(self.X)
                    else:
                        dtree.max_features_ = int(len(self.X) / 3 + 1)
                else:
                    dtree.max_features_ = params["max_features"]
                dtree.n_features_ = len(self.X)
                dtree.n_outputs_ = 1
                dtree.tree_ = ti
                model.estimators_ += [dtree]
        else:
            raise FunctionError(
                "The method 'to_sklearn' is not available for model type '{}'.".format(
                    self.type
                )
            )
        return model

    # ---#
    def to_python(self, name: str = "predict", return_proba: bool = False, return_distance_clusters: bool = False, return_str: bool = False,):
        """
    ---------------------------------------------------------------------------
    Returns the Python code needed to deploy the model without using built-in
    Vertica functions.

    Parameters
    ----------
    name: str, optional
        Function Name.
    return_proba: bool, optional
        If set to True and the model is a classifier, the function will return 
        the model probabilities.
    return_distance_clusters: bool, optional
        If set to True and the model type is KMeans or NearestCentroid, the function 
        will return the model clusters distances.
    return_str: bool, optional
        If set to True, the function str will be returned.


    Returns
    -------
    str / func
        Python function
        """
        if not(return_str):
            func = self.to_python(name=name, return_proba=return_proba, return_distance_clusters=return_distance_clusters, return_str=True,)
            all_vars = {}
            exec(func, {}, all_vars)
            return all_vars[name]
        func = "def {}(X):\n\timport numpy as np\n\t".format(name)
        if self.type in ("LinearRegression", "LinearSVR", "LogisticRegression", "LinearSVC",):
            result = "{} + np.sum(np.array({}) * np.array(X), axis=1)".format(self.coef_["coefficient"][0], self.coef_["coefficient"][1:])
            if self.type in ("LogisticRegression", "LinearSVC",):
                func += f"result = 1 / (1 + np.exp(- ({result})))"
            else:
                func += "result =  " + result
            if return_proba and self.type in ("LogisticRegression", "LinearSVC",):
                func += "\n\treturn np.column_stack((1 - result, result))"
            elif not(return_proba) and self.type in ("LogisticRegression", "LinearSVC",):
                func += "\n\treturn np.where(result > 0.5, 1, 0)"
            else:
                func += "\n\treturn result"
            return func
        elif self.type in ("BisectingKMeans",):
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
            func += "\t\t\tif np.sum((row - centroids[left_node]) ** 2) < np.sum((row - centroids[right_node]) ** 2):\n"
            func += "\t\t\t\treturn predict_tree(right_child, left_child, row, left_node, centroids)\n"
            func += "\t\t\telse:\n"
            func += "\t\t\t\treturn predict_tree(right_child, left_child, row, right_node, centroids)\n"
            func += "\tdef predict_tree_final(row):\n"
            func += "\t\treturn predict_tree(right_child, left_child, row, 0, centroids)\n"
            func += "\treturn np.apply_along_axis(predict_tree_final, 1, X)\n"
            return func
        elif self.type in ("NearestCentroid", "KMeans",):
            centroids = self.centroids_.to_list() if self.type == "NearestCentroid" else self.cluster_centers_.to_list()
            if self.type == "NearestCentroid":
                for center in centroids:
                    del center[-1]
            func += "centroids = np.array({})\n".format(centroids)
            if self.type == "NearestCentroid":
                func += "\tclasses = np.array({})\n".format(self.classes_)
            func += "\tresult = []\n"
            func += "\tfor centroid in centroids:\n"
            func += "\t\tresult += [np.sum((np.array(centroid) - X) ** {}, axis=1) ** (1 / {})]\n".format(self.parameters["p"] if self.type == "NearestCentroid" else 2, self.parameters["p"] if self.type == "NearestCentroid" else 2)
            func += "\tresult = np.column_stack(result)\n"
            if self.type == "NearestCentroid" and return_proba and not(return_distance_clusters):
                func += "\tresult = 1 / (result + 1e-99) / np.sum(1 / (result + 1e-99), axis=1)[:,None]\n"
            elif not(return_distance_clusters):
                func += "\tresult = np.argmin(result, axis=1)\n"
                if self.type == "NearestCentroid" and self.classes_ != [i for i in range(len(self.classes_))]:
                    func += "\tclass_is_str = isinstance(classes[0], str)\n"
                    func += "\tfor idx, c in enumerate(classes):\n"
                    func += "\t\ttmp_idx = str(idx) if class_is_str and idx > 0 else idx\n"
                    func += "\t\tresult = np.where(result == tmp_idx, c, result)\n"
            func += "\treturn result\n"
            return func
        elif self.type in ("PCA", "MCA",):
            avg = self.get_attr("columns")["mean"]
            pca = []
            attr = self.get_attr("principal_components")
            n = len(attr["PC1"])
            for i in range(1, n + 1):
                pca += [attr["PC{}".format(i)]]
            func += "avg_values = np.array({})\n".format(avg)
            func += "\tpca_values = np.array({})\n".format(pca)
            func += "\tresult = (X - avg_values)\n"
            func += "\tL = []\n"
            func += "\tfor i in range({}):\n".format(n)
            func += "\t\tL += [np.sum(result * pca_values[i], axis=1)]\n"
            func += "\tresult = np.column_stack(L)\n"
            func += "\treturn result\n"
            return func
        elif self.type in ("Normalizer",):
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
        elif self.type in ("SVD",):
            sv = []
            attr = self.get_attr("right_singular_vectors")
            n = len(attr["vector1"])
            for i in range(1, n + 1):
                sv += [attr["vector{}".format(i)]]
            value = self.get_attr("singular_values")["value"]
            func += "singular_values = np.array({})\n".format(value)
            func += "\tright_singular_vectors = np.array({})\n".format(sv)
            func += "\tL = []\n"
            n = len(sv[0])
            func += "\tfor i in range({}):\n".format(n)
            func += "\t\tL += [np.sum(X * right_singular_vectors[i] / singular_values[i], axis=1)]\n"
            func += "\tresult = np.column_stack(L)\n"
            func += "\treturn result\n"
            return func
        elif self.type in ("NaiveBayes",):
            var_info_simplified = nb_var_info(self)
            prior = self.get_attr("prior").values["probability"]
            func += "var_info_simplified = {}\n".format(var_info_simplified)
            func += "\tprior = np.array({})\n".format(prior)
            func += "\tclasses = {}\n".format(self.classes_)
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
            func += "\t\t\t\t\tprob = 1 / np.sqrt(2 * np.pi * prob[c]['sigma_sq']) * np.exp(- (float(X[idx]) - prob[c]['mu']) ** 2 / (2 * prob[c]['sigma_sq']))\n"
            func += "\t\t\t\tsub_result += [prob]\n"
            func += "\t\t\tresult += [sub_result]\n"
            func += "\t\tresult = np.array(result).prod(axis=1) * prior\n"
            if return_proba:
                func += "\t\treturn result / result.sum()\n"
            else:
                func += "\t\treturn classes[np.argmax(result)]\n"
            func += "\treturn np.apply_along_axis(naive_bayes_score_row, 1, X)\n"
            return func
        elif self.type in ("OneHotEncoder",):
            predictors = self.X
            details = self.param_.values
            n, m = len(predictors), len(details["category_name"])
            positions = {}
            for i in range(m):
                val = str_column(details["category_name"][i])
                if val not in positions:
                    positions[val] = [i]
                else:
                    positions[val] += [i]
            category_level = []
            for p in predictors:
                pos = positions[p]
                category_level += [details["category_level"][pos[0]:pos[-1] + 1]]
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
        elif self.type in ("RandomForestClassifier", "RandomForestRegressor", "XGBoostRegressor", "XGBoostClassifier",):
            result = []
            if self.type in ("RandomForestClassifier", "RandomForestRegressor"):
                n = self.parameters["n_estimators"]
            else:
                n = self.get_attr("tree_count")["tree_count"][0]
            func += "n = {}\n".format(n)
            if self.type in ("XGBoostClassifier", "RandomForestClassifier"):
                func += "\tclasses = np.array({})\n".format([str(elem) for elem in self.classes_])
            func += "\ttree_list = []\n"
            for i in range(n):
                tree = self.get_tree(i)
                func += "\ttree_list += [{}]\n".format(tree_attributes_list(tree, self.X, self.type))
            func += "\tdef predict_tree(tree, node_id, X,):\n"
            func += "\t\tif tree[0][node_id] == tree[1][node_id]:\n"
            if self.type in ("RandomForestRegressor", "XGBoostRegressor",):
                func += "\t\t\treturn float(tree[4][node_id])\n"
            elif self.type in ("RandomForestClassifier",):
                func += "\t\t\treturn tree[4][node_id]\n"
            else:
                func += "\t\t\treturn tree[6][node_id]\n"
            func += "\t\telse:\n"
            func += "\t\t\tidx, right_node, left_node = tree[2][node_id], tree[1][node_id], tree[0][node_id]\n"
            func += "\t\t\tif (tree[5][node_id] and str(X[idx]) == tree[3][node_id]) or (not(tree[5][node_id]) and float(X[idx]) < float(tree[3][node_id])):\n"
            func += "\t\t\t\treturn predict_tree(tree, left_node, X)\n"
            func += "\t\t\telse:\n"
            func += "\t\t\t\treturn predict_tree(tree, right_node, X)\n"
            func += "\tdef predict_tree_final(X):\n"
            func += "\t\tresult = [predict_tree(tree, 0, X,) for tree in tree_list]\n"
            if self.type in ("XGBoostClassifier", "RandomForestClassifier"):
                func += "\t\tall_classes_score = {}\n"
                func += "\t\tfor elem in classes:\n"
                func += "\t\t\tall_classes_score[elem] = 0\n"
            if self.type in ("XGBoostRegressor", "XGBoostClassifier",):
                if self.type in ("XGBoostRegressor",):
                    avg = self.prior_
                    func += "\t\treturn {} + {} * np.sum(result)\n".format(avg, self.parameters["learning_rate"])
                else:
                    if not(isinstance(self.prior_, list)):
                        func += "\t\tlogodds = np.array([{}, {}])\n".format(np.log((1 - self.prior_) / self.prior_), np.log(self.prior_ / (1 - self.prior_)))
                    else:
                        func += "\t\tlogodds = np.array({})\n".format(self.prior_)
                    func += "\t\tfor idx, elem in enumerate(all_classes_score):\n"
                    func += "\t\t\tfor val in result:\n"
                    func += "\t\t\t\tall_classes_score[elem] += val[elem]\n"
                    func += "\t\t\tall_classes_score[elem] = 1 / (1 + np.exp( - (logodds[idx] + {} * all_classes_score[elem])))\n".format(self.parameters["learning_rate"])
                    func += "\t\tresult = [all_classes_score[elem] for elem in all_classes_score]\n"
            elif self.type in ("RandomForestRegressor",):
                func += "\t\treturn np.mean(result)\n"
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
            raise ModelError("Function to_python not yet available for model type '{}'.".format(self.type))

    # ---#
    def to_sql(self, X: list = [], return_proba: bool = False,):
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
    def fit(
        self,
        input_relation: Union[str, vDataFrame],
        X: list,
        y: str,
        test_relation: Union[str, vDataFrame] = "",
    ):
        """
	---------------------------------------------------------------------------
	Trains the model.

	Parameters
	----------
	input_relation: str/vDataFrame
		Training relation.
	X: list
		List of the predictors.
	y: str
		Response column.
	test_relation: str/vDataFrame, optional
		Relation used to test the model.

	Returns
	-------
	object
		model
		"""
        if isinstance(X, str):
            X = [X]
        check_types(
            [
                ("input_relation", input_relation, [str, vDataFrame],),
                ("X", X, [list],),
                ("y", y, [str],),
                ("test_relation", test_relation, [str, vDataFrame],),
            ]
        )
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
                input_relation = vdf_from_relation(input_relation, cursor=self.cursor)
            input_relation.astype(new_types)
        self.cursor = check_cursor(self.cursor, input_relation, True)[0]
        does_model_exist(name=self.name, cursor=self.cursor, raise_error=True)
        if isinstance(input_relation, vDataFrame):
            self.input_relation = input_relation.__genSQL__()
            schema = schema_relation(self.name)[0]
            relation = "{}._VERTICAPY_TEMPORARY_VIEW_{}".format(
                str_column(schema), get_session(self.cursor)
            )
            self.cursor.execute("DROP VIEW IF EXISTS {}".format(relation))
            self.cursor.execute(
                "CREATE VIEW {} AS SELECT * FROM {}".format(
                    relation, input_relation.__genSQL__()
                )
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
        self.X = [str_column(column) for column in X]
        self.y = str_column(y)
        parameters = vertica_param_dict(self)
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
        fun = self.get_model_fun()[0]
        query = "SELECT {}('{}', '{}', '{}', '{}' USING PARAMETERS "
        query = query.format(fun, self.name, relation, self.y, ", ".join(self.X))
        query += ", ".join(
            ["{} = {}".format(elem, parameters[elem]) for elem in parameters]
        )
        if alpha != None:
            query += ", alpha = {}".format(alpha)
        if self.type in (
            "RandomForestClassifier",
            "RandomForestRegressor",
            "XGBoostRegressor",
            "XGBoostRegressor",
        ) and isinstance(verticapy.options["random_state"], int):
            query += ", seed={}, id_column='{}'".format(
                verticapy.options["random_state"], X[0],
            )
        if self.type in ("BisectingKMeans",) and isinstance(
            verticapy.options["random_state"], int
        ):
            query += ", kmeans_seed={}, id_column='{}'".format(
                verticapy.options["random_state"], X[0],
            )
        query += ")"
        try:
            executeSQL(self.cursor, query, "Fitting the model.")
            if isinstance(input_relation, vDataFrame):
                self.cursor.execute("DROP VIEW {};".format(relation))
        except:
            if isinstance(input_relation, vDataFrame):
                self.cursor.execute("DROP VIEW {};".format(relation))
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
                self.cursor.execute(
                    "SELECT DISTINCT {} FROM {} WHERE {} IS NOT NULL ORDER BY 1".format(
                        self.y, input_relation, self.y
                    )
                )
                classes = self.cursor.fetchall()
                self.classes_ = [item[0] for item in classes]
            else:
                self.classes_ = input_relation[self.y].distinct()
        if self.type in ("XGBoostClassifier", "XGBoostRegressor",):
            self.prior_ = xgb_prior(self)
        return self


# ---#
class Tree:

    # ---#
    def to_graphviz(self, 
                    tree_id: int = 0,
                    classes_color: list = [],
                    round_pred: int = 2,
                    percent: bool = False,
                    vertical: bool = True,
                    node_style: dict = {},
                    arrow_style: dict = {},
                    leaf_style: dict = {},):
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
        return self.to_memmodel(return_tree = tree_id).to_graphviz(feature_names = self.X,
                                                                   classes_color = classes_color,
                                                                   round_pred = round_pred,
                                                                   percent = percent,
                                                                   vertical = vertical,
                                                                   node_style = node_style,
                                                                   arrow_style = arrow_style,
                                                                   leaf_style = leaf_style,)

    # ---#
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
        check_types([("tree_id", tree_id, [int, float],)])
        version(cursor=self.cursor, condition=[9, 1, 1])
        name = self.tree_name if self.type in ("KernelDensity") else self.name
        query = "SELECT * FROM (SELECT READ_TREE ( USING PARAMETERS model_name = '{}', tree_id = {}, format = 'tabular')) x ORDER BY node_id;".format(
            name, tree_id
        )
        result = to_tablesample(query=query, cursor=self.cursor, title="Reading Tree.",)
        return result

    # ---#
    def plot_tree(self,
                  pic_path: str = "",
                  tree_id: int = 0,
                  classes_color: list = [],
                  round_pred: int = 2,
                  percent: bool = False,
                  vertical: bool = True,
                  node_style: dict = {},
                  arrow_style: dict = {},
                  leaf_style: dict = {},):
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
        return self.to_memmodel(return_tree = tree_id).plot_tree(feature_names = self.X,
                                                                 classes_color = classes_color,
                                                                 round_pred = round_pred,
                                                                 percent = percent,
                                                                 vertical = vertical,
                                                                 node_style = node_style,
                                                                 arrow_style = arrow_style,
                                                                 leaf_style = leaf_style,
                                                                 pic_path = pic_path,)


# ---#
class Classifier(Supervised):
    pass


# ---#
class BinaryClassifier(Classifier):

    classes_ = [0, 1]

    # ---#
    def classification_report(self, cutoff: float = 0.5):
        """
	---------------------------------------------------------------------------
	Computes a classification report using multiple metrics to evaluate the model
	(AUC, accuracy, PRC AUC, F1...). 

	Parameters
	----------
	cutoff: float, optional
		Probability cutoff.

	Returns
	-------
	tablesample
		An object containing the result. For more information, see
		utilities.tablesample.
		"""
        check_types([("cutoff", cutoff, [int, float],)])
        if cutoff > 1 or cutoff < 0:
            cutoff = self.score(method="best_cutoff")
        return classification_report(
            self.y,
            [self.deploySQL(), self.deploySQL(cutoff)],
            self.test_relation,
            self.cursor,
            cutoff=cutoff,
        )

    # ---#
    def confusion_matrix(self, cutoff: float = 0.5):
        """
	---------------------------------------------------------------------------
	Computes the model confusion matrix.

	Parameters
	----------
	cutoff: float, optional
		Probability cutoff.

	Returns
	-------
	tablesample
		An object containing the result. For more information, see
		utilities.tablesample.
		"""
        check_types([("cutoff", cutoff, [int, float],)])
        return confusion_matrix(
            self.y, self.deploySQL(cutoff), self.test_relation, self.cursor
        )

    # ---#
    def deploySQL(self, cutoff: float = -1, X: list = []):
        """
	---------------------------------------------------------------------------
	Returns the SQL code needed to deploy the model. 

	Parameters
	----------
	cutoff: float, optional
		Probability cutoff. If this number is not between 0 and 1, the method 
		will return the probability to be of class 1.
	X: list, optional
		List of the columns used to deploy the model. If empty, the model
		predictors will be used.

	Returns
	-------
	str
		the SQL code needed to deploy the model.
		"""
        if isinstance(X, str):
            X = [X]
        check_types([("cutoff", cutoff, [int, float],), ("X", X, [list],)])
        X = [str_column(elem) for elem in X]
        fun = self.get_model_fun()[1]
        sql = "{}({} USING PARAMETERS model_name = '{}', type = 'probability', match_by_pos = 'true')"
        if cutoff <= 1 and cutoff >= 0:
            sql = "(CASE WHEN {} > {} THEN 1 ELSE 0 END)".format(sql, cutoff)
        return sql.format(fun, ", ".join(self.X if not (X) else X), self.name)

    # ---#
    def lift_chart(
        self, ax=None, nbins: int = 1000, **style_kwds,
    ):
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
            self.cursor,
            ax=ax,
            nbins=nbins,
            **style_kwds,
        )

    # ---#
    def prc_curve(
        self, ax=None, nbins: int = 30, **style_kwds,
    ):
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
            self.cursor,
            ax=ax,
            nbins=nbins,
            **style_kwds,
        )

    # ---#
    def predict(
        self,
        vdf: Union[str, vDataFrame],
        X: list = [],
        name: str = "",
        cutoff: float = -1,
        inplace: bool = True,
    ):
        """
	---------------------------------------------------------------------------
	Predicts using the input relation.

	Parameters
	----------
	vdf: str/vDataFrame
		Object to use to run the prediction. You can also specify a customized 
        relation, but you must enclose it with an alias. For example "(SELECT 1) x" 
        is correct whereas "(SELECT 1)" and "SELECT 1" are incorrect.
	X: list, optional
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
        if isinstance(X, str):
            X = [X]
        check_types(
            [
                ("name", name, [str],),
                ("cutoff", cutoff, [int, float],),
                ("X", X, [list],),
                ("vdf", vdf, [str, vDataFrame],),
            ],
        )
        if isinstance(vdf, str):
            vdf = vdf_from_relation(relation=vdf, cursor=self.cursor)
        X = [str_column(elem) for elem in X]
        name = (
            "{}_".format(self.type) + "".join(ch for ch in self.name if ch.isalnum())
            if not (name)
            else name
        )
        if inplace:
            return vdf.eval(name, self.deploySQL(cutoff=cutoff, X=X))
        else:
            return vdf.copy().eval(name, self.deploySQL(cutoff=cutoff, X=X))

    # ---#
    def cutoff_curve(
        self, ax=None, nbins: int = 30, **style_kwds,
    ):
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
            self.cursor,
            ax=ax,
            cutoff_curve=True,
            nbins=nbins,
            **style_kwds,
        )

    # ---#
    def roc_curve(
        self, ax=None, nbins: int = 30, **style_kwds,
    ):
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
            self.cursor,
            ax=ax,
            nbins=nbins,
            **style_kwds,
        )

    # ---#
    def score(self, 
              method: str = "accuracy", 
              cutoff: float = 0.5,
              nbins: int = 10000,):
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
	cutoff: float, optional
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
        check_types([("cutoff", cutoff, [int, float],), 
                     ("method", method, [str],),
                     ("nbins", nbins, [int],)])
        if method in ("accuracy", "acc"):
            return accuracy_score(
                self.y, self.deploySQL(cutoff), self.test_relation, self.cursor, pos_label=1,
            )
        elif method == "aic":
            return aic_bic(self.y, self.deploySQL(), self.test_relation, self.cursor, len(self.X))[0]
        elif method == "bic":
            return aic_bic(self.y, self.deploySQL(), self.test_relation, self.cursor, len(self.X))[1]
        elif method == "prc_auc":
            return prc_curve(self.y, self.deploySQL(), self.test_relation, self.cursor, auc_prc=True, nbins=nbins,)
        elif method == "auc":
            return roc_curve(self.y, self.deploySQL(), self.test_relation, self.cursor, auc_roc=True, nbins=nbins,)
        elif method in ("best_cutoff", "best_threshold"):
            return roc_curve(
                self.y,
                self.deploySQL(),
                self.test_relation,
                self.cursor,
                best_threshold=True,
                nbins=nbins,
            )
        elif method in ("recall", "tpr"):
            return recall_score(
                self.y, self.deploySQL(cutoff), self.test_relation, self.cursor
            )
        elif method in ("precision", "ppv"):
            return precision_score(
                self.y, self.deploySQL(cutoff), self.test_relation, self.cursor
            )
        elif method in ("specificity", "tnr"):
            return specificity_score(
                self.y, self.deploySQL(cutoff), self.test_relation, self.cursor
            )
        elif method in ("negative_predictive_value", "npv"):
            return precision_score(
                self.y, self.deploySQL(cutoff), self.test_relation, self.cursor
            )
        elif method in ("log_loss", "logloss"):
            return log_loss(self.y, self.deploySQL(), self.test_relation, self.cursor)
        elif method == "f1":
            return f1_score(
                self.y, self.deploySQL(cutoff), self.test_relation, self.cursor
            )
        elif method == "mcc":
            return matthews_corrcoef(
                self.y, self.deploySQL(cutoff), self.test_relation, self.cursor
            )
        elif method in ("bm", "informedness"):
            return informedness(
                self.y, self.deploySQL(cutoff), self.test_relation, self.cursor
            )
        elif method in ("mk", "markedness"):
            return markedness(
                self.y, self.deploySQL(cutoff), self.test_relation, self.cursor
            )
        elif method in ("csi", "critical_success_index"):
            return critical_success_index(
                self.y, self.deploySQL(cutoff), self.test_relation, self.cursor
            )
        else:
            raise ParameterError(
                "The parameter 'method' must be in accuracy|auc|prc_auc|best_cutoff|recall|precision|log_loss|negative_predictive_value|specificity|mcc|informedness|markedness|critical_success_index|aic|bic"
            )


# ---#
class MulticlassClassifier(Classifier):

    # ---#
    def classification_report(self, cutoff: Union[float, list] = [], labels: list = []):
        """
	---------------------------------------------------------------------------
	Computes a classification report using multiple metrics to evaluate the model
	(AUC, accuracy, PRC AUC, F1...). For multiclass classification, it will consider 
    each category as positive and switch to the next one during the computation.

	Parameters
	----------
	cutoff: float/list, optional
		Cutoff for which the tested category will be accepted as a prediction.
		For multiclass classification, each tested category becomes 
		the positives and the others are merged into the negatives. The list will 
		represent the classes threshold. If it is empty, the best cutoff will be used.
	labels: list, optional
		List of the different labels to be used during the computation.

	Returns
	-------
	tablesample
		An object containing the result. For more information, see
		utilities.tablesample.
		"""
        if isinstance(labels, str):
            labels = [labels]
        check_types(
            [("cutoff", cutoff, [int, float, list],), ("labels", labels, [list],),]
        )
        if not (labels):
            labels = self.classes_
        return classification_report(
            cutoff=cutoff, estimator=self, labels=labels, cursor=self.cursor
        )

    # ---#
    def confusion_matrix(self, pos_label: Union[int, float, str] = None, cutoff: float = -1):
        """
	---------------------------------------------------------------------------
	Computes the model confusion matrix.

	Parameters
	----------
	pos_label: int/float/str, optional
		Label to consider as positive. All the other classes will be merged and
		considered as negative for multiclass classification.
	cutoff: float, optional
		Cutoff for which the tested category will be accepted as a prediction.If the 
		cutoff is not between 0 and 1, the entire confusion matrix will be drawn.

	Returns
	-------
	tablesample
		An object containing the result. For more information, see
		utilities.tablesample.
		"""
        check_types([("cutoff", cutoff, [int, float],)])
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
                self.cursor,
                pos_label=pos_label,
            )
        else:
            return multilabel_confusion_matrix(
                self.y, self.deploySQL(), self.test_relation, self.classes_, self.cursor
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
            deploySQL_str = self.deploySQL(allSQL=True)[get_index(pos_label, self.classes_, False,)]
        else:
            deploySQL_str = self.deploySQL(allSQL=True)[0].format(pos_label)
        return roc_curve(
            self.y,
            deploySQL_str,
            self.test_relation,
            self.cursor,
            pos_label,
            ax=ax,
            cutoff_curve=True,
            nbins=nbins,
            **style_kwds,
        )

    # ---#
    def deploySQL(
        self,
        pos_label: Union[int, float, str] = None,
        cutoff: float = -1,
        allSQL: bool = False,
        X: list = [],
    ):
        """
	---------------------------------------------------------------------------
	Returns the SQL code needed to deploy the model. 

	Parameters
	----------
	pos_label: int/float/str, optional
		Label to consider as positive. All the other classes will be merged and
		considered as negative for multiclass classification.
	cutoff: float, optional
		Cutoff for which the tested category will be accepted as a prediction.If 
		the cutoff is not between 0 and 1, a probability will be returned.
	allSQL: bool, optional
		If set to True, the output will be a list of the different SQL codes 
		needed to deploy the different categories score.
	X: list, optional
		List of the columns used to deploy the model. If empty, the model
		predictors will be used.

	Returns
	-------
	str / list
		the SQL code needed to deploy the self.
		"""
        if isinstance(X, str):
            X = [X]
        check_types(
            [
                ("cutoff", cutoff, [int, float],),
                ("allSQL", allSQL, [bool],),
                ("X", X, [list],),
            ]
        )
        X = [str_column(elem) for elem in X]
        fun = self.get_model_fun()[1]
        if allSQL:
            if self.type == "NearestCentroid":
                sql = self.to_memmodel().predict_proba_sql(self.X if not (X) else X)
            else:
                sql = "{}({} USING PARAMETERS model_name = '{}', class = '{}', type = 'probability', match_by_pos = 'true')".format(
                    fun, ", ".join(self.X if not (X) else X), self.name, "{}"
                )
                sql = [
                    sql,
                    "{}({} USING PARAMETERS model_name = '{}', match_by_pos = 'true')".format(
                        fun, ", ".join(self.X if not (X) else X), self.name
                    ),
                ]
        else:
            if pos_label in self.classes_:
                if self.type == "NearestCentroid":
                    sql = self.to_memmodel().predict_proba_sql(self.X if not (X) else X)[get_index(pos_label, self.classes_, False,)]
                else:
                    sql = "{}({} USING PARAMETERS model_name = '{}', class = '{}', type = 'probability', match_by_pos = 'true')".format(
                        fun, ", ".join(self.X if not (X) else X), self.name, pos_label
                    )
            if pos_label in self.classes_ and cutoff <= 1 and cutoff >= 0:
                if len(self.classes_) > 2:
                    sql = "(CASE WHEN {} >= {} THEN '{}' WHEN {} IS NULL THEN NULL ELSE 'Non-{}' END)".format(
                        sql, cutoff, pos_label, sql, pos_label
                    )
                else:
                    non_pos_label = (
                        self.classes_[0]
                        if (self.classes_[0] != pos_label)
                        else self.classes_[1]
                    )
                    sql = "(CASE WHEN {} >= {} THEN '{}' WHEN {} IS NULL THEN NULL ELSE '{}' END)".format(
                        sql, cutoff, pos_label, sql, non_pos_label
                    )
            elif pos_label not in self.classes_:
                if self.type == "NearestCentroid":
                    sql = self.to_memmodel().predict_sql(self.X if not (X) else X)
                else:
                    sql = "{}({} USING PARAMETERS model_name = '{}', match_by_pos = 'true')".format(
                        fun, ", ".join(self.X if not (X) else X), self.name
                    )
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
            deploySQL_str = self.deploySQL(allSQL=True)[get_index(pos_label, self.classes_, False,)]
        else:
            deploySQL_str = self.deploySQL(allSQL=True)[0].format(pos_label)
        return lift_chart(
            self.y,
            deploySQL_str,
            self.test_relation,
            self.cursor,
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
            deploySQL_str = self.deploySQL(allSQL=True)[get_index(pos_label, self.classes_, False,)]
        else:
            deploySQL_str = self.deploySQL(allSQL=True)[0].format(pos_label)
        return prc_curve(
            self.y,
            deploySQL_str,
            self.test_relation,
            self.cursor,
            pos_label,
            ax=ax,
            nbins=nbins,
            **style_kwds,
        )

    # ---#
    def predict(
        self,
        vdf: Union[str, vDataFrame],
        X: list = [],
        name: str = "",
        cutoff: float = -1,
        pos_label: Union[int, str, float] = None,
        inplace: bool = True,
    ):
        """
	---------------------------------------------------------------------------
	Predicts using the input relation.

	Parameters
	----------
	vdf: str/vDataFrame
		Object to use to run the prediction. You can also specify a customized 
        relation, but you must enclose it with an alias. For example "(SELECT 1) x" 
        is correct whereas "(SELECT 1)" and "SELECT 1" are incorrect.
	X: list, optional
		List of the columns used to deploy the models. If empty, the model
		predictors will be used.
	name: str, optional
		Name of the added vcolumn. If empty, a name will be generated.
	cutoff: float, optional
		Cutoff for which the tested category will be accepted as a prediction.
		If the parameter is not between 0 and 1, the class probability will
		be returned.
	pos_label: int/float/str, optional
		Class label.
	inplace: bool, optional
		If set to True, the prediction will be added to the vDataFrame.

	Returns
	-------
	vDataFrame
		the input object.
		"""
        if isinstance(X, str):
            X = [X]
        check_types(
            [
                ("name", name, [str],),
                ("cutoff", cutoff, [int, float],),
                ("X", X, [list],),
                ("vdf", vdf, [str, vDataFrame],),
            ],
        )
        if isinstance(vdf, str):
            vdf = vdf_from_relation(relation=vdf, cursor=self.cursor)
        X = [str_column(elem) for elem in X]
        name = (
            "{}_".format(self.type) + "".join(ch for ch in self.name if ch.isalnum())
            if not (name)
            else name
        )
        if len(self.classes_) == 2 and pos_label == None:
            pos_label = self.classes_[1]
        if inplace:
            return vdf.eval(
                name, self.deploySQL(pos_label=pos_label, cutoff=cutoff, X=X)
            )
        else:
            return vdf.copy().eval(
                name, self.deploySQL(pos_label=pos_label, cutoff=cutoff, X=X)
            )

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
            deploySQL_str = self.deploySQL(allSQL=True)[get_index(pos_label, self.classes_, False,)]
        else:
            deploySQL_str = self.deploySQL(allSQL=True)[0].format(pos_label)
        return roc_curve(
            self.y,
            deploySQL_str,
            self.test_relation,
            self.cursor,
            pos_label,
            ax=ax,
            nbins=nbins,
            **style_kwds,
        )

    # ---#
    def score(
        self,
        method: str = "accuracy",
        pos_label: Union[int, float, str] = None,
        cutoff: float = -1,
        nbins: int = 10000,
    ):
        """
	---------------------------------------------------------------------------
	Computes the model score.

	Parameters
	----------
	pos_label: int/float/str, optional
		Label to consider as positive. All the other classes will be merged and
		considered as negative for multiclass classification.
	cutoff: float, optional
		Cutoff for which the tested category will be accepted as a prediction.
		If the parameter is not between 0 and 1, an automatic cutoff is 
		computed.
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
        check_types([("cutoff", cutoff, [int, float],), 
                     ("method", method, [str],),
                     ("nbins", nbins, [int],)])
        pos_label = (
            self.classes_[1]
            if (pos_label == None and len(self.classes_) == 2)
            else pos_label
        )
        if (pos_label not in self.classes_) and (method != "accuracy"):
            raise ParameterError(
                "'pos_label' must be one of the response column classes"
            )
        elif (cutoff >= 1 or cutoff <= 0) and (method != "accuracy"):
            cutoff = self.score("best_cutoff", pos_label, 0.5)
        if self.type == "NearestCentroid":
            deploySQL_str = self.deploySQL(allSQL=True)[get_index(pos_label, self.classes_, False,)]
        else:
            deploySQL_str = self.deploySQL(allSQL=True)[0].format(pos_label)
        if method in ("accuracy", "acc"):
            return accuracy_score(
                self.y,
                self.deploySQL(pos_label, cutoff),
                self.test_relation,
                self.cursor,
                pos_label,
            )
        elif method == "auc":
            return auc(
                "DECODE({}, '{}', 1, 0)".format(self.y, pos_label),
                deploySQL_str,
                self.test_relation,
                self.cursor,
                nbins=nbins,
            )
        elif method == "aic":
            return aic_bic(
                "DECODE({}, '{}', 1, 0)".format(self.y, pos_label),
                deploySQL_str,
                self.test_relation,
                self.cursor,
                len(self.X),
            )[0]
        elif method == "bic":
            return aic_bic(
                "DECODE({}, '{}', 1, 0)".format(self.y, pos_label),
                deploySQL_str,
                self.test_relation,
                self.cursor,
                len(self.X),
            )[1]
        elif method == "prc_auc":
            return prc_auc(
                "DECODE({}, '{}', 1, 0)".format(self.y, pos_label),
                deploySQL_str,
                self.test_relation,
                self.cursor,
                nbins=nbins,
            )
        elif method in ("best_cutoff", "best_threshold"):
            return roc_curve(
                "DECODE({}, '{}', 1, 0)".format(self.y, pos_label),
                deploySQL_str,
                self.test_relation,
                self.cursor,
                best_threshold=True,
                nbins=nbins,
            )
        elif method in ("recall", "tpr"):
            return recall_score(
                self.y,
                self.deploySQL(pos_label, cutoff),
                self.test_relation,
                self.cursor,
            )
        elif method in ("precision", "ppv"):
            return precision_score(
                self.y,
                self.deploySQL(pos_label, cutoff),
                self.test_relation,
                self.cursor,
            )
        elif method in ("specificity", "tnr"):
            return specificity_score(
                self.y,
                self.deploySQL(pos_label, cutoff),
                self.test_relation,
                self.cursor,
            )
        elif method in ("negative_predictive_value", "npv"):
            return precision_score(
                self.y,
                self.deploySQL(pos_label, cutoff),
                self.test_relation,
                self.cursor,
            )
        elif method in ("log_loss", "logloss"):
            return log_loss(
                "DECODE({}, '{}', 1, 0)".format(self.y, pos_label),
                deploySQL_str,
                self.test_relation,
                self.cursor,
            )
        elif method == "f1":
            return f1_score(
                self.y,
                self.deploySQL(pos_label, cutoff),
                self.test_relation,
                self.cursor,
            )
        elif method == "mcc":
            return matthews_corrcoef(
                self.y,
                self.deploySQL(pos_label, cutoff),
                self.test_relation,
                self.cursor,
            )
        elif method in ("bm", "informedness"):
            return informedness(
                self.y,
                self.deploySQL(pos_label, cutoff),
                self.test_relation,
                self.cursor,
            )
        elif method in ("mk", "markedness"):
            return markedness(
                self.y,
                self.deploySQL(pos_label, cutoff),
                self.test_relation,
                self.cursor,
            )
        elif method in ("csi", "critical_success_index"):
            return critical_success_index(
                self.y,
                self.deploySQL(pos_label, cutoff),
                self.test_relation,
                self.cursor,
            )
        else:
            raise ParameterError(
                "The parameter 'method' must be in accuracy|auc|prc_auc|best_cutoff|recall|precision|log_loss|negative_predictive_value|specificity|mcc|informedness|markedness|critical_success_index|aic|bic"
            )


# ---#
class Regressor(Supervised):

    # ---#
    def predict(
        self, vdf: Union[str, vDataFrame], X: list = [], name: str = "", inplace: bool = True
    ):
        """
	---------------------------------------------------------------------------
	Predicts using the input relation.

	Parameters
	----------
	vdf: str/vDataFrame
		Object to use to run the prediction. You can also specify a customized 
        relation, but you must enclose it with an alias. For example "(SELECT 1) x" 
        is correct whereas "(SELECT 1)" and "SELECT 1" are incorrect.
	X: list, optional
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
        check_types(
            [
                ("name", name, [str],),
                ("X", X, [list],),
                ("vdf", vdf, [str, vDataFrame],),
            ],
        )
        if isinstance(vdf, str):
            vdf = vdf_from_relation(relation=vdf, cursor=self.cursor)
        X = [str_column(elem) for elem in X]
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
        check_types([("method", method, ["anova", "metrics", "details"],)])
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
                    self.cursor,
                    len(self.X) * self.parameters["p"],
                ).values["value"]
            return result
        elif self.type == "KernelDensity":
            test_relation = self.map
        else:
            test_relation = self.test_relation
        if method == "metrics":
            return regression_report(
                self.y, prediction, test_relation, self.cursor, len(self.X)
            )
        elif method == "anova":
            return anova_table(
                self.y, prediction, test_relation, len(self.X), self.cursor
            )
        elif method == "details":
            vdf = vdf_from_relation(
                "(SELECT {} FROM ".format(self.y)
                + self.input_relation
                + ") VERTICAPY_SUBTABLE",
                cursor=self.cursor,
            )
            n = vdf[self.y].count()
            kurt = vdf[self.y].kurt()
            skew = vdf[self.y].skew()
            jb = vdf[self.y].agg(["jb"])[self.y][0]
            R2 = self.score()
            R2_adj = 1 - ((1 - R2) * (n - 1) / (n - len(self.X) - 1))
            anova_T = anova_table(
                self.y, prediction, test_relation, len(self.X), self.cursor
            )
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

    # ---#
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
        check_types([("method", method, [str],)])
        method = method.lower()
        if method in ("r2a", "r2adj", "r2adjusted"):
            method = "r2"
            adj = True
        else:
            adj = False
        if method in ("rmse",):
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
                test_relation = test_relation.replace("[X{}]".format(idx), elem)
            prediction = "prediction"
        elif self.type == "VAR":
            relation = self.transform_relation.replace(
                "[VerticaPy_ts]", self.ts
            ).format(self.test_relation)
            for idx, elem in enumerate(self.X):
                relation = relation.replace("[X{}]".format(idx), elem)
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
        if method in ("aic",):
            if self.type == "VAR":
                for idx, y in enumerate(self.X):
                    result.values[y] = [
                        aic_bic(
                            y,
                            self.deploySQL()[idx],
                            relation,
                            self.cursor,
                            len(self.X),
                        )[0]
                    ]
            else:
                return aic_bic(
                    self.y, prediction, test_relation, self.cursor, len(self.X),
                )[0]
        elif method in ("bic",):
            if self.type == "VAR":
                for idx, y in enumerate(self.X):
                    result.values[y] = [
                        aic_bic(
                            y, self.deploySQL()[idx], relation, self.cursor, len(self.X)
                        )[1]
                    ]
            else:
                return aic_bic(
                    self.y, prediction, test_relation, self.cursor, len(self.X)
                )[1]
        elif method in ("r2", "rsquared"):
            if self.type == "VAR":
                for idx, y in enumerate(self.X):
                    result.values[y] = [
                        r2_score(
                            y,
                            self.deploySQL()[idx],
                            relation,
                            self.cursor,
                            len(self.X) * self.parameters["p"],
                            adj,
                        )
                    ]
            else:
                return r2_score(
                    self.y, prediction, test_relation, self.cursor, len(self.X), adj
                )
        elif method in ("mae", "mean_absolute_error"):
            if self.type == "VAR":
                for idx, y in enumerate(self.X):
                    result.values[y] = [
                        mean_absolute_error(
                            y, self.deploySQL()[idx], relation, self.cursor
                        )
                    ]
            else:
                return mean_absolute_error(
                    self.y, prediction, test_relation, self.cursor
                )
        elif method in ("mse", "mean_squared_error"):
            if self.type == "VAR":
                for idx, y in enumerate(self.X):
                    result.values[y] = [
                        mean_squared_error(
                            y, self.deploySQL()[idx], relation, self.cursor, root
                        )
                    ]
            else:
                return mean_squared_error(
                    self.y, prediction, test_relation, self.cursor, root
                )
        elif method in ("msle", "mean_squared_log_error"):
            if self.type == "VAR":
                for idx, y in enumerate(self.X):
                    result.values[y] = [
                        mean_squared_log_error(
                            y, self.deploySQL()[idx], relation, self.cursor
                        )
                    ]
            else:
                return mean_squared_log_error(
                    self.y, prediction, test_relation, self.cursor
                )
        elif method in ("max", "max_error"):
            if self.type == "VAR":
                for idx, y in enumerate(self.X):
                    result.values[y] = [
                        max_error(y, self.deploySQL()[idx], relation, self.cursor)
                    ]
            else:
                return max_error(self.y, prediction, test_relation, self.cursor)
        elif method in ("median", "median_absolute_error"):
            if self.type == "VAR":
                for idx, y in enumerate(self.X):
                    result.values[y] = [
                        median_absolute_error(
                            y, self.deploySQL()[idx], relation, self.cursor
                        )
                    ]
            else:
                return median_absolute_error(
                    self.y, prediction, test_relation, self.cursor
                )
        elif method in ("var", "explained_variance"):
            if self.type == "VAR":
                for idx, y in enumerate(self.X):
                    result.values[y] = [
                        explained_variance(
                            y, self.deploySQL()[idx], relation, self.cursor
                        )
                    ]
            else:
                return explained_variance(
                    self.y, prediction, test_relation, self.cursor
                )
        else:
            raise ParameterError(
                "The parameter 'method' must be in r2|mae|mse|msle|max|median|var"
            )
        return result.transpose()


# ---#
class Unsupervised(vModel):

    # ---#
    def fit(self, input_relation: Union[str, vDataFrame], X: list = []):
        """
	---------------------------------------------------------------------------
	Trains the model.

	Parameters
	----------
	input_relation: str/vDataFrame
		Training relation.
	X: list, optional
		List of the predictors. If empty, all the numerical columns will be used.

	Returns
	-------
	object
		model
		"""
        if isinstance(X, str):
            X = [X]
        check_types(
            [("input_relation", input_relation, [str, vDataFrame],), ("X", X, [list],)]
        )
        self.cursor = check_cursor(self.cursor, input_relation, True)[0]
        does_model_exist(name=self.name, cursor=self.cursor, raise_error=True)
        if isinstance(input_relation, str) and self.type == "MCA":
            input_relation = vdf_from_relation(input_relation, cursor=self.cursor)
        if isinstance(input_relation, vDataFrame):
            if self.type == "MCA":
                result = input_relation.sum(columns=X)
                if isinstance(result, (int, float)):
                    result = [result]
                else:
                    result = result["sum"]
                result = sum(result) + (input_relation.shape()[0] - 1) * len(result)
                assert abs(result) < 0.01, ConversionError("MCA can only work on a transformed complete disjunctive table. You should transform your relation first.\nTips: Use the vDataFrame.cdt method to transform the relation.")
            self.input_relation = input_relation.__genSQL__()
            schema = schema_relation(self.name)[0]
            relation = "{}._VERTICAPY_TEMPORARY_VIEW_{}".format(
                str_column(schema), get_session(self.cursor)
            )
            self.cursor.execute("DROP VIEW IF EXISTS {}".format(relation))
            self.cursor.execute(
                "CREATE VIEW {} AS SELECT * FROM {}".format(
                    relation, input_relation.__genSQL__()
                )
            )
            if not (X):
                X = input_relation.numcol()
        else:
            self.input_relation = input_relation
            relation = input_relation
            if not (X):
                X = vDataFrame(input_relation, self.cursor).numcol()
        self.X = [str_column(column) for column in X]
        parameters = vertica_param_dict(self)
        if "num_components" in parameters and not (parameters["num_components"]):
            del parameters["num_components"]
        fun = self.get_model_fun()[0] if self.type != "MCA" else "PCA"
        query = "SELECT {}('{}', '{}', '{}'".format(
            fun, self.name, relation, ", ".join(self.X)
        )
        if self.type in ("BisectingKMeans", "KMeans"):
            query += ", {}".format(parameters["n_cluster"])
        elif self.type == "Normalizer":
            query += ", {}".format(parameters["method"])
            del parameters["method"]
        if self.type not in ("Normalizer", "MCA",):
            query += " USING PARAMETERS "
        if (
            "init_method" in parameters
            and not (isinstance(parameters["init_method"], str))
            and self.type in ("KMeans", "BisectingKMeans")
        ):
            schema = schema_relation(self.name)[0]
            name = "VERTICAPY_KMEANS_INITIAL_{}".format(get_session(self.cursor))
            del parameters["init_method"]
            try:
                self.cursor.execute("DROP TABLE IF EXISTS {}.{}".format(schema, name))
            except:
                pass
            if len(self.parameters["init"]) != self.parameters["n_cluster"]:
                raise ParameterError(
                    "'init' must be a list of 'n_cluster' = {} points".format(
                        self.parameters["n_cluster"]
                    )
                )
            else:
                for item in self.parameters["init"]:
                    if len(X) != len(item):
                        raise ParameterError(
                            "Each points of 'init' must be of size len(X) = {}".format(
                                len(self.X)
                            )
                        )
                query0 = []
                for i in range(len(self.parameters["init"])):
                    line = []
                    for j in range(len(self.parameters["init"][0])):
                        line += [str(self.parameters["init"][i][j]) + " AS " + X[j]]
                    line = ",".join(line)
                    query0 += ["SELECT " + line]
                query0 = " UNION ".join(query0)
                query0 = "CREATE TABLE {}.{} AS {}".format(schema, name, query0)
                self.cursor.execute(query0)
                query += "initial_centers_table = '{}.{}', ".format(schema, name)
        elif "init_method" in parameters:
            del parameters["init_method"]
            query += "init_method = '{}', ".format(self.parameters["init"])
        query += ", ".join(
            ["{} = {}".format(elem, parameters[elem]) for elem in parameters]
        )
        query += ")"
        try:
            executeSQL(self.cursor, query, "Fitting the model.")
            if isinstance(input_relation, vDataFrame):
                self.cursor.execute("DROP VIEW {};".format(relation))
        except:
            if isinstance(input_relation, vDataFrame):
                self.cursor.execute("DROP VIEW {};".format(relation))
            raise
        if self.type == "KMeans":
            try:
                self.cursor.execute("DROP TABLE IF EXISTS {}.{}".format(schema, name))
            except:
                pass
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
                    result.split("Between-Cluster Sum of Squares: ")[1].split("\n")[0]
                ),
                float(result.split("Total Sum of Squares: ")[1].split("\n")[0]),
                float(
                    result.split("Total Within-Cluster Sum of Squares: ")[1].split(
                        "\n"
                    )[0]
                ),
                float(
                    result.split("Between-Cluster Sum of Squares: ")[1].split("\n")[0]
                )
                / float(result.split("Total Sum of Squares: ")[1].split("\n")[0]),
                result.split("Converged: ")[1].split("\n")[0] == "True",
            ]
            self.metrics_ = tablesample(values)
        elif self.type in ("BisectingKMeans"):
            self.metrics_ = self.get_attr("Metrics")
            self.cluster_centers_ = self.get_attr("BKTree")
        elif self.type in ("PCA", "MCA",):
            self.components_ = self.get_attr("principal_components")
            if self.type in ("MCA",):
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
        elif self.type in ("SVD"):
            self.singular_values_ = self.get_attr("right_singular_vectors")
            self.explained_variance_ = self.get_attr("singular_values")
        elif self.type in ("Normalizer"):
            self.param_ = self.get_attr("details")
        elif self.type == "OneHotEncoder":
            try:
                self.param_ = to_tablesample(
                    query="SELECT category_name, category_level::varchar, category_level_index FROM (SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'integer_categories')) VERTICAPY_SUBTABLE UNION ALL SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'varchar_categories')".format(
                        self.name, self.name
                    ),
                    cursor=self.cursor,
                    title="Getting Model Attributes.",
                )
            except:
                try:
                    self.param_ = to_tablesample(
                        query="SELECT category_name, category_level::varchar, category_level_index FROM (SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'integer_categories')) VERTICAPY_SUBTABLE".format(
                            self.name
                        ),
                        cursor=self.cursor,
                        title="Getting Model Attributes.",
                    )
                except:
                    self.param_ = self.get_attr("varchar_categories")
        return self


# ---#
class Preprocessing(Unsupervised):

    # ---#
    def deploySQL(
        self, key_columns: list = [], exclude_columns: list = [], X: list = [],
    ):
        """
    ---------------------------------------------------------------------------
    Returns the SQL code needed to deploy the model. 

    Parameters
    ----------
    key_columns: list, optional
        Predictors used during the algorithm computation which will be deployed
        with the principal components.
    exclude_columns: list, optional
        Columns to exclude from the prediction.
    X: list, optional
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
        check_types(
            [
                ("key_columns", key_columns, [list],),
                ("exclude_columns", exclude_columns, [list],),
                ("X", X, [list],),
            ]
        )
        X = [str_column(elem) for elem in X]
        fun = self.get_model_fun()[1]
        sql = "{}({} USING PARAMETERS model_name = '{}', match_by_pos = 'true'"
        if key_columns:
            sql += ", key_columns = '{}'".format(
                ", ".join([str_column(item) for item in key_columns])
            )
        if exclude_columns:
            sql += ", exclude_columns = '{}'".format(
                ", ".join([str_column(item) for item in exclude_columns])
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
            sql += ", drop_first = {}, ignore_null = {}, separator = {}, column_naming = '{}'".format(
                self.parameters["drop_first"],
                self.parameters["ignore_null"],
                separator,
                self.parameters["column_naming"],
            )
            if self.parameters["column_naming"].lower() in ("values", "values_relaxed"):
                sql += ", null_column_name = {}".format(null_column_name,)
        sql += ")"
        return sql.format(fun, ", ".join(self.X if not (X) else X), self.name)

    # ---#
    def deployInverseSQL(
        self, key_columns: list = [], exclude_columns: list = [], X: list = []
    ):
        """
    ---------------------------------------------------------------------------
    Returns the SQL code needed to deploy the inverse model. 

    Parameters
    ----------
    key_columns: list, optional
        Predictors used during the algorithm computation which will be deployed
        with the principal components.
    exclude_columns: list, optional
        Columns to exclude from the prediction.
    X: list, optional
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
        check_types([("key_columns", key_columns, [list],), ("X", X, [list],)])
        X = [str_column(elem) for elem in X]
        fun = self.get_model_fun()[2]
        sql = "{}({} USING PARAMETERS model_name = '{}', match_by_pos = 'true'"
        if key_columns:
            sql += ", key_columns = '{}'".format(
                ", ".join([str_column(item) for item in key_columns])
            )
        if exclude_columns:
            sql += ", exclude_columns = '{}'".format(
                ", ".join([str_column(item) for item in exclude_columns])
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
        X = [str_column(elem) for elem in X]
        if not (X):
            X = self.X
        if self.type in ("PCA", "SVD", "MCA",) and not (inverse):
            if self.type in ("PCA", "SVD",):
                n = self.parameters["n_components"]
                if not (n):
                    n = len(self.X)
            else:
                n = len(self.X)
            return [f"col{i}" for i in range(1, n + 1)]
        elif self.type in ("OneHotEncoder") and not (inverse):
            names = []
            for column in self.X:
                k = 0
                for i in range(len(self.param_["category_name"])):
                    if str_column(self.param_["category_name"][i]) == str_column(
                        column
                    ):
                        if (k != 0 or not (self.parameters["drop_first"])) and (
                            not (self.parameters["ignore_null"])
                            or self.param_["category_level"][i] != None
                        ):
                            if self.parameters["column_naming"] == "indices":
                                names += [
                                    '"'
                                    + str_column(column)[1:-1]
                                    + "{}{}".format(
                                        self.parameters["separator"],
                                        self.param_["category_level_index"][i],
                                    )
                                    + '"'
                                ]
                            else:
                                names += [
                                    '"'
                                    + str_column(column)[1:-1]
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
    def inverse_transform(
        self, vdf: Union[str, vDataFrame], X: list = [],
    ):
        """
    ---------------------------------------------------------------------------
    Applies the Inverse Model on a vDataFrame.

    Parameters
    ----------
    vdf: str/vDataFrame
        input vDataFrame. You can also specify a customized relation, 
        but you must enclose it with an alias. For example "(SELECT 1) x" is 
        correct whereas "(SELECT 1)" and "SELECT 1" are incorrect.
    X: list, optional
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
        check_types([("X", X, [list],)])
        if not (vdf):
            vdf = self.input_relation
        if not (X):
            X = self.get_names()
        check_types([("vdf", vdf, [str, vDataFrame],),],)
        if isinstance(vdf, str):
            vdf = vdf_from_relation(relation=vdf, cursor=self.cursor)
        X = vdf_columns_names(X, vdf)
        relation = vdf.__genSQL__()
        exclude_columns = vdf.get_columns(exclude_columns=X)
        all_columns = vdf.get_columns()
        main_relation = "(SELECT {} FROM {}) VERTICAPY_SUBTABLE".format(
            self.deployInverseSQL(exclude_columns, exclude_columns, all_columns),
            relation,
        )
        return vdf_from_relation(main_relation, "Inverse Transformation", self.cursor,)

    # ---#
    def transform(
        self, vdf: Union[str, vDataFrame] = None, X: list = [],
    ):
        """
    ---------------------------------------------------------------------------
    Applies the model on a vDataFrame.

    Parameters
    ----------
    vdf: str/vDataFrame, optional
        Input vDataFrame. You can also specify a customized relation, 
        but you must enclose it with an alias. For example "(SELECT 1) x" is 
        correct whereas "(SELECT 1)" and "SELECT 1" are incorrect.
    X: list, optional
        List of the input vcolumns.

    Returns
    -------
    vDataFrame
        object result of the model transformation.
        """
        if isinstance(X, str):
            X = [X]
        check_types(
            [("X", X, [list],),]
        )
        if not (vdf):
            vdf = self.input_relation
        if not (X):
            X = self.X
        check_types([("vdf", vdf, [str, vDataFrame],),],)
        if isinstance(vdf, str):
            vdf = vdf_from_relation(relation=vdf, cursor=self.cursor)
        columns_check(X, vdf)
        X = vdf_columns_names(X, vdf)
        relation = vdf.__genSQL__()
        exclude_columns = vdf.get_columns(exclude_columns=X)
        all_columns = vdf.get_columns()
        main_relation = "(SELECT {} FROM {}) VERTICAPY_SUBTABLE".format(
            self.deploySQL(exclude_columns, exclude_columns, all_columns), relation
        )
        return vdf_from_relation(main_relation, "Inverse Transformation", self.cursor,)


# ---#
class Decomposition(Preprocessing):

    # ---#
    def deploySQL(
        self,
        n_components: int = 0,
        cutoff: float = 1,
        key_columns: list = [],
        exclude_columns: list = [],
        X: list = [],
    ):
        """
    ---------------------------------------------------------------------------
    Returns the SQL code needed to deploy the model. 

    Parameters
    ----------
    n_components: int, optional
        Number of components to return. If set to 0, all the components will be
        deployed.
    cutoff: float, optional
        Specifies the minimum accumulated explained variance. Components are taken 
        until the accumulated explained variance reaches this value.
    key_columns: list, optional
        Predictors used during the algorithm computation which will be deployed
        with the principal components.
    exclude_columns: list, optional
        Columns to exclude from the prediction.
    X: list, optional
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
        check_types(
            [
                ("n_components", n_components, [int, float],),
                ("cutoff", cutoff, [int, float],),
                ("key_columns", key_columns, [list],),
                ("exclude_columns", exclude_columns, [list],),
                ("X", X, [list],),
            ]
        )
        X = [str_column(elem) for elem in X]
        fun = self.get_model_fun()[1]
        sql = "{}({} USING PARAMETERS model_name = '{}', match_by_pos = 'true'"
        if key_columns:
            sql += ", key_columns = '{}'".format(
                ", ".join([str_column(item) for item in key_columns])
            )
        if exclude_columns:
            sql += ", exclude_columns = '{}'".format(
                ", ".join([str_column(item) for item in exclude_columns])
            )
        if n_components:
            sql += ", num_components = {}".format(n_components)
        else:
            sql += ", cutoff = {}".format(cutoff)
        sql += ")"
        return sql.format(fun, ", ".join(self.X if not (X) else X), self.name)

    # ---#
    def plot(
        self, dimensions: tuple = (1, 2), ax=None, **style_kwds,
    ):
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
        check_types([("dimensions", dimensions, [tuple],),])
        vdf = vdf_from_relation(self.input_relation, cursor=self.cursor)
        ax = self.transform(vdf).scatter(columns=["col{}".format(dimensions[0]), "col{}".format(dimensions[1])], max_nb_points=100000, ax=ax, **style_kwds)
        explained_variance = self.explained_variance_["explained_variance"]
        ax.set_xlabel("Dim{} {}".format(dimensions[0], "" if not(explained_variance[dimensions[0] - 1]) else "({}%)".format(round(explained_variance[dimensions[0] - 1] * 100, 1))))
        ax.set_ylabel("Dim{} {}".format(dimensions[1], "" if not(explained_variance[dimensions[1] - 1]) else "({}%)".format(round(explained_variance[dimensions[1] - 1] * 100, 1))))
        return ax

    # ---#
    def plot_circle(
        self, dimensions: tuple = (1, 2), ax=None, **style_kwds
    ):
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
        check_types([("dimensions", dimensions, [tuple],),])
        if self.type == "SVD":
            x = self.singular_values_["vector{}".format(dimensions[0])]
            y = self.singular_values_["vector{}".format(dimensions[1])]
        else:
            x = self.components_["PC{}".format(dimensions[0])]
            y = self.components_["PC{}".format(dimensions[1])]
        explained_variance = self.explained_variance_["explained_variance"]
        return plot_pca_circle(x, y, self.X, (explained_variance[dimensions[0] - 1], explained_variance[dimensions[1] - 1]), dimensions, ax, **style_kwds,)

    # ---#
    def plot_scree(
        self, ax=None, **style_kwds
    ):
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
        explained_variance, n = [100 * elem for elem in explained_variance], len(explained_variance)
        information = tablesample({"dimensions": [i + 1 for i in range(n)], "percentage_explained_variance": explained_variance}).to_vdf(self.cursor)
        information["dimensions_center"] = information["dimensions"] + 0.5
        ax = information["dimensions"].hist(method="avg", of="percentage_explained_variance", h=1, max_cardinality=1, ax=ax, **style_kwds)
        ax = information["percentage_explained_variance"].plot(ts="dimensions_center", ax=ax, color="black")
        ax.set_xlim(1, n + 1)
        ax.set_xticks([i + 1.5 for i in range(n)],)
        ax.set_xticklabels([i + 1 for i in range(n)],)
        ax.set_ylabel('"percentage_explained_variance"')
        ax.set_xlabel('"dimensions"')
        for i in range(n):
            ax.text(i + 1.5, explained_variance[i] + 1, "{}%".format(round(explained_variance[i], 1)))
        return ax

    # ---#
    def score(
        self, X: list = [], input_relation: str = "", method: str = "avg", p: int = 2
    ):
        """
    ---------------------------------------------------------------------------
    Returns the decomposition score on a dataset for each transformed column. It
    is the average / median of the p-distance between the real column and its 
    result after applying the decomposition model and its inverse.  

    Parameters
    ----------
    X: list, optional
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
        if isinstance(X, str):
            X = [X]
        check_types(
            [
                ("X", X, [list],),
                ("input_relation", input_relation, [str],),
                ("method", str(method).lower(), ["avg", "median"],),
                ("p", p, [int, float],),
            ]
        )
        fun = self.get_model_fun()
        if not (X):
            X = self.X
        if not (input_relation):
            input_relation = self.input_relation
        method = str(method).upper()
        if method == "MEDIAN":
            method = "APPROXIMATE_MEDIAN"
        if self.type in ("PCA", "SVD",):
            n_components = self.parameters["n_components"]
            if not (n_components):
                n_components = len(X)
        else:
            n_components = len(X)
        col_init_1 = ["{} AS col_init{}".format(X[idx], idx) for idx in range(len(X))]
        col_init_2 = ["col_init{}".format(idx) for idx in range(len(X))]
        cols = ["col{}".format(idx + 1) for idx in range(n_components)]
        query = "SELECT {}({} USING PARAMETERS model_name = '{}', key_columns = '{}', num_components = {}) OVER () FROM {}".format(
            fun[1],
            ", ".join(self.X),
            self.name,
            ", ".join(self.X),
            n_components,
            input_relation,
        )
        query = "SELECT {} FROM ({}) VERTICAPY_SUBTABLE".format(
            ", ".join(col_init_1 + cols), query
        )
        query = "SELECT {}({} USING PARAMETERS model_name = '{}', key_columns = '{}', exclude_columns = '{}', num_components = {}) OVER () FROM ({}) y".format(
            fun[2],
            ", ".join(col_init_2 + cols),
            self.name,
            ", ".join(col_init_2),
            ", ".join(col_init_2),
            n_components,
            query,
        )
        query = "SELECT 'Score' AS 'index', {} FROM ({}) z".format(
            ", ".join(
                [
                    "{}(POWER(ABS(POWER({}, {}) - POWER({}, {})), {})) AS {}".format(
                        method,
                        X[idx],
                        p,
                        "col_init{}".format(idx),
                        p,
                        float(1 / p),
                        X[idx],
                    )
                    for idx in range(len(X))
                ]
            ),
            query,
        )
        result = to_tablesample(
            query, cursor=self.cursor, title="Getting Model Score.",
        ).transpose()
        return result

    # ---#
    def transform(
        self,
        vdf: Union[str, vDataFrame] = None,
        X: list = [],
        n_components: int = 0,
        cutoff: float = 1,
    ):
        """
    ---------------------------------------------------------------------------
    Applies the model on a vDataFrame.

    Parameters
    ----------
    vdf: str/vDataFrame, optional
        Input vDataFrame. You can also specify a customized relation, 
        but you must enclose it with an alias. For example "(SELECT 1) x" is 
        correct whereas "(SELECT 1)" and "SELECT 1" are incorrect.
    X: list, optional
        List of the input vcolumns.
    n_components: int, optional
        Number of components to return. If set to 0, all the components will 
        be deployed.
    cutoff: float, optional
        Specifies the minimum accumulated explained variance. Components are 
        taken until the accumulated explained variance reaches this value.

    Returns
    -------
    vDataFrame
        object result of the model transformation.
        """
        if isinstance(X, str):
            X = [X]
        check_types(
            [
                ("n_components", n_components, [int, float],),
                ("cutoff", cutoff, [int, float],),
                ("X", X, [list],),
            ]
        )
        if not (vdf):
            vdf = self.input_relation
        if not (X):
            X = self.X
        check_types([("vdf", vdf, [str, vDataFrame],),],)
        if isinstance(vdf, str):
            vdf = vdf_from_relation(relation=vdf, cursor=self.cursor)
        columns_check(X, vdf)
        X = vdf_columns_names(X, vdf)
        relation = vdf.__genSQL__()
        exclude_columns = vdf.get_columns(exclude_columns=X)
        all_columns = vdf.get_columns()
        main_relation = "(SELECT {} FROM {}) VERTICAPY_SUBTABLE".format(
            self.deploySQL(
                n_components, cutoff, exclude_columns, exclude_columns, all_columns
            ),
            relation,
        )
        return vdf_from_relation(main_relation, "Inverse Transformation", self.cursor,)


# ---#
class Clustering(Unsupervised):

    # ---#
    def predict(
        self, vdf: Union[str, vDataFrame], X: list = [], name: str = "", inplace: bool = True,
    ):
        """
	---------------------------------------------------------------------------
	Predicts using the input relation.

	Parameters
	----------
	vdf: str/vDataFrame
		Object to use to run the prediction. You can also specify a customized 
        relation, but you must enclose it with an alias. For example "(SELECT 1) x" 
        is correct whereas "(SELECT 1)" and "SELECT 1" are incorrect.
	X: list, optional
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
        check_types(
            [
                ("name", name, [str],),
                ("X", X, [list],),
                ("vdf", vdf, [str, vDataFrame],),
            ],
        )
        if isinstance(vdf, str):
            vdf = vdf_from_relation(relation=vdf, cursor=self.cursor)
        X = [str_column(elem) for elem in X]
        name = (
            "{}_".format(self.type) + "".join(ch for ch in self.name if ch.isalnum())
            if not (name)
            else name
        )
        if inplace:
            return vdf.eval(name, self.deploySQL(X=X))
        else:
            return vdf.copy().eval(name, self.deploySQL(X=X))
