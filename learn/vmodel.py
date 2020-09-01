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
# Standard Python Modules
import os

# VerticaPy Modules
from verticapy import vDataFrame
from verticapy.learn.plot import *
from verticapy.utilities import *
from verticapy.toolbox import *
from verticapy.connections.connect import read_auto_connect
from verticapy.errors import *
from verticapy.learn.metrics import *

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
            if self.type not in ("DBSCAN", "NearestCentroid"):
                try:
                    version(cursor=self.cursor, condition=[9, 0, 0])
                    self.cursor.execute(
                        "SELECT GET_MODEL_SUMMARY(USING PARAMETERS model_name = '{}')".format(
                            self.name
                        )
                    )
                except:
                    self.cursor.execute(
                        "SELECT SUMMARIZE_MODEL('{}')".format(self.name)
                    )
                return self.cursor.fetchone()[0]
            elif self.type == "DBSCAN":
                rep = "<DBSCAN>\nNumber of Clusters: {}\nNumber of Outliers: {}".format(
                    self.n_cluster_, self.n_noise_
                )
                return rep
            elif self.type == "NearestCentroid":
                rep = "<NearestCentroid>" + self.centroids_.__repr__()
                return rep
            else:
                raise
        except:
            return "<{}>".format(self.type)

    # ---#
    def deploySQL(self, X: list = []):
        """
	---------------------------------------------------------------------------
	Returns the SQL code needed to deploy the model. 

	Parameters
	----------
	X: list, optional
		List of the columns used to deploy the self. If empty, the model
		predictors will be used.

	Returns
	-------
	str
		the SQL code needed to deploy the model.
		"""
        if self.type not in ("DBSCAN"):
            check_types([("X", X, [list], False)])
            X = [str_column(elem) for elem in X]
            fun = self.get_model_fun()[1]
            sql = "{}({} USING PARAMETERS model_name = '{}', match_by_pos = 'true')"
            return sql.format(fun, ", ".join(self.X if not (X) else X), self.name)
        else:
            raise FunctionError(
                "Method 'deploySQL' for '{}' doesn't exist.".format(self.type)
            )

    # ---#
    def drop(self):
        """
	---------------------------------------------------------------------------
	Drops the model from the Vertica DB.
		"""
        if self.type not in (
            "DBSCAN",
            "NearestCentroid",
            "KNeighborsClassifier",
            "KNeighborsRegressor",
            "LocalOutlierFactor",
        ):
            drop_model(self.name, self.cursor, print_info=False)
        elif self.type in (
            "NearestCentroid",
            "KNeighborsClassifier",
            "KNeighborsRegressor",
            "DBSCAN",
            "LocalOutlierFactor",
        ):
            try:
                path = os.path.dirname(
                    verticapy.__file__
                ) + "/learn/models/{}.verticapy".format(self.name)
                file = open(path, "r")
                os.remove(path)
            except:
                pass
            if self.type in ("DBSCAN", "LocalOutlierFactor"):
                drop_table(self.name, self.cursor, print_info=False)
            elif self.type in ("CountVectorizer"):
                drop_text_index(self.name, self.cursor, print_info=False)

    # ---#
    def features_importance(self):
        """
		---------------------------------------------------------------------------
		Computes the model features importance.

		Returns
		-------
		tablesample
			An object containing the result. For more information, see
			utilities.tablesample.
		"""
        if self.type in ("RandomForestClassifier", "RandomForestRegressor"):
            version(cursor=self.cursor, condition=[9, 1, 1])
            query = "SELECT predictor_name AS predictor, ROUND(100 * importance_value / SUM(importance_value) OVER (), 2) AS importance, SIGN(importance_value) AS sign FROM (SELECT RF_PREDICTOR_IMPORTANCE ( USING PARAMETERS model_name = '{}')) x ORDER BY 2 DESC;".format(
                self.name
            )
            print_legend = False
        elif self.type in (
            "LinearRegression",
            "LogisticRegression",
            "LinearSVC",
            "LinearSVR",
        ):
            version(cursor=self.cursor, condition=[8, 1, 1])
            query = "SELECT predictor, ROUND(100 * importance / SUM(importance) OVER(), 2) AS importance, sign FROM "
            query += "(SELECT stat.predictor AS predictor, ABS(coefficient * (max - min)) AS importance, SIGN(coefficient) AS sign FROM "
            query += '(SELECT LOWER("column") AS predictor, min, max FROM (SELECT SUMMARIZE_NUMCOL({}) OVER() '.format(
                ", ".join(self.X)
            )
            query += " FROM {}) x) stat NATURAL JOIN (SELECT GET_MODEL_ATTRIBUTE (USING PARAMETERS model_name = '{}', ".format(
                self.input_relation, self.name
            )
            query += "attr_name = 'details')) coeff) importance_t ORDER BY 2 DESC;"
            print_legend = True
        else:
            raise FunctionError(
                "Method 'features_importance' for '{}' doesn't exist.".format(self.type)
            )
        self.cursor.execute(query)
        result = self.cursor.fetchall()
        coeff_importances, coeff_sign = {}, {}
        for elem in result:
            coeff_importances[elem[0]] = elem[1]
            coeff_sign[elem[0]] = elem[2]
        try:
            plot_importance(coeff_importances, coeff_sign, print_legend=print_legend)
        except:
            pass
        importances = {"index": ["importance"]}
        for elem in coeff_importances:
            importances[elem] = [coeff_importances[elem]]
        return tablesample(values=importances, table_info=False).transpose()

    # ---#
    def get_model_attribute(self, attr_name: str = ""):
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
        if self.type not in ("DBSCAN", "LocalOutlierFactor"):
            version(cursor=self.cursor, condition=[8, 1, 1])
            result = to_tablesample(
                query="SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}'{})".format(
                    self.name,
                    ", attr_name = '{}'".format(attr_name) if attr_name else "",
                ),
                cursor=self.cursor,
            )
            result.table_info = False
            return result
        elif self.type in ("DBSCAN"):
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
                    name="Attributes",
                    table_info=False,
                )
                return result
            else:
                raise ParameterError("Attribute '' doesn't exist.".format(attr_name))
        elif self.type in ("LocalOutlierFactor"):
            if attr_name == "n_errors":
                return self.n_errors_
            elif not (attr_name):
                result = tablesample(
                    values={"attr_name": ["n_errors"], "value": [self.n_errors_]},
                    name="Attributes",
                    table_info=False,
                )
                return result
            else:
                raise ParameterError("Attribute '' doesn't exist.".format(attr_name))
        else:
            raise FunctionError(
                "Method 'get_model_attribute' for '{}' doesn't exist.".format(self.type)
            )

    # ---#
    def get_model_fun(self):
        """
	---------------------------------------------------------------------------
	Returns the Vertica associated functions.

	Returns
	-------
	tuple
		(FIT, PREDICT, INVERSE)
		"""
        if self.type == "LinearRegression":
            return ("LINEAR_REG", "PREDICT_LINEAR_REG", "")
        elif self.type == "LogisticRegression":
            return ("LOGISTIC_REG", "PREDICT_LOGISTIC_REG", "")
        elif self.type == "LinearSVC":
            return ("SVM_CLASSIFIER", "PREDICT_SVM_CLASSIFIER", "")
        elif self.type == "LinearSVR":
            return ("SVM_REGRESSOR", "PREDICT_SVM_REGRESSOR", "")
        elif self.type == "RandomForestRegressor":
            return ("RF_REGRESSOR", "PREDICT_RF_REGRESSOR", "")
        elif self.type == "RandomForestClassifier":
            return ("RF_CLASSIFIER", "PREDICT_RF_CLASSIFIER", "")
        elif self.type == "MultinomialNB":
            return ("NAIVE_BAYES", "PREDICT_NAIVE_BAYES", "")
        elif self.type == "KMeans":
            return ("KMEANS", "APPLY_KMEANS", "")
        elif self.type == "BisectingKMeans":
            return ("BISECTING_KMEANS", "APPLY_BISECTING_KMEANS", "")
        elif self.type == "PCA":
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
	Returns the model Parameters.

	Returns
	-------
	dict
		model parameters
		"""
        return self.parameters

    # ---#
    def plot(self, max_nb_points: int = 100):
        """
	---------------------------------------------------------------------------
	Draws the Model.

	Parameters
	----------
	max_nb_points: int
		Maximum number of points to display.

	Returns
	-------
	Figure
		Matplotlib Figure
		"""
        check_types([("max_nb_points", max_nb_points, [int, float], False)])
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
                )
            elif self.type == "LinearSVC":
                return svm_classifier_plot(
                    self.X,
                    self.y,
                    self.input_relation,
                    coefficients,
                    self.cursor,
                    max_nb_points,
                )
            else:
                return regression_plot(
                    self.X,
                    self.y,
                    self.input_relation,
                    coefficients,
                    self.cursor,
                    max_nb_points,
                )
        elif self.type in ("KMeans", "BisectingKMeans", "DBSCAN"):
            if self.type != "DBSCAN":
                vdf = vDataFrame(self.input_relation, self.cursor)
                self.predict(vdf, name="kmeans_cluster")
                catcol = "kmeans_cluster"
            else:
                vdf = vDataFrame(self.name, self.cursor)
                catcol = "dbscan_cluster"
            if 2 <= len(self.X) <= 3:
                return vdf.scatter(
                    columns=self.X,
                    catcol=catcol,
                    max_cardinality=100,
                    max_nb_points=max_nb_points,
                )
            else:
                raise Exception("Clustering Plots are only available in 2D or 3D")
        elif self.type in ("PCA", "SVD"):
            if 2 <= self.parameters["n_components"] or (
                self.parameters["n_components"] <= 0 and len(self.X) > 1
            ):
                X = [
                    "col{}".format(i + 1)
                    for i in range(min(max(self.parameters["n_components"], 2), 3))
                ]
                return self.transform().scatter(columns=X, max_nb_points=max_nb_points,)
            else:
                raise Exception("Decomposition Plots are not available in 1D")
        elif self.type in ("LocalOutlierFactor"):
            query = "SELECT COUNT(*) FROM {}".format(self.name)
            tablesample = 100 * min(
                float(max_nb_points / self.cursor.execute(query).fetchone()[0]), 1
            )
            return lof_plot(self.name, self.X, "lof_score", self.cursor, 100)
        else:
            raise FunctionError(
                "Method 'plot' for '{}' doesn't exist.".format(self.type)
            )

    # ---#
    def set_cursor(self, cursor):
        """
	---------------------------------------------------------------------------
	Sets a new DB cursor. It can be very usefull if the connection to the DB is 
	lost.

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
        model_parameters = {}
        default_parameters = default_model_parameters(self.type)
        if self.type in ("LinearRegression", "LogisticRegression"):
            if "solver" in parameters:
                check_types([("solver", parameters["solver"], [str], False)])
                assert str(parameters["solver"]).lower() in [
                    "newton",
                    "bfgs",
                    "cgd",
                ], "Incorrect parameter 'solver'.\nThe optimizer must be in (Newton | BFGC | CGD), found '{}'.".format(
                    parameters["solver"]
                )
                model_parameters["solver"] = parameters["solver"]
            else:
                model_parameters["solver"] = default_parameters["solver"]
            if "penalty" in parameters:
                check_types([("penalty", parameters["penalty"], [str], False)])
                assert str(parameters["penalty"]).lower() in [
                    "none",
                    "l1",
                    "l2",
                    "enet",
                ], "Incorrect parameter 'penalty'.\nThe regularization must be in (None | L1 | L2 | ENet), found '{}'.".format(
                    parameters["penalty"]
                )
                model_parameters["penalty"] = parameters["penalty"]
            else:
                model_parameters["penalty"] = default_parameters["penalty"]
            if "max_iter" in parameters:
                check_types([("max_iter", parameters["max_iter"], [int, float], False)])
                assert (
                    0 <= parameters["max_iter"]
                ), "Incorrect parameter 'max_iter'.\nThe maximum number of iterations must be positive."
                model_parameters["max_iter"] = parameters["max_iter"]
            else:
                model_parameters["max_iter"] = default_parameters["max_iter"]
            if "l1_ratio" in parameters:
                check_types([("l1_ratio", parameters["l1_ratio"], [int, float], False)])
                assert (
                    0 <= parameters["l1_ratio"] <= 1
                ), "Incorrect parameter 'l1_ratio'.\nThe ENet Mixture must be between 0 and 1."
                model_parameters["l1_ratio"] = parameters["l1_ratio"]
            else:
                model_parameters["l1_ratio"] = default_parameters["l1_ratio"]
            if "C" in parameters:
                check_types([("C", parameters["C"], [int, float], False)])
                assert (
                    0 <= parameters["C"]
                ), "Incorrect parameter 'C'.\nThe regularization parameter value must be positive."
                model_parameters["C"] = parameters["C"]
            else:
                model_parameters["C"] = default_parameters["C"]
            if "tol" in parameters:
                check_types([("tol", parameters["tol"], [int, float], False)])
                assert (
                    0 <= parameters["tol"]
                ), "Incorrect parameter 'tol'.\nThe tolerance parameter value must be positive."
                model_parameters["tol"] = parameters["tol"]
            else:
                model_parameters["tol"] = default_parameters["tol"]
        elif self.type in ("RandomForestClassifier", "RandomForestRegressor"):
            if "n_estimators" in parameters:
                check_types(
                    [("n_estimators", parameters["n_estimators"], [int], False)]
                )
                assert (
                    0 <= parameters["n_estimators"] <= 1000
                ), "Incorrect parameter 'n_estimators'.\nThe number of trees must be lesser than 1000."
                model_parameters["n_estimators"] = parameters["n_estimators"]
            else:
                model_parameters["n_estimators"] = default_parameters["n_estimators"]
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
                if type(parameters["max_features"]) == str:
                    assert str(parameters["max_features"]).lower() in [
                        "max",
                        "auto",
                    ], "Incorrect parameter 'init'.\nThe maximum number of features to test must be in (max | auto) or an integer, found '{}'.".format(
                        parameters["max_features"]
                    )
                model_parameters["max_features"] = parameters["max_features"]
            else:
                model_parameters["max_features"] = default_parameters["max_features"]
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
                assert (
                    1 <= parameters["max_leaf_nodes"] <= 1e9
                ), "Incorrect parameter 'max_leaf_nodes'.\nThe maximum number of leaf nodes must be between 1 and 1e9, inclusive."
                model_parameters["max_leaf_nodes"] = parameters["max_leaf_nodes"]
            else:
                model_parameters["max_leaf_nodes"] = default_parameters[
                    "max_leaf_nodes"
                ]
            if "sample" in parameters:
                check_types([("sample", parameters["sample"], [int, float], False)])
                assert (
                    0 <= parameters["sample"] <= 1
                ), "Incorrect parameter 'sample'.\nThe portion of the input data set that is randomly picked for training each tree must be between 0.0 and 1.0, inclusive."
                model_parameters["sample"] = parameters["sample"]
            else:
                model_parameters["sample"] = default_parameters["sample"]
            if "max_depth" in parameters:
                check_types([("max_depth", parameters["max_depth"], [int], False)])
                assert (
                    1 <= parameters["max_depth"] <= 100
                ), "Incorrect parameter 'max_depth'.\nThe maximum depth for growing each tree must be between 1 and 100, inclusive."
                model_parameters["max_depth"] = parameters["max_depth"]
            else:
                model_parameters["max_depth"] = default_parameters["max_depth"]
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
                assert (
                    1 <= parameters["min_samples_leaf"] <= 1e6
                ), "Incorrect parameter 'min_samples_leaf'.\nThe minimum number of samples each branch must have after splitting a node must be between 1 and 1e6, inclusive."
                model_parameters["min_samples_leaf"] = parameters["min_samples_leaf"]
            else:
                model_parameters["min_samples_leaf"] = default_parameters[
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
                assert (
                    0 <= parameters["min_info_gain"] <= 1
                ), "Incorrect parameter 'min_info_gain'.\nThe minimum threshold for including a split must be between 0.0 and 1.0, inclusive."
                model_parameters["min_info_gain"] = parameters["min_info_gain"]
            else:
                model_parameters["min_info_gain"] = default_parameters["min_info_gain"]
            if "nbins" in parameters:
                check_types([("nbins", parameters["nbins"], [int, float], False)])
                assert (
                    2 <= parameters["nbins"] <= 1000
                ), "Incorrect parameter 'min_info_gain'.\nThe number of bins to use for continuous features must be between 2 and 1000, inclusive."
                model_parameters["nbins"] = parameters["nbins"]
            else:
                model_parameters["nbins"] = default_parameters["nbins"]
        elif self.type in ("MultinomialNB"):
            if "alpha" in parameters:
                check_types([("alpha", parameters["alpha"], [int, float], False)])
                assert (
                    0 <= parameters["alpha"]
                ), "Incorrect parameter 'alpha'.\nThe smoothing factor must be positive."
                model_parameters["alpha"] = parameters["alpha"]
            else:
                model_parameters["alpha"] = default_parameters["alpha"]
        elif self.type in ("KMeans", "BisectingKMeans"):
            if "max_iter" in parameters:
                check_types([("max_iter", parameters["max_iter"], [int, float], False)])
                assert (
                    0 <= parameters["max_iter"]
                ), "Incorrect parameter 'max_iter'.\nThe maximum number of iterations must be positive."
                model_parameters["max_iter"] = parameters["max_iter"]
            else:
                model_parameters["max_iter"] = default_parameters["max_iter"]
            if "tol" in parameters:
                check_types([("tol", parameters["tol"], [int, float], False)])
                assert (
                    0 <= parameters["tol"]
                ), "Incorrect parameter 'tol'.\nThe tolerance parameter value must be positive."
                model_parameters["tol"] = parameters["tol"]
            else:
                model_parameters["tol"] = default_parameters["tol"]
            if "n_cluster" in parameters:
                check_types(
                    [("n_cluster", parameters["n_cluster"], [int, float], False)]
                )
                assert (
                    1 <= parameters["n_cluster"] <= 10000
                ), "Incorrect parameter 'n_cluster'.\nThe number of clusters must be between 1 and 10000, inclusive."
                model_parameters["n_cluster"] = parameters["n_cluster"]
            else:
                model_parameters["n_cluster"] = default_parameters["n_cluster"]
            if "init" in parameters:
                check_types([("init", parameters["init"], [str, list], False)])
                if type(parameters["init"]) == str:
                    assert str(parameters["init"]).lower() in [
                        "random",
                        "kmeanspp",
                    ], "Incorrect parameter 'init'.\nThe initialization method of the clusters must be in (random | kmeanspp) or a list of the initial clusters position, found '{}'.".format(
                        parameters["init"]
                    )
                model_parameters["init"] = parameters["init"]
            else:
                model_parameters["init"] = default_parameters["init"]
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
                ), "Incorrect parameter 'bisection_iterations'.\nThe number of iterations the bisecting k-means algorithm performs for each bisection step must be between 1 and 1e6, inclusive."
                model_parameters["bisection_iterations"] = parameters[
                    "bisection_iterations"
                ]
            elif self.type == "BisectingKMeans":
                model_parameters["bisection_iterations"] = default_parameters[
                    "bisection_iterations"
                ]
            if "split_method" in parameters:
                check_types(
                    [("split_method", parameters["split_method"], [str], False)]
                )
                assert str(parameters["split_method"]).lower() in [
                    "size",
                    "sum_squares",
                ], "Incorrect parameter 'split_method'.\nThe split method must be in (size | sum_squares), found '{}'.".format(
                    parameters["split_method"]
                )
                model_parameters["split_method"] = parameters["split_method"]
            elif self.type == "BisectingKMeans":
                model_parameters["split_method"] = default_parameters["split_method"]
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
                assert (
                    2 <= parameters["min_divisible_cluster_size"]
                ), "Incorrect parameter 'min_divisible_cluster_size'.\nThe minimum number of points of a divisible cluster must be greater than or equal to 2."
                model_parameters["min_divisible_cluster_size"] = parameters[
                    "min_divisible_cluster_size"
                ]
            elif self.type == "BisectingKMeans":
                model_parameters["min_divisible_cluster_size"] = default_parameters[
                    "min_divisible_cluster_size"
                ]
            if "distance_method" in parameters:
                check_types(
                    [("distance_method", parameters["distance_method"], [str], False)]
                )
                assert str(parameters["distance_method"]).lower() in [
                    "euclidean"
                ], "Incorrect parameter 'distance_method'.\nThe distance method must be in (euclidean), found '{}'.".format(
                    parameters["distance_method"]
                )
                model_parameters["distance_method"] = parameters["distance_method"]
            elif self.type == "BisectingKMeans":
                model_parameters["distance_method"] = default_parameters[
                    "distance_method"
                ]
        elif self.type in ("LinearSVC", "LinearSVR"):
            if "tol" in parameters:
                check_types([("tol", parameters["tol"], [int, float], False)])
                assert (
                    0 <= parameters["tol"]
                ), "Incorrect parameter 'tol'.\nThe tolerance parameter value must be positive."
                model_parameters["tol"] = parameters["tol"]
            else:
                model_parameters["tol"] = default_parameters["tol"]
            if "C" in parameters:
                check_types([("C", parameters["C"], [int, float], False)])
                assert (
                    0 <= parameters["C"]
                ), "Incorrect parameter 'C'.\nThe weight for misclassification cost must be positive."
                model_parameters["C"] = parameters["C"]
            else:
                model_parameters["C"] = default_parameters["C"]
            if "max_iter" in parameters:
                check_types([("max_iter", parameters["max_iter"], [int, float], False)])
                assert (
                    0 <= parameters["max_iter"]
                ), "Incorrect parameter 'max_iter'.\nThe maximum number of iterations must be positive."
                model_parameters["max_iter"] = parameters["max_iter"]
            else:
                model_parameters["max_iter"] = default_parameters["max_iter"]
            if "fit_intercept" in parameters:
                check_types(
                    [("fit_intercept", parameters["fit_intercept"], [bool], False)]
                )
                model_parameters["fit_intercept"] = parameters["fit_intercept"]
            else:
                model_parameters["fit_intercept"] = default_parameters["fit_intercept"]
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
                assert (
                    0 <= parameters["intercept_scaling"]
                ), "Incorrect parameter 'intercept_scaling'.\nThe Intercept Scaling parameter value must be positive."
                model_parameters["intercept_scaling"] = parameters["intercept_scaling"]
            else:
                model_parameters["intercept_scaling"] = default_parameters[
                    "intercept_scaling"
                ]
            if "intercept_mode" in parameters:
                check_types(
                    [("intercept_mode", parameters["intercept_mode"], [str], False)]
                )
                assert str(parameters["intercept_mode"]).lower() in [
                    "regularized",
                    "unregularized",
                ], "Incorrect parameter 'intercept_mode'.\nThe Intercept Mode must be in (size | sum_squares), found '{}'.".format(
                    parameters["intercept_mode"]
                )
                model_parameters["intercept_mode"] = parameters["intercept_mode"]
            else:
                model_parameters["intercept_mode"] = default_parameters[
                    "intercept_mode"
                ]
            if ("class_weight" in parameters) and self.type in ("LinearSVC"):
                check_types(
                    [("class_weight", parameters["class_weight"], [list, tuple], False)]
                )
                model_parameters["class_weight"] = parameters["class_weight"]
            elif self.type in ("LinearSVC"):
                model_parameters["class_weight"] = default_parameters["class_weight"]
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
                assert (
                    0 <= parameters["acceptable_error_margin"]
                ), "Incorrect parameter 'acceptable_error_margin'.\nThe Acceptable Error Margin parameter value must be positive."
                model_parameters["acceptable_error_margin"] = parameters[
                    "acceptable_error_margin"
                ]
            elif self.type in ("LinearSVR"):
                model_parameters["acceptable_error_margin"] = default_parameters[
                    "acceptable_error_margin"
                ]
        elif self.type in ("PCA", "SVD"):
            if ("scale" in parameters) and self.type in ("PCA"):
                check_types([("scale", parameters["scale"], [bool], False)])
                model_parameters["scale"] = parameters["scale"]
            elif self.type in ("PCA"):
                model_parameters["scale"] = default_parameters["scale"]
            if "method" in parameters:
                check_types([("method", parameters["method"], [str], False)])
                assert str(parameters["method"]).lower() in [
                    "lapack"
                ], "Incorrect parameter 'method'.\nThe decomposition method must be in (lapack), found '{}'.".format(
                    parameters["method"]
                )
                model_parameters["method"] = parameters["method"]
            else:
                model_parameters["method"] = default_parameters["method"]
            if "n_components" in parameters:
                check_types(
                    [("n_components", parameters["n_components"], [int, float], False)]
                )
                assert (
                    0 <= parameters["n_components"]
                ), "Incorrect parameter 'n_components'.\nThe number of components must be positive. If it is equal to 0, all the components will be considered."
                model_parameters["n_components"] = parameters["n_components"]
            else:
                model_parameters["n_components"] = default_parameters["n_components"]
        elif self.type in ("OneHotEncoder"):
            if "extra_levels" in parameters:
                check_types(
                    [("extra_levels", parameters["extra_levels"], [dict], False)]
                )
                model_parameters["extra_levels"] = parameters["extra_levels"]
            else:
                model_parameters["extra_levels"] = default_parameters["extra_levels"]
        elif self.type in ("Normalizer"):
            if "method" in parameters:
                check_types([("method", parameters["method"], [str], False)])
                assert str(parameters["method"]).lower() in [
                    "size",
                    "sum_squares",
                ], "Incorrect parameter 'method'.\nThe normalization method must be in (zscore | robust_zscore | minmax), found '{}'.".format(
                    parameters["method"]
                )
                model_parameters["method"] = parameters["method"]
            else:
                model_parameters["method"] = default_parameters["method"]
        elif self.type in ("DBSCAN"):
            if "eps" in parameters:
                check_types([("eps", parameters["eps"], [int, float], False)])
                assert (
                    0 < parameters["eps"]
                ), "Incorrect parameter 'eps'.\nThe radius of a neighborhood must be strictly positive."
                model_parameters["eps"] = parameters["eps"]
            else:
                model_parameters["eps"] = default_parameters["eps"]
            if "p" in parameters:
                check_types([("p", parameters["p"], [int, float], False)])
                assert (
                    0 < parameters["p"]
                ), "Incorrect parameter 'p'.\nThe p of the p-distance must be strictly positive."
                model_parameters["p"] = parameters["p"]
            else:
                model_parameters["p"] = default_parameters["p"]
            if "min_samples" in parameters:
                check_types(
                    [("min_samples", parameters["min_samples"], [int, float], False)]
                )
                assert (
                    0 < parameters["min_samples"]
                ), "Incorrect parameter 'min_samples'.\nThe minimum number of points required to form a dense region must be strictly positive."
                model_parameters["min_samples"] = parameters["min_samples"]
            else:
                model_parameters["min_samples"] = default_parameters["min_samples"]
        elif self.type in (
            "NearestCentroid",
            "KNeighborsClassifier",
            "KNeighborsRegressor",
            "LocalOutlierFactor",
        ):
            if "p" in parameters:
                check_types([("p", parameters["p"], [int, float], False)])
                assert (
                    0 < parameters["p"]
                ), "Incorrect parameter 'p'.\nThe p of the p-distance must be strictly positive."
                model_parameters["p"] = parameters["p"]
            else:
                model_parameters["p"] = default_parameters["p"]
            if ("n_neighbors" in parameters) and (self.type != "NearestCentroid"):
                check_types(
                    [("n_neighbors", parameters["n_neighbors"], [int, float], False)]
                )
                assert (
                    0 < parameters["n_neighbors"]
                ), "Incorrect parameter 'n_neighbors'.\nThe number of neighbors must be strictly positive."
                model_parameters["n_neighbors"] = parameters["n_neighbors"]
            elif self.type != "NearestCentroid":
                model_parameters["n_neighbors"] = default_parameters["n_neighbors"]
        elif self.type in ("CountVectorizer"):
            if "max_df" in parameters:
                check_types([("max_df", parameters["max_df"], [int, float], False)])
                assert (
                    0 <= parameters["max_df"] <= 1
                ), "Incorrect parameter 'max_df'.\nIt must be between 0 and 1, inclusive."
                model_parameters["max_df"] = parameters["max_df"]
            else:
                model_parameters["max_df"] = default_parameters["max_df"]
            if "min_df" in parameters:
                check_types([("min_df", parameters["min_df"], [int, float], False)])
                assert (
                    0 <= parameters["min_df"] <= 1
                ), "Incorrect parameter 'min_df'.\nIt must be between 0 and 1, inclusive."
                model_parameters["min_df"] = parameters["min_df"]
            else:
                model_parameters["min_df"] = default_parameters["min_df"]
            if "lowercase" in parameters:
                check_types([("lowercase", parameters["lowercase"], [bool], False)])
                model_parameters["lowercase"] = parameters["lowercase"]
            else:
                model_parameters["lowercase"] = default_parameters["lowercase"]
            if "ignore_special" in parameters:
                check_types(
                    [("ignore_special", parameters["ignore_special"], [bool], False)]
                )
                model_parameters["ignore_special"] = parameters["ignore_special"]
            else:
                model_parameters["ignore_special"] = default_parameters[
                    "ignore_special"
                ]
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
                assert (
                    0 < parameters["max_text_size"]
                ), "Incorrect parameter 'max_text_size'.\nThe maximum text size must be positive."
                model_parameters["max_text_size"] = parameters["max_text_size"]
            else:
                model_parameters["max_text_size"] = default_parameters["max_text_size"]
            if "max_features" in parameters:
                check_types(
                    [("max_features", parameters["max_features"], [int, float], False)]
                )
                model_parameters["max_features"] = parameters["max_features"]
            else:
                model_parameters["max_features"] = default_parameters["max_features"]
        self.parameters = model_parameters


# ---#
class Supervised(vModel):

    # ---#
    def fit(self, input_relation: str, X: list, y: str, test_relation: str = ""):
        """
	---------------------------------------------------------------------------
	Trains the self.

	Parameters
	----------
	input_relation: str
		Train relation.
	X: list
		List of the predictors.
	y: str
		Response column.
	test_relation: str, optional
		Relation to use to test the self.

	Returns
	-------
	object
		model
		"""
        check_types(
            [
                ("input_relation", input_relation, [str], False),
                ("X", X, [list], False),
                ("y", y, [str], False),
                ("test_relation", test_relation, [str], False),
            ]
        )
        check_model(name=self.name, cursor=self.cursor)
        self.input_relation = input_relation
        self.test_relation = test_relation if (test_relation) else input_relation
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
        query = query.format(fun, self.name, input_relation, self.y, ", ".join(self.X))
        query += ", ".join(
            ["{} = {}".format(elem, parameters[elem]) for elem in parameters]
        )
        if alpha != None:
            query += ", alpha = {}".format(alpha)
        query += ")"
        self.cursor.execute(query)
        if self.type in (
            "LinearSVC",
            "LinearSVR",
            "LogisticRegression",
            "LinearRegression",
        ):
            self.coef_ = self.get_model_attribute("details")
        elif self.type in ("RandomForestClassifier", "MultinomialNB"):
            self.cursor.execute(
                "SELECT DISTINCT {} FROM {} WHERE {} IS NOT NULL ORDER BY 1".format(
                    self.y, input_relation, self.y
                )
            )
            classes = self.cursor.fetchall()
            self.classes_ = [item[0] for item in classes]
        return self


# ---#
class Tree:

    # ---#
    def export_graphviz(self, tree_id: int = 0):
        """
	---------------------------------------------------------------------------
	Converts the input tree to graphviz.

	Parameters
	----------
	tree_id: int, optional
		Unique tree identifier. It is an integer between 0 and n_estimators - 1

	Returns
	-------
	str
		graphviz formatted tree.
		"""
        check_types([("tree_id", tree_id, [int, float], False)])
        version(cursor=self.cursor, condition=[9, 1, 1])
        query = "SELECT READ_TREE ( USING PARAMETERS model_name = '{}', tree_id = {}, format = 'graphviz');".format(
            self.name, tree_id
        )
        self.cursor.execute(query)
        return self.cursor.fetchone()[1]

    # ---#
    def get_tree(self, tree_id: int = 0):
        """
	---------------------------------------------------------------------------
	Returns a table with all the input tree information.

	Parameters
	----------
	tree_id: int, optional
		Unique tree identifier. It is an integer between 0 and n_estimators - 1

	Returns
	-------
	tablesample
		An object containing the result. For more information, see
		utilities.tablesample.
		"""
        check_types([("tree_id", tree_id, [int, float], False)])
        version(cursor=self.cursor, condition=[9, 1, 1])
        query = "SELECT READ_TREE ( USING PARAMETERS model_name = '{}', tree_id = {}, format = 'tabular');".format(
            self.name, tree_id
        )
        result = to_tablesample(query=query, cursor=self.cursor)
        result.table_info = False
        return result

    # ---#
    def plot_tree(self, tree_id: int = 0, pic_path: str = ""):
        """
	---------------------------------------------------------------------------
	Draws the input tree. The module anytree must be installed in the machine.

	Parameters
	----------
	tree_id: int, optional
		Unique tree identifier. It is an integer between 0 and n_estimators - 1
	pic_path: str, optional
		Absolute path to save the image of the tree.
		"""
        check_types(
            [
                ("tree_id", tree_id, [int, float], False),
                ("pic_path", pic_path, [str], False),
            ]
        )
        plot_tree(
            self.get_tree(tree_id=tree_id).values, metric="variance", pic_path=pic_path
        )


# ---#
class NeighborsClassifier:

    # ---#
    def classification_report(self, cutoff=[], labels: list = []):
        """
	---------------------------------------------------------------------------
	Computes a classification report using multiple metrics to evaluate the model
	(AUC, accuracy, PRC AUC, F1...). In case of multiclass classification, it will 
	consider each category as positive and switch to the next one during the computation.

	Parameters
	----------
	cutoff: float/list, optional
		Cutoff for which the tested category will be accepted as prediction. 
		In case of multiclass classification, each tested category becomes 
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
        check_types(
            [
                ("cutoff", cutoff, [int, float, list], False),
                ("labels", labels, [list], False),
            ]
        )
        if not (labels):
            labels = self.classes_
        return classification_report(cutoff=cutoff, estimator=self, labels=labels)

    # ---#
    def confusion_matrix(self, pos_label=None, cutoff: float = -1):
        """
	---------------------------------------------------------------------------
	Computes the model confusion matrix.

	Parameters
	----------
	pos_label: int/float/str, optional
		Label to consider as positive. All the other classes will be merged and
		considered as negative in case of multi classification.
	cutoff: float, optional
		Cutoff for which the tested category will be accepted as prediction. If the 
		cutoff is not between 0 and 1, the entire confusion matrix will be drawn.

	Returns
	-------
	tablesample
 		An object containing the result. For more information, see
 		utilities.tablesample.
		"""
        check_types([("cutoff", cutoff, [int, float], False)])
        pos_label = (
            self.classes_[1]
            if (pos_label == None and len(self.classes_) == 2)
            else pos_label
        )
        if pos_label in self.classes_ and cutoff <= 1 and cutoff >= 0:
            input_relation = self.deploySQL() + " WHERE predict_neighbors = '{}'".format(
                pos_label
            )
            y_score = "(CASE WHEN proba_predict > {} THEN 1 ELSE 0 END)".format(cutoff)
            y_true = "DECODE({}, '{}', 1, 0)".format(self.y, pos_label)
            result = confusion_matrix(y_true, y_score, input_relation, self.cursor)
            if pos_label == 1:
                return result
            else:
                return tablesample(
                    values={
                        "index": ["Non-{}".format(pos_label), "{}".format(pos_label)],
                        "Non-{}".format(pos_label): result.values[0],
                        "{}".format(pos_label): result.values[1],
                    },
                    table_info=False,
                )
        else:
            input_relation = "(SELECT *, ROW_NUMBER() OVER(PARTITION BY {}, row_id ORDER BY proba_predict DESC) AS pos FROM {}) neighbors_table WHERE pos = 1".format(
                ", ".join(self.X), self.deploySQL()
            )
            return multilabel_confusion_matrix(
                self.y, "predict_neighbors", input_relation, self.classes_, self.cursor
            )

    # ---#
    def lift_chart(self, pos_label=None):
        """
	---------------------------------------------------------------------------
	Draws the model Lift Chart.

	Parameters
	----------
	pos_label: int/float/str
		To draw a lift chart, one of the response column class has to be the 
		positive one. The parameter 'pos_label' represents this class.

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
        input_relation = self.deploySQL() + " WHERE predict_neighbors = '{}'".format(
            pos_label
        )
        return lift_chart(
            self.y, "proba_predict", input_relation, self.cursor, pos_label
        )

    # ---#
    def prc_curve(self, pos_label=None):
        """
	---------------------------------------------------------------------------
	Draws the model PRC curve.

	Parameters
	----------
	pos_label: int/float/str
		To draw the PRC curve, one of the response column class has to be the 
		positive one. The parameter 'pos_label' represents this class.

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
        input_relation = self.deploySQL() + " WHERE predict_neighbors = '{}'".format(
            pos_label
        )
        return prc_curve(
            self.y, "proba_predict", input_relation, self.cursor, pos_label
        )

    # ---#
    def predict(
        self,
        vdf,
        X: list = [],
        name: str = "",
        cutoff: float = -1,
        all_classes: bool = False,
    ):
        """
	---------------------------------------------------------------------------
	Predicts using the input relation.

	Parameters
	----------
	vdf: vDataFrame
		Object to use to run the prediction.
	X: list, optional
		List of the columns used to deploy the models. If empty, the model
		predictors will be used.
	name: str, optional
		Name of the added vcolumn. If empty, a name will be generated.
	cutoff: float, optional
		Cutoff used in case of binary classification. It is the probability to
		accept the category 1.
	all_classes: bool, optional
		If set to True, all the classes probabilities will be generated (one 
		column per category).

	Returns
	-------
	vDataFrame
 		the vDataFrame of the prediction
		"""
        check_types(
            [
                ("cutoff", cutoff, [int, float], False),
                ("all_classes", all_classes, [bool], False),
                ("name", name, [str], False),
                ("cutoff", cutoff, [int, float], False),
                ("X", X, [list], False),
            ],
            vdf=["vdf", vdf],
        )
        X = [str_column(elem) for elem in X] if (X) else self.X
        key_columns = vdf.get_columns(exclude_columns=X)
        name = (
            "{}_".format(self.type) + "".join(ch for ch in self.name if ch.isalnum())
            if not (name)
            else name
        )
        if all_classes:
            predict = [
                "ZEROIFNULL(AVG(DECODE(predict_neighbors, '{}', proba_predict, NULL))) AS \"{}_{}\"".format(
                    elem, name, elem
                )
                for elem in self.classes_
            ]
            sql = "SELECT {}{}, {} FROM {} GROUP BY {}".format(
                ", ".join(X),
                ", " + ", ".join(key_columns) if key_columns else "",
                ", ".join(predict),
                self.deploySQL(
                    X=X, test_relation=vdf.__genSQL__(), key_columns=key_columns
                ),
                ", ".join(X + key_columns),
            )
        else:
            if (len(self.classes_) == 2) and (cutoff <= 1 and cutoff >= 0):
                sql = "SELECT {}{}, (CASE WHEN proba_predict > {} THEN '{}' ELSE '{}' END) AS {} FROM {} WHERE predict_neighbors = '{}'".format(
                    ", ".join(X),
                    ", " + ", ".join(key_columns) if key_columns else "",
                    cutoff,
                    self.classes_[1],
                    self.classes_[0],
                    name,
                    self.deploySQL(
                        X=X, test_relation=vdf.__genSQL__(), key_columns=key_columns
                    ),
                    self.classes_[1],
                )
            elif len(self.classes_) == 2:
                sql = "SELECT {}{}, proba_predict AS {} FROM {} WHERE predict_neighbors = '{}'".format(
                    ", ".join(X),
                    ", " + ", ".join(key_columns) if key_columns else "",
                    name,
                    self.deploySQL(
                        X=X, test_relation=vdf.__genSQL__(), key_columns=key_columns
                    ),
                    self.classes_[1],
                )
            else:
                sql = "SELECT {}{}, predict_neighbors AS {} FROM {}".format(
                    ", ".join(X),
                    ", " + ", ".join(key_columns) if key_columns else "",
                    name,
                    self.deploySQL(
                        X=X,
                        test_relation=vdf.__genSQL__(),
                        key_columns=key_columns,
                        predict=True,
                    ),
                )
        sql = "({}) x".format(sql)
        return vdf_from_relation(name="Neighbors", relation=sql, cursor=self.cursor)

    # ---#
    def roc_curve(self, pos_label=None):
        """
	---------------------------------------------------------------------------
	Draws the model ROC curve.

	Parameters
	----------
	pos_label: int/float/str
		To draw the ROC curve, one of the response column class has to be the 
		positive one. The parameter 'pos_label' represents this class.

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
        input_relation = self.deploySQL() + " WHERE predict_neighbors = '{}'".format(
            pos_label
        )
        return roc_curve(
            self.y, "proba_predict", input_relation, self.cursor, pos_label
        )

    # ---#
    def score(self, method: str = "accuracy", pos_label=None, cutoff: float = -1):
        """
	---------------------------------------------------------------------------
	Computes the model score.

	Parameters
	----------
	pos_label: int/float/str, optional
		Label to consider as positive. All the other classes will be merged and
		considered as negative in case of multi classification.
	cutoff: float, optional
		Cutoff for which the tested category will be accepted as prediction. 
	method: str, optional
		The method to use to compute the score.
			accuracy    : Accuracy
			auc         : Area Under the Curve (ROC)
			best_cutoff : Cutoff which optimised the ROC Curve prediction.
			bm	        : Informedness = tpr + tnr - 1
			csi         : Critical Success Index = tp / (tp + fn + fp)
			f1	        : F1 Score 
			logloss     : Log Loss
			mcc         : Matthews Correlation Coefficient 
			mk	        : Markedness = ppv + npv - 1
			npv         : Negative Predictive Value = tn / (tn + fn)
			prc_auc     : Area Under the Curve (PRC)
			precision   : Precision = tp / (tp + fp)
			recall      : Recall = tp / (tp + fn)
			specificity : Specificity = tn / (tn + fp) 

	Returns
	-------
	float
 		score
		"""
        check_types(
            [("cutoff", cutoff, [int, float], False), ("method", method, [str], False)]
        )
        if pos_label == None and len(self.classes_) == 2:
            pos_label = self.classes_[1]
        input_relation = "(SELECT * FROM {} WHERE predict_neighbors = '{}') final_centroids_relation".format(
            self.deploySQL(), pos_label
        )
        y_score = "(CASE WHEN proba_predict > {} THEN 1 ELSE 0 END)".format(cutoff)
        y_proba = "proba_predict"
        y_true = "DECODE({}, '{}', 1, 0)".format(self.y, pos_label)
        if (pos_label not in self.classes_) and (method != "accuracy"):
            raise ParameterError(
                "'pos_label' must be one of the response column classes"
            )
        elif (cutoff >= 1 or cutoff <= 0) and (method != "accuracy"):
            cutoff = self.score(pos_label=pos_label, cutoff=0.5, method="best_cutoff")
        if method in ("accuracy", "acc"):
            if pos_label not in self.classes_:
                return accuracy_score(
                    self.y,
                    "predict_neighbors",
                    self.deploySQL(predict=True),
                    self.cursor,
                    pos_label=None,
                )
            else:
                return accuracy_score(y_true, y_score, input_relation, self.cursor)
        elif method == "auc":
            return auc(y_true, y_proba, input_relation, self.cursor)
        elif method == "prc_auc":
            return prc_auc(y_true, y_proba, input_relation, self.cursor)
        elif method in ("best_cutoff", "best_threshold"):
            return roc_curve(
                y_true, y_proba, input_relation, self.cursor, best_threshold=True
            )
        elif method in ("recall", "tpr"):
            return recall_score(y_true, y_score, input_relation, self.cursor)
        elif method in ("precision", "ppv"):
            return precision_score(y_true, y_score, input_relation, self.cursor)
        elif method in ("specificity", "tnr"):
            return specificity_score(y_true, y_score, input_relation, self.cursor)
        elif method in ("negative_predictive_value", "npv"):
            return precision_score(y_true, y_score, input_relation, self.cursor)
        elif method in ("log_loss", "logloss"):
            return log_loss(y_true, y_proba, input_relation, self.cursor)
        elif method == "f1":
            return f1_score(y_true, y_score, input_relation, self.cursor)
        elif method == "mcc":
            return matthews_corrcoef(y_true, y_score, input_relation, self.cursor)
        elif method in ("bm", "informedness"):
            return informedness(y_true, y_score, input_relation, self.cursor)
        elif method in ("mk", "markedness"):
            return markedness(y_true, y_score, input_relation, self.cursor)
        elif method in ("csi", "critical_success_index"):
            return critical_success_index(y_true, y_score, input_relation, self.cursor)
        else:
            raise ParameterError(
                "The parameter 'method' must be in accuracy|auc|prc_auc|best_cutoff|recall|precision|log_loss|negative_predictive_value|specificity|mcc|informedness|markedness|critical_success_index"
            )


# ---#
class NeighborsRegressor:

    # ---#
    def predict(self, vdf, X: list = [], name: str = ""):
        """
	---------------------------------------------------------------------------
	Predicts using the input relation.

	Parameters
	----------
	vdf: vDataFrame
		Object to use to run the prediction.
	X: list, optional
		List of the columns used to deploy the models. If empty, the model
		predictors will be used.
	name: str, optional
		Name of the added vcolumn. If empty, a name will be generated.

	Returns
	-------
	vDataFrame
 		the vDataFrame of the prediction
		"""
        check_types(
            [("name", name, [str], False), ("X", X, [list], False),], vdf=["vdf", vdf],
        )
        X = [str_column(elem) for elem in X] if (X) else self.X
        key_columns = vdf.get_columns(exclude_columns=X)
        name = (
            "{}_".format(self.type) + "".join(ch for ch in self.name if ch.isalnum())
            if not (name)
            else name
        )
        sql = "(SELECT {}{}, {} AS {} FROM {}) x".format(
            ", ".join(X),
            ", " + ", ".join(key_columns) if key_columns else "",
            "predict_neighbors",
            name,
            self.deploySQL(
                X=X, test_relation=vdf.__genSQL__(), key_columns=key_columns
            ),
        )
        return vdf_from_relation(name="Neighbors", relation=sql, cursor=self.cursor)

    # ---#
    def regression_report(self):
        """
	---------------------------------------------------------------------------
	Computes a regression report using multiple metrics to evaluate the model
	(r2, mse, max error...). 

	Returns
	-------
	tablesample
		An object containing the result. For more information, see
		utilities.tablesample.
		"""
        return regression_report(
            self.y, "predict_neighbors", self.deploySQL(), self.cursor
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
			max    : Max Error
			mae    : Mean Absolute Error
			median : Median Absolute Error
			mse    : Mean Squared Error
			msle   : Mean Squared Log Error
			r2     : R squared coefficient
			var    : Explained Variance 

	Returns
	-------
	float
 		score
		"""
        check_types([("method", method, [str], False)])
        if method in ("r2", "rsquared"):
            return r2_score(self.y, "predict_neighbors", self.deploySQL(), self.cursor)
        elif method in ("mae", "mean_absolute_error"):
            return mean_absolute_error(
                self.y, "predict_neighbors", self.deploySQL(), self.cursor
            )
        elif method in ("mse", "mean_squared_error"):
            return mean_squared_error(
                self.y, "predict_neighbors", self.deploySQL(), self.cursor
            )
        elif method in ("msle", "mean_squared_log_error"):
            return mean_squared_log_error(
                self.y, "predict_neighbors", self.deploySQL(), self.cursor
            )
        elif method in ("max", "max_error"):
            return max_error(self.y, "predict_neighbors", self.deploySQL(), self.cursor)
        elif method in ("median", "median_absolute_error"):
            return median_absolute_error(
                self.y, "predict_neighbors", self.deploySQL(), self.cursor
            )
        elif method in ("var", "explained_variance"):
            return explained_variance(
                self.y, "predict_neighbors", self.deploySQL(), self.cursor
            )
        else:
            raise ParameterError(
                "The parameter 'method' must be in r2|mae|mse|msle|max|median|var"
            )


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
        check_types([("cutoff", cutoff, [int, float], False)])
        if cutoff > 1 or cutoff < 0:
            cutoff = self.score(method="best_cutoff")
        return classification_report(
            self.y,
            [self.deploySQL(), self.deploySQL(cutoff)],
            self.test_relation,
            self.cursor,
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
        check_types([("cutoff", cutoff, [int, float], False)])
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
        check_types([("cutoff", cutoff, [int, float], False), ("X", X, [list], False)])
        X = [str_column(elem) for elem in X]
        fun = self.get_model_fun()[1]
        sql = "{}({} USING PARAMETERS model_name = '{}', type = 'probability', match_by_pos = 'true')"
        if cutoff <= 1 and cutoff >= 0:
            sql = "(CASE WHEN {} > {} THEN 1 ELSE 0 END)".format(sql, cutoff)
        return sql.format(fun, ", ".join(self.X if not (X) else X), self.name)

    # ---#
    def lift_chart(self):
        """
	---------------------------------------------------------------------------
	Draws the model Lift Chart.

	Returns
	-------
	tablesample
		An object containing the result. For more information, see
		utilities.tablesample.
		"""
        return lift_chart(self.y, self.deploySQL(), self.test_relation, self.cursor)

    # ---#
    def prc_curve(self):
        """
	---------------------------------------------------------------------------
	Draws the model PRC curve.

	Returns
	-------
	tablesample
		An object containing the result. For more information, see
		utilities.tablesample.
		"""
        return prc_curve(self.y, self.deploySQL(), self.test_relation, self.cursor)

    # ---#
    def predict(
        self,
        vdf,
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
	vdf: vDataFrame
		Object to use to run the prediction.
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
        check_types(
            [
                ("name", name, [str], False),
                ("cutoff", cutoff, [int, float], False),
                ("X", X, [list], False),
            ],
            vdf=["vdf", vdf],
        )
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
    def roc_curve(self):
        """
	---------------------------------------------------------------------------
	Draws the model ROC curve.

	Returns
	-------
	tablesample
		An object containing the result. For more information, see
		utilities.tablesample.
		"""
        return roc_curve(self.y, self.deploySQL(), self.test_relation, self.cursor)

    # ---#
    def score(self, method: str = "accuracy", cutoff: float = 0.5):
        """
	---------------------------------------------------------------------------
	Computes the model score.

	Parameters
	----------
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

	cutoff: float, optional
		Cutoff for which the tested category will be accepted as prediction. 

	Returns
	-------
	float
		score
		"""
        check_types(
            [("cutoff", cutoff, [int, float], False), ("method", method, [str], False)]
        )
        if method in ("accuracy", "acc"):
            return accuracy_score(
                self.y, self.deploySQL(cutoff), self.test_relation, self.cursor
            )
        elif method == "auc":
            return auc(self.y, self.deploySQL(), self.test_relation, self.cursor)
        elif method == "prc_auc":
            return prc_auc(self.y, self.deploySQL(), self.test_relation, self.cursor)
        elif method in ("best_cutoff", "best_threshold"):
            return roc_curve(
                self.y,
                self.deploySQL(),
                self.test_relation,
                self.cursor,
                best_threshold=True,
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
                "The parameter 'method' must be in accuracy|auc|prc_auc|best_cutoff|recall|precision|log_loss|negative_predictive_value|specificity|mcc|informedness|markedness|critical_success_index"
            )


# ---#
class MulticlassClassifier(Classifier):

    # ---#
    def classification_report(self, cutoff=[], labels: list = []):
        """
	---------------------------------------------------------------------------
	Computes a classification report using multiple metrics to evaluate the model
	(AUC, accuracy, PRC AUC, F1...). In case of multiclass classification, it will 
	consider each category as positive and switch to the next one during the computation.

	Parameters
	----------
	cutoff: float/list, optional
		Cutoff for which the tested category will be accepted as prediction. 
		In case of multiclass classification, each tested category becomes 
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
        check_types(
            [
                ("cutoff", cutoff, [int, float, list], False),
                ("labels", labels, [list], False),
            ]
        )
        if not (labels):
            labels = self.classes_
        return classification_report(cutoff=cutoff, estimator=self, labels=labels)

    # ---#
    def confusion_matrix(self, pos_label=None, cutoff: float = -1):
        """
	---------------------------------------------------------------------------
	Computes the model confusion matrix.

	Parameters
	----------
	pos_label: int/float/str, optional
		Label to consider as positive. All the other classes will be merged and
		considered as negative in case of multi classification.
	cutoff: float, optional
		Cutoff for which the tested category will be accepted as prediction. If the 
		cutoff is not between 0 and 1, the entire confusion matrix will be drawn.

	Returns
	-------
	tablesample
		An object containing the result. For more information, see
		utilities.tablesample.
		"""
        check_types([("cutoff", cutoff, [int, float], False)])
        pos_label = (
            self.classes_[1]
            if (pos_label == None and len(self.classes_) == 2)
            else pos_label
        )
        if pos_label in self.classes_ and cutoff <= 1 and cutoff >= 0:
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
    def deploySQL(
        self, pos_label=None, cutoff: float = -1, allSQL: bool = False, X: list = []
    ):
        """
	---------------------------------------------------------------------------
	Returns the SQL code needed to deploy the model. 

	Parameters
	----------
	pos_label: int/float/str, optional
		Label to consider as positive. All the other classes will be merged and
		considered as negative in case of multi classification.
	cutoff: float, optional
		Cutoff for which the tested category will be accepted as prediction. If 
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
        check_types(
            [
                ("cutoff", cutoff, [int, float], False),
                ("allSQL", allSQL, [bool], False),
                ("X", X, [list], False),
            ]
        )
        X = [str_column(elem) for elem in X]
        fun = self.get_model_fun()[1]
        if allSQL:
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
            if pos_label in self.classes_ and cutoff <= 1 and cutoff >= 0:
                sql = "{}({} USING PARAMETERS model_name = '{}', class = '{}', type = 'probability', match_by_pos = 'true')".format(
                    fun, ", ".join(self.X if not (X) else X), self.name, pos_label
                )
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
            elif pos_label in self.classes_:
                sql = "{}({} USING PARAMETERS model_name = '{}', class = '{}', type = 'probability', match_by_pos = 'true')".format(
                    fun, ", ".join(self.X if not (X) else X), self.name, pos_label
                )
            else:
                sql = "{}({} USING PARAMETERS model_name = '{}', match_by_pos = 'true')".format(
                    fun, ", ".join(self.X if not (X) else X), self.name
                )
        return sql

    # ---#
    def lift_chart(self, pos_label=None):
        """
	---------------------------------------------------------------------------
	Draws the model Lift Chart.

	Parameters
	----------
	pos_label: int/float/str, optional
		To draw a lift chart, one of the response column class has to be the 
		positive one. The parameter 'pos_label' represents this class.

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
        return lift_chart(
            self.y,
            self.deploySQL(allSQL=True)[0].format(pos_label),
            self.test_relation,
            self.cursor,
            pos_label,
        )

    # ---#
    def prc_curve(self, pos_label=None):
        """
	---------------------------------------------------------------------------
	Draws the model PRC curve.

	Parameters
	----------
	pos_label: int/float/str, optional
		To draw the PRC curve, one of the response column class has to be the 
		positive one. The parameter 'pos_label' represents this class.

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
        return prc_curve(
            self.y,
            self.deploySQL(allSQL=True)[0].format(pos_label),
            self.test_relation,
            self.cursor,
            pos_label,
        )

    # ---#
    def predict(
        self,
        vdf,
        X: list = [],
        name: str = "",
        cutoff: float = -1,
        pos_label=None,
        inplace: bool = True,
    ):
        """
	---------------------------------------------------------------------------
	Predicts using the input relation.

	Parameters
	----------
	vdf: vDataFrame
		Object to use to run the prediction.
	X: list, optional
		List of the columns used to deploy the models. If empty, the model
		predictors will be used.
	name: str, optional
		Name of the added vcolumn. If empty, a name will be generated.
	cutoff: float, optional
		Cutoff for which the tested category will be accepted as prediction. 
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
        check_types(
            [
                ("name", name, [str], False),
                ("cutoff", cutoff, [int, float], False),
                ("X", X, [list], False),
            ],
            vdf=["vdf", vdf],
        )
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
    def roc_curve(self, pos_label=None):
        """
	---------------------------------------------------------------------------
	Draws the model ROC curve.

	Parameters
	----------
	pos_label: int/float/str, optional
		To draw the ROC curve, one of the response column class has to be the 
		positive one. The parameter 'pos_label' represents this class.

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
        return roc_curve(
            self.y,
            self.deploySQL(allSQL=True)[0].format(pos_label),
            self.test_relation,
            self.cursor,
            pos_label,
        )

    # ---#
    def score(self, method: str = "accuracy", pos_label=None, cutoff: float = -1):
        """
	---------------------------------------------------------------------------
	Computes the model score.

	Parameters
	----------
	pos_label: int/float/str, optional
		Label to consider as positive. All the other classes will be merged and
		considered as negative in case of multi classification.
	cutoff: float, optional
		Cutoff for which the tested category will be accepted as prediction. 
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

	Returns
	-------
	float
		score
		"""
        check_types(
            [("cutoff", cutoff, [int, float], False), ("method", method, [str], False)]
        )
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
                self.deploySQL(allSQL=True)[0].format(pos_label),
                self.test_relation,
                self.cursor,
            )
        elif method == "prc_auc":
            return prc_auc(
                "DECODE({}, '{}', 1, 0)".format(self.y, pos_label),
                self.deploySQL(allSQL=True)[0].format(pos_label),
                self.test_relation,
                self.cursor,
            )
        elif method in ("best_cutoff", "best_threshold"):
            return roc_curve(
                "DECODE({}, '{}', 1, 0)".format(self.y, pos_label),
                self.deploySQL(allSQL=True)[0].format(pos_label),
                self.test_relation,
                self.cursor,
                best_threshold=True,
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
                self.deploySQL(allSQL=True)[0].format(pos_label),
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
                "The parameter 'method' must be in accuracy|auc|prc_auc|best_cutoff|recall|precision|log_loss|negative_predictive_value|specificity|mcc|informedness|markedness|critical_success_index"
            )


# ---#
class Regressor(Supervised):

    # ---#
    def predict(self, vdf, X: list = [], name: str = "", inplace: bool = True):
        """
	---------------------------------------------------------------------------
	Predicts using the input relation.

	Parameters
	----------
	vdf: vDataFrame
		Object to use to run the prediction.
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
        check_types(
            [("name", name, [str], False), ("X", X, [list], False)], vdf=["vdf", vdf]
        )
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
    def regression_report(self):
        """
	---------------------------------------------------------------------------
	Computes a regression report using multiple metrics to evaluate the model
	(r2, mse, max error...). 

	Returns
	-------
	tablesample
		An object containing the result. For more information, see
		utilities.tablesample.
		"""
        return regression_report(
            self.y, self.deploySQL(), self.test_relation, self.cursor
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
			max	   : Max Error
			mae	   : Mean Absolute Error
			median : Median Absolute Error
			mse	   : Mean Squared Error
			msle   : Mean Squared Log Error
			r2	   : R squared coefficient
			var	   : Explained Variance 

	Returns
	-------
	float
		score
		"""
        check_types([("method", method, [str], False)])
        if method in ("r2", "rsquared"):
            return r2_score(self.y, self.deploySQL(), self.test_relation, self.cursor)
        elif method in ("mae", "mean_absolute_error"):
            return mean_absolute_error(
                self.y, self.deploySQL(), self.test_relation, self.cursor
            )
        elif method in ("mse", "mean_squared_error"):
            return mean_squared_error(
                self.y, self.deploySQL(), self.test_relation, self.cursor
            )
        elif method in ("msle", "mean_squared_log_error"):
            return mean_squared_log_error(
                self.y, self.deploySQL(), self.test_relation, self.cursor
            )
        elif method in ("max", "max_error"):
            return max_error(self.y, self.deploySQL(), self.test_relation, self.cursor)
        elif method in ("median", "median_absolute_error"):
            return median_absolute_error(
                self.y, self.deploySQL(), self.test_relation, self.cursor
            )
        elif method in ("var", "explained_variance"):
            return explained_variance(
                self.y, self.deploySQL(), self.test_relation, self.cursor
            )
        else:
            raise ParameterError(
                "The parameter 'method' must be in r2|mae|mse|msle|max|median|var"
            )


# ---#
class Unsupervised(vModel):

    # ---#
    def fit(self, input_relation: str, X: list):
        """
	---------------------------------------------------------------------------
	Trains the self.

	Parameters
	----------
	input_relation: str
		Train relation.
	X: list
		List of the predictors.

	Returns
	-------
	object
		model
		"""
        check_types(
            [("input_relation", input_relation, [str], False), ("X", X, [list], False)]
        )
        check_model(name=self.name, cursor=self.cursor)
        self.input_relation = input_relation
        self.X = [str_column(column) for column in X]
        parameters = vertica_param_dict(self)
        if "num_components" in parameters and not (parameters["num_components"]):
            del parameters["num_components"]
        fun = self.get_model_fun()[0]
        query = "SELECT {}('{}', '{}', '{}'".format(
            fun, self.name, input_relation, ", ".join(self.X)
        )
        if self.type in ("BisectingKMeans", "KMeans"):
            query += ", {}".format(parameters["n_cluster"])
        elif self.type == "Normalizer":
            query += ", {}".format(parameters["method"])
            del parameters["method"]
        if self.type != "Normalizer":
            query += " USING PARAMETERS "
        if (
            "init_method" in parameters
            and type(parameters["init_method"]) != str
            and self.type == "KMeans"
        ):
            schema = schema_relation(self.name)[0]
            name = "VERTICAPY_KMEANS_INITIAL"
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
        self.cursor.execute(query)
        if self.type == "KMeans":
            try:
                self.cursor.execute("DROP TABLE IF EXISTS {}.{}".format(schema, name))
            except:
                pass
            self.cluster_centers_ = self.get_model_attribute("centers")
            result = self.get_model_attribute("metrics").values["metrics"][0]
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
            self.metrics_ = tablesample(values, table_info=False)
        elif self.type in ("BisectingKMeans"):
            self.metrics_ = self.get_model_attribute("Metrics")
            self.cluster_centers_ = self.get_model_attribute("BKTree")
        elif self.type in ("PCA"):
            self.components_ = self.get_model_attribute("principal_components")
            self.explained_variance_ = self.get_model_attribute("singular_values")
            self.mean_ = self.get_model_attribute("columns")
        elif self.type in ("SVD"):
            self.singular_values_ = self.get_model_attribute("right_singular_vectors")
            self.explained_variance_ = self.get_model_attribute("singular_values")
        elif self.type in ("Normalizer"):
            self.param_ = self.get_model_attribute("details")
        elif self.type == "OneHotEncoder":
            try:
                self.param_ = to_tablesample(
                    query="SELECT category_name, category_level::varchar, category_level_index FROM (SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'integer_categories')) x UNION ALL SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'varchar_categories')".format(
                        self.name, self.name
                    ),
                    cursor=self.cursor,
                )
            except:
                try:
                    self.param_ = to_tablesample(
                        query="SELECT category_name, category_level::varchar, category_level_index FROM (SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'integer_categories')) x".format(
                            self.name
                        ),
                        cursor=self.cursor,
                    )
                except:
                    self.param_ = self.get_model_attribute("varchar_categories")
            self.param_.table_info = False
        return self


# ---#
class Decomposition(Unsupervised):

    # ---#
    def deploySQL(
        self,
        n_components: int = 0,
        cutoff: float = 1,
        key_columns: list = [],
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
	X: list, optional
		List of the columns used to deploy the self. If empty, the model
		predictors will be used.

	Returns
	-------
	str
		the SQL code needed to deploy the model.
		"""
        check_types(
            [
                ("n_components", n_components, [int, float], False),
                ("cutoff", cutoff, [int, float], False),
                ("key_columns", key_columns, [list], False),
                ("X", X, [list], False),
            ]
        )
        X = [str_column(elem) for elem in X]
        fun = self.get_model_fun()[1]
        sql = "{}({} USING PARAMETERS model_name = '{}', match_by_pos = 'true'"
        if key_columns:
            sql += ", key_columns = '{}'".format(
                ", ".join([str_column(item) for item in key_columns])
            )
        if n_components:
            sql += ", num_components = {}".format(n_components)
        else:
            sql += ", cutoff = {}".format(cutoff)
        sql += ")"
        return sql.format(fun, ", ".join(self.X if not (X) else X), self.name)

    # ---#
    def deployInverseSQL(self, key_columns: list = [], X: list = []):
        """
	---------------------------------------------------------------------------
	Returns the SQL code needed to deploy the inverse model. 

	Parameters
	----------
	key_columns: list, optional
		Predictors used during the algorithm computation which will be deployed
		with the principal components.
	X: list, optional
		List of the columns used to deploy the self. If empty, the model
		predictors will be used.

	Returns
	-------
	str
		the SQL code needed to deploy the inverse model.
		"""
        check_types(
            [("key_columns", key_columns, [list], False), ("X", X, [list], False)]
        )
        X = [str_column(elem) for elem in X]
        fun = self.get_model_fun()[2]
        sql = "{}({} USING PARAMETERS model_name = '{}', match_by_pos = 'true'"
        if key_columns:
            sql += ", key_columns = '{}'".format(
                ", ".join([str_column(item) for item in key_columns])
            )
        sql += ")"
        return sql.format(fun, ", ".join(self.X if not (X) else X), self.name)

    # ---#
    def inverse_transform(self, vdf=None, X: list = [], key_columns: list = []):
        """
	---------------------------------------------------------------------------
	Applies the Inverse Model on a vDataFrame.

	Parameters
	----------
	vdf: vDataFrame, optional
		input vDataFrame.
	X: list, optional
		List of the input vcolumns.
	key_columns: list, optional
		Predictors to keep unchanged during the transformation.

	Returns
	-------
	vDataFrame
		object result of the model transformation.
		"""
        check_types(
            [("key_columns", key_columns, [list], False), ("X", X, [list], False)]
        )
        if vdf:
            check_types(vdf=["vdf", vdf])
            X = vdf_columns_names(X, vdf)
            relation = vdf.__genSQL__()
        else:
            relation = self.input_relation
            X = [str_column(elem) for elem in X]
        main_relation = "(SELECT {} FROM {}) x".format(
            self.deployInverseSQL(key_columns, self.X if not (X) else X), relation
        )
        return vdf_from_relation(main_relation, "Inverse Transformation", self.cursor,)

    # ---#
    def score(
        self, X: list = [], input_relation: str = "", method: str = "avg", p: int = 2
    ):
        """
	---------------------------------------------------------------------------
	Returns the decomposition Score on a dataset for each trasformed column. It 
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
			avg	: The average is used as aggregation.
			median : The mdeian is used as aggregation.
	p: int, optional
		The p of the p-distance.

	Returns
	-------
	tablesample
		An object containing the result. For more information, see
		utilities.tablesample.
		"""
        check_types(
            [
                ("X", X, [list], False),
                ("input_relation", input_relation, [str], False),
                ("method", str(method).lower(), ["avg", "median"], True),
                ("p", p, [int, float], False),
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
        n_components = self.parameters["n_components"]
        if not (n_components):
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
        query = "SELECT {} FROM ({}) x".format(", ".join(col_init_1 + cols), query)
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
        result = to_tablesample(query, cursor=self.cursor).transpose()
        result.table_info = False
        return result

    # ---#
    def transform(
        self,
        vdf=None,
        X: list = [],
        n_components: int = 0,
        cutoff: float = 1,
        key_columns: list = [],
    ):
        """
	---------------------------------------------------------------------------
	Applies the model on a vDataFrame.

	Parameters
	----------
	vdf: vDataFrame, optional
		Input vDataFrame.
	X: list, optional
		List of the input vcolumns.
	n_components: int, optional
		Number of components to return. If set to 0, all the components will 
		be deployed.
	cutoff: float, optional
		Specifies the minimum accumulated explained variance. Components are 
		taken until the accumulated explained variance reaches this value.
	key_columns: list, optional
		Predictors to keep unchanged during the transformation.

	Returns
	-------
	vDataFrame
		object result of the model transformation.
		"""
        check_types(
            [
                ("n_components", n_components, [int, float], False),
                ("cutoff", cutoff, [int, float], False),
                ("key_columns", key_columns, [list], False),
                ("X", X, [list], False),
            ]
        )
        if vdf:
            check_types(vdf=["vdf", vdf])
            X = vdf_columns_names(X, vdf)
            relation = vdf.__genSQL__()
        else:
            relation = self.input_relation
            X = [str_column(elem) for elem in X]
        main_relation = "(SELECT {} FROM {}) x".format(
            self.deploySQL(n_components, cutoff, key_columns, self.X if not (X) else X),
            relation,
        )
        return vdf_from_relation(main_relation, "Transformation", self.cursor,)


# ---#
class Preprocessing(Unsupervised):

    # ---#
    def transform(self, vdf=None, X: list = []):
        """
	---------------------------------------------------------------------------
	Applies the model on a vDataFrame.

	Parameters
	----------
	vdf: vDataFrame, optional
		Input vDataFrame.
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
                self.deploySQL(self.X if not (X) else X), relation
            ),
            self.name,
            self.cursor,
        )


# ---#
class Clustering(Unsupervised):

    # ---#
    def predict(self, vdf, X: list = [], name: str = "", inplace: bool = True):
        """
	---------------------------------------------------------------------------
	Predicts using the input relation.

	Parameters
	----------
	vdf: vDataFrame
		Object to use to run the prediction.
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
        check_types(
            [("name", name, [str], False), ("X", X, [list], False)], vdf=["vdf", vdf]
        )
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
