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
# VerticaPy Modules
from verticapy.toolbox import *

#
# ---#
def check_model(name: str, cursor=None):
    """
---------------------------------------------------------------------------
Checks if the model already exists.

Parameters
----------
name: str
    Model name.
cursor: DBcursor, optional
    Vertica DB cursor.
    """
    check_types([("name", name, [str],)])
    cursor, conn = check_cursor(cursor)[0:2]
    cursor.execute("SELECT * FROM MODELS WHERE model_name='{}'".format(name))
    result = cursor.fetchone()
    if result:
        if conn:
            conn.close()
        raise NameError("The model '{}' already exists !".format(name))
    else:
        try:
            cursor.execute(
                "SELECT * FROM verticapy.models WHERE LOWER(model_name) = '{}'".format(
                    str_column(name)
                )
            )
            result = cursor.fetchone()
        except:
            result = []
        if result:
            raise NameError("The model '{}' already exists !".format(name))
    if conn:
        conn.close()


# ---#
def load_model(name: str, cursor=None, test_relation: str = ""):
    """
---------------------------------------------------------------------------
Loads a Vertica model and returns the associated object.

Parameters
----------
name: str
    Model Name.
cursor: DBcursor, optional
    Vertica DB cursor.
test_relation: str, optional
    Relation to use to do the testing. All the methods will use this relation 
    for the scoring. If empty, the training relation will be used as testing.

Returns
-------
model
    The model.
    """
    check_types([("name", name, [str],), ("test_relation", test_relation, [str],)])
    cursor = check_cursor(cursor)[0]
    try:
        check_model(name=name, cursor=cursor)
        raise NameError("The model '{}' doesn't exist.".format(name))
    except:
        pass
    try:
        cursor.execute(
            "SELECT GET_MODEL_ATTRIBUTE (USING PARAMETERS model_name = '"
            + name
            + "', attr_name = 'call_string')"
        )
        info = cursor.fetchone()[0].replace("\n", " ")
    except:
        try:
            cursor.execute(
                "SELECT GET_MODEL_SUMMARY (USING PARAMETERS model_name = '"
                + name
                + "')"
            )
            info = cursor.fetchone()[0].replace("\n", " ")
            info = "kmeans(" + info.split("kmeans(")[1]
        except:
            try:
                from verticapy.learn.preprocessing import Normalizer

                model = Normalizer(name, cursor)
                model.param_ = self.get_attr("details")
                model.X = [
                    '"' + item + '"' for item in model.param_.values["column_name"]
                ]
                if "avg" in model.param_.values:
                    model.parameters["method"] = "zscore"
                elif "max" in model.param_.values:
                    model.parameters["method"] = "minmax"
                else:
                    model.parameters["method"] = "robust_zscore"
                return model
            except:
                try:
                    cursor.execute(
                        "SELECT attr_name, value FROM verticapy.attr WHERE LOWER(model_name) = '{}'".format(
                            str_column(name.lower())
                        )
                    )
                    result = cursor.fetchall()
                    model_save = {}
                    for elem in result:
                        ldic = {}
                        try:
                            exec("result_tmp = {}".format(elem[1]), globals(), ldic)
                        except:
                            exec(
                                "result_tmp = '{}'".format(elem[1].replace("'", "''")),
                                globals(),
                                ldic,
                            )
                        result_tmp = ldic["result_tmp"]
                        try:
                            result_tmp = float(result_tmp)
                        except:
                            pass
                        if result_tmp == None:
                            result_tmp = "None"
                        model_save[elem[0]] = result_tmp
                except:
                    raise
                    model_save = {}
                if not (model_save):
                    raise NameError("The model named {} doesn't exist.".format(name))
                if model_save["type"] == "NearestCentroid":
                    from verticapy.learn.neighbors import NearestCentroid

                    model = NearestCentroid(name, cursor, model_save["p"])
                    model.centroids_ = tablesample(model_save["centroids"])
                    model.classes_ = model_save["classes"]
                elif model_save["type"] == "KNeighborsClassifier":
                    from verticapy.learn.neighbors import KNeighborsClassifier

                    model = KNeighborsClassifier(
                        name, cursor, model_save["n_neighbors"], model_save["p"]
                    )
                    model.classes_ = model_save["classes"]
                elif model_save["type"] == "KNeighborsRegressor":
                    from verticapy.learn.neighbors import KNeighborsRegressor

                    model = KNeighborsRegressor(
                        name, cursor, model_save["n_neighbors"], model_save["p"]
                    )
                elif model_save["type"] == "KernelDensity":
                    from verticapy.learn.neighbors import KernelDensity

                    model = KernelDensity(
                        name,
                        cursor,
                        model_save["bandwidth"],
                        model_save["kernel"],
                        model_save["leaf_size"],
                    )
                    model.map_ = tablesample(
                        {"x": model_save["map"][0], "y": model_save["map"][1]}
                    )
                elif model_save["type"] == "LocalOutlierFactor":
                    from verticapy.learn.neighbors import LocalOutlierFactor

                    model = LocalOutlierFactor(
                        name, cursor, model_save["n_neighbors"], model_save["p"]
                    )
                    model.n_errors_ = model_save["n_errors"]
                elif model_save["type"] == "DBSCAN":
                    from verticapy.learn.cluster import DBSCAN

                    model = DBSCAN(
                        name,
                        cursor,
                        model_save["eps"],
                        model_save["min_samples"],
                        model_save["p"],
                    )
                    model.n_cluster_ = model_save["n_cluster"]
                    model.n_noise_ = model_save["n_noise"]
                elif model_save["type"] == "CountVectorizer":
                    from verticapy.learn.preprocessing import CountVectorizer

                    model = CountVectorizer(
                        name,
                        cursor,
                        model_save["lowercase"],
                        model_save["max_df"],
                        model_save["min_df"],
                        model_save["max_features"],
                        model_save["ignore_special"],
                        model_save["max_text_size"],
                    )
                    model.vocabulary_ = model_save["vocabulary"]
                    model.stop_words_ = model_save["stop_words"]
                elif model_save["type"] == "SARIMAX":
                    from verticapy.learn.tsa import SARIMAX

                    model = SARIMAX(
                        name,
                        cursor,
                        model_save["p"],
                        model_save["d"],
                        model_save["q"],
                        model_save["P"],
                        model_save["D"],
                        model_save["Q"],
                        model_save["s"],
                        model_save["tol"],
                        model_save["max_iter"],
                        model_save["solver"],
                        model_save["max_pik"],
                        model_save["papprox_ma"],
                    )
                    model.transform_relation = model_save["transform_relation"]
                    model.coef_ = tablesample(model_save["coef"])
                    model.ma_avg_ = model_save["ma_avg"]
                    if isinstance(model_save["ma_piq"], dict):
                        model.ma_piq_ = tablesample(model_save["ma_piq"])
                    else:
                        model.ma_piq_ = None
                    model.ts = model_save["ts"]
                    model.exogenous = model_save["exogenous"]
                    model.deploy_predict_ = model_save["deploy_predict"]
                elif model_save["type"] == "VAR":
                    from verticapy.learn.tsa import VAR

                    model = VAR(
                        name,
                        cursor,
                        model_save["p"],
                        model_save["tol"],
                        model_save["max_iter"],
                        model_save["solver"],
                    )
                    model.transform_relation = model_save["transform_relation"]
                    model.coef_ = []
                    for i in range(len(model_save["X"])):
                        model.coef_ += [tablesample(model_save["coef_{}".format(i)])]
                    model.ts = model_save["ts"]
                    model.deploy_predict_ = model_save["deploy_predict"]
                    model.X = model_save["X"]
                model.input_relation = model_save["input_relation"]
                model.X = model_save["X"]
                if model_save["type"] in (
                    "KNeighborsRegressor",
                    "KNeighborsClassifier",
                    "NearestCentroid",
                    "SARIMAX",
                ):
                    model.y = model_save["y"]
                    model.test_relation = model_save["test_relation"]
                elif model_save["type"] not in ("CountVectorizer", "VAR"):
                    model.key_columns = model_save["key_columns"]
                return model
    try:
        info = info.split("SELECT ")[1].split("(")
    except:
        info = info.split("(")
    model_type = info[0].lower()
    info = info[1].split(")")[0].replace(" ", "").split("USINGPARAMETERS")
    if model_type == "svm_classifier":
        parameters = "".join(info[1].split("class_weights=")[1].split("'"))
        parameters = parameters[3 : len(parameters)].split(",")
        del parameters[0]
        parameters += [
            "class_weights=" + info[1].split("class_weights=")[1].split("'")[1]
        ]
    elif model_type != "svd":
        parameters = info[1].split(",")
    if model_type != "svd":
        parameters = [item.split("=") for item in parameters]
        parameters_dict = {}
        for item in parameters:
            try:
                parameters_dict[item[0]] = item[1]
            except:
                pass
    info = info[0]
    for elem in parameters_dict:
        if isinstance(parameters_dict[elem], str):
            parameters_dict[elem] = parameters_dict[elem].replace("'", "")
    if model_type == "rf_regressor":
        from verticapy.learn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(
            name,
            cursor,
            int(parameters_dict["ntree"]),
            int(parameters_dict["mtry"]),
            int(parameters_dict["max_breadth"]),
            float(parameters_dict["sampling_size"]),
            int(parameters_dict["max_depth"]),
            int(parameters_dict["min_leaf_size"]),
            float(parameters_dict["min_info_gain"]),
            int(parameters_dict["nbins"]),
        )
    elif model_type == "rf_classifier":
        from verticapy.learn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            name,
            cursor,
            int(parameters_dict["ntree"]),
            int(parameters_dict["mtry"]),
            int(parameters_dict["max_breadth"]),
            float(parameters_dict["sampling_size"]),
            int(parameters_dict["max_depth"]),
            int(parameters_dict["min_leaf_size"]),
            float(parameters_dict["min_info_gain"]),
            int(parameters_dict["nbins"]),
        )
    elif model_type == "xgb_classifier":
        from verticapy.learn.ensemble import XGBoostClassifier

        model = XGBoostClassifier(
            name,
            cursor,
            int(parameters_dict["max_ntree"]),
            int(parameters_dict["max_depth"]),
            int(parameters_dict["nbins"]),
            parameters_dict["objective"],
            parameters_dict["split_proposal_method"],
            float(parameters_dict["epsilon"]),
            float(parameters_dict["learning_rate"]),
            float(parameters_dict["min_split_loss"]),
            float(parameters_dict["weight_reg"]),
            float(parameters_dict["sampling_size"]),
        )
    elif model_type == "xgb_regressor":
        from verticapy.learn.ensemble import XGBoostRegressor

        model = XGBoostRegressor(
            name,
            cursor,
            int(parameters_dict["max_ntree"]),
            int(parameters_dict["max_depth"]),
            int(parameters_dict["nbins"]),
            parameters_dict["objective"],
            parameters_dict["split_proposal_method"],
            float(parameters_dict["epsilon"]),
            float(parameters_dict["learning_rate"]),
            float(parameters_dict["min_split_loss"]),
            float(parameters_dict["weight_reg"]),
            float(parameters_dict["sampling_size"]),
        )
    elif model_type == "logistic_reg":
        from verticapy.learn.linear_model import LogisticRegression

        model = LogisticRegression(
            name,
            cursor,
            parameters_dict["regularization"],
            float(parameters_dict["epsilon"]),
            float(parameters_dict["lambda"]),
            int(parameters_dict["max_iterations"]),
            parameters_dict["optimizer"],
            float(parameters_dict["alpha"]),
        )
    elif model_type == "linear_reg":
        from verticapy.learn.linear_model import (
            LinearRegression,
            Lasso,
            Ridge,
            ElasticNet,
        )

        if parameters_dict["regularization"] == "none":
            model = LinearRegression(
                name,
                cursor,
                float(parameters_dict["epsilon"]),
                int(parameters_dict["max_iterations"]),
                parameters_dict["optimizer"],
            )
        elif parameters_dict["regularization"] == "l1":
            model = Lasso(
                name,
                cursor,
                float(parameters_dict["epsilon"]),
                float(parameters_dict["lambda"]),
                int(parameters_dict["max_iterations"]),
                parameters_dict["optimizer"],
            )
        elif parameters_dict["regularization"] == "l2":
            model = Ridge(
                name,
                cursor,
                float(parameters_dict["epsilon"]),
                float(parameters_dict["lambda"]),
                int(parameters_dict["max_iterations"]),
                parameters_dict["optimizer"],
            )
        else:
            model = ElasticNet(
                name,
                cursor,
                float(parameters_dict["epsilon"]),
                float(parameters_dict["lambda"]),
                int(parameters_dict["max_iterations"]),
                parameters_dict["optimizer"],
                float(parameters_dict["alpha"]),
            )
    elif model_type == "naive_bayes":
        from verticapy.learn.naive_bayes import NaiveBayes

        model = NaiveBayes(name, cursor, float(parameters_dict["alpha"]))
    elif model_type == "svm_regressor":
        from verticapy.learn.svm import LinearSVR

        model = LinearSVR(
            name,
            cursor,
            float(parameters_dict["epsilon"]),
            float(parameters_dict["C"]),
            True,
            float(parameters_dict["intercept_scaling"]),
            parameters_dict["intercept_mode"],
            float(parameters_dict["error_tolerance"]),
            int(parameters_dict["max_iterations"]),
        )
    elif model_type == "svm_classifier":
        from verticapy.learn.svm import LinearSVC

        class_weights = parameters_dict["class_weights"].split(",")
        for idx, elem in enumerate(class_weights):
            try:
                class_weights[idx] = float(class_weights[idx])
            except:
                class_weights[idx] = None
        model = LinearSVC(
            name,
            cursor,
            float(parameters_dict["epsilon"]),
            float(parameters_dict["C"]),
            True,
            float(parameters_dict["intercept_scaling"]),
            parameters_dict["intercept_mode"],
            class_weights,
            int(parameters_dict["max_iterations"]),
        )
    elif model_type == "kmeans":
        from verticapy.learn.cluster import KMeans

        model = KMeans(
            name,
            cursor,
            int(info.split(",")[-1]),
            parameters_dict["init_method"],
            int(parameters_dict["max_iterations"]),
            float(parameters_dict["epsilon"]),
        )
        model.cluster_centers_ = model.get_attr("centers")
        result = model.get_attr("metrics").values["metrics"][0]
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
            float(result.split("Between-Cluster Sum of Squares: ")[1].split("\n")[0]),
            float(result.split("Total Sum of Squares: ")[1].split("\n")[0]),
            float(
                result.split("Total Within-Cluster Sum of Squares: ")[1].split("\n")[0]
            ),
            float(result.split("Between-Cluster Sum of Squares: ")[1].split("\n")[0])
            / float(result.split("Total Sum of Squares: ")[1].split("\n")[0]),
            result.split("Converged: ")[1].split("\n")[0] == "True",
        ]
        model.metrics_ = tablesample(values)
    elif model_type == "bisecting_kmeans":
        from verticapy.learn.cluster import BisectingKMeans

        model = BisectingKMeans(
            name,
            cursor,
            int(info.split(",")[-1]),
            int(parameters_dict["bisection_iterations"]),
            parameters_dict["split_method"],
            int(parameters_dict["min_divisible_cluster_size"]),
            parameters_dict["distance_method"],
            parameters_dict["kmeans_center_init_method"],
            int(parameters_dict["kmeans_max_iterations"]),
            float(parameters_dict["kmeans_epsilon"]),
        )
        model.metrics_ = model.get_attr("Metrics")
        model.cluster_centers_ = model.get_attr("BKTree")
    elif model_type == "pca":
        from verticapy.learn.decomposition import PCA

        model = PCA(name, cursor, 0, bool(parameters_dict["scale"]))
        model.components_ = model.get_attr("principal_components")
        model.explained_variance_ = model.get_attr("singular_values")
        model.mean_ = model.get_attr("columns")
    elif model_type == "svd":
        from verticapy.learn.decomposition import SVD

        model = SVD(name, cursor)
        model.singular_values_ = model.get_attr("right_singular_vectors")
        model.explained_variance_ = model.get_attr("singular_values")
    elif model_type == "one_hot_encoder_fit":
        from verticapy.learn.preprocessing import OneHotEncoder

        model = OneHotEncoder(name, cursor)
        try:
            model.param_ = to_tablesample(
                query="SELECT category_name, category_level::varchar, category_level_index FROM (SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'integer_categories')) VERTICAPY_SUBTABLE UNION ALL SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'varchar_categories')".format(
                    model.name, model.name
                ),
                cursor=model.cursor,
            )
        except:
            try:
                model.param_ = to_tablesample(
                    query="SELECT category_name, category_level::varchar, category_level_index FROM (SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'integer_categories')) VERTICAPY_SUBTABLE".format(
                        model.name
                    ),
                    cursor=model.cursor,
                )
            except:
                model.param_ = model.get_attr("varchar_categories")
    model.input_relation = info.split(",")[1].replace("'", "").replace("\\", "")
    model.test_relation = test_relation if (test_relation) else model.input_relation
    if model_type not in ("kmeans", "pca", "svd", "one_hot_encoder_fit"):
        model.X = info.split(",")[3 : len(info.split(","))]
        model.X = [item.replace("'", "").replace("\\", "") for item in model.X]
        model.y = info.split(",")[2].replace("'", "").replace("\\", "")
    elif model_type in (
        "svd",
        "pca",
        "one_hot_encoder_fit",
        "normalizer",
        "kmeans",
        "bisectingkmeans",
    ):
        model.X = info.split(",")[2 : len(info.split(","))]
        model.X = [item.replace("'", "").replace("\\", "") for item in model.X]
    if model_type in ("naive_bayes", "rf_classifier", "xgb_classifier"):
        try:
            cursor.execute(
                "SELECT DISTINCT {} FROM {} WHERE {} IS NOT NULL ORDER BY 1".format(
                    model.y, model.input_relation, model.y
                )
            )
            classes = cursor.fetchall()
            model.classes_ = [item[0] for item in classes]
        except:
            model.classes_ = None
    elif model_type in ("svm_classifier", "logistic_reg"):
        model.classes_ = [0, 1]
    if model_type in ("svm_classifier", "svm_regressor", "logistic_reg", "linear_reg",):
        model.coef_ = model.get_attr("details")
    return model
