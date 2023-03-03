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
from verticapy._typing import SQLRelation
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import schema_relation
from verticapy._utils._sql._sys import _executeSQL
from verticapy._utils._sql._vertica_version import vertica_version

from verticapy.core.tablesample.base import TableSample

from verticapy.machine_learning.sys.model_checking import does_model_exist
from verticapy.machine_learning.vertica.base import VerticaModel
from verticapy.machine_learning.vertica.cluster import (
    BisectingKMeans,
    KMeans,
    KPrototypes,
)
from verticapy.machine_learning.vertica.decomposition import PCA, SVD
from verticapy.machine_learning.vertica.ensemble import (
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
    XGBClassifier,
    XGBRegressor,
)
from verticapy.machine_learning.vertica.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from verticapy.machine_learning.vertica.model_management import load_model
from verticapy.machine_learning.vertica.naive_bayes import NaiveBayes
from verticapy.machine_learning.vertica.preprocessing import Scaler, OneHotEncoder
from verticapy.machine_learning.vertica.svm import LinearSVC, LinearSVR


@save_verticapy_logs
def load_model(
    name: str, input_relation: SQLRelation = "", test_relation: SQLRelation = ""
) -> VerticaModel:
    """
    Loads a Vertica model and returns the associated 
    object.

    Parameters
    ----------
    name: str
        Model Name.
    input_relation: str, optional
        Some automated functions may depend on the 
        input relation. If the load_model function 
        cannot  find the  input relation from  the 
        call string, you should fill it manually.
    test_relation: str, optional
        Relation to use to do the testing. All the 
        methods  will  use this  relation for  the 
        scoring.  If empty, the training  relation 
        will be used as testing.

    Returns
    -------
    model
        The model.
    """
    model_type = does_model_exist(name=name, raise_error=False, return_model_type=True)
    schema, model_name = schema_relation(name)
    schema, model_name = schema[1:-1], name[1:-1]
    if not (model_type):
        raise NameError(f"The model '{name}' doesn't exist.")
    if model_type.lower() in ("kmeans", "kprototypes",):
        info = _executeSQL(
            query=f"""
                SELECT 
                    /*+LABEL('learn.tools.load_model')*/ 
                    GET_MODEL_SUMMARY 
                    (USING PARAMETERS 
                    model_name = '{name}')""",
            method="fetchfirstelem",
            print_time_sql=False,
        ).replace("\n", " ")
        mtype = model_type.lower() + "("
        info = mtype + info.split(mtype)[1]
    elif model_type.lower() == "normalize_fit":
        model = Scaler(name)
        param = model.get_vertica_attributes("details")
        model.X = ['"' + item + '"' for item in param.values["column_name"]]
        if "avg" in param.values:
            model.parameters["method"] = "zscore"
        elif "max" in param.values:
            model.parameters["method"] = "minmax"
        else:
            model.parameters["method"] = "robust_zscore"
        model._compute_attributes()
        return model
    else:
        info = _executeSQL(
            query=f"""
                SELECT 
                    /*+LABEL('learn.tools.load_model')*/ 
                    GET_MODEL_ATTRIBUTE 
                    (USING PARAMETERS 
                    model_name = '{name}',
                    attr_name = 'call_string')""",
            method="fetchfirstelem",
            print_time_sql=False,
        ).replace("\n", " ")
    if "SELECT " in info:
        info = info.split("SELECT ")[1].split("(")
    else:
        info = info.split("(")
    model_type = info[0].lower()
    info = info[1].split(")")[0].replace(" ", "").split("USINGPARAMETERS")
    if (
        model_type == "svm_classifier"
        and "class_weights='none'" not in " ".join(info).lower()
    ):
        parameters = "".join(info[1].split("class_weights=")[1].split("'"))
        parameters = parameters[3 : len(parameters)].split(",")
        del parameters[0]
        parameters += [
            "class_weights=" + info[1].split("class_weights=")[1].split("'")[1]
        ]
    elif model_type != "svd":
        parameters = info[1].split(",")
    else:
        parameters = []
    parameters = [item.split("=") for item in parameters]
    parameters_dict = {}
    for item in parameters:
        if len(item) > 1:
            parameters_dict[item[0]] = item[1]
    info = info[0]
    for elem in parameters_dict:
        if isinstance(parameters_dict[elem], str):
            parameters_dict[elem] = parameters_dict[elem].replace("'", "")
    if "split_proposal_method" in parameters_dict:
        split_proposal_method = parameters_dict["split_proposal_method"]
    else:
        split_proposal_method = "global"
    if "epsilon" in parameters_dict:
        epsilon = parameters_dict["epsilon"]
    else:
        epsilon = 0.001
    if model_type == "rf_regressor":
        model = RandomForestRegressor(
            name,
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
        model = RandomForestClassifier(
            name,
            int(parameters_dict["ntree"]),
            int(parameters_dict["mtry"]),
            int(parameters_dict["max_breadth"]),
            float(parameters_dict["sampling_size"]),
            int(parameters_dict["max_depth"]),
            int(parameters_dict["min_leaf_size"]),
            float(parameters_dict["min_info_gain"]),
            int(parameters_dict["nbins"]),
        )
    elif model_type == "iforest":
        model = IsolationForest(
            name,
            int(parameters_dict["ntree"]),
            int(parameters_dict["max_depth"]),
            int(parameters_dict["nbins"]),
            float(parameters_dict["sampling_size"]),
            float(parameters_dict["col_sample_by_tree"]),
        )
    elif model_type == "xgb_classifier":
        model = XGBClassifier(
            name,
            int(parameters_dict["max_ntree"]),
            int(parameters_dict["max_depth"]),
            int(parameters_dict["nbins"]),
            split_proposal_method,
            float(epsilon),
            float(parameters_dict["learning_rate"]),
            float(parameters_dict["min_split_loss"]),
            float(parameters_dict["weight_reg"]),
            float(parameters_dict["sampling_size"]),
        )
    elif model_type == "xgb_regressor":
        model = XGBRegressor(
            name,
            int(parameters_dict["max_ntree"]),
            int(parameters_dict["max_depth"]),
            int(parameters_dict["nbins"]),
            split_proposal_method,
            float(epsilon),
            float(parameters_dict["learning_rate"]),
            float(parameters_dict["min_split_loss"]),
            float(parameters_dict["weight_reg"]),
            float(parameters_dict["sampling_size"]),
        )
    elif model_type == "logistic_reg":
        model = LogisticRegression(
            name,
            parameters_dict["regularization"],
            float(parameters_dict["epsilon"]),
            float(parameters_dict["lambda"]),
            int(parameters_dict["max_iterations"]),
            parameters_dict["optimizer"],
            float(parameters_dict["alpha"]),
        )
    elif model_type == "linear_reg":
        if parameters_dict["regularization"] == "none":
            model = LinearRegression(
                name,
                float(parameters_dict["epsilon"]),
                int(parameters_dict["max_iterations"]),
                parameters_dict["optimizer"],
            )
        elif parameters_dict["regularization"] == "l1":
            model = Lasso(
                name,
                float(parameters_dict["epsilon"]),
                float(parameters_dict["lambda"]),
                int(parameters_dict["max_iterations"]),
                parameters_dict["optimizer"],
            )
        elif parameters_dict["regularization"] == "l2":
            model = Ridge(
                name,
                float(parameters_dict["epsilon"]),
                float(parameters_dict["lambda"]),
                int(parameters_dict["max_iterations"]),
                parameters_dict["optimizer"],
            )
        else:
            model = ElasticNet(
                name,
                float(parameters_dict["epsilon"]),
                float(parameters_dict["lambda"]),
                int(parameters_dict["max_iterations"]),
                parameters_dict["optimizer"],
                float(parameters_dict["alpha"]),
            )
    elif model_type == "naive_bayes":
        model = NaiveBayes(name, float(parameters_dict["alpha"]))
    elif model_type == "svm_regressor":
        model = LinearSVR(
            name,
            float(parameters_dict["epsilon"]),
            float(parameters_dict["C"]),
            True,
            float(parameters_dict["intercept_scaling"]),
            parameters_dict["intercept_mode"],
            float(parameters_dict["error_tolerance"]),
            int(parameters_dict["max_iterations"]),
        )
    elif model_type == "svm_classifier":
        class_weights = parameters_dict["class_weights"].split(",")
        for idx, elem in enumerate(class_weights):
            try:
                class_weights[idx] = float(class_weights[idx])
            except:
                class_weights[idx] = None
        model = LinearSVC(
            name,
            float(parameters_dict["epsilon"]),
            float(parameters_dict["C"]),
            True,
            float(parameters_dict["intercept_scaling"]),
            parameters_dict["intercept_mode"],
            class_weights,
            int(parameters_dict["max_iterations"]),
        )
    elif model_type == "kmeans":
        model = KMeans(
            name,
            int(info.split(",")[-1]),
            parameters_dict["init_method"],
            int(parameters_dict["max_iterations"]),
            float(parameters_dict["epsilon"]),
        )
    elif model_type == "kprototypes":
        model = KPrototypes(
            name,
            int(info.split(",")[-1]),
            parameters_dict["init_method"],
            int(parameters_dict["max_iterations"]),
            float(parameters_dict["epsilon"]),
            float(parameters_dict["gamma"]),
        )
    elif model_type == "bisecting_kmeans":
        model = BisectingKMeans(
            name,
            int(info.split(",")[-1]),
            int(parameters_dict["bisection_iterations"]),
            parameters_dict["split_method"],
            int(parameters_dict["min_divisible_cluster_size"]),
            parameters_dict["distance_method"],
            parameters_dict["kmeans_center_init_method"],
            int(parameters_dict["kmeans_max_iterations"]),
            float(parameters_dict["kmeans_epsilon"]),
        )
    elif model_type == "pca":
        model = PCA(name, 0, bool(parameters_dict["scale"]))
    elif model_type == "svd":
        model = SVD(name)
    elif model_type == "one_hot_encoder_fit":
        model = OneHotEncoder(name)
    if not (input_relation):
        model.input_relation = info.split(",")[1].replace("'", "").replace("\\", "")
    else:
        model.input_relation = input_relation
    model.test_relation = test_relation if (test_relation) else model.input_relation
    if model_type not in (
        "kmeans",
        "kprototypes",
        "pca",
        "svd",
        "one_hot_encoder_fit",
        "bisecting_kmeans",
        "iforest",
        "normalizer",
    ):
        start = 3
        model.y = info.split(",")[2].replace("'", "").replace("\\", "")
    else:
        start = 2
    end = len(info.split(","))
    if model_type in ("bisecting_kmeans",):
        end -= 1
    model.X = info.split(",")[start:end]
    model.X = [item.replace("'", "").replace("\\", "") for item in model.X]
    if model_type in ("xgb_classifier", "xgb_regressor"):
        v = vertica_version()
        v = v[0] > 11 or (v[0] == 11 and (v[1] >= 1 or v[2] >= 1))
        if v:
            model.set_params(
                {
                    "col_sample_by_tree": float(parameters_dict["col_sample_by_tree"]),
                    "col_sample_by_node": float(parameters_dict["col_sample_by_node"]),
                }
            )
    model._compute_attributes()
    return model
