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
        Some automated functions depend on the
        input relation. If the load_model function
        cannot  find the  input relation from  the
        call string, you should fill it manually.
    test_relation: str, optional
        Relation used for testing. All the methods
        use this  relation for scoring. If empty,
        the training relation is used for testing.

    Returns
    -------
    model
        The model.
    """
    model_type = VerticaModel.does_model_exists(
        name=name, raise_error=False, return_model_type=True
    )
    schema, model_name = schema_relation(name)
    schema, model_name = schema[1:-1], name[1:-1]
    if not model_type:
        raise NameError(f"The model '{name}' doesn't exist.")
    if model_type.lower() in (
        "kmeans",
        "kprototypes",
    ):
        info = (
            _executeSQL(
                query=f"""
                SELECT 
                    /*+LABEL('learn.tools.load_model')*/ 
                    GET_MODEL_SUMMARY 
                    (USING PARAMETERS 
                    model_name = '{name}')""",
                method="fetchfirstelem",
                print_time_sql=False,
            )
            .replace("\n", " ")
            .strip()
        )
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
        info = (
            _executeSQL(
                query=f"""
                SELECT 
                    /*+LABEL('learn.tools.load_model')*/ 
                    GET_MODEL_ATTRIBUTE 
                    (USING PARAMETERS 
                    model_name = '{name}',
                    attr_name = 'call_string')""",
                method="fetchfirstelem",
                print_time_sql=False,
            )
            .replace("\n", " ")
            .strip()
        )
    if info.lower().startswith("select "):
        info = info[7:].split("(")
    else:
        info = info.split("(")
    model_type = info[0].lower()
    info = ")".join("(".join(info[1:]).split(")")[0:-1])
    if " USING PARAMETERS " in info:
        info = info.split(" USING PARAMETERS ")
        parameters = info[1]
        info = info[0]
    info = eval("[" + info + "]")
    lookup_table = {
        "rf_regressor": RandomForestRegressor,
        "rf_classifier": RandomForestClassifier,
        "iforest": IsolationForest,
        "xgb_classifier": XGBClassifier,
        "xgb_regressor": XGBRegressor,
        "logistic_reg": LogisticRegression,
        "naive_bayes": NaiveBayes,
        "svm_regressor": LinearSVR,
        "svm_classifier": LinearSVC,
        "linear_reg": LinearRegression,
        "kmeans": KMeans,
        "kprototypes": KPrototypes,
        "bisecting_kmeans": BisectingKMeans,
        "pca": PCA,
        "svd": SVD,
        "one_hot_encoder_fit": OneHotEncoder,
    }
    model = lookup_table[model_type](name)
    if model_type != "svd":
        true, false = True, False
        squarederror = "squarederror"
        crossentropy = "crossentropy"
        if " lambda=" in parameters:
            parameters = parameters.replace(" lambda=", " C=")
        try:
            parameters = eval("model._get_verticapy_param_dict(" + parameters + ")")
        except SyntaxError:
            parameters = parameters.replace("''", ' """ ')
            parameters = eval("model._get_verticapy_param_dict(" + parameters + ")")
        if model_type in ("kmeans", "bisecting_kmeans", "kprototypes"):
            parameters["n_cluster"] = info[-1]
        if model_type == "linear_reg":
            lr_lookup_table = {
                "none": LinearRegression,
                "l1": Lasso,
                "l2": Ridge,
                "enet": ElasticNet,
            }
            model = lr_lookup_table[parameters["penalty"]](name)
    else:
        parameters = {}
    model.set_params(**parameters)
    model.input_relation = input_relation if (input_relation) else info[1]
    if model._model_category == "SUPERVISED":
        model.y = info[2]
        model.X = eval("[" + info[3] + "]")
        model.test_relation = test_relation if (test_relation) else model.input_relation
    else:
        model.X = eval("[" + info[2] + "]")
    model._compute_attributes()
    return model
