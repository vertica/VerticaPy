"""
Copyright  (c)  2018-2024 Open Text  or  one  of its
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
from typing import Literal, Optional

from verticapy._typing import NoneType, SQLRelation
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
    PoissonRegressor,
    Ridge,
)
from verticapy.machine_learning.vertica.naive_bayes import NaiveBayes
from verticapy.machine_learning.vertica.pmml import PMMLModel
from verticapy.machine_learning.vertica.preprocessing import Scaler, OneHotEncoder
from verticapy.machine_learning.vertica.svm import LinearSVC, LinearSVR
from verticapy.machine_learning.vertica.tensorflow import TensorFlowModel
from verticapy.machine_learning.vertica.tsa import ARIMA, AR, MA


@save_verticapy_logs
def export_models(
    name: str,
    path: str,
    kind: Literal["pmml", "vertica", "vertica_models", "tensorflow", "tf", None] = None,
) -> bool:
    """
    Exports machine learning models.

    Parameters
    ----------
    name: str
        Specifies which models to export as follows:

        ``[schema.]{ model-name | * }``

        where schema optionally specifies to export
        models from the specified schema. If omitted,
        ``export_models`` uses the default schema.
        Supply * (asterisk) to export all models from
        the schema.
    path: str
        Absolute path of an output directory to store
        the exported models.

        .. warning::

            This function operates solely on the server
            side and is not accessible locally.
            The 'path' provided should match the location
            where the file(s) will be exported on the server.
    kind: str, optional
        The category of models to export, one of the
        following:

            - vertica
            - pmml
            - tensorflow

        ``export_models`` exports models of the specified
        category according to the scope of the export
        operation—that is, whether it applies to a single
        model, or to all models within a schema.

        If you omit this parameter, ``export_models``
        exports the model, or models in the specified
        schema, according to their model type.

    Returns
    -------
    bool
        True if the model(s) was(were) successfully
        exported.
    """
    return VerticaModel.export_models(name=name, path=path, kind=kind)


@save_verticapy_logs
def import_models(
    path: str,
    schema: Optional[str] = None,
    kind: Literal["pmml", "vertica", "vertica_models", "tensorflow", "tf", None] = None,
) -> bool:
    """
    Imports machine learning models.

    Parameters
    ----------
    path: str
        The absolute path of the location from which
        to import models, one of the following:

         - The directory of a single model:
            ``path/model-directory``
         - The parent directory of multiple model
         directories:
            ``parent-dir-path/*``

        .. warning::

            This function only operates on the server
            side and is not accessible locally.
            The 'path' should correspond to the location
            of the file(s) on the server. Please make
            sure you have successfully transferred your
            file(s) to the server.
    schema: str, optional
        An existing schema where the machine learning
        models are imported. If omitted, models are
        imported to the default schema.
        ``import_models`` extracts the name of the
        imported model from its metadata.json file,
        if it exists. Otherwise, the function uses
        the name of the model directory.
    kind: str, optional
        The category of models to import, one of the
        following:

            - vertica
            - pmml
            - tensorflow

        This parameter is required if the model directory
        has no metadata.json file. ``import_models``
        returns with an error if one of the following
        cases is true:

            - No category is specified and the model
              directory has no metadata.json.
            - The specified category does not match the
              model type.

        .. note::

            If the category is `tensorflow`, ``import_models``
            only imports the following files from the model
            directory:

                - model-name.pb
                - model-name.json
                - model-name.pbtxt (optional)

    Returns
    -------
    bool
        True if the model(s) was(were) successfully
        imported.
    """
    return VerticaModel.import_models(path=path, schema=schema, kind=kind)


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
    res = VerticaModel.does_model_exists(
        name=name, raise_error=False, return_model_type=True
    )
    if isinstance(res, NoneType):
        raise NameError(f"The model '{name}' doesn't exist.")
    model_category, model_type = res
    model_category = model_category.lower()
    if model_category == "pmml":
        return PMMLModel(name)
    elif model_category == "tensorflow":
        return TensorFlowModel(name)
    schema, model_name = schema_relation(name)
    schema, model_name = schema[1:-1], name[1:-1]
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
        "arima": ARIMA,
        "autoregressor": AR,
        "moving_average": MA,
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
        "poisson_reg": PoissonRegressor,
        "kmeans": KMeans,
        "kprototypes": KPrototypes,
        "bisecting_kmeans": BisectingKMeans,
        "pca": PCA,
        "svd": SVD,
        "one_hot_encoder_fit": OneHotEncoder,
    }
    model = lookup_table[model_type](name)
    if model_type != "svd":
        # Variables used in the CALL STRING
        true, false = True, False
        squarederror = "squarederror"
        crossentropy = "crossentropy"
        ols = "ols"
        hr = "hr"
        linear_interpolation = "linear_interpolation"
        zero = "zero"
        error = "error"
        drop = "drop"
        if "method=yule-walker," in parameters:
            parameters = parameters.replace(
                "method=yule-walker,", "method='yule-walker',"
            )
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
        if '"' in info[3]:
            model.X = eval("[" + info[3] + "]")
        else:
            model.X = info[3].split(",")
        model.test_relation = test_relation if (test_relation) else model.input_relation
    elif model._model_category == "TIMESERIES":
        model.y = info[2]
        model.ts = info[3]
        model.test_relation = test_relation if (test_relation) else model.input_relation
        if model._model_type == "ARIMA":
            p = int(model.get_vertica_attributes("p")["p"][0])
            d = int(model.get_vertica_attributes("d")["d"][0])
            q = int(model.get_vertica_attributes("q")["q"][0])
            model.set_params({"order": (p, d, q)})
    else:
        model.X = eval("[" + info[2] + "]")
    model._compute_attributes()
    return model
