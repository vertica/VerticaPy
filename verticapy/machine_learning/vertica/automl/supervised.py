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
from typing import Literal, Union
from tqdm.auto import tqdm

from verticapy._config.config import OPTIONS
from verticapy._utils._collect import save_verticapy_logs
from verticapy._utils._gen import gen_tmp_name
from verticapy._utils._sql._format import schema_relation
from verticapy._version import vertica_version
from verticapy.errors import ParameterError

from verticapy.core.TableSample.base import TableSample
from verticapy.core.vdataframe.base import vDataFrame

from verticapy.plotting._matplotlib.mlplot import plot_bubble_ml, plot_stepwise_ml

from verticapy.machine_learning._utils import reverse_score
from verticapy.machine_learning.vertica.automl import AutoDataPrep
from verticapy.machine_learning.vertica.base import vModel
from verticapy.machine_learning.vertica.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    XGBoostClassifier,
    XGBoostRegressor,
)
from verticapy.machine_learning.vertica.naive_bayes import NaiveBayes
from verticapy.machine_learning.vertica.linear_model import (
    LogisticRegression,
    LinearRegression,
    ElasticNet,
    Lasso,
    Ridge,
)
from verticapy.machine_learning.model_selection import (
    gen_params_grid,
    grid_search_cv,
    stepwise,
)
from verticapy.machine_learning.vertica.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
)
from verticapy.machine_learning.vertica.svm import LinearSVC, LinearSVR


class AutoML(vModel):
    """
Tests multiple models to find those that maximize the input score.

Parameters
----------
name: str
    Name of the model.
estimator: list / 'native' / 'all' / 'fast' / object
    List of Vertica estimators with a fit method.
    Alternatively, you can specify 'native' for all native Vertica models,
    'all' for all VerticaPy models and 'fast' for quick modeling.
estimator_type: str, optional
    Estimator Type.
        auto      : Automatically detects the estimator type.
        regressor : The estimator will be used to perform a regression.
        binary    : The estimator will be used to perform a binary classification.
        multi     : The estimator will be used to perform a multiclass classification.
metric: str, optional
    Metric used to do the model evaluation.
        auto: logloss for classification & rmse for regression.
    For Classification:
        accuracy    : Accuracy
        auc         : Area Under the Curve (ROC)
        bm          : Informedness = tpr + tnr - 1
        csi         : Critical Success Index = tp / (tp + fn + fp)
        f1          : F1 Score 
        logloss     : Log Loss
        mcc         : Matthews Correlation Coefficient 
        mk          : Markedness = ppv + npv - 1
        npv         : Negative Predictive Value = tn / (tn + fn)
        prc_auc     : Area Under the Curve (PRC)
        precision   : Precision = tp / (tp + fp)
        recall      : Recall = tp / (tp + fn)
        specificity : Specificity = tn / (tn + fp)
    For Regression:
        max    : Max error
        mae    : Mean absolute error
        median : Median absolute error
        mse    : Mean squared error
        msle   : Mean squared log error
        r2     : R-squared coefficient
        r2a    : R2 adjusted
        rmse   : Root-mean-squared error
        var    : Explained variance
cv: int, optional
    Number of folds.
pos_label: int/float/str, optional
    The main class to be considered as positive (classification only).
cutoff: float, optional
    The model cutoff (classification only).
nbins: int, optional
    Number of bins used to compute the different parameter categories.
lmax: int, optional
    Maximum length of each parameter list.
optimized_grid: int, optional
    If set to 0, the randomness is based on the input parameters.
    If set to 1, the randomness is limited to some parameters while others
    are picked based on a default grid.
    If set to 2, no randomness is used and a default grid is returned.
stepwise: bool, optional
    If True, the stepwise algorithm will be used to determine the
    final model list of parameters.
stepwise_criterion: str, optional
    Criterion used when doing the final estimator stepwise.
        aic : Akaike’s information criterion
        bic : Bayesian information criterion
stepwise_direction: str, optional
    Which direction to start the stepwise search. Can be done 'backward' or 'forward'.
stepwise_max_steps: int, optional
    The maximum number of steps to be considered when doing the final estimator 
    stepwise.
x_order: str, optional
    Method for preprocessing X before using the stepwise algorithm.
        pearson  : X is ordered based on the Pearson's correlation coefficient.
        spearman : X is ordered based on Spearman's rank correlation coefficient.
        random   : Shuffles the vector X before applying the stepwise algorithm.
        none     : Does not change the order of X.
preprocess_data: bool, optional
    If True, the data will be preprocessed.
preprocess_dict: dict, optional
    Dictionary to pass to the AutoDataPrep class in order to 
    preprocess the data before the clustering.
print_info: bool
    If True, prints the model information at each step.

Attributes
----------
preprocess_: object
    Model used to preprocess the data.
best_model_: object
    Most efficient models found during the search.
model_grid_ : TableSample
    Grid containing the different models information.
    """

    @save_verticapy_logs
    def __init__(
        self,
        name: str,
        estimator: Union[list, str] = "fast",
        estimator_type: Literal["auto", "regressor", "binary", "multi"] = "auto",
        metric: str = "auto",
        cv: int = 3,
        pos_label: Union[int, float, str] = None,
        cutoff: float = -1,
        nbins: int = 100,
        lmax: int = 5,
        optimized_grid: int = 2,
        stepwise: bool = True,
        stepwise_criterion: Literal["aic", "bic"] = "aic",
        stepwise_direction: Literal["forward", "backward"] = "backward",
        stepwise_max_steps: int = 100,
        stepwise_x_order: Literal["pearson", "spearman", "random", "none"] = "pearson",
        preprocess_data: bool = True,
        preprocess_dict: dict = {"identify_ts": False},
        print_info: bool = True,
    ):
        assert optimized_grid in [0, 1, 2], ParameterError(
            "Optimized Grid must be an integer between 0 and 2."
        )
        self.type, self.name = "AutoML", name
        self.parameters = {
            "estimator": estimator,
            "estimator_type": estimator_type,
            "metric": metric,
            "cv": cv,
            "pos_label": pos_label,
            "cutoff": cutoff,
            "nbins": nbins,
            "lmax": lmax,
            "optimized_grid": optimized_grid,
            "print_info": print_info,
            "stepwise": stepwise,
            "stepwise_criterion": stepwise_criterion,
            "stepwise_direction": stepwise_direction,
            "stepwise_max_steps": stepwise_max_steps,
            "stepwise_x_order": stepwise_x_order,
            "preprocess_data": preprocess_data,
            "preprocess_dict": preprocess_dict,
        }

    def fit(self, input_relation: Union[str, vDataFrame], X: list = [], y: str = ""):
        """
    Trains the model.

    Parameters
    ----------
    input_relation: str/vDataFrame
        Training Relation.
    X: list, optional
        List of the predictors.
    y: str, optional
        Response column.
    Returns
    -------
    object
        model grid
        """
        if OPTIONS["overwrite_model"]:
            self.drop()
        else:
            does_model_exist(name=self.name, raise_error=True)
        if not (X):
            if not (y):
                exclude_columns = []
            else:
                exclude_columns = [y]
            if not (isinstance(input_relation, vDataFrame)):
                X = vDataFrame(input_relation).get_columns(
                    exclude_columns=exclude_columns
                )
            else:
                X = input_relation.get_columns(exclude_columns=exclude_columns)
        if isinstance(self.parameters["estimator"], str):
            v = vertica_version()
            self.parameters["estimator"] = self.parameters["estimator"].lower()
            modeltype = None
            estimator_method = self.parameters["estimator"]
            if not (isinstance(input_relation, vDataFrame)):
                vdf = vDataFrame(input_relation)
            else:
                vdf = input_relation
            if self.parameters["estimator_type"].lower() == "binary" or (
                self.parameters["estimator_type"].lower() == "auto"
                and sorted(vdf[y].distinct()) == [0, 1]
            ):
                self.parameters["estimator_type"] = "binary"
                self.parameters["estimator"] = [
                    LogisticRegression(self.name),
                    NaiveBayes(self.name),
                ]
                if estimator_method in ("native", "all"):
                    if v[0] > 10 or (v[0] == 10 and v[1] >= 1):
                        self.parameters["estimator"] += [XGBoostClassifier(self.name)]
                    if v[0] >= 9:
                        self.parameters["estimator"] += [
                            LinearSVC(self.name),
                            RandomForestClassifier(self.name),
                        ]
                if estimator_method == "all":
                    self.parameters["estimator"] += [
                        KNeighborsClassifier(self.name),
                        NearestCentroid(self.name),
                    ]
            elif self.parameters["estimator_type"].lower() == "regressor" or (
                self.parameters["estimator_type"].lower() == "auto" and vdf[y].isnum()
            ):
                self.parameters["estimator_type"] = "regressor"
                self.parameters["estimator"] = [
                    LinearRegression(self.name),
                    ElasticNet(self.name),
                    Ridge(self.name),
                    Lasso(self.name),
                ]
                if estimator_method in ("native", "all"):
                    if v[0] > 10 or (v[0] == 10 and v[1] >= 1):
                        self.parameters["estimator"] += [XGBoostRegressor(self.name)]
                    if v[0] >= 9:
                        self.parameters["estimator"] += [
                            LinearSVR(self.name),
                            RandomForestRegressor(self.name),
                        ]
                if estimator_method == "all":
                    self.parameters["estimator"] += [KNeighborsRegressor(self.name)]
            elif self.parameters["estimator_type"].lower() in ("multi", "auto"):
                self.parameters["estimator_type"] = "multi"
                self.parameters["estimator"] = [NaiveBayes(self.name)]
                if estimator_method in ("native", "all"):
                    if v[0] >= 10 and v[1] >= 1:
                        self.parameters["estimator"] += [XGBoostClassifier(self.name)]
                    if v[0] >= 9:
                        self.parameters["estimator"] += [
                            RandomForestClassifier(self.name)
                        ]
                if estimator_method == "all":
                    self.parameters["estimator"] += [
                        KNeighborsClassifier(self.name),
                        NearestCentroid(self.name),
                    ]
            else:
                raise ParameterError(
                    f"Parameter 'estimator_type' must be in auto|binary|multi|regressor. Found {estimator_type}."
                )
        elif isinstance(
            self.parameters["estimator"],
            (
                RandomForestRegressor,
                RandomForestClassifier,
                XGBoostRegressor,
                XGBoostClassifier,
                NaiveBayes,
                LinearRegression,
                ElasticNet,
                Lasso,
                Ridge,
                LogisticRegression,
                KNeighborsRegressor,
                KNeighborsClassifier,
                NearestCentroid,
                LinearSVC,
                LinearSVR,
            ),
        ):
            self.parameters["estimator"] = [self.parameters["estimator"]]
        else:
            for elem in self.parameters["estimator"]:
                assert isinstance(
                    elem,
                    (
                        RandomForestRegressor,
                        RandomForestClassifier,
                        XGBoostRegressor,
                        XGBoostClassifier,
                        NaiveBayes,
                        LinearRegression,
                        ElasticNet,
                        Lasso,
                        Ridge,
                        LogisticRegression,
                        KNeighborsRegressor,
                        KNeighborsClassifier,
                        NearestCentroid,
                        LinearSVC,
                        LinearSVR,
                    ),
                ), ParameterError(
                    f"estimator must be a list of VerticaPy estimators. Found {elem}."
                )
        if self.parameters["estimator_type"] == "auto":
            self.parameters["estimator_type"] = self.parameters["estimator"][0].type
        for elem in self.parameters["estimator"]:
            assert (
                self.parameters["estimator_type"] in ("binary", "multi")
                and elem.MODEL_SUBTYPE == "CLASSIFIER"
                or self.parameters["estimator_type"] == "regressor"
                and elem.MODEL_SUBTYPE == "REGRESSOR"
            ), ParameterError(
                f"Incorrect list for parameter 'estimator'. Expected type '{self.parameters['estimator_type']}', found type '{elem.MODEL_SUBTYPE}'."
            )
        if (
            self.parameters["estimator_type"] == "regressor"
            and self.parameters["metric"] == "auto"
        ):
            self.parameters["metric"] = "rmse"
        elif self.parameters["metric"] == "auto":
            self.parameters["metric"] = "logloss"
        result = TableSample(
            {
                "model_type": [],
                "parameters": [],
                "avg_score": [],
                "avg_train_score": [],
                "avg_time": [],
                "score_std": [],
                "score_train_std": [],
                "model_class": [],
            }
        )
        if self.parameters["preprocess_data"]:
            schema, name = schema_relation(self.name)
            name = gen_tmp_name(schema=schema, name="autodataprep")
            model_preprocess = AutoDataPrep(
                name=name, **self.parameters["preprocess_dict"]
            )
            input_relation = model_preprocess.fit(input_relation, X=X)
            X = [elem for elem in model_preprocess.X_out]
            self.preprocess_ = model_preprocess
        else:
            self.preprocess_ = None
        if self.parameters["print_info"]:
            print(f"\033[1m\033[4mStarting AutoML\033[0m\033[0m\n")
        if OPTIONS["tqdm"] and self.parameters["print_info"]:
            loop = tqdm(self.parameters["estimator"])
        else:
            loop = self.parameters["estimator"]
        for elem in loop:
            if self.parameters["print_info"]:
                print(
                    f"\n\033[1m\033[4mTesting Model - {str(elem.__class__).split('.')[-1][:-2]}\033[0m\033[0m\n"
                )
            param_grid = gen_params_grid(
                elem,
                self.parameters["nbins"],
                len(X),
                self.parameters["lmax"],
                self.parameters["optimized_grid"],
            )
            gs = grid_search_cv(
                elem,
                param_grid,
                input_relation,
                X,
                y,
                self.parameters["metric"],
                self.parameters["cv"],
                self.parameters["pos_label"],
                self.parameters["cutoff"],
                True,
                "no_print",
                self.parameters["print_info"],
            )
            if (
                gs["parameters"] != []
                and gs["avg_score"] != []
                and gs["avg_train_score"] != []
                and gs["avg_time"] != []
                and gs["score_std"] != []
                and gs["score_train_std"] != []
            ):
                result.values["model_type"] += [
                    str(elem.__class__).split(".")[-1][:-2]
                ] * len(gs["parameters"])
                result.values["parameters"] += gs["parameters"]
                result.values["avg_score"] += gs["avg_score"]
                result.values["avg_train_score"] += gs["avg_train_score"]
                result.values["avg_time"] += gs["avg_time"]
                result.values["score_std"] += gs["score_std"]
                result.values["score_train_std"] += gs["score_train_std"]
                result.values["model_class"] += [elem.__class__] * len(gs["parameters"])

        data = [
            (
                result["model_type"][i],
                result["parameters"][i],
                result["avg_score"][i],
                result["avg_train_score"][i],
                result["avg_time"][i],
                result["score_std"][i],
                result["score_train_std"][i],
                result["model_class"][i],
            )
            for i in range(len(result["score_train_std"]))
        ]
        reverse = reverse_score(self.parameters["metric"])
        data.sort(key=lambda tup: tup[2], reverse=reverse)
        result = TableSample(
            {
                "model_type": [elem[0] for elem in data],
                "parameters": [elem[1] for elem in data],
                "avg_score": [elem[2] for elem in data],
                "avg_train_score": [elem[3] for elem in data],
                "avg_time": [elem[4] for elem in data],
                "score_std": [elem[5] for elem in data],
                "score_train_std": [elem[6] for elem in data],
                "model_class": [elem[7] for elem in data],
            }
        )
        if self.parameters["print_info"]:
            print(f"\033[1m\033[4mFinal Model\033[0m\033[0m\n")
            print(
                f"{result['model_type'][0]}; Best_Parameters: {result['parameters'][0]}; \033[91mBest_Test_score: {result['avg_score'][0]}\033[0m; \033[92mTrain_score: {result['avg_train_score'][0]}\033[0m; \033[94mTime: {result['avg_time'][0]}\033[0m;\n\n"
            )
        best_model = result["model_class"][0](self.name)
        best_model.set_params(result["parameters"][0])
        self.stepwise_ = None
        if self.parameters["stepwise"]:
            self.stepwise_ = stepwise(
                best_model,
                input_relation,
                X,
                y,
                criterion=self.parameters["stepwise_criterion"],
                direction=self.parameters["stepwise_direction"],
                max_steps=self.parameters["stepwise_max_steps"],
                x_order=self.parameters["stepwise_x_order"],
                print_info=self.parameters["print_info"],
                drop_final_estimator=False,
                show=False,
                criterion_threshold=2,
            )
        else:
            best_model.fit(input_relation, X, y)
        self.best_model_ = best_model
        self.VERTICA_FIT_FUNCTION_SQL = best_model.VERTICA_FIT_FUNCTION_SQL
        self.VERTICA_PREDICT_FUNCTION_SQL = best_model.VERTICA_PREDICT_FUNCTION_SQL
        self.model_grid_ = result
        self.parameters["reverse"] = not (reverse)
        if self.preprocess_ != None:
            self.preprocess_.drop()
            self.preprocess_.final_relation_ = vDataFrame(self.preprocess_.sql_)
        return self.model_grid_

    def plot(self, mltype: str = "champion", ax=None, **style_kwds):
        """
    Draws the AutoML plot.

    Parameters
    ----------
    mltype: str, optional
        The plot type.
            champion: champion challenger plot.
            step    : stepwise plot.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object
        """
        if mltype == "champion":
            return plot_bubble_ml(
                self.model_grid_["avg_time"],
                self.model_grid_["avg_score"],
                self.model_grid_["score_std"],
                self.model_grid_["model_type"],
                x_label="time",
                y_label="score",
                title="Model Type",
                ax=ax,
                reverse=(True, self.parameters["reverse"]),
                **style_kwds,
            )
        else:
            return plot_stepwise_ml(
                [len(elem) for elem in self.stepwise_["features"]],
                self.stepwise_[self.parameters["stepwise_criterion"]],
                self.stepwise_["variable"],
                self.stepwise_["change"],
                [self.stepwise_["features"][0], self.stepwise_.best_list_],
                x_label="n_features",
                y_label=self.parameters["stepwise_criterion"],
                direction=self.parameters["stepwise_direction"],
                ax=ax,
                **style_kwds,
            )
