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
import copy
from typing import Literal, Optional, Union
from tqdm.auto import tqdm
import numpy as np

import verticapy._config.config as conf
from verticapy._typing import (
    PlottingObject,
    PythonScalar,
    NoneType,
    SQLRelation,
    SQLColumns,
)
from verticapy._utils._gen import gen_tmp_name
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import format_type, schema_relation
from verticapy._utils._sql._vertica_version import vertica_version
from verticapy.errors import ModelError

from verticapy.core.tablesample.base import TableSample
from verticapy.core.vdataframe.base import vDataFrame

import verticapy.machine_learning.memmodel as mm
from verticapy.machine_learning.model_selection.hp_tuning.cv import grid_search_cv
from verticapy.machine_learning.model_selection.hp_tuning.param_gen import (
    gen_params_grid,
)
from verticapy.machine_learning.model_selection.variables_selection import stepwise
from verticapy.machine_learning.vertica.automl.dataprep import AutoDataPrep
from verticapy.machine_learning.vertica.base import VerticaModel
from verticapy.machine_learning.vertica.cluster import NearestCentroid
from verticapy.machine_learning.vertica.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    XGBClassifier,
    XGBRegressor,
)
from verticapy.machine_learning.vertica.naive_bayes import NaiveBayes
from verticapy.machine_learning.vertica.linear_model import (
    LogisticRegression,
    LinearRegression,
    ElasticNet,
    Lasso,
    Ridge,
)
from verticapy.machine_learning.vertica.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
)
from verticapy.machine_learning.vertica.svm import LinearSVC, LinearSVR


class AutoML(VerticaModel):
    """
    Tests multiple models to find those that maximize
    the input score.

    Parameters
    ----------
    name: str, optional
        Name of the model.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    estimator: list / 'native' / 'all' / 'fast' / object
        List  of Vertica  estimators with a fit  method.
        Alternatively,  you can specify 'native' for all
        native  Vertica models, 'all' for all  VerticaPy
        models, and 'fast' for quick modeling.
    estimator_type: str, optional
        Estimator Type.
            auto      : Automatically     detects     the
                        estimator type.
            regressor : The estimator  is  used  to
                        perform a regression.
            binary    : The  estimator is used  to
                        perform  a binary classification.
            multi     : The  estimator  is  used  to
                        perform  a multiclass
                        classification.
    metric: str, optional
        Metric used for the model evaluation.
            auto: logloss for  classification & RMSE for
                  regression.
        For Classification:
            accuracy    : Accuracy
            auc         : Area Under the Curve
                          (ROC)
            ba          : Balanced Accuracy
                          = (tpr + tnr) / 2
            bm          : Informedness
                          = tpr + tnr - 1
            csi         : Critical Success Index
                          = tp / (tp + fn + fp)
            f1          : F1 Score
            fdr         : False Discovery Rate = 1 - ppv
            fm          : Fowlkes–Mallows index
                          = sqrt(ppv * tpr)
            fnr         : False Negative Rate
                          = fn / (fn + tp)
            for         : False Omission Rate = 1 - npv
            fpr         : False Positive Rate
                          = fp / (fp + tn)
            logloss     : Log Loss
            lr+         : Positive Likelihood Ratio
                          = tpr / fpr
            lr-         : Negative Likelihood Ratio
                          = fnr / tnr
            dor         : Diagnostic Odds Ratio
            mcc         : Matthews Correlation Coefficient
            mk          : Markedness
                          = ppv + npv - 1
            npv         : Negative Predictive Value
                          = tn / (tn + fn)
            prc_auc     : Area Under the Curve
                          (PRC)
            precision   : Precision
                          = tp / (tp + fp)
            pt          : Prevalence Threshold
                          = sqrt(fpr) / (sqrt(tpr) + sqrt(fpr))
            recall      : Recall
                          = tp / (tp + fn)
            specificity : Specificity
                          = tn / (tn + fp)
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
    pos_label: PythonScalar, optional
        The main class to  be considered as positive
        (classification only).
    cutoff: float, optional
        The model cutoff (classification only).
    nbins: int, optional
        Number of bins used to compute the different
        parameter categories.
    lmax: int, optional
        Maximum length of each parameter list.
    optimized_grid: int, optional
        If set to zero, the randomness is based on the
        input parameters.
        If set to one, the randomness  is limited  to
        some parameters while others are picked based
        on a default grid.
        If  set to two, no randomness is used  and  a
        default grid is returned.
    stepwise: bool, optional
        If True, the stepwise algorithm is used to
        determine the final model list of parameters.
    stepwise_criterion: str, optional
        Criterion used when performing the final
        estimator stepwise.
            aic : Akaike’s information criterion
            bic : Bayesian information criterion
    stepwise_direction: str, optional
        Direction to start the stepwise search,
        either 'backward' or 'forward'.
    stepwise_max_steps: int, optional
        The maximum number of steps to be considered
        when performing the final estimator stepwise.
    x_order: str, optional
        Method for preprocessing  X before using the
        stepwise algorithm.
            pearson  : X  is ordered  based  on  the
                       Pearson's         correlation
                       coefficient.
            spearman : X   is   ordered   based   on
                       Spearman's  rank  correlation
                       coefficient.
            random   : Shuffles the  vector X before
                       applying     the     stepwise
                       algorithm.
            none     : Does not  change the order of
                       X.
    preprocess_data: bool, optional
        If True, the data will be preprocessed.
    preprocess_dict: dict, optional
        Dictionary to pass to the AutoDataPrep class
        in  order to preprocess the data before
        clustering.
    print_info: bool
        If  True,  prints the model  information  at
        each step.

    Attributes
    ----------
    preprocess_: object
        Model used to preprocess the data.
    best_model_: object
        Most  efficient   models  found  during  the
        search.
    model_grid_ : TableSample
        Grid   containing   the   different   models
        information.
    """

    # Properties.

    @property
    def _is_native(self) -> Literal[False]:
        return False

    @property
    def _vertica_fit_sql(self) -> Literal[""]:
        return ""

    @property
    def _vertica_predict_sql(self) -> Literal[""]:
        return ""

    @property
    def _model_category(self) -> Literal["SUPERVISED"]:
        return "SUPERVISED"

    @property
    def _model_subcategory(self) -> Literal[""]:
        return ""

    @property
    def _model_type(self) -> Literal["AutoML"]:
        return "AutoML"

    @property
    def _attributes(self) -> list[str]:
        return ["preprocess_", "best_model_", "model_grid_"]

    # System & Special Methods.

    @save_verticapy_logs
    def __init__(
        self,
        name: Optional[str] = None,
        overwrite_model: bool = False,
        estimator: Union[list, str] = "fast",
        estimator_type: Literal["auto", "regressor", "binary", "multi"] = "auto",
        metric: str = "auto",
        cv: int = 3,
        pos_label: Optional[PythonScalar] = None,
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
    ) -> None:
        if optimized_grid not in [0, 1, 2]:
            raise ValueError("Optimized Grid must be an integer between 0 and 2.")
        super().__init__(name, overwrite_model)
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

    # Attributes Methods.

    def get_vertica_attributes(self, attr_name: Optional[str] = None) -> TableSample:
        """
        Returns the model attribute.

        Parameters
        ----------
        attr_name: str, optional
            Attribute Name.

        Returns
        -------
        TableSample
            model attributes.
        """
        return self.best_model_.get_vertica_attributes(attr_name)

    # I/O Methods.

    def deploySQL(self, X: Optional[SQLColumns] = None) -> str:
        """
        Returns the SQL code needed to deploy the model.

        Parameters
        ----------
        X: SQLColumns, optional
            List of the columns used to deploy the model.
            If empty, the model  predictors are used.

        Returns
        -------
        str
            the SQL code needed to deploy the model.
        """
        return self.best_model_.deploySQL(X)

    def to_memmodel(self) -> mm.InMemoryModel:
        """
        Converts  the model to  an InMemory object  that
        can be used for different types of predictions.
        """
        return self.best_model_.to_memmodel()

    # Model Fitting Method.

    def fit(
        self,
        input_relation: SQLRelation,
        X: Optional[SQLColumns] = None,
        y: Optional[str] = None,
        return_report: bool = False,
    ) -> None:
        """
        Trains the model.

        Parameters
        ----------
        input_relation: SQLRelation
            Training Relation.
        X: SQLColumns, optional
            List of the predictors.
        y: str, optional
            Response column.
        """
        if self.overwrite_model:
            self.drop()
        else:
            self._is_already_stored(raise_error=True)
        if isinstance(X, NoneType):
            if not y:
                exclude_columns = []
            else:
                exclude_columns = [y]
            if not isinstance(input_relation, vDataFrame):
                X = vDataFrame(input_relation).get_columns(
                    exclude_columns=exclude_columns
                )
            else:
                X = input_relation.get_columns(exclude_columns=exclude_columns)
        X = format_type(X, dtype=list)
        if isinstance(self.parameters["estimator"], str):
            v = vertica_version()
            self.parameters["estimator"] = self.parameters["estimator"].lower()
            estimator_method = self.parameters["estimator"]
            if not isinstance(input_relation, vDataFrame):
                vdf = vDataFrame(input_relation)
            else:
                vdf = input_relation
            if self.parameters["estimator_type"].lower() == "binary" or (
                self.parameters["estimator_type"].lower() == "auto"
                and sorted(vdf[y].distinct()) == [0, 1]
            ):
                self.parameters["estimator_type"] = "binary"
                self.parameters["estimator"] = [
                    LogisticRegression(self.model_name),
                    NaiveBayes(self.model_name),
                ]
                if estimator_method in ("native", "all"):
                    if v[0] > 10 or (v[0] == 10 and v[1] >= 1):
                        self.parameters["estimator"] += [XGBClassifier(self.model_name)]
                    if v[0] >= 9:
                        self.parameters["estimator"] += [
                            LinearSVC(self.model_name),
                            RandomForestClassifier(self.model_name),
                        ]
                if estimator_method == "all":
                    self.parameters["estimator"] += [
                        KNeighborsClassifier(self.model_name),
                        NearestCentroid(self.model_name),
                    ]
            elif self.parameters["estimator_type"].lower() == "regressor" or (
                self.parameters["estimator_type"].lower() == "auto" and vdf[y].isnum()
            ):
                self.parameters["estimator_type"] = "regressor"
                self.parameters["estimator"] = [
                    LinearRegression(self.model_name),
                    ElasticNet(self.model_name),
                    Ridge(self.model_name),
                    Lasso(self.model_name),
                ]
                if estimator_method in ("native", "all"):
                    if v[0] > 10 or (v[0] == 10 and v[1] >= 1):
                        self.parameters["estimator"] += [XGBRegressor(self.model_name)]
                    if v[0] >= 9:
                        self.parameters["estimator"] += [
                            LinearSVR(self.model_name),
                            RandomForestRegressor(self.model_name),
                        ]
                if estimator_method == "all":
                    self.parameters["estimator"] += [
                        KNeighborsRegressor(self.model_name)
                    ]
            elif self.parameters["estimator_type"].lower() in ("multi", "auto"):
                self.parameters["estimator_type"] = "multi"
                self.parameters["estimator"] = [NaiveBayes(self.model_name)]
                if estimator_method in ("native", "all"):
                    if v[0] >= 10 and v[1] >= 1:
                        self.parameters["estimator"] += [XGBClassifier(self.model_name)]
                    if v[0] >= 9:
                        self.parameters["estimator"] += [
                            RandomForestClassifier(self.model_name)
                        ]
                if estimator_method == "all":
                    self.parameters["estimator"] += [
                        KNeighborsClassifier(self.model_name),
                        NearestCentroid(self.model_name),
                    ]
            else:
                raise ValueError(
                    f"Parameter 'estimator_type' must be in auto|binary|multi|regressor. Found {self.parameters['estimator_type']}."
                )
        elif isinstance(
            self.parameters["estimator"],
            (
                RandomForestRegressor,
                RandomForestClassifier,
                XGBRegressor,
                XGBClassifier,
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
                        XGBRegressor,
                        XGBClassifier,
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
                ), ValueError(
                    f"estimator must be a list of VerticaPy estimators. Found {elem}."
                )
        if self.parameters["estimator_type"] == "auto":
            self.parameters["estimator_type"] = self.parameters["estimator"][
                0
            ]._model_type
        for elem in self.parameters["estimator"]:
            assert (
                self.parameters["estimator_type"] in ("binary", "multi")
                and elem._model_subcategory == "CLASSIFIER"
                or self.parameters["estimator_type"] == "regressor"
                and elem._model_subcategory == "REGRESSOR"
            ), ValueError(
                f"Incorrect list for parameter 'estimator'. Expected type '{self.parameters['estimator_type']}', found type '{elem._model_subcategory}'."
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
            model_preprocess = AutoDataPrep(
                **self.parameters["preprocess_dict"],
                overwrite_model=True,
            )
            model_preprocess.fit(
                input_relation,
                X=X,
                return_report=True,
            )
            input_relation = model_preprocess.final_relation_
            X = copy.deepcopy(model_preprocess.X_out_)
            self.preprocess_ = model_preprocess
        else:
            self.preprocess_ = None
        if self.parameters["print_info"]:
            print(f"\033[1m\033[4mStarting AutoML\033[0m\033[0m\n")
        if conf.get_option("tqdm") and self.parameters["print_info"]:
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
                metric=self.parameters["metric"],
                cv=self.parameters["cv"],
                pos_label=self.parameters["pos_label"],
                cutoff=self.parameters["cutoff"],
                training_score=True,
                skip_error="no_print",
                print_info=self.parameters["print_info"],
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
        reverse = True
        if self.parameters["metric"] in [
            "logloss",
            "max",
            "mae",
            "median",
            "mse",
            "msle",
            "rmse",
            "aic",
            "bic",
            "auto",
        ]:
            reverse = False
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
        if len(result["model_type"]) == 0:
            raise ModelError(
                "Error: 'AutoML' failed to converge. Please retry fitting the estimator."
            )
        if self.parameters["print_info"]:
            print(f"\033[1m\033[4mFinal Model\033[0m\033[0m\n")
            print(
                f"{result['model_type'][0]}; Best_Parameters: {result['parameters'][0]}; \033[91mBest_Test_score: {result['avg_score'][0]}\033[0m; \033[92mTrain_score: {result['avg_train_score'][0]}\033[0m; \033[94mTime: {result['avg_time'][0]}\033[0m;\n\n"
            )
        best_model = result["model_class"][0](self.model_name)
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
            best_model.fit(
                input_relation,
                X,
                y,
                return_report=True,
            )
        self.best_model_ = best_model
        self.model_grid_ = result
        self.parameters["reverse"] = not reverse
        if not isinstance(self.preprocess_, NoneType):
            self.preprocess_.drop()
            self.preprocess_.final_relation_ = vDataFrame(self.preprocess_.sql_)

    # Features Importance Methods.

    def features_importance(
        self, chart: Optional[PlottingObject] = None, **style_kwargs
    ) -> PlottingObject:
        """
        Computes the model's features importance.

        Parameters
        ----------
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any optional parameter to pass to the
            Plotting functions.

        Returns
        -------
        obj
            features importance.
        """
        if self.stepwise_:
            data = {
                "importance": self.stepwise_["importance"],
            }
            layout = {"columns": copy.deepcopy(self.stepwise_["variable"])}
            vpy_plt, kwargs = self.get_plotting_lib(
                class_name="ImportanceBarChart",
                chart=chart,
                style_kwargs=style_kwargs,
            )
            vpy_plt.ImportanceBarChart(data=data, layout=layout).draw(**kwargs)
        return self.best_model_.features_importance(**kwargs)

    # Plotting Methods.

    def plot(
        self,
        mltype: Literal["champion", "step"] = "champion",
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the AutoML plot.

        Parameters
        ----------
        mltype: str, optional
            The plot type.
                champion : champion challenger plot.
                step     : stepwise plot.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any optional  parameter  to pass to  the
            Plotting functions.

        Returns
        -------
        obj
            Plotting Object.
        """
        if mltype == "champion":
            data = {
                "x": np.array(self.model_grid_["avg_time"]).astype(float),
                "y": np.array(self.model_grid_["avg_score"]).astype(float),
                "s": np.array(self.model_grid_["score_std"]).astype(float),
                "c": np.array(self.model_grid_["model_type"]),
            }
            layout = {
                "x_label": "time",
                "y_label": self.parameters["metric"],
                "z_label": self.parameters["metric"] + "_std",
                "title": "Model Type",
                "reverse": (True, self.parameters["reverse"]),
            }
            vpy_plt, kwargs = self.get_plotting_lib(
                class_name="ChampionChallengerPlot",
                chart=chart,
                matplotlib_kwargs={"plt_text": True},
                style_kwargs=style_kwargs,
            )
            return vpy_plt.ChampionChallengerPlot(data=data, layout=layout).draw(
                **kwargs
            )
        else:
            vpy_plt, kwargs = self.get_plotting_lib(
                class_name="StepwisePlot",
                chart=chart,
                style_kwargs=style_kwargs,
            )
            data = {
                "x": np.array([len(x) for x in self.stepwise_["features"]]).astype(int),
                "y": np.array(
                    self.stepwise_[self.parameters["stepwise_criterion"]]
                ).astype(float),
                "c": np.array(self.stepwise_["variable"]),
                "sign": np.array(self.stepwise_["change"]),
            }
            layout = {
                "in_variables": self.stepwise_["features"][0],
                "out_variables": self.stepwise_.best_list_,
                "x_label": "n_features",
                "y_label": self.parameters["stepwise_criterion"],
                "direction": self.parameters["stepwise_direction"],
            }
            return vpy_plt.StepwisePlot(data=data, layout=layout).draw(**kwargs)
