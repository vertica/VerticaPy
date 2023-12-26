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
from typing import Literal, Optional, Union
from collections.abc import Iterable

import numpy as np

from verticapy._typing import PlottingObject, PythonScalar, SQLColumns, SQLRelation
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import format_type

from verticapy.core.tablesample.base import TableSample
from verticapy.core.vdataframe.base import vDataFrame

from verticapy.machine_learning.model_selection.hp_tuning.cv import grid_search_cv
from verticapy.machine_learning.vertica.base import VerticaModel

from verticapy.plotting._utils import PlottingUtils

"""
Tracking Over-fitting.
"""


@save_verticapy_logs
def validation_curve(
    estimator: VerticaModel,
    param_name: str,
    param_range: list,
    input_relation: SQLRelation,
    X: SQLColumns,
    y: str,
    metric: str = "auto",
    cv: int = 3,
    average: Literal["binary", "micro", "macro", "weighted"] = "weighted",
    pos_label: Optional[PythonScalar] = None,
    cutoff: float = -1,
    std_coeff: float = 1,
    chart: Optional[PlottingObject] = None,
    show: Optional[bool] = False,
    **style_kwargs,
) -> TableSample:
    """
    Draws the validation curve.

    Parameters
    ----------
    estimator: VerticaModel
        Vertica estimator with a fit method.
    param_name: str
        Parameter name.
    param_range: list
        Parameter Range.
    input_relation: SQLRelation
        Relation used to train the model.
    X: SQLColumns
        List of the predictor columns.
    y: str
        Response Column.
    metric: str, optional
        Metric used to for model evaluation.
        - auto:
            logloss for classification
            & RMSE for regression.

        **For Classification**

        - accuracy:
            Accuracy.

            .. math::

                Accuracy = \\frac{TP + TN}{TP + TN + FP + FN}

        - auc:
            Area Under the Curve (ROC).

            .. math::

                AUC = \int_{0}^{1} TPR(FPR) \, dFPR

        - ba:
            Balanced Accuracy.

            .. math::

                BA = \\frac{TPR + TNR}{2}

        - bm:
            Informedness

            .. math::

                BM = TPR + TNR - 1

        - csi:
            Critical Success Index

            .. math::

                index = \\frac{TP}{TP + FN + FP}

        - f1:
            F1 Score
            .. math::

                F_1 Score = 2 \\times \\frac{Precision \\times Recall}{Precision + Recall}

        - fdr:
            False Discovery Rate

            .. math::

                FDR = 1 - PPV

        - fm:
            Fowlkes-Mallows index

            .. math::

                FM = \\sqrt{PPV * TPR}

        - fnr:
            False Negative Rate

            .. math::

                FNR = \\frac{FN}{FN + TP}

        - for:
            False Omission Rate

            .. math::

                FOR = 1 - NPV

        - fpr:
            False Positive Rate

            .. math::

                FPR = \\frac{FP}{FP + TN}

        - logloss:
            Log Loss

            .. math::

                Loss = -\\frac{1}{N} \sum_{i=1}^{N} \left( y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \\right)

        - lr+:
            Positive Likelihood Ratio.

            .. math::

                LR+ = \\frac{TPR}{FPR}

        - lr-:
            Negative Likelihood Ratio.

            .. math::

                LR- = \\frac{FNR}{TNR}

        - dor:
            Diagnostic Odds Ratio.

            .. math::

                DOR = \\frac{TP \\times TN}{FP \\times FN}

        - mcc:
            Matthews Correlation Coefficient

        - mk:
            Markedness

            .. math::

                MK = PPV + NPV - 1

        - npv:
            Negative Predictive Value

            .. math::

                NPV = \\frac{TN}{TN + FN}

        - prc_auc:
            Area Under the Curve (PRC)

            .. math::

                AUC = \int_{0}^{1} Precision(Recall) \, dRecall

        - precision:
            Precision

            .. math::

                TP / (TP + FP)

        - pt:
            Prevalence Threshold.

            .. math::

                \\frac{\\sqrt{FPR}}{\\sqrt{TPR} + \\sqrt{FPR}}

        - recall:
            Recall.

            .. math::
                TP / (TP + FN)

        - specificity:
            Specificity.

            .. math::

                TN / (TN + FP)

        **For Regression**

        - max:
            Max Error.

            .. math::

                ME = \max_{i=1}^{n} \left| y_i - \hat{y}_i \\right|

        - mae:
            Mean Absolute Error.

            .. math::

                MAE = \\frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \\right|

        - median:
            Median Absolute Error.

            .. math::

                MedAE = \\text{median}_{i=1}^{n} \left| y_i - \hat{y}_i \\right|

        - mse:
            Mean Squared Error.

            .. math::

                MSE = \\frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \\right)^2

        - msle:
            Mean Squared Log Error.

            .. math::

                MSLE = \\frac{1}{n} \sum_{i=1}^{n} (\log(1 + y_i) - \log(1 + \hat{y}_i))^2

        - r2:
            R squared coefficient.

            .. math::

                R^2 = 1 - \\frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \\bar{y})^2}

        - r2a:
            R2 adjusted

            .. math::

                \\text{Adjusted } R^2 = 1 - \\frac{(1 - R^2)(n - 1)}{n - k - 1}

        - var:
            Explained Variance.

            .. math::

                VAR = 1 - \\frac{Var(y - \hat{y})}{Var(y)}

        - rmse:
            Root-mean-squared error

            .. math::

                RMSE = \sqrt{\\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}

    cv: int, optional
        Number of folds.
    average: str, optional
        The method used to  compute the final score for
        multiclass-classification.

        - binary:
            considers one of the
            classes as positive
            and use the binary
            confusion matrix to
            compute the score.

        - micro:
            positive and negative
            values globally.

        - macro:
            average of the score
            of each class.

        - weighted:
            weighted average of
            the score of each
            class.

    pos_label: PythonScalar, optional
        The main class to be
        considered as positive
        (classification only).
    cutoff: float, optional
        The model cutoff
        (classification only).
    std_coeff: float, optional
        Value of the standard deviation
        coefficient used to compute the
        area plot around each score.
    chart: PlottingObject, optional
        The chart object to plot on.
    show: bool, optional
        Select whether you want to get
        the chart as the output only.
    **style_kwargs
        Any  optional parameter to
        pass to the Plotting functions.

    Returns
    -------
    TableSample
        ``training_score_lower, training_score,training_score_upper, test_score_lower,test_score,test_score_upper``


    Examples
    --------

    .. note::

        The below example is a very basic one. For
        other more detailed examples and customization
        options, please see :ref:`chart_gallery.learning`_

    We import :py:mod:`verticapy`:

    .. ipython:: python

        import verticapy as vp

    .. hint::

        By assigning an alias to :py:mod:`verticapy`,
        we mitigate the risk of code collisions with
        other libraries. This precaution is necessary
        because verticapy uses commonly known function
        names like "average" and "median", which can
        potentially lead to naming conflicts. The use
        of an alias ensures that the functions from
        :py:mod:`verticapy` are used as intended
        without interfering with functions from other
        libraries.

    Let's generate a dataset
    using the following data.

    .. ipython:: python

        import random

        N = 500 # Number of Records
        k = 10 # step

        # Normal Distributions
        x = np.random.normal(5, 1, round(N / 2))
        y = np.random.normal(3, 1, round(N / 2))
        z = np.random.normal(3, 1, round(N / 2))

        # Creating a vDataFrame with two clusters
        data = vp.vDataFrame({
            "x": np.concatenate([x, x + k]),
            "y": np.concatenate([y, y + k]),
            "z": np.concatenate([z, z + k]),
            "c": [random.randint(0, 1) for _ in range(N)]
        })

    Let's proceed by creating a
    :py:class:`~verticapy.machine_learning.vertica.ensemble.RandomForestClassifier`
    model using the complete dataset.

    .. ipython:: python

        # Importing the Vertica ML module
        import verticapy.machine_learning.vertica as vml

        # Importing the model selection module
        import verticapy.machine_learning.model_selection as vms

        # Defining the Model
        model = vml.RandomForestClassifier()

    Let's draw the validation curve.

    .. code-block:: python

        vms.validation_curve(
          model,
          param_name = "max_depth",
          param_range = [1, 2, 3],
          input_relation = data,
          X = ["x", "y", "z"],
          y = "c",
          cv = 3,
          metric = "auc",
          show = True,
        )

    .. ipython:: python
        :suppress:
        :okwarning:

        vp.set_option("plotting_lib", "plotly")
        fig = vms.validation_curve(
          model,
          param_name = "max_depth",
          param_range = [1, 2, 3],
          input_relation = data,
          X = ["x", "y", "z"],
          y = "c",
          cv = 3,
          metric = "auc",
          show = True,
        )
        fig.write_html("figures/machine_learning_model_selection_hp_tuning_validation_curve.html")

    .. raw:: html
          :file: SPHINX_DIRECTORY/figures/machine_learning_model_selection_hp_tuning_validation_curve.html

    .. note::

        VerticaPy's Learning Curve tool is
        an essential asset for evaluating
        machine learning models. It enables
        users to visualize a model's performance
        by plotting key metrics against varying
        training dataset sizes. By analyzing
        these curves, data analysts can identify
        issues such as overfitting or underfitting,
        make informed decisions about dataset size,
        and optimize model performance. This feature
        plays a crucial role in enhancing model
        robustness and facilitating data-driven
        decision-making.

    .. seealso::

        | :py:func:`~verticapy.machine_learning.model_selection.learning_curve` :
            Draws the learning curve.
    """
    X = format_type(X, dtype=list)
    if not isinstance(param_range, Iterable) or isinstance(param_range, str):
        param_range = [param_range]
    gs_result = grid_search_cv(
        estimator,
        {param_name: param_range},
        input_relation,
        X,
        y,
        metric=metric,
        cv=cv,
        average=average,
        pos_label=pos_label,
        cutoff=cutoff,
        training_score=True,
        skip_error=False,
        print_info=False,
    )
    gs_result_final = [
        (
            gs_result["parameters"][i][param_name],
            gs_result["avg_score"][i],
            gs_result["avg_train_score"][i],
            gs_result["score_std"][i],
            gs_result["score_train_std"][i],
        )
        for i in range(len(param_range))
    ]
    gs_result_final.sort(key=lambda tup: tup[0])
    x = np.array([s[0] for s in gs_result_final])
    Y = np.column_stack(
        (
            [s[2] - std_coeff * s[4] for s in gs_result_final],
            [s[2] for s in gs_result_final],
            [s[2] + std_coeff * s[4] for s in gs_result_final],
            [s[1] - std_coeff * s[3] for s in gs_result_final],
            [s[1] for s in gs_result_final],
            [s[1] + std_coeff * s[3] for s in gs_result_final],
        )
    )
    result = TableSample(
        {
            param_name: x,
            "training_score_lower": Y[:, 0],
            "training_score": Y[:, 1],
            "training_score_upper": Y[:, 2],
            "test_score_lower": Y[:, 3],
            "test_score": Y[:, 4],
            "test_score_upper": Y[:, 5],
        }
    )
    vpy_plt, kwargs = PlottingUtils().get_plotting_lib(
        class_name="RangeCurve",
        chart=chart,
        style_kwargs=style_kwargs,
    )
    data = {"x": x, "Y": Y}
    layout = {"columns": ["train", "test"], "order_by": param_name, "y_label": metric}
    vpy_plt.RangeCurve(data=data, layout=layout).draw(**kwargs)
    if show:
        return vpy_plt.RangeCurve(data=data, layout=layout).draw(**kwargs)
    return result


"""
TSA - Finding ARIMA parameters.
"""


@save_verticapy_logs
def plot_acf_pacf(
    vdf: vDataFrame,
    column: str,
    ts: str,
    by: Optional[SQLColumns] = None,
    p: Union[int, list] = 15,
    show: bool = True,
    **style_kwargs,
) -> TableSample:
    """
    Draws the ACF and PACF Charts.

    Parameters
    ----------
    vdf: vDataFrame
        Input vDataFrame.
    column: str
        Response column.
    ts: str
        vDataColumn used as timeline to order the data.
        It can be a numerical or date-like type (date,
        datetime,   timestamp...) vDataColumn.
    by: list, optional
        vDataColumns used in the partition.
    p: int | list, optional
        Integer equal to the maximum  number  of lags to
        consider during the computation or a list of the
        different lags to include during the computation.
        p must be positive or a list of positive integers.
    show: bool, optional
        If  set to  True,  the  Plotting  object is
        returned.
    **style_kwargs
        Any optional  parameter to pass to the Plotting
        functions.

    Returns
    -------
    TableSample
        ``acf, pacf, confidence``

    Examples
    --------

    We import :py:mod:`verticapy`:

    .. ipython:: python

        import verticapy as vp

    .. hint::

        By assigning an alias to :py:mod:`verticapy`,
        we mitigate the risk of code collisions with
        other libraries. This precaution is necessary
        because verticapy uses commonly known function
        names like "average" and "median", which can
        potentially lead to naming conflicts. The use
        of an alias ensures that the functions from
        :py:mod:`verticapy` are used as intended
        without interfering with functions from other
        libraries.

    For this example, we will use the Amazon dataset.

    .. code-block:: python

        import verticapy.datasets as vpd

        amazon = vpd.load_amazon()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_amazon.html

    .. note::

        VerticaPy offers a wide range of sample
        datasets that are ideal for training
        and testing purposes. You can explore
        the full list of available datasets in
        the :ref:`api.datasets`, which provides
        detailed information on each dataset and
        how to use them effectively. These datasets
        are invaluable resources for honing your
        data analysis and machine learning skills
        within the VerticaPy environment.

    .. ipython:: python
        :suppress:

        import verticapy.datasets as vpd
        amazon = vpd.load_amazon()

    Let's select only one state to get a refined plot.

    .. ipython:: python

        amazon = amazon[amazon["state"] == "ACRE"]

    We can have a look at the time-series plot
    using the ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.plot`:

    .. code-block::

        amazon["number"].plot(ts = "date")

    .. ipython:: python
        :suppress:
        :okwarning:

        vp.set_option("plotting_lib", "plotly")
        fig = amazon["number"].plot(ts = "date")
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_model_selection_hp_tuning_plotting_plot_data.html", "w")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_model_selection_hp_tuning_plotting_plot_data.html

    Now we can plot the ACF and PACF plots together:

    .. code-block::

        from verticapy.machine_learning.model_selection import plot_acf_pacf

        plot_acf_pacf(
            amazon,
            column = "number",
            ts = "date",
            p = 40,
        )

    .. ipython:: python
        :suppress:
        :okwarning:

        vp.set_option("plotting_lib", "plotly")
        from verticapy.machine_learning.model_selection import plot_acf_pacf
        result = plot_acf_pacf(
            amazon,
            column = "number",
            ts = "date",
            p = 40,
            show = False,
        )
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_model_selection_hp_tuning_plotting_plot_acf_pacf.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_model_selection_hp_tuning_plotting_plot_acf_pacf.html

    .. seealso::

        | :py:func:`~vDataFrame.acf` : ACF plot from a :py:class:`~vDataFrame`.
        | :py:func:`~vDataFrame.pacf` : PACF plot from a :py:class:`~vDataFrame`.
    """
    by = format_type(by, dtype=list)
    by, column, ts = vdf.format_colnames(by, column, ts)
    acf = vdf.acf(ts=ts, column=column, by=by, p=p, show=False)
    pacf = vdf.pacf(ts=ts, column=column, by=by, p=p, show=False)
    index = [i for i in range(0, len(acf.values["value"]))]
    if show:
        vpy_plt, kwargs = PlottingUtils().get_plotting_lib(
            class_name="ACFPACFPlot",
            style_kwargs=style_kwargs,
        )
        data = {
            "x": np.array(index),
            "y0": np.array(acf.values["value"]),
            "y1": np.array(pacf.values["value"]),
            "z": np.array(pacf.values["confidence"]),
        }
        layout = {
            "y0_label": "Autocorrelation",
            "y1_label": "Partial Autocorrelation",
        }
        return vpy_plt.ACFPACFPlot(data=data, layout=layout).draw(**kwargs)
    return TableSample(
        {
            "index": index,
            "acf": acf.values["value"],
            "pacf": pacf.values["value"],
            "confidence": pacf.values["confidence"],
        }
    )
