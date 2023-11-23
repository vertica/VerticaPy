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
import numpy as np

from verticapy._typing import PlottingObject
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._vertica_version import check_minimum_version

from verticapy.core.vdataframe.base import vDataFrame

from verticapy.machine_learning.vertica.base import BinaryClassifier, Regressor
from verticapy.machine_learning.vertica.linear_model import (
    LinearModel,
    LinearModelClassifier,
)

"""
Algorithms used for regression.
"""


class LinearSVR(LinearModel, Regressor):
    """
    Creates  a  LinearSVR  object  using the Vertica  SVM
    (Support Vector Machine)  algorithm.  This  algorithm
    finds the hyperplane used to approximate distribution
    of the data.

    Parameters
    ----------
    name: str, optional
        Name of the model. The model is stored in
        the database.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    tol: float, optional
        Tolerance for stopping criteria. This is
        used to control accuracy.
    C: float, optional
        Weight  for  misclassification  cost. The
        algorithm minimizes the regularization
        cost and the misclassification cost.
    intercept_scaling: float
        A  float value, serves  as the value of a
        dummy feature  whose  coefficient Vertica
        uses to calculate the model intercept.
        Because  the dummy feature is not in  the
        training data,  its values  are  set to a
        constant, by default set to 1.
    intercept_mode: str, optional
        Specify how to treat the intercept.

        - regularized:
            Fits  the intercept  and applies a
            regularization.
        - unregularized:
            Fits the  intercept  but does not  include
            it in regularization.
    acceptable_error_margin: float, optional
        Defines the acceptable error margin. Any data
        points  outside this region add a penalty  to
        the cost function.
    max_iter: int, optional
        The  maximum  number of iterations  that  the
        algorithm performs.

    Attributes
    ----------
    Many attributes are created during the fitting phase.

    coef_: numpy.array
        The regression coefficients. The order of
        coefficients is the same as the order of
        columns used during the fitting phase.
    intercept_: float
        The expected value of the dependent variable
        when all independent variables are zero,
        serving as the baseline or constant term in
        the model.
    features_importance_: numpy.array
        The importance of features is computed through
        the model coefficients, which are normalized
        based on their range. Subsequently, an
        activation function calculates the final score.
        It is necessary to use the
        :py:meth:`verticapy.machine_learning.vertica.linear_model.LinearModel.features_importance`
        method to compute it initially, and the computed
        values will be subsequently utilized for subsequent
        calls.

    .. note::

        All attributes can be accessed using the
        :py:meth:`verticapy.machine_learning.vertica.linear_model.LinearModel.get_attributes``
        method.

    .. note::

        Several other attributes can be accessed by using the
        :py:meth:`verticapy.machine_learning.vertica.linear_model.LinearModel.get_vertica_attributes``
        method.

    Examples
    ---------

    The following examples provide a basic understanding of usage.
    For more detailed examples, please refer to the
    :ref:`user_guide.machine_learning` or the
    `Examples <https://www.vertica.com/python/examples/>`_
    section on the website.

    Load data for machine learning
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    We import :py:mod:`verticapy`:

    .. code-block:: python

        import verticapy as vp

    .. hint::

        By assigning an alias to :py:mod:`verticapy`, we mitigate the risk
        of code collisions with other libraries. This precaution is
        necessary because verticapy uses commonly known function names
        like "average" and "median", which can potentially lead to naming
        conflicts. The use of an alias ensures that the functions from
        verticapy are used as intended without interfering with functions
        from other libraries.

    For this example, we will use the winequality dataset.

    .. code-block:: python

        import verticapy.datasets as vpd

        data = vpd.load_winequality()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_winequality.html

    .. note::

        VerticaPy offers a wide range of sample datasets that are
        ideal for training and testing purposes. You can explore
        the full list of available datasets in the :ref:`api.datasets`,
        which provides detailed information on each dataset
        and how to use them effectively. These datasets are invaluable
        resources for honing your data analysis and machine learning
        skills within the VerticaPy environment.

    You can easily divide your dataset into training and testing subsets
    using the :py:meth:`vDataFrame.train_test_split` method. This is a
    crucial step when preparing your data for machine learning, as it
    allows you to evaluate the performance of your models accurately.

    .. code-block:: python

        data = vpd.load_winequality()
        train, test = data.train_test_split(test_size = 0.2)

    .. warning::

        In this case, VerticaPy utilizes seeded randomization to guarantee
        the reproducibility of your data split. However, please be aware
        that this approach may lead to reduced performance. For a more
        efficient data split, you can use the :py:meth:`vDataFrame.to_db`
        method to save your results into ``tables`` or ``temporary tables``.
        This will help enhance the overall performance of the process.

    .. ipython:: python
        :suppress:

        import verticapy as vp
        import verticapy.datasets as vpd
        data = vpd.load_winequality()
        train, test = data.train_test_split(test_size = 0.2)

    Model Initialization
    ^^^^^^^^^^^^^^^^^^^^^

    First we import the ``LinearSVR`` model:

    .. code-block::

        from verticapy.machine_learning.vertica import LinearSVR

    Then we can create the model:

    .. code-block::

        model = LinearSVR(
            tol = 1e-4,
            C = 1.0,
            intercept_scaling = 1.0,
            intercept_mode = "regularized",
            acceptable_error_margin = 0.1,
            max_iter = 100,
        )

    .. hint::

        In :py:mod:`verticapy` 1.0.x and higher, you do not need to specify the
        model name, as the name is automatically assigned. If you need to
        re-use the model, you can fetch the model name from the model's
        attributes.

    .. important::

        The model name is crucial for the model management system and
        versioning. It's highly recommended to provide a name if you
        plan to reuse the model later.

    .. ipython:: python
        :suppress:

        from verticapy.machine_learning.vertica import LinearSVR
        model = LinearSVR(
            tol = 1e-4,
            C = 1.0,
            intercept_scaling = 1.0,
            intercept_mode = "regularized",
            acceptable_error_margin = 0.1,
            max_iter = 100,
        )

    Model Training
    ^^^^^^^^^^^^^^^

    We can now fit the model:

    .. ipython:: python

        model.fit(
            train,
            [
                "fixed_acidity",
                "volatile_acidity",
                "citric_acid",
                "residual_sugar",
                "chlorides",
                "density",
            ],
            "quality",
            test,
        )

    .. important::

        To train a model, you can directly use the :py:class:`vDataFrame` or the
        name of the relation stored in the database. The test set is optional
        and is only used to compute the test metrics. In :py:mod:`verticapy`, we
        don't work using ``X`` matrices and ``y`` vectors. Instead, we work
        directly with lists of predictors and the response name.

    Features Importance
    ^^^^^^^^^^^^^^^^^^^^

    We can conveniently get the features importance:

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")
        fig = model.features_importance()
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_lsvr_feature.html")

    .. code-block:: python

        result = model.features_importance()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_lsvr_feature.html

    .. note::

        For ``LinearModel``, feature importance is computed using the coefficients.
        These coefficients are then normalized using the feature distribution. An
        activation function is applied to get the final score.

    Metrics
    ^^^^^^^^

    We can get the entire report using:

    .. ipython:: python
        :suppress:

        result = model.report()
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_lsvr_report.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_lsvr_report.html

    .. important::

        Most metrics are computed using a single SQL query, but some of them might
        require multiple SQL queries. Selecting only the necessary metrics in the
        report can help optimize performance.
        E.g. ``model.report(metrics = ["mse", "r2"])``.

    For ``LinearModel``, we can easily get the ANOVA table using:

    .. ipython:: python
        :suppress:

        result = model.report(metrics = "anova")
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_lsvr_report_anova.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report(metrics = "anova")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_lsvr_report_anova.html

    You can also use the ``LinearModel.score`` function to compute the R-squared
    value:

    .. ipython:: python

        model.score()

    Prediction
    ^^^^^^^^^^^

    Prediction is straight-forward:

    .. ipython:: python
        :suppress:

        result = model.predict(
            test,
            [
                "fixed_acidity",
                "volatile_acidity",
                "citric_acid",
                "residual_sugar",
                "chlorides",
                "density",
            ],
            "prediction",
        )
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_lsvr_prediction.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.predict(
            test,
            [
                "fixed_acidity",
                "volatile_acidity",
                "citric_acid",
                "residual_sugar",
                "chlorides",
                "density",
            ],
            "prediction",
        )

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_lsvr_prediction.html

    .. note::

        Predictions can be made automatically using the test set, in which
        case you don't need to specify the predictors. Alternatively, you
        can pass only the :py:class:`vDataFrame` to the
        :py:meth:`verticapy.machine_learning.vertica.linear_model.LinearModel.predict`
        function, but in this case, it's essential that the column names of
        the :py:class:`vDataFrame` match the predictors and response name in the
        model.

    Plots
    ^^^^^^

    If the model allows, you can also generate relevant plots. For example,
    regression plots can be found in the :ref:`chart_gallery.regression_plot`.

    .. code-block:: python

        model.plot()

    .. important::

        The plotting feature is typically suitable for models with fewer than
        three predictors.

    **Contour plot** is another useful plot that can be produced
    for models with two predictors.

    .. code-block:: python

        model.contour()

    .. important::

        Machine learning models with two predictors can usually
        benefit from their own contour plot. This visual representation
        aids in exploring predictions and gaining a deeper understanding
        of how these models perform in different scenarios.
        Please refer to  :ref:`chart_gallery.contour` for more examples.

    Parameter Modification
    ^^^^^^^^^^^^^^^^^^^^^^^

    In order to see the parameters:

    .. ipython:: python

        model.get_params()

    And to manually change some of the parameters:

    .. ipython:: python

        model.set_params({'tol': 0.001})

    Model Register
    ^^^^^^^^^^^^^^

    In order to register the model for tracking and versioning:

    .. code-block:: python

        model.register("model_v1")

    Please refer to :ref:`notebooks/ml/model_tracking_versioning/index.html`
    for more details on model tracking and versioning.

    Model Exporting
    ^^^^^^^^^^^^^^^^

    **To Memmodel**

    .. code-block:: python

        model.to_memmodel()

    .. note::

        ``MemModel`` objects serve as in-memory representations of machine
        learning models. They can be used for both in-database and in-memory
        prediction tasks. These objects can be pickled in the same way that
        you would pickle a ``scikit-learn`` model.

    The following methods for exporting the model use ``MemModel``, and it
    is recommended to use ``MemModel`` directly.

    **To SQL**

    You can get the SQL code by:

    .. ipython:: python

        model.to_sql()

    **To Python**

    To obtain the prediction function in Python syntax, use the following code:

    .. ipython:: python

        X = [[4.2, 0.17, 0.36, 1.8, 0.029, 0.9899]]
        model.to_python()(X)

    .. hint::

        The
        :py:meth:`verticapy.machine_learning.vertica.linear_model.LinearModel.to_python`
        method is used to retrieve predictions,
        probabilities, or cluster distances. For specific details on how to
        use this method for different model types, refer to the relevant
        documentation for each model.
    """

    # Properties.

    @property
    def _vertica_fit_sql(self) -> Literal["SVM_REGRESSOR"]:
        return "SVM_REGRESSOR"

    @property
    def _vertica_predict_sql(self) -> Literal["PREDICT_SVM_REGRESSOR"]:
        return "PREDICT_SVM_REGRESSOR"

    @property
    def _model_subcategory(self) -> Literal["REGRESSOR"]:
        return "REGRESSOR"

    @property
    def _model_type(self) -> Literal["LinearSVR"]:
        return "LinearSVR"

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str = None,
        overwrite_model: bool = False,
        tol: float = 1e-4,
        C: float = 1.0,
        intercept_scaling: float = 1.0,
        intercept_mode: Literal["regularized", "unregularized"] = "regularized",
        acceptable_error_margin: float = 0.1,
        max_iter: int = 100,
    ) -> None:
        super().__init__(name, overwrite_model)
        self.parameters = {
            "tol": tol,
            "C": C,
            "intercept_scaling": intercept_scaling,
            "intercept_mode": str(intercept_mode).lower(),
            "acceptable_error_margin": acceptable_error_margin,
            "max_iter": max_iter,
        }

    # Parameters Methods.

    @staticmethod
    def _map_to_vertica_param_dict() -> dict:
        return {
            "tol": "epsilon",
            "max_iter": "max_iterations",
            "acceptable_error_margin": "error_tolerance",
        }


"""
Algorithms used for classification.
"""


class LinearSVC(LinearModelClassifier, BinaryClassifier):
    """
    Creates  a LinearSVC object  using the  Vertica
    Support Vector Machine  (SVM)  algorithm on the
    data. Given a set of training examples, where
    each is marked as belonging to one of two
    categories, an SVM training algorithm builds a
    model that assigns new examples to one category
    or  the other,  making it  a  non-probabilistic
    binary linear classifier.

    Parameters
    ----------
    name: str, optional
        Name  of the  model. The model is stored
        in the database.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    tol: float, optional
        Tolerance for stopping criteria. This is
        used to control accuracy.
    C: float, optional
        Weight for misclassification cost.  The
        algorithm minimizes the regularization cost
        and the misclassification cost.
    intercept_scaling: float
        A  float  value,  serves as  the  value of a
        dummy feature whose coefficient Vertica uses
        to calculate the model intercept.
        Because  the  dummy  feature  is not in  the
        training  data,  its  values  are  set to  a
        constant, by default set to 1.
    intercept_mode: str, optional
        Specify how to treat the intercept.

        - regularized:
            Fits  the intercept  and applies a
            regularization.
        - unregularized:
            Fits the  intercept  but does not  include
            it in regularization.
    class_weight: str / list, optional
        Specifies how to determine weights for the two
        classes.  It can be a  list of 2 elements  or
        one of the following methods:

        - auto:
            Weights  each class  according  to
            the number of samples.
        - none:
            No weights are used.
    max_iter: int, optional
        The  maximum  number of iterations  that  the
        algorithm performs.

    Attributes
    ----------
    Many attributes are created during the fitting phase.

    coef_: numpy.array
        The regression coefficients. The order of
        coefficients is the same as the order of
        columns used during the fitting phase.
    intercept_: float
        The expected value of the dependent variable
        when all independent variables are zero,
        serving as the baseline or constant term in
        the model.
    features_importance_: numpy.array
        The importance of features is computed through
        the model coefficients, which are normalized
        based on their range. Subsequently, an
        activation function calculates the final score.
        It is necessary to use the
        :py:meth:`verticapy.machine_learning.vertica.linear_model.LinearModel.features_importance`
        method to compute it initially, and the computed
        values will be subsequently utilized for subsequent
        calls.
    classes_: numpy.array
        The classes labels.

    .. note::

        All attributes can be accessed using the
        :py:meth:`verticapy.machine_learning.vertica.linear_model.LinearModel.get_attributes``
        method.

    .. note::

        Several other attributes can be accessed by using the
        :py:meth:`verticapy.machine_learning.vertica.linear_model.LinearModel.get_vertica_attributes``
        method.

    Examples
    ---------

    The following examples provide a basic understanding of usage.
    For more detailed examples, please refer to the
    :ref:`user_guide.machine_learning` or the
    `Examples <https://www.vertica.com/python/examples/>`_
    section on the website.

    Load data for machine learning
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    We import :py:mod:`verticapy`:

    .. code-block:: python

        import verticapy as vp

    .. hint::

        By assigning an alias to :py:mod:`verticapy`, we mitigate the risk of code
        collisions with other libraries. This precaution is necessary
        because verticapy uses commonly known function names like "average"
        and "median", which can potentially lead to naming conflicts.
        The use of an alias ensures that the functions from verticapy are
        used as intended without interfering with functions from other
        libraries.

    For this example, we will use the winequality dataset.

    .. code-block:: python

        import verticapy.datasets as vpd

        data = vpd.load_winequality()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_winequality.html

    .. note::

        VerticaPy offers a wide range of sample datasets that are
        ideal for training and testing purposes. You can explore
        the full list of available datasets in the :ref:`api.datasets`,
        which provides detailed information on each dataset
        and how to use them effectively. These datasets are invaluable
        resources for honing your data analysis and machine learning
        skills within the VerticaPy environment.

    You can easily divide your dataset into training and testing subsets
    using the :py:meth:`vDataFrame.train_test_split` method. This is a
    crucial step when preparing your data for machine learning, as it
    allows you to evaluate the performance of your models accurately.

    .. code-block:: python

        data = vpd.load_winequality()
        train, test = data.train_test_split(test_size = 0.2)

    .. warning::

        In this case, VerticaPy utilizes seeded randomization to guarantee
        the reproducibility of your data split. However, please be aware
        that this approach may lead to reduced performance. For a more
        efficient data split, you can use the :py:meth:`vDataFrame.to_db`
        method to save your results into ``tables`` or ``temporary tables``.
        This will help enhance the overall performance of the process.

    .. ipython:: python
        :suppress:

        import verticapy as vp
        import verticapy.datasets as vpd
        data = vpd.load_winequality()
        train, test = data.train_test_split(test_size = 0.2)

    Model Initialization
    ^^^^^^^^^^^^^^^^^^^^^

    First we import the ``LinearSVC`` model:

    .. code-block::

        from verticapy.machine_learning.vertica import LinearSVC

    .. ipython:: python
        :suppress:

        from verticapy.machine_learning.vertica import LinearSVC

    Then we can create the model:

    .. ipython:: python

        model = LinearSVC(
            tol = 1e-4,
            C = 1.0,
            intercept_scaling = 1.0,
            intercept_mode = "regularized",
            class_weight = [1, 1],
            max_iter = 100,
        )

    .. hint::

        In :py:mod:`verticapy` 1.0.x and higher, you do not need to specify the
        model name, as the name is automatically assigned. If you need to
        re-use the model, you can fetch the model name from the model's
        attributes.

    .. important::

        The model name is crucial for the model management system and
        versioning. It's highly recommended to provide a name if you
        plan to reuse the model later.



    Model Training
    ^^^^^^^^^^^^^^^

    We can now fit the model:

    .. ipython:: python

        model.fit(
            train,
            [
                "fixed_acidity",
                "volatile_acidity",
                "citric_acid",
                "residual_sugar",
                "chlorides",
                "density"
            ],
            "good",
            test,
        )

    .. important::

        To train a model, you can directly use the :py:class:`vDataFrame` or the
        name of the relation stored in the database. The test set is optional
        and is only used to compute the test metrics. In :py:mod:`verticapy`, we
        don't work using ``X`` matrices and ``y`` vectors. Instead, we work
        directly with lists of predictors and the response name.

    Features Importance
    ^^^^^^^^^^^^^^^^^^^^

    We can conveniently get the features importance:

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")
        fig = model.features_importance()
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_lscv_feature.html")

    .. code-block:: python

        result = model.features_importance()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_lscv_feature.html

    .. note::

        For ``LinearModel``, feature importance is computed using the coefficients.
        These coefficients are then normalized using the feature distribution. An
        activation function is applied to get the final score.

    Metrics
    ^^^^^^^^

    We can get the entire report using:

    .. ipython:: python
        :suppress:

        result = model.report()
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_lscv_report.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_lscv_report.html

    .. important::

        Most metrics are computed using a single SQL query, but some of them might
        require multiple SQL queries. Selecting only the necessary metrics in the
        report can help optimize performance.
        E.g. ``model.report(metrics = ["auc", "accuracy"])``.

    For classification models, we can easily modify the ``cutoff`` to observe
    the effect on different metrics:

    .. ipython:: python
        :suppress:

        result = model.report(cutoff = 0.2)
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_lscv_report_cutoff.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report(cutoff = 0.2)

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_lscv_report_cutoff.html


    You can also use the ``LinearModel.score`` function to compute any
    classification metric. The default metric is the accuracy:

    .. ipython:: python

        model.score()

    Prediction
    ^^^^^^^^^^^

    Prediction is straight-forward:

    .. ipython:: python
        :suppress:

        result = model.predict(
            test,
            [
                "fixed_acidity",
                "volatile_acidity",
                "citric_acid",
                "residual_sugar",
                "chlorides",
                "density"
            ],
            "prediction",
        )
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_lscv_prediction.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.predict(
            test,
            [
                "fixed_acidity",
                "volatile_acidity",
                "citric_acid",
                "residual_sugar",
                "chlorides",
                "density"
            ],
            "prediction",
        )

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_lscv_prediction.html

    .. note::

        Predictions can be made automatically using the test set, in which
        case you don't need to specify the predictors. Alternatively, you
        can pass only the :py:class:`vDataFrame` to the
        :py:meth:`verticapy.machine_learning.vertica.linear_model.LinearModel.predict`
        function, but in this case, it's essential that the column names of
        the :py:class:`vDataFrame` match the predictors and response name in the
        model.

    Probabilities
    ^^^^^^^^^^^^^^

    It is also easy to get the model's probabilities:

    .. ipython:: python
        :suppress:

        result = model.predict_proba(
            test,
            [
                "fixed_acidity",
                "volatile_acidity",
                "citric_acid",
                "residual_sugar",
                "chlorides",
                "density"
            ],
            "prediction",
        )
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_lscv_proba.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.predict_proba(
            test,
            [
                "fixed_acidity",
                "volatile_acidity",
                "citric_acid",
                "residual_sugar",
                "chlorides",
                "density"
            ],
            "prediction",
        )

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_lscv_proba.html

    .. note::

        Probabilities are added to the vDataFrame, and VerticaPy uses the
        corresponding probability function in SQL behind the scenes. You
        can use the ``pos_label`` parameter to add only the probability
        of the selected category.

    Confusion Matrix
    ^^^^^^^^^^^^^^^^^

    You can obtain the confusion matrix of your choice by specifying
    the desired cutoff.

    .. ipython:: python

        model.confusion_matrix(cutoff = 0.5)

    .. note::

        In classification, the ``cutoff`` is a threshold value used to
        determine class assignment based on predicted probabilities or
        scores from a classification model. In binary classification,
        if the predicted probability for a specific class is greater
        than or equal to the cutoff, the instance is assigned to the
        positive class; otherwise, it is assigned to the negative class.
        Adjusting the cutoff allows for trade-offs between true positives
        and false positives, enabling the model to be optimized for
        specific objectives or to consider the relative costs of different
        classification errors. The choice of cutoff is critical for
        tailoring the model's performance to meet specific needs.

    Main Plots (Classification Curves)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Classification models allow for the creation of various plots that
    are very helpful in understanding the model, such as the ROC Curve,
    PRC Curve, Cutoff Curve, Gain Curve, and more.

    Most of the classification curves can be found in the
    :ref:`chart_gallery.classification_curve`.

    For example, let's draw the model's ROC curve.

    .. code-block:: python

        model.roc_curve()

    .. ipython:: python
        :suppress:

        fig = model.roc_curve()
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_lscv_roc.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_lscv_roc.html

    .. important::

        Most of the curves have a parameter called ``nbins``, which is essential
        for estimating metrics. The larger the ``nbins``, the more precise the
        estimation, but it can significantly impact performance. Exercise caution
        when increasing this parameter excessively.

    .. hint::

        In binary classification, various curves can be easily plotted. However,
        in multi-class classification, it's important to select the ``pos_label``
        , representing the class to be treated as positive when drawing the curve.

    Other Plots
    ^^^^^^^^^^^^

    If the model allows, you can also generate relevant plots.
    For example, classification plots can be found in the
    :ref:`chart_gallery.classification_plot`.

    .. code-block:: python

        model.plot()

    .. important::

        The plotting feature is typically suitable for models with
        fewer than three predictors.

    **Contour plot** is another useful plot that can be produced
    for models with two predictors.

    .. code-block:: python

        model.contour()

    .. important::

        Machine learning models with two predictors can usually
        benefit from their own contour plot. This visual representation
        aids in exploring predictions and gaining a deeper understanding
        of how these models perform in different scenarios.
        Please refer to  :ref:`chart_gallery.contour` for more examples.

    Parameter Modification
    ^^^^^^^^^^^^^^^^^^^^^^^

    In order to see the parameters:

    .. ipython:: python

        model.get_params()

    And to manually change some of the parameters:

    .. ipython:: python

        model.set_params({'tol': 0.001})

    Model Register
    ^^^^^^^^^^^^^^

    In order to register the model for tracking and versioning:

    .. code-block:: python

        model.register("model_v1")

    Please refer to :ref:`notebooks/ml/model_tracking_versioning/index.html`
    for more details on model tracking and versioning.

    Model Exporting
    ^^^^^^^^^^^^^^^^

    **To Memmodel**

    .. code-block:: python

        model.to_memmodel()

    .. note::

        ``MemModel`` objects serve as in-memory representations of machine
        learning models. They can be used for both in-database and in-memory
        prediction tasks. These objects can be pickled in the same way that
        you would pickle a ``scikit-learn`` model.

    The following methods for exporting the model use ``MemModel``, and it
    is recommended to use ``MemModel`` directly.

    **To SQL**

    You can get the SQL code by:

    .. ipython:: python

        model.to_sql()

    **To Python**

    To obtain the prediction function in Python syntax, use the following code:

    .. ipython:: python

        X = [[4.2, 0.17, 0.36, 1.8, 0.029, 0.9899]]
        model.to_python()(X)

    .. hint::

        The
        :py:meth:`verticapy.machine_learning.vertica.linear_model.LinearModel.to_python`
        method is used to retrieve predictions,
        probabilities, or cluster distances. For specific details on how to
        use this method for different model types, refer to the relevant
        documentation for each model.
    """

    # Properties.

    @property
    def _vertica_fit_sql(self) -> Literal["SVM_CLASSIFIER"]:
        return "SVM_CLASSIFIER"

    @property
    def _vertica_predict_sql(self) -> Literal["PREDICT_SVM_CLASSIFIER"]:
        return "PREDICT_SVM_CLASSIFIER"

    @property
    def _model_subcategory(self) -> Literal["CLASSIFIER"]:
        return "CLASSIFIER"

    @property
    def _model_type(self) -> Literal["LinearSVC"]:
        return "LinearSVC"

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str = None,
        overwrite_model: bool = False,
        tol: float = 1e-4,
        C: float = 1.0,
        intercept_scaling: float = 1.0,
        intercept_mode: Literal["regularized", "unregularized"] = "regularized",
        class_weight: Union[Literal["auto", "none"], list] = [1, 1],
        max_iter: int = 100,
    ) -> None:
        super().__init__(name, overwrite_model)
        self.parameters = {
            "tol": tol,
            "C": C,
            "intercept_scaling": intercept_scaling,
            "intercept_mode": str(intercept_mode).lower(),
            "class_weight": class_weight,
            "max_iter": max_iter,
        }

    # Parameters Methods.

    @staticmethod
    def _map_to_vertica_param_dict() -> dict[str, str]:
        return {
            "class_weights": "class_weight",
            "tol": "epsilon",
            "max_iter": "max_iterations",
        }

    # Plotting Methods.

    def plot(
        self,
        max_nb_points: int = 100,
        chart: Optional[PlottingObject] = None,
        **style_kwargs
    ) -> PlottingObject:
        """
        Draws the model.

        Parameters
        ----------
        max_nb_points: int
            Maximum  number of points to display.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any optional parameter to pass to the
            Plotting functions.

        Returns
        -------
        obj
            Plotting Object.
        """
        vpy_plt, kwargs = self.get_plotting_lib(
            class_name="SVMClassifierPlot",
            chart=chart,
            style_kwargs=style_kwargs,
        )
        return vpy_plt.SVMClassifierPlot(
            vdf=vDataFrame(self.input_relation),
            columns=self.X + [self.y],
            max_nb_points=max_nb_points,
            misc_data={"coef": np.concatenate(([self.intercept_], self.coef_))},
        ).draw(**kwargs)
