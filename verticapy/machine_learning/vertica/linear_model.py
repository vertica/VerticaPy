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
from abc import abstractmethod
from typing import Literal, Optional

import numpy as np

from verticapy._typing import PlottingObject, PythonNumber
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import quote_ident
from verticapy._utils._sql._sys import _executeSQL
from verticapy._utils._sql._vertica_version import (
    check_minimum_version,
    vertica_version,
)
from verticapy.errors import VersionError

from verticapy.core.tablesample.base import TableSample
from verticapy.core.vdataframe.base import vDataFrame

import verticapy.machine_learning.memmodel as mm
from verticapy.machine_learning.vertica.base import Regressor, BinaryClassifier


"""
General Classes.
"""


class LinearModel:
    """
    Base Class for Vertica Linear Models.
    """

    # Properties.

    @property
    def _attributes(self) -> list[str]:
        return ["coef_", "intercept_", "features_importance_"]

    # Attributes Methods.

    def _compute_attributes(self) -> None:
        """
        Computes the model's attributes.
        """
        details = self.get_vertica_attributes("details")
        self.coef_ = np.array(details["coefficient"][1:])
        self.intercept_ = details["coefficient"][0]

    # Features Importance Methods.

    def _compute_features_importance(self) -> None:
        """
        Computes the features importance.
        """
        vertica_version(condition=[8, 1, 1])
        query = f"""
        SELECT /*+LABEL('learn.VerticaModel.features_importance')*/
            predictor, 
            sign * ROUND(100 * importance / SUM(importance) OVER(), 2) AS importance
        FROM (SELECT 
                stat.predictor AS predictor, 
                ABS(coefficient * (max - min))::float AS importance, 
                SIGN(coefficient)::int AS sign 
              FROM (SELECT 
                        LOWER("column") AS predictor, 
                        min, 
                        max 
                    FROM (SELECT 
                            SUMMARIZE_NUMCOL({', '.join(self.X)}) OVER() 
                          FROM {self.input_relation}) VERTICAPY_SUBTABLE) stat 
                          NATURAL JOIN (SELECT GET_MODEL_ATTRIBUTE(
                                                USING PARAMETERS 
                                                model_name='{self.model_name}',
                                                attr_name='details') ) coeff) importance_t 
                          ORDER BY 2 DESC;"""
        importance = _executeSQL(
            query=query, title="Computing Features Importance.", method="fetchall"
        )
        self.features_importance_ = self._format_vector(self.X, importance)

    def _get_features_importance(self) -> np.ndarray:
        """
        Returns the features' importance.
        """
        if not hasattr(self, "features_importance_"):
            self._compute_features_importance()
        return copy.deepcopy(self.features_importance_)

    def features_importance(
        self, show: bool = True, chart: Optional[PlottingObject] = None, **style_kwargs
    ) -> PlottingObject:
        """
        Computes the model's features importance.

        Parameters
        ----------
        show: bool
            If set to True,  draw the feature's importance.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any optional parameter to pass to the Plotting
            functions.

        Returns
        -------
        obj
            features importance.
        """
        fi = self._get_features_importance()
        if show:
            data = {
                "importance": fi,
            }
            layout = {"columns": copy.deepcopy(self.X)}
            vpy_plt, kwargs = self.get_plotting_lib(
                class_name="ImportanceBarChart",
                chart=chart,
                style_kwargs=style_kwargs,
            )
            return vpy_plt.ImportanceBarChart(data=data, layout=layout).draw(**kwargs)
        importances = {
            "index": [quote_ident(x)[1:-1].lower() for x in self.X],
            "importance": list(abs(fi)),
            "sign": list(np.sign(fi)),
        }
        return TableSample(values=importances).sort(column="importance", desc=True)

    # I/O Methods.

    def to_memmodel(self) -> mm.LinearModel:
        """
        Converts  the model to an InMemory object  that
        can be used for different types of predictions.
        """
        return mm.LinearModel(self.coef_, self.intercept_)

    # Plotting Methods.

    def plot(
        self,
        max_nb_points: int = 100,
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the model.

        Parameters
        ----------
        max_nb_points: int
            Maximum number of points to display.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any optional parameter to pass to the
            Plotting functions.

        Returns
        -------
        object
            Plotting Object.
        """
        vpy_plt, kwargs = self.get_plotting_lib(
            class_name="RegressionPlot",
            chart=chart,
            style_kwargs=style_kwargs,
        )
        return vpy_plt.RegressionPlot(
            vdf=vDataFrame(self.input_relation),
            columns=self.X + [self.y],
            max_nb_points=max_nb_points,
            misc_data={"coef": np.concatenate(([self.intercept_], self.coef_))},
        ).draw(**kwargs)


class LinearModelClassifier(LinearModel):
    """
    Base Class for Vertica Linear Models Classifiers.
    """

    # Properties.

    @property
    def _attributes(self) -> list[str]:
        return ["coef_", "intercept_", "classes_", "features_importance_"]

    # Attributes Methods.

    def _compute_attributes(self) -> None:
        """
        Computes the model's attributes.
        """
        details = self.get_vertica_attributes("details")
        self.coef_ = np.array(details["coefficient"][1:])
        self.intercept_ = details["coefficient"][0]

    # I/O Methods.

    def to_memmodel(self) -> mm.LinearModelClassifier:
        """
        Converts  the  model  to an InMemory object that
        can be used for different types of predictions.
        """
        return mm.LinearModelClassifier(self.coef_, self.intercept_)

    # Plotting Methods.

    def plot(
        self,
        max_nb_points: int = 100,
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the model.

        Parameters
        ----------
        max_nb_points: int
            Maximum number of points to display.
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
            class_name="LogisticRegressionPlot",
            chart=chart,
            style_kwargs=style_kwargs,
        )
        return vpy_plt.LogisticRegressionPlot(
            vdf=vDataFrame(self.input_relation),
            columns=self.X + [self.y],
            max_nb_points=max_nb_points,
            misc_data={"coef": np.concatenate(([self.intercept_], self.coef_))},
        ).draw(**kwargs)


"""
Algorithms used for regression.
"""


class ElasticNet(LinearModel, Regressor):
    """
    Creates an ElasticNet object using the Vertica
    Linear Regression  algorithm. The Elastic Net
    is a regularized regression method that linearly
    combines the L1 and L2 penalties of the Lasso and
    Ridge methods.

    Parameters
    ----------
    name: str, optional
        Name of the  model.  The model is stored in
        the database.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    tol: float, optional
        Determines  whether the algorithm has reached
        the specified accuracy result.
    C: PythonNumber, optional
        The regularization parameter value. The value
        must be zero or non-negative.
    max_iter: int, optional
        Determines  the maximum number of  iterations
        the  algorithm performs before  achieving the
        specified accuracy result.
    solver: str, optional
        The optimizer method used to train the model.

        - newton:
            Newton Method.
        - bfgs:
            Broyden Fletcher Goldfarb Shanno.
        - cgd:
            Coordinate Gradient Descent.
    l1_ratio: float, optional
        ENet mixture parameter that defines the provided
        ratio of L1 versus L2 regularization.
    fit_intercept: bool, optional
        Boolean,  specifies whether the model includes an
        intercept. If set to false, no intercept is used
        in training the model.  Note that setting
        fit_intercept  to false does  not work well with
        the BFGS optimizer.

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

    First we import the ``ElasticNet`` model:

    .. code-block::

        from verticapy.machine_learning.vertica import ElasticNet

    Then we can create the model:

    .. code-block::

        model = ElasticNet(
            tol = 1e-6,
            C = 1,
            max_iter = 100,
            solver = 'CGD',
            l1_ratio = 0.5,
            fit_intercept = True,
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

        from verticapy.machine_learning.vertica import ElasticNet
        model = ElasticNet(
            tol = 1e-6,
            C = 1,
            max_iter = 100,
            solver = 'CGD',
            l1_ratio = 0.5,
            fit_intercept = True,
        )

    Model Training
    ^^^^^^^^^^^^^^^

    We can now fit the model:

    .. ipython:: python
        :okwarning:

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
            "quality",
            test,
        )

    .. important::

        To train a model, you can directly use the :py:class:`vDataFrame` or the
        name of the relation stored in the database. The test set is optional
        and is only used to compute the test metrics. In :py:mod:`verticapy`, we
        don't work using ``X`` matrices and ``y`` vectors. Instead, we work
        directly with lists of predictors and the response name.

    Metrics
    ^^^^^^^^

    We can get the entire report using:

    .. ipython:: python
        :suppress:

        result = model.report()
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_elasticnet_report.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_elasticnet_report.html

    .. important::

        Most metrics are computed using a single SQL query, but some of them might
        require multiple SQL queries. Selecting only the necessary metrics in the
        report can help optimize performance.
        E.g. ``model.report(metrics = ["mse", "r2"])``.

    For ``LinearModel``, we can easily get the ANOVA table using:

    .. ipython:: python
        :suppress:

        result = model.report(metrics = "anova")
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_elasticnet_report_anova.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report(metrics = "anova")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_elasticnet_report_anova.html

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
                "density"
            ],
            "prediction",
        )
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_elasticnet_prediction.html", "w")
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
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_elasticnet_prediction.html

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

    Please refer to :ref:`notebooks/ml/model_tracking_versioning/index.html` for
    more details on model tracking and versioning.

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
    def _vertica_fit_sql(self) -> Literal["LINEAR_REG"]:
        return "LINEAR_REG"

    @property
    def _vertica_predict_sql(self) -> Literal["PREDICT_LINEAR_REG"]:
        return "PREDICT_LINEAR_REG"

    @property
    def _model_subcategory(self) -> Literal["REGRESSOR"]:
        return "REGRESSOR"

    @property
    def _model_type(self) -> Literal["LinearRegression"]:
        return "LinearRegression"

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str = None,
        overwrite_model: bool = False,
        tol: float = 1e-6,
        C: PythonNumber = 1.0,
        max_iter: int = 100,
        solver: Literal["newton", "bfgs", "cgd"] = "cgd",
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
    ) -> None:
        super().__init__(name, overwrite_model)
        if vertica_version()[0] < 12 and not fit_intercept:
            raise VersionError(
                "The parameter 'fit_intercept' can be activated for "
                "Vertica versions greater or equal to 12."
            )
        self.parameters = {
            "penalty": "enet",
            "tol": tol,
            "C": C,
            "max_iter": max_iter,
            "solver": str(solver).lower(),
            "l1_ratio": l1_ratio,
            "fit_intercept": fit_intercept,
        }


class Lasso(LinearModel, Regressor):
    """
    Creates  a  Lasso  object using the  Vertica
    Linear  Regression  algorithm.
    Lasso is a regularized regression method
    that uses an L1 penalty.

    Parameters
    ----------
    name: str, optional
        Name of the model. The  model is stored in the
        database.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    tol: float, optional
        Determines  whether the  algorithm has reached
        the specified accuracy result.
    C: PythonNumber, optional
        The regularization  parameter value. The value
        must be zero or non-negative.
    max_iter: int, optional
        Determines  the  maximum number of  iterations
        the  algorithm  performs before achieving  the
        specified accuracy result.
    solver: str, optional
        The optimizer method used to train the model.
                newton : Newton Method.
                bfgs   : Broyden Fletcher Goldfarb Shanno.
                cgd    : Coordinate Gradient Descent.
    fit_intercept: bool, optional
        Boolean,  specifies  whether the model includes an
        intercept.  If set to false, no intercept is
        used in  training the model.  Note that setting
        fit_intercept to false does not work well with the
        BFGS optimizer.

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

    First we import the ``Lasso`` model:

    .. code-block::

        from verticapy.machine_learning.vertica import Lasso

    Then we can create the model:

    .. code-block::

        model = Lasso(
            tol = 1e-6,
            C = 0.5,
            max_iter = 100,
            solver = 'CGD',
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

        from verticapy.machine_learning.vertica import Lasso
        model = Lasso(
            tol = 1e-6,
            C = 0.5,
            max_iter = 100,
            solver = 'CGD',
        )

    Model Training
    ^^^^^^^^^^^^^^^

    We can now fit the model:

    .. ipython:: python
        :okwarning:

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
            "quality",
            test,
        )

    .. important::

        To train a model, you can directly use the :py:class:`vDataFrame` or the
        name of the relation stored in the database. The test set is optional
        and is only used to compute the test metrics. In :py:mod:`verticapy`, we
        don't work using ``X`` matrices and ``y`` vectors. Instead, we work
        directly with lists of predictors and the response name.

    Metrics
    ^^^^^^^^

    We can get the entire report using:

    .. ipython:: python
        :suppress:

        result = model.report()
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_lasso_report.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_lasso_report.html

    .. important::

        Most metrics are computed using a single SQL query, but some of them might
        require multiple SQL queries. Selecting only the necessary metrics in the
        report can help optimize performance.
        E.g. ``model.report(metrics = ["mse", "r2"])``.

    For ``LinearModel``, we can easily get the ANOVA table using:

    .. ipython:: python
        :suppress:

        result = model.report(metrics = "anova")
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_lasso_report_anova.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report(metrics = "anova")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_lasso_report_anova.html

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
                "density"
            ],
            "prediction",
        )
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_lasso_prediction.html", "w")
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
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_lasso_prediction.html

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

    Please refer to :ref:`notebooks/ml/model_tracking_versioning/index.html` for
    more details on model tracking and versioning.

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
    def _vertica_fit_sql(self) -> Literal["LINEAR_REG"]:
        return "LINEAR_REG"

    @property
    def _vertica_predict_sql(self) -> Literal["PREDICT_LINEAR_REG"]:
        return "PREDICT_LINEAR_REG"

    @property
    def _model_subcategory(self) -> Literal["REGRESSOR"]:
        return "REGRESSOR"

    @property
    def _model_type(self) -> Literal["LinearRegression"]:
        return "LinearRegression"

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str = None,
        overwrite_model: bool = False,
        tol: float = 1e-6,
        C: PythonNumber = 1.0,
        max_iter: int = 100,
        solver: Literal["newton", "bfgs", "cgd"] = "cgd",
        fit_intercept: bool = True,
    ) -> None:
        super().__init__(name, overwrite_model)
        if vertica_version()[0] < 12 and not fit_intercept:
            raise VersionError(
                "The parameter 'fit_intercept' can be activated for "
                "Vertica versions greater or equal to 12."
            )
        self.parameters = {
            "penalty": "l1",
            "tol": tol,
            "C": C,
            "max_iter": max_iter,
            "solver": str(solver).lower(),
            "fit_intercept": fit_intercept,
        }


class LinearRegression(LinearModel, Regressor):
    """
    Creates a LinearRegression object using the Vertica
    Linear Regression algorithm.

    Parameters
    ----------
    name: str, optional
        Name of the model. The  model is stored  in the
        database.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    tol: float, optional
        Determines whether the  algorithm has reached the
        specified accuracy result.
    max_iter: int, optional
        Determines the  maximum number of  iterations the
        algorithm performs before achieving the specified
        accuracy result.
    solver: str, optional
        The optimizer method used to train the model.
                newton : Newton Method.
                bfgs   : Broyden Fletcher Goldfarb Shanno.
    fit_intercept: bool, optional
        Boolean,  specifies  whether the model includes an
        intercept. If set to false,  no intercept is
        used in  training the model.  Note that setting
        fit_intercept to false does not work well with the
        BFGS optimizer.

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
        It is necessary to use the :py:meth:`verticapy.machine_learning.vertica.linear_model.LinearModel.features_importance`method
        to compute it initially, and the computed values
        will be subsequently utilized for subsequent
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

    First we import the ``LinearRegression`` model:

    .. code-block::

        from verticapy.machine_learning.vertica import LinearRegression

    Then we can create the model:

    .. code-block::

        model = LinearRegression(
            tol = 1e-6,
            max_iter = 100,
            solver = 'Newton',
            fit_intercept = True,
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

        from verticapy.machine_learning.vertica import LinearRegression
        model = LinearRegression(
            tol = 1e-6,
            max_iter = 100,
            solver = 'Newton',
            fit_intercept = True,
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
                "density"
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
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_lr_feature.html")

    .. code-block:: python

        result = model.features_importance()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_lr_feature.html

    .. note::

        For ``LinearModel``, feature importance is computed using
        the coefficients. These coefficients are then normalized using the
        feature distribution. An activation function is applied to
        get the final score.

    Metrics
    ^^^^^^^^

    We can get the entire report using:

    .. ipython:: python
        :suppress:

        result = model.report()
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_lr_report.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        result = model.report()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_lr_report.html

    .. important::

        Most metrics are computed using a single SQL query, but some
        of them might require multiple SQL queries. Selecting only the
        necessary metrics in the report can help optimize performance.
        E.g. ``model.report(metrics = ["mse", "r2"])``.

    For ``LinearModel``, we can easily get the ANOVA table using:

    .. ipython:: python
        :suppress:

        result = model.report(metrics = "anova")
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_lr_report_anova.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        result = model.report(metrics = "anova")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_lr_report_anova.html

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
                "density"
            ],
            "prediction",
        )
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_lr_prediction.html", "w")
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
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_lr_prediction.html

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

        The plotting feature is typically suitable for models with fewer
        than three predictors.

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

    Please refer to :ref:`notebooks/ml/model_tracking_versioning/index.html` for
    more details on model tracking and versioning.

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
        probabilities, or cluster distances. For specific details on how
        to use this method for different model types, refer to the
        relevant documentation for each model.
    """

    # Properties.

    @property
    def _vertica_fit_sql(self) -> Literal["LINEAR_REG"]:
        return "LINEAR_REG"

    @property
    def _vertica_predict_sql(self) -> Literal["PREDICT_LINEAR_REG"]:
        return "PREDICT_LINEAR_REG"

    @property
    def _model_subcategory(self) -> Literal["REGRESSOR"]:
        return "REGRESSOR"

    @property
    def _model_type(self) -> Literal["LinearRegression"]:
        return "LinearRegression"

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str = None,
        overwrite_model: bool = False,
        tol: float = 1e-6,
        max_iter: int = 100,
        solver: Literal["newton", "bfgs"] = "newton",
        fit_intercept: bool = True,
    ) -> None:
        super().__init__(name, overwrite_model)
        if vertica_version()[0] < 12 and not fit_intercept:
            raise VersionError(
                "The parameter 'fit_intercept' can be activated for "
                "Vertica versions greater or equal to 12."
            )
        self.parameters = {
            "penalty": "none",
            "tol": tol,
            "max_iter": max_iter,
            "solver": str(solver).lower(),
            "fit_intercept": fit_intercept,
        }


class PoissonRegressor(LinearModel, Regressor):
    """
    Creates an PoissonRegressor object using the
    Vertica Poisson Regression algorithm.

    Parameters
    ----------
    name: str, optional
        Name of the  model.  The model is stored in
        the database.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    tol: float, optional
        Determines  whether the algorithm has reached
        the specified accuracy result.
    penalty: str, optional
        Method of regularization.

        - None:
            No regularization.
        - L2:
            L2 regularization.
    C: PythonNumber, optional
        The regularization parameter value. The value
        must be zero or non-negative.
    max_iter: int, optional
        Determines  the maximum number of  iterations
        the  algorithm performs before  achieving the
        specified accuracy result.
    solver: str, optional
        The optimizer method used to train the model.

        - newton:
            Newton Method.
    fit_intercept: bool, optional
        Boolean,  specifies whether the model includes an
        intercept. If set to false, no intercept is used
        in training the model.  Note that setting
        fit_intercept  to false does  not work well with
        the BFGS optimizer.

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
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

    First we import the ``PoissonRegressor`` model:

    .. code-block::

        from verticapy.machine_learning.vertica import PoissonRegressor

    Then we can create the model:

    .. code-block::

        model = PoissonRegressor(
            tol = 1e-6,
            penalty = 'L2',
            C = 1,
            max_iter = 100,
            fit_intercept = True,
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

        from verticapy.machine_learning.vertica import PoissonRegressor
        model = PoissonRegressor(
            tol = 1e-6,
            penalty = 'L2',
            C = 1,
            max_iter = 100,
            fit_intercept = True,
        )

    Model Training
    ^^^^^^^^^^^^^^^

    We can now fit the model:

    .. ipython:: python
        :okwarning:

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
            "quality",
            test,
        )

    .. important::

        To train a model, you can directly use the :py:class:`vDataFrame` or the
        name of the relation stored in the database. The test set is optional
        and is only used to compute the test metrics. In :py:mod:`verticapy`, we
        don't work using ``X`` matrices and ``y`` vectors. Instead, we work
        directly with lists of predictors and the response name.

    Metrics
    ^^^^^^^^

    We can get the entire report using:

    .. ipython:: python
        :suppress:

        result = model.report()
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_poisson_report.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_poisson_report.html

    .. important::

        Most metrics are computed using a single SQL query, but some of them might
        require multiple SQL queries. Selecting only the necessary metrics in the
        report can help optimize performance.
        E.g. ``model.report(metrics = ["mse", "r2"])``.

    For ``LinearModel``, we can easily get the ANOVA table using:

    .. ipython:: python
        :suppress:

        result = model.report(metrics = "anova")
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_poisson_report_anova.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report(metrics = "anova")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_poisson_report_anova.html

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
                "density"
            ],
            "prediction",
        )
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_poisson_prediction.html", "w")
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
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_poisson_prediction.html

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
    def _vertica_fit_sql(self) -> Literal["POISSON_REG"]:
        return "POISSON_REG"

    @property
    def _vertica_predict_sql(self) -> Literal["PREDICT_POISSON_REG"]:
        return "PREDICT_POISSON_REG"

    @property
    def _model_subcategory(self) -> Literal["REGRESSOR"]:
        return "REGRESSOR"

    @property
    def _model_type(self) -> Literal["PoissonRegressor"]:
        return "PoissonRegressor"

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str = None,
        overwrite_model: bool = False,
        tol: float = 1e-6,
        penalty: Literal["none", "l2", None] = "none",
        C: PythonNumber = 1.0,
        max_iter: int = 100,
        solver: Literal["newton"] = "newton",
        fit_intercept: bool = True,
    ) -> None:
        super().__init__(name, overwrite_model)
        penalty = str(penalty).lower()
        solver = str(solver).lower()
        self.parameters = {
            "penalty": penalty,
            "tol": tol,
            "C": C,
            "max_iter": max_iter,
            "solver": solver,
            "fit_intercept": fit_intercept,
        }


class Ridge(LinearModel, Regressor):
    """
    Creates  a  Ridge  object using the  Vertica
    Linear  Regression  algorithm.
    Ridge is a regularized regression method
    which uses an L2 penalty.

    Parameters
    ----------
    name: str, optional
        Name of the model. The model is stored in the
        database.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    tol: float, optional
        Determines whether  the algorithm has reached
        the specified accuracy result.
    C: PythonNumber, optional
        The regularization parameter value. The value
        must be zero or non-negative.
    max_iter: int, optional
        Determines  the maximum number of  iterations
        the  algorithm performs before  achieving the
        specified accuracy result.
    solver: str, optional
        The optimizer method used to train the model.
                newton : Newton Method.
                bfgs   : Broyden Fletcher Goldfarb Shanno.
    fit_intercept: bool, optional
        Boolean,  specifies whether the model  includes
        an intercept. If set to false, no intercept
        is used in training the model.
        Note  that setting fit_intercept to false  does
        not work well with the BFGS optimizer.

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

    First we import the ``Ridge`` model:

    .. code-block::

        from verticapy.machine_learning.vertica import Ridge

    Then we can create the model:

    .. code-block::

        model = Ridge(
            tol = 1e-6,
            C = 0.5,
            max_iter = 100,
            solver = 'Newton',
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

        from verticapy.machine_learning.vertica import Ridge
        model = Ridge(
            tol = 1e-6,
            C = 0.5,
            max_iter = 100,
            solver = 'Newton',
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
                "density"
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
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_ridge_feature.html")

    .. code-block:: python

        result = model.features_importance()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_ridge_feature.html

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
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_ridge_report.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_ridge_report.html

    .. important::

        Most metrics are computed using a single SQL query, but some of them might
        require multiple SQL queries. Selecting only the necessary metrics in the
        report can help optimize performance.
        E.g. ``model.report(metrics = ["mse", "r2"])``.

    For ``LinearModel``, we can easily get the ANOVA table using:

    .. ipython:: python
        :suppress:

        result = model.report(metrics = "anova")
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_ridge_report_anova.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report(metrics = "anova")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_ridge_report_anova.html

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
                "density"
            ],
            "prediction",
        )
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_ridge_prediction.html", "w")
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
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_ridge_prediction.html

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
    def _vertica_fit_sql(self) -> Literal["LINEAR_REG"]:
        return "LINEAR_REG"

    @property
    def _vertica_predict_sql(self) -> Literal["PREDICT_LINEAR_REG"]:
        return "PREDICT_LINEAR_REG"

    @property
    def _model_subcategory(self) -> Literal["REGRESSOR"]:
        return "REGRESSOR"

    @property
    def _model_type(self) -> Literal["LinearRegression"]:
        return "LinearRegression"

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str = None,
        overwrite_model: bool = False,
        tol: float = 1e-6,
        C: PythonNumber = 1.0,
        max_iter: int = 100,
        solver: Literal["newton", "bfgs"] = "newton",
        fit_intercept: bool = True,
    ) -> None:
        super().__init__(name, overwrite_model)
        if vertica_version()[0] < 12 and not fit_intercept:
            raise VersionError(
                "The parameter 'fit_intercept' can be activated for "
                "Vertica versions greater or equal to 12."
            )
        self.parameters = {
            "penalty": "l2",
            "tol": tol,
            "C": C,
            "max_iter": max_iter,
            "solver": str(solver).lower(),
            "fit_intercept": fit_intercept,
        }


"""
Algorithms used for classification.
"""


class LogisticRegression(LinearModelClassifier, BinaryClassifier):
    """
    Creates a LogisticRegression  object using the Vertica
    Logistic Regression algorithm.

    Parameters
    ----------
    name: str, optional
        Name of the model.  The model is stored in the
        database.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    penalty: str, optional
        Determines the method of regularization.
            None : No Regularization.
            l1   : L1 Regularization.
            l2   : L2 Regularization.
            enet : Combination between L1 and L2.
    tol: float, optional
        Determines  whether  the  algorithm has reached  the
        specified accuracy result.
    C: PythonNumber, optional
        The  regularization parameter value.  The value must
        be zero or non-negative.
    max_iter: int, optional
        Determines  the  maximum number  of  iterations  the
        algorithm  performs  before achieving  the specified
        accuracy result.
    solver: str, optional
        The  optimizer method  used to train the  model.
            newton : Newton Method.
            bfgs   : Broyden Fletcher Goldfarb Shanno.
            cgd    : Coordinate Gradient Descent.
    l1_ratio: float, optional
        ENet  mixture parameter  that  defines the provided
        ratio of L1 versus L2 regularization.
    fit_intercept: bool, optional
        Boolean,  specifies  whether  the model  includes an
        intercept.
        If set to false,  no intercept is used in
        training the model.  Note that setting fit_intercept
        to false does not work well with the BFGS optimizer.

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

    Balancing the Dataset
    ^^^^^^^^^^^^^^^^^^^^^^

    In VerticaPy, balancing a dataset to address class imbalances
    is made straightforward through the
    :py:meth:`verticapy.machine_learning.vertica.preprocessing.balance`
    function within the ``preprocessing`` module. This function
    enables users to rectify skewed class distributions efficiently.
    By specifying the target variable and setting parameters like
    the method for balancing, users can effortlessly achieve a more
    equitable representation of classes in their dataset.
    Whether opting for over-sampling, under-sampling, or a combination
    of both, VerticaPy's
    :py:meth:`verticapy.machine_learning.vertica.preprocessing.balance`
    function streamlines the process, empowering users to enhance the
    performance and fairness of their machine learning models trained
    on imbalanced data.

    To balance the dataset, use the following syntax.

    .. code-block:: python

        from verticapy.machine_learning.vertica.preprocessing import balance

        balanced_train = balance(
            name = "my_schema.train_balanced",
            input_relation = train,
            y = "good",
            method = "hybrid",
        )

    .. note::

        With this code, a table named `train_balanced` is created in the
        `my_schema` schema. It can then be used to train the model. In the
        rest of the example, we will work with the full dataset.

    .. hint::

        Balancing the dataset is a crucial step in improving the accuracy
        of machine learning models, particularly when faced with imbalanced
        class distributions. By addressing disparities in the number of
        instances across different classes, the model becomes more adept at
        learning patterns from all classes rather than being biased towards
        the majority class. This, in turn, enhances the model's ability to
        make accurate predictions for under-represented classes. The balanced
        dataset ensures that the model is not dominated by the majority class
        and, as a result, leads to more robust and unbiased model performance.
        Therefore, by employing techniques such as over-sampling, under-sampling,
        or a combination of both during dataset preparation, practitioners can
        significantly contribute to achieving higher accuracy and better
        generalization of their machine learning models.

    Model Initialization
    ^^^^^^^^^^^^^^^^^^^^^

    First we import the ``LogisticRegression`` model:

    .. code-block::

        from verticapy.machine_learning.vertica import LogisticRegression

    Then we can create the model:

    .. code-block::

        model = LogisticRegression(
            tol = 1e-6,
            max_iter = 100,
            solver = 'Newton',
            fit_intercept = True,
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

        from verticapy.machine_learning.vertica import LogisticRegression
        model = LogisticRegression(
            tol = 1e-6,
            max_iter = 100,
            solver = 'Newton',
            fit_intercept = True,
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
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_logr_feature.html")

    .. code-block:: python

        result = model.features_importance()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_logr_feature.html

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
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_logr_report.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_logr_report.html

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
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_logr_report_cutoff.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report(cutoff = 0.2)

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_logr_report_cutoff.html


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
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_logr_prediction.html", "w")
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
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_logr_prediction.html

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
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_logr_proba.html", "w")
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
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_logr_proba.html

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
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_logr_roc.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_logr_roc.html

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
    def _vertica_fit_sql(self) -> Literal["LOGISTIC_REG"]:
        return "LOGISTIC_REG"

    @property
    def _vertica_predict_sql(self) -> Literal["PREDICT_LOGISTIC_REG"]:
        return "PREDICT_LOGISTIC_REG"

    @property
    def _model_subcategory(self) -> Literal["CLASSIFIER"]:
        return "CLASSIFIER"

    @property
    def _model_type(self) -> Literal["LogisticRegression"]:
        return "LogisticRegression"

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str = None,
        overwrite_model: bool = False,
        penalty: Literal["none", "l1", "l2", "enet", None] = "none",
        tol: float = 1e-6,
        C: PythonNumber = 1.0,
        max_iter: int = 100,
        solver: Literal["newton", "bfgs", "cgd"] = "newton",
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
    ) -> None:
        super().__init__(name, overwrite_model)
        penalty = str(penalty).lower()
        solver = str(solver).lower()
        if vertica_version()[0] < 12 and not fit_intercept:
            raise VersionError(
                "The parameter 'fit_intercept' can be activated for "
                "Vertica versions greater or equal to 12."
            )
        self.parameters = {
            "penalty": penalty,
            "tol": tol,
            "C": C,
            "max_iter": max_iter,
            "solver": solver,
            "l1_ratio": l1_ratio,
            "fit_intercept": fit_intercept,
        }
        if str(penalty).lower() == "none":
            del self.parameters["l1_ratio"]
            del self.parameters["C"]
            if "solver" == "cgd":
                raise ValueError(
                    "solver can not be set to 'cgd' when there is no regularization."
                )
        elif str(penalty).lower() in ("l1", "l2"):
            del self.parameters["l1_ratio"]
