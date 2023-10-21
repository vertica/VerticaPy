"""
Copyright  (c)  2018-2023 Open Text  or  one  of its
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
    # Properties.

    @property
    def _attributes(self) -> list[str]:
        return ["coef_", "intercept_", "features_importance_"]

    # System & Special Methods.

    @abstractmethod
    def __init__(self) -> None:
        """Must be overridden in the child class"""
        return None
        # self.input_relation = None
        # self.test_relation = None
        # self.X = None
        # self.y = None
        # self.parameters = {}
        # for att in self._attributes:
        #    setattr(self, att, None)

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
    # Properties.

    @property
    def _attributes(self) -> list[str]:
        return ["coef_", "intercept_", "classes_", "features_importance_"]

    # System & Special Methods.

    @abstractmethod
    def __init__(self) -> None:
        """Must be overridden in the child class"""
        return None
        # self.input_relation = None
        # self.test_relation = None
        # self.X = None
        # self.y = None
        # self.classes_ = None

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


class ElasticNet(Regressor, LinearModel):
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
                newton : Newton Method.
                bfgs   : Broyden Fletcher Goldfarb Shanno.
                cgd    : Coordinate Gradient Descent.
    l1_ratio: float, optional
        ENet mixture parameter that defines the provided
        ratio of L1 versus L2 regularization.
    fit_intercept: bool, optional
        Boolean,  specifies whether the model includes an
        intercept. If set to false, no intercept is used
        in training the model.  Note that setting
        fit_intercept  to false does  not work well with
        the BFGS optimizer.
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


class Lasso(Regressor, LinearModel):
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


class LinearRegression(Regressor, LinearModel):
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
    
    Examples
    ---------


    Load data for machine learning
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    We import ``verticapy``:

    .. code-block:: python

        import verticapy as vp

    .. hint:: 

        By giving an alias to ``verticapy``, we avoid code colission with other libraries.
        This is verticapy uses some well-known functions names like average, median etc. 

    For this example, we will use the winequality dataset.

    .. code-block:: python

        import verticapy.datasets as vpd

        data = vpd.load_winequality()        


    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_winequality.html
    

    .. note:: 

        VerticaPy provides many sample datasets that can be used for training/testing purposes. Check out
        :ref:`` for all the available datasets.

    We can use the ``vDataFrame.train_test_split`` method to split-up the dataset into training and testing
    portions.

    .. code-block:: python

        data = vpd.load_winequality()
        train, test = data.train_test_split(test_size = 0.2)        

    .. warning:: 

        In this case, ``verticapy`` will use seeded random to ensure reproducibility. However,
        this process can drastically reduce the performance. In order to performa more efficient split
        you can use ``vDataFrame.to_db`` to save your result in to ``tables`` or ``temporary tables``.

    .. ipython:: python
        :suppress:

        import verticapy as vp
        import verticapy.datasets as vpd
        data = vpd.load_winequality()
        train, test = data.train_test_split(test_size = 0.2)



    Model Initialziation
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

        In ``verticapy`` 1.0 and higher, you do not need to specify the model name as the name is automatically
        assigned. If you need to re-use the model, you can fetch the model name from model attributes.

    .. important:: 

        The model name is really important for the model management system and versioning. It is highly recommended
        to give a name if you plan to re-use the model later. 

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
            ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "density"],
            "quality",
            test,
        )
    
    .. important::

        To train a model you can use directly use the ``vDataFrame`` or the name of the relation stored in
        the database. The test set is optional and is only used to compute the test metrics. In ``verticapy``
        we do not work using ``X`` matrix and ``y`` vectors. We work directly with list of predictors and response name.

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

        For ``LinearModel`` the features importance is computed using the coefficients. They are normalized
        using the feature distribution. Then an activation function is applied to get the final score. 

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

        Most metrics are computed using a single SQL query but some of them might need multiple SQL queries. 
        By selecting only the necessary metric in the report 
        you may be able to optimize performance. E.g. ``model.report(metrics = ["mse", "r2"])``.
    
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

    You can also use the ``score`` function to compute the R-squared value:

    .. ipython:: python

        model.score()
    
    Prediction
    ^^^^^^^^^^^

    Prediction is straight-forward: 

    .. ipython:: python
        :suppress:

        result = model.predict(
            test,
            ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "density"],
            "prediction",
        )
        html_file = open("figures/machine_learning_vertica_linear_model_lr_prediction.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()


    .. code-block:: python

        model.predict(
            test,
            ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "density"],
            "prediction",
        )

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_lr_prediction.html

    .. note:: 

        Prediction can be done automatically using the test set in which case you do not nee to 
        specify the predictors. It is also possible to pass only the the ``vDataFrame`` to the ``predict``
        function. But in this case, it is important that the column names of the ``vDataFrame`` are
        matching the predictors and response name in the model.

    Plots
    ^^^^^^

    If the models allows, you can also draw the relevant plots. Example of regression plots can 
    be found in :ref:`chart_gallery.regression_plot`.

    .. code-block:: python

        model.plot()    

    .. important:: The plot generally works for models which have less than 3 predictors.

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

    Please refer to :ref:`notebooks/ml/model_tracking_versioning/index.html` for more details on model tracking
    and versioning.

    Model Exporting
    ^^^^^^^^^^^^^^^^

    To Memmodel
    """ """""" """""" """"

    .. code-block:: python

        model.to_memmodel()

    .. note::

        ``Memmodel`` are in-memory representation of ML models. They can be used to do in-database and
        in-memory prediction. They can pickled the same way that you pickle your ``scikit-learn`` model.


    The following ways of exporting the model uses memmodel and it is perfered to use the ``memmodels`` directly.

    To SQL
    """ """""" """""" """"

    You can get the SQL code by:

    .. ipython:: python

        model.to_sql()

    To Python
    """ """""" """""" """"

    To get the prediction function in a python sytanx:

    .. ipython:: python

        X = [[4.2, 0.17, 0.36, 1.8, 0.029, 0.9899]]
        model.to_python()(X)
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


class Ridge(Regressor, LinearModel):
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


class LogisticRegression(BinaryClassifier, LinearModelClassifier):
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
