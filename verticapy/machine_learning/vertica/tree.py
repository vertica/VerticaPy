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
from typing import Literal, Union

from verticapy._typing import PythonNumber
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._vertica_version import check_minimum_version

from verticapy.machine_learning.vertica.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)

"""
Algorithms used for regression.
"""


class DecisionTreeRegressor(RandomForestRegressor):
    """
    A DecisionTreeRegressor consisting of a single tree.

    Parameters
    ----------
    name: str, optional
        Name of the model. The model is stored in the
        database.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    max_features: str / int, optional
        The number of randomly  chosen features from which
        to pick the best feature  to split on a given tree
        node.  It  can  be  an integer  or one of the  two
        following methods:

        - auto:
            square root of the total number of predictors.
        - max:
            number of predictors.
    max_leaf_nodes: PythonNumber, optional
        The maximum number of leaf nodes for a tree in the
        forest, an integer between 1 and 1e9, inclusive.
    max_depth: int, optional
        The maximum depth for growing each tree, an integer
        between 1 and 100, inclusive.
    min_samples_leaf: int, optional
        The minimum number of samples each branch must have
        after a node is split, an integer between 1 and 1e6,
        inclusive. Any split that results in fewer remaining
        samples is discarded.
    min_info_gain: PythonNumber, optional
        The  minimum  threshold  for including a  split,  a
        float between 0.0 and 1.0,  inclusive. A split with
        information  gain  less  than   this  threshold  is
        discarded.
    nbins: int, optional
        The number of bins to use for continuous  features,
        an integer between 2 and 1000, inclusive.

    Examples
    ---------

    The following examples provide a basic understanding of usage.
    For more detailed examples, please refer to the
    :ref:`user_guide.machine_learning` or the
    `Examples <https://www.vertica.com/python/examples/>`_
    section on the website.

    Load data for machine learning
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    We import ``verticapy``:

    .. code-block:: python

        import verticapy as vp

    .. hint::

        By assigning an alias to ``verticapy``, we mitigate the risk
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
    using the :py:mod:`vDataFrame.train_test_split` method. This is a
    crucial step when preparing your data for machine learning, as it
    allows you to evaluate the performance of your models accurately.

    .. code-block:: python

        data = vpd.load_winequality()
        train, test = data.train_test_split(test_size = 0.2)

    .. warning::

        In this case, VerticaPy utilizes seeded randomization to guarantee
        the reproducibility of your data split. However, please be aware
        that this approach may lead to reduced performance. For a more
        efficient data split, you can use the :py:mod:`vDataFrame.to_db`
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

    First we import the ``DecisionTreeRegressor`` model:

    .. ipython:: python

        from verticapy.machine_learning.vertica import DecisionTreeRegressor

    Then we can create the model:

    .. ipython:: python

        model = DecisionTreeRegressor(
            max_features = "auto",
            max_leaf_nodes = 32,
            max_depth = 3,
            min_samples_leaf = 5,
            min_info_gain = 0.0,
            nbins = 32
        )

    .. hint::

        In ``verticapy`` 1.0.x and higher, you do not need to specify the
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

        To train a model, you can directly use the ``vDataFrame`` or the
        name of the relation stored in the database. The test set is optional
        and is only used to compute the test metrics. In ``verticapy``, we
        don't work using ``X`` matrices and ``y`` vectors. Instead, we work
        directly with lists of predictors and the response name.

    Features Importance
    ^^^^^^^^^^^^^^^^^^^^

    We can conveniently get the features importance:

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")
        fig = model.features_importance()
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_dtreereg_feature.html")

    .. code-block:: python

        result = model.features_importance()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_dtreereg_feature.html

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
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_dtreereg_report.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_dtreereg_report.html

    .. important::

        Most metrics are computed using a single SQL query, but some of them might
        require multiple SQL queries. Selecting only the necessary metrics in the
        report can help optimize performance.
        E.g. ``model.report(metrics = ["mse", "r2"])``.

    For ``LinearModel``, we can easily get the ANOVA table using:

    .. ipython:: python
        :suppress:

        result = model.report(metrics = "anova")
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_dtreereg_report_anova.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report(metrics = "anova")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_dtreereg_report_anova.html

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
        html_file = open("figures/machine_learning_vertica_linear_model_dtreereg_prediction.html", "w")
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
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_dtreereg_prediction.html

    .. note::

        Predictions can be made automatically using the test set, in which
        case you don't need to specify the predictors. Alternatively, you
        can pass only the ``vDataFrame`` to the
        :py:mod:`verticapy.machine_learning.vertica.linear_model.LinearModel.predict`
        function, but in this case, it's essential that the column names of
        the ``vDataFrame`` match the predictors and response name in the
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

        model.set_params({'max_depth': 5})

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
        :py:mod:`verticapy.machine_learning.vertica.linear_model.LinearModel.to_python`
        method is used to retrieve predictions,
        probabilities, or cluster distances. For specific details on how to
        use this method for different model types, refer to the relevant
        documentation for each model.
    """

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str = None,
        overwrite_model: bool = False,
        max_features: Union[Literal["auto", "max"], int] = "auto",
        max_leaf_nodes: PythonNumber = 1e9,
        max_depth: int = 100,
        min_samples_leaf: int = 1,
        min_info_gain: PythonNumber = 0.0,
        nbins: int = 32,
    ) -> None:
        super().__init__(name, overwrite_model)
        self.parameters = {
            "n_estimators": 1,
            "max_features": max_features,
            "max_leaf_nodes": int(max_leaf_nodes),
            "sample": 1.0,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_info_gain": min_info_gain,
            "nbins": nbins,
        }


class DummyTreeRegressor(RandomForestRegressor):
    """
    A regressor that overfits the training data.
    These models are typically used as a control
    to compare with your other models.

    Parameters
    ----------
    name: str, optional
        Name of the model. The model is stored
        in the database.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.

    Examples
    ---------

    The following examples provide a basic understanding of usage.
    For more detailed examples, please refer to the
    :ref:`user_guide.machine_learning` or the
    `Examples <https://www.vertica.com/python/examples/>`_
    section on the website.

    Load data for machine learning
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    We import ``verticapy``:

    .. code-block:: python

        import verticapy as vp

    .. hint::

        By assigning an alias to ``verticapy``, we mitigate the risk
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
    using the :py:mod:`vDataFrame.train_test_split` method. This is a
    crucial step when preparing your data for machine learning, as it
    allows you to evaluate the performance of your models accurately.

    .. code-block:: python

        data = vpd.load_winequality()
        train, test = data.train_test_split(test_size = 0.2)

    .. warning::

        In this case, VerticaPy utilizes seeded randomization to guarantee
        the reproducibility of your data split. However, please be aware
        that this approach may lead to reduced performance. For a more
        efficient data split, you can use the :py:mod:`vDataFrame.to_db`
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

    First we import the ``DummyTreeRegressor`` model:

    .. ipython:: python

        from verticapy.machine_learning.vertica import DummyTreeRegressor

    Then we can create the model:

    .. ipython:: python

        model = DummyTreeRegressor()

    .. hint::

        In ``verticapy`` 1.0.x and higher, you do not need to specify the
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

        To train a model, you can directly use the ``vDataFrame`` or the
        name of the relation stored in the database. The test set is optional
        and is only used to compute the test metrics. In ``verticapy``, we
        don't work using ``X`` matrices and ``y`` vectors. Instead, we work
        directly with lists of predictors and the response name.

    Features Importance
    ^^^^^^^^^^^^^^^^^^^^

    We can conveniently get the features importance:

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")
        fig = model.features_importance()
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_dummytreereg_feature.html")

    .. code-block:: python

        result = model.features_importance()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_dummytreereg_feature.html

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
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_dummytreereg_report.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_dummytreereg_report.html

    .. important::

        Most metrics are computed using a single SQL query, but some of them might
        require multiple SQL queries. Selecting only the necessary metrics in the
        report can help optimize performance.
        E.g. ``model.report(metrics = ["mse", "r2"])``.

    For ``LinearModel``, we can easily get the ANOVA table using:

    .. ipython:: python
        :suppress:

        result = model.report(metrics = "anova")
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_dummytreereg_report_anova.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report(metrics = "anova")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_dummytreereg_report_anova.html

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
        html_file = open("figures/machine_learning_vertica_linear_model_dummytreereg_prediction.html", "w")
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
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_dummytreereg_prediction.html

    .. note::

        Predictions can be made automatically using the test set, in which
        case you don't need to specify the predictors. Alternatively, you
        can pass only the ``vDataFrame`` to the
        :py:mod:`verticapy.machine_learning.vertica.linear_model.LinearModel.predict`
        function, but in this case, it's essential that the column names of
        the ``vDataFrame`` match the predictors and response name in the
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
        :py:mod:`verticapy.machine_learning.vertica.linear_model.LinearModel.to_python`
        method is used to retrieve predictions,
        probabilities, or cluster distances. For specific details on how to
        use this method for different model types, refer to the relevant
        documentation for each model.
    """

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(self, name: str = None, overwrite_model: bool = False) -> None:
        super().__init__(name, overwrite_model)
        self.parameters = {
            "n_estimators": 1,
            "max_features": "max",
            "max_leaf_nodes": int(1e9),
            "sample": 1.0,
            "max_depth": 100,
            "min_samples_leaf": 1,
            "min_info_gain": 0.0,
            "nbins": 1000,
        }


"""
Algorithms used for classification.
"""


class DecisionTreeClassifier(RandomForestClassifier):
    """
    A DecisionTreeClassifier consisting of a single tree.

    Parameters
    ----------
    name: str, optional
        Name of the model. The model is stored in the
        database.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    max_features: str / int, optional
        The number of randomly  chosen features from which
        to pick the best feature  to split on a given tree
        node.  It  can  be  an integer  or one of the  two
        following methods.
            auto : square root of the total number of
                   predictors.
            max  : number of predictors.
    max_leaf_nodes: PythonNumber, optional
        The maximum number of leaf nodes for a tree in the
        forest, an integer between 1 and 1e9, inclusive.
    max_depth: int, optional
        The maximum depth for growing each tree, an integer
        between 1 and 100, inclusive.
    min_samples_leaf: int, optional
        The minimum number of samples each branch must have
        after a node is split, an integer between 1 and 1e6,
        inclusive. Any split that results in fewer remaining
        samples is discarded.
    min_info_gain: PythonNumber, optional
        The  minimum  threshold  for including a  split,  a
        float between 0.0 and 1.0,  inclusive. A split with
        information  gain  less  than   this  threshold  is
        discarded.
    nbins: int, optional
        The number of bins to use for continuous  features,
        an integer between 2 and 1000, inclusive.
    """

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str = None,
        overwrite_model: bool = False,
        max_features: Union[Literal["auto", "max"], int] = "auto",
        max_leaf_nodes: PythonNumber = 1e9,
        max_depth: int = 100,
        min_samples_leaf: int = 1,
        min_info_gain: PythonNumber = 0.0,
        nbins: int = 32,
    ) -> None:
        super().__init__(name, overwrite_model)
        self.parameters = {
            "n_estimators": 1,
            "max_features": max_features,
            "max_leaf_nodes": int(max_leaf_nodes),
            "sample": 1.0,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_info_gain": min_info_gain,
            "nbins": nbins,
        }


class DummyTreeClassifier(RandomForestClassifier):
    """
    A classifier that overfits the training data.
    These models are  typically used as a control
    to compare with your other models.

    Parameters
    ----------
    name: str, optional
        Name of  the  model. The model is stored
        in the database.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    """

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(self, name: str = None, overwrite_model: bool = False) -> None:
        super().__init__(name, overwrite_model)
        self.parameters = {
            "n_estimators": 1,
            "max_features": "max",
            "max_leaf_nodes": int(1e9),
            "sample": 1.0,
            "max_depth": 100,
            "min_samples_leaf": 1,
            "min_info_gain": 0.0,
            "nbins": 1000,
        }
