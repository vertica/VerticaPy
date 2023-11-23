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

    Attributes
    ----------
    Many attributes are created during the fitting phase.

    trees_: list of one BinaryTreeRegressor
        One tree model which is instance of
        ``BinaryTreeRegressor``. It possess various
        attributes. For more detailed information,
        refer to the documentation for
        :py:meth:`verticapy.machine_learning.memmodel.tree.BinaryTreeRegressor`.
    features_importance_: numpy.array
        The importance of features. It is calculated
        using the MDI (Mean Decreased Impurity). To
        determine the final score, VerticaPy sums the
        scores of each tree, normalizes them and applies
        an activation function to scale them.
        It is necessary to use the
        :py:meth:`verticapy.machine_learning.vertica.base.Tree.features_importance`
        method to compute it initially, and the computed
        values will be subsequently utilized for subsequent
        calls.

    .. note::

        All attributes can be accessed using the
        :py:meth:`verticapy.machine_learning.vertica.base.Tree.get_attributes``
        method.

    .. note::

        Several other attributes can be accessed by using the
        :py:meth:`verticapy.machine_learning.vertica.base.Tree.get_vertica_attributes``
        method.

    Examples
    ---------

    The following examples provide a basic understanding of usage.
    For more detailed examples, please refer to the
    :ref:`user_guide.machine_learning` or the
    `Examples <https://www.vertica.com/python/examples/>`_
    section on the website.

    .. important::

        Many tree-based models inherit from the ``RandomForest``
        base class, and it's recommended to use it directly for
        access to a wider range of options.

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

    First we import the ``DecisionTreeRegressor`` model:

    .. ipython:: python

        from verticapy.machine_learning.vertica import DecisionTreeRegressor

    Then we can create the model:

    .. ipython:: python
        :okwarning:

        model = DecisionTreeRegressor(
            max_features = "auto",
            max_leaf_nodes = 32,
            max_depth = 3,
            min_samples_leaf = 5,
            min_info_gain = 0.0,
            nbins = 32
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

    Features Importance
    ^^^^^^^^^^^^^^^^^^^^

    We can conveniently get the features importance:

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")
        fig = model.features_importance()
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_dtreereg_feature.html")

    .. code-block:: python

        result = model.features_importance()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_dtreereg_feature.html

    .. note::

        In models such as ``RandomForest``, feature importance is calculated
        using the MDI (Mean Decreased Impurity). To determine the final score,
        VerticaPy sums the scores of each tree, normalizes them and applies an
        activation function to scale them.

    Metrics
    ^^^^^^^^

    We can get the entire report using:

    .. ipython:: python
        :suppress:

        result = model.report()
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_dtreereg_report.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_dtreereg_report.html

    .. important::

        Most metrics are computed using a single SQL query, but some of them might
        require multiple SQL queries. Selecting only the necessary metrics in the
        report can help optimize performance.
        E.g. ``model.report(metrics = ["mse", "r2"])``.

    You can utilize the
    :py:meth:`verticapy.machine_learning.vertica.ensemble.RandomForestRegressor.score`
    function to calculate various regression metrics, with the R-squared being the default.

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
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_dtreereg_prediction.html", "w")
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
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_dtreereg_prediction.html

    .. note::

        Predictions can be made automatically using the test set, in which
        case you don't need to specify the predictors. Alternatively, you
        can pass only the :py:class:`vDataFrame` to the
        :py:meth:`verticapy.machine_learning.vertica.ensemble.RandomForestRegressor.predict`
        function, but in this case, it's essential that the column names of
        the :py:class:`vDataFrame` match the predictors and response name in the
        model.

    Plots
    ^^^^^^

    Tree models can be visualized by drawing their tree plots.
    For more examples, check out :ref:`chart_gallery.tree`.

    .. code-block:: python

        model.plot_tree()

    .. ipython:: python
        :suppress:

        res = model.plot_tree()
        res.render(filename='figures/machine_learning_vertica_tree_decision', format='png')

    .. image:: /../figures/machine_learning_vertica_tree_decision.png

    .. note::

        The above example may not render properly in the doc because
        of the huge size of the tree. But it should render nicely
        in jupyter environment.

    In order to plot graph using `graphviz <https://graphviz.org/>`_
    separately, you can extract the graphviz DOT file code as follows:

    .. ipython:: python

        model.to_graphviz()

    This string can then be copied into a DOT file which can be
    parsed by graphviz.

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
        :py:meth:`verticapy.machine_learning.vertica.tree.DecisionTreeRegressor.to_python`
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

    Attributes
    ----------
    Many attributes are created during the fitting phase.

    trees_: list of one BinaryTreeRegressor
        One tree model which is instance of
        ``BinaryTreeRegressor``. It possess various
        attributes. For more detailed information,
        refer to the documentation for
        :py:meth:`verticapy.machine_learning.memmodel.tree.BinaryTreeRegressor`.
    features_importance_: numpy.array
        The importance of features. It is calculated
        using the MDI (Mean Decreased Impurity). To
        determine the final score, VerticaPy sums the
        scores of each tree, normalizes them and applies
        an activation function to scale them.
        It is necessary to use the
        :py:meth:`verticapy.machine_learning.vertica.base.Tree.features_importance`
        method to compute it initially, and the computed
        values will be subsequently utilized for subsequent
        calls.

    .. note::

        All attributes can be accessed using the
        :py:meth:`verticapy.machine_learning.vertica.base.Tree.get_attributes``
        method.

    .. note::

        Several other attributes can be accessed by using the
        :py:meth:`verticapy.machine_learning.vertica.base.Tree.get_vertica_attributes``
        method.

    Examples
    ---------

    The following examples provide a basic understanding of usage.
    For more detailed examples, please refer to the
    :ref:`user_guide.machine_learning` or the
    `Examples <https://www.vertica.com/python/examples/>`_
    section on the website.

    .. important::

        Many tree-based models inherit from the ``RandomForest``
        base class, and it's recommended to use it directly for
        access to a wider range of options.

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

    First we import the ``DummyTreeRegressor`` model:

    .. ipython:: python

        from verticapy.machine_learning.vertica import DummyTreeRegressor

    Then we can create the model:

    .. ipython:: python

        model = DummyTreeRegressor()

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

    Features Importance
    ^^^^^^^^^^^^^^^^^^^^

    We can conveniently get the features importance:

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")
        fig = model.features_importance()
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_dummytreereg_feature.html")

    .. code-block:: python

        result = model.features_importance()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_dummytreereg_feature.html

    .. note::

        In models such as ``RandomForest``, feature importance is calculated
        using the MDI (Mean Decreased Impurity). To determine the final score,
        VerticaPy sums the scores of each tree, normalizes them and applies an
        activation function to scale them.

    Metrics
    ^^^^^^^^

    We can get the entire report using:

    .. ipython:: python
        :suppress:

        result = model.report()
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_dummytreereg_report.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_dummytreereg_report.html

    .. important::

        Most metrics are computed using a single SQL query, but some of them might
        require multiple SQL queries. Selecting only the necessary metrics in the
        report can help optimize performance.
        E.g. ``model.report(metrics = ["mse", "r2"])``.

    You can utilize the
    :py:meth:`verticapy.machine_learning.vertica.ensemble.RandomForestRegressor.score`
    function to calculate various regression metrics, with the R-squared being the default.

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
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_dummytreereg_prediction.html", "w")
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
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_dummytreereg_prediction.html

    .. note::

        Predictions can be made automatically using the test set, in which
        case you don't need to specify the predictors. Alternatively, you
        can pass only the :py:class:`vDataFrame` to the
        :py:meth:`verticapy.machine_learning.vertica.ensemble.RandomForestRegressor.predict`
        function, but in this case, it's essential that the column names of
        the :py:class:`vDataFrame` match the predictors and response name in the
        model.

    Plots
    ^^^^^^

    Tree models can be visualized by drawing their tree plots.
    For more examples, check out :ref:`chart_gallery.tree`.

    .. code-block:: python

        model.plot_tree()

    .. ipython:: python
        :suppress:

        res = model.plot_tree()
        res.render(filename='figures/machine_learning_vertica_tree_dummy', format='png')

    .. image:: /../figures/machine_learning_vertica_tree_dummy.png

    .. note::

        The above example may not render properly in the doc because
        of the huge size of the tree. But it should render nicely
        in jupyter environment.

    In order to plot graph using `graphviz <https://graphviz.org/>`_
    separately, you can extract the graphviz DOT file code as follows:

    .. ipython:: python

        model.to_graphviz()

    This string can then be copied into a DOT file which can be
    parsed by graphviz.

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
        :py:meth:`verticapy.machine_learning.vertica.tree.DummyTreeRegressor.to_python`
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

    Attributes
    ----------
    Many attributes are created during the fitting phase.

    trees_: list of one BinaryTreeClassifier
        One tree model which is instance of
        ``BinaryTreeClassifier``. It possess various
        attributes. For more detailed information,
        refer to the documentation for
        :py:meth:`verticapy.machine_learning.memmodel.tree.BinaryTreeClassifier`.
    features_importance_: numpy.array
        The importance of features. It is calculated
        using the MDI (Mean Decreased Impurity). To
        determine the final score, VerticaPy sums the
        scores of each tree, normalizes them and applies
        an activation function to scale them.
        It is necessary to use the
        :py:meth:`verticapy.machine_learning.vertica.base.Tree.features_importance`
        method to compute it initially, and the computed
        values will be subsequently utilized for subsequent
        calls.
    classes_: numpy.array
        The classes labels.

    .. note::

        All attributes can be accessed using the
        :py:meth:`verticapy.machine_learning.vertica.base.Tree.get_attributes``
        method.

    .. note::

        Several other attributes can be accessed by using the
        :py:meth:`verticapy.machine_learning.vertica.base.Tree.get_vertica_attributes``
        method.

    Examples
    ---------

    The following examples provide a basic understanding of usage.
    For more detailed examples, please refer to the
    :ref:`user_guide.machine_learning` or the
    `Examples <https://www.vertica.com/python/examples/>`_
    section on the website.

    .. important::

        Many tree-based models inherit from the ``RandomForest``
        base class, and it's recommended to use it directly for
        access to a wider range of options.

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

    First we import the ``DecisionTreeClassifier`` model:

    .. code-block::

        from verticapy.machine_learning.vertica import DecisionTreeClassifier

    .. ipython:: python
        :suppress:

        from verticapy.machine_learning.vertica import DecisionTreeClassifier

    Then we can create the model:

    .. ipython:: python
        :okwarning:

        model = DecisionTreeClassifier(
            max_features = "auto",
            max_leaf_nodes = 32,
            max_depth = 3,
            min_samples_leaf = 5,
            min_info_gain = 0.0,
            nbins = 32,
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
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_dtreeclass_feature.html")

    .. code-block:: python

        result = model.features_importance()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_dtreeclass_feature.html

    .. note::

        In models such as ``RandomForest``, feature importance is calculated
        using the MDI (Mean Decreased Impurity). To determine the final score,
        VerticaPy sums the scores of each tree, normalizes them and applies an
        activation function to scale them.

    Metrics
    ^^^^^^^^

    We can get the entire report using:

    .. ipython:: python
        :suppress:

        result = model.report()
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_dtreeclass_report.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_dtreeclass_report.html

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
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_dtreeclass_report_cutoff.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report(cutoff = 0.2)

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_dtreeclass_report_cutoff.html


    You can also use the
    :py:meth:`verticapy.machine_learning.vertica.ensemble.RandomForestClassifier.score`
    function to compute any classification metric. The default metric is the accuracy:

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
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_dtreeclass_prediction.html", "w")
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
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_dtreeclass_prediction.html

    .. note::

        Predictions can be made automatically using the test set, in which
        case you don't need to specify the predictors. Alternatively, you
        can pass only the :py:class:`vDataFrame` to the
        :py:meth:`verticapy.machine_learning.vertica.ensemble.RandomForestClassifier.predict`
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
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_dtreeclass_proba.html", "w")
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
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_dtreeclass_proba.html

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
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_dtreeclass_roc.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_dtreeclass_roc.html

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

    Tree models can be visualized by drawing their tree plots.
    For more examples, check out :ref:`chart_gallery.tree`.

    .. code-block:: python

        model.plot_tree()

    .. ipython:: python
        :suppress:

        res = model.plot_tree()
        res.render(filename='figures/machine_learning_vertica_tree_decision_classifier', format='png')

    .. image:: /../figures/machine_learning_vertica_tree_decision_classifier.png

    .. note::

        The above example may not render properly in the doc because
        of the huge size of the tree. But it should render nicely
        in jupyter environment.

    In order to plot graph using `graphviz <https://graphviz.org/>`_
    separately, you can extract the graphviz DOT file code as follows:

    .. ipython:: python

        model.to_graphviz()

    This string can then be copied into a DOT file which can be
    parsed by graphviz.

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
        :py:meth:`verticapy.machine_learning.vertica.tree.DecisionTreeClassifier.to_python`
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

    Attributes
    ----------
    Many attributes are created during the fitting phase.

    trees_: list of one BinaryTreeClassifier
        One tree model which is instance of
        ``BinaryTreeClassifier``. It possess various
        attributes. For more detailed information,
        refer to the documentation for
        :py:meth:`verticapy.machine_learning.memmodel.tree.BinaryTreeClassifier`.
    features_importance_: numpy.array
        The importance of features. It is calculated
        using the MDI (Mean Decreased Impurity). To
        determine the final score, VerticaPy sums the
        scores of each tree, normalizes them and applies
        an activation function to scale them.
        It is necessary to use the
        :py:meth:`verticapy.machine_learning.vertica.base.Tree.features_importance`
        method to compute it initially, and the computed
        values will be subsequently utilized for subsequent
        calls.
    classes_: numpy.array
        The classes labels.

    .. note::

        All attributes can be accessed using the
        :py:meth:`verticapy.machine_learning.vertica.base.Tree.get_attributes``
        method.

    .. note::

        Several other attributes can be accessed by using the
        :py:meth:`verticapy.machine_learning.vertica.base.Tree.get_vertica_attributes``
        method.

    Examples
    ---------

    The following examples provide a basic understanding of usage.
    For more detailed examples, please refer to the
    :ref:`user_guide.machine_learning` or the
    `Examples <https://www.vertica.com/python/examples/>`_
    section on the website.

    .. important::

        Many tree-based models inherit from the ``RandomForest``
        base class, and it's recommended to use it directly for
        access to a wider range of options.

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

    First we import the ``DummyTreeClassifier`` model:

    .. code-block::

        from verticapy.machine_learning.vertica import DummyTreeClassifier

    .. ipython:: python
        :suppress:

        from verticapy.machine_learning.vertica import DummyTreeClassifier

    Then we can create the model:

    .. ipython:: python
        :okwarning:

        model = DummyTreeClassifier()


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
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_dummytreecl_feature.html")

    .. code-block:: python

        result = model.features_importance()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_dummytreecl_feature.html

    .. note::

        In models such as ``RandomForest``, feature importance is calculated
        using the MDI (Mean Decreased Impurity). To determine the final score,
        VerticaPy sums the scores of each tree, normalizes them and applies an
        activation function to scale them.

    Metrics
    ^^^^^^^^

    We can get the entire report using:

    .. ipython:: python
        :suppress:

        result = model.report()
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_dummytreecl_report.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_dummytreecl_report.html

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
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_dummytreecl_report_cutoff.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report(cutoff = 0.2)

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_dummytreecl_report_cutoff.html


    You can also use the
    :py:meth:`verticapy.machine_learning.vertica.ensemble.RandomForestClassifier.score`
    function to compute any classification metric. The default metric is the accuracy:

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
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_dummytreecl_prediction.html", "w")
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
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_dummytreecl_prediction.html

    .. note::

        Predictions can be made automatically using the test set, in which
        case you don't need to specify the predictors. Alternatively, you
        can pass only the :py:class:`vDataFrame` to the
        :py:meth:`verticapy.machine_learning.vertica.ensemble.RandomForestClassifier.predict`
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
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_dummytreecl_proba.html", "w")
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
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_dummytreecl_proba.html

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
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_dummytreecl_roc.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_dummytreecl_roc.html

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

    Tree models can be visualized by drawing their tree plots.
    For more examples, check out :ref:`chart_gallery.tree`.

    .. code-block:: python

        model.plot_tree()

    .. ipython:: python
        :suppress:

        res = model.plot_tree()
        res.render(filename='figures/machine_learning_vertica_tree_decision_classifier', format='png')

    .. image:: /../figures/machine_learning_vertica_tree_decision_classifier.png

    .. note::

        The above example may not render properly in the doc because
        of the huge size of the tree. But it should render nicely
        in jupyter environment.

    In order to plot graph using `graphviz <https://graphviz.org/>`_
    separately, you can extract the graphviz DOT file code as follows:

    .. ipython:: python

        model.to_graphviz()

    This string can then be copied into a DOT file which can be
    parsed by graphviz.

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
        :py:meth:`verticapy.machine_learning.vertica.tree.DummyTreeClassifier.to_python`
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
