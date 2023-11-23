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
import random
from typing import Literal, Optional, Union
import numpy as np

from vertica_python.errors import MissingRelation, QueryError

from verticapy._typing import (
    NoneType,
    PlottingObject,
    PythonNumber,
    SQLColumns,
    SQLRelation,
)
from verticapy._utils._gen import gen_name
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import clean_query, format_type, quote_ident
from verticapy._utils._sql._sys import _executeSQL
from verticapy._utils._sql._vertica_version import check_minimum_version

from verticapy.core.vdataframe.base import vDataFrame

import verticapy.machine_learning.memmodel as mm
from verticapy.machine_learning.vertica.base import (
    MulticlassClassifier,
    Regressor,
    Tree,
)
from verticapy.machine_learning.vertica.cluster import Clustering

"""
General Classes.
"""


class RandomForest(Tree):
    # Properties.

    @property
    def _model_importance_function(self) -> Literal["RF_PREDICTOR_IMPORTANCE"]:
        return "RF_PREDICTOR_IMPORTANCE"

    @property
    def _model_importance_feature(self) -> Literal["IMPORTANCE_VALUE"]:
        return "IMPORTANCE_VALUE"


class XGBoost(Tree):
    # Properties.

    @property
    def _model_importance_function(self) -> Literal["XGB_PREDICTOR_IMPORTANCE"]:
        return "XGB_PREDICTOR_IMPORTANCE"

    @property
    def _model_importance_feature(self) -> Literal["AVG_GAIN"]:
        return "AVG_GAIN"

    # Attributes Methods.

    def _compute_prior(self) -> Union[float, list[float]]:
        """
        Returns the XGB priors.

        Returns
        -------
        float / list
            XGB priors.
        """
        condition = [f"{x} IS NOT NULL" for x in self.X] + [f"{self.y} IS NOT NULL"]
        query = f"""
            SELECT 
                /*+LABEL('learn.ensemble.XGBoost._compute_prior')*/ 
                {{}}
            FROM {self.input_relation} 
            WHERE {' AND '.join(condition)}{{}}"""
        if self._model_type == "XGBRegressor" or (
            len(self.classes_) == 2 and self.classes_[1] == 1 and self.classes_[0] == 0
        ):
            prior_ = _executeSQL(
                query=query.format(f"AVG({self.y})", ""),
                method="fetchfirstelem",
                print_time_sql=False,
            )
        else:
            prior_ = np.array([0.0 for p in self.classes_])
        return prior_

    # I/O Methods.

    def _to_json_tree_dict(self, tree_id: int, c: str = None) -> dict:
        """
        Method used to convert the model to JSON.
        """
        tree = self.get_tree(tree_id)
        attributes = self._compute_trees_arrays(tree, self.X)
        n_nodes = len(attributes[0])
        split_conditions = []
        parents = [0 for i in range(n_nodes)]
        parents[0] = random.randint(n_nodes + 1, 999999999)
        for i in range(n_nodes):
            left_child = attributes[0][i]
            right_child = attributes[1][i]
            if left_child != right_child:
                parents[left_child] = i
                parents[right_child] = i
            if attributes[5][i]:
                split_conditions += [attributes[3][i]]
            elif isinstance(attributes[5][i], NoneType):
                if self._model_type == "XGBRegressor":
                    split_conditions += [
                        float(attributes[4][i]) * self.parameters["learning_rate"]
                    ]
                elif (
                    len(self.classes_) == 2
                    and self.classes_[1] == 1
                    and self.classes_[0] == 0
                ):
                    split_conditions += [
                        self.parameters["learning_rate"] * float(attributes[6][i]["1"])
                    ]
                else:
                    split_conditions += [
                        self.parameters["learning_rate"] * float(attributes[6][i][c])
                    ]
            else:
                split_conditions += [float(attributes[3][i])]
        return {
            "base_weights": [0.0 for i in range(n_nodes)],
            "categories": [],
            "categories_nodes": [],
            "categories_segments": [],
            "categories_sizes": [],
            "default_left": [True for i in range(n_nodes)],
            "id": tree_id,
            "left_children": [-1 if x is None else x for x in attributes[0]],
            "loss_changes": [0.0 for i in range(n_nodes)],
            "parents": parents,
            "right_children": [-1 if x is None else x for x in attributes[1]],
            "split_conditions": split_conditions,
            "split_indices": [0 if x is None else x for x in attributes[2]],
            "split_type": [
                int(x) if isinstance(x, bool) else int(attributes[5][0])
                for x in attributes[5]
            ],
            "sum_hessian": [0.0 for i in range(n_nodes)],
            "tree_param": {
                "num_deleted": "0",
                "num_feature": str(len(self.X)),
                "num_nodes": str(n_nodes),
                "size_leaf_vector": "0",
            },
        }

    def _to_json_tree_dict_list(self) -> dict:
        """
        Method used to convert the model to JSON.
        """
        if self._model_type == "XGBClassifier" and (
            len(self.classes_) > 2 or self.classes_[1] != 1 or self.classes_[0] != 0
        ):
            trees = []
            for i in range(self.n_estimators_):
                for c in self.classes_:
                    trees += [self._to_json_tree_dict(i, str(c))]
            tree_info = [i for i in range(len(self.classes_))] * self.n_estimators_
            for idx, tree in enumerate(trees):
                tree["id"] = idx
        else:
            trees = [self._to_json_tree_dict(i) for i in range(self.n_estimators_)]
            tree_info = [0 for i in range(self.n_estimators_)]
        return {
            "model": {
                "trees": trees,
                "tree_info": tree_info,
                "gbtree_model_param": {
                    "num_trees": str(len(trees)),
                    "size_leaf_vector": "0",
                },
            },
            "name": "gbtree",
        }

    def _to_json_learner(self) -> dict:
        """
        Method used to convert the model to JSON.
        """
        if self._model_type == "XGBRegressor" or (
            len(self.classes_) == 2 and self.classes_[1] == 1 and self.classes_[0] == 0
        ):
            bs, num_class, param, param_val = (
                self.mean_,
                "0",
                "reg_loss_param",
                {"scale_pos_weight": "1"},
            )
            if self._model_type == "XGBRegressor":
                objective = "reg:squarederror"
                attributes_dict = {
                    "scikit_learn": '{"n_estimators": '
                    + str(self.n_estimators_)
                    + ', "objective": "reg:squarederror", "max_depth": '
                    + str(self.parameters["max_depth"])
                    + ', "learning_rate": '
                    + str(self.parameters["learning_rate"])
                    + ', "verbosity": null, "booster": null, "tree_method": null,'
                    + ' "gamma": null, "min_child_weight": null, "max_delta_step":'
                    + ' null, "subsample": null, "colsample_bytree": '
                    + str(self.parameters["col_sample_by_tree"])
                    + ', "colsample_bylevel": null, "colsample_bynode": '
                    + str(self.parameters["col_sample_by_node"])
                    + ', "reg_alpha": null, "reg_lambda": null, "scale_pos_weight":'
                    + ' null, "base_score": null, "missing": NaN, "num_parallel_tree"'
                    + ': null, "kwargs": {}, "random_state": null, "n_jobs": null, '
                    + '"monotone_constraints": null, "interaction_constraints": null,'
                    + ' "importance_type": "gain", "gpu_id": null, "validate_parameters"'
                    + ': null, "_estimator_type": "regressor"}'
                }
            else:
                objective = "binary:logistic"
                attributes_dict = {
                    "scikit_learn": '{"use_label_encoder": true, "n_estimators": '
                    + str(self.n_estimators_)
                    + ', "objective": "binary:logistic", "max_depth": '
                    + str(self.parameters["max_depth"])
                    + ', "learning_rate": '
                    + str(self.parameters["learning_rate"])
                    + ', "verbosity": null, "booster": null, "tree_method": null,'
                    + ' "gamma": null, "min_child_weight": null, "max_delta_step":'
                    + ' null, "subsample": null, "colsample_bytree": '
                    + str(self.parameters["col_sample_by_tree"])
                    + ', "colsample_bylevel": null, "colsample_bynode": '
                    + str(self.parameters["col_sample_by_node"])
                    + ', "reg_alpha": null, "reg_lambda": null, "scale_pos_weight":'
                    + ' null, "base_score": null, "missing": NaN, "num_parallel_tree"'
                    + ': null, "kwargs": {}, "random_state": null, "n_jobs": null,'
                    + ' "monotone_constraints": null, "interaction_constraints": null,'
                    + ' "importance_type": "gain", "gpu_id": null, "validate_parameters"'
                    + ': null, "classes_": [0, 1], "n_classes_": 2, "_le": {"classes_": '
                    + '[0, 1]}, "_estimator_type": "classifier"}'
                }
        else:
            objective, bs, num_class, param, param_val = (
                "multi:softprob",
                0.5,
                str(len(self.classes_)),
                "softmax_multiclass_param",
                {"num_class": str(len(self.classes_))},
            )
            attributes_dict = {
                "scikit_learn": '{"use_label_encoder": true, "n_estimators": '
                + str(self.n_estimators_)
                + ', "objective": "multi:softprob", "max_depth": '
                + str(self.parameters["max_depth"])
                + ', "learning_rate": '
                + str(self.parameters["learning_rate"])
                + ', "verbosity": null, "booster": null, "tree_method": null, '
                + '"gamma": null, "min_child_weight": null, "max_delta_step": '
                + 'null, "subsample": null, "colsample_bytree": '
                + str(self.parameters["col_sample_by_tree"])
                + ', "colsample_bylevel": null, "colsample_bynode": '
                + str(self.parameters["col_sample_by_node"])
                + ', "reg_alpha": null, "reg_lambda": null, "scale_pos_weight":'
                + ' null, "base_score": null, "missing": NaN, "num_parallel_tree":'
                + ' null, "kwargs": {}, "random_state": null, "n_jobs": null, '
                + '"monotone_constraints": null, "interaction_constraints": null, '
                + '"importance_type": "gain", "gpu_id": null, "validate_parameters":'
                + ' null, "classes_": '
                + str(list(self.classes_))
                + ', "n_classes_": '
                + str(len(self.classes_))
                + ', "_le": {"classes_": '
                + str(list(self.classes_))
                + '}, "_estimator_type": "classifier"}'
            }
        attributes_dict["scikit_learn"] = attributes_dict["scikit_learn"].replace(
            '"', "++++"
        )
        gradient_booster = self._to_json_tree_dict_list()
        return {
            "attributes": attributes_dict,
            "feature_names": [],
            "feature_types": [],
            "gradient_booster": gradient_booster,
            "learner_model_param": {
                "base_score": np.format_float_scientific(bs, precision=7).upper(),
                "num_class": num_class,
                "num_feature": str(len(self.X)),
            },
            "objective": {"name": objective, param: param_val},
        }

    def to_json(self, path: Optional[str] = None) -> Optional[str]:
        """
        Creates  a  Python  XGBoost  JSON  file  that  can
        be imported into the Python XGBoost API.

        \u26A0 Warning :   For    multiclass   classifiers,
        the   probabilities  returned   by  the   VerticaPy
        and  exported  models might differ slightly because
        of  normalization;  while Vertica uses  multinomial
        logistic  regression, XGBoost Python uses  Softmax.
        This  difference does not affect the model's  final
        predictions. Categorical predictors must be encoded.

        Parameters
        ----------
        path: str, optional
            The path and name of the output file. If a file
            with the same name already exists, the function
            returns an error.

        Returns
        -------
        None / str
            The content of the JSON file if variable 'path'
            is empty. Otherwise, nothing is returned.
        """
        res = {"learner": self._to_json_learner(), "version": [1, 6, 2]}
        res = (
            str(res)
            .replace("'", '"')
            .replace("True", "true")
            .replace("False", "false")
            .replace("++++", '\\"')
        )
        if path:
            with open(path, "w+", encoding="utf-8") as f:
                f.write(res)

        else:
            return res


"""
Algorithms used for regression.
"""


class RandomForestRegressor(Regressor, RandomForest):
    """
    Creates a RandomForestRegressor object using the
    Vertica RF_REGRESSOR function. It is an ensemble
    learning method for regression  that operates by
    constructing a multitude of decision trees  at
    training-time and outputting a class with the
    mode.

    Parameters
    ----------
    name: str, optional
        Name of the model. The model is stored in the
        database.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    n_estimators: int, optional
        The number of trees  in the forest, an integer
        between 1 and 1000, inclusive.
    max_features: int / str, optional
        The  number  of randomly chosen features  from
        which  to pick the best feature to split a
        given  tree node. It can be an integer or  one
        of the two following methods.

        - auto:
            square root  of the total number of predictors.
        - max :
            number of predictors.
    max_leaf_nodes: PythonNumber, optional
        The maximum number of leaf nodes for a tree in
        the forest, an integer between 1 and 1e9,
        inclusive.
    sample: float, optional
        The  portion  of  the  input  data set that  is
        randomly selected for training each tree, a
        float between 0.0 and 1.0, inclusive.
    max_depth: int, optional
        The  maximum  depth  for growing each tree,  an
        integer between 1 and 100, inclusive.
    min_samples_leaf: int, optional
        The minimum number of  samples each branch must
        have after splitting a node, an integer between
        1 and 1e6, inclusive. A split that results in
        remaining samples less than this value
        is discarded.
    min_info_gain: PythonNumber, optional
        The  minimum threshold for including a split, a
        float  between 0.0 and 1.0, inclusive. A  split
        with information gain  less than this threshold
        is discarded.
    nbins: int, optional
        The  number  of  bins  to  use  for  continuous
        features,   an  integer  between  2  and  1000,
        inclusive.

    Attributes
    ----------
    Many attributes are created during the fitting phase.

    trees_: list of BinaryTreeRegressor
        Tree models are instances of ``BinaryTreeRegressor``,
        each possessing various attributes. For more
        detailed information, refer to the documentation
        for
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
    features_importance_trees_: dict of numpy.array
        Each element of the array represents the feature
        importance of tree i.
        The importance of features is calculated
        using the MDI (Mean Decreased Impurity).
        It is necessary to use the
        :py:meth:`verticapy.machine_learning.vertica.base.Tree.features_importance`
        method to compute it initially, and the computed
        values will be subsequently utilized for subsequent
        calls.
    n_estimators_: int
        The number of model estimators.

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

    First we import the ``RandomForestRegressor`` model:

    .. ipython:: python

        from verticapy.machine_learning.vertica import RandomForestRegressor

    Then we can create the model:

    .. ipython:: python
        :okwarning:

        model = RandomForestRegressor(
            max_features = "auto",
            max_leaf_nodes = 32,
            sample = 0.5,
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
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_rfreg_feature.html")

    .. code-block:: python

        result = model.features_importance()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_rfreg_feature.html

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
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_rfreg_report.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_rfreg_report.html

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
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_rfreg_prediction.html", "w")
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
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_rfreg_prediction.html

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
        res.render(filename='figures/machine_learning_vertica_rfreg', format='png')

    .. image:: /../figures/machine_learning_vertica_rfreg.png

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
        :py:meth:`verticapy.machine_learning.vertica.tree.RandomForestRegressor.to_python`
        method is used to retrieve predictions,
        probabilities, or cluster distances. For specific details on how to
        use this method for different model types, refer to the relevant
        documentation for each model.
    """

    # Properties.

    @property
    def _vertica_fit_sql(self) -> Literal["RF_REGRESSOR"]:
        return "RF_REGRESSOR"

    @property
    def _vertica_predict_sql(self) -> Literal["PREDICT_RF_REGRESSOR"]:
        return "PREDICT_RF_REGRESSOR"

    @property
    def _model_subcategory(self) -> Literal["REGRESSOR"]:
        return "REGRESSOR"

    @property
    def _model_type(self) -> Literal["RandomForestRegressor"]:
        return "RandomForestRegressor"

    @property
    def _attributes(self) -> list[str]:
        return [
            "n_estimators_",
            "trees_",
            "features_importance_",
            "features_importance_trees_",
        ]

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str = None,
        overwrite_model: bool = False,
        n_estimators: int = 10,
        max_features: Union[Literal["auto", "max"], int] = "auto",
        max_leaf_nodes: PythonNumber = 1e9,
        sample: float = 0.632,
        max_depth: int = 5,
        min_samples_leaf: int = 1,
        min_info_gain: PythonNumber = 0.0,
        nbins: int = 32,
    ) -> None:
        super().__init__(name, overwrite_model)
        self.parameters = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "max_leaf_nodes": int(max_leaf_nodes),
            "sample": sample,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_info_gain": min_info_gain,
            "nbins": nbins,
        }

    # Attributes Methods.

    def _compute_attributes(self) -> None:
        """
        Computes the model's attributes.
        """
        self.n_estimators_ = self.parameters["n_estimators"]
        trees = []
        for i in range(self.n_estimators_):
            tree = self._compute_trees_arrays(self.get_tree(i), self.X)
            tree_d = {
                "children_left": tree[0],
                "children_right": tree[1],
                "feature": tree[2],
                "threshold": tree[3],
                "value": tree[4],
            }
            for j in range(len(tree[5])):
                if not tree[5][j] and isinstance(tree_d["threshold"][j], str):
                    tree_d["threshold"][j] = float(tree_d["threshold"][j])
            tree_d["value"] = [
                float(val) if isinstance(val, str) else val for val in tree_d["value"]
            ]
            model = mm.BinaryTreeRegressor(**tree_d)
            trees += [model]
        self.trees_ = trees

    # I/O Methods.

    def to_memmodel(self) -> Union[mm.RandomForestRegressor, mm.BinaryTreeRegressor]:
        """
        Converts  the model  to an InMemory object  that
        can be used for different types of predictions.
        """
        if self.n_estimators_ == 1:
            return self.trees_[0]
        else:
            return mm.RandomForestRegressor(self.trees_)


class XGBRegressor(Regressor, XGBoost):
    """
    Creates  an  XGBRegressor  object  using the  Vertica
    XGB_REGRESSOR algorithm.

    Parameters
    ----------
    name: str, optional
        Name  of the  model.  The  model  is  stored
        in the DB.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    max_ntree: int, optional
        Maximum  number  of trees that  can be  created.
    max_depth: int, optional
        Maximum depth  of each tree, an  integer between 1
        and 20, inclusive.
    nbins: int, optional
        Number of bins used to find splits in each column,
        where more splits  leads to a longer runtime but
        more  fine-grained, possibly  better splits. Must
        be an integer between 2 and 1000, inclusive.
    split_proposal_method: str, optional
        Approximate  splitting  strategy, either 'global'
        or 'local' (not yet supported).
    tol: float, optional
        Approximation error of quantile summary structures
        used in the approximate split finding method.
    learning_rate: float, optional
        Weight applied to each tree's prediction. This
        reduces each  tree's impact, allowing for  later
        trees  to contribute and keeping earlier trees from
        dominating.
    min_split_loss: float, optional
        Each  split  must improve the model's objective
        function value by  at least this much in order
        to avoid pruning.  A value  of  0 is the same  as
        turning off this parameter (trees are still pruned
        based  on  positive / negative  objective function
        values).
    weight_reg: float, optional
        Regularization term that is applied to the weights
        of  the leaves in the regression tree. A higher
        value leads to more sparse/smooth weights, which
        often helps to prevent overfitting.
    sample: float, optional
        Fraction of rows used per iteration in training.
    col_sample_by_tree: float, optional
        Float  in  the  range  (0,1]  that  specifies  the
        fraction of columns (features), chosen at  random,
        to use when building each tree.
    col_sample_by_node: float, optional
        Float  in  the  range  (0,1]  that  specifies  the
        fraction of columns (features), chosen at  random,
        to use when evaluating each split.

    Attributes
    ----------
    Many attributes are created during the fitting phase.

    trees_: list of BinaryTreeRegressor
        Tree models are instances of ``BinaryTreeRegressor``,
        each possessing various attributes. For more
        detailed information, refer to the documentation
        for
        :py:meth:`verticapy.machine_learning.memmodel.tree.BinaryTreeRegressor`.
    features_importance_: numpy.array
        The importance of features. It is calculated
        using the average gain of each tree. To determine
        the final score, VerticaPy sums the scores of each
        tree, normalizes them and applies an activation
        function to scale them.
        It is necessary to use the
        :py:meth:`verticapy.machine_learning.vertica.base.Tree.features_importance`
        method to compute it initially, and the computed
        values will be subsequently utilized for subsequent
        calls.
    features_importance_trees_: dict of numpy.array
        Each element of the array represents the feature
        importance of tree i.
        The importance of features is calculated
        using the average gain of each tree.
        It is necessary to use the
        :py:meth:`verticapy.machine_learning.vertica.base.Tree.features_importance`
        method to compute it initially, and the computed
        values will be subsequently utilized for subsequent
        calls.
    mean_: float
        The mean of the response column.
    eta_: float
        The learning rate, is a crucial hyperparameter in
        machine learning algorithms. It determines the step
        size at each iteration during the model training
        process. A well-chosen learning rate is essential
        for achieving optimal convergence and preventing
        overshooting or slow convergence in the training
        phase. Adjusting the learning rate is often necessary
        to strike a balance between model accuracy and
        computational efficiency.
    n_estimators_: int
        The number of model estimators.

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

        Many tree-based models inherit from the ``XGB``
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

    First we import the ``XGBRegressor`` model:

    .. ipython:: python

        from verticapy.machine_learning.vertica import XGBRegressor

    Then we can create the model:

    .. ipython:: python
        :okwarning:

        model = XGBRegressor(
            max_ntree = 3,
            max_depth = 3,
            nbins = 6,
            split_proposal_method = 'global',
            tol = 0.001,
            learning_rate = 0.1,
            min_split_loss = 0,
            weight_reg = 0,
            sample = 0.7,
            col_sample_by_tree = 1,
            col_sample_by_node = 1,
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
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_xgbreg_feature.html")

    .. code-block:: python

        result = model.features_importance()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_xgbreg_feature.html

    .. note::

        In models such as ``XGBoost``, feature importance is calculated
        using the average gain of each tree. To determine the final score,
        VerticaPy sums the scores of each tree, normalizes them and applies an
        activation function to scale them.

    Metrics
    ^^^^^^^^

    We can get the entire report using:

    .. ipython:: python
        :suppress:

        result = model.report()
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_xgbreg_report.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_xgbreg_report.html

    .. important::

        Most metrics are computed using a single SQL query, but some of them might
        require multiple SQL queries. Selecting only the necessary metrics in the
        report can help optimize performance.
        E.g. ``model.report(metrics = ["mse", "r2"])``.

    You can utilize the
    :py:meth:`verticapy.machine_learning.vertica.ensemble.XGBRegressor.score`
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
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_xgbreg_prediction.html", "w")
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
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_xgbreg_prediction.html

    .. note::

        Predictions can be made automatically using the test set, in which
        case you don't need to specify the predictors. Alternatively, you
        can pass only the :py:class:`vDataFrame` to the
        :py:meth:`verticapy.machine_learning.vertica.ensemble.XGBRegressor.predict`
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
        res.render(filename='figures/machine_learning_vertica_xgbreg', format='png')

    .. image:: /../figures/machine_learning_vertica_xgbreg.png

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

    The preceding methods for exporting the model use ``MemModel``, and it
    is recommended to use ``MemModel`` directly.

    **To SQL**

    You can get the SQL query equivalent of the XGB model by:

    .. ipython:: python

        model.to_sql()

    .. note:: This SQL query can be directly used in any database.

    **Deploy SQL**

    To get the SQL query which uses Vertica functions use below:

    .. ipython:: python

        model.deploySQL()

    **To Python**

    To obtain the prediction function in Python syntax, use the following code:

    .. ipython:: python

        X = [[4.2, 0.17, 0.36, 1.8, 0.029, 0.9899]]
        model.to_python()(X)

    .. hint::

        The
        :py:meth:`verticapy.machine_learning.vertica.tree.XGBRegressor.to_python`
        method is used to retrieve predictions,
        probabilities, or cluster distances. For specific details on how to
        use this method for different model types, refer to the relevant
        documentation for each model.
    """

    # Properties.

    @property
    def _vertica_fit_sql(self) -> Literal["XGB_REGRESSOR"]:
        return "XGB_REGRESSOR"

    @property
    def _vertica_predict_sql(self) -> Literal["PREDICT_XGB_REGRESSOR"]:
        return "PREDICT_XGB_REGRESSOR"

    @property
    def _model_subcategory(self) -> Literal["REGRESSOR"]:
        return "REGRESSOR"

    @property
    def _model_type(self) -> Literal["XGBRegressor"]:
        return "XGBRegressor"

    @property
    def _attributes(self) -> list[str]:
        return [
            "n_estimators_",
            "eta_",
            "mean_",
            "trees_",
            "features_importance_",
            "features_importance_trees_",
        ]

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str = None,
        overwrite_model: bool = False,
        max_ntree: int = 10,
        max_depth: int = 5,
        nbins: int = 32,
        split_proposal_method: Literal["local", "global"] = "global",
        tol: float = 0.001,
        learning_rate: float = 0.1,
        min_split_loss: float = 0.0,
        weight_reg: float = 0.0,
        sample: float = 1.0,
        col_sample_by_tree: float = 1.0,
        col_sample_by_node: float = 1.0,
    ) -> None:
        super().__init__(name, overwrite_model)
        params = {
            "max_ntree": max_ntree,
            "max_depth": max_depth,
            "nbins": nbins,
            "split_proposal_method": str(split_proposal_method).lower(),
            "tol": tol,
            "learning_rate": learning_rate,
            "min_split_loss": min_split_loss,
            "weight_reg": weight_reg,
            "sample": sample,
            "col_sample_by_tree": col_sample_by_tree,
            "col_sample_by_node": col_sample_by_node,
        }
        self.parameters = params

    # Attributes Methods.

    def _compute_attributes(self) -> None:
        """
        Computes the model's attributes.
        """
        self.eta_ = self.parameters["learning_rate"]
        self.n_estimators_ = self.get_vertica_attributes("tree_count")["tree_count"][0]
        try:
            self.mean_ = float(
                self.get_vertica_attributes("initial_prediction")["initial_prediction"][
                    0
                ]
            )
        except:
            self.mean_ = self._compute_prior()
        trees = []
        for i in range(self.n_estimators_):
            tree = self._compute_trees_arrays(self.get_tree(i), self.X)
            tree_d = {
                "children_left": tree[0],
                "children_right": tree[1],
                "feature": tree[2],
                "threshold": tree[3],
                "value": tree[4],
            }
            for j in range(len(tree[5])):
                if not tree[5][j] and isinstance(tree_d["threshold"][j], str):
                    tree_d["threshold"][j] = float(tree_d["threshold"][j])
            tree_d["value"] = [
                float(val) if isinstance(val, str) else val for val in tree_d["value"]
            ]
            model = mm.BinaryTreeRegressor(**tree_d)
            trees += [model]
        self.trees_ = trees

    # I/O Methods.

    def to_memmodel(self) -> mm.XGBRegressor:
        """
        Converts  the  model to an InMemory object  that
        can be used for different types of predictions.
        """
        return mm.XGBRegressor(self.trees_, self.mean_, self.eta_)


"""
Algorithms used for classification.
"""


class RandomForestClassifier(MulticlassClassifier, RandomForest):
    """
    Creates a RandomForestClassifier object using the
    Vertica  RF_CLASSIFIER function. It is an ensemble
    learning method for classification that operates
    by constructing a multitude of decision trees  at
    training-time and outputting a class with the mode.

    Parameters
    ----------
    name: str, optional
        Name of the model. The model is stored in the
        database.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    n_estimators: int, optional
        The number of trees  in the forest, an integer
        between 1 and 1000, inclusive.
    max_features: int / str, optional
        The  number  of randomly chosen features  from
        which  to pick the best feature to split  a
        given  tree node. It can be an integer or  one
        of the two following methods.

        - auto:
            square root  of the total number of predictors.
        - max :
            number of predictors.
    max_leaf_nodes: PythonNumber, optional
        The maximum number of leaf nodes for a tree in
        the forest, an integer between 1 and 1e9,
        inclusive.
    sample: float, optional
        The  portion  of  the  input  data set that  is
        randomly selected for training each tree, a
        float between 0.0 and 1.0, inclusive.
    max_depth: int, optional
        The  maximum  depth  for growing each tree,  an
        integer between 1 and 100, inclusive.
    min_samples_leaf: int, optional
        The minimum number of  samples each branch must
        have after splitting a node, an integer between
        1 and 1e6, inclusive. A split that results in
        remaining samples less than this value
        is discarded.
    min_info_gain: PythonNumber, optional
        The  minimum threshold for including a split, a
        float  between 0.0 and 1.0, inclusive. A  split
        with information gain  less than this threshold
        is discarded.
    nbins: int, optional
        The  number  of  bins  to  use  for  continuous
        features,   an  integer  between  2  and  1000,
        inclusive.

    Attributes
    ----------
    Many attributes are created during the fitting phase.

    trees_: list of BinaryTreeClassifier
        Tree models are instances of ``BinaryTreeClassifier``,
        each possessing various attributes. For more
        detailed information, refer to the documentation
        for
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
    features_importance_trees_: dict of numpy.array
        Each element of the array represents the feature
        importance of tree i.
        The importance of features is calculated
        using the MDI (Mean Decreased Impurity).
        It is necessary to use the
        :py:meth:`verticapy.machine_learning.vertica.base.Tree.features_importance`
        method to compute it initially, and the computed
        values will be subsequently utilized for subsequent
        calls.
    n_estimators_: int
        The number of model estimators.
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

    First we import the ``RandomForestClassifier`` model:

    .. code-block::

        from verticapy.machine_learning.vertica import RandomForestClassifier

    .. ipython:: python
        :suppress:

        from verticapy.machine_learning.vertica import RandomForestClassifier

    Then we can create the model:

    .. ipython:: python
        :okwarning:

        model = RandomForestClassifier(
            max_features = "auto",
            max_leaf_nodes = 32,
            sample = 0.5,
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
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_rf_classifier_feature.html")

    .. code-block:: python

        result = model.features_importance()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_rf_classifier_feature.html

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
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_rf_classifier_report.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_rf_classifier_report.html

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
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_rf_classifier_report_cutoff.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report(cutoff = 0.2)

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_rf_classifier_report_cutoff.html

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
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_rf_classifier_prediction.html", "w")
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
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_rf_classifier_prediction.html

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
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_rf_classifier_proba.html", "w")
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
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_rf_classifier_proba.html

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
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_rf_classifier_roc.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_rf_classifier_roc.html

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
        res.render(filename='figures/machine_learning_vertica_tree_rf_classifier_', format='png')

    .. image:: /../figures/machine_learning_vertica_tree_rf_classifier_.png

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
        :py:meth:`verticapy.machine_learning.vertica.tree.RandomForestClassifier.to_python`
        method is used to retrieve predictions,
        probabilities, or cluster distances. For specific details on how to
        use this method for different model types, refer to the relevant
        documentation for each model.
    """

    # Properties.

    @property
    def _vertica_fit_sql(self) -> Literal["RF_CLASSIFIER"]:
        return "RF_CLASSIFIER"

    @property
    def _vertica_predict_sql(self) -> Literal["PREDICT_RF_CLASSIFIER"]:
        return "PREDICT_RF_CLASSIFIER"

    @property
    def _model_subcategory(self) -> Literal["CLASSIFIER"]:
        return "CLASSIFIER"

    @property
    def _model_type(self) -> Literal["RandomForestClassifier"]:
        return "RandomForestClassifier"

    @property
    def _attributes(self) -> list[str]:
        return [
            "n_estimators_",
            "classes_",
            "trees_",
            "features_importance_",
            "features_importance_trees_",
        ]

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str = None,
        overwrite_model: bool = False,
        n_estimators: int = 10,
        max_features: Union[Literal["auto", "max"], int] = "auto",
        max_leaf_nodes: PythonNumber = 1e9,
        sample: float = 0.632,
        max_depth: int = 5,
        min_samples_leaf: int = 1,
        min_info_gain: PythonNumber = 0.0,
        nbins: int = 32,
    ) -> None:
        super().__init__(name, overwrite_model)
        self.parameters = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "max_leaf_nodes": int(max_leaf_nodes),
            "sample": sample,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_info_gain": min_info_gain,
            "nbins": nbins,
        }

    # Attributes Methods.

    def _compute_attributes(self) -> None:
        """
        Computes the model's attributes.
        """
        self.n_estimators_ = self.parameters["n_estimators"]
        try:
            self.classes_ = self._array_to_int(self._get_classes())
        except MissingRelation:
            self.classes_ = np.array([])

        trees = []
        for i in range(self.n_estimators_):
            tree = self._compute_trees_arrays(self.get_tree(i), self.X, True)
            tree_d = {
                "children_left": tree[0],
                "children_right": tree[1],
                "feature": tree[2],
                "threshold": tree[3],
                "value": tree[4],
                "classes": self.classes_,
            }
            n_classes = len(self.classes_)
            for j in range(len(tree[5])):
                if not tree[5][j] and isinstance(tree_d["threshold"][j], str):
                    tree_d["threshold"][j] = float(tree_d["threshold"][j])
            for j in range(len(tree_d["value"])):
                if not isinstance(tree_d["value"][j], NoneType):
                    prob = [0.0 for i in range(n_classes)]
                    for k, c in enumerate(self.classes_):
                        if str(c) == str(tree_d["value"][j]):
                            prob[k] = tree[6][j]
                            break
                    other_proba = (1 - tree[6][j]) / (n_classes - 1)
                    for k, p in enumerate(prob):
                        if p == 0.0:
                            prob[k] = other_proba
                    tree_d["value"][j] = prob
            model = mm.BinaryTreeClassifier(**tree_d)
            trees += [model]
        self.trees_ = trees

    # I/O Methods.

    def to_memmodel(self) -> Union[mm.RandomForestClassifier, mm.BinaryTreeClassifier]:
        """
        Converts the model to an InMemory object that
        can be used for different types of predictions.
        """
        if self.n_estimators_ == 1:
            return self.trees_[0]
        else:
            return mm.RandomForestClassifier(self.trees_, self.classes_)


class XGBClassifier(MulticlassClassifier, XGBoost):
    """
    Creates  an  XGBClassifier  object using the  Vertica
    XGB_CLASSIFIER algorithm.

    Parameters
    ----------
    name: str, optional
        Name  of the  model. The model  is  stored in the
        database.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    max_ntree: int, optional
        Maximum  number  of trees that can be  created.
    max_depth: int, optional
        Maximum depth  of each tree, an  integer between 1
        and 20, inclusive.
    nbins: int, optional
        Number of bins used to find splits in each column,
        where more splits  leads to a longer runtime but
        more  fine-grained, possibly  better splits. Must
        be an integer between 2 and 1000, inclusive.
    split_proposal_method: str, optional
        Approximate  splitting  strategy, either 'global'
        or 'local' (not yet supported)
    tol: float, optional
        Approximation error of quantile summary structures
        used in the approximate split finding method.
    learning_rate: float, optional
        Weight applied to each tree's prediction. This
        reduces each  tree's impact, allowing for  later
        trees  to contribute and keeping earlier trees from
        dominating.
    min_split_loss: float, optional
        Each  split  must improve the model's objective
        function value by  at least this much in order
        to avoid pruning.  A value  of  0 is the same  as
        turning off this parameter (trees are still pruned
        based  on  positive / negative  objective function
        values).
    weight_reg: float, optional
        Regularization term that is applied to the weights
        of  the leaves in the regression tree. A higher
        value leads to more sparse/smooth weights, which
        often helps to prevent overfitting.
    sample: float, optional
        Fraction of rows used per iteration in training.
    col_sample_by_tree: float, optional
        Float  in  the  range  (0,1]  that  specifies  the
        fraction of columns (features), chosen at  random,
        to use when building each tree.
    col_sample_by_node: float, optional
        Float  in  the  range  (0,1]  that  specifies  the
        fraction of columns (features), chosen at  random,
        to use when evaluating each split.

    Attributes
    ----------
    Many attributes are created during the fitting phase.

    trees_: list of BinaryTreeClassifier
        Tree models are instances of ``BinaryTreeClassifier``,
        each possessing various attributes. For more
        detailed information, refer to the documentation
        for
        :py:meth:`verticapy.machine_learning.memmodel.tree.BinaryTreeClassifier`.
    features_importance_: numpy.array
        The importance of features. It is calculated
        using the average gain of each tree. To determine
        the final score, VerticaPy sums the scores of each
        tree, normalizes them and applies an activation
        function to scale them.
        It is necessary to use the
        :py:meth:`verticapy.machine_learning.vertica.base.Tree.features_importance`
        method to compute it initially, and the computed
        values will be subsequently utilized for subsequent
        calls.
    features_importance_trees_: dict of numpy.array
        Each element of the array represents the feature
        importance of tree i.
        The importance of features is calculated
        using the average gain of each tree.
        It is necessary to use the
        :py:meth:`verticapy.machine_learning.vertica.base.Tree.features_importance`
        method to compute it initially, and the computed
        values will be subsequently utilized for subsequent
        calls.
    logodds_: numpy.array
        The log-odds. It quantifies the logarithm of the
        odds ratio, providing a measure of the likelihood
        of an event occurring.
    eta_: float
        The learning rate, is a crucial hyperparameter in
        machine learning algorithms. It determines the step
        size at each iteration during the model training
        process. A well-chosen learning rate is essential
        for achieving optimal convergence and preventing
        overshooting or slow convergence in the training
        phase. Adjusting the learning rate is often necessary
        to strike a balance between model accuracy and
        computational efficiency.
    n_estimators_: int
        The number of model estimators.
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

        Many tree-based models inherit from the ``XGB``
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

    First we import the ``XGBClassifier`` model:

    .. code-block::

        from verticapy.machine_learning.vertica import XGBClassifier

    .. ipython:: python
        :suppress:

        from verticapy.machine_learning.vertica import XGBClassifier

    Then we can create the model:

    .. ipython:: python
        :okwarning:

        model = XGBClassifier(
            max_ntree = 3,
            max_depth = 3,
            nbins = 6,
            split_proposal_method = 'global',
            tol = 0.001,
            learning_rate = 0.1,
            min_split_loss = 0,
            weight_reg = 0,
            sample = 0.7,
            col_sample_by_tree = 1,
            col_sample_by_node = 1,
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
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_xgb_classifier_feature.html")

    .. code-block:: python

        result = model.features_importance()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_xgb_classifier_feature.html

    .. note::

        In models such as ``XGBoost``, feature importance is calculated
        using the average gain of each tree. To determine the final score,
        VerticaPy sums the scores of each tree, normalizes them and applies an
        activation function to scale them.

    Metrics
    ^^^^^^^^

    We can get the entire report using:

    .. ipython:: python
        :suppress:

        result = model.report()
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_xgb_classifier_report.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_xgb_classifier_report.html

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
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_xgb_classifier_report_cutoff.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report(cutoff = 0.2)

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_xgb_classifier_report_cutoff.html

    You can also use the
    :py:meth:`verticapy.machine_learning.vertica.ensemble.XGBClassifier.score`
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
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_xgb_classifier_prediction.html", "w")
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
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_xgb_classifier_prediction.html

    .. note::

        Predictions can be made automatically using the test set, in which
        case you don't need to specify the predictors. Alternatively, you
        can pass only the :py:class:`vDataFrame` to the
        :py:meth:`verticapy.machine_learning.vertica.ensemble.XGBClassifier.predict`
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
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_xgb_classifier_proba.html", "w")
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
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_xgb_classifier_proba.html

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
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_xgb_classifier_roc.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_xgb_classifier_roc.html

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
        res.render(filename='figures/machine_learning_vertica_tree_xgb_classifier_', format='png')

    .. image:: /../figures/machine_learning_vertica_tree_xgb_classifier_.png

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

    The preceding methods for exporting the model use ``MemModel``, and it
    is recommended to use ``MemModel`` directly.

    **To SQL**

    You can get the SQL query equivalent of the XGB model by:

    .. ipython:: python

        model.to_sql()

    .. note:: This SQL query can be directly used in any database.

    **Deploy SQL**

    To get the SQL query which uses Vertica functions use below:

    .. ipython:: python

        model.deploySQL()

    **To Python**

    To obtain the prediction function in Python syntax, use the following code:

    .. ipython:: python

        X = [[4.2, 0.17, 0.36, 1.8, 0.029, 0.9899]]
        model.to_python()(X)

    .. hint::

        The
        :py:meth:`verticapy.machine_learning.vertica.tree.XGBClassifier.to_python`
        method is used to retrieve predictions,
        probabilities, or cluster distances. For specific details on how to
        use this method for different model types, refer to the relevant
        documentation for each model.
    """

    # Properties.

    @property
    def _vertica_fit_sql(self) -> Literal["XGB_CLASSIFIER"]:
        return "XGB_CLASSIFIER"

    @property
    def _vertica_predict_sql(self) -> Literal["PREDICT_XGB_CLASSIFIER"]:
        return "PREDICT_XGB_CLASSIFIER"

    @property
    def _model_subcategory(self) -> Literal["CLASSIFIER"]:
        return "CLASSIFIER"

    @property
    def _model_type(self) -> Literal["XGBClassifier"]:
        return "XGBClassifier"

    @property
    def _attributes(self) -> list[str]:
        return [
            "n_estimators_",
            "classes_",
            "eta_",
            "logodds_",
            "trees_",
            "features_importance_",
            "features_importance_trees_",
        ]

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str = None,
        overwrite_model: bool = False,
        max_ntree: int = 10,
        max_depth: int = 5,
        nbins: int = 32,
        split_proposal_method: Literal["local", "global"] = "global",
        tol: float = 0.001,
        learning_rate: float = 0.1,
        min_split_loss: float = 0.0,
        weight_reg: float = 0.0,
        sample: float = 1.0,
        col_sample_by_tree: float = 1.0,
        col_sample_by_node: float = 1.0,
    ) -> None:
        super().__init__(name, overwrite_model)
        params = {
            "max_ntree": max_ntree,
            "max_depth": max_depth,
            "nbins": nbins,
            "split_proposal_method": str(split_proposal_method).lower(),
            "tol": tol,
            "learning_rate": learning_rate,
            "min_split_loss": min_split_loss,
            "weight_reg": weight_reg,
            "sample": sample,
            "col_sample_by_tree": col_sample_by_tree,
            "col_sample_by_node": col_sample_by_node,
        }
        self.parameters = params

    # Attributes Methods.

    def _compute_attributes(self) -> None:
        """
        Computes the model's attributes.
        """
        self.eta_ = self.parameters["learning_rate"]
        self.n_estimators_ = self.get_vertica_attributes("tree_count")["tree_count"][0]
        try:
            self.classes_ = self._array_to_int(
                np.array(
                    self.get_vertica_attributes("initial_prediction")["response_label"]
                )
            )
            # Handling NULL Values.
            null_ = False
            if self.classes_[0] == "":
                self.classes_ = self.classes_[1:]
                null_ = True
            if self._is_binary_classifier():
                prior = self._compute_prior()
            else:
                prior = np.array(
                    self.get_vertica_attributes("initial_prediction")["value"]
                )
                if null_:
                    prior = prior[1:]
        except QueryError:
            try:
                self.classes_ = self._array_to_int(self._get_classes())
            except MissingRelation:
                self.classes_ = np.array([])
            prior = self._compute_prior()
        if isinstance(prior, (int, float)):
            self.mean_ = prior
            self.logodds_ = [
                np.log((1 - prior) / prior),
                np.log(prior / (1 - prior)),
            ]
        else:
            self.logodds_ = prior
        trees = []
        for i in range(self.n_estimators_):
            tree = self._compute_trees_arrays(self.get_tree(i), self.X)
            tree_d = {
                "children_left": tree[0],
                "children_right": tree[1],
                "feature": tree[2],
                "threshold": tree[3],
                "value": tree[6],
                "classes": self.classes_,
            }
            for j in range(len(tree[5])):
                if not tree[5][j] and isinstance(tree_d["threshold"][j], str):
                    tree_d["threshold"][j] = float(tree_d["threshold"][j])
            for j in range(len(tree[6])):
                if not isinstance(tree[6][j], NoneType):
                    all_classes_logodss = []
                    for c in self.classes_:
                        all_classes_logodss += [tree[6][j][str(c)]]
                    tree_d["value"][j] = all_classes_logodss
            model = mm.BinaryTreeClassifier(**tree_d)
            trees += [model]
        self.trees_ = trees

    # I/O Methods.

    def to_memmodel(self) -> mm.XGBClassifier:
        """
        Converts the model  to an InMemory  object  that
        can be used for different types of predictions.
        """
        return mm.XGBClassifier(self.trees_, self.logodds_, self.classes_, self.eta_)


"""
Algorithms used for anomaly detection.
"""


class IsolationForest(Clustering, Tree):
    """
    Creates an IsolationForest object using the Vertica
    IFOREST algorithm.

    Parameters
    ----------
    name: str, optional
        Name  of  the model. The model  is stored in the
        database.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    n_estimators: int, optional
        The number  of  trees in the forest,  an integer
        between 1 and 1000, inclusive.
    max_depth: int, optional
        Maximum  depth of each tree,  an integer between
        1 and 100, inclusive.
    nbins: int, optional
        Number of bins used to find splits in each column,
        where more splits  leads to a longer runtime but
        more  fine-grained, possibly  better splits. Must
        be an integer between 2 and 1000, inclusive.
    sample: float, optional
        The  portion  of  the input  data  set  that  is
        randomly  selected  for  training  each tree,  a
        float between 0.0 and 1.0, inclusive.
    col_sample_by_tree: float, optional
        Float  in  the  range (0,1] that  specifies  the
        fraction of columns (features), chosen at random,
        used when building each tree.

    Attributes
    ----------
    Many attributes are created during the fitting phase.

    trees_: list of BinaryTreeAnomaly
        Tree models are instances of ``BinaryTreeAnomaly``,
        each possessing various attributes. For more
        detailed information, refer to the documentation
        for
        :py:meth:`verticapy.machine_learning.memmodel.tree.BinaryTreeAnomaly`.
    psy_: int
        Sampling size used to compute the final score.
    n_estimators_: int
        The number of model estimators.

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

    Load data for machine learning
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    We import :py:mod:`verticapy`:

    .. ipython:: python

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

    .. ipython:: python
        :suppress:

        import verticapy.datasets as vpd
        data = vpd.load_winequality()

    Model Initialization
    ^^^^^^^^^^^^^^^^^^^^^

    First we import the ``IsolationForest`` model:

    .. code-block::

        from verticapy.machine_learning.vertica import IsolationForest

    .. ipython:: python
        :suppress:

        from verticapy.machine_learning.vertica import IsolationForest

    Then we can create the model:

    .. ipython:: python
        :okwarning:

        model = IsolationForest(
            n_estimators = 10,
            max_depth = 3,
            nbins = 6,
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

        model.fit(data, X = ["density", "sulphates"])

    .. important::

        To train a model, you can directly use the :py:class:`vDataFrame` or the
        name of the relation stored in the database. The test set is optional
        and is only used to compute the test metrics. In :py:mod:`verticapy`, we
        don't work using ``X`` matrices and ``y`` vectors. Instead, we work
        directly with lists of predictors and the response name.

    .. hint::

        For clustering and anomaly detection, the use of predictors is
        optional. In such cases, all available predictors are considered,
        which can include solely numerical variables or a combination of
        numerical and categorical variables, depending on the model's
        capabilities.

    Prediction
    ^^^^^^^^^^^

    Prediction is straight-forward:

    .. ipython:: python
        :suppress:

        result = model.predict(data, ["density", "sulphates"])
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_isolation_for_prediction.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.predict(data, ["density", "sulphates"])

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_isolation_for_prediction.html

    Plots - Anomaly Detection
    ^^^^^^^^^^^^^^^^^^^^^^^^^^

    Plots highlighting the outliers can be easily drawn using:

    .. code-block:: python

        model.plot()

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")
        fig = model.plot()
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_isolation_for_plot.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_isolation_for_plot.html

    .. note::

        Most anomaly detection methods produce a score. In scenarios involving
        2 or 3 predictors, using a bubble plot to visualize the model's results
        is a straightforward approach. In such plots, the size of each bubble
        corresponds to the anomaly score.

    Plots - Tree
    ^^^^^^^^^^^^^

    Tree models can be visualized by drawing their tree plots.
    For more examples, check out :ref:`chart_gallery.tree`.

    .. code-block:: python

        model.plot_tree()

    .. ipython:: python
        :suppress:

        res = model.plot_tree()
        res.render(filename='figures/machine_learning_vertica_tree_isolation_for_', format='png')

    .. image:: /../figures/machine_learning_vertica_tree_isolation_for_.png

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

    Plots - Contour
    ^^^^^^^^^^^^^^^^

    In order to understand the parameter space, we can also look
    at the contour plots:

    .. code-block:: python

        model.contour()

    .. ipython:: python
        :suppress:

        fig = model.contour()
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_isolation_for_contour.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_isolation_for_contour.html

    .. note::

        Machine learning models with two predictors can usually benefit
        from their own contour plot. This visual representation aids in
        exploring predictions and gaining a deeper understanding of how
        these models perform in different scenarios. Please refer to
        :ref:`chart_gallery.contour_plot` for more examples.

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

    The preceding methods for exporting the model use ``MemModel``, and it
    is recommended to use ``MemModel`` directly.

    **To SQL**

    You can get the SQL query equivalent of the ``IsolationForest`` model by:

    .. ipython:: python

        model.to_sql()

    .. note:: This SQL query can be directly used in any database.

    **Deploy SQL**

    To get the SQL query which uses Vertica functions use below:

    .. ipython:: python

        model.deploySQL()

    **To Python**

    To obtain the prediction function in Python syntax, use the following code:

    .. ipython:: python

        X = [[0.9, 0.5]]
        model.to_python()(X)

    .. hint::

        The
        :py:meth:`verticapy.machine_learning.vertica.tree.IsolationForest.to_python`
        method is used to retrieve the anomaly score.
        For specific details on how to
        use this method for different model types, refer to the relevant
        documentation for each model.
    """

    # Properties.

    @property
    def _vertica_fit_sql(self) -> Literal["IFOREST"]:
        return "IFOREST"

    @property
    def _vertica_predict_sql(self) -> Literal["APPLY_IFOREST"]:
        return "APPLY_IFOREST"

    @property
    def _model_subcategory(self) -> Literal["ANOMALY_DETECTION"]:
        return "ANOMALY_DETECTION"

    @property
    def _model_type(self) -> Literal["IsolationForest"]:
        return "IsolationForest"

    @property
    def _attributes(self) -> list[str]:
        return ["n_estimators_", "psy_", "trees_"]

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str = None,
        overwrite_model: bool = False,
        n_estimators: int = 100,
        max_depth: int = 10,
        nbins: int = 32,
        sample: float = 0.632,
        col_sample_by_tree: float = 1.0,
    ) -> None:
        super().__init__(name, overwrite_model)
        self.parameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "nbins": nbins,
            "sample": sample,
            "col_sample_by_tree": col_sample_by_tree,
        }

    # Attributes Methods.

    def _compute_attributes(self) -> None:
        """
        Computes the model's attributes.
        """
        self.n_estimators_ = self.parameters["n_estimators"]
        self.psy_ = int(
            self.parameters["sample"]
            * int(
                self.get_vertica_attributes("accepted_row_count")["accepted_row_count"][
                    0
                ]
            )
        )
        trees = []
        for i in range(self.n_estimators_):
            tree = self._compute_trees_arrays(
                self.get_tree(i),
                self.X,
            )
            tree_d = {
                "children_left": tree[0],
                "children_right": tree[1],
                "feature": tree[2],
                "threshold": tree[3],
                "value": tree[4],
                "psy": self.psy_,
            }
            for idx in range(len(tree[5])):
                if not tree[5][idx] and isinstance(tree_d["threshold"][idx], str):
                    tree_d["threshold"][idx] = float(tree_d["threshold"][idx])
            model = mm.BinaryTreeAnomaly(**tree_d)
            trees += [model]
        self.trees_ = trees

    # I/O Methods.

    def deploySQL(
        self,
        X: Optional[SQLColumns] = None,
        cutoff: PythonNumber = 0.7,
        contamination: Optional[PythonNumber] = None,
        return_score: bool = False,
    ) -> str:
        """
        Returns  the SQL code needed to deploy the model.

        Parameters
        ----------
        X: SQLColumns, optional
            List of the columns used to  deploy the model.
            If empty, the model predictors are used.
        cutoff: PythonNumber, optional
            Float in the range  (0.0, 1.0),  specifies the
            threshold that determines  if a data  point is
            an anomaly.  If the  anomaly_score  for a data
            point is greater  than or equal to the cutoff,
            the data point is marked as an anomaly.
        contamination: PythonNumber, optional
            Float in the range (0,1), the approximate ratio
            of data points in the training data that should
            be labeled  as anomalous.  If this parameter is
            specified, the cutoff parameter is ignored.
        return_score: bool, optional
            If  set to True, the anomaly score is returned,
            and the parameters 'cutoff' and 'contamination'
            are ignored.

        Returns
        -------
        str
            the SQL code needed to deploy the model.
        """
        X = format_type(X, dtype=list, na_out=self.X)
        X = quote_ident(X)
        if contamination and not return_score:
            assert 0 < contamination < 1, ValueError(
                "Incorrect parameter 'contamination'.\nThe parameter "
                "'contamination' must be between 0.0 and 1.0, exclusive."
            )
        elif not return_score:
            assert 0 < cutoff < 1, ValueError(
                "Incorrect parameter 'cutoff'.\nThe parameter "
                "'cutoff' must be between 0.0 and 1.0, exclusive."
            )
        if return_score:
            other_parameters = ""
        elif contamination:
            other_parameters = f", contamination = {contamination}"
        else:
            other_parameters = f", threshold = {cutoff}"
        sql = f"""{self._vertica_predict_sql}({', '.join(X)} 
                   USING PARAMETERS 
                   model_name = '{self.model_name}', 
                   match_by_pos = 'true'{other_parameters})"""
        if return_score:
            sql = f"({sql}).anomaly_score"
        else:
            sql = f"(({sql}).is_anomaly)::int"
        return clean_query(sql)

    def to_memmodel(self) -> Union[mm.IsolationForest, mm.BinaryTreeAnomaly]:
        """
        Converts  the model  to an InMemory object  that
        can be used for different types of predictions.
        """
        if self.n_estimators_ == 1:
            return self.trees_[0]
        else:
            return mm.IsolationForest(self.trees_)

    # Prediction / Transformation Methods.

    def decision_function(
        self,
        vdf: SQLRelation,
        X: Optional[SQLColumns] = None,
        name: Optional[str] = None,
        inplace: bool = True,
    ) -> vDataFrame:
        """
        Returns  the  anomaly  score using the  input
        relation.

        Parameters
        ----------
        vdf: SQLRelation
            Object to use for the prediction. You can
            specify  a customized  relation if it  is
            enclosed  with  an  alias.  For  example,
            "(SELECT 1) x"   is   valid,    whereas
            "(SELECT 1)" and "SELECT 1" are invalid.
        X: SQLColumns, optional
            List of columns used to deploy the models.
            If empty,  the model  predictors are used.
        name: str, optional
            Name  of the  additional  vDataColumn.  If
            empty, a name is generated.
        inplace: bool, optional
            If  True,  the prediction is added to  the
            vDataFrame.

        Returns
        -------
        vDataFrame
            the input object.
        """
        # Inititalization
        if isinstance(vdf, str):
            vdf = vDataFrame(vdf)
        if not name:
            name = gen_name([self._model_type, self.model_name])

        # In Place
        vdf_return = vdf if inplace else vdf.copy()

        # Result
        return vdf_return.eval(
            name,
            self.deploySQL(
                X=X,
                return_score=True,
            ),
        )

    def predict(
        self,
        vdf: SQLRelation,
        X: Optional[SQLColumns] = None,
        name: Optional[str] = None,
        cutoff: PythonNumber = 0.7,
        contamination: Optional[PythonNumber] = None,
        inplace: bool = True,
    ) -> vDataFrame:
        """
        Predicts using the input relation.

        Parameters
        ----------
        vdf: SQLRelation
            Object to use for the prediction. You can
            specify  a customized  relation if it  is
            enclosed  with  an  alias.  For  example,
            "(SELECT 1) x"   is   valid,    whereas
            "(SELECT 1)" and "SELECT 1" are invalid.
        X: list, optional
            List of columns used to deploy the models.
            If empty,  the model  predictors are used.
        name: str, optional
            Name  of the  additional  vDataColumn.  If
            empty, a name is generated.
        cutoff: PythonNumber, optional
            Float  in the range (0.0, 1.0),  specifies
            the  threshold  that determines if a  data
            point is an anomaly.  If the anomaly_score
            for a data point  is greater than or equal
            to the cutfoff,  the data  point is marked
            as an anomaly.
        contamination: PythonNumber, optional
            Float  in the range (0,1), the approximate
            ratio of data points  in the training data
            that  should  be labeled as anomalous.  If
            this  parameter is specified,  the  cutoff
            parameter is ignored.
        inplace: bool, optional
            If  True,  the prediction is added to  the
            vDataFrame.

        Returns
        -------
        vDataFrame
            the input object.
        """
        # Initialization
        if isinstance(vdf, str):
            vdf = vDataFrame(vdf)
        if not name:
            name = gen_name([self._model_type, self.model_name])

        # In Place
        vdf_return = vdf if inplace else vdf.copy()

        # Result
        return vdf_return.eval(
            name, self.deploySQL(cutoff=cutoff, contamination=contamination, X=X)
        )

    # Plotting Methods.

    def _get_plot_args(self, method: Optional[str] = None) -> list:
        """
        Returns the args used by plotting methods.
        """
        if method == "contour":
            args = [self.X, self.deploySQL(X=self.X, return_score=True)]
        else:
            raise NotImplementedError
        return args

    def _get_plot_kwargs(
        self,
        nbins: int = 30,
        chart: Optional[PlottingObject] = None,
        method: Optional[str] = None,
    ) -> dict:
        """
        Returns the kwargs used by plotting methods.
        """
        res = {"nbins": nbins, "chart": chart}
        if method == "contour":
            res["func_name"] = "anomaly_score"
        else:
            raise NotImplementedError
        return res
