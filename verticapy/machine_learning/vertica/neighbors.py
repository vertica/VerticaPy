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
import itertools
import warnings
from typing import Literal, Optional
import numpy as np

from vertica_python.errors import QueryError

import verticapy._config.config as conf
from verticapy._typing import (
    NoneType,
    PlottingObject,
    PythonNumber,
    PythonScalar,
    SQLColumns,
    SQLRelation,
)
from verticapy._utils._gen import gen_name, gen_tmp_name
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import (
    clean_query,
    format_type,
    quote_ident,
    schema_relation,
)
from verticapy._utils._sql._sys import _executeSQL


from verticapy.core.tablesample.base import TableSample
from verticapy.core.vdataframe.base import vDataFrame

from verticapy.plotting._utils import PlottingUtils

import verticapy.machine_learning.metrics as mt
from verticapy.machine_learning.vertica.base import (
    MulticlassClassifier,
    Regressor,
    Tree,
    VerticaModel,
)
from verticapy.machine_learning.vertica.tree import DecisionTreeRegressor

from verticapy.sql.drop import drop


"""
Algorithms used for regression.
"""


class KNeighborsRegressor(Regressor):
    """
    [Beta Version]
    Creates a  KNeighborsRegressor object using the
    k-nearest neighbors algorithm. This object uses
    pure SQL to compute all the distances and final
    score.

    .. warning::

        This   algorithm   uses  a   CROSS  JOIN
        during   computation  and  is  therefore
        computationally  expensive at  O(n * n),
        where n is the total number of elements.
        Since  KNeighborsRegressor  uses  the p-
        distance,  it  is  highly  sensitive  to
        unnormalized data.

    .. important::

        This algorithm is not Vertica Native and relies solely
        on SQL for attribute computation. While this model does
        not take advantage of the benefits provided by a model
        management system, including versioning and tracking,
        the SQL code it generates can still be used to create a
        pipeline.

    Parameters
    ----------
    n_neighbors: int, optional
        Number of neighbors to consider when computing
        the score.
    p: int, optional
        The p of the p-distances (distance metric used
        during the model computation).

    Attributes
    ----------
    Many attributes are created during the fitting phase.

    n_neighbors_: int
        Number of neighbors.
    p_: int
        The p of the p-distances.

    .. note::

        All attributes can be accessed using the
        :py:meth:`verticapy.machine_learning.vertica.base.VerticaModel.get_attributes``
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

    First we import the ``KNeighborsRegressor`` model:

    .. ipython:: python

        from verticapy.machine_learning.vertica import KNeighborsRegressor

    Then we can create the model:

    .. ipython:: python

        model = KNeighborsRegressor()

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


    Metrics
    ^^^^^^^^

    We can get the entire report using:

    .. ipython:: python
        :suppress:

        result = model.report()
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_knnreg_report.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        result = model.report()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_knnreg_report.html

    .. important::

        Most metrics are computed using a single SQL query, but some
        of them might require multiple SQL queries. Selecting only the
        necessary metrics in the report can help optimize performance.
        E.g. ``model.report(metrics = ["mse", "r2"])``.

    For ``KNeighborsRegressor``, we can easily get the ANOVA table using:

    .. ipython:: python
        :suppress:

        result = model.report(metrics = "anova")
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_knnreg_report_anova.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        result = model.report(metrics = "anova")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_knnreg_report_anova.html

    You can also use the ``KNeighborsRegressor.score`` function to compute the R-squared
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
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_knnreg_prediction.html", "w")
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
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_linear_model_knnreg_prediction.html

    .. note::

        Predictions can be made automatically using the test set, in which
        case you don't need to specify the predictors. Alternatively, you
        can pass only the :py:class:`vDataFrame` to the
        :py:meth:`verticapy.machine_learning.vertica.linear_model.LinearModel.predict`
        function, but in this case, it's essential that the column names of
        the :py:class:`vDataFrame` match the predictors and response name in the
        model.

    Parameter Modification
    ^^^^^^^^^^^^^^^^^^^^^^^

    In order to see the parameters:

    .. ipython:: python

        model.get_params()

    And to manually change some of the parameters:

    .. ipython:: python

        model.set_params({'n_neighbors': 3})

    Model Register
    ^^^^^^^^^^^^^^

    In order to register the model for tracking and versioning:

    .. code-block:: python

        model.register("model_v1")

    Please refer to :ref:`notebooks/ml/model_tracking_versioning/index.html` for
    more details on model tracking and versioning.
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
    def _model_subcategory(self) -> Literal["REGRESSOR"]:
        return "REGRESSOR"

    @property
    def _model_type(self) -> Literal["KNeighborsRegressor"]:
        return "KNeighborsRegressor"

    @property
    def _attributes(self) -> list[str]:
        return ["n_neighbors_", "p_"]

    # System & Special Methods.

    @save_verticapy_logs
    def __init__(
        self,
        name: str = None,
        overwrite_model: bool = False,
        n_neighbors: int = 5,
        p: int = 2,
    ) -> None:
        super().__init__(name, overwrite_model)
        self.parameters = {"n_neighbors": n_neighbors, "p": p}

    def drop(self) -> bool:
        """
        KNN models are not stored in the Vertica DB.
        """
        return False

    # Attributes Methods.

    def _compute_attributes(self) -> None:
        self.p_ = self.parameters["p"]
        self.n_neighbors_ = self.parameters["n_neighbors"]

    # I/O Methods.

    def deploySQL(
        self,
        X: Optional[SQLColumns] = None,
        test_relation: Optional[str] = None,
        key_columns: Optional[SQLColumns] = None,
    ) -> str:
        """
        Returns the SQL code needed to deploy the model.

        Parameters
        ----------
        X: SQLColumns
            List of the predictors.
        test_relation: str, optional
            Relation used to do the predictions.
        key_columns: SQLColumns, optional
            A  list  of columns  to  include in  the  results,
            but to exclude from computation of the prediction.

        Returns
        -------
        str
            the SQL code needed to deploy the model.
        """
        key_columns = format_type(key_columns, dtype=list)
        X = format_type(X, dtype=list, na_out=self.X)
        X = quote_ident(X)
        if not test_relation:
            test_relation = self.test_relation
            if not key_columns:
                key_columns = [self.y]
        p = self.parameters["p"]
        X_str = ", ".join([f"x.{x}" for x in X])
        if key_columns:
            key_columns_str = ", " + ", ".join(
                ["x." + quote_ident(x) for x in key_columns]
            )
        else:
            key_columns_str = ""
        sql = [f"POWER(ABS(x.{X[i]} - y.{self.X[i]}), {p})" for i in range(len(self.X))]
        sql = f"""
            SELECT 
                {X_str}{key_columns_str}, 
                ROW_NUMBER() OVER(PARTITION BY {X_str}, row_id 
                                  ORDER BY POWER({' + '.join(sql)}, 1 / {p})) 
                                  AS ordered_distance, 
                y.{self.y} AS predict_neighbors, 
                row_id 
            FROM
                (SELECT 
                    *, 
                    ROW_NUMBER() OVER() AS row_id 
                 FROM {test_relation} 
                 WHERE {" AND ".join([f"{x} IS NOT NULL" for x in X])}) x 
                 CROSS JOIN 
                 (SELECT 
                    * 
                 FROM {self.input_relation} 
                 WHERE {" AND ".join([f"{x} IS NOT NULL" for x in self.X])}) y"""
        if key_columns:
            key_columns_str = ", " + ", ".join(quote_ident(key_columns))
        n_neighbors = self.parameters["n_neighbors"]
        sql = f"""
            (SELECT 
                {", ".join(X)}{key_columns_str}, 
                AVG(predict_neighbors) AS predict_neighbors 
             FROM ({sql}) z 
             WHERE ordered_distance <= {n_neighbors} 
             GROUP BY {", ".join(X)}{key_columns_str}, row_id) knr_table"""
        return clean_query(sql)

    # Prediction / Transformation Methods.

    def _predict(
        self,
        vdf: SQLRelation,
        X: Optional[SQLColumns] = None,
        name: Optional[str] = None,
        inplace: bool = True,
        **kwargs,
    ) -> vDataFrame:
        """
        Predicts using the input relation.
        """
        X = format_type(X, dtype=list)
        if isinstance(vdf, str):
            vdf = vDataFrame(vdf)
        X = quote_ident(X) if (X) else self.X
        key_columns = vdf.get_columns(exclude_columns=X)
        if "key_columns" in kwargs:
            key_columns_arg = None
        else:
            key_columns_arg = key_columns
        if not name:
            name = f"{self._model_type}_" + "".join(
                ch for ch in self.model_name if ch.isalnum()
            )
        if key_columns:
            key_columns_str = ", " + ", ".join(key_columns)
        else:
            key_columns_str = ""
        table = self.deploySQL(
            X=X, test_relation=vdf.current_relation(), key_columns=key_columns_arg
        )
        sql = f"""
            SELECT 
                {", ".join(X)}{key_columns_str}, 
                predict_neighbors AS {name} 
             FROM {table}"""
        if inplace:
            vdf.__init__(sql)
            return vdf
        else:
            return vDataFrame(sql)

    # Plotting Methods.

    def _get_plot_args(self, method: Optional[str] = None) -> list:
        """
        Returns the args used by plotting methods.
        """
        if method == "contour":
            args = [
                self.X,
                self.deploySQL(X=self.X, test_relation="{1}").replace(
                    "predict_neighbors", "{0}"
                ),
            ]
        else:
            raise NotImplementedError
        return args


"""
Algorithms used for classification.
"""


class KNeighborsClassifier(MulticlassClassifier):
    """
    [Beta Version]
    Creates a KNeighborsClassifier object using the
    k-nearest neighbors algorithm. This object uses
    pure SQL to compute all the distances and final
    score.

    .. warning::

        This   algorithm   uses  a   CROSS  JOIN
        during   computation  and  is  therefore
        computationally  expensive at  O(n * n),
        where n is the total number of elements.
        Since  KNeighborsClassifier uses  the p-
        distance,  it  is  highly  sensitive  to
        unnormalized data.

    .. important::

        This algorithm is not Vertica Native and relies solely
        on SQL for attribute computation. While this model does
        not take advantage of the benefits provided by a model
        management system, including versioning and tracking,
        the SQL code it generates can still be used to create a
        pipeline.

    Parameters
    ----------
    n_neighbors: int, optional
        Number  of neighbors to consider when computing  the
        score.
    p: int, optional
        The p of the p-distances (distance metric used
        during the model computation).

    Attributes
    ----------
    Many attributes are created during the fitting phase.

    n_neighbors_: int
        Number of neighbors.
    p_: int
        The p of the p-distances.
    classes_: numpy.array
        The classes labels.

    .. note::

        All attributes can be accessed using the
        :py:meth:`verticapy.machine_learning.vertica.base.VerticaModel.get_attributes``
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

    There are multiple classes for the "quality" column. Let us
    filter the data for classes between 5 and 7:

    .. code-block:: python

        data = data[data["quality"]>=5]
        data = data[data["quality"]<=7]

    We can the balance the dataset to ensure equal representation:

    .. code-block:: python

        data = data.balance(column="quality", x = 1)

    You can easily divide your dataset into training and testing subsets
    using the :py:meth:`vDataFrame.train_test_split` method. This is a
    crucial step when preparing your data for machine learning, as it
    allows you to evaluate the performance of your models accurately.

    .. code-block:: python

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
        data = data[data["quality"]>=5]
        data = data[data["quality"]<=7]
        data = data.balance(column="quality", x = 1)
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

    First we import the ``KNeighborsClassifier`` model:

    .. code-block::

        from verticapy.machine_learning.vertica import KNeighborsClassifier

    .. ipython:: python
        :suppress:

        from verticapy.machine_learning.vertica import KNeighborsClassifier

    Then we can create the model:

    .. ipython:: python
        :okwarning:

        model = KNeighborsClassifier(
           n_neighbors = 10,
           p = 2,
        )

    .. hint::

        In :py:mod:`verticapy` 1.0.x and higher, you do not need to specify the
        model name, as the name is automatically assigned. If you need to
        re-use the model, you can fetch the model name from the model's
        attributes.

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
                "density",
                "pH",
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

    .. important::

        As this model is not native, it solely relies on SQL statements to
        compute various attributes, storing them within the object. No data
        is saved in the database.

    Metrics
    ^^^^^^^^

    We can get the entire report using:

    .. ipython:: python
        :suppress:

        result = model.report()
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_neighbors_knc_report.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_neighbors_knc_report.html

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
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_neighbors_knc_report_cutoff.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report(cutoff = 0.2)

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_neighbors_knc_report_cutoff.html


    You can also use the ``KNeighborsClassifier.score`` function to compute any
    classification metric. The default metric is the accuracy:

    .. ipython:: python

        model.score(metric = "f1", average = "macro")

    .. note::

        For multi-class scoring, :py:mod:`verticapy` allows the
        flexibility to use three averaging techniques:
        micro, macro and weighted. Please refer to
        `this link <https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f>`_
        for more details on how they are calculated.

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
                "density",
                "pH",
            ],
            "prediction",
        )
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_neighbors_knc_prediction.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.predict(
            test,
            [
                "fixed_acidity",
                "volatile_acidity",
                "density",
                "pH",
            ],
            "prediction",
        )

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_neighbors_knc_prediction.html

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
                "density",
                "pH",
            ],
            "prediction",
        )
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_neighbors_knc_proba.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.predict_proba(
            test,
            [
                "fixed_acidity",
                "volatile_acidity",
                "density",
                "pH",
            ],
            "prediction",
        )

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_neighbors_knc_proba.html

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

    .. hint::

        In the context of multi-class classification, you typically work
        with an overall confusion matrix that summarizes the classification
        efficiency across all classes. However, you have the flexibility to
        specify a ``pos_label`` and adjust the cutoff threshold. In this case,
        a binary confusion matrix is computed, where the chosen class is treated
        as the positive class, allowing you to evaluate its efficiency as if it
        were a binary classification problem.

        .. ipython:: python

            model.confusion_matrix(pos_label = "5", cutoff = 0.6)

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

        model.roc_curve(pos_label = "5")

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")
        fig = model.roc_curve(pos_label = "5")
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_neighbors_knc_roc.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_neighbors_knc_roc.html

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

    **Contour plot** is another useful plot that can be produced
    for models with two predictors.

    .. code-block:: python

        model.contour(pos_label = "5")

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

        model.set_params({'n_neighbors': 8})

    Model Register
    ^^^^^^^^^^^^^^

    As this model is not native, it does not support model management and
    versioning. However, it is possible to use the SQL code it generates
    for deployment.

    Model Exporting
    ^^^^^^^^^^^^^^^^

    It is not possible to export this type of model, but you can still
    examine the SQL code generated by using the
    :py:meth:`verticapy.machine_learning.vertica.neighbors.KNeighborsClassifier.deploySQL`
    method.
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
    def _model_subcategory(self) -> Literal["CLASSIFIER"]:
        return "CLASSIFIER"

    @property
    def _model_type(self) -> Literal["KNeighborsClassifier"]:
        return "KNeighborsClassifier"

    @property
    def _attributes(self) -> list[str]:
        return ["classes_", "n_neighbors_", "p_"]

    # System & Special Methods.

    @save_verticapy_logs
    def __init__(
        self,
        name: str = None,
        overwrite_model: bool = False,
        n_neighbors: int = 5,
        p: int = 2,
    ) -> None:
        super().__init__(name, overwrite_model)
        self.parameters = {"n_neighbors": n_neighbors, "p": p}

    def drop(self) -> bool:
        """
        KNN models are not stored in the Vertica DB.
        """
        return False

    def _check_cutoff(
        self, cutoff: Optional[PythonNumber] = None
    ) -> Optional[PythonNumber]:
        if isinstance(cutoff, NoneType):
            return 1.0 / len(self.classes_)
        elif not 0 <= cutoff <= 1:
            ValueError(
                "Incorrect parameter 'cutoff'.\nThe cutoff "
                "must be between 0 and 1, inclusive."
            )
        else:
            return cutoff

    # Attributes Methods.

    def _compute_attributes(self) -> None:
        """
        Computes the model's attributes.
        """
        self.classes_ = self._get_classes()
        self.p_ = self.parameters["p"]
        self.n_neighbors_ = self.parameters["n_neighbors"]

    # I/O Methods.

    def deploySQL(
        self,
        X: Optional[SQLColumns] = None,
        test_relation: Optional[str] = None,
        predict: bool = False,
        key_columns: Optional[SQLColumns] = None,
    ) -> str:
        """
        Returns the SQL code needed to deploy the model.

        Parameters
        ----------
        X: SQLColumns
            List of the predictors.
        test_relation: str, optional
            Relation used to do the predictions.
        predict: bool, optional
            If set to True, returns the prediction instead
            of the probability.
        key_columns: SQLColumns, optional
            A  list of columns to include in the  results,
            but  to   exclude  from   computation  of  the
            prediction.

        Returns
        -------
        SQLExpression
            the SQL code needed to deploy the model.
        """
        key_columns = format_type(key_columns, dtype=list)
        X = format_type(X, dtype=list, na_out=self.X)
        X = quote_ident(X)
        if not test_relation:
            test_relation = self.test_relation
            if not key_columns:
                key_columns = [self.y]
        p = self.parameters["p"]
        n_neighbors = self.parameters["n_neighbors"]
        X_str = ", ".join([f"x.{x}" for x in X])
        if key_columns:
            key_columns_str = ", " + ", ".join(
                ["x." + quote_ident(x) for x in key_columns]
            )
        else:
            key_columns_str = ""
        sql = [f"POWER(ABS(x.{X[i]} - y.{self.X[i]}), {p})" for i in range(len(self.X))]
        sql = f"""
            SELECT 
                {X_str}{key_columns_str}, 
                ROW_NUMBER() OVER(PARTITION BY 
                                  {X_str}, row_id 
                                  ORDER BY POWER({' + '.join(sql)}, 1 / {p})) 
                                  AS ordered_distance, 
                y.{self.y} AS predict_neighbors, 
                row_id 
            FROM 
                (SELECT 
                    *, 
                    ROW_NUMBER() OVER() AS row_id 
                 FROM {test_relation} 
                 WHERE {" AND ".join([f"{x} IS NOT NULL" for x in X])}) x 
                 CROSS JOIN 
                (SELECT * FROM {self.input_relation} 
                 WHERE {" AND ".join([f"{x} IS NOT NULL" for x in self.X])}) y"""

        if key_columns:
            key_columns_str = ", " + ", ".join(quote_ident(key_columns))

        sql = f"""
            (SELECT 
                row_id, 
                {", ".join(X)}{key_columns_str}, 
                predict_neighbors, 
                COUNT(*) / {n_neighbors} AS proba_predict 
             FROM ({sql}) z 
             WHERE ordered_distance <= {n_neighbors} 
             GROUP BY {", ".join(X)}{key_columns_str}, 
                      row_id, 
                      predict_neighbors) kneighbors_table"""
        if predict:
            sql = f"""
                (SELECT 
                    {", ".join(X)}{key_columns_str}, 
                    predict_neighbors 
                 FROM 
                    (SELECT 
                        {", ".join(X)}{key_columns_str}, 
                        predict_neighbors, 
                        ROW_NUMBER() OVER (PARTITION BY {", ".join(X)} 
                                           ORDER BY proba_predict DESC) 
                                           AS order_prediction 
                     FROM {sql}) VERTICAPY_SUBTABLE 
                     WHERE order_prediction = 1) predict_neighbors_table"""
        return clean_query(sql)

    # Prediction / Transformation Methods.

    def _get_final_relation(
        self,
        pos_label: Optional[PythonScalar] = None,
    ) -> str:
        """
        Returns the final relation used to do the predictions.
        """
        filter_sql = ""
        if not (isinstance(pos_label, NoneType)):
            filter_sql = f"WHERE predict_neighbors = '{pos_label}'"
        return f"""
            (SELECT 
                * 
                FROM {self.deploySQL()}
            {filter_sql}) 
            final_centroids_relation"""

    def _get_y_proba(
        self,
        pos_label: Optional[PythonScalar] = None,
    ) -> str:
        """
        Returns the input which represents the model's probabilities.
        """
        return "proba_predict"

    def _get_y_score(
        self,
        pos_label: Optional[PythonScalar] = None,
        cutoff: Optional[PythonNumber] = None,
        allSQL: bool = False,
    ) -> str:
        """
        Returns the input that represents the model's scoring.
        """
        cutoff = self._check_cutoff(cutoff=cutoff)
        if isinstance(pos_label, NoneType) and not (self._is_binary_classifier()):
            return "predict_neighbors"
        elif self._is_binary_classifier():
            return f"""
                (CASE 
                    WHEN proba_predict > {cutoff} THEN '{self.classes_[1]}'
                    ELSE '{self.classes_[0]}'
                 END)"""
        elif allSQL:
            return f"""
                (CASE 
                    WHEN predict_neighbors = '{pos_label}' THEN proba_predict
                    ELSE NULL 
                 END)"""
        else:
            return f"""
                (CASE 
                    WHEN proba_predict < {cutoff} AND predict_neighbors = '{pos_label}' THEN NULL
                    ELSE predict_neighbors 
                 END)"""

    def _compute_accuracy(self) -> float:
        """
        Computes the model accuracy.
        """
        return mt.accuracy_score(
            self.y, "predict_neighbors", self.deploySQL(predict=True)
        )

    def _confusion_matrix(
        self,
        pos_label: Optional[PythonScalar] = None,
        cutoff: Optional[PythonNumber] = None,
    ) -> TableSample:
        """
        Computes the model confusion matrix.
        """
        if isinstance(pos_label, NoneType):
            input_relation = f"""
                (SELECT 
                    *, 
                    ROW_NUMBER() OVER(PARTITION BY {", ".join(self.X)}, row_id 
                                      ORDER BY proba_predict DESC) AS pos 
                 FROM {self.deploySQL()}) neighbors_table WHERE pos = 1"""
            return mt.confusion_matrix(
                self.y, "predict_neighbors", input_relation, labels=self.classes_
            )
        else:
            cutoff = self._check_cutoff(cutoff=cutoff)
            pos_label = self._check_pos_label(pos_label=pos_label)
            input_relation = (
                self.deploySQL() + f" WHERE predict_neighbors = '{pos_label}'"
            )
            y_score = f"(CASE WHEN proba_predict > {cutoff} THEN 1 ELSE 0 END)"
            y_true = f"DECODE({self.y}, '{pos_label}', 1, 0)"
            return mt.confusion_matrix(y_true, y_score, input_relation)

    # Model Evaluation Methods.

    def _predict(
        self,
        vdf: SQLRelation,
        X: Optional[SQLColumns] = None,
        name: Optional[str] = None,
        cutoff: Optional[PythonNumber] = None,
        inplace: bool = True,
        **kwargs,
    ) -> vDataFrame:
        """
        Predicts using the input relation.
        """
        X = format_type(X, dtype=list)
        cutoff = self._check_cutoff(cutoff=cutoff)
        if isinstance(vdf, str):
            vdf = vDataFrame(vdf)
        X = quote_ident(X) if (X) else self.X
        key_columns = vdf.get_columns(exclude_columns=X)
        if "key_columns" in kwargs:
            key_columns_arg = None
        else:
            key_columns_arg = key_columns
        if key_columns:
            key_columns_str = ", " + ", ".join(key_columns)
        else:
            key_columns_str = ""
        if not name:
            name = gen_name([self._model_type, self.model_name])

        if self._is_binary_classifier():
            table = self.deploySQL(
                X=X, test_relation=vdf.current_relation(), key_columns=key_columns_arg
            )
            sql = f"""
                (SELECT 
                    {", ".join(X)}{key_columns_str}, 
                    (CASE 
                        WHEN proba_predict > {cutoff} 
                            THEN '{self.classes_[1]}' 
                        ELSE '{self.classes_[0]}' 
                     END) AS {name} 
                 FROM {table} 
                 WHERE predict_neighbors = '{self.classes_[1]}') VERTICAPY_SUBTABLE"""
        else:
            table = self.deploySQL(
                X=X,
                test_relation=vdf.current_relation(),
                key_columns=key_columns_arg,
                predict=True,
            )
            sql = f"""
                SELECT 
                    {", ".join(X)}{key_columns_str}, 
                    predict_neighbors AS {name} 
                 FROM {table}"""
        if inplace:
            vdf.__init__(sql)
            return vdf
        else:
            return vDataFrame(sql)

    def _predict_proba(
        self,
        vdf: SQLRelation,
        X: Optional[SQLColumns] = None,
        name: Optional[str] = None,
        pos_label: Optional[PythonScalar] = None,
        inplace: bool = True,
        **kwargs,
    ) -> vDataFrame:
        """
        Returns the model's probabilities using the
        input relation.
        """
        # Inititalization
        X = format_type(X, dtype=list)
        assert pos_label is None or pos_label in self.classes_, ValueError(
            (
                "Incorrect parameter 'pos_label'.\nThe class label "
                f"must be in [{'|'.join([str(c) for c in self.classes_])}]. "
                f"Found '{pos_label}'."
            )
        )
        if isinstance(vdf, str):
            vdf = vDataFrame(vdf)
        X = quote_ident(X) if (X) else self.X
        key_columns = vdf.get_columns(exclude_columns=X)
        if not name:
            name = gen_name([self._model_type, self.model_name])
        if "key_columns" in kwargs:
            key_columns_arg = None
        else:
            key_columns_arg = key_columns

        # Generating the probabilities
        if isinstance(pos_label, NoneType):
            predict = [
                f"""ZEROIFNULL(AVG(DECODE(predict_neighbors, 
                                          '{c}', 
                                          proba_predict, 
                                          NULL))) AS {gen_name([name, c])}"""
                for c in self.classes_
            ]
        else:
            predict = [
                f"""ZEROIFNULL(AVG(DECODE(predict_neighbors, 
                                          '{pos_label}', 
                                          proba_predict, 
                                          NULL))) AS {name}"""
            ]
        if key_columns:
            key_columns_str = ", " + ", ".join(key_columns)
        else:
            key_columns_str = ""
        table = self.deploySQL(
            X=X, test_relation=vdf.current_relation(), key_columns=key_columns_arg
        )
        sql = f"""
            SELECT 
                {", ".join(X)}{key_columns_str}, 
                {", ".join(predict)} 
             FROM {table} 
             GROUP BY {", ".join(X + key_columns)}"""

        # Result
        if inplace:
            vdf.__init__(sql)
            return vdf
        else:
            return vDataFrame(sql)

    # Plotting Methods.

    def _get_plot_args(
        self, pos_label: Optional[PythonScalar] = None, method: Optional[str] = None
    ) -> list:
        """
        Returns the args used by plotting methods.
        """
        pos_label = self._check_pos_label(pos_label)
        if method == "contour":
            sql = (
                f"""
                SELECT
                    {', '.join(self.X)},
                    ZEROIFNULL(AVG(DECODE(predict_neighbors, 
                                          '{pos_label}', 
                                          proba_predict, 
                                          NULL))) AS {{0}}
                FROM """
                + self.deploySQL(X=self.X, test_relation="{1}")
                + f" GROUP BY {', '.join(self.X)}"
            )
            args = [self.X, sql]
        else:
            input_relation = (
                self.deploySQL() + f" WHERE predict_neighbors = '{pos_label}'"
            )
            args = [self.y, "proba_predict", input_relation, pos_label]
        return args

    def _get_plot_kwargs(
        self,
        pos_label: Optional[PythonScalar] = None,
        nbins: int = 30,
        chart: Optional[PlottingObject] = None,
        method: Optional[str] = None,
    ) -> dict:
        """
        Returns the kwargs used by plotting methods.
        """
        pos_label = self._check_pos_label(pos_label)
        res = {"nbins": nbins, "chart": chart}
        if method == "contour":
            res["func_name"] = f"p({self.y} = '{pos_label}')"
        elif method == "cutoff":
            res["cutoff_curve"] = True
        return res


"""
Algorithms used for density analysis.
"""


class KernelDensity(Regressor, Tree):
    """
    [Beta Version]
    Creates a KernelDensity object.
    This object uses pure SQL to compute the final score.

    Parameters
    ----------
    name: str, optional
        Name of the model. This is not a built-in model, so
        this name is used  to build the final table.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    bandwidth: PythonNumber, optional
        The bandwidth of the kernel.
    kernel: str, optional
        The kernel used during the learning phase.
            gaussian  : Gaussian Kernel.
            logistic  : Logistic Kernel.
            sigmoid   : Sigmoid Kernel.
            silverman : Silverman Kernel.
    p: int, optional
        The p of the p-distances (distance metric used
        during the model computation).
    max_leaf_nodes: PythonNumber, optional
        The maximum number of leaf nodes,  an integer between
        1 and 1e9, inclusive.
    max_depth: int, optional
        The maximum tree depth,  an integer between 1 and 100,
        inclusive.
    min_samples_leaf: int, optional
        The  minimum number of  samples each branch must  have
        after splitting a node,  an integer between 1 and 1e6,
        inclusive. A split that results in fewer remaining
        samples is discarded.
    nbins: int, optional
        The  number  of  bins used to discretize  the  input
        features.
    xlim: list, optional
        List of tuples used to compute the kernel window.

    Attributes
    ----------
    Several attributes are computed during the fitting phase,
    and in the case of kernel density estimation (KDE), a
    :py:meth:`verticapy.machine_learning.vertica.ensemble.RandomForestRegressor``
    is employed to approximate the k-nearest neighbors (KNN)
    computation. This reliance on RandomForestRegressor
    enhances the efficiency and accuracy of the KDE algorithm.
    """

    # Properties.

    @property
    def _is_native(self) -> Literal[False]:
        return False

    @property
    def _is_using_native(self) -> Literal[True]:
        return True

    @property
    def _vertica_fit_sql(self) -> Literal["RF_REGRESSOR"]:
        return "RF_REGRESSOR"

    @property
    def _vertica_predict_sql(self) -> Literal["PREDICT_RF_REGRESSOR"]:
        return "PREDICT_RF_REGRESSOR"

    # This is an exception. Although KernelDensity is a subclass of Regressor,
    # but it is UNSUPERVISED.
    @property
    def _model_category(self) -> Literal["UNSUPERVISED"]:
        return "UNSUPERVISED"

    @property
    def _model_subcategory(self) -> Literal["PREPROCESSING"]:
        return "PREPROCESSING"

    @property
    def _model_type(self) -> Literal["KernelDensity"]:
        return "KernelDensity"

    # System & Special Methods.

    @save_verticapy_logs
    def __init__(
        self,
        name: str = None,
        overwrite_model: bool = False,
        bandwidth: PythonNumber = 1.0,
        kernel: Literal["gaussian", "logistic", "sigmoid", "silverman"] = "gaussian",
        p: int = 2,
        max_leaf_nodes: PythonNumber = 1e9,
        max_depth: int = 5,
        min_samples_leaf: int = 1,
        nbins: int = 5,
        xlim: Optional[list] = None,
        **kwargs,
    ) -> None:
        super().__init__(name, overwrite_model)
        self.parameters = {
            "nbins": nbins,
            "p": p,
            "bandwidth": bandwidth,
            "kernel": str(kernel).lower(),
            "max_leaf_nodes": int(max_leaf_nodes),
            "max_depth": int(max_depth),
            "min_samples_leaf": int(min_samples_leaf),
            "xlim": format_type(xlim, dtype=list),
        }
        self._verticapy_store = "store" not in kwargs or kwargs["store"]
        self.verticapy_x = None
        self.verticapy_y = None

    def drop(self) -> bool:
        """
        Drops the model from the Vertica database.
        """
        try:
            table_name = self.model_name.replace('"', "") + "_KernelDensity_Map"
            _executeSQL(
                query=f"SELECT KDE FROM {table_name} LIMIT 0;",
                title="Looking if the KDE table exists.",
            )
            drop(table_name, method="table")
        except QueryError:
            return False
        return drop(self.model_name, method="model")

    # Attributes Methods.

    def _density_kde(
        self, vdf: vDataFrame, columns: SQLColumns, kernel: str, x, p: int, h=None
    ) -> str:
        """
        Returns the result of the KDE.
        """
        for col in columns:
            if not vdf[col].isnum():
                raise TypeError(
                    f"Cannot compute KDE for non-numerical columns. {col} is not numerical."
                )
        if kernel == "gaussian":
            fkernel = "EXP(-1 / 2 * POWER({0}, 2)) / SQRT(2 * PI())"

        elif kernel == "logistic":
            fkernel = "1 / (2 + EXP({0}) + EXP(-{0}))"

        elif kernel == "sigmoid":
            fkernel = "2 / (PI() * (EXP({0}) + EXP(-{0})))"

        elif kernel == "silverman":
            fkernel = (
                "EXP(-1 / SQRT(2) * ABS({0})) / 2 * SIN(ABS({0}) / SQRT(2) + PI() / 4)"
            )

        else:
            raise ValueError(
                "The parameter 'kernel' must be in [gaussian|logistic|sigmoid|silverman]."
            )
        if isinstance(x, (tuple)):
            return self._density_kde(vdf, columns, kernel, [x], p, h)[0]
        elif isinstance(x, (list)):
            N = vdf.shape()[0]
            L = []
            for xj in x:
                distance = []
                for i in range(len(columns)):
                    distance += [f"POWER({columns[i]} - {xj[i]}, {p})"]
                distance = " + ".join(distance)
                distance = f"POWER({distance}, {1.0 / p})"
                fkernel_tmp = fkernel.format(f"{distance} / {h}")
                L += [f"SUM({fkernel_tmp}) / ({h} * {N})"]
            query = f"""
                SELECT 
                    /*+LABEL('learn.neighbors.KernelDensity.fit')*/ 
                    {", ".join(L)} 
                FROM {vdf}"""
            result = _executeSQL(
                query=query, title="Computing the KDE", method="fetchrow"
            )
            return list(result)
        else:
            return 0

    def _density_compute(
        self,
        vdf: vDataFrame,
        columns: SQLColumns,
        h=None,
        kernel: str = "gaussian",
        nbins: int = 5,
        p: int = 2,
    ) -> list:
        """
        Returns the result of the KDE for all the data points.
        """
        columns = vdf.format_colnames(columns)
        x_vars = []
        y = []
        for idx, column in enumerate(columns):
            if self.parameters["xlim"]:
                try:
                    x_min, x_max = self.parameters["xlim"][idx]
                except:
                    warning_message = (
                        f"Wrong xlim for the vDataColumn {column}.\n"
                        "The max and the min will be used instead."
                    )
                    warnings.warn(warning_message, Warning)
                    x_min, x_max = vdf.agg(
                        func=["min", "max"], columns=[column]
                    ).transpose()[column]
            else:
                x_min, x_max = vdf.agg(
                    func=["min", "max"], columns=[column]
                ).transpose()[column]
            x_vars += [
                [(x_max - x_min) * i / nbins + x_min for i in range(0, nbins + 1)]
            ]
        x = list(itertools.product(*x_vars))
        try:
            y = self._density_kde(vdf, columns, kernel, x, p, h)
        except:
            for xi in x:
                K = self._density_kde(vdf, columns, kernel, xi, p, h)
                y += [K]
        return [x, y]

    # Model Fitting Method.

    def fit(
        self,
        input_relation: SQLRelation,
        X: Optional[SQLColumns] = None,
        return_report: bool = False,
    ) -> None:
        """
        Trains the model.

        Parameters
        ----------
        input_relation: SQLRelation
            Training relation.
        X: list, optional
            List of the predictors.
        """
        X = format_type(X, dtype=list)
        X = quote_ident(X)
        if self.overwrite_model:
            self.drop()
        else:
            self._is_already_stored(raise_error=True)
        if isinstance(input_relation, vDataFrame):
            if not X:
                X = input_relation.numcol()
            vdf = input_relation
            input_relation = input_relation.current_relation()
        else:
            vdf = vDataFrame(input_relation)
            if not X:
                X = vdf.numcol()
        X = vdf.format_colnames(X)
        x, y = self._density_compute(
            vdf,
            X,
            self.parameters["bandwidth"],
            self.parameters["kernel"],
            self.parameters["nbins"],
            self.parameters["p"],
        )
        table_name = self.model_name.replace('"', "") + "_KernelDensity_Map"
        if self._verticapy_store:
            _executeSQL(
                query=f"""
                    CREATE TABLE {table_name} AS    
                        SELECT 
                            /*+LABEL('learn.neighbors.KernelDensity.fit')*/
                            {", ".join(X)}, 0.0::float AS KDE 
                        FROM {vdf} 
                        LIMIT 0""",
                print_time_sql=False,
            )
            r, idx = 0, 0
            while r < len(y):
                values = []
                m = min(r + 100, len(y))
                for i in range(r, m):
                    values += ["SELECT " + str(x[i] + (y[i],))[1:-1]]
                _executeSQL(
                    query=f"""
                    INSERT /*+LABEL('learn.neighbors.KernelDensity.fit')*/ 
                    INTO {table_name}
                    ({", ".join(X)}, KDE) {" UNION ".join(values)}""",
                    title=f"Computing the KDE [Step {idx}].",
                )
                _executeSQL("COMMIT;", print_time_sql=False)
                r += 100
                idx += 1
            self.X, self.input_relation = X, input_relation
            self.map = table_name
            self.y = "KDE"
            model = DecisionTreeRegressor(
                name=self.model_name,
                max_features=len(self.X),
                max_leaf_nodes=self.parameters["max_leaf_nodes"],
                max_depth=self.parameters["max_depth"],
                min_samples_leaf=self.parameters["min_samples_leaf"],
                nbins=1000,
            )
            model.fit(self.map, self.X, "KDE")
        else:
            self.X, self.input_relation = X, input_relation
            self.verticapy_x = x
            self.verticapy_y = y

    # Plotting Methods.

    def _compute_plot_params(self) -> tuple[dict, dict]:
        if len(self.X) == 1:
            if self._verticapy_store:
                query = f"""
                    SELECT 
                        /*+LABEL('learn.neighbors.KernelDensity.plot')*/ 
                        {self.X[0]}, KDE 
                    FROM {self.map} ORDER BY 1"""
                result = _executeSQL(query, method="fetchall", print_time_sql=False)
                x, y = [v[0] for v in result], [v[1] for v in result]
            else:
                x, y = [v[0] for v in self.verticapy_x], self.verticapy_y
            data = {
                "x": np.array(x).astype(float),
                "y": np.array(y).astype(float),
            }
            layout = {
                "x_label": self.X[0],
                "y_label": "density",
            }
        elif len(self.X) == 2:
            n = self.parameters["nbins"]
            if self._verticapy_store:
                query = f"""
                    SELECT 
                        /*+LABEL('learn.neighbors.KernelDensity.plot')*/ 
                        {self.X[0]}, 
                        {self.X[1]}, 
                        KDE 
                    FROM {self.map} 
                    ORDER BY 1, 2"""
                result = _executeSQL(query, method="fetchall", print_time_sql=False)
                x, y, z = (
                    [v[0] for v in result],
                    [v[1] for v in result],
                    [v[2] for v in result],
                )
            else:
                x, y, z = (
                    [v[0] for v in self.verticapy_x],
                    [v[1] for v in self.verticapy_x],
                    self.verticapy_y,
                )
            X, idx = [], 0
            while idx < (n + 1) * (n + 1):
                X += [[z[idx + i] for i in range(n + 1)]]
                idx += n + 1
            extent = [
                float(np.nanmin(x)),
                float(np.nanmax(x)),
                float(np.nanmin(y)),
                float(np.nanmax(y)),
            ]
            data = {
                "X": np.array(X).astype(float),
            }
            layout = {
                "x_label": self.X[0],
                "y_label": self.X[1],
                "extent": extent,
            }
        else:
            raise AttributeError("KDE Plots are only available in 1D or 2D.")
        return data, layout

    def plot(
        self, chart: Optional[PlottingObject] = None, **style_kwargs
    ) -> PlottingObject:
        """
        Draws the Model.

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
            Plotting Object.
        """
        data, layout = self._compute_plot_params()
        if len(self.X) == 1:
            vpy_plt, kwargs = PlottingUtils().get_plotting_lib(
                class_name="DensityPlot",
                chart=chart,
                style_kwargs=style_kwargs,
            )
            fun = vpy_plt.DensityPlot
        elif len(self.X) == 2:
            vpy_plt, kwargs = PlottingUtils().get_plotting_lib(
                class_name="DensityPlot2D",
                chart=chart,
                style_kwargs=style_kwargs,
            )
            fun = vpy_plt.DensityPlot2D
        else:
            raise AttributeError("KDE Plots are only available in 1D or 2D.")
        return fun(data=data, layout=layout).draw(**kwargs)


"""
Algorithms used for anomaly detection.
"""


class LocalOutlierFactor(VerticaModel):
    """
    [Beta Version]
    Creates a LocalOutlierFactor object by using the
    Local Outlier Factor algorithm as defined by Markus
    M. Breunig, Hans-Peter Kriegel, Raymond T. Ng and Jörg
    Sander. This object is using pure SQL to compute all
    the distances and final score.

    .. warning :

        This   algorithm   uses  a   CROSS  JOIN
        during   computation  and  is  therefore
        computationally  expensive at  O(n * n),
        where n is the total number of elements.
        Since  LocalOutlierFactor   uses  the p-
        distance,  it  is  highly  sensitive  to
        unnormalized data.
        A  table  is created at the  end of
        the learning phase.

    .. important::

        This algorithm is not Vertica Native and relies solely
        on SQL for attribute computation. While this model does
        not take advantage of the benefits provided by a model
        management system, including versioning and tracking,
        the SQL code it generates can still be used to create a
        pipeline.

    Parameters
    ----------
    name: str, optional
        Name  of the  model.  This is not a  built-in
        model, so this name is used to build the
        final table.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    n_neighbors: int, optional
        Number of neighbors to consider when computing
        the score.
    p: int, optional
        The p of the p-distances (distance metric used
        during the model computation).

    Attributes
    ----------
    Many attributes are created during the fitting phase.

    n_neighbors_: int
        Number of neighbors.
    p_: int
        The p of the p-distances.
    n_errors_: int
        Number of errors during the model fitting phase.
    cnt_: int
        Number of elements accepted during the model
        fitting phase.

    .. note::

        All attributes can be accessed using the
        :py:meth:`verticapy.machine_learning.vertica.base.VerticaModel.get_attributes``
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

    First we import the ``LocalOutlierFactor`` model:

    .. code-block::

        from verticapy.machine_learning.vertica import LocalOutlierFactor

    .. ipython:: python
        :suppress:

        from verticapy.machine_learning.vertica import LocalOutlierFactor

    Then we can create the model:

    .. ipython:: python
        :okwarning:

        model = LocalOutlierFactor(
            n_neighbors = 10,
            p = 2,
        )

    .. important::

        As this model is not native, it solely relies on SQL statements to
        compute various attributes, storing them within the object. No data
        is saved in the database.

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

    .. important::

        As this model is not native, it solely relies on SQL statements to
        compute various attributes, storing them within the object. No data
        is saved in the database.

    Prediction
    ^^^^^^^^^^^

    To find out the LOF score for each datapoint:

    .. ipython:: python
        :suppress:

        result = model.predict()
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_lof_prediction.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.predict()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_lof_prediction.html

    As shown above, a new column has been created, containing
    the lof score.


    Plots - Outliers
    ^^^^^^^^^^^^^^^^^

    Plots highlighting the outliers can be easily drawn using:

    .. code-block:: python

        model.plot()

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")
        fig = model.plot(width = 600)
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_lof_plot.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_lof_plot.html

    .. important::

        Please refer to :ref:`chart_gallery.lof` for more examples.

    Parameter Modification
    ^^^^^^^^^^^^^^^^^^^^^^^

    In order to see the parameters:

    .. ipython:: python

        model.get_params()

    And to manually change some of the parameters:

    .. ipython:: python

        model.set_params({'p': 3})

    Model Register
    ^^^^^^^^^^^^^^

    As this model is not native, it does not support model management and
    versioning. However, it is possible to use the SQL code it generates
    for deployment.
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
    def _model_category(self) -> Literal["UNSUPERVISED"]:
        return "UNSUPERVISED"

    @property
    def _model_subcategory(self) -> Literal["ANOMALY_DETECTION"]:
        return "ANOMALY_DETECTION"

    @property
    def _model_type(self) -> Literal["LocalOutlierFactor"]:
        return "LocalOutlierFactor"

    @property
    def _attributes(self) -> list[str]:
        return ["n_neighbors_", "p_", "n_errors_", "cnt_"]

    # System & Special Methods.

    @save_verticapy_logs
    def __init__(
        self,
        name: str = None,
        overwrite_model: bool = False,
        n_neighbors: int = 20,
        p: int = 2,
    ) -> None:
        super().__init__(name, overwrite_model)
        self.parameters = {"n_neighbors": n_neighbors, "p": p}

    def drop(self) -> bool:
        """
        Drops the model from the Vertica database.
        """
        try:
            _executeSQL(
                query=f"SELECT lof_score FROM {self.model_name} LIMIT 0;",
                title="Looking if the LOF table exists.",
            )
            return drop(self.model_name, method="table")
        except QueryError:
            return False

    # Attributes Methods.

    def _compute_attributes(self) -> None:
        """
        Computes the model's attributes.
        """
        self.p_ = self.parameters["p"]
        self.n_neighbors_ = self.parameters["n_neighbors"]
        self.cnt_ = _executeSQL(
            query=f"SELECT /*+LABEL('learn.VerticaModel.plot')*/ COUNT(*) FROM {self.model_name}",
            method="fetchfirstelem",
            print_time_sql=False,
        )

    # Model Fitting Method.

    def fit(
        self,
        input_relation: SQLRelation,
        X: Optional[SQLColumns] = None,
        key_columns: Optional[SQLColumns] = None,
        index: Optional[str] = None,
        return_report: bool = False,
    ) -> None:
        """
        Trains the model.

        Parameters
        ----------
        input_relation: SQLRelation
                Training relation.
        X: SQLColumns, optional
                List of the predictors.
        key_columns: SQLColumns, optional
                Columns  not  used   during  the   algorithm
            computation  but   which  are  used  to
            create the final relation.
        index: str, optional
                Index  used to seperately identify each row.
            To avoid the creation of temporary tables,
            it is recommended that you already have an
            index in the main table.
        """
        X, key_columns = format_type(X, key_columns, dtype=list)
        X = quote_ident(X)
        if self.overwrite_model:
            self.drop()
        else:
            self._is_already_stored(raise_error=True)
        self.key_columns = quote_ident(key_columns)
        if isinstance(input_relation, vDataFrame):
            self.input_relation = input_relation.current_relation()
            if not X:
                X = input_relation.numcol()
        else:
            self.input_relation = input_relation
            if not X:
                X = vDataFrame(input_relation).numcol()
        self.X = X
        n_neighbors = self.parameters["n_neighbors"]
        p = self.parameters["p"]
        schema = schema_relation(input_relation)[0]
        tmp_main_table_name = gen_tmp_name(name="main")
        tmp_distance_table_name = gen_tmp_name(name="distance")
        tmp_lrd_table_name = gen_tmp_name(name="lrd")
        tmp_lof_table_name = gen_tmp_name(name="lof")
        try:
            if not index:
                index = "id"
                main_table = tmp_main_table_name
                schema = "v_temp_schema"
                drop(f"v_temp_schema.{tmp_main_table_name}", method="table")
                _executeSQL(
                    query=f"""
                        CREATE LOCAL TEMPORARY TABLE {main_table} 
                        ON COMMIT PRESERVE ROWS AS 
                            SELECT 
                                /*+LABEL('learn.neighbors.LocalOutlierFactor.fit')*/ 
                                ROW_NUMBER() OVER() AS id, 
                                {', '.join(X + key_columns)} 
                            FROM {self.input_relation} 
                            WHERE {' AND '.join([f"{x} IS NOT NULL" for x in X])}""",
                    print_time_sql=False,
                )
            else:
                main_table = self.input_relation
            sql = [f"POWER(ABS(x.{X[i]} - y.{X[i]}), {p})" for i in range(len(X))]
            distance = f"POWER({' + '.join(sql)}, 1 / {p})"
            drop(f"v_temp_schema.{tmp_distance_table_name}", method="table")
            _executeSQL(
                query=f"""
                    CREATE LOCAL TEMPORARY TABLE {tmp_distance_table_name} 
                    ON COMMIT PRESERVE ROWS AS 
                        SELECT 
                            /*+LABEL('learn.neighbors.LocalOutlierFactor.fit')*/ 
                            node_id, 
                            nn_id, 
                            distance, 
                            knn 
                        FROM 
                            (SELECT 
                                x.{index} AS node_id, 
                                y.{index} AS nn_id, 
                                {distance} AS distance, 
                                ROW_NUMBER() OVER(PARTITION BY x.{index} 
                                                  ORDER BY {distance}) AS knn 
                             FROM {schema}.{main_table} AS x 
                             CROSS JOIN 
                             {schema}.{main_table} AS y) distance_table 
                        WHERE knn <= {n_neighbors + 1}""",
                title="Computing the LOF [Step 0].",
            )
            drop(f"v_temp_schema.{tmp_lrd_table_name}", method="table")
            _executeSQL(
                query=f"""
                    CREATE LOCAL TEMPORARY TABLE {tmp_lrd_table_name} 
                    ON COMMIT PRESERVE ROWS AS 
                        SELECT 
                            /*+LABEL('learn.neighbors.LocalOutlierFactor.fit')*/ 
                            distance_table.node_id, 
                            {n_neighbors} / SUM(
                                    CASE 
                                        WHEN distance_table.distance 
                                             > kdistance_table.distance 
                                        THEN distance_table.distance 
                                        ELSE kdistance_table.distance 
                                     END) AS lrd 
                        FROM 
                            (v_temp_schema.{tmp_distance_table_name} AS distance_table 
                             LEFT JOIN 
                             (SELECT 
                                 node_id, 
                                 nn_id, 
                                 distance AS distance 
                              FROM v_temp_schema.{tmp_distance_table_name} 
                              WHERE knn = {n_neighbors + 1}) AS kdistance_table
                             ON distance_table.nn_id = kdistance_table.node_id) x 
                        GROUP BY 1""",
                title="Computing the LOF [Step 1].",
            )
            drop(f"v_temp_schema.{tmp_lof_table_name}", method="table")
            _executeSQL(
                query=f"""
                    CREATE LOCAL TEMPORARY TABLE {tmp_lof_table_name} 
                    ON COMMIT PRESERVE ROWS AS 
                    SELECT 
                        /*+LABEL('learn.neighbors.LocalOutlierFactor.fit')*/ 
                        x.node_id, 
                        SUM(y.lrd) / (MAX(x.node_lrd) * {n_neighbors}) AS LOF 
                    FROM 
                        (SELECT 
                            n_table.node_id, 
                            n_table.nn_id, 
                            lrd_table.lrd AS node_lrd 
                         FROM 
                            v_temp_schema.{tmp_distance_table_name} AS n_table 
                         LEFT JOIN 
                            v_temp_schema.{tmp_lrd_table_name} AS lrd_table 
                        ON n_table.node_id = lrd_table.node_id) x 
                    LEFT JOIN 
                        v_temp_schema.{tmp_lrd_table_name} AS y 
                    ON x.nn_id = y.node_id GROUP BY 1""",
                title="Computing the LOF [Step 2].",
            )
            _executeSQL(
                query=f"""
                    CREATE TABLE {self.model_name} AS 
                        SELECT 
                            /*+LABEL('learn.neighbors.LocalOutlierFactor.fit')*/ 
                            {', '.join(X + self.key_columns)}, 
                            (CASE WHEN lof > 1e100 OR lof != lof THEN 0 ELSE lof END) AS lof_score
                        FROM 
                            {main_table} AS x 
                        LEFT JOIN 
                            v_temp_schema.{tmp_lof_table_name} AS y 
                        ON x.{index} = y.node_id""",
                title="Computing the LOF [Step 3].",
            )
            self.n_errors_ = _executeSQL(
                query=f"""
                    SELECT 
                        /*+LABEL('learn.neighbors.LocalOutlierFactor.fit')*/ 
                        COUNT(*) 
                    FROM {schema}.{tmp_lof_table_name} z 
                    WHERE lof > 1e100 OR lof != lof""",
                method="fetchfirstelem",
                print_time_sql=False,
            )
            self._compute_attributes()
        finally:
            drop(f"v_temp_schema.{tmp_main_table_name}", method="table")
            drop(f"v_temp_schema.{tmp_distance_table_name}", method="table")
            drop(f"v_temp_schema.{tmp_lrd_table_name}", method="table")
            drop(f"v_temp_schema.{tmp_lof_table_name}", method="table")

    # Prediction / Transformation Methods.

    def predict(self) -> vDataFrame:
        """
        Creates a vDataFrame of the model.

        Returns
        -------
        vDataFrame
            the vDataFrame including the prediction.
        """
        return vDataFrame(self.model_name)

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
            class_name="LOFPlot",
            chart=chart,
            style_kwargs=style_kwargs,
        )
        return vpy_plt.LOFPlot(
            vdf=vDataFrame(self.model_name),
            columns=self.X + ["lof_score"],
            max_nb_points=max_nb_points,
        ).draw(**kwargs)
