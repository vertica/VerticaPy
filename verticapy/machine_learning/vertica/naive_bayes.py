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
from typing import Literal
import numpy as np

from verticapy._typing import PythonNumber
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import quote_ident
from verticapy._utils._sql._vertica_version import check_minimum_version
from verticapy.errors import MissingRelation

from verticapy.core.vdataframe.base import vDataFrame

import verticapy.machine_learning.memmodel as mm

from verticapy.machine_learning.vertica.base import MulticlassClassifier

"""
Algorithms used for classification.
"""


class NaiveBayes(MulticlassClassifier):
    """
    Creates  a  NaiveBayes object using the Vertica
    Naive  Bayes  algorithm.  It is a "probabilistic
    classifier"  based  on  applying Bayes' theorem
    with strong (naïve) independence assumptions
    between the features.

    Parameters
    ----------
    name: str, optional
        Name  of  the  model.  The  model is stored
        in the database.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    alpha: float, optional
        A  float  that  specifies  use  of  Laplace
        smoothing if the event model is categorical,
        multinomial, or Bernoulli.
    nbtype: str, optional
        Naive Bayes type.

        - auto:
            Vertica NaiveBayes objects
            treat columns according to data type:

            - FLOAT:
                values are assumed to follow some
                Gaussian distribution.
            - INTEGER:
                values are assumed to belong to
                one multinomial distribution.
            - CHAR/VARCHAR:
                values  are  assumed  to
                follow  some categorical distribution.
                The  string  values  stored  in  these
                columns  must be no greater than  128
                characters.
            - BOOLEAN:
                values    are   treated   as
                categorical with two values.

        - bernoulli:
            Casts the variables to boolean.
        - categorical:
            Casts the variables to categorical.
        - multinomial:
            Casts the variables to integer.
        - gaussian:
            Casts the variables to float.

    Attributes
    ----------
    Many attributes are created during the fitting phase.

    prior_: numpy.array
        The model's classes probabilities.
    attributes_: list of dict
        List  of the model's attributes. Each feature  is
        represented by a dictionary, which differs based
        on the distribution.
    classes_: numpy.array
        The classes labels.

    .. note::

        All attributes can be accessed using the
        :py:meth:`verticapy.machine_learning.vertica.base.MulticlassClassifier.get_attributes``
        method.

    .. note::

        Several other attributes can be accessed by using the
        :py:meth:`verticapy.machine_learning.vertica.base.MulticlassClassifier.get_vertica_attributes``
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

    For this example, we will use the iris dataset.

    .. code-block:: python

        import verticapy.datasets as vpd

        data = vpd.load_iris()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_iris.html

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

        data = vpd.load_iris()
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
        data = vpd.load_iris()
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

    First we import the ``NaiveBayes`` model:

    .. ipython:: python

        from verticapy.machine_learning.vertica import NaiveBayes

    Then we can create the model:

    .. ipython:: python

        model = NaiveBayes()

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
                "SepalLengthCm",
                "SepalWidthCm",
                "PetalLengthCm",
                "PetalWidthCm",
            ],
            "Species",
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
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_NB_naivebayes_report.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_NB_naivebayes_report.html

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
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_NB_naivebayes_report_cutoff.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report(cutoff = 0.2)

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_NB_naivebayes_report_cutoff.html


    You can also use the ``NaiveBayes.score`` function to compute any
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
                "SepalLengthCm",
                "SepalWidthCm",
                "PetalLengthCm",
                "PetalWidthCm",
            ],
            "prediction",
        )
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_NB_naivebayes_prediction.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.predict(
            test,
            [
                "SepalLengthCm",
                "SepalWidthCm",
                "PetalLengthCm",
                "PetalWidthCm",
            ],
            "prediction",
        )

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_NB_naivebayes_prediction.html

    .. note::

        Predictions can be made automatically using the test set, in which
        case you don't need to specify the predictors. Alternatively, you
        can pass only the :py:class:`vDataFrame` to the
        :py:meth:`verticapy.machine_learning.vertica.naive_bayes.NaiveBayes.predict`
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
                "SepalLengthCm",
                "SepalWidthCm",
                "PetalLengthCm",
                "PetalWidthCm",
            ],
            "prediction",
        )
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_NB_naivebayes_proba.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.predict_proba(
            test,
            [
                "SepalLengthCm",
                "SepalWidthCm",
                "PetalLengthCm",
                "PetalWidthCm",
            ],
            "prediction",
        )

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_NB_naivebayes_proba.html

    .. note::

        Probabilities are added to the vDataFrame, and VerticaPy uses the
        corresponding probability function in SQL behind the scenes. You
        can use the ``pos_label`` parameter to add only the probability
        of the selected category.

    Confusion Matrix
    ^^^^^^^^^^^^^^^^^

    You can obtain the confusion matrix.

    .. ipython:: python

        model.confusion_matrix()

    .. hint::

        In the context of multi-class classification, you typically work
        with an overall confusion matrix that summarizes the classification
        efficiency across all classes. However, you have the flexibility to
        specify a ``pos_label`` and adjust the cutoff threshold. In this case,
        a binary confusion matrix is computed, where the chosen class is treated
        as the positive class, allowing you to evaluate its efficiency as if it
        were a binary classification problem.

        .. ipython:: python

            model.confusion_matrix(pos_label = "Iris-setosa", cutoff = 0.6)

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

        model.roc_curve(pos_label = "Iris-setosa")

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")
        fig = model.roc_curve(pos_label = "Iris-setosa")
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_NB_naivebayes_roc.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_NB_naivebayes_roc.html

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

        model.contour(pos_label = "Iris-setosa")

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

        model.set_params({'alpha': 0.9})

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

        X = [[5, 2, 3, 1]]
        model.to_python()(X)

    .. hint::

        The
        :py:meth:`verticapy.machine_learning.vertica.naive_bayes.NaiveBayes.to_python`
        method is used to retrieve predictions,
        probabilities, or cluster distances. For specific details on how to
        use this method for different model types, refer to the relevant
        documentation for each model.
    """

    # Properties.

    @property
    def _vertica_fit_sql(self) -> Literal["NAIVE_BAYES"]:
        return "NAIVE_BAYES"

    @property
    def _vertica_predict_sql(self) -> Literal["PREDICT_NAIVE_BAYES"]:
        return "PREDICT_NAIVE_BAYES"

    @property
    def _model_subcategory(self) -> Literal["CLASSIFIER"]:
        return "CLASSIFIER"

    @property
    def _model_type(self) -> Literal["NaiveBayes"]:
        return "NaiveBayes"

    @property
    def _attributes(self) -> list[str]:
        return ["attributes_", "prior_", "classes_"]

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str = None,
        overwrite_model: bool = False,
        alpha: PythonNumber = 1.0,
        nbtype: Literal[
            "auto", "bernoulli", "categorical", "multinomial", "gaussian"
        ] = "auto",
    ) -> None:
        super().__init__(name, overwrite_model)
        self.parameters = {"alpha": alpha, "nbtype": str(nbtype).lower()}

    # Attributes Methods.

    def _compute_attributes(self) -> None:
        """
        Computes the model's attributes.
        """
        self.classes_ = self._array_to_int(
            np.array(self.get_vertica_attributes("prior")["class"])
        )
        self.prior_ = np.array(self.get_vertica_attributes("prior")["probability"])
        self.attributes_ = self._get_nb_attributes()

    def _get_nb_attributes(self) -> list[dict]:
        """
        Returns a list of dictionary for each of the NB
        variables. It is used to translate NB to Python.
        """
        try:
            vdf = vDataFrame(self.input_relation)
        except MissingRelation:
            return []
        var_info = {}
        gaussian_incr, bernoulli_incr, multinomial_incr = 0, 0, 0
        for idx, elem in enumerate(self.X):
            var_info[elem] = {"rank": idx}
            if vdf[elem].isbool():
                var_info[elem]["type"] = "bernoulli"
                for c in self.classes_:
                    var_info[elem][c] = self.get_vertica_attributes(f"bernoulli.{c}")[
                        "probability"
                    ][bernoulli_incr]
                bernoulli_incr += 1
            elif vdf[elem].category() == "int":
                var_info[elem]["type"] = "multinomial"
                for c in self.classes_:
                    multinomial = self.get_vertica_attributes(f"multinomial.{c}")
                    var_info[elem][c] = multinomial["probability"][multinomial_incr]
                multinomial_incr += 1
            elif vdf[elem].isnum():
                var_info[elem]["type"] = "gaussian"
                for c in self.classes_:
                    gaussian = self.get_vertica_attributes(f"gaussian.{c}")
                    var_info[elem][c] = {
                        "mu": gaussian["mu"][gaussian_incr],
                        "sigma_sq": gaussian["sigma_sq"][gaussian_incr],
                    }
                gaussian_incr += 1
            else:
                var_info[elem]["type"] = "categorical"
                my_cat = "categorical." + quote_ident(elem)[1:-1]
                attr = self.get_vertica_attributes()["attr_name"]
                for item in attr:
                    if item.lower() == my_cat.lower():
                        my_cat = item
                        break
                val = self.get_vertica_attributes(my_cat).values
                for item in val:
                    if item != "category":
                        if item not in var_info[elem]:
                            var_info[elem][item] = {}
                        for i, p in enumerate(val[item]):
                            var_info[elem][item][val["category"][i]] = p
        var_info_simplified = []
        for i in range(len(var_info)):
            for elem in var_info:
                if var_info[elem]["rank"] == i:
                    var_info_simplified += [var_info[elem]]
                    break
        for elem in var_info_simplified:
            del elem["rank"]
        return var_info_simplified

    # Parameters Methods.

    @staticmethod
    def _map_to_vertica_param_dict() -> dict:
        return {}

    # I/O Methods.

    def to_memmodel(self) -> mm.NaiveBayes:
        """
        Converts  the model to an InMemory object  that
        can be used for different types of predictions.
        """
        return mm.NaiveBayes(
            self.attributes_,
            self.prior_,
            self.classes_,
        )


class BernoulliNB(NaiveBayes):
    """NaiveBayes with parameter nbtype = 'bernoulli'"""

    def __init__(
        self, name: str = None, overwrite_model: bool = False, alpha: float = 1.0
    ) -> None:
        super().__init__(name, overwrite_model, alpha, nbtype="bernoulli")


class CategoricalNB(NaiveBayes):
    """NaiveBayes with parameter nbtype = 'categorical'"""

    def __init__(
        self, name: str = None, overwrite_model: bool = False, alpha: float = 1.0
    ) -> None:
        super().__init__(name, overwrite_model, alpha, nbtype="categorical")


class GaussianNB(NaiveBayes):
    """NaiveBayes with parameter nbtype = 'gaussian'"""

    def __init__(self, name: str = None, overwrite_model: bool = False) -> None:
        super().__init__(name, overwrite_model, nbtype="gaussian")


class MultinomialNB(NaiveBayes):
    """NaiveBayes with parameter nbtype = 'multinomial'"""

    def __init__(
        self, name: str = None, overwrite_model: bool = False, alpha: float = 1.0
    ) -> None:
        super().__init__(name, overwrite_model, alpha, nbtype="multinomial")
