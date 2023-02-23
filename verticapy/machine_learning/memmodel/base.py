"""
(c)  Copyright  [2018-2023]  OpenText  or one of its
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
import numpy as np

from verticapy._config.config import GRAPHVIZ_ON
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import clean_query
from verticapy.errors import ParameterError, FunctionError

from verticapy.machine_learning.memmodel.cluster import (
    bisecting_kmeans_to_graphviz,
    predict_from_bisecting_kmeans,
    predict_from_clusters,
    predict_from_clusters_kprotypes,
    sql_from_bisecting_kmeans,
    sql_from_clusters,
    sql_from_clusters_kprotypes,
)
from verticapy.machine_learning.memmodel.decomposition import (
    matrix_rotation,
    transform_from_pca,
    transform_from_svd,
    sql_from_pca,
    sql_from_svd,
)
from verticapy.machine_learning.memmodel.linear_model import (
    predict_from_coef,
    sql_from_coef,
)
from verticapy.machine_learning.memmodel.naive_bayes import predict_from_nb, sql_from_nb
from verticapy.machine_learning.memmodel.preprocessing import (
    sql_from_normalizer,
    sql_from_one_hot_encoder,
    transform_from_one_hot_encoder,
    transform_from_normalizer,
)
from verticapy.machine_learning.memmodel.tree import (
    binary_tree_to_graphviz,
    chaid_to_graphviz,
    predict_from_binary_tree,
    predict_from_chaid_tree,
    sql_from_binary_tree,
    sql_from_chaid_tree,
)

if GRAPHVIZ_ON:
    import graphviz


class memModel:
    """
Independent machine learning models that can easily be deployed 
using raw SQL or Python code.

Parameters
----------
model_type: str
    The model type, one of the following: 
    'BinaryTreeClassifier', 'BinaryTreeRegressor', 'BisectingKMeans',  
    'KMeans', 'KPrototypes', 'LinearSVC', 'LinearSVR', 'LinearRegression', 
    'LogisticRegression', 'NaiveBayes', 'NearestCentroid', 'Normalizer', 
    'OneHotEncoder', 'PCA', 'RandomForestClassifier', 'RandomForestRegressor', 
    'SVD', 'XGBoostClassifier', 'XGBoostRegressor'.
attributes: dict
    Dictionary which includes all the model's attributes.
        For BisectingKMeans: 
            {"clusters": List of the model's cluster centers.
             "left_child": List of the model's left children IDs.
             "right_child": List of the model's right children IDs.
             "p": The p corresponding to the one of the p-distances.}
        For BinaryTreeClassifier, BinaryTreeRegressor, BinaryTreeAnomaly:
            {children_left:  A list of node IDs, where children_left[i] is 
                             the node id of the left child of node i.
             children_right: A list of node IDs, where children_right[i] is 
                             the node id of the right child of node i.
             feature: A list of features, where feature[i] is the feature to 
                      split on, for the internal node i.
             threshold: threshold[i] is the threshold for the internal node i.
             value: Contains the constant prediction value of each node.
             classes: [Only for Classifier] 
                      The classes for the binary tree model.}
             psy: [Only for Anomaly Detection] 
                  Sampling size used to compute the the Isolation Forest Score.
        For CHAID:         
            {"tree": CHAID tree. This tree can be generated using the 
                     vDataFrame.chaid method.
             "classes": The classes for the CHAID model.}
        For KMeans:        
            {"clusters": List of the model's cluster centers.
             "p": The p corresponding to the one of the p-distances.}
        For KPrototypes:   
            {"clusters": List of the model's cluster centers.
             "p": The p corresponding to one of the p-distances.
             "gamma": Weighting factor for categorical columns.
             "is_categorical": List of booleans to indicate whether 
                               X[idx] is a categorical variable or not.}
        For LinearSVC, LinearSVR, LinearSVC, 
            LinearRegression, LogisticRegression: 
            {"coefficients": List of the model's coefficients.
             "intercept": Intercept or constant value.}
        For NaiveBayes:     
            {classes: The classes for the naive bayes model.
             prior: The model probabilities of each class.
             attributes: 
                List of the model's attributes. Each feature is represented 
                by a dictionary, the contents of which differs for each 
                distribution type.
                For 'gaussian':
                  Key 'type' must have the value 'gaussian'.
                  Each of the model's classes must include a dictionary with 
                  two keys:
                    sigma_sq: Square root of the standard deviation.
                    mu: Average.
                  Example: {'type': 'gaussian', 
                            'C': {'mu': 63.9878308300395, 
                                  'sigma_sq': 7281.87598377196}, 
                            'Q': {'mu': 13.0217386792453, 
                                  'sigma_sq': 211.626862330204}, 
                            'S': {'mu': 27.6928120412844, 
                                  'sigma_sq': 1428.57067393938}}
                For 'multinomial':
                  Key 'type' must have the value 'multinomial'.
                  Each of the model's classes must be represented by a key with its 
                  probability as the value.
                  Example: {'type': 'multinomial', 
                            'C': 0.771666666666667, 
                            'Q': 0.910714285714286, 
                            'S': 0.878216123499142}
                For 'bernoulli':
                  Key 'type' must have the value 'bernoulli'.
                  Each of the model's classes must be represented by a key with its 
                  probability as the value.
                  Example: {'type': 'bernoulli', 
                            'C': 0.537254901960784, 
                            'Q': 0.277777777777778, 
                            'S': 0.324942791762014}
                For 'categorical':
                  Key 'type' must have the value 'categorical'.
                  Each of the model's classes must include a dictionary with all 
                  the feature categories.
                  Example: {'type': 'categorical', 
                            'C': {'female': 0.407843137254902, 
                                  'male': 0.592156862745098}, 
                            'Q': {'female': 0.416666666666667, 
                                  'male': 0.583333333333333}, 
                            'S': {'female': 0.311212814645309, 
                                  'male': 0.688787185354691}}}
        For NearestCentroid:
            {"clusters": List of the model's cluster centers.
             "p": The p corresponding to the one of the p-distances.
             "classes": Represents the classes of the nearest centroids.}
        For Normalizer:    
            {"values": List of tuples including the model's attributes.
                The required tuple depends on the specified method: 
                    'zscore': (mean, std)
                    'robust_zscore': (median, mad)
                    'minmax': (min, max)
             "method": The model's category, one of the following: 'zscore', 
                      'robust_zscore', or 'minmax'.}
        For OneHotEncoder: 
            {"categories": List of the different feature categories.
             "drop_first": Boolean, whether the first category
                           should be dropped.
             "column_naming": Appends categorical levels to column names 
                             according to the specified method. 
                             It can be set to 'indices' or 'values'.}
        For PCA:           
            {"principal_components": Matrix of the principal components.
             "mean": List of the input predictors average.}
        For RandomForestClassifier, RandomForestRegressor, 
            XGBoostClassifier, XGBoostRegressor, IsolationForest:
            {trees: list of memModels of type 'BinaryTreeRegressor' or 
                    'BinaryTreeClassifier' or 'BinaryTreeAnomaly'
             learning_rate: [Only for XGBoostClassifier 
                                  and XGBoostRegressor]
                            Learning rate.
             mean: [Only for XGBoostRegressor]
                   Average of the response column.
             logodds: [Only for XGBoostClassifier]
                   List of the logodds of the response classes.}
        For SVD:           
            {"vectors": Matrix of the right singular vectors.
             "values": List of the singular values.}
    """

    #
    # Special Methods
    #

    @save_verticapy_logs
    def __init__(
        self,
        model_type: Literal[
            "OneHotEncoder",
            "Normalizer",
            "SVD",
            "PCA",
            "CHAID",
            "BisectingKMeans",
            "KMeans",
            "KPrototypes",
            "NaiveBayes",
            "XGBoostClassifier",
            "XGBoostRegressor",
            "RandomForestClassifier",
            "BinaryTreeClassifier",
            "BinaryTreeRegressor",
            "BinaryTreeAnomaly",
            "RandomForestRegressor",
            "LinearSVR",
            "LinearSVC",
            "LogisticRegression",
            "LinearRegression",
            "NearestCentroid",
            "IsolationForest",
        ],
        attributes: dict,
    ):
        attributes_ = {}
        if model_type == "NaiveBayes":
            if (
                "attributes" not in attributes
                or "prior" not in attributes
                or "classes" not in attributes
            ):
                raise ParameterError(
                    f"{model_type}'s attributes must include at least the following "
                    "lists: attributes, prior, classes."
                )
            attributes_["prior"] = np.copy(attributes["prior"])
            attributes_["classes"] = np.copy(attributes["classes"])
            attributes_["attributes"] = []
            for att in attributes["attributes"]:
                assert isinstance(att, dict), ParameterError(
                    "All the elements of the 'attributes' key must be dictionaries."
                )
                assert "type" in att and att["type"] in (
                    "categorical",
                    "bernoulli",
                    "multinomial",
                    "gaussian",
                ), ParameterError(
                    "All the elements of the 'attributes' key must be dictionaries "
                    "including a 'type' key with a value in (categorical, bernoulli,"
                    " multinomial, gaussian)."
                )
                attributes_["attributes"] += [att.copy()]
        elif model_type in (
            "RandomForestRegressor",
            "XGBoostRegressor",
            "RandomForestClassifier",
            "XGBoostClassifier",
            "IsolationForest",
        ):
            if "trees" not in attributes:
                raise ParameterError(
                    f"{model_type}'s attributes must include a list of memModels "
                    "representing each tree."
                )
            attributes_["trees"] = []
            for tree in attributes["trees"]:
                assert isinstance(tree, memModel), ParameterError(
                    f"Each tree of the model must be a memModel, found '{tree}'."
                )
                if model_type in ("RandomForestClassifier", "XGBoostClassifier"):
                    assert tree.model_type_ == "BinaryTreeClassifier", ParameterError(
                        "Each tree of the model must be a BinaryTreeClassifier"
                        f", found '{tree.model_type_}'."
                    )
                elif model_type == "IsolationForest":
                    assert tree.model_type_ == "BinaryTreeAnomaly", ParameterError(
                        "Each tree of the model must be a BinaryTreeAnomaly"
                        f", found '{tree.model_type_}'."
                    )
                else:
                    assert tree.model_type_ == "BinaryTreeRegressor", ParameterError(
                        "Each tree of the model must be a BinaryTreeRegressor"
                        f", found '{tree.model_type_}'."
                    )
                attributes_["trees"] += [tree]
            if model_type == "XGBoostRegressor":
                if "learning_rate" not in attributes or "mean" not in attributes:
                    raise ParameterError(
                        f"{model_type}'s attributes must include the response "
                        "average and the learning rate."
                    )
                attributes_["mean"] = float(attributes["mean"])
            if model_type == "XGBoostClassifier":
                if "learning_rate" not in attributes or "logodds" not in attributes:
                    raise ParameterError(
                        f"{model_type}'s attributes must include the response "
                        "classes logodds and the learning rate."
                    )
                attributes_["logodds"] = np.copy(attributes["logodds"])
            if model_type in ("XGBoostRegressor", "XGBoostClassifier"):
                attributes_["learning_rate"] = float(attributes["learning_rate"])
        elif model_type in (
            "BinaryTreeClassifier",
            "BinaryTreeRegressor",
            "BinaryTreeAnomaly",
        ):
            if (
                "children_left" not in attributes
                or "children_right" not in attributes
                or "feature" not in attributes
                or "threshold" not in attributes
                or "value" not in attributes
            ):
                raise ParameterError(
                    f"{model_type}'s attributes must include at least the following "
                    "lists: children_left, children_right, feature, threshold, value."
                )
            for elem in (
                "children_left",
                "children_right",
                "feature",
                "threshold",
                "value",
            ):
                if isinstance(attributes[elem], list):
                    attributes_[elem] = attributes[elem].copy()
                else:
                    attributes_[elem] = np.copy(attributes[elem])
            if model_type == "BinaryTreeClassifier":
                if "classes" not in attributes:
                    attributes_["classes"] = []
                else:
                    attributes_["classes"] = np.copy(attributes["classes"])
            if model_type == "BinaryTreeAnomaly":
                assert "psy" in attributes, ParameterError(
                    "BinaryTreeAnomaly's must include the sampling size 'psy'."
                )
                attributes_["psy"] = int(attributes["psy"])
        elif model_type == "CHAID":
            assert "tree" in attributes, ParameterError(
                f"{model_type}'s attributes must include at least the CHAID tree."
            )
            attributes_["tree"] = dict(attributes["tree"])
            if "classes" not in attributes:
                attributes_["classes"] = []
            else:
                attributes_["classes"] = np.copy(attributes["classes"])
        elif model_type == "OneHotEncoder":
            assert "categories" in attributes, ParameterError(
                "OneHotEncoder's attributes must include a list with all "
                "the feature categories for the 'categories' parameter."
            )
            attributes_["categories"] = attributes["categories"].copy()
            if "drop_first" not in attributes:
                attributes_["drop_first"] = False
            else:
                attributes_["drop_first"] = bool(attributes["drop_first"])
            if "column_naming" not in attributes:
                attributes_["column_naming"] = "indices"
            elif not (attributes["column_naming"]):
                attributes_["column_naming"] = None
            else:
                if attributes["column_naming"] not in ["indices", "values"]:
                    raise ValueError(
                        f"Attribute 'column_naming' must be in <{' | '.join(attributes['column_naming'])}>"
                    )
                attributes_["column_naming"] = attributes["column_naming"]
        elif model_type in (
            "LinearSVR",
            "LinearSVC",
            "LogisticRegression",
            "LinearRegression",
        ):
            if "coefficients" not in attributes or "intercept" not in attributes:
                raise ParameterError(
                    f"{model_type}'s attributes must include a list with the 'coefficients' and the 'intercept' value."
                )
            attributes_["coefficients"] = np.copy(attributes["coefficients"])
            attributes_["intercept"] = float(attributes["intercept"])
        elif model_type == "BisectingKMeans":
            if (
                "clusters" not in attributes
                or "left_child" not in attributes
                or "right_child" not in attributes
            ):
                raise ParameterError(
                    "BisectingKMeans's attributes must include three lists: one with "
                    "all the 'clusters' centers, one with all the cluster's right "
                    "children, and one with all the cluster's left children."
                )
            attributes_["clusters"] = np.copy(attributes["clusters"])
            attributes_["left_child"] = np.copy(attributes["left_child"])
            attributes_["right_child"] = np.copy(attributes["right_child"])
            if "p" not in attributes:
                attributes_["p"] = 2
            else:
                attributes_["p"] = int(attributes["p"])
            if "cluster_size" not in attributes:
                attributes_["cluster_size"] = []
            else:
                attributes_["cluster_size"] = np.copy(attributes["cluster_size"])
            if "cluster_score" not in attributes:
                attributes_["cluster_score"] = []
            else:
                attributes_["cluster_score"] = np.copy(attributes["cluster_score"])
        elif model_type in ("KMeans", "NearestCentroid", "KPrototypes"):
            if "clusters" not in attributes:
                raise ParameterError(
                    f"{model_type}'s attributes must include a list with all the 'clusters' centers."
                )
            attributes_["clusters"] = np.copy(attributes["clusters"])
            if "p" not in attributes:
                attributes_["p"] = 2
            else:
                attributes_["p"] = int(attributes["p"])
            if model_type == "KPrototypes":
                if "gamma" not in attributes:
                    attributes_["gamma"] = 1.0
                else:
                    attributes_["gamma"] = attributes["gamma"]
                if "is_categorical" not in attributes:
                    attributes_["is_categorical"] = []
                else:
                    attributes_["is_categorical"] = attributes["is_categorical"]
            if model_type == "NearestCentroid":
                if "classes" not in attributes:
                    attributes_["classes"] = None
                else:
                    attributes_["classes"] = [c for c in attributes["classes"]]
        elif model_type == "PCA":
            if "principal_components" not in attributes or "mean" not in attributes:
                raise ParameterError(
                    "PCA's attributes must include two lists: one with all the principal "
                    "components and one with all the averages of each input feature."
                )
            attributes_["principal_components"] = np.copy(
                attributes["principal_components"]
            )
            attributes_["mean"] = np.copy(attributes["mean"])
        elif model_type == "SVD":
            if "vectors" not in attributes or "values" not in attributes:
                raise ParameterError(
                    "SVD's attributes must include 2 lists: one with all the right singular "
                    "vectors and one with the singular values of each input feature."
                )
            attributes_["vectors"] = np.copy(attributes["vectors"])
            attributes_["values"] = np.copy(attributes["values"])
        elif model_type == "Normalizer":
            assert "values" in attributes and "method" in attributes, ParameterError(
                "Normalizer's attributes must include a list including the model's "
                "aggregations and a string representing the model's method."
            )
            if attributes["method"] not in ["minmax", "zscore", "robust_zscore"]:
                raise ValueError(
                    f"Attribute 'method' must be in <{' | '.join(attributes['method'])}>"
                )
            attributes_["values"] = np.copy(attributes["values"])
            attributes_["method"] = attributes["method"]
        else:
            raise ParameterError(f"Model type '{model_type}' is not yet available.")
        self.attributes_ = attributes_
        self.model_type_ = model_type
        self.represent_ = f"<{model_type}>\n\nattributes = {attributes_}"

    def __repr__(self):
        return self.represent_

    #
    # Methods
    #

    def get_attributes(self) -> dict:
        """
    Returns model's attributes.
        """
        return self.attributes_

    def set_attributes(self, attributes: dict):
        """
    Sets new model's attributes.

    Parameters
    ----------
    attributes: dict
        New attributes. See method '__init__' for more information.
        """
        attributes_tmp = {}
        for elem in self.attributes_:
            attributes_tmp[elem] = self.attributes_[elem]
        for elem in attributes:
            attributes_tmp[elem] = attributes[elem]
        self.__init__(model_type=self.model_type_, attributes=attributes_tmp)

    def plot_tree(
        self,
        pic_path: str = "",
        tree_id: int = 0,
        feature_names: Union[list, np.ndarray] = [],
        classes_color: list = [],
        round_pred: int = 2,
        percent: bool = False,
        vertical: bool = True,
        node_style: dict = {},
        arrow_style: dict = {},
        leaf_style: dict = {},
    ):
        """
        Draws the input tree. Requires the graphviz module.

        Parameters
        ----------
        pic_path: str, optional
            Absolute path to save the image of the tree.
        tree_id: int, optional
            Unique tree identifier, an integer in the range [0, n_estimators - 1].
        feature_names: list / numpy.array, optional
            List of the names of each feature.
        classes_color: list, optional
            Colors that represent the different classes.
        round_pred: int, optional
            The number of decimals to round the prediction to. 0 rounds to an integer.
        percent: bool, optional
            If set to True, the probabilities are returned as percents.
        vertical: bool, optional
            If set to True, the function generates a vertical tree.
        node_style: dict, optional
            Dictionary of options to customize each node of the tree. For a list of options, see
            the Graphviz API: https://graphviz.org/doc/info/attrs.html
        arrow_style: dict, optional
            Dictionary of options to customize each arrow of the tree. For a list of options, see
            the Graphviz API: https://graphviz.org/doc/info/attrs.html
        leaf_style: dict, optional
            Dictionary of options to customize each leaf of the tree. For a list of options, see
            the Graphviz API: https://graphviz.org/doc/info/attrs.html

        Returns
        -------
        graphviz.Source
            graphviz object.
        """
        if not (GRAPHVIZ_ON):
            raise ImportError(
                "The graphviz module doesn't seem to be installed in your environment.\n"
                "To be able to use this method, you'll have to install it.\n"
                "[Tips] Run: 'pip3 install graphviz' in your terminal to install the module."
            )
        graphviz_str = self.to_graphviz(
            tree_id=tree_id,
            feature_names=feature_names,
            classes_color=classes_color,
            round_pred=round_pred,
            percent=percent,
            vertical=vertical,
            node_style=node_style,
            arrow_style=arrow_style,
            leaf_style=leaf_style,
        )
        res = graphviz.Source(graphviz_str)
        if pic_path:
            res.view(pic_path)
        return res

    def predict(self, X: list) -> np.ndarray:
        """
    Predicts using the model's attributes.

    Parameters
    ----------
    X: list / numpy.array
        data.

    Returns
    -------
    numpy.array
        Predicted values
        """
        if self.model_type_ in (
            "LinearRegression",
            "LinearSVC",
            "LinearSVR",
            "LogisticRegression",
        ):
            return predict_from_coef(
                X,
                self.attributes_["coefficients"],
                self.attributes_["intercept"],
                self.model_type_,
            )
        elif self.model_type_ == "NaiveBayes":
            return predict_from_nb(
                X,
                self.attributes_["attributes"],
                classes=self.attributes_["classes"],
                prior=self.attributes_["prior"],
                return_proba=False,
            )
        elif self.model_type_ == "KMeans":
            return predict_from_clusters(
                X, self.attributes_["clusters"], p=self.attributes_["p"]
            )
        elif self.model_type_ == "KPrototypes":
            return predict_from_clusters_kprotypes(
                X,
                self.attributes_["clusters"],
                p=self.attributes_["p"],
                gamma=self.attributes_["gamma"],
            )
        elif self.model_type_ == "NearestCentroid":
            return predict_from_clusters(
                X,
                self.attributes_["clusters"],
                p=self.attributes_["p"],
                classes=self.attributes_["classes"],
            )
        elif self.model_type_ == "BisectingKMeans":
            return predict_from_bisecting_kmeans(
                X,
                self.attributes_["clusters"],
                self.attributes_["left_child"],
                self.attributes_["right_child"],
                p=self.attributes_["p"],
            )
        elif self.model_type_ in (
            "BinaryTreeRegressor",
            "BinaryTreeClassifier",
            "BinaryTreeAnomaly",
        ):
            return predict_from_binary_tree(
                X,
                self.attributes_["children_left"],
                self.attributes_["children_right"],
                self.attributes_["feature"],
                self.attributes_["threshold"],
                self.attributes_["value"],
                self.attributes_["classes"]
                if self.model_type_ == "BinaryTreeClassifier"
                else [],
                is_regressor=(self.model_type_ == "BinaryTreeRegressor"),
                is_anomaly=(self.model_type_ == "BinaryTreeAnomaly"),
                psy=self.attributes_["psy"]
                if (self.model_type_ == "BinaryTreeAnomaly")
                else -1,
            )
        elif self.model_type_ in (
            "RandomForestRegressor",
            "XGBoostRegressor",
            "IsolationForest",
        ):
            result = [tree.predict(X) for tree in self.attributes_["trees"]]
            if self.model_type_ in ("RandomForestRegressor", "IsolationForest"):
                res = np.average(np.column_stack(result), axis=1)
                if self.model_type_ == "IsolationForest":
                    res = 2 ** (-res)
                return res
            else:
                return (
                    np.sum(np.column_stack(result), axis=1)
                    * self.attributes_["learning_rate"]
                    + self.attributes_["mean"]
                )
        elif self.model_type_ in ("RandomForestClassifier", "XGBoostClassifier"):
            result = np.argmax(self.predict_proba(X), axis=1)
            result = np.array(
                [self.attributes_["trees"][0].attributes_["classes"][i] for i in result]
            )
            return result
        elif self.model_type_ == "CHAID":
            return predict_from_chaid_tree(
                X, self.attributes_["tree"], self.attributes_["classes"], False
            )
        else:
            raise FunctionError(
                f"Method 'predict' is not available for model type '{self.model_type_}'."
            )

    def predict_sql(self, X: list) -> Union[list, str]:
        """
    Returns the SQL code needed to deploy the model.

    Parameters
    ----------
    X: list
        Names or values of the input predictors.

    Returns
    -------
    str
        SQL code
        """
        if self.model_type_ in (
            "LinearRegression",
            "LinearSVC",
            "LinearSVR",
            "LogisticRegression",
        ):
            result = sql_from_coef(
                X,
                self.attributes_["coefficients"],
                self.attributes_["intercept"],
                self.model_type_,
            )
            if self.model_type_ in ("LinearSVC", "LogisticRegression"):
                result = f"(({result}) > 0.5)::int"
        elif self.model_type_ == "KMeans":
            result = sql_from_clusters(
                X, self.attributes_["clusters"], p=self.attributes_["p"]
            )
        elif self.model_type_ == "KPrototypes":
            result = sql_from_clusters_kprotypes(
                X,
                self.attributes_["clusters"],
                p=self.attributes_["p"],
                gamma=self.attributes_["gamma"],
                is_categorical=self.attributes_["is_categorical"],
            )
        elif self.model_type_ == "NearestCentroid":
            result = sql_from_clusters(
                X,
                self.attributes_["clusters"],
                p=self.attributes_["p"],
                classes=self.attributes_["classes"],
            )
        elif self.model_type_ == "BisectingKMeans":
            result = sql_from_bisecting_kmeans(
                X,
                self.attributes_["clusters"],
                self.attributes_["left_child"],
                self.attributes_["right_child"],
                p=self.attributes_["p"],
            )
        elif self.model_type_ in (
            "BinaryTreeRegressor",
            "BinaryTreeClassifier",
            "BinaryTreeAnomaly",
        ):
            result = sql_from_binary_tree(
                X,
                self.attributes_["children_left"],
                self.attributes_["children_right"],
                self.attributes_["feature"],
                self.attributes_["threshold"],
                self.attributes_["value"],
                self.attributes_["classes"]
                if (self.model_type_ == "BinaryTreeClassifier")
                else [],
                is_regressor=(self.model_type_ == "BinaryTreeRegressor"),
                is_anomaly=(self.model_type_ == "BinaryTreeAnomaly"),
                psy=self.attributes_["psy"]
                if (self.model_type_ == "BinaryTreeAnomaly")
                else -1,
            )
        elif self.model_type_ in (
            "RandomForestRegressor",
            "XGBoostRegressor",
            "IsolationForest",
        ):
            result = [str(tree.predict_sql(X)) for tree in self.attributes_["trees"]]
            if self.model_type_ in ("RandomForestRegressor", "IsolationForest"):
                result = f"({' + '.join(result)}) / {len(result)}"
                if self.model_type_ == "IsolationForest":
                    result = f"POWER(2, - ({result}))"
            else:
                result = f"({' + '.join(result)}) * {self.attributes_['learning_rate']}"
                result += f" + {self.attributes_['mean']}"
        elif self.model_type_ in (
            "RandomForestClassifier",
            "XGBoostClassifier",
            "NaiveBayes",
        ):
            if self.model_type_ == "NaiveBayes":
                classes = self.attributes_["classes"]
                result_proba = sql_from_nb(
                    X,
                    self.attributes_["attributes"],
                    classes=self.attributes_["classes"],
                    prior=self.attributes_["prior"],
                )
            else:
                classes = self.attributes_["trees"][0].attributes_["classes"]
                result_proba = self.predict_proba_sql(X)
            m = len(classes)
            if m == 2:
                result = f"""
                    (CASE 
                        WHEN {result_proba[1]} > 0.5 
                            THEN {classes[1]} 
                        ELSE {classes[0]} 
                    END)"""
            else:
                sql = []
                for i in range(m):
                    list_tmp = []
                    for j in range(i):
                        list_tmp += [f"{result_proba[i]} >= {result_proba[j]}"]
                    sql += [" AND ".join(list_tmp)]
                sql = sql[1:]
                sql.reverse()
                result = f"""
                    CASE 
                        WHEN {' OR '.join([f"{x} IS NULL" for x in X])} 
                        THEN NULL"""
                for i in range(m - 1):
                    class_i = classes[m - i - 1]
                    if isinstance(class_i, str):
                        class_i_str = f"'{class_i}'"
                    else:
                        class_i_str = class_i
                    result += f" WHEN {sql[i]} THEN {class_i_str}"
                if isinstance(classes[0], str):
                    classes_0 = f"'{classes[0]}'"
                else:
                    classes_0 = classes[0]
                result += f" ELSE {classes_0} END"
        elif self.model_type_ == "CHAID":
            return sql_from_chaid_tree(
                X, self.attributes_["tree"], self.attributes_["classes"], False
            )
        else:
            raise FunctionError(
                f"Method 'predict_sql' is not available for model type '{self.model_type_}'"
            )
        if isinstance(result, str):
            result = clean_query(result.replace("\xa0", " "))
        return result

    def predict_proba(self, X: list) -> np.ndarray:
        """
    Predicts probabilities using the model's attributes.

    Parameters
    ----------
    X: list / numpy.array
        data.

    Returns
    -------
    numpy.array
        Predicted values
        """
        if self.model_type_ in ("LinearSVC", "LogisticRegression"):
            return predict_from_coef(
                X,
                self.attributes_["coefficients"],
                self.attributes_["intercept"],
                self.model_type_,
                return_proba=True,
            )
        elif self.model_type_ == "NaiveBayes":
            return predict_from_nb(
                X,
                self.attributes_["attributes"],
                classes=self.attributes_["classes"],
                prior=self.attributes_["prior"],
                return_proba=True,
            )
        elif self.model_type_ == "KMeans":
            return predict_from_clusters(
                X,
                self.attributes_["clusters"],
                p=self.attributes_["p"],
                return_proba=True,
            )
        elif self.model_type_ == "KPrototypes":
            return predict_from_clusters_kprotypes(
                X,
                self.attributes_["clusters"],
                p=self.attributes_["p"],
                gamma=self.attributes_["gamma"],
                return_proba=True,
            )
        elif self.model_type_ == "NearestCentroid":
            return predict_from_clusters(
                X,
                self.attributes_["clusters"],
                p=self.attributes_["p"],
                classes=self.attributes_["classes"],
                return_proba=True,
            )
        elif self.model_type_ == "BinaryTreeClassifier":
            return predict_from_binary_tree(
                X,
                self.attributes_["children_left"],
                self.attributes_["children_right"],
                self.attributes_["feature"],
                self.attributes_["threshold"],
                self.attributes_["value"],
                self.attributes_["classes"],
                True,
                is_regressor=False,
            )
        elif self.model_type_ == "RandomForestClassifier":
            result, n = 0, len(self.attributes_["trees"])
            for i in range(n):
                result_tmp = self.attributes_["trees"][i].predict_proba(X)
                result_tmp_arg = np.zeros_like(result_tmp)
                result_tmp_arg[np.arange(len(result_tmp)), result_tmp.argmax(1)] = 1
                result += result_tmp_arg
            return result / n
        elif self.model_type_ == "XGBoostClassifier":
            result = 0
            for tree in self.attributes_["trees"]:
                result += tree.predict_proba(X)
            result = (
                self.attributes_["logodds"] + self.attributes_["learning_rate"] * result
            )
            result = 1 / (1 + np.exp(-result))
            result /= np.sum(result, axis=1)[:, None]
            return result
        elif self.model_type_ == "CHAID":
            return predict_from_chaid_tree(
                X, self.attributes_["tree"], self.attributes_["classes"], True
            )
        else:
            raise FunctionError(
                "Method 'predict_proba' is not available "
                f"for model type '{self.model_type_}'."
            )

    def predict_proba_sql(self, X: list) -> list:
        """
    Returns the SQL code needed to deploy the probabilities model.

    Parameters
    ----------
    X: list
        Names or values of the input predictors.

    Returns
    -------
    str
        SQL code
        """
        if self.model_type_ in ("LinearSVC", "LogisticRegression"):
            result = sql_from_coef(
                X,
                self.attributes_["coefficients"],
                self.attributes_["intercept"],
                self.model_type_,
            )
            result = [f"1 - ({result})", result]
        elif self.model_type_ == "NaiveBayes":
            result = sql_from_nb(
                X,
                self.attributes_["attributes"],
                classes=self.attributes_["classes"],
                prior=self.attributes_["prior"],
            )
            div = "(" + " + ".join(result) + ")"
            for idx in range(len(result)):
                result[idx] = "(" + result[idx] + ") / " + div
            result = result
        elif self.model_type_ == "KMeans":
            result = sql_from_clusters(
                X,
                self.attributes_["clusters"],
                p=self.attributes_["p"],
                return_proba=True,
            )
        elif self.model_type_ == "KPrototypes":
            result = sql_from_clusters_kprotypes(
                X,
                self.attributes_["clusters"],
                p=self.attributes_["p"],
                gamma=self.attributes_["gamma"],
                is_categorical=self.attributes_["is_categorical"],
                return_proba=True,
            )
        elif self.model_type_ == "NearestCentroid":
            result = sql_from_clusters(
                X,
                self.attributes_["clusters"],
                p=self.attributes_["p"],
                classes=self.attributes_["classes"],
                return_proba=True,
            )
        elif self.model_type_ == "BinaryTreeClassifier":
            result = sql_from_binary_tree(
                X,
                self.attributes_["children_left"],
                self.attributes_["children_right"],
                self.attributes_["feature"],
                self.attributes_["threshold"],
                self.attributes_["value"],
                self.attributes_["classes"],
                True,
                is_regressor=False,
            )
        elif self.model_type_ == "RandomForestClassifier":
            trees, n, m = (
                [],
                len(self.attributes_["trees"]),
                len(self.attributes_["trees"][0].attributes_["classes"]),
            )
            for i in range(n):
                val = []
                for elem in self.attributes_["trees"][i].attributes_["value"]:
                    if isinstance(elem, type(None)):
                        val += [elem]
                    else:
                        value_tmp = np.zeros_like([elem])
                        value_tmp[np.arange(1), np.array([elem]).argmax(1)] = 1
                        val += [list(value_tmp[0])]
                tree = memModel(
                    "BinaryTreeClassifier",
                    {
                        "children_left": self.attributes_["trees"][i].attributes_[
                            "children_left"
                        ],
                        "children_right": self.attributes_["trees"][i].attributes_[
                            "children_right"
                        ],
                        "feature": self.attributes_["trees"][i].attributes_["feature"],
                        "threshold": self.attributes_["trees"][i].attributes_[
                            "threshold"
                        ],
                        "value": val,
                        "classes": self.attributes_["trees"][i].attributes_["classes"],
                    },
                )
                trees += [tree]
            result = [trees[i].predict_proba_sql(X) for i in range(n)]
            classes_proba = []
            for i in range(m):
                classes_proba += [f"({' + '.join([val[i] for val in result])}) / {n}"]
            result = classes_proba
        elif self.model_type_ == "XGBoostClassifier":
            result, n, m = (
                [],
                len(self.attributes_["trees"]),
                len(self.attributes_["trees"][0].attributes_["classes"]),
            )
            all_probas = [
                self.attributes_["trees"][i].predict_proba_sql(X) for i in range(n)
            ]
            for i in range(m):
                result += [
                    f"""(1 / (1 + EXP(- ({ self.attributes_['logodds'][i]} 
                                + {self.attributes_['learning_rate']} 
                                * ({' + '.join([p[i] for p in all_probas])})))))"""
                ]
            sum_result = f"({' + '.join(result)})"
            result = [clean_query(f"{x} / {sum_result}") for x in result]
        elif self.model_type_ == "CHAID":
            return sql_from_chaid_tree(
                X, self.attributes_["tree"], self.attributes_["classes"], True
            )
        else:
            raise FunctionError(
                "Method 'predict_proba_sql' is not available "
                f"for model type '{self.model_type_}'."
            )
        return [r.replace("\xa0", " ") for r in result]

    def to_graphviz(
        self,
        tree_id: int = 0,
        feature_names: Union[list, np.ndarray] = [],
        classes_color: list = [],
        round_pred: int = 2,
        percent: bool = False,
        vertical: bool = True,
        node_style: dict = {},
        arrow_style: dict = {},
        leaf_style: dict = {},
    ):
        """
        Returns the code for a Graphviz tree.

        Parameters
        ----------
        tree_id: int, optional
            Unique tree identifier, an integer in the range [0, n_estimators - 1].
        feature_names: list / numpy.array, optional
            List of the names of each feature.
        classes_color: list, optional
            Colors that represent the different classes.
        round_pred: int, optional
            The number of decimals to which to round the prediction/score. 0 rounds to an integer.
        percent: bool, optional
            If set to True, the probabilities/scores are returned as a percent.
        vertical: bool, optional
            If set to True, the function generates a vertical tree.
        node_style: dict, optional
            Dictionary of options to customize each node of the tree. For a list of options, see
            the Graphviz API: https://graphviz.org/doc/info/attrs.html
        arrow_style: dict, optional
            Dictionary of options to customize each arrow of the tree. For a list of options, see
            the Graphviz API: https://graphviz.org/doc/info/attrs.html
        leaf_style: dict, optional
            Dictionary of options to customize each leaf of the tree. For a list of options, see
            the Graphviz API: https://graphviz.org/doc/info/attrs.html

        Returns
        -------
        str
            Graphviz code.
        """
        if len(node_style) == 0 and self.model_type_ != "BisectingKMeans":
            node_style = {"shape": "box", "style": "filled"}
        else:
            node_style = {"shape": "none"}
        classes = self.attributes_["classes"] if "classes" in self.attributes_ else []
        if self.model_type_ in (
            "BinaryTreeRegressor",
            "BinaryTreeClassifier",
            "BinaryTreeAnomaly",
        ):
            prefix_pred = "prob"
            for elem in self.attributes_["value"]:
                if isinstance(elem, list) and not (0.99 < sum(elem) <= 1.0):
                    prefix_pred = "logodds"
                    break
                elif (
                    isinstance(elem, list)
                    and len(elem) == 2
                    and isinstance(elem[0], int)
                    and isinstance(elem[1], int)
                ):
                    prefix_pred = "contamination"
                    break
            return binary_tree_to_graphviz(
                children_left=self.attributes_["children_left"],
                children_right=self.attributes_["children_right"],
                feature=self.attributes_["feature"],
                threshold=self.attributes_["threshold"],
                value=self.attributes_["value"],
                feature_names=feature_names,
                classes=classes,
                classes_color=classes_color,
                prefix_pred=prefix_pred,
                round_pred=round_pred,
                percent=percent,
                vertical=vertical,
                node_style=node_style,
                arrow_style=arrow_style,
                leaf_style=leaf_style,
                psy=self.attributes_["psy"]
                if (self.model_type_ == "BinaryTreeAnomaly")
                else -1,
            )
        elif self.model_type_ == "BisectingKMeans":
            cluster_size = (
                self.attributes_["cluster_size"]
                if "cluster_size" in self.attributes_
                else []
            )
            cluster_score = (
                self.attributes_["cluster_score"]
                if "cluster_score" in self.attributes_
                else []
            )
            return bisecting_kmeans_to_graphviz(
                children_left=self.attributes_["left_child"],
                children_right=self.attributes_["right_child"],
                cluster_size=cluster_size,
                cluster_score=cluster_score,
                round_score=round_pred,
                percent=percent,
                vertical=vertical,
                node_style=node_style,
                arrow_style=arrow_style,
                leaf_style=leaf_style,
            )
        elif self.model_type_ == "CHAID":
            return chaid_to_graphviz(
                tree=self.attributes_["tree"],
                classes=classes,
                classes_color=classes_color,
                round_pred=round_pred,
                percent=percent,
                vertical=vertical,
                node_style=node_style,
                arrow_style=arrow_style,
                leaf_style=leaf_style,
            )
        elif self.model_type_ in (
            "RandomForestClassifier",
            "XGBoostClassifier",
            "RandomForestRegressor",
            "XGBoostRegressor",
            "IsolationForest",
        ):
            return self.attributes_["trees"][tree_id].to_graphviz(
                feature_names=feature_names,
                classes_color=classes_color,
                round_pred=round_pred,
                percent=percent,
                vertical=vertical,
                node_style=node_style,
                arrow_style=arrow_style,
                leaf_style=leaf_style,
            )
        else:
            raise FunctionError(
                f"Method 'to_graphviz' does not exist for model type '{self.model_type_}'."
            )

    def transform(self, X: list) -> np.ndarray:
        """
    Transforms the data using the model's attributes.

    Parameters
    ----------
    X: list / numpy.array
        Data to transform.

    Returns
    -------
    numpy.array
        Transformed data
        """
        if self.model_type_ == "Normalizer":
            return transform_from_normalizer(
                X, self.attributes_["values"], self.attributes_["method"]
            )
        elif self.model_type_ == "PCA":
            return transform_from_pca(
                X, self.attributes_["principal_components"], self.attributes_["mean"],
            )
        elif self.model_type_ == "SVD":
            return transform_from_svd(
                X, self.attributes_["vectors"], self.attributes_["values"]
            )
        elif self.model_type_ == "OneHotEncoder":
            return transform_from_one_hot_encoder(
                X, self.attributes_["categories"], self.attributes_["drop_first"]
            )
        elif self.model_type_ in ("KMeans", "NearestCentroid", "BisectingKMeans",):
            return predict_from_clusters(
                X, self.attributes_["clusters"], return_distance_clusters=True
            )
        elif self.model_type_ == "KPrototypes":
            return predict_from_clusters_kprotypes(
                X,
                self.attributes_["clusters"],
                return_distance_clusters=True,
                gamma=self.attributes_["gamma"],
            )
        else:
            raise FunctionError(
                f"Method 'transform' is not available for model type '{self.model_type_}'."
            )

    def transform_sql(self, X: list) -> list:
        """
    Returns the SQL code needed to deploy the model.

    Parameters
    ----------
    X: list
        Name or values of the input predictors.

    Returns
    -------
    list
        SQL code
        """
        if self.model_type_ == "Normalizer":
            result = sql_from_normalizer(
                X, self.attributes_["values"], self.attributes_["method"]
            )
        elif self.model_type_ == "PCA":
            result = sql_from_pca(
                X, self.attributes_["principal_components"], self.attributes_["mean"],
            )
        elif self.model_type_ == "SVD":
            result = sql_from_svd(
                X, self.attributes_["vectors"], self.attributes_["values"]
            )
        elif self.model_type_ == "OneHotEncoder":
            result = sql_from_one_hot_encoder(
                X,
                self.attributes_["categories"],
                self.attributes_["drop_first"],
                self.attributes_["column_naming"],
            )
        elif self.model_type_ in ("KMeans", "NearestCentroid", "BisectingKMeans"):
            result = sql_from_clusters(
                X, self.attributes_["clusters"], return_distance_clusters=True
            )
        elif self.model_type_ == "KPrototypes":
            result = sql_from_clusters_kprotypes(
                X,
                self.attributes_["clusters"],
                return_distance_clusters=True,
                gamma=self.attributes_["gamma"],
                is_categorical=self.attributes_["is_categorical"],
            )
        else:
            raise FunctionError(
                f"Method 'transform_sql' is not available for model type '{self.model_type_}'."
            )
        if self.model_type_ == "OneHotEncoder":
            for idx in range(len(result)):
                result[idx] = [r.replace("\xa0", " ") for r in result[idx]]
            return result
        else:
            return [r.replace("\xa0", " ") for r in result]

    def rotate(self, gamma: float = 1.0, q: int = 20, tol: float = 1e-6):
        """
    Performs a Oblimin (Varimax, Quartimax) rotation on the the model's PCA 
    matrix.

    Parameters
    ----------
    gamma: float, optional
        Oblimin rotation factor, determines the type of rotation.
        It must be between 0.0 and 1.0.
            gamma = 0.0 results in a Quartimax rotation.
            gamma = 1.0 results in a Varimax rotation.
    q: int, optional
        Maximum number of iterations.
    tol: float, optional
        The algorithm stops when the Frobenius norm of gradient is less than tol.

    Returns
    -------
    self
        memModel
        """
        if self.model_type_ == "PCA":
            principal_components = matrix_rotation(
                self.get_attributes()["principal_components"], gamma, q, tol
            )
            self.set_attributes({"principal_components": principal_components})
        else:
            raise FunctionError(
                f"Method 'rotate' is not available for model type '{self.model_type_}'."
            )
        return self
