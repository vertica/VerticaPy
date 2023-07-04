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
            auto : square root  of the total number of
                   predictors.
            max  : number of predictors.
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
            auto : square root  of the total number of
                   predictors.
            max  : number of predictors.
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
