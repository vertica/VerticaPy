# (c) Copyright [2018-2022] Micro Focus or one of its affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# |_     |~) _  _| _  /~\    _ |.
# |_)\/  |_)(_|(_||   \_/|_|(_|||
#    /
#              ____________       ______
#             / __        `\     /     /
#            |  \/         /    /     /
#            |______      /    /     /
#                   |____/    /     /
#          _____________     /     /
#          \           /    /     /
#           \         /    /     /
#            \_______/    /     /
#             ______     /     /
#             \    /    /     /
#              \  /    /     /
#               \/    /     /
#                    /     /
#                   /     /
#                   \    /
#                    \  /
#                     \/
#                    _
# \  / _  __|_. _ _ |_)
#  \/ (/_|  | |(_(_|| \/
#                     /
# VerticaPy is a Python library with scikit-like functionality for conducting
# data science projects on data stored in Vertica, taking advantage Vertica’s
# speed and built-in analytics and machine learning features. It supports the
# entire data science life cycle, uses a ‘pipeline’ mechanism to sequentialize
# data transformation operations, and offers beautiful graphical options.
#
# VerticaPy aims to do all of the above. The idea is simple: instead of moving
# data around for processing, VerticaPy brings the logic to the data.
#
#
# Modules
#
# Standard Python Modules
from typing import Union
import random
import numpy as np

# VerticaPy Modules
from verticapy.learn.metrics import *
from verticapy.learn.mlplot import *
from verticapy.utilities import *
from verticapy.toolbox import *
from verticapy import vDataFrame
from verticapy.errors import *
from verticapy.learn.vmodel import *
from verticapy.learn.tree import get_tree_list_of_arrays

# ---#
class XGBoost_utils:
    # Class:
    # - to export Vertica XGBoost to the Python XGBoost JSON format.
    # - to get the XGB priors

    def to_json(self, path: str = ""):
        """
        ---------------------------------------------------------------------------
        Creates a Python XGBoost JSON file that can be imported into the Python
        XGBoost API.
        
        \u26A0 Warning : For multiclass classifiers, the probabilities returned 
        by the VerticaPy and exported model may differ slightly because of 
        normalization; while Vertica uses multinomial logistic regression,  
        XGBoost Python uses Softmax. This difference does not affect the model's 
        final predictions. Categorical predictors must be encoded.

        Parameters
        ----------
        path: str, optional
            The path and name of the output file. If a file with the same name 
            already exists, the function returns an error.
            
        Returns
        -------
        str
            The content of the JSON file if variable 'path' is empty. Otherwise,
            nothing is returned.
        """

        def xgboost_to_json(model):
            def xgboost_dummy_tree_dict(model, i: int = 0):
                # Dummy trees are used to store the prior probabilities.
                # The Python XGBoost API do not use those information and start
                # the training with priors = 0
                result = {
                    "base_weights": [0.0],
                    "categories": [],
                    "categories_nodes": [],
                    "categories_segments": [],
                    "categories_sizes": [],
                    "default_left": [True],
                    "id": -1,
                    "left_children": [-1],
                    "loss_changes": [0.0],
                    "parents": [random.randint(2, 999999999)],
                    "right_children": [-1],
                    "split_conditions": [model.prior_[i]],
                    "split_indices": [0],
                    "split_type": [0],
                    "sum_hessian": [0.0],
                    "tree_param": {
                        "num_deleted": "0",
                        "num_feature": str(len(model.X)),
                        "num_nodes": "1",
                        "size_leaf_vector": "0",
                    },
                }
                return result

            def xgboost_tree_dict(model, tree_id: int, c: str = None):
                tree = model.get_tree(tree_id)
                attributes = get_tree_list_of_arrays(tree, model.X, model.type)
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
                    elif attributes[5][i] == None:
                        if model.type == "XGBoostRegressor":
                            split_conditions += [
                                float(attributes[4][i])
                                * model.parameters["learning_rate"]
                            ]
                        elif (
                            len(model.classes_) == 2
                            and model.classes_[1] == 1
                            and model.classes_[0] == 0
                        ):
                            split_conditions += [
                                model.parameters["learning_rate"]
                                * float(attributes[6][i]["1"])
                            ]
                        else:
                            split_conditions += [
                                model.parameters["learning_rate"]
                                * float(attributes[6][i][c])
                            ]
                    else:
                        split_conditions += [float(attributes[3][i])]
                result = {
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
                        "num_feature": str(len(model.X)),
                        "num_nodes": str(n_nodes),
                        "size_leaf_vector": "0",
                    },
                }
                return result

            def xgboost_tree_dict_list(model):
                n = model.get_attr("tree_count")["tree_count"][0]
                if model.type == "XGBoostClassifier" and (
                    len(model.classes_) > 2
                    or model.classes_[1] != 1
                    or model.classes_[0] != 0
                ):
                    trees = []
                    for i in range(n):
                        for c in model.classes_:
                            trees += [xgboost_tree_dict(model, i, str(c))]
                    v = version()
                    v = v[0] > 11 or (v[0] == 11 and (v[1] >= 1 or v[2] >= 1))
                    if not (v):
                        for i in range(len(model.classes_)):
                            trees += [xgboost_dummy_tree_dict(model, i)]
                    tree_info = [i for i in range(len(model.classes_))] * (
                        n + int(not (v))
                    )
                    for idx, tree in enumerate(trees):
                        tree["id"] = idx
                else:
                    trees = [xgboost_tree_dict(model, i) for i in range(n)]
                    tree_info = [0 for i in range(n)]
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

            def xgboost_learner(model):
                v = version()
                v = v[0] > 11 or (v[0] == 11 and (v[1] >= 1 or v[2] >= 1))
                if v:
                    col_sample_by_tree = model.parameters["col_sample_by_tree"]
                    col_sample_by_node = model.parameters["col_sample_by_node"]
                else:
                    col_sample_by_tree = "null"
                    col_sample_by_node = "null"
                condition = ["{} IS NOT NULL".format(elem) for elem in model.X] + [
                    "{} IS NOT NULL".format(model.y)
                ]
                n = model.get_attr("tree_count")["tree_count"][0]
                if model.type == "XGBoostRegressor" or (
                    len(model.classes_) == 2
                    and model.classes_[1] == 1
                    and model.classes_[0] == 0
                ):
                    bs, num_class, param, param_val = (
                        model.prior_,
                        "0",
                        "reg_loss_param",
                        {"scale_pos_weight": "1"},
                    )
                    if model.type == "XGBoostRegressor":
                        objective = "reg:squarederror"
                        attributes_dict = {
                            "scikit_learn": '{"n_estimators": '
                            + str(n)
                            + ', "objective": "reg:squarederror", "max_depth": '
                            + str(model.parameters["max_depth"])
                            + ', "learning_rate": '
                            + str(model.parameters["learning_rate"])
                            + ', "verbosity": null, "booster": null, "tree_method": null,'
                            + ' "gamma": null, "min_child_weight": null, "max_delta_step":'
                            + ' null, "subsample": null, "colsample_bytree": '
                            + str(col_sample_by_tree)
                            + ', "colsample_bylevel": null, "colsample_bynode": '
                            + str(col_sample_by_node)
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
                            + str(n)
                            + ', "objective": "binary:logistic", "max_depth": '
                            + str(model.parameters["max_depth"])
                            + ', "learning_rate": '
                            + str(model.parameters["learning_rate"])
                            + ', "verbosity": null, "booster": null, "tree_method": null,'
                            + ' "gamma": null, "min_child_weight": null, "max_delta_step":'
                            + ' null, "subsample": null, "colsample_bytree": '
                            + str(col_sample_by_tree)
                            + ', "colsample_bylevel": null, "colsample_bynode": '
                            + str(col_sample_by_node)
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
                        str(len(model.classes_)),
                        "softmax_multiclass_param",
                        {"num_class": str(len(model.classes_))},
                    )
                    attributes_dict = {
                        "scikit_learn": '{"use_label_encoder": true, "n_estimators": '
                        + str(n)
                        + ', "objective": "multi:softprob", "max_depth": '
                        + str(model.parameters["max_depth"])
                        + ', "learning_rate": '
                        + str(model.parameters["learning_rate"])
                        + ', "verbosity": null, "booster": null, "tree_method": null, '
                        + '"gamma": null, "min_child_weight": null, "max_delta_step": '
                        + 'null, "subsample": null, "colsample_bytree": '
                        + str(col_sample_by_tree)
                        + ', "colsample_bylevel": null, "colsample_bynode": '
                        + str(col_sample_by_node)
                        + ', "reg_alpha": null, "reg_lambda": null, "scale_pos_weight":'
                        + ' null, "base_score": null, "missing": NaN, "num_parallel_tree":'
                        + ' null, "kwargs": {}, "random_state": null, "n_jobs": null, '
                        + '"monotone_constraints": null, "interaction_constraints": null, '
                        + '"importance_type": "gain", "gpu_id": null, "validate_parameters":'
                        + ' null, "classes_": '
                        + str(model.classes_)
                        + ', "n_classes_": '
                        + str(len(model.classes_))
                        + ', "_le": {"classes_": '
                        + str(model.classes_)
                        + '}, "_estimator_type": "classifier"}'
                    }
                attributes_dict["scikit_learn"] = attributes_dict[
                    "scikit_learn"
                ].replace('"', "++++")
                gradient_booster = xgboost_tree_dict_list(model)
                return {
                    "attributes": attributes_dict,
                    "feature_names": [],
                    "feature_types": [],
                    "gradient_booster": gradient_booster,
                    "learner_model_param": {
                        "base_score": np.format_float_scientific(
                            bs, precision=7
                        ).upper(),
                        "num_class": num_class,
                        "num_feature": str(len(model.X)),
                    },
                    "objective": {"name": objective, param: param_val},
                }

            res = {"learner": xgboost_learner(model), "version": [1, 4, 2]}
            res = str(res)
            res = (
                res.replace("'", '"')
                .replace("True", "true")
                .replace("False", "false")
                .replace("++++", '\\"')
            )
            return res

        result = xgboost_to_json(self)
        if path:
            f = open(path, "w+")
            f.write(result)
        else:
            return result

    # ---#
    def get_prior(self):
        """
        ---------------------------------------------------------------------------
        Returns the XGB Priors.
            
        Returns
        -------
        list
            XGB Priors.
        """
        from verticapy.utilities import version

        condition = ["{} IS NOT NULL".format(elem) for elem in self.X] + [
            "{} IS NOT NULL".format(self.y)
        ]
        v = version()
        v = v[0] > 11 or (v[0] == 11 and (v[1] >= 1 or v[2] >= 1))
        if self.type == "XGBoostRegressor" or (
            len(self.classes_) == 2 and self.classes_[1] == 1 and self.classes_[0] == 0
        ):
            prior_ = executeSQL(
                "SELECT /*+LABEL('learn.ensemble.XGBoost_utils.get_prior')*/ AVG({}) FROM {} WHERE {}".format(
                    self.y, self.input_relation, " AND ".join(condition)
                ),
                method="fetchfirstelem",
                print_time_sql=False,
            )
        elif not (v):
            prior_ = []
            for elem in self.classes_:
                avg = executeSQL(
                    "SELECT /*+LABEL('learn.ensemble.XGBoost_utils.get_prior')*/ COUNT(*) FROM {} WHERE {} AND {} = '{}'".format(
                        self.input_relation, " AND ".join(condition), self.y, elem
                    ),
                    method="fetchfirstelem",
                    print_time_sql=False,
                )
                avg /= executeSQL(
                    "SELECT /*+LABEL('learn.ensemble.XGBoost_utils.get_prior')*/ COUNT(*) FROM {} WHERE {}".format(
                        self.input_relation, " AND ".join(condition)
                    ),
                    method="fetchfirstelem",
                    print_time_sql=False,
                )
                logodds = np.log(avg / (1 - avg))
                prior_ += [logodds]
        else:
            prior_ = [0.0 for elem in self.classes_]
        return prior_


# ---#
class IsolationForest(Clustering, Tree):
    """
---------------------------------------------------------------------------
Creates an IsolationForest object using the Vertica IFOREST algorithm.

Parameters
----------
name: str
    Name of the the model. The model is stored in the DB.
n_estimators: int, optional
    The number of trees in the forest, an integer between 1 and 1000, inclusive.
max_depth: int, optional
    Maximum depth of each tree, an integer between 1 and 100, inclusive.
nbins: int, optional
    Number of bins used for finding splits in each column. A larger number 
    of splits leads to a longer runtime but results in more fine-grained and 
    possibly better splits, an integer between 2 and 1000, inclusive.
sample: float, optional
    The portion of the input data set that is randomly selected for training each tree, 
    a float between 0.0 and 1.0, inclusive. 
col_sample_by_tree: float, optional
    Float in the range (0,1] that specifies the fraction of columns (features), 
    which are chosen at random, used when building each tree.
    """

    def __init__(
        self,
        name: str,
        n_estimators: int = 100,
        max_depth: int = 10,
        nbins: int = 32,
        sample: float = 0.632,
        col_sample_by_tree: float = 1.0,
    ):
        # Saving information to the query profile table
        save_to_query_profile(
            name="IsolationForest",
            path="learn.ensemble",
            json_dict={
                "name": name,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "nbins": nbins,
                "sample": sample,
                "col_sample_by_tree": col_sample_by_tree,
            },
        )
        # -#
        version(condition=[12, 0, 0])
        check_types([("name", name, [str], False)])
        self.type, self.name = "IsolationForest", name
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "nbins": nbins,
            "sample": sample,
            "col_sample_by_tree": col_sample_by_tree,
        }
        self.set_params(params)

    # ---#
    def decision_function(
        self,
        vdf: Union[str, vDataFrame],
        X: list = [],
        name: str = "",
        inplace: bool = True,
    ):
        """
    ---------------------------------------------------------------------------
    Returns the anomaly score using the input relation.

    Parameters
    ----------
    vdf: str/vDataFrame
        Object to use for the prediction. You can specify a customized 
        relation if it is enclosed with an alias. For example, 
        "(SELECT 1) x" is correct, whereas "(SELECT 1)" and "SELECT 1" are 
        incorrect.
    X: list, optional
        List of columns used to deploy the models. If empty, the model
        predictors are used.
    name: str, optional
        Name of the additional vColumn. If empty, a name is generated.
    inplace: bool, optional
        If True, the prediction is added to the vDataFrame.

    Returns
    -------
    vDataFrame
        the input object.
        """
        # Inititalization
        check_types(
            [
                ("name", name, [str]),
                ("vdf", vdf, [str, vDataFrame]),
                ("inplace", inplace, [bool]),
            ],
        )
        if isinstance(vdf, str):
            vdf = vDataFrameSQL(relation=vdf)
        if not (name):
            name = gen_name([self.type, self.name])

        # In Place
        vdf_return = vdf if inplace else vdf.copy()

        # Result
        return vdf_return.eval(name, self.deploySQL(X=X, return_score=True,))

    # ---#
    def deploySQL(
        self,
        X: list = [],
        cutoff: float = 0.7,
        contamination: float = None,
        return_score: bool = False,
    ):
        """
    ---------------------------------------------------------------------------
    Returns the SQL code needed to deploy the model. 

    Parameters
    ----------
    X: list, optional
        List of the columns used to deploy the model. If empty, the model
        predictors are used.
    cutoff: float, optional
        Float in the range (0.0, 1.0), specifies the threshold that 
        determines if a data point is an anomaly. If the anomaly_score 
        for a data point is greater than or equal to the cutoff, 
        the data point is marked as an anomaly.
    contamination: float, optional
        Float in the range (0,1), the approximate ratio of data points in the 
        training data that should be labeled as anomalous. If this parameter is 
        specified, the cutoff parameter is ignored.
    return_score: bool, optional
        If set to True, the anomaly score is returned, and the parameters 'cutoff'
        and 'contamination' are ignored.

    Returns
    -------
    str
        the SQL code needed to deploy the model.
        """
        if isinstance(X, str):
            X = [X]
        check_types(
            [
                ("X", X, [list]),
                ("cutoff", cutoff, [float]),
                ("contamination", contamination, [float]),
                ("return_score", return_score, [bool]),
            ]
        )
        if contamination and not (return_score):
            assert 0 < contamination < 1, ParameterError(
                "Incorrect parameter 'contamination'.\nThe parameter "
                "'contamination' must be between 0.0 and 1.0, exclusive."
            )
        elif not (return_score):
            assert 0 < cutoff < 1, ParameterError(
                "Incorrect parameter 'cutoff'.\nThe parameter "
                "'cutoff' must be between 0.0 and 1.0, exclusive."
            )
        X = [quote_ident(elem) for elem in X]
        if return_score:
            other_parameters = ""
        elif contamination:
            other_parameters = f", contamination = {contamination}"
        else:
            other_parameters = f", threshold = {cutoff}"
        fun = self.get_model_fun()[1]
        sql = "{}({} USING PARAMETERS model_name = '{}', match_by_pos = 'true'{})".format(
            fun, ", ".join(self.X if not (X) else X), self.name, other_parameters,
        )
        if return_score:
            sql = f"({sql}).anomaly_score"
        else:
            sql = f"(({sql}).is_anomaly)::int"
        return sql

    # ---#
    def predict(
        self,
        vdf: Union[str, vDataFrame],
        X: list = [],
        name: str = "",
        cutoff: float = 0.7,
        contamination: float = None,
        inplace: bool = True,
    ):
        """
    ---------------------------------------------------------------------------
    Predicts using the input relation.

    Parameters
    ----------
    vdf: str/vDataFrame
        Object to use for the prediction. You can specify a customized 
        relation if it is enclosed with an alias. For example, 
        "(SELECT 1) x" is correct, whereas "(SELECT 1)" and "SELECT 1" are 
        incorrect.
    X: list, optional
        List of the columns used to deploy the model. If empty, the model
        predictors are used.
    name: str, optional
        Name of the additional vColumn. If empty, a name is generated.
    cutoff: float, optional
        Float in the range (0.0, 1.0), specifies the threshold that 
        determines if a data point is an anomaly. If the anomaly_score 
        for a data point is greater than or equal to the cutfoff, 
        the data point is marked as an anomaly.
    contamination: float, optional
        Float in the range (0,1), the approximate ratio of data points in the
        training data that should be labeled as anomalous. If this parameter is 
        specified, the cutoff parameter is ignored.
    inplace: bool, optional
        If set to True, the prediction is added to the vDataFrame.

    Returns
    -------
    vDataFrame
        the input object.
        """
        # Inititalization
        check_types(
            [
                ("name", name, [str]),
                ("vdf", vdf, [str, vDataFrame]),
                ("inplace", inplace, [bool]),
            ],
        )
        if isinstance(vdf, str):
            vdf = vDataFrameSQL(relation=vdf)
        if not (name):
            name = gen_name([self.type, self.name])

        # In Place
        vdf_return = vdf if inplace else vdf.copy()

        # Result
        return vdf_return.eval(
            name, self.deploySQL(cutoff=cutoff, contamination=contamination, X=X)
        )


# ---#
class RandomForestClassifier(MulticlassClassifier, Tree):
    """
---------------------------------------------------------------------------
Creates a RandomForestClassifier object using the Vertica RF_CLASSIFIER 
function. It is one of the ensemble learning methods for classification 
that operates by constructing a multitude of decision trees at 
training-time and outputting a class with the mode.

Parameters
----------
name: str
  Name of the the model. The model will be stored in the DB.
n_estimators: int, optional
  The number of trees in the forest, an integer between 1 and 1000, inclusive.
max_features: int/str, optional
  The number of randomly chosen features from which to pick the best feature 
  to split on a given tree node. It can be an integer or one of the two following
  methods.
    auto : square root of the total number of predictors.
    max  : number of predictors.
max_leaf_nodes: int, optional
  The maximum number of leaf nodes a tree in the forest can have, an integer 
  between 1 and 1e9, inclusive.
sample: float, optional
  The portion of the input data set that is randomly picked for training each tree, 
  a float between 0.0 and 1.0, inclusive. 
max_depth: int, optional
  The maximum depth for growing each tree, an integer between 1 and 100, inclusive.
min_samples_leaf: int, optional
  The minimum number of samples each branch must have after splitting a node, an 
  integer between 1 and 1e6, inclusive. A split that causes fewer remaining samples 
  is discarded. 
min_info_gain: float, optional
  The minimum threshold for including a split, a float between 0.0 and 1.0, inclusive. 
  A split with information gain less than this threshold is discarded.
nbins: int, optional 
  The number of bins to use for continuous features, an integer between 2 and 1000, 
  inclusive.
  """

    def __init__(
        self,
        name: str,
        n_estimators: int = 10,
        max_features: Union[int, str] = "auto",
        max_leaf_nodes: int = 1e9,
        sample: float = 0.632,
        max_depth: int = 5,
        min_samples_leaf: int = 1,
        min_info_gain: float = 0.0,
        nbins: int = 32,
    ):
        # Saving information to the query profile table
        save_to_query_profile(
            name="RandomForestClassifier",
            path="learn.ensemble",
            json_dict={
                "name": name,
                "n_estimators": n_estimators,
                "max_features": max_features,
                "max_leaf_nodes": max_leaf_nodes,
                "sample": sample,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "min_info_gain": min_info_gain,
                "nbins": nbins,
            },
        )
        # -#
        version(condition=[8, 1, 1])
        check_types([("name", name, [str], False)])
        self.type, self.name = "RandomForestClassifier", name
        self.set_params(
            {
                "n_estimators": n_estimators,
                "max_features": max_features,
                "max_leaf_nodes": max_leaf_nodes,
                "sample": sample,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "min_info_gain": min_info_gain,
                "nbins": nbins,
            }
        )


# ---#
class RandomForestRegressor(Regressor, Tree):
    """
---------------------------------------------------------------------------
Creates a RandomForestRegressor object using the Vertica RF_REGRESSOR 
function. It is one of the ensemble learning methods for regression that 
operates by constructing a multitude of decision trees at training-time 
and outputting a class with the mode.

Parameters
----------
name: str
  Name of the the model. The model will be stored in the DB.
n_estimators: int, optional
  The number of trees in the forest, an integer between 1 and 1000, inclusive.
max_features: int/str, optional
  The number of randomly chosen features from which to pick the best feature 
  to split on a given tree node. It can be an integer or one of the two following
  methods.
    auto : square root of the total number of predictors.
    max  : number of predictors.
max_leaf_nodes: int, optional
  The maximum number of leaf nodes a tree in the forest can have, an integer 
  between 1 and 1e9, inclusive.
sample: float, optional
  The portion of the input data set that is randomly picked for training each tree, 
  a float between 0.0 and 1.0, inclusive. 
max_depth: int, optional
  The maximum depth for growing each tree, an integer between 1 and 100, inclusive.
min_samples_leaf: int, optional
  The minimum number of samples each branch must have after splitting a node, an 
  integer between 1 and 1e6, inclusive. A split that causes fewer remaining samples 
  is discarded. 
min_info_gain: float, optional
  The minimum threshold for including a split, a float between 0.0 and 1.0, inclusive. 
  A split with information gain less than this threshold is discarded.
nbins: int, optional 
  The number of bins to use for continuous features, an integer between 2 and 1000, 
  inclusive.
  """

    def __init__(
        self,
        name: str,
        n_estimators: int = 10,
        max_features: Union[int, str] = "auto",
        max_leaf_nodes: int = 1e9,
        sample: float = 0.632,
        max_depth: int = 5,
        min_samples_leaf: int = 1,
        min_info_gain: float = 0.0,
        nbins: int = 32,
    ):
        # Saving information to the query profile table
        save_to_query_profile(
            name="RandomForestRegressor",
            path="learn.ensemble",
            json_dict={
                "name": name,
                "n_estimators": n_estimators,
                "max_features": max_features,
                "max_leaf_nodes": max_leaf_nodes,
                "sample": sample,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "min_info_gain": min_info_gain,
                "nbins": nbins,
            },
        )
        # -#
        version(condition=[9, 0, 1])
        check_types([("name", name, [str], False)])
        self.type, self.name = "RandomForestRegressor", name
        self.set_params(
            {
                "n_estimators": n_estimators,
                "max_features": max_features,
                "max_leaf_nodes": max_leaf_nodes,
                "sample": sample,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "min_info_gain": min_info_gain,
                "nbins": nbins,
            }
        )


# ---#
class XGBoostClassifier(MulticlassClassifier, Tree, XGBoost_utils):
    """
---------------------------------------------------------------------------
Creates an XGBoostClassifier object using the Vertica XGB_CLASSIFIER 
algorithm.

Parameters
----------
name: str
    Name of the the model. The model will be stored in the DB.
max_ntree: int, optional
    Maximum number of trees that will be created.
max_depth: int, optional
    Maximum depth of each tree, an integer between 1 and 20, inclusive.
nbins: int, optional
    Number of bins to use for finding splits in each column, more 
    splits leads to longer runtime but more fine-grained and possibly 
    better splits, an integer between 2 and 1000, inclusive.
split_proposal_method: str, optional
    approximate splitting strategy. Can be 'global' or 'local'
    (not yet supported)
tol: float, optional
    approximation error of quantile summary structures used in the 
    approximate split finding method.
learning_rate: float, optional
    weight applied to each tree's prediction, reduces each tree's 
    impact allowing for later trees to contribute, keeping earlier 
    trees from 'hogging' all the improvements.
min_split_loss: float, optional
    Each split must improve the objective function value of the model 
    by at least this much in order to not be pruned. Value of 0 is the 
    same as turning off this parameter (trees will still be pruned based 
    on positive/negative objective function values).
weight_reg: float, optional
    Regularization term that is applied to the weights of the leaves in 
    the regression tree. The higher this value is, the more sparse/smooth 
    the weights will be, which often helps prevent overfitting.
sample: float, optional
    Fraction of rows to use in training per iteration.
col_sample_by_tree: float, optional
    Float in the range (0,1] that specifies the fraction of columns (features), 
    chosen at random, to use when building each tree.
col_sample_by_node: float, optional
    Float in the range (0,1] that specifies the fraction of columns (features), 
    chosen at random, to use when evaluating each split.
    """

    def __init__(
        self,
        name: str,
        max_ntree: int = 10,
        max_depth: int = 5,
        nbins: int = 32,
        split_proposal_method: str = "global",
        tol: float = 0.001,
        learning_rate: float = 0.1,
        min_split_loss: float = 0.0,
        weight_reg: float = 0.0,
        sample: float = 1.0,
        col_sample_by_tree: float = 1.0,
        col_sample_by_node: float = 1.0,
    ):
        # Saving information to the query profile table
        save_to_query_profile(
            name="XGBoostClassifier",
            path="learn.ensemble",
            json_dict={
                "name": name,
                "max_ntree": max_ntree,
                "max_depth": max_depth,
                "nbins": nbins,
                "split_proposal_method": split_proposal_method,
                "tol": tol,
                "learning_rate": learning_rate,
                "min_split_loss": min_split_loss,
                "weight_reg": weight_reg,
                "sample": sample,
                "col_sample_by_tree": col_sample_by_tree,
                "col_sample_by_node": col_sample_by_node,
            },
        )
        # -#
        version(condition=[10, 1, 0])
        check_types([("name", name, [str], False)])
        self.type, self.name = "XGBoostClassifier", name
        params = {
            "max_ntree": max_ntree,
            "max_depth": max_depth,
            "nbins": nbins,
            "split_proposal_method": split_proposal_method,
            "tol": tol,
            "learning_rate": learning_rate,
            "min_split_loss": min_split_loss,
            "weight_reg": weight_reg,
            "sample": sample,
        }
        v = version()
        v = v[0] > 11 or (v[0] == 11 and (v[1] >= 1 or v[2] >= 1))
        if v:
            params["col_sample_by_tree"] = col_sample_by_tree
            params["col_sample_by_node"] = col_sample_by_node
        self.set_params(params)


# ---#
class XGBoostRegressor(Regressor, Tree, XGBoost_utils):
    """
---------------------------------------------------------------------------
Creates an XGBoostRegressor object using the Vertica XGB_REGRESSOR 
algorithm.

Parameters
----------
name: str
    Name of the the model. The model will be stored in the DB.
max_ntree: int, optional
    Maximum number of trees that will be created.
max_depth: int, optional
    Maximum depth of each tree, an integer between 1 and 20, inclusive.
nbins: int, optional
    Number of bins to use for finding splits in each column, more 
    splits leads to longer runtime but more fine-grained and possibly 
    better splits, an integer between 2 and 1000, inclusive.
split_proposal_method: str, optional
    approximate splitting strategy. Can be 'global' or 'local'
    (not yet supported)
tol: float, optional
    approximation error of quantile summary structures used in the 
    approximate split finding method.
learning_rate: float, optional
    weight applied to each tree's prediction, reduces each tree's 
    impact allowing for later trees to contribute, keeping earlier 
    trees from 'hogging' all the improvements.
min_split_loss: float, optional
    Each split must improve the objective function value of the model 
    by at least this much in order to not be pruned. Value of 0 is the 
    same as turning off this parameter (trees will still be pruned based 
    on positive/negative objective function values).
weight_reg: float, optional
    Regularization term that is applied to the weights of the leaves in 
    the regression tree. The higher this value is, the more sparse/smooth 
    the weights will be, which often helps prevent overfitting.
sample: float, optional
    Fraction of rows to use in training per iteration.
col_sample_by_tree: float, optional
    Float in the range (0,1] that specifies the fraction of columns (features), 
    chosen at random, to use when building each tree.
col_sample_by_node: float, optional
    Float in the range (0,1] that specifies the fraction of columns (features), 
    chosen at random, to use when evaluating each split.
    """

    def __init__(
        self,
        name: str,
        max_ntree: int = 10,
        max_depth: int = 5,
        nbins: int = 32,
        split_proposal_method: str = "global",
        tol: float = 0.001,
        learning_rate: float = 0.1,
        min_split_loss: float = 0.0,
        weight_reg: float = 0.0,
        sample: float = 1.0,
        col_sample_by_tree: float = 1.0,
        col_sample_by_node: float = 1.0,
    ):
        # Saving information to the query profile table
        save_to_query_profile(
            name="XGBoostRegressor",
            path="learn.ensemble",
            json_dict={
                "name": name,
                "max_ntree": max_ntree,
                "max_depth": max_depth,
                "nbins": nbins,
                "split_proposal_method": split_proposal_method,
                "tol": tol,
                "learning_rate": learning_rate,
                "min_split_loss": min_split_loss,
                "weight_reg": weight_reg,
                "sample": sample,
                "col_sample_by_tree": col_sample_by_tree,
                "col_sample_by_node": col_sample_by_node,
            },
        )
        # -#
        version(condition=[10, 1, 0])
        check_types([("name", name, [str], False)])
        self.type, self.name = "XGBoostRegressor", name
        params = {
            "max_ntree": max_ntree,
            "max_depth": max_depth,
            "nbins": nbins,
            "split_proposal_method": split_proposal_method,
            "tol": tol,
            "learning_rate": learning_rate,
            "min_split_loss": min_split_loss,
            "weight_reg": weight_reg,
            "sample": sample,
        }
        v = version()
        v = v[0] > 11 or (v[0] == 11 and (v[1] >= 1 or v[2] >= 1))
        if v:
            params["col_sample_by_tree"] = col_sample_by_tree
            params["col_sample_by_node"] = col_sample_by_node
        self.set_params(params)
