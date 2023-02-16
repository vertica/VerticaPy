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

#
#
# Modules
#
# Standard Python Modules
from typing import Union, Literal
import random
import numpy as np

# VerticaPy Modules
from verticapy._version import check_minimum_version
from verticapy._utils._collect import save_verticapy_logs
from verticapy._version import vertica_version
from verticapy._utils._gen import gen_name
from verticapy._utils._sql._execute import _executeSQL
from verticapy.core.vdataframe.base import vDataFrame
from verticapy.learn.vmodel import Clustering, Tree, MulticlassClassifier, Regressor
from verticapy.learn.tree import get_tree_list_of_arrays
from verticapy._utils._sql._format import quote_ident


class XGBoost:
    # Class:
    # - to export Vertica XGBoost to the Python XGBoost JSON format.
    # - to get the XGB priors

    def to_json(self, path: str = ""):
        """
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
                    v = vertica_version()
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
                v = vertica_version()
                v = v[0] > 11 or (v[0] == 11 and (v[1] >= 1 or v[2] >= 1))
                if v:
                    col_sample_by_tree = model.parameters["col_sample_by_tree"]
                    col_sample_by_node = model.parameters["col_sample_by_node"]
                else:
                    col_sample_by_tree = "null"
                    col_sample_by_node = "null"
                condition = [f"{predictor} IS NOT NULL" for predictor in model.X] + [
                    f"{model.y} IS NOT NULL"
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

    def get_prior(self):
        """
        Returns the XGB Priors.
            
        Returns
        -------
        list
            XGB Priors.
        """
        condition = [f"{x} IS NOT NULL" for x in self.X] + [f"{self.y} IS NOT NULL"]
        v = vertica_version()
        v = v[0] > 11 or (v[0] == 11 and (v[1] >= 1 or v[2] >= 1))
        query = f"""
            SELECT 
                /*+LABEL('learn.ensemble.XGBoost.get_prior')*/ 
                {{}}
            FROM {self.input_relation} 
            WHERE {' AND '.join(condition)}{{}}"""
        if self.type == "XGBoostRegressor" or (
            len(self.classes_) == 2 and self.classes_[1] == 1 and self.classes_[0] == 0
        ):
            prior_ = _executeSQL(
                query=query.format(f"AVG({self.y})", ""),
                method="fetchfirstelem",
                print_time_sql=False,
            )
        elif not (v):
            prior_ = []
            for c in self.classes_:
                avg = _executeSQL(
                    query=query.format("COUNT(*)", f" AND {self.y} = '{c}'"),
                    method="fetchfirstelem",
                    print_time_sql=False,
                )
                avg /= _executeSQL(
                    query=query.format("COUNT(*)", ""),
                    method="fetchfirstelem",
                    print_time_sql=False,
                )
                logodds = np.log(avg / (1 - avg))
                prior_ += [logodds]
        else:
            prior_ = [0.0 for p in self.classes_]
        return prior_


class IsolationForest(Clustering, Tree):
    """
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

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str,
        n_estimators: int = 100,
        max_depth: int = 10,
        nbins: int = 32,
        sample: float = 0.632,
        col_sample_by_tree: float = 1.0,
    ):
        self.type, self.name = "IsolationForest", name
        self.VERTICA_FIT_FUNCTION_SQL = "IFOREST"
        self.VERTICA_PREDICT_FUNCTION_SQL = "APPLY_IFOREST"
        self.MODEL_TYPE = "UNSUPERVISED"
        self.MODEL_SUBTYPE = "ANOMALY_DETECTION"
        self.parameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "nbins": nbins,
            "sample": sample,
            "col_sample_by_tree": col_sample_by_tree,
        }

    def decision_function(
        self,
        vdf: Union[str, vDataFrame],
        X: list = [],
        name: str = "",
        inplace: bool = True,
    ):
        """
    Returns the anomaly score using the input relation.

    Parameters
    ----------
    vdf: str / vDataFrame
        Object to use for the prediction. You can specify a customized 
        relation if it is enclosed with an alias. For example, 
        "(SELECT 1) x" is correct, whereas "(SELECT 1)" and "SELECT 1" are 
        incorrect.
    X: list, optional
        List of columns used to deploy the models. If empty, the model
        predictors are used.
    name: str, optional
        Name of the additional vDataColumn. If empty, a name is generated.
    inplace: bool, optional
        If True, the prediction is added to the vDataFrame.

    Returns
    -------
    vDataFrame
        the input object.
        """
        # Inititalization
        if isinstance(vdf, str):
            vdf = vDataFrameSQL(relation=vdf)
        if not (name):
            name = gen_name([self.type, self.name])

        # In Place
        vdf_return = vdf if inplace else vdf.copy()

        # Result
        return vdf_return.eval(name, self.deploySQL(X=X, return_score=True,))

    def deploySQL(
        self,
        X: Union[str, list] = [],
        cutoff: Union[int, float] = 0.7,
        contamination: Union[int, float] = None,
        return_score: bool = False,
    ):
        """
    Returns the SQL code needed to deploy the model. 

    Parameters
    ----------
    X: str / list, optional
        List of the columns used to deploy the model. If empty, the model
        predictors are used.
    cutoff: int / float, optional
        Float in the range (0.0, 1.0), specifies the threshold that 
        determines if a data point is an anomaly. If the anomaly_score 
        for a data point is greater than or equal to the cutoff, 
        the data point is marked as an anomaly.
    contamination: int / float, optional
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
        X = self.X if not (X) else [quote_ident(elem) for elem in X]
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
        if return_score:
            other_parameters = ""
        elif contamination:
            other_parameters = f", contamination = {contamination}"
        else:
            other_parameters = f", threshold = {cutoff}"
        sql = f"{self.VERTICA_PREDICT_FUNCTION_SQL}({', '.join(X)} USING PARAMETERS model_name = '{self.name}', match_by_pos = 'true'{other_parameters})"
        if return_score:
            sql = f"({sql}).anomaly_score"
        else:
            sql = f"(({sql}).is_anomaly)::int"
        return sql

    def predict(
        self,
        vdf: Union[str, vDataFrame],
        X: Union[str, list] = [],
        name: str = "",
        cutoff: Union[int, float] = 0.7,
        contamination: Union[int, float] = None,
        inplace: bool = True,
    ):
        """
    Predicts using the input relation.

    Parameters
    ----------
    vdf: str / vDataFrame
        Object to use for the prediction. You can specify a customized 
        relation if it is enclosed with an alias. For example, 
        "(SELECT 1) x" is correct, whereas "(SELECT 1)" and "SELECT 1" are 
        incorrect.
    X: list, optional
        List of the columns used to deploy the model. If empty, the model
        predictors are used.
    name: str, optional
        Name of the additional vDataColumn. If empty, a name is generated.
    cutoff: int / float, optional
        Float in the range (0.0, 1.0), specifies the threshold that 
        determines if a data point is an anomaly. If the anomaly_score 
        for a data point is greater than or equal to the cutfoff, 
        the data point is marked as an anomaly.
    contamination: int / float, optional
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


class RandomForestClassifier(MulticlassClassifier, Tree):
    """
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
max_features: int / str, optional
    The number of randomly chosen features from which to pick the best feature 
    to split on a given tree node. It can be an integer or one of the two following
    methods.
        auto : square root of the total number of predictors.
        max  : number of predictors.
max_leaf_nodes: int / float, optional
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
min_info_gain: int / float, optional
    The minimum threshold for including a split, a float between 0.0 and 1.0, inclusive. 
    A split with information gain less than this threshold is discarded.
nbins: int, optional 
    The number of bins to use for continuous features, an integer between 2 and 1000, 
    inclusive.
    """

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str,
        n_estimators: int = 10,
        max_features: Union[Literal["auto", "max"], int] = "auto",
        max_leaf_nodes: Union[int, float] = 1e9,
        sample: float = 0.632,
        max_depth: int = 5,
        min_samples_leaf: int = 1,
        min_info_gain: Union[int, float] = 0.0,
        nbins: int = 32,
    ):
        self.type, self.name = "RandomForestClassifier", name
        self.VERTICA_FIT_FUNCTION_SQL = "RF_CLASSIFIER"
        self.VERTICA_PREDICT_FUNCTION_SQL = "PREDICT_RF_CLASSIFIER"
        self.MODEL_TYPE = "SUPERVISED"
        self.MODEL_SUBTYPE = "CLASSIFIER"
        self.parameters = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "max_leaf_nodes": max_leaf_nodes,
            "sample": sample,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_info_gain": min_info_gain,
            "nbins": nbins,
        }


class RandomForestRegressor(Regressor, Tree):
    """
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
max_features: int / str, optional
    The number of randomly chosen features from which to pick the best feature 
    to split on a given tree node. It can be an integer or one of the two following
    methods.
        auto : square root of the total number of predictors.
        max  : number of predictors.
max_leaf_nodes: int / float, optional
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
min_info_gain: int / float, optional
    The minimum threshold for including a split, a float between 0.0 and 1.0, inclusive. 
    A split with information gain less than this threshold is discarded.
nbins: int, optional 
    The number of bins to use for continuous features, an integer between 2 and 1000, 
    inclusive.
    """

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str,
        n_estimators: int = 10,
        max_features: Union[Literal["auto", "max"], int] = "auto",
        max_leaf_nodes: Union[int, float] = 1e9,
        sample: float = 0.632,
        max_depth: int = 5,
        min_samples_leaf: int = 1,
        min_info_gain: Union[int, float] = 0.0,
        nbins: int = 32,
    ):
        self.type, self.name = "RandomForestRegressor", name
        self.VERTICA_FIT_FUNCTION_SQL = "RF_REGRESSOR"
        self.VERTICA_PREDICT_FUNCTION_SQL = "PREDICT_RF_REGRESSOR"
        self.MODEL_TYPE = "SUPERVISED"
        self.MODEL_SUBTYPE = "REGRESSOR"
        self.parameters = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "max_leaf_nodes": max_leaf_nodes,
            "sample": sample,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_info_gain": min_info_gain,
            "nbins": nbins,
        }


class XGBoostClassifier(MulticlassClassifier, Tree, XGBoost):
    """
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

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str,
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
    ):
        self.type, self.name = "XGBoostClassifier", name
        self.VERTICA_FIT_FUNCTION_SQL = "XGB_CLASSIFIER"
        self.VERTICA_PREDICT_FUNCTION_SQL = "PREDICT_XGB_CLASSIFIER"
        self.MODEL_TYPE = "SUPERVISED"
        self.MODEL_SUBTYPE = "CLASSIFIER"
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
        }
        v = vertica_version()
        v = v[0] > 11 or (v[0] == 11 and (v[1] >= 1 or v[2] >= 1))
        if v:
            params["col_sample_by_tree"] = col_sample_by_tree
            params["col_sample_by_node"] = col_sample_by_node
        self.parameters = params


class XGBoostRegressor(Regressor, Tree, XGBoost):
    """
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

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str,
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
    ):
        self.type, self.name = "XGBoostRegressor", name
        self.VERTICA_FIT_FUNCTION_SQL = "XGB_REGRESSOR"
        self.VERTICA_PREDICT_FUNCTION_SQL = "PREDICT_XGB_REGRESSOR"
        self.MODEL_TYPE = "SUPERVISED"
        self.MODEL_SUBTYPE = "REGRESSOR"
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
        }
        v = vertica_version()
        v = v[0] > 11 or (v[0] == 11 and (v[1] >= 1 or v[2] >= 1))
        if v:
            params["col_sample_by_tree"] = col_sample_by_tree
            params["col_sample_by_node"] = col_sample_by_node
        self.parameters = params
