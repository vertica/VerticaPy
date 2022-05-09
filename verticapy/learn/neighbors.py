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
# VerticaPy Modules
from verticapy.learn.metrics import *
from verticapy.learn.mlplot import *
from verticapy.utilities import *
from verticapy.toolbox import *
from verticapy import vDataFrame
from verticapy.learn.mlplot import *
from verticapy.learn.model_selection import *
from verticapy.errors import *
from verticapy.learn.vmodel import *
from verticapy.learn.tools import *

# Standard Python Modules
import warnings
from typing import Union

# ---#
class NearestCentroid(MulticlassClassifier):
    """
---------------------------------------------------------------------------
[Beta Version]
Creates a NearestCentroid object using the k-nearest centroid algorithm. 
This object uses pure SQL to compute the distances and final score. 

\u26A0 Warning : Because this algorithm uses p-distances, it is highly 
                 sensitive to unnormalized data.

Parameters
----------
p: int, optional
	The p corresponding to the one of the p-distances (distance metric used
	to compute the model).
	"""

    def __init__(self, name: str, p: int = 2):
        check_types([("name", name, [str], False)])
        self.type, self.name = "NearestCentroid", name
        self.set_params({"p": p})

    # ---#
    def fit(
        self,
        input_relation: Union[str, vDataFrame],
        X: list,
        y: str,
        test_relation: Union[str, vDataFrame] = "",
    ):
        """
	---------------------------------------------------------------------------
	Trains the model.

	Parameters
	----------
	input_relation: str/vDataFrame
		Training relation.
	X: list
		List of the predictors.
	y: str
		Response column.
	test_relation: str/vDataFrame, optional
		Relation used to test the model.

	Returns
	-------
	object
 		self
		"""
        if isinstance(X, str):
            X = [X]
        check_types(
            [
                ("input_relation", input_relation, [str, vDataFrame], False),
                ("X", X, [list], False),
                ("y", y, [str], False),
                ("test_relation", test_relation, [str, vDataFrame], False),
            ]
        )
        if verticapy.options["overwrite_model"]:
            self.drop()
        else:
            does_model_exist(name=self.name, raise_error=True)
        func = "APPROXIMATE_MEDIAN" if (self.parameters["p"] == 1) else "AVG"
        if isinstance(input_relation, vDataFrame):
            self.input_relation = input_relation.__genSQL__()
        else:
            self.input_relation = input_relation
        if isinstance(test_relation, vDataFrame):
            self.test_relation = test_relation.__genSQL__()
        elif test_relation:
            self.test_relation = test_relation
        else:
            self.test_relation = self.input_relation
        self.X = [quote_ident(column) for column in X]
        self.y = quote_ident(y)
        query = "SELECT {}, {} FROM {} WHERE {} IS NOT NULL GROUP BY {} ORDER BY {} ASC".format(
            ", ".join(
                ["{}({}) AS {}".format(func, column, column) for column in self.X]
            ),
            self.y,
            self.input_relation,
            self.y,
            self.y,
            self.y,
        )
        self.centroids_ = to_tablesample(query=query, title="Getting Model Centroids.",)
        self.classes_ = self.centroids_.values[y]
        model_save = {
            "type": "NearestCentroid",
            "input_relation": self.input_relation,
            "test_relation": self.test_relation,
            "X": self.X,
            "y": self.y,
            "p": self.parameters["p"],
            "centroids": self.centroids_.values,
            "classes": self.classes_,
        }
        insert_verticapy_schema(
            model_name=self.name, model_type="NearestCentroid", model_save=model_save,
        )
        return self


# ---#
class KNeighborsClassifier(vModel):
    """
---------------------------------------------------------------------------
[Beta Version]
Creates a KNeighborsClassifier object using the k-nearest neighbors algorithm. 
This object uses pure SQL to compute the distances and final score.

\u26A0 Warning : This algorithm uses a CROSS JOIN during computation and
                 is therefore computationally expensive at O(n * n), where
                 n is the total number of elements. Because this algorithm  
                 uses the p-distance, it is highly sensitive to unnormalized 
                 data.

Parameters
----------
n_neighbors: int, optional
	Number of neighbors to consider when computing the score.
p: int, optional
	The p corresponding to the one of the p-distances (distance metric used  
	to compute the model).
	"""

    def __init__(self, name: str, n_neighbors: int = 5, p: int = 2):
        check_types([("name", name, [str], False)])
        self.type, self.name = "KNeighborsClassifier", name
        self.set_params({"n_neighbors": n_neighbors, "p": p})

    # ---#
    def deploySQL(
        self,
        X: list = [],
        test_relation: str = "",
        predict: bool = False,
        key_columns: list = [],
    ):
        """
	---------------------------------------------------------------------------
	Returns the SQL code needed to deploy the model. 

    Parameters
    ----------
    X: list
        List of the predictors.
    test_relation: str, optional
        Relation to use to do the predictions.
    predict: bool, optional
        If set to True, returns the prediction instead of the probability.
    key_columns: list, optional
        A list of columns to include in the results, but to exclude from 
        computation of the prediction.

    Returns
    -------
    str/list
        the SQL code needed to deploy the model.
		"""
        if isinstance(X, str):
            X = [X]
        if isinstance(key_columns, str):
            key_columns = [key_columns]
        check_types(
            [
                ("test_relation", test_relation, [str], False),
                ("predict", predict, [bool], False),
                ("X", X, [list], False),
                ("key_columns", key_columns, [list], False),
            ],
        )
        X = [quote_ident(elem) for elem in X] if (X) else self.X
        if not (test_relation):
            test_relation = self.test_relation
        if not (key_columns) and key_columns != None:
            key_columns = [self.y]
        sql = [
            "POWER(ABS(x.{} - y.{}), {})".format(X[i], self.X[i], self.parameters["p"])
            for i in range(len(self.X))
        ]
        sql = "POWER({}, 1 / {})".format(" + ".join(sql), self.parameters["p"])
        sql = "ROW_NUMBER() OVER(PARTITION BY {}, row_id ORDER BY {})".format(
            ", ".join(["x.{}".format(item) for item in X]), sql
        )
        sql = "SELECT {}{}, {} AS ordered_distance, y.{} AS predict_neighbors, row_id FROM (SELECT *, ROW_NUMBER() OVER() AS row_id FROM {} WHERE {}) x CROSS JOIN (SELECT * FROM {} WHERE {}) y".format(
            ", ".join(["x.{}".format(item) for item in X]),
            ", " + ", ".join(["x." + quote_ident(elem) for elem in key_columns])
            if (key_columns)
            else "",
            sql,
            self.y,
            test_relation,
            " AND ".join(["{} IS NOT NULL".format(item) for item in X]),
            self.input_relation,
            " AND ".join(["{} IS NOT NULL".format(item) for item in self.X]),
        )
        sql = "(SELECT row_id, {}{}, predict_neighbors, COUNT(*) / {} AS proba_predict FROM ({}) z WHERE ordered_distance <= {} GROUP BY {}{}, row_id, predict_neighbors) kneighbors_table".format(
            ", ".join(X),
            ", " + ", ".join([quote_ident(elem) for elem in key_columns])
            if (key_columns)
            else "",
            self.parameters["n_neighbors"],
            sql,
            self.parameters["n_neighbors"],
            ", ".join(X),
            ", " + ", ".join([quote_ident(elem) for elem in key_columns])
            if (key_columns)
            else "",
        )
        if predict:
            sql = "(SELECT {}{}, predict_neighbors FROM (SELECT {}{}, predict_neighbors, ROW_NUMBER() OVER (PARTITION BY {} ORDER BY proba_predict DESC) AS order_prediction FROM {}) VERTICAPY_SUBTABLE WHERE order_prediction = 1) predict_neighbors_table".format(
                ", ".join(X),
                ", " + ", ".join([quote_ident(elem) for elem in key_columns])
                if (key_columns)
                else "",
                ", ".join(X),
                ", " + ", ".join([quote_ident(elem) for elem in key_columns])
                if (key_columns)
                else "",
                ", ".join(X),
                sql,
            )
        return sql

    # ---#
    def fit(
        self,
        input_relation: Union[str, vDataFrame],
        X: list,
        y: str,
        test_relation: Union[str, vDataFrame] = "",
    ):
        """
	---------------------------------------------------------------------------
	Trains the model.

	Parameters
	----------
	input_relation: str/vDataFrame
		Training relation.
	X: list
		List of the predictors.
	y: str
		Response column.
	test_relation: str/vDataFrame, optional
		Relation used to test the model.

	Returns
	-------
	object
 		self
		"""
        if isinstance(X, str):
            X = [X]
        check_types(
            [
                ("input_relation", input_relation, [str, vDataFrame], False),
                ("X", X, [list], False),
                ("y", y, [str], False),
                ("test_relation", test_relation, [str, vDataFrame], False),
            ]
        )
        if verticapy.options["overwrite_model"]:
            self.drop()
        else:
            does_model_exist(name=self.name, raise_error=True)
        if isinstance(input_relation, vDataFrame):
            self.input_relation = input_relation.__genSQL__()
        else:
            self.input_relation = input_relation
        if isinstance(test_relation, vDataFrame):
            self.test_relation = test_relation.__genSQL__()
        elif test_relation:
            self.test_relation = test_relation
        else:
            self.test_relation = self.input_relation
        self.X = [quote_ident(column) for column in X]
        self.y = quote_ident(y)
        classes = executeSQL(
            "SELECT DISTINCT {} FROM {} WHERE {} IS NOT NULL ORDER BY {} ASC".format(
                self.y, self.input_relation, self.y, self.y
            ),
            method="fetchall",
            print_time_sql=False,
        )
        self.classes_ = [item[0] for item in classes]
        model_save = {
            "type": "KNeighborsClassifier",
            "input_relation": self.input_relation,
            "test_relation": self.test_relation,
            "X": self.X,
            "y": self.y,
            "p": self.parameters["p"],
            "n_neighbors": self.parameters["n_neighbors"],
            "classes": self.classes_,
        }
        insert_verticapy_schema(
            model_name=self.name,
            model_type="KNeighborsClassifier",
            model_save=model_save,
        )
        return self

    # ---#
    def classification_report(self, cutoff: Union[float, list] = [], labels: list = []):
        """
    ---------------------------------------------------------------------------
    Computes a classification report using multiple metrics to evaluate the model
    (AUC, accuracy, PRC AUC, F1, etc.). For multiclass classification, this 
    function tests the model by considering one class as the sole positive case, 
    repeating the process until it tests all classes.

    Parameters
    ----------
    cutoff: float/list, optional
        Cutoff for which the tested category is accepted as a prediction. 
        For multiclass classification, each tested category becomes positive case
        and untested categories are merged into the negative cases. This list 
        represents the threshold for each class. If empty, the best cutoff is be used.
    labels: list, optional
        List of the different labels to be used during the computation.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.
        """
        if not (isinstance(labels, Iterable)) or isinstance(labels, str):
            labels = [labels]
        check_types(
            [("cutoff", cutoff, [int, float, list]), ("labels", labels, [list])]
        )
        if not (labels):
            labels = self.classes_
        return classification_report(cutoff=cutoff, estimator=self, labels=labels)

    report = classification_report

    # ---#
    def cutoff_curve(
        self, pos_label: Union[int, float, str] = None, ax=None, **style_kwds
    ):
        """
    ---------------------------------------------------------------------------
    Draws the ROC curve of a classification model.

    Parameters
    ----------
    pos_label: int/float/str
        The response column class to be considered positive.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.
        """
        pos_label = (
            self.classes_[1]
            if (pos_label == None and len(self.classes_) == 2)
            else pos_label
        )
        if pos_label not in self.classes_:
            raise ParameterError(
                "'pos_label' must be one of the response column classes"
            )
        input_relation = self.deploySQL() + " WHERE predict_neighbors = '{}'".format(
            pos_label
        )
        return roc_curve(
            self.y,
            "proba_predict",
            input_relation,
            pos_label,
            ax=ax,
            cutoff_curve=True,
            **style_kwds,
        )

    # ---#
    def confusion_matrix(
        self, pos_label: Union[int, float, str] = None, cutoff: float = -1
    ):
        """
    ---------------------------------------------------------------------------
    Computes the model confusion matrix.

    Parameters
    ----------
    pos_label: int/float/str, optional
        Label to consider as positive. All the other classes will be merged and
        considered as negative for multiclass classification.
    cutoff: float, optional
        Cutoff for which the tested category will be accepted as a prediction. If the 
        cutoff is not between 0 and 1, the entire confusion matrix will be drawn.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.
        """
        check_types([("cutoff", cutoff, [int, float])])
        pos_label = (
            self.classes_[1]
            if (pos_label == None and len(self.classes_) == 2)
            else pos_label
        )
        if pos_label in self.classes_ and cutoff <= 1 and cutoff >= 0:
            input_relation = self.deploySQL() + " WHERE predict_neighbors = '{}'".format(
                pos_label
            )
            y_score = "(CASE WHEN proba_predict > {} THEN 1 ELSE 0 END)".format(cutoff)
            y_true = "DECODE({}, '{}', 1, 0)".format(self.y, pos_label)
            result = confusion_matrix(y_true, y_score, input_relation)
            if pos_label == 1:
                return result
            else:
                return tablesample(
                    values={
                        "index": ["Non-{}".format(pos_label), "{}".format(pos_label)],
                        "Non-{}".format(pos_label): result.values[0],
                        "{}".format(pos_label): result.values[1],
                    },
                )
        else:
            input_relation = "(SELECT *, ROW_NUMBER() OVER(PARTITION BY {}, row_id ORDER BY proba_predict DESC) AS pos FROM {}) neighbors_table WHERE pos = 1".format(
                ", ".join(self.X), self.deploySQL()
            )
            return multilabel_confusion_matrix(
                self.y, "predict_neighbors", input_relation, self.classes_
            )

    # ---#
    def lift_chart(
        self, pos_label: Union[int, float, str] = None, ax=None, **style_kwds
    ):
        """
    ---------------------------------------------------------------------------
    Draws the model Lift Chart.

    Parameters
    ----------
    pos_label: int/float/str
        To draw a lift chart, one of the response column classes must be the 
        positive one. The parameter 'pos_label' represents this class.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.
        """
        pos_label = (
            self.classes_[1]
            if (pos_label == None and len(self.classes_) == 2)
            else pos_label
        )
        if pos_label not in self.classes_:
            raise ParameterError(
                "'pos_label' must be one of the response column classes"
            )
        input_relation = self.deploySQL() + " WHERE predict_neighbors = '{}'".format(
            pos_label
        )
        return lift_chart(
            self.y, "proba_predict", input_relation, pos_label, ax=ax, **style_kwds,
        )

    # ---#
    def prc_curve(
        self, pos_label: Union[int, float, str] = None, ax=None, **style_kwds
    ):
        """
    ---------------------------------------------------------------------------
    Draws the model PRC curve.

    Parameters
    ----------
    pos_label: int/float/str
        To draw the PRC curve, one of the response column classes must be the 
        positive one. The parameter 'pos_label' represents this class.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.
        """
        pos_label = (
            self.classes_[1]
            if (pos_label == None and len(self.classes_) == 2)
            else pos_label
        )
        if pos_label not in self.classes_:
            raise ParameterError(
                "'pos_label' must be one of the response column classes"
            )
        input_relation = self.deploySQL() + " WHERE predict_neighbors = '{}'".format(
            pos_label
        )
        return prc_curve(
            self.y, "proba_predict", input_relation, pos_label, ax=ax, **style_kwds,
        )

    # ---#
    def predict(
        self,
        vdf: Union[str, vDataFrame],
        X: list = [],
        name: str = "",
        cutoff: float = 0.5,
        inplace: bool = True,
        **kwargs,
    ):
        """
    ---------------------------------------------------------------------------
    Predicts using the input relation.

    Parameters
    ----------
    vdf: str/vDataFrame
        Object to use to run the prediction. You can also specify a customized 
        relation, but you must enclose it with an alias. For example,  
        "(SELECT 1) x" is correct, whereas "(SELECT 1)" and "SELECT 1" are 
        incorrect.
    X: list, optional
        List of the columns used to deploy the models. If empty, the model
        predictors will be used.
    name: str, optional
        Name of the added vcolumn. If empty, a name will be generated.
    cutoff: float, optional
        Cutoff for which the tested category will be accepted as a prediction.
        This parameter is only used for binary classification.
    inplace: bool, optional
        If set to True, the prediction will be added to the vDataFrame.

    Returns
    -------
    vDataFrame
        the vDataFrame of the prediction
        """
        if isinstance(X, str):
            X = [X]
        check_types(
            [
                ("name", name, [str]),
                ("cutoff", cutoff, [int, float]),
                ("X", X, [list]),
                ("inplace", inplace, [bool]),
                ("vdf", vdf, [str, vDataFrame]),
            ],
        )
        assert 0 <= cutoff <= 1, ParameterError(
            "Incorrect parameter 'cutoff'.\nThe cutoff "
            "must be between 0 and 1, inclusive."
        )
        if isinstance(vdf, str):
            vdf = vDataFrameSQL(relation=vdf)
        X = [quote_ident(elem) for elem in X] if (X) else self.X
        key_columns = vdf.get_columns(exclude_columns=X)
        if "key_columns" in kwargs:
            key_columns_arg = None
        else:
            key_columns_arg = key_columns
        if not (name):
            name = gen_name([self.type, self.name])

        if (
            len(self.classes_) == 2
            and self.classes_[0] in [0, "0"]
            and self.classes_[1] in [1, "1"]
        ):
            sql = (
                "(SELECT {0}{1}, (CASE WHEN proba_predict > {2} THEN '{3}' ELSE '{4}' END)"
                " AS {5} FROM {6} WHERE predict_neighbors = '{3}') VERTICAPY_SUBTABLE"
            ).format(
                ", ".join(X),
                ", " + ", ".join(key_columns) if key_columns else "",
                cutoff,
                self.classes_[1],
                self.classes_[0],
                name,
                self.deploySQL(
                    X=X, test_relation=vdf.__genSQL__(), key_columns=key_columns_arg
                ),
            )
        else:
            sql = "(SELECT {0}{1}, predict_neighbors AS {2} FROM {3}) VERTICAPY_SUBTABLE".format(
                ", ".join(X),
                ", " + ", ".join(key_columns) if key_columns else "",
                name,
                self.deploySQL(
                    X=X,
                    test_relation=vdf.__genSQL__(),
                    key_columns=key_columns_arg,
                    predict=True,
                ),
            )
        if inplace:
            return vDataFrameSQL(name="Neighbors", relation=sql, vdf=vdf)
        else:
            return vDataFrameSQL(name="Neighbors", relation=sql)

    # ---#
    def predict_proba(
        self,
        vdf: Union[str, vDataFrame],
        X: list = [],
        name: str = "",
        pos_label: Union[int, str, float] = None,
        inplace: bool = True,
        **kwargs,
    ):
        """
    ---------------------------------------------------------------------------
    Returns the model's probabilities using the input relation.

    Parameters
    ----------
    vdf: str/vDataFrame
        Object to use to run the prediction. You can also specify a customized 
        relation, but you must enclose it with an alias. For example, "(SELECT 1) x" 
        is correct, whereas "(SELECT 1)" and "SELECT 1" are incorrect.
    X: list, optional
        List of the columns used to deploy the models. If empty, the model
        predictors will be used.
    name: str, optional
        Name of the additional prediction vColumn. If unspecified, a name is 
	generated based on the model and class names.
    pos_label: int/float/str, optional
        Class label, the class for which the probability is calculated. 
	If name is specified and pos_label is unspecified, the probability column 
	names use the following format: name_class1, name_class2, etc.
    inplace: bool, optional
        If set to True, the prediction will be added to the vDataFrame.

    Returns
    -------
    vDataFrame
        the vDataFrame of the prediction
        """
        # Inititalization
        if isinstance(X, str):
            X = [X]
        check_types(
            [
                ("name", name, [str]),
                ("X", X, [list]),
                ("inplace", inplace, [bool]),
                ("vdf", vdf, [str, vDataFrame]),
                ("pos_label", pos_label, [int, float, str]),
            ],
        )
        assert pos_label is None or pos_label in self.classes_, ParameterError(
            (
                "Incorrect parameter 'pos_label'.\nThe class label "
                "must be in [{0}]. Found '{1}'."
            ).format("|".join(["{}".format(c) for c in self.classes_]), pos_label)
        )
        if isinstance(vdf, str):
            vdf = vDataFrameSQL(relation=vdf)
        X = [quote_ident(elem) for elem in X] if (X) else self.X
        key_columns = vdf.get_columns(exclude_columns=X)
        if not (name):
            name = gen_name([self.type, self.name])
        if "key_columns" in kwargs:
            key_columns_arg = None
        else:
            key_columns_arg = key_columns

        # Generating the probabilities
        if pos_label == None:
            predict = [
                (
                    "ZEROIFNULL(AVG(DECODE(predict_neighbors, '{0}', "
                    "proba_predict, NULL))) AS {1}"
                ).format(elem, gen_name([name, elem]))
                for elem in self.classes_
            ]
        else:
            predict = [
                (
                    "ZEROIFNULL(AVG(DECODE(predict_neighbors, '{0}', "
                    "proba_predict, NULL))) AS {1}"
                ).format(pos_label, name)
            ]
        sql = "(SELECT {0}{1}, {2} FROM {3} GROUP BY {4}) VERTICAPY_SUBTABLE".format(
            ", ".join(X),
            ", " + ", ".join(key_columns) if key_columns else "",
            ", ".join(predict),
            self.deploySQL(
                X=X, test_relation=vdf.__genSQL__(), key_columns=key_columns_arg
            ),
            ", ".join(X + key_columns),
        )

        # Result
        if inplace:
            return vDataFrameSQL(name="Neighbors", relation=sql, vdf=vdf)
        else:
            return vDataFrameSQL(name="Neighbors", relation=sql)

    # ---#
    def roc_curve(
        self, pos_label: Union[int, float, str] = None, ax=None, **style_kwds
    ):
        """
    ---------------------------------------------------------------------------
    Draws the model ROC curve.

    Parameters
    ----------
    pos_label: int/float/str
        To draw the ROC curve, one of the response column classes must be the 
        positive one. The parameter 'pos_label' represents this class.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.
        """
        pos_label = (
            self.classes_[1]
            if (pos_label == None and len(self.classes_) == 2)
            else pos_label
        )
        if pos_label not in self.classes_:
            raise ParameterError(
                "'pos_label' must be one of the response column classes"
            )
        input_relation = self.deploySQL() + " WHERE predict_neighbors = '{}'".format(
            pos_label
        )
        return roc_curve(
            self.y, "proba_predict", input_relation, pos_label, ax=ax, **style_kwds,
        )

    # ---#
    def score(
        self,
        method: str = "accuracy",
        pos_label: Union[int, float, str] = None,
        cutoff: float = -1,
        nbins: int = 10000,
    ):
        """
    ---------------------------------------------------------------------------
    Computes the model score.

    Parameters
    ----------
    pos_label: int/float/str, optional
        Label to consider as positive. All the other classes will be merged and
        considered as negative for multiclass classification.
    cutoff: float, optional
        Cutoff for which the tested category will be accepted as a prediction. 
    method: str, optional
        The method to use to compute the score.
            accuracy    : Accuracy
            auc         : Area Under the Curve (ROC)
            best_cutoff : Cutoff which optimised the ROC Curve prediction.
            bm          : Informedness = tpr + tnr - 1
            csi         : Critical Success Index = tp / (tp + fn + fp)
            f1          : F1 Score 
            logloss     : Log Loss
            mcc         : Matthews Correlation Coefficient 
            mk          : Markedness = ppv + npv - 1
            npv         : Negative Predictive Value = tn / (tn + fn)
            prc_auc     : Area Under the Curve (PRC)
            precision   : Precision = tp / (tp + fp)
            recall      : Recall = tp / (tp + fn)
            specificity : Specificity = tn / (tn + fp)
    nbins: int, optional
        [Only when method is set to auc|prc_auc|best_cutoff] 
        An integer value that determines the number of decision boundaries. 
        Decision boundaries are set at equally spaced intervals between 0 and 1, 
        inclusive. Greater values for nbins give more precise estimations of the AUC, 
        but can potentially decrease performance. The maximum value is 999,999. 
        If negative, the maximum value is used.

    Returns
    -------
    float
        score
        """
        check_types(
            [
                ("cutoff", cutoff, [int, float]),
                ("method", method, [str]),
                ("nbins", nbins, [int]),
            ]
        )
        if pos_label == None and len(self.classes_) == 2:
            pos_label = self.classes_[1]
        input_relation = "(SELECT * FROM {0} WHERE predict_neighbors = '{1}') final_centroids_relation".format(
            self.deploySQL(), pos_label
        )
        y_score = "(CASE WHEN proba_predict > {} THEN 1 ELSE 0 END)".format(cutoff)
        y_proba = "proba_predict"
        y_true = "DECODE({}, '{}', 1, 0)".format(self.y, pos_label)
        if (pos_label not in self.classes_) and (method != "accuracy"):
            raise ParameterError(
                "'pos_label' must be one of the response column classes"
            )
        elif (cutoff >= 1 or cutoff <= 0) and (method != "accuracy"):
            cutoff = self.score(pos_label=pos_label, cutoff=0.5, method="best_cutoff")
        if method in ("accuracy", "acc"):
            if pos_label not in self.classes_:
                return accuracy_score(
                    self.y,
                    "predict_neighbors",
                    self.deploySQL(predict=True),
                    pos_label=None,
                )
            else:
                return accuracy_score(
                    y_true, y_score, input_relation, pos_label=pos_label
                )
        elif method == "auc":
            return auc(y_true, y_proba, input_relation, nbins=nbins)
        elif method == "prc_auc":
            return prc_auc(y_true, y_proba, input_relation, nbins=nbins)
        elif method in ("best_cutoff", "best_threshold"):
            return roc_curve(
                y_true, y_proba, input_relation, best_threshold=True, nbins=nbins
            )
        elif method in ("recall", "tpr"):
            return recall_score(y_true, y_score, input_relation)
        elif method in ("precision", "ppv"):
            return precision_score(y_true, y_score, input_relation)
        elif method in ("specificity", "tnr"):
            return specificity_score(y_true, y_score, input_relation)
        elif method in ("negative_predictive_value", "npv"):
            return precision_score(y_true, y_score, input_relation)
        elif method in ("log_loss", "logloss"):
            return log_loss(y_true, y_proba, input_relation)
        elif method == "f1":
            return f1_score(y_true, y_score, input_relation)
        elif method == "mcc":
            return matthews_corrcoef(y_true, y_score, input_relation)
        elif method in ("bm", "informedness"):
            return informedness(y_true, y_score, input_relation)
        elif method in ("mk", "markedness"):
            return markedness(y_true, y_score, input_relation)
        elif method in ("csi", "critical_success_index"):
            return critical_success_index(y_true, y_score, input_relation)
        else:
            raise ParameterError(
                "The parameter 'method' must be in accuracy|auc|prc_auc|best_cutoff|"
                "recall|precision|log_loss|negative_predictive_value|specificity|"
                "mcc|informedness|markedness|critical_success_index"
            )


# ---#
class KernelDensity(Regressor, Tree):
    """
---------------------------------------------------------------------------
[Beta Version]
Creates a KernelDensity object. 
This object uses pure SQL to compute the final score.

Parameters
----------
bandwidth: float, optional
    The bandwidth of the kernel.
kernel: str, optional
    The kernel used during the learning phase.
        gaussian  : Gaussian Kernel.
        logistic  : Logistic Kernel.
        sigmoid   : Sigmoid Kernel.
        silverman : Silverman Kernel.
p: int, optional
    The p corresponding to the one of the p-distances (distance metric used during 
    the model computation).
max_leaf_nodes: int, optional
    The maximum number of leaf nodes, an integer between 1 and 1e9, inclusive.
max_depth: int, optional
    The maximum tree depth, an integer between 1 and 100, inclusive.
min_samples_leaf: int, optional
    The minimum number of samples each branch must have after splitting a node, an 
    integer between 1 and 1e6, inclusive. A split that causes fewer remaining samples 
    is discarded.
nbins: int, optional 
    The number of bins to use to discretize the input features.
xlim: list, optional
    List of tuples use to compute the kernel window.
    """

    def __init__(
        self,
        name: str,
        bandwidth: float = 1,
        kernel: str = "gaussian",
        p: int = 2,
        max_leaf_nodes: int = 1e9,
        max_depth: int = 5,
        min_samples_leaf: int = 1,
        nbins: int = 5,
        xlim: list = [],
        **kwargs,
    ):
        check_types(
            [
                ("name", name, [str], False),
                ("bandwidth", bandwidth, [int, float], False),
                ("kernel", kernel, ["gaussian", "logistic", "sigmoid", "silverman"]),
                ("max_leaf_nodes", max_leaf_nodes, [int, float], False),
                ("max_depth", max_depth, [int, float], False),
                ("min_samples_leaf", min_samples_leaf, [int, float], False),
                ("nbins", nbins, [int, float], False),
                ("xlim", xlim, [list], False),
            ]
        )
        self.type, self.name = "KernelDensity", name
        self.set_params(
            {
                "nbins": nbins,
                "p": p,
                "bandwidth": bandwidth,
                "kernel": kernel,
                "max_leaf_nodes": int(max_leaf_nodes),
                "max_depth": int(max_depth),
                "min_samples_leaf": int(min_samples_leaf),
                "xlim": xlim,
            }
        )
        if "store" not in kwargs or kwargs["store"]:
            self.verticapy_store = True
        else:
            self.verticapy_store = False

    # ---#
    def fit(self, input_relation: Union[str, vDataFrame], X: list = []):
        """
    ---------------------------------------------------------------------------
    Trains the model.

    Parameters
    ----------
    input_relation: str/vDataFrame
        Training relation.
    X: list, optional
        List of the predictors.

    Returns
    -------
    object
        self
        """
        if isinstance(X, str):
            X = [X]
        check_types(
            [("input_relation", input_relation, [str, vDataFrame]), ("X", X, [list])]
        )
        if verticapy.options["overwrite_model"]:
            self.drop()
        else:
            does_model_exist(name=self.name, raise_error=True)
        if isinstance(input_relation, vDataFrame):
            if not (X):
                X = input_relation.numcol()
            vdf = input_relation
            input_relation = input_relation.__genSQL__()
        else:
            try:
                vdf = vDataFrame(input_relation)
            except:
                vdf = vDataFrameSQL(input_relation)
            if not (X):
                X = vdf.numcol()
        vdf.are_namecols_in(X)
        X = vdf.format_colnames(X)

        # ---#
        def density_compute(
            vdf: vDataFrame,
            columns: list,
            h=None,
            kernel: str = "gaussian",
            nbins: int = 5,
            p: int = 2,
        ):
            # ---#
            def density_kde(vdf, columns: list, kernel: str, x, p: int, h=None):
                for elem in columns:
                    if not (vdf[elem].isnum()):
                        raise TypeError(
                            f"Cannot compute KDE for non-numerical columns. {elem} is not numerical."
                        )
                if kernel == "gaussian":
                    fkernel = "EXP(-1 / 2 * POWER({0}, 2)) / SQRT(2 * PI())"

                elif kernel == "logistic":
                    fkernel = "1 / (2 + EXP({0}) + EXP(-{0}))"

                elif kernel == "sigmoid":
                    fkernel = "2 / (PI() * (EXP({0}) + EXP(-{0})))"

                elif kernel == "silverman":
                    fkernel = "EXP(-1 / SQRT(2) * ABS({0})) / 2 * SIN(ABS({0}) / SQRT(2) + PI() / 4)"

                else:
                    raise ParameterError(
                        "The parameter 'kernel' must be in [gaussian|logistic|sigmoid|silverman]."
                    )
                if isinstance(x, (tuple)):
                    return density_kde(vdf, columns, kernel, [x], p, h)[0]
                elif isinstance(x, (list)):
                    N = vdf.shape()[0]
                    L = []
                    for elem in x:
                        distance = []
                        for i in range(len(columns)):
                            distance += [
                                "POWER({0} - {1}, {2})".format(columns[i], elem[i], p)
                            ]
                        distance = " + ".join(distance)
                        distance = "POWER({0}, {1})".format(distance, 1.0 / p)
                        fkernel_tmp = fkernel.format(f"{distance} / {h}")
                        L += [f"SUM({fkernel_tmp}) / ({h} * {N})"]
                    query = "SELECT {0} FROM {1}".format(", ".join(L), vdf.__genSQL__())
                    result = executeSQL(
                        query, title="Computing the KDE", method="fetchrow"
                    )
                    return [elem for elem in result]
                else:
                    return 0

            vdf.are_namecols_in(columns)
            columns = vdf.format_colnames(columns)
            x_vars = []
            y = []
            for idx, column in enumerate(columns):
                if self.parameters["xlim"]:
                    try:
                        x_min, x_max = self.parameters["xlim"][idx]
                        N = vdf[column].count()
                    except:
                        warning_message = (
                            f"Wrong xlim for the vcolumn {column}.\n"
                            "The max and the min will be used instead."
                        )
                        warnings.warn(warning_message, Warning)
                        x_min, x_max, N = vdf.agg(
                            func=["min", "max", "count"], columns=[column]
                        ).transpose()[column]
                else:
                    x_min, x_max, N = vdf.agg(
                        func=["min", "max", "count"], columns=[column]
                    ).transpose()[column]
                x_vars += [
                    [(x_max - x_min) * i / nbins + x_min for i in range(0, nbins + 1)]
                ]
            import itertools

            x = list(itertools.product(*x_vars))
            try:
                y = density_kde(vdf, columns, kernel, x, p, h)
            except:
                for xi in x:
                    K = density_kde(vdf, columns, kernel, xi, p, h)
                    y += [K]
            return [x, y]

        x, y = density_compute(
            vdf,
            X,
            self.parameters["bandwidth"],
            self.parameters["kernel"],
            self.parameters["nbins"],
            self.parameters["p"],
        )
        if self.verticapy_store:
            query = """CREATE TABLE {0}_KernelDensity_Map AS    
                            SELECT 
                                {1}, 0.0::float AS KDE 
                            FROM {2} 
                            LIMIT 0""".format(
                self.name.replace('"', ""), ", ".join(X), vdf.__genSQL__()
            )
            executeSQL(query, print_time_sql=False)
            r, idx = 0, 0
            while r < len(y):
                values = []
                m = min(r + 100, len(y))
                for i in range(r, m):
                    values += ["SELECT " + str(x[i] + (y[i],))[1:-1]]
                query = "INSERT INTO {0}_KernelDensity_Map ({1}, KDE) {2}".format(
                    self.name.replace('"', ""), ", ".join(X), " UNION ".join(values)
                )
                executeSQL(query, f"Computing the KDE [Step {idx}].")
                executeSQL("COMMIT;", print_time_sql=False)
                r += 100
                idx += 1
            self.X, self.input_relation = X, input_relation
            self.map = "{0}_KernelDensity_Map".format(self.name.replace('"', ""))
            self.tree_name = "{0}_KernelDensity_Tree".format(self.name.replace('"', ""))
            self.y = "KDE"

            from verticapy.learn.tree import DecisionTreeRegressor

            model = DecisionTreeRegressor(
                name=self.tree_name,
                max_features=len(self.X),
                max_leaf_nodes=self.parameters["max_leaf_nodes"],
                max_depth=self.parameters["max_depth"],
                min_samples_leaf=self.parameters["min_samples_leaf"],
                nbins=1000,
            )
            model.fit(self.map, self.X, "KDE")
            model_save = {
                "type": "KernelDensity",
                "input_relation": self.input_relation,
                "X": self.X,
                "map": self.map,
                "tree_name": self.tree_name,
                "bandwidth": self.parameters["bandwidth"],
                "kernel": self.parameters["kernel"],
                "p": self.parameters["p"],
                "max_leaf_nodes": self.parameters["max_leaf_nodes"],
                "max_depth": self.parameters["max_depth"],
                "min_samples_leaf": self.parameters["min_samples_leaf"],
                "nbins": self.parameters["nbins"],
                "xlim": self.parameters["xlim"],
            }
            insert_verticapy_schema(
                model_name=self.name, model_type="KernelDensity", model_save=model_save
            )
        else:
            self.X, self.input_relation = X, input_relation
            self.verticapy_x = x
            self.verticapy_y = y
        return self

    # ---#
    def plot(self, ax=None, **style_kwds):
        """
    ---------------------------------------------------------------------------
    Draws the Model.

    Parameters
    ----------
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object
        """
        if len(self.X) == 1:
            if self.verticapy_store:
                query = "SELECT {}, KDE FROM {} ORDER BY 1".format(self.X[0], self.map)
                result = executeSQL(query, method="fetchall", print_time_sql=False)
                x, y = [elem[0] for elem in result], [elem[1] for elem in result]
            else:
                x, y = [elem[0] for elem in self.verticapy_x], self.verticapy_y
            if not (ax):
                fig, ax = plt.subplots()
                if isnotebook():
                    fig.set_size_inches(7, 5)
                ax.grid()
                ax.set_axisbelow(True)
            from verticapy.plot import gen_colors

            param = {
                "color": gen_colors()[0],
            }
            ax.plot(x, y, **updated_dict(param, style_kwds))
            ax.fill_between(
                x, y, facecolor=updated_dict(param, style_kwds)["color"], alpha=0.7
            )
            ax.set_xlim(min(x), max(x))
            ax.set_ylim(bottom=0)
            ax.set_ylabel("density")
            return ax
        elif len(self.X) == 2:
            n = self.parameters["nbins"]
            if self.verticapy_store:
                query = "SELECT {}, {}, KDE FROM {} ORDER BY 1, 2".format(
                    self.X[0], self.X[1], self.map,
                )
                result = executeSQL(query, method="fetchall", print_time_sql=False)
                x, y, z = (
                    [elem[0] for elem in result],
                    [elem[1] for elem in result],
                    [elem[2] for elem in result],
                )
            else:
                x, y, z = (
                    [elem[0] for elem in self.verticapy_x],
                    [elem[1] for elem in self.verticapy_x],
                    self.verticapy_y,
                )
            result, idx = [], 0
            while idx < (n + 1) * (n + 1):
                result += [[z[idx + i] for i in range(n + 1)]]
                idx += n + 1
            if not (ax):
                fig, ax = plt.subplots()
                if isnotebook():
                    fig.set_size_inches(8, 6)
            else:
                fig = plt
            param = {
                "cmap": "Reds",
                "origin": "lower",
                "interpolation": "bilinear",
            }
            extent = [min(x), max(x), min(y), max(y)]
            extent = [float(elem) for elem in extent]
            im = ax.imshow(result, extent=extent, **updated_dict(param, style_kwds))
            fig.colorbar(im, ax=ax)
            ax.set_ylabel(self.X[1])
            ax.set_xlabel(self.X[0])
            return ax
        else:
            raise Exception("KDE Plots are only available in 1D or 2D.")


# ---#
class KNeighborsRegressor(Regressor):
    """
---------------------------------------------------------------------------
[Beta Version]
Creates a KNeighborsRegressor object using the k-nearest neighbors 
algorithm. This object uses pure SQL to compute all the distances and 
final score.

\u26A0 Warning : This algorithm uses a CROSS JOIN during computation and
                 is therefore computationally expensive at O(n * n), where
                 n is the total number of elements. Since KNeighborsRegressor 
                 uses the p-distance, it is highly sensitive to unnormalized 
                 data.

Parameters
----------
n_neighbors: int, optional
	Number of neighbors to consider when computing the score.
p: int, optional
	The p corresponding to the one of the p-distances (distance metric used during 
	the model computation).
	"""

    def __init__(self, name: str, n_neighbors: int = 5, p: int = 2):
        check_types([("name", name, [str], False)])
        self.type, self.name = "KNeighborsRegressor", name
        self.set_params({"n_neighbors": n_neighbors, "p": p})

    # ---#
    def deploySQL(self, X: list = [], test_relation: str = "", key_columns: list = []):
        """
    ---------------------------------------------------------------------------
    Returns the SQL code needed to deploy the model. 

    Parameters
    ----------
    X: list
        List of the predictors.
    test_relation: str, optional
        Relation to use to do the predictions.
    key_columns: list, optional
        A list of columns to include in the results, but to exclude from 
        computation of the prediction.

    Returns
    -------
    str/list
        the SQL code needed to deploy the model.
        """
        if isinstance(X, str):
            X = [X]
        if isinstance(key_columns, str):
            key_columns = [key_columns]
        check_types(
            [
                ("test_relation", test_relation, [str], False),
                ("X", X, [list], False),
                ("key_columns", key_columns, [list], False),
            ],
        )
        X = [quote_ident(elem) for elem in X] if (X) else self.X
        if not (test_relation):
            test_relation = self.test_relation
        if not (key_columns) and key_columns != None:
            key_columns = [self.y]
        sql = [
            "POWER(ABS(x.{} - y.{}), {})".format(X[i], self.X[i], self.parameters["p"])
            for i in range(len(self.X))
        ]
        sql = "POWER({}, 1 / {})".format(" + ".join(sql), self.parameters["p"])
        sql = "ROW_NUMBER() OVER(PARTITION BY {}, row_id ORDER BY {})".format(
            ", ".join(["x.{}".format(item) for item in X]), sql
        )
        sql = "SELECT {}{}, {} AS ordered_distance, y.{} AS predict_neighbors, row_id FROM (SELECT *, ROW_NUMBER() OVER() AS row_id FROM {} WHERE {}) x CROSS JOIN (SELECT * FROM {} WHERE {}) y".format(
            ", ".join(["x.{}".format(item) for item in X]),
            ", " + ", ".join(["x." + quote_ident(elem) for elem in key_columns])
            if (key_columns)
            else "",
            sql,
            self.y,
            test_relation,
            " AND ".join(["{} IS NOT NULL".format(item) for item in X]),
            self.input_relation,
            " AND ".join(["{} IS NOT NULL".format(item) for item in self.X]),
        )
        sql = "(SELECT {}{}, AVG(predict_neighbors) AS predict_neighbors FROM ({}) z WHERE ordered_distance <= {} GROUP BY {}{}, row_id) knr_table".format(
            ", ".join(X),
            ", " + ", ".join([quote_ident(elem) for elem in key_columns])
            if (key_columns)
            else "",
            sql,
            self.parameters["n_neighbors"],
            ", ".join(X),
            ", " + ", ".join([quote_ident(elem) for elem in key_columns])
            if (key_columns)
            else "",
        )
        return sql

    # ---#
    def fit(
        self,
        input_relation: Union[str, vDataFrame],
        X: list,
        y: str,
        test_relation: Union[str, vDataFrame] = "",
    ):
        """
	---------------------------------------------------------------------------
	Trains the model.

	Parameters
	----------
	input_relation: str/vDataFrame
		Training relation.
	X: list
		List of the predictors.
	y: str
		Response column.
	test_relation: str/vDataFrame, optional
		Relation used to test the model.

	Returns
	-------
	object
 		self
		"""
        if isinstance(X, str):
            X = [X]
        check_types(
            [
                ("input_relation", input_relation, [str, vDataFrame], False),
                ("X", X, [list], False),
                ("y", y, [str], False),
                ("test_relation", test_relation, [str, vDataFrame], False),
            ]
        )
        if verticapy.options["overwrite_model"]:
            self.drop()
        else:
            does_model_exist(name=self.name, raise_error=True)
        if isinstance(input_relation, vDataFrame):
            self.input_relation = input_relation.__genSQL__()
        else:
            self.input_relation = input_relation
        if isinstance(test_relation, vDataFrame):
            self.test_relation = test_relation.__genSQL__()
        elif test_relation:
            self.test_relation = test_relation
        else:
            self.test_relation = self.input_relation
        self.X = [quote_ident(column) for column in X]
        self.y = quote_ident(y)
        model_save = {
            "type": "KNeighborsRegressor",
            "input_relation": self.input_relation,
            "test_relation": self.test_relation,
            "X": self.X,
            "y": self.y,
            "p": self.parameters["p"],
            "n_neighbors": self.parameters["n_neighbors"],
        }
        insert_verticapy_schema(
            model_name=self.name,
            model_type="KNeighborsRegressor",
            model_save=model_save,
        )
        return self

    # ---#
    def predict(
        self,
        vdf: Union[str, vDataFrame],
        X: list = [],
        name: str = "",
        inplace: bool = True,
        **kwargs,
    ):
        """
    ---------------------------------------------------------------------------
    Predicts using the input relation.

    Parameters
    ----------
    vdf: str/vDataFrame
        Object to use to run the prediction. You can also specify a customized 
        relation, but you must enclose it with an alias. For example "(SELECT 1) x" 
        is correct whereas "(SELECT 1)" and "SELECT 1" are incorrect.
    X: list, optional
        List of the columns used to deploy the models. If empty, the model
        predictors will be used.
    name: str, optional
        Name of the added vcolumn. If empty, a name will be generated.
    inplace: bool, optional
        If set to True, the prediction will be added to the vDataFrame.

    Returns
    -------
    vDataFrame
        the vDataFrame of the prediction
        """
        if isinstance(X, str):
            X = [X]
        check_types(
            [
                ("name", name, [str]),
                ("X", X, [list]),
                ("vdf", vdf, [str, vDataFrame]),
                ("inplace", inplace, [bool]),
            ],
        )
        if isinstance(vdf, str):
            vdf = vDataFrameSQL(vdf)
        X = [quote_ident(elem) for elem in X] if (X) else self.X
        key_columns = vdf.get_columns(exclude_columns=X)
        if "key_columns" in kwargs:
            key_columns_arg = None
        else:
            key_columns_arg = key_columns
        name = (
            "{}_".format(self.type) + "".join(ch for ch in self.name if ch.isalnum())
            if not (name)
            else name
        )
        sql = "(SELECT {}{}, {} AS {} FROM {}) VERTICAPY_SUBTABLE".format(
            ", ".join(X),
            ", " + ", ".join(key_columns) if key_columns else "",
            "predict_neighbors",
            name,
            self.deploySQL(
                X=X, test_relation=vdf.__genSQL__(), key_columns=key_columns_arg
            ),
        )
        if inplace:
            return vDataFrameSQL(name="Neighbors", relation=sql, vdf=vdf)
        else:
            return vDataFrameSQL(name="Neighbors", relation=sql)


# ---#
class LocalOutlierFactor(vModel):
    """
---------------------------------------------------------------------------
[Beta Version]
Creates a LocalOutlierFactor object by using the Local Outlier Factor algorithm 
as defined by Markus M. Breunig, Hans-Peter Kriegel, Raymond T. Ng and Jörg 
Sander. This object is using pure SQL to compute all the distances and final 
score.

\u26A0 Warning : This algorithm uses a CROSS JOIN during computation and
                 is therefore computationally expensive at O(n * n), where
                 n is the total number of elementss. Since LocalOutlierFactor 
                 is uses the p-distance, it is highly sensitive to unnormalized 
                 data. A table will be created at the end of the learning phase.

Parameters
----------
name: str
	Name of the the model. This is not a built-in model, so this name will be used
	to build the final table.
n_neighbors: int, optional
	Number of neighbors to consider when computing the score.
p: int, optional
	The p of the p-distances (distance metric used during the model computation).
	"""

    def __init__(self, name: str, n_neighbors: int = 20, p: int = 2):
        check_types([("name", name, [str], False)])
        self.type, self.name = "LocalOutlierFactor", name
        self.set_params({"n_neighbors": n_neighbors, "p": p})

    # ---#
    def fit(
        self,
        input_relation: Union[str, vDataFrame],
        X: list = [],
        key_columns: list = [],
        index: str = "",
    ):
        """
	---------------------------------------------------------------------------
	Trains the model.

	Parameters
	----------
	input_relation: str/vDataFrame
		Training relation.
	X: list, optional
		List of the predictors.
	key_columns: list, optional
		Columns not used during the algorithm computation but which will be used
		to create the final relation.
	index: str, optional
		Index used to identify each row separately. It is highly recommanded to
        have one already in the main table to avoid creating temporary tables.

	Returns
	-------
	object
 		self
		"""
        if isinstance(X, str):
            X = [X]
        if isinstance(key_columns, str):
            key_columns = [key_columns]
        check_types(
            [
                ("input_relation", input_relation, [str, vDataFrame], False),
                ("X", X, [list], False),
                ("key_columns", key_columns, [list], False),
                ("index", index, [str], False),
            ]
        )
        if verticapy.options["overwrite_model"]:
            self.drop()
        else:
            does_model_exist(name=self.name, raise_error=True)
        self.key_columns = [quote_ident(column) for column in key_columns]
        if isinstance(input_relation, vDataFrame):
            self.input_relation = input_relation.__genSQL__()
            if not (X):
                X = input_relation.numcol()
        else:
            self.input_relation = input_relation
            if not (X):
                X = vDataFrame(input_relation).numcol()
        X = [quote_ident(column) for column in X]
        self.X = X
        n_neighbors = self.parameters["n_neighbors"]
        p = self.parameters["p"]
        schema, relation = schema_relation(input_relation)
        name_list = [
            gen_tmp_name(name="main"),
            gen_tmp_name(name="distance"),
            gen_tmp_name(name="lrd"),
            gen_tmp_name(name="lof"),
        ]
        tmp_main_table_name = gen_tmp_name(name="main")
        tmp_distance_table_name = gen_tmp_name(name="distance")
        tmp_lrd_table_name = gen_tmp_name(name="lrd")
        tmp_lof_table_name = gen_tmp_name(name="lof")
        try:
            if not (index):
                index = "id"
                main_table = tmp_main_table_name
                schema = "v_temp_schema"
                sql = "CREATE LOCAL TEMPORARY TABLE {} ON COMMIT PRESERVE ROWS AS SELECT ROW_NUMBER() OVER() AS id, {} FROM {} WHERE {}".format(
                    main_table,
                    ", ".join(X + key_columns),
                    self.input_relation,
                    " AND ".join(["{} IS NOT NULL".format(item) for item in X]),
                )
                drop("v_temp_schema.{}".format(tmp_main_table_name), method="table")
                executeSQL(sql, print_time_sql=False)
            else:
                main_table = self.input_relation
            sql = [
                "POWER(ABS(x.{} - y.{}), {})".format(X[i], X[i], p)
                for i in range(len(X))
            ]
            distance = "POWER({}, 1 / {})".format(" + ".join(sql), p)
            sql = "SELECT x.{} AS node_id, y.{} AS nn_id, {} AS distance, ROW_NUMBER() OVER(PARTITION BY x.{} ORDER BY {}) AS knn FROM {}.{} AS x CROSS JOIN {}.{} AS y".format(
                index,
                index,
                distance,
                index,
                distance,
                schema,
                main_table,
                schema,
                main_table,
            )
            sql = "SELECT node_id, nn_id, distance, knn FROM ({}) distance_table WHERE knn <= {}".format(
                sql, n_neighbors + 1
            )
            sql = "CREATE LOCAL TEMPORARY TABLE {} ON COMMIT PRESERVE ROWS AS {}".format(
                tmp_distance_table_name, sql
            )
            drop("v_temp_schema.{}".format(tmp_distance_table_name), method="table")
            executeSQL(sql, "Computing the LOF [Step 0].")
            kdistance = "(SELECT node_id, nn_id, distance AS distance FROM v_temp_schema.{} WHERE knn = {}) AS kdistance_table".format(
                tmp_distance_table_name, n_neighbors + 1
            )
            lrd = "SELECT distance_table.node_id, {} / SUM(CASE WHEN distance_table.distance > kdistance_table.distance THEN distance_table.distance ELSE kdistance_table.distance END) AS lrd FROM (v_temp_schema.{} AS distance_table LEFT JOIN {} ON distance_table.nn_id = kdistance_table.node_id) x GROUP BY 1".format(
                n_neighbors, tmp_distance_table_name, kdistance
            )
            sql = "CREATE LOCAL TEMPORARY TABLE {} ON COMMIT PRESERVE ROWS AS {}".format(
                tmp_lrd_table_name, lrd
            )
            drop("v_temp_schema.{}".format(tmp_lrd_table_name), method="table")
            executeSQL(sql, "Computing the LOF [Step 1].")
            sql = "SELECT x.node_id, SUM(y.lrd) / (MAX(x.node_lrd) * {}) AS LOF FROM (SELECT n_table.node_id, n_table.nn_id, lrd_table.lrd AS node_lrd FROM v_temp_schema.{} AS n_table LEFT JOIN v_temp_schema.{} AS lrd_table ON n_table.node_id = lrd_table.node_id) x LEFT JOIN v_temp_schema.{} AS y ON x.nn_id = y.node_id GROUP BY 1".format(
                n_neighbors,
                tmp_distance_table_name,
                tmp_lrd_table_name,
                tmp_lrd_table_name,
            )
            sql = "CREATE LOCAL TEMPORARY TABLE {} ON COMMIT PRESERVE ROWS AS {}".format(
                tmp_lof_table_name, sql
            )
            drop("v_temp_schema.{}".format(tmp_lof_table_name), method="table")
            executeSQL(sql, "Computing the LOF [Step 2].")
            sql = "SELECT {}, (CASE WHEN lof > 1e100 OR lof != lof THEN 0 ELSE lof END) AS lof_score FROM {} AS x LEFT JOIN v_temp_schema.{} AS y ON x.{} = y.node_id".format(
                ", ".join(X + self.key_columns), main_table, tmp_lof_table_name, index
            )
            executeSQL(
                "CREATE TABLE {} AS {}".format(self.name, sql),
                title="Computing the LOF [Step 3].",
            )
            self.n_errors_ = executeSQL(
                "SELECT COUNT(*) FROM {}.{} z WHERE lof > 1e100 OR lof != lof".format(
                    schema, tmp_lof_table_name
                ),
                method="fetchfirstelem",
                print_time_sql=False,
            )
        except:
            drop("v_temp_schema.{}".format(tmp_main_table_name), method="table")
            drop("v_temp_schema.{}".format(tmp_distance_table_name), method="table")
            drop("v_temp_schema.{}".format(tmp_lrd_table_name), method="table")
            drop("v_temp_schema.{}".format(tmp_lof_table_name), method="table")
            raise
        drop("v_temp_schema.{}".format(tmp_main_table_name), method="table")
        drop("v_temp_schema.{}".format(tmp_distance_table_name), method="table")
        drop("v_temp_schema.{}".format(tmp_lrd_table_name), method="table")
        drop("v_temp_schema.{}".format(tmp_lof_table_name), method="table")
        model_save = {
            "type": "LocalOutlierFactor",
            "input_relation": self.input_relation,
            "key_columns": self.key_columns,
            "X": self.X,
            "p": self.parameters["p"],
            "n_neighbors": self.parameters["n_neighbors"],
            "n_errors": self.n_errors_,
        }
        insert_verticapy_schema(
            model_name=self.name,
            model_type="LocalOutlierFactor",
            model_save=model_save,
        )
        return self

    # ---#
    def predict(self):
        """
	---------------------------------------------------------------------------
	Creates a vDataFrame of the model.

	Returns
	-------
	vDataFrame
 		the vDataFrame including the prediction.
		"""
        return vDataFrame(self.name)
