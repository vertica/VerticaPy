# (c) Copyright [2018-2020] Micro Focus or one of its affiliates.
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
# VerticaPy is a Python library with scikit-like functionality to use to conduct
# data science projects on data stored in Vertica, taking advantage Vertica’s
# speed and built-in analytics and machine learning features. It supports the
# entire data science life cycle, uses a ‘pipeline’ mechanism to sequentialize
# data transformation operations, and offers beautiful graphical options.
#
# VerticaPy aims to solve all of these problems. The idea is simple: instead
# of moving data around for processing, VerticaPy brings the logic to the data.
#
#
# Modules
#
# VerticaPy Modules
from verticapy.learn.metrics import *
from verticapy.learn.plot import *
from verticapy.utilities import *
from verticapy.toolbox import *
from verticapy import vDataFrame
from verticapy.learn.plot import lof_plot
from verticapy.connections.connect import read_auto_connect
from verticapy.errors import *
from verticapy.learn.vmodel import *

# ---#
class NeighborsClassifier(vModel):

    # ---#
    def classification_report(self, cutoff=[], labels: list = []):
        """
    ---------------------------------------------------------------------------
    Computes a classification report using multiple metrics to evaluate the model
    (AUC, accuracy, PRC AUC, F1...). In case of multiclass classification, it will 
    consider each category as positive and switch to the next one during the computation.

    Parameters
    ----------
    cutoff: float/list, optional
        Cutoff for which the tested category will be accepted as prediction. 
        In case of multiclass classification, each tested category becomes 
        the positives and the others are merged into the negatives. The list will 
        represent the classes threshold. If it is empty, the best cutoff will be used.
    labels: list, optional
        List of the different labels to be used during the computation.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.
        """
        check_types(
            [("cutoff", cutoff, [int, float, list],), ("labels", labels, [list],),]
        )
        if not (labels):
            labels = self.classes_
        return classification_report(cutoff=cutoff, estimator=self, labels=labels)

    # ---#
    def confusion_matrix(self, pos_label=None, cutoff: float = -1):
        """
    ---------------------------------------------------------------------------
    Computes the model confusion matrix.

    Parameters
    ----------
    pos_label: int/float/str, optional
        Label to consider as positive. All the other classes will be merged and
        considered as negative in case of multi classification.
    cutoff: float, optional
        Cutoff for which the tested category will be accepted as prediction. If the 
        cutoff is not between 0 and 1, the entire confusion matrix will be drawn.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.
        """
        check_types([("cutoff", cutoff, [int, float],)])
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
            result = confusion_matrix(y_true, y_score, input_relation, self.cursor)
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
                self.y, "predict_neighbors", input_relation, self.classes_, self.cursor
            )

    # ---#
    def lift_chart(self, pos_label=None):
        """
    ---------------------------------------------------------------------------
    Draws the model Lift Chart.

    Parameters
    ----------
    pos_label: int/float/str
        To draw a lift chart, one of the response column class has to be the 
        positive one. The parameter 'pos_label' represents this class.

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
            self.y, "proba_predict", input_relation, self.cursor, pos_label
        )

    # ---#
    def prc_curve(self, pos_label=None):
        """
    ---------------------------------------------------------------------------
    Draws the model PRC curve.

    Parameters
    ----------
    pos_label: int/float/str
        To draw the PRC curve, one of the response column class has to be the 
        positive one. The parameter 'pos_label' represents this class.

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
            self.y, "proba_predict", input_relation, self.cursor, pos_label
        )

    # ---#
    def predict(
        self,
        vdf,
        X: list = [],
        name: str = "",
        cutoff: float = -1,
        all_classes: bool = False,
    ):
        """
    ---------------------------------------------------------------------------
    Predicts using the input relation.

    Parameters
    ----------
    vdf: vDataFrame
        Object to use to run the prediction.
    X: list, optional
        List of the columns used to deploy the models. If empty, the model
        predictors will be used.
    name: str, optional
        Name of the added vcolumn. If empty, a name will be generated.
    cutoff: float, optional
        Cutoff used in case of binary classification. It is the probability to
        accept the category 1.
    all_classes: bool, optional
        If set to True, all the classes probabilities will be generated (one 
        column per category).

    Returns
    -------
    vDataFrame
        the vDataFrame of the prediction
        """
        check_types(
            [
                ("cutoff", cutoff, [int, float],),
                ("all_classes", all_classes, [bool],),
                ("name", name, [str],),
                ("cutoff", cutoff, [int, float],),
                ("X", X, [list],),
                ("vdf", vdf, [vDataFrame],),
            ],
        )
        X = [str_column(elem) for elem in X] if (X) else self.X
        key_columns = vdf.get_columns(exclude_columns=X)
        name = (
            "{}_".format(self.type) + "".join(ch for ch in self.name if ch.isalnum())
            if not (name)
            else name
        )
        if all_classes:
            predict = [
                "ZEROIFNULL(AVG(DECODE(predict_neighbors, '{}', proba_predict, NULL))) AS \"{}_{}\"".format(
                    elem, name, elem
                )
                for elem in self.classes_
            ]
            sql = "SELECT {}{}, {} FROM {} GROUP BY {}".format(
                ", ".join(X),
                ", " + ", ".join(key_columns) if key_columns else "",
                ", ".join(predict),
                self.deploySQL(
                    X=X, test_relation=vdf.__genSQL__(), key_columns=key_columns
                ),
                ", ".join(X + key_columns),
            )
        else:
            if (len(self.classes_) == 2) and (cutoff <= 1 and cutoff >= 0):
                sql = "SELECT {}{}, (CASE WHEN proba_predict > {} THEN '{}' ELSE '{}' END) AS {} FROM {} WHERE predict_neighbors = '{}'".format(
                    ", ".join(X),
                    ", " + ", ".join(key_columns) if key_columns else "",
                    cutoff,
                    self.classes_[1],
                    self.classes_[0],
                    name,
                    self.deploySQL(
                        X=X, test_relation=vdf.__genSQL__(), key_columns=key_columns
                    ),
                    self.classes_[1],
                )
            elif len(self.classes_) == 2:
                sql = "SELECT {}{}, proba_predict AS {} FROM {} WHERE predict_neighbors = '{}'".format(
                    ", ".join(X),
                    ", " + ", ".join(key_columns) if key_columns else "",
                    name,
                    self.deploySQL(
                        X=X, test_relation=vdf.__genSQL__(), key_columns=key_columns
                    ),
                    self.classes_[1],
                )
            else:
                sql = "SELECT {}{}, predict_neighbors AS {} FROM {}".format(
                    ", ".join(X),
                    ", " + ", ".join(key_columns) if key_columns else "",
                    name,
                    self.deploySQL(
                        X=X,
                        test_relation=vdf.__genSQL__(),
                        key_columns=key_columns,
                        predict=True,
                    ),
                )
        sql = "({}) VERTICAPY_SUBTABLE".format(sql)
        return vdf_from_relation(name="Neighbors", relation=sql, cursor=self.cursor)

    # ---#
    def roc_curve(self, pos_label=None):
        """
    ---------------------------------------------------------------------------
    Draws the model ROC curve.

    Parameters
    ----------
    pos_label: int/float/str
        To draw the ROC curve, one of the response column class has to be the 
        positive one. The parameter 'pos_label' represents this class.

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
            self.y, "proba_predict", input_relation, self.cursor, pos_label
        )

    # ---#
    def score(self, method: str = "accuracy", pos_label=None, cutoff: float = -1):
        """
    ---------------------------------------------------------------------------
    Computes the model score.

    Parameters
    ----------
    pos_label: int/float/str, optional
        Label to consider as positive. All the other classes will be merged and
        considered as negative in case of multi classification.
    cutoff: float, optional
        Cutoff for which the tested category will be accepted as prediction. 
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

    Returns
    -------
    float
        score
        """
        check_types([("cutoff", cutoff, [int, float],), ("method", method, [str],)])
        if pos_label == None and len(self.classes_) == 2:
            pos_label = self.classes_[1]
        input_relation = "(SELECT * FROM {} WHERE predict_neighbors = '{}') final_centroids_relation".format(
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
                    self.cursor,
                    pos_label=None,
                )
            else:
                return accuracy_score(y_true, y_score, input_relation, self.cursor)
        elif method == "auc":
            return auc(y_true, y_proba, input_relation, self.cursor)
        elif method == "prc_auc":
            return prc_auc(y_true, y_proba, input_relation, self.cursor)
        elif method in ("best_cutoff", "best_threshold"):
            return roc_curve(
                y_true, y_proba, input_relation, self.cursor, best_threshold=True
            )
        elif method in ("recall", "tpr"):
            return recall_score(y_true, y_score, input_relation, self.cursor)
        elif method in ("precision", "ppv"):
            return precision_score(y_true, y_score, input_relation, self.cursor)
        elif method in ("specificity", "tnr"):
            return specificity_score(y_true, y_score, input_relation, self.cursor)
        elif method in ("negative_predictive_value", "npv"):
            return precision_score(y_true, y_score, input_relation, self.cursor)
        elif method in ("log_loss", "logloss"):
            return log_loss(y_true, y_proba, input_relation, self.cursor)
        elif method == "f1":
            return f1_score(y_true, y_score, input_relation, self.cursor)
        elif method == "mcc":
            return matthews_corrcoef(y_true, y_score, input_relation, self.cursor)
        elif method in ("bm", "informedness"):
            return informedness(y_true, y_score, input_relation, self.cursor)
        elif method in ("mk", "markedness"):
            return markedness(y_true, y_score, input_relation, self.cursor)
        elif method in ("csi", "critical_success_index"):
            return critical_success_index(y_true, y_score, input_relation, self.cursor)
        else:
            raise ParameterError(
                "The parameter 'method' must be in accuracy|auc|prc_auc|best_cutoff|recall|precision|log_loss|negative_predictive_value|specificity|mcc|informedness|markedness|critical_success_index"
            )


# ---#
class NearestCentroid(NeighborsClassifier):
    """
---------------------------------------------------------------------------
[Beta Version]
Creates a NearestCentroid object by using the K Nearest Centroid Algorithm. 
This object is using pure SQL to compute all the distances and final score. 

\u26A0 Warning : As NearestCentroid is using the p-distance, it is highly 
                 sensible to un-normalized data.  

Parameters
----------
cursor: DBcursor, optional
	Vertica DB cursor. 
p: int, optional
	The p corresponding to the one of the p-distance (distance metric used 
	during the model computation).
	"""

    def __init__(self, name: str, cursor=None, p: int = 2):
        check_types([("name", name, [str], False)])
        self.type, self.name = "NearestCentroid", name
        self.set_params({"p": p})
        if not (cursor):
            cursor = read_auto_connect().cursor()
        else:
            check_cursor(cursor)
        self.cursor = cursor

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
        Columns which are not used but to keep during the computations.

	Returns
	-------
	str/list
 		the SQL code needed to deploy the model.
		"""
        check_types(
            [
                ("test_relation", test_relation, [str], False),
                ("predict", predict, [bool], False),
                ("X", X, [list], False),
                ("key_columns", key_columns, [list], False),
            ],
        )
        X = [str_column(elem) for elem in X] if (X) else self.X
        if not (key_columns):
            key_columns = [self.y]
        if not (test_relation):
            test_relation = self.test_relation
        sql = [
            "POWER(ABS(x.{} - y.{}), {})".format(X[i], self.X[i], self.parameters["p"])
            for i in range(len(self.X))
        ]
        distance = "POWER({}, 1 / {})".format(" + ".join(sql), self.parameters["p"])
        sql = "ROW_NUMBER() OVER(PARTITION BY {}, row_id ORDER BY {})".format(
            ", ".join(["x.{}".format(item) for item in X]), distance
        )
        where = " AND ".join(["{} IS NOT NULL".format(item) for item in X])
        sql = "(SELECT {}, {} AS ordered_distance, {} AS distance, y.{} AS predict_neighbors{} FROM (SELECT *, ROW_NUMBER() OVER() AS row_id FROM {} WHERE {}) x CROSS JOIN ({}) y) nc_distance_table".format(
            ", ".join(["x.{}".format(item) for item in X]),
            sql,
            distance,
            self.y,
            ", " + ", ".join(["x." + str_column(elem) for elem in key_columns])
            if (key_columns)
            else "",
            test_relation,
            where,
            self.centroids_.to_sql(),
        )
        if predict:
            sql = "(SELECT {}{}, predict_neighbors FROM {} WHERE ordered_distance = 1) neighbors_table".format(
                ", ".join(X),
                ", " + ", ".join([str_column(elem) for elem in key_columns])
                if (key_columns)
                else "",
                sql,
            )
        else:
            sql = "(SELECT {}{}, predict_neighbors, (1 - DECODE(distance, 0, 0, (distance / SUM(distance) OVER (PARTITION BY {})))) / {} AS proba_predict, ordered_distance FROM {}) neighbors_table".format(
                ", ".join(X),
                ", " + ", ".join([str_column(elem) for elem in key_columns])
                if (key_columns)
                else "",
                ", ".join(X),
                len(self.classes_) - 1,
                sql,
            )
        return sql

    # ---#
    def fit(self, input_relation: str, X: list, y: str, test_relation: str = ""):
        """
	---------------------------------------------------------------------------
	Trains the model.

	Parameters
	----------
	input_relation: str
		Train relation.
	X: list
		List of the predictors.
	y: str
		Response column.
	test_relation: str, optional
		Relation to use to test the model.

	Returns
	-------
	object
 		self
		"""
        check_types(
            [
                ("input_relation", input_relation, [str], False),
                ("X", X, [list], False),
                ("y", y, [str], False),
                ("test_relation", test_relation, [str], False),
            ]
        )
        check_model(name=self.name, cursor=self.cursor)
        func = "APPROXIMATE_MEDIAN" if (self.parameters["p"] == 1) else "AVG"
        self.input_relation = input_relation
        self.test_relation = test_relation if (test_relation) else input_relation
        self.X = [str_column(column) for column in X]
        self.y = str_column(y)
        query = "SELECT {}, {} FROM {} WHERE {} IS NOT NULL GROUP BY {} ORDER BY {} ASC".format(
            ", ".join(
                ["{}({}) AS {}".format(func, column, column) for column in self.X]
            ),
            self.y,
            input_relation,
            self.y,
            self.y,
            self.y,
        )
        self.centroids_ = to_tablesample(query=query, cursor=self.cursor)
        self.centroids_.table_info = False
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
        path = os.path.dirname(
            verticapy.__file__
        ) + "/learn/models/{}.verticapy".format(self.name)
        file = open(path, "x")
        file.write("model_save = " + str(model_save))
        return self


# ---#
class KNeighborsClassifier(NeighborsClassifier):
    """
---------------------------------------------------------------------------
[Beta Version]
Creates a KNeighborsClassifier object by using the K Nearest Neighbors Algorithm. 
This object is using pure SQL to compute all the distances and final score.

\u26A0 Warning : This Algorithm is computationally expensive. It is using a CROSS 
                 JOIN during the computation. The complexity is O(n * n), n 
                 being the total number of elements. As KNeighborsClassifier 
                 is using the p-distance, it is highly sensible to un-normalized 
                 data. 

Parameters
----------
cursor: DBcursor, optional
	Vertica DB cursor. 
n_neighbors: int, optional
	Number of neighbors to consider when computing the score.
p: int, optional
	The p corresponding to the one of the p-distance (distance metric used during 
	the model computation).
	"""

    def __init__(self, name: str, cursor=None, n_neighbors: int = 5, p: int = 2):
        check_types([("name", name, [str], False)])
        self.type, self.name = "KNeighborsClassifier", name
        self.set_params({"n_neighbors": n_neighbors, "p": p})
        if not (cursor):
            cursor = read_auto_connect().cursor()
        else:
            check_cursor(cursor)
        self.cursor = cursor

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
        Columns which are not used but to keep during the computations.

    Returns
    -------
    str/list
        the SQL code needed to deploy the model.
		"""
        check_types(
            [
                ("test_relation", test_relation, [str], False),
                ("predict", predict, [bool], False),
                ("X", X, [list], False),
                ("key_columns", key_columns, [list], False),
            ],
        )
        X = [str_column(elem) for elem in X] if (X) else self.X
        if not (test_relation):
            test_relation = self.test_relation
        if not (key_columns):
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
            ", " + ", ".join(["x." + str_column(elem) for elem in key_columns])
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
            ", " + ", ".join([str_column(elem) for elem in key_columns])
            if (key_columns)
            else "",
            self.parameters["n_neighbors"],
            sql,
            self.parameters["n_neighbors"],
            ", ".join(X),
            ", " + ", ".join([str_column(elem) for elem in key_columns])
            if (key_columns)
            else "",
        )
        if predict:
            sql = "(SELECT {}{}, predict_neighbors FROM (SELECT {}{}, predict_neighbors, ROW_NUMBER() OVER (PARTITION BY {} ORDER BY proba_predict DESC) AS order_prediction FROM {}) VERTICAPY_SUBTABLE WHERE order_prediction = 1) predict_neighbors_table".format(
                ", ".join(X),
                ", " + ", ".join([str_column(elem) for elem in key_columns])
                if (key_columns)
                else "",
                ", ".join(X),
                ", " + ", ".join([str_column(elem) for elem in key_columns])
                if (key_columns)
                else "",
                ", ".join(X),
                sql,
            )
        return sql

    # ---#
    def fit(self, input_relation: str, X: list, y: str, test_relation: str = ""):
        """
	---------------------------------------------------------------------------
	Trains the model.

	Parameters
	----------
	input_relation: str
		Train relation.
	X: list
		List of the predictors.
	y: str
		Response column.
	test_relation: str, optional
		Relation to use to test the model.

	Returns
	-------
	object
 		self
		"""
        check_types(
            [
                ("input_relation", input_relation, [str], False),
                ("X", X, [list], False),
                ("y", y, [str], False),
                ("test_relation", test_relation, [str], False),
            ]
        )
        check_model(name=self.name, cursor=self.cursor)
        self.input_relation = input_relation
        self.test_relation = test_relation if (test_relation) else input_relation
        self.X = [str_column(column) for column in X]
        self.y = str_column(y)
        self.cursor.execute(
            "SELECT DISTINCT {} FROM {} WHERE {} IS NOT NULL ORDER BY {} ASC".format(
                self.y, input_relation, self.y, self.y
            )
        )
        classes = self.cursor.fetchall()
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
        path = os.path.dirname(
            verticapy.__file__
        ) + "/learn/models/{}.verticapy".format(self.name)
        file = open(path, "x")
        file.write("model_save = " + str(model_save))
        return self


# ---#
class KNeighborsRegressor(Regressor):
    """
---------------------------------------------------------------------------
[Beta Version]
Creates a KNeighborsRegressor object by using the K Nearest Neighbors Algorithm. 
This object is using pure SQL to compute all the distances and final score. 

\u26A0 Warning : This Algorithm is computationally expensive. It is using a CROSS 
                 JOIN during the computation. The complexity is O(n * n), n 
                 being the total number of elements. As KNeighborsRegressor 
                 is using the p-distance, it is highly sensible to un-normalized 
                 data.

Parameters
----------
cursor: DBcursor, optional
	Vertica DB cursor. 
n_neighbors: int, optional
	Number of neighbors to consider when computing the score.
p: int, optional
	The p corresponding to the one of the p-distance (distance metric used during 
	the model computation).
	"""

    def __init__(self, name: str, cursor=None, n_neighbors: int = 5, p: int = 2):
        check_types([("name", name, [str], False)])
        self.type, self.name = "KNeighborsRegressor", name
        self.set_params({"n_neighbors": n_neighbors, "p": p})
        if not (cursor):
            cursor = read_auto_connect().cursor()
        else:
            check_cursor(cursor)
        self.cursor = cursor

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
        Columns which are not used but to keep during the computations.

    Returns
    -------
    str/list
        the SQL code needed to deploy the model.
        """
        check_types(
            [
                ("test_relation", test_relation, [str], False),
                ("X", X, [list], False),
                ("key_columns", key_columns, [list], False),
            ],
        )
        X = [str_column(elem) for elem in X] if (X) else self.X
        if not (test_relation):
            test_relation = self.test_relation
        if not (key_columns):
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
            ", " + ", ".join(["x." + str_column(elem) for elem in key_columns])
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
            ", " + ", ".join([str_column(elem) for elem in key_columns])
            if (key_columns)
            else "",
            sql,
            self.parameters["n_neighbors"],
            ", ".join(X),
            ", " + ", ".join([str_column(elem) for elem in key_columns])
            if (key_columns)
            else "",
        )
        return sql

    # ---#
    def fit(self, input_relation: str, X: list, y: str, test_relation: str = ""):
        """
	---------------------------------------------------------------------------
	Trains the model.

	Parameters
	----------
	input_relation: str
		Train relation.
	X: list
		List of the predictors.
	y: str
		Response column.
	test_relation: str, optional
		Relation to use to test the model.

	Returns
	-------
	object
 		self
		"""
        check_types(
            [
                ("input_relation", input_relation, [str], False),
                ("X", X, [list], False),
                ("y", y, [str], False),
                ("test_relation", test_relation, [str], False),
            ]
        )
        check_model(name=self.name, cursor=self.cursor)
        self.input_relation = input_relation
        self.test_relation = test_relation if (test_relation) else input_relation
        self.X = [str_column(column) for column in X]
        self.y = str_column(y)
        model_save = {
            "type": "KNeighborsRegressor",
            "input_relation": self.input_relation,
            "test_relation": self.test_relation,
            "X": self.X,
            "y": self.y,
            "p": self.parameters["p"],
            "n_neighbors": self.parameters["n_neighbors"],
        }
        path = os.path.dirname(
            verticapy.__file__
        ) + "/learn/models/{}.verticapy".format(self.name)
        file = open(path, "x")
        file.write("model_save = " + str(model_save))
        return self

    # ---#
    def predict(self, vdf, X: list = [], name: str = ""):
        """
    ---------------------------------------------------------------------------
    Predicts using the input relation.

    Parameters
    ----------
    vdf: vDataFrame
        Object to use to run the prediction.
    X: list, optional
        List of the columns used to deploy the models. If empty, the model
        predictors will be used.
    name: str, optional
        Name of the added vcolumn. If empty, a name will be generated.

    Returns
    -------
    vDataFrame
        the vDataFrame of the prediction
        """
        check_types(
            [("name", name, [str], False), ("X", X, [list], False),], vdf=["vdf", vdf],
        )
        X = [str_column(elem) for elem in X] if (X) else self.X
        key_columns = vdf.get_columns(exclude_columns=X)
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
                X=X, test_relation=vdf.__genSQL__(), key_columns=key_columns
            ),
        )
        return vdf_from_relation(name="Neighbors", relation=sql, cursor=self.cursor)


# ---#
class LocalOutlierFactor(vModel):
    """
---------------------------------------------------------------------------
[Beta Version]
Creates a LocalOutlierFactor object by using the Local Outlier Factor algorithm 
as defined by Markus M. Breunig, Hans-Peter Kriegel, Raymond T. Ng and Jörg 
Sander. This object is using pure SQL to compute all the distances and final 
score.

\u26A0 Warning : This Algorithm is computationally expensive. It is using a CROSS 
                 JOIN during the computation. The complexity is O(n * n), n 
                 being the total number of elements. As LocalOutlierFactor 
                 is using the p-distance, it is highly sensible to un-normalized 
                 data. A table will be created at the end of the learning phase.

Parameters
----------
name: str
	Name of the the model. As it is not a built in model, this name will be used
	to build the final table.
cursor: DBcursor, optional
	Vertica DB cursor.
n_neighbors: int, optional
	Number of neighbors to consider when computing the score.
p: int, optional
	The p of the p-distance (distance metric used during the model computation).
	"""

    def __init__(self, name: str, cursor=None, n_neighbors: int = 20, p: int = 2):
        check_types([("name", name, [str], False)])
        self.type, self.name = "LocalOutlierFactor", name
        self.set_params({"n_neighbors": n_neighbors, "p": p})
        if not (cursor):
            cursor = read_auto_connect().cursor()
        else:
            check_cursor(cursor)
        self.cursor = cursor

    # ---#
    def fit(
        self, input_relation: str, X: list, key_columns: list = [], index: str = ""
    ):
        """
	---------------------------------------------------------------------------
	Trains the model.

	Parameters
	----------
	input_relation: str
		Train relation.
	X: list
		List of the predictors.
	key_columns: list, optional
		Columns not used during the algorithm computation but which will be used
		to create the final relation.
	index: str, optional
		Index to use to identify each row separately. It is highly recommanded to
		have one already in the main table to avoid creation of temporary tables.

	Returns
	-------
	object
 		self
		"""
        check_types(
            [
                ("input_relation", input_relation, [str], False),
                ("X", X, [list], False),
                ("key_columns", key_columns, [list], False),
                ("index", index, [str], False),
            ]
        )
        check_model(name=self.name, cursor=self.cursor)
        X = [str_column(column) for column in X]
        self.X = X
        self.key_columns = [str_column(column) for column in key_columns]
        self.input_relation = input_relation
        cursor = self.cursor
        n_neighbors = self.parameters["n_neighbors"]
        p = self.parameters["p"]
        relation_alpha = "".join(ch for ch in input_relation if ch.isalnum())
        schema, relation = schema_relation(input_relation)
        if not (index):
            index = "id"
            relation_alpha = "".join(ch for ch in relation if ch.isalnum())
            main_table = "VERTICAPY_MAIN_{}".format(relation_alpha)
            schema = "v_temp_schema"
            try:
                cursor.execute(
                    "DROP TABLE IF EXISTS v_temp_schema.{}".format(main_table)
                )
            except:
                pass
            sql = "CREATE LOCAL TEMPORARY TABLE {} ON COMMIT PRESERVE ROWS AS SELECT ROW_NUMBER() OVER() AS id, {} FROM {} WHERE {}".format(
                main_table,
                ", ".join(X + key_columns),
                input_relation,
                " AND ".join(["{} IS NOT NULL".format(item) for item in X]),
            )
            cursor.execute(sql)
        else:
            main_table = input_relation
        sql = [
            "POWER(ABS(x.{} - y.{}), {})".format(X[i], X[i], p) for i in range(len(X))
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
        try:
            cursor.execute(
                "DROP TABLE IF EXISTS v_temp_schema.VERTICAPY_DISTANCE_{}".format(
                    relation_alpha
                )
            )
        except:
            pass
        sql = "CREATE LOCAL TEMPORARY TABLE VERTICAPY_DISTANCE_{} ON COMMIT PRESERVE ROWS AS {}".format(
            relation_alpha, sql
        )
        cursor.execute(sql)
        kdistance = "(SELECT node_id, nn_id, distance AS distance FROM v_temp_schema.VERTICAPY_DISTANCE_{} WHERE knn = {}) AS kdistance_table".format(
            relation_alpha, n_neighbors + 1
        )
        lrd = "SELECT distance_table.node_id, {} / SUM(CASE WHEN distance_table.distance > kdistance_table.distance THEN distance_table.distance ELSE kdistance_table.distance END) AS lrd FROM (v_temp_schema.VERTICAPY_DISTANCE_{} AS distance_table LEFT JOIN {} ON distance_table.nn_id = kdistance_table.node_id) x GROUP BY 1".format(
            n_neighbors, relation_alpha, kdistance
        )
        try:
            cursor.execute(
                "DROP TABLE IF EXISTS v_temp_schema.VERTICAPY_LRD_{}".format(
                    relation_alpha
                )
            )
        except:
            pass
        sql = "CREATE LOCAL TEMPORARY TABLE VERTICAPY_LRD_{} ON COMMIT PRESERVE ROWS AS {}".format(
            relation_alpha, lrd
        )
        cursor.execute(sql)
        sql = "SELECT x.node_id, SUM(y.lrd) / (MAX(x.node_lrd) * {}) AS LOF FROM (SELECT n_table.node_id, n_table.nn_id, lrd_table.lrd AS node_lrd FROM v_temp_schema.VERTICAPY_DISTANCE_{} AS n_table LEFT JOIN v_temp_schema.VERTICAPY_LRD_{} AS lrd_table ON n_table.node_id = lrd_table.node_id) x LEFT JOIN v_temp_schema.VERTICAPY_LRD_{} AS y ON x.nn_id = y.node_id GROUP BY 1".format(
            n_neighbors, relation_alpha, relation_alpha, relation_alpha
        )
        try:
            cursor.execute(
                "DROP TABLE IF EXISTS v_temp_schema.VERTICAPY_LOF_{}".format(
                    relation_alpha
                )
            )
        except:
            pass
        sql = "CREATE LOCAL TEMPORARY TABLE VERTICAPY_LOF_{} ON COMMIT PRESERVE ROWS AS {}".format(
            relation_alpha, sql
        )
        cursor.execute(sql)
        sql = "SELECT {}, (CASE WHEN lof > 1e100 OR lof != lof THEN 0 ELSE lof END) AS lof_score FROM {} AS x LEFT JOIN v_temp_schema.VERTICAPY_LOF_{} AS y ON x.{} = y.node_id".format(
            ", ".join(X + self.key_columns), main_table, relation_alpha, index
        )
        cursor.execute("CREATE TABLE {} AS {}".format(self.name, sql))
        cursor.execute(
            "SELECT COUNT(*) FROM {}.VERTICAPY_LOF_{} z WHERE lof > 1e100 OR lof != lof".format(
                schema, relation_alpha
            )
        )
        self.n_errors_ = cursor.fetchone()[0]
        cursor.execute(
            "DROP TABLE IF EXISTS v_temp_schema.VERTICAPY_MAIN_{}".format(
                relation_alpha
            )
        )
        cursor.execute(
            "DROP TABLE IF EXISTS v_temp_schema.VERTICAPY_DISTANCE_{}".format(
                relation_alpha
            )
        )
        cursor.execute(
            "DROP TABLE IF EXISTS v_temp_schema.VERTICAPY_LRD_{}".format(relation_alpha)
        )
        cursor.execute(
            "DROP TABLE IF EXISTS v_temp_schema.VERTICAPY_LOF_{}".format(relation_alpha)
        )
        model_save = {
            "type": "LocalOutlierFactor",
            "input_relation": self.input_relation,
            "key_columns": self.key_columns,
            "X": self.X,
            "p": self.parameters["p"],
            "n_neighbors": self.parameters["n_neighbors"],
            "n_errors": self.n_errors_,
        }
        path = os.path.dirname(
            verticapy.__file__
        ) + "/learn/models/{}.verticapy".format(self.name)
        file = open(path, "x")
        file.write("model_save = " + str(model_save))
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
        return vDataFrame(self.name, self.cursor)
