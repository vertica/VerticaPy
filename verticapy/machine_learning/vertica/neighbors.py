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
# VerticaPy Modules
from verticapy.utils._decorators import save_verticapy_logs
from verticapy.learn.metrics import *
from verticapy.sql.drop import drop
from verticapy.sql.read import to_tablesample, vDataFrameSQL
from verticapy.sql.insert import insert_verticapy_schema
from verticapy.core.tablesample import tablesample
from verticapy._config._notebook import ISNOTEBOOK
from verticapy.utils._gen import gen_name, gen_tmp_name
from verticapy.sql.read import _executeSQL
from verticapy import vDataFrame
from verticapy.plotting._matplotlib import *
from verticapy.learn.model_selection import *
from verticapy.errors import *
from verticapy.learn.vmodel import *
from verticapy.learn.tools import *
from verticapy.sql._utils._format import quote_ident, schema_relation
from verticapy.sql._utils._format import clean_query
from verticapy.plotting._matplotlib.core import updated_dict

# Standard Python Modules
import warnings, itertools
from typing import Union, Literal


class NearestCentroid(MulticlassClassifier):
    """
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

    @save_verticapy_logs
    def __init__(self, name: str, p: int = 2):
        self.type, self.name = "NearestCentroid", name
        self.VERTICA_FIT_FUNCTION_SQL = ""
        self.VERTICA_PREDICT_FUNCTION_SQL = ""
        self.MODEL_TYPE = "SUPERVISED"
        self.MODEL_SUBTYPE = "CLASSIFIER"
        self.parameters = {"p": p}

    def fit(
        self,
        input_relation: Union[str, vDataFrame],
        X: Union[str, list],
        y: str,
        test_relation: Union[str, vDataFrame] = "",
    ):
        """
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
        if verticapy.OPTIONS["overwrite_model"]:
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
        X_str = ", ".join([f"{func}({column}) AS {column}" for column in self.X])
        self.centroids_ = to_tablesample(
            query=f"""
            SELECT 
                {X_str}, 
                {self.y} 
            FROM {self.input_relation} 
            WHERE {self.y} IS NOT NULL 
            GROUP BY {self.y} 
            ORDER BY {self.y} ASC""",
            title="Getting Model Centroids.",
        )
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


class KNeighborsClassifier(vModel):
    """
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

    @save_verticapy_logs
    def __init__(self, name: str, n_neighbors: int = 5, p: int = 2):
        self.type, self.name = "KNeighborsClassifier", name
        self.VERTICA_FIT_FUNCTION_SQL = ""
        self.VERTICA_PREDICT_FUNCTION_SQL = ""
        self.MODEL_TYPE = "SUPERVISED"
        self.MODEL_SUBTYPE = "CLASSIFIER"
        self.parameters = {"n_neighbors": n_neighbors, "p": p}

    def deploySQL(
        self,
        X: Union[str, list] = [],
        test_relation: str = "",
        predict: bool = False,
        key_columns: Union[str, list] = [],
    ):
        """
	Returns the SQL code needed to deploy the model. 

    Parameters
    ----------
    X: str / list
        List of the predictors.
    test_relation: str, optional
        Relation to use to do the predictions.
    predict: bool, optional
        If set to True, returns the prediction instead of the probability.
    key_columns: str / list, optional
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
        X = [quote_ident(x) for x in X] if (X) else self.X
        if not (test_relation):
            test_relation = self.test_relation
        if not (key_columns) and key_columns != None:
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
            key_columns_str = ", " + ", ".join([quote_ident(x) for x in key_columns])

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

    def fit(
        self,
        input_relation: Union[str, vDataFrame],
        X: Union[str, list],
        y: str,
        test_relation: Union[str, vDataFrame] = "",
    ):
        """
	Trains the model.

	Parameters
	----------
	input_relation: str/vDataFrame
		Training relation.
	X: str / list
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
        if verticapy.OPTIONS["overwrite_model"]:
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
        classes = _executeSQL(
            query=f"""
                SELECT 
                    /*+LABEL('learn.neighbors.KNeighborsClassifier.fit')*/ 
                    DISTINCT {self.y} 
                FROM {self.input_relation} 
                WHERE {self.y} IS NOT NULL 
                ORDER BY {self.y} ASC""",
            method="fetchall",
            print_time_sql=False,
        )
        self.classes_ = [c[0] for c in classes]
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

    def classification_report(
        self, cutoff: Union[int, float, list] = [], labels: list = []
    ):
        """
    Computes a classification report using multiple metrics to evaluate the model
    (AUC, accuracy, PRC AUC, F1, etc.). For multiclass classification, this 
    function tests the model by considering one class as the sole positive case, 
    repeating the process until it tests all classes.

    Parameters
    ----------
    cutoff: int / float / list, optional
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
        if not (labels):
            labels = self.classes_
        return classification_report(cutoff=cutoff, estimator=self, labels=labels)

    report = classification_report

    def cutoff_curve(
        self, pos_label: Union[int, float, str] = None, ax=None, **style_kwds
    ):
        """
    Draws the ROC curve of a classification model.

    Parameters
    ----------
    pos_label: int / float / str
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
        if pos_label == None and len(self.classes_) == 2:
            pos_label = self.classes_[1]
        if pos_label not in self.classes_:
            raise ParameterError(
                "'pos_label' must be one of the response column classes"
            )
        input_relation = self.deploySQL() + f" WHERE predict_neighbors = '{pos_label}'"
        return roc_curve(
            self.y,
            "proba_predict",
            input_relation,
            pos_label,
            ax=ax,
            cutoff_curve=True,
            **style_kwds,
        )

    def confusion_matrix(
        self, pos_label: Union[int, float, str] = None, cutoff: Union[int, float] = -1,
    ):
        """
    Computes the model confusion matrix.

    Parameters
    ----------
    pos_label: int / float / str, optional
        Label to consider as positive. All the other classes will be merged and
        considered as negative for multiclass classification.
    cutoff: int / float, optional
        Cutoff for which the tested category will be accepted as a prediction. If the 
        cutoff is not between 0 and 1, the entire confusion matrix will be drawn.

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
        if pos_label in self.classes_ and cutoff <= 1 and cutoff >= 0:
            input_relation = (
                self.deploySQL() + f" WHERE predict_neighbors = '{pos_label}'"
            )
            y_score = f"(CASE WHEN proba_predict > {cutoff} THEN 1 ELSE 0 END)"
            y_true = f"DECODE({self.y}, '{pos_label}', 1, 0)"
            result = confusion_matrix(y_true, y_score, input_relation)
            if pos_label == 1:
                return result
            else:
                return tablesample(
                    values={
                        "index": [f"Non-{pos_label}", str(pos_label),],
                        f"Non-{pos_label}": result.values[0],
                        str(pos_label): result.values[1],
                    },
                )
        else:
            input_relation = f"""
                (SELECT 
                    *, 
                    ROW_NUMBER() OVER(PARTITION BY {", ".join(self.X)}, row_id 
                                      ORDER BY proba_predict DESC) AS pos 
                 FROM {self.deploySQL()}) neighbors_table WHERE pos = 1"""
            return multilabel_confusion_matrix(
                self.y, "predict_neighbors", input_relation, self.classes_
            )

    def lift_chart(
        self, pos_label: Union[int, float, str] = None, ax=None, **style_kwds
    ):
        """
    Draws the model Lift Chart.

    Parameters
    ----------
    pos_label: int / float / str
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
        if pos_label == None and len(self.classes_) == 2:
            pos_label = self.classes_[1]
        if pos_label not in self.classes_:
            raise ParameterError(
                "'pos_label' must be one of the response column classes"
            )
        input_relation = self.deploySQL() + f" WHERE predict_neighbors = '{pos_label}'"
        return lift_chart(
            self.y, "proba_predict", input_relation, pos_label, ax=ax, **style_kwds,
        )

    def prc_curve(
        self, pos_label: Union[int, float, str] = None, ax=None, **style_kwds
    ):
        """
    Draws the model PRC curve.

    Parameters
    ----------
    pos_label: int / float / str
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
        if pos_label == None and len(self.classes_) == 2:
            pos_label = self.classes_[1]
        if pos_label not in self.classes_:
            raise ParameterError(
                "'pos_label' must be one of the response column classes"
            )
        input_relation = self.deploySQL() + f" WHERE predict_neighbors = '{pos_label}'"
        return prc_curve(
            self.y, "proba_predict", input_relation, pos_label, ax=ax, **style_kwds,
        )

    def predict(
        self,
        vdf: Union[str, vDataFrame],
        X: Union[str, list] = [],
        name: str = "",
        cutoff: Union[int, float] = 0.5,
        inplace: bool = True,
        **kwargs,
    ):
        """
    Predicts using the input relation.

    Parameters
    ----------
    vdf: str / vDataFrame
        Object to use to run the prediction. You can also specify a customized 
        relation, but you must enclose it with an alias. For example,  
        "(SELECT 1) x" is correct, whereas "(SELECT 1)" and "SELECT 1" are 
        incorrect.
    X: str / list, optional
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
        if key_columns:
            key_columns_str = ", " + ", ".join(key_columns)
        else:
            key_columns_str = ""
        if not (name):
            name = gen_name([self.type, self.name])

        if (
            len(self.classes_) == 2
            and self.classes_[0] in [0, "0"]
            and self.classes_[1] in [1, "1"]
        ):
            table = self.deploySQL(
                X=X, test_relation=vdf.__genSQL__(), key_columns=key_columns_arg
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
                test_relation=vdf.__genSQL__(),
                key_columns=key_columns_arg,
                predict=True,
            )
            sql = f"""
                (SELECT 
                    {", ".join(X)}{key_columns_str}, 
                    predict_neighbors AS {name} 
                 FROM {table}) VERTICAPY_SUBTABLE"""
        if inplace:
            return vDataFrameSQL(name="Neighbors", relation=sql, vdf=vdf)
        else:
            return vDataFrameSQL(name="Neighbors", relation=sql)

    def predict_proba(
        self,
        vdf: Union[str, vDataFrame],
        X: Union[str, list] = [],
        name: str = "",
        pos_label: Union[int, str, float] = None,
        inplace: bool = True,
        **kwargs,
    ):
        """
    Returns the model's probabilities using the input relation.

    Parameters
    ----------
    vdf: str / vDataFrame
        Object to use to run the prediction. You can also specify a customized 
        relation, but you must enclose it with an alias. For example, "(SELECT 1) x" 
        is correct, whereas "(SELECT 1)" and "SELECT 1" are incorrect.
    X: str / list, optional
        List of the columns used to deploy the models. If empty, the model
        predictors will be used.
    name: str, optional
        Name of the additional prediction vColumn. If unspecified, a name is 
	    generated based on the model and class names.
    pos_label: int / float / str, optional
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
        assert pos_label is None or pos_label in self.classes_, ParameterError(
            (
                "Incorrect parameter 'pos_label'.\nThe class label "
                f"must be in [{'|'.join([str(c) for c in self.classes_])}]. "
                f"Found '{pos_label}'."
            )
        )
        if isinstance(vdf, str):
            vdf = vDataFrameSQL(relation=vdf)
        X = [quote_ident(x) for x in X] if (X) else self.X
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
            X=X, test_relation=vdf.__genSQL__(), key_columns=key_columns_arg
        )
        sql = f"""
            (SELECT 
                {", ".join(X)}{key_columns_str}, 
                {", ".join(predict)} 
             FROM {table} 
             GROUP BY {", ".join(X + key_columns)}) VERTICAPY_SUBTABLE"""

        # Result
        if inplace:
            return vDataFrameSQL(name="Neighbors", relation=sql, vdf=vdf)
        else:
            return vDataFrameSQL(name="Neighbors", relation=sql)

    def roc_curve(
        self, pos_label: Union[int, float, str] = None, ax=None, **style_kwds
    ):
        """
    Draws the model ROC curve.

    Parameters
    ----------
    pos_label: int / float / str
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
        input_relation = self.deploySQL() + f" WHERE predict_neighbors = '{pos_label}'"
        return roc_curve(
            self.y, "proba_predict", input_relation, pos_label, ax=ax, **style_kwds,
        )

    def score(
        self,
        method: str = "accuracy",
        pos_label: Union[str, int, float] = None,
        cutoff: Union[int, float] = -1,
        nbins: int = 10000,
    ):
        """
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
        if pos_label == None and len(self.classes_) == 2:
            pos_label = self.classes_[1]
        input_relation = f"""
            (SELECT 
                * 
             FROM {self.deploySQL()} 
             WHERE predict_neighbors = '{pos_label}') final_centroids_relation"""
        y_score = f"(CASE WHEN proba_predict > {cutoff} THEN 1 ELSE 0 END)"
        y_proba = "proba_predict"
        y_true = f"DECODE({self.y}, '{pos_label}', 1, 0)"
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


class KernelDensity(Regressor, Tree):
    """
[Beta Version]
Creates a KernelDensity object. 
This object uses pure SQL to compute the final score.

Parameters
----------
bandwidth: int / float, optional
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
max_leaf_nodes: int / float, optional
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

    @save_verticapy_logs
    def __init__(
        self,
        name: str,
        bandwidth: Union[int, float] = 1.0,
        kernel: Literal["gaussian", "logistic", "sigmoid", "silverman"] = "gaussian",
        p: int = 2,
        max_leaf_nodes: Union[int, float] = 1e9,
        max_depth: int = 5,
        min_samples_leaf: int = 1,
        nbins: int = 5,
        xlim: list = [],
        **kwargs,
    ):
        self.type, self.name = "KernelDensity", name
        self.VERTICA_FIT_FUNCTION_SQL = "RF_REGRESSOR"
        self.VERTICA_PREDICT_FUNCTION_SQL = "PREDICT_RF_REGRESSOR"
        self.MODEL_TYPE = "UNSUPERVISED"
        self.MODEL_SUBTYPE = "PREPROCESSING"
        self.parameters = {
            "nbins": nbins,
            "p": p,
            "bandwidth": bandwidth,
            "kernel": str(kernel).lower(),
            "max_leaf_nodes": int(max_leaf_nodes),
            "max_depth": int(max_depth),
            "min_samples_leaf": int(min_samples_leaf),
            "xlim": xlim,
        }
        if "store" not in kwargs or kwargs["store"]:
            self.verticapy_store = True
        else:
            self.verticapy_store = False

    def fit(self, input_relation: Union[str, vDataFrame], X: Union[str, list] = []):
        """
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
        if verticapy.OPTIONS["overwrite_model"]:
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
        X = vdf.format_colnames(X)

        def density_compute(
            vdf: vDataFrame,
            columns: list,
            h=None,
            kernel: str = "gaussian",
            nbins: int = 5,
            p: int = 2,
        ):
            def density_kde(vdf, columns: list, kernel: str, x, p: int, h=None):
                for col in columns:
                    if not (vdf[col].isnum()):
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
                        FROM {vdf.__genSQL__()}"""
                    result = _executeSQL(
                        query, title="Computing the KDE", method="fetchrow"
                    )
                    return [x for x in result]
                else:
                    return 0

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
        name_str = self.name.replace('"', "")
        if self.verticapy_store:
            _executeSQL(
                query=f"""CREATE TABLE {name_str}_KernelDensity_Map AS    
                            SELECT 
                                /*+LABEL('learn.neighbors.KernelDensity.fit')*/
                                {", ".join(X)}, 0.0::float AS KDE 
                            FROM {vdf.__genSQL__()} 
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
                    INTO {name_str}_KernelDensity_Map 
                    ({", ".join(X)}, KDE) {" UNION ".join(values)}""",
                    title=f"Computing the KDE [Step {idx}].",
                )
                _executeSQL("COMMIT;", print_time_sql=False)
                r += 100
                idx += 1
            self.X, self.input_relation = X, input_relation
            self.map = f"{name_str}_KernelDensity_Map"
            self.tree_name = f"{name_str}_KernelDensity_Tree"
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
                model_name=self.name, model_type="KernelDensity", model_save=model_save,
            )
        else:
            self.X, self.input_relation = X, input_relation
            self.verticapy_x = x
            self.verticapy_y = y
        return self

    def plot(self, ax=None, **style_kwds):
        """
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
                query = f"""
                    SELECT 
                        /*+LABEL('learn.neighbors.KernelDensity.plot')*/ 
                        {self.X[0]}, KDE 
                    FROM {self.map} ORDER BY 1"""
                result = _executeSQL(query, method="fetchall", print_time_sql=False)
                x, y = [v[0] for v in result], [v[1] for v in result]
            else:
                x, y = [v[0] for v in self.verticapy_x], self.verticapy_y
            if not (ax):
                fig, ax = plt.subplots()
                if ISNOTEBOOK:
                    fig.set_size_inches(7, 5)
                ax.grid()
                ax.set_axisbelow(True)
            from verticapy.plotting._colors import gen_colors

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
            result, idx = [], 0
            while idx < (n + 1) * (n + 1):
                result += [[z[idx + i] for i in range(n + 1)]]
                idx += n + 1
            if not (ax):
                fig, ax = plt.subplots()
                if ISNOTEBOOK:
                    fig.set_size_inches(8, 6)
            else:
                fig = plt
            param = {
                "cmap": "Reds",
                "origin": "lower",
                "interpolation": "bilinear",
            }
            extent = [min(x), max(x), min(y), max(y)]
            extent = [float(v) for v in extent]
            im = ax.imshow(result, extent=extent, **updated_dict(param, style_kwds))
            fig.colorbar(im, ax=ax)
            ax.set_ylabel(self.X[1])
            ax.set_xlabel(self.X[0])
            return ax
        else:
            raise Exception("KDE Plots are only available in 1D or 2D.")


class KNeighborsRegressor(Regressor):
    """
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

    @save_verticapy_logs
    def __init__(self, name: str, n_neighbors: int = 5, p: int = 2):
        self.type, self.name = "KNeighborsRegressor", name
        self.VERTICA_FIT_FUNCTION_SQL = ""
        self.VERTICA_PREDICT_FUNCTION_SQL = ""
        self.MODEL_TYPE = "SUPERVISED"
        self.MODEL_SUBTYPE = "REGRESSOR"
        self.parameters = {"n_neighbors": n_neighbors, "p": p}

    def deploySQL(
        self,
        X: Union[str, list] = [],
        test_relation: str = "",
        key_columns: Union[str, list] = [],
    ):
        """
    Returns the SQL code needed to deploy the model. 

    Parameters
    ----------
    X: str / list
        List of the predictors.
    test_relation: str, optional
        Relation to use to do the predictions.
    key_columns: str / list, optional
        A list of columns to include in the results, but to exclude from 
        computation of the prediction.

    Returns
    -------
    str/list
        the SQL code needed to deploy the model.
        """
        from verticapy.sql._utils._format import clean_query

        if isinstance(X, str):
            X = [X]
        if isinstance(key_columns, str):
            key_columns = [key_columns]
        X = [quote_ident(elem) for elem in X] if (X) else self.X
        if not (test_relation):
            test_relation = self.test_relation
        if not (key_columns) and key_columns != None:
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
            key_columns_str = ", " + ", ".join([quote_ident(x) for x in key_columns])
        n_neighbors = self.parameters["n_neighbors"]
        sql = f"""
            (SELECT 
                {", ".join(X)}{key_columns_str}, 
                AVG(predict_neighbors) AS predict_neighbors 
             FROM ({sql}) z 
             WHERE ordered_distance <= {n_neighbors} 
             GROUP BY {", ".join(X)}{key_columns_str}, row_id) knr_table"""
        return clean_query(sql)

    def fit(
        self,
        input_relation: Union[str, vDataFrame],
        X: Union[str, list],
        y: str,
        test_relation: Union[str, vDataFrame] = "",
    ):
        """
	Trains the model.

	Parameters
	----------
	input_relation: str / vDataFrame
		Training relation.
	X: str / list
		List of the predictors.
	y: str
		Response column.
	test_relation: str / vDataFrame, optional
		Relation used to test the model.

	Returns
	-------
	object
 		self
		"""
        if isinstance(X, str):
            X = [X]
        if verticapy.OPTIONS["overwrite_model"]:
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

    def predict(
        self,
        vdf: Union[str, vDataFrame],
        X: Union[str, list] = [],
        name: str = "",
        inplace: bool = True,
        **kwargs,
    ):
        """
    Predicts using the input relation.

    Parameters
    ----------
    vdf: str / vDataFrame
        Object to use to run the prediction. You can also specify a customized 
        relation, but you must enclose it with an alias. For example "(SELECT 1) x" 
        is correct whereas "(SELECT 1)" and "SELECT 1" are incorrect.
    X: str / list, optional
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
        if isinstance(vdf, str):
            vdf = vDataFrameSQL(vdf)
        X = [quote_ident(elem) for elem in X] if (X) else self.X
        key_columns = vdf.get_columns(exclude_columns=X)
        if "key_columns" in kwargs:
            key_columns_arg = None
        else:
            key_columns_arg = key_columns
        if not (name):
            name = f"{self.type}_" + "".join(ch for ch in self.name if ch.isalnum())
        if key_columns:
            key_columns_str = ", " + ", ".join(key_columns)
        else:
            key_columns_str = ""
        table = self.deploySQL(
            X=X, test_relation=vdf.__genSQL__(), key_columns=key_columns_arg
        )
        sql = f"""
            (SELECT 
                {", ".join(X)}{key_columns_str}, 
                predict_neighbors AS {name} 
             FROM {table}) VERTICAPY_SUBTABLE"""
        if inplace:
            return vDataFrameSQL(name="Neighbors", relation=sql, vdf=vdf)
        else:
            return vDataFrameSQL(name="Neighbors", relation=sql)


class LocalOutlierFactor(vModel):
    """
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

    @save_verticapy_logs
    def __init__(self, name: str, n_neighbors: int = 20, p: int = 2):
        self.type, self.name = "LocalOutlierFactor", name
        self.VERTICA_FIT_FUNCTION_SQL = ""
        self.VERTICA_PREDICT_FUNCTION_SQL = ""
        self.MODEL_TYPE = "UNSUPERVISED"
        self.MODEL_SUBTYPE = "ANOMALY_DETECTION"
        self.parameters = {"n_neighbors": n_neighbors, "p": p}

    def fit(
        self,
        input_relation: Union[str, vDataFrame],
        X: Union[str, list] = [],
        key_columns: Union[str, list] = [],
        index: str = "",
    ):
        """
	Trains the model.

	Parameters
	----------
	input_relation: str / vDataFrame
		Training relation.
	X: str / list, optional
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
        if verticapy.OPTIONS["overwrite_model"]:
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
        tmp_main_table_name = gen_tmp_name(name="main")
        tmp_distance_table_name = gen_tmp_name(name="distance")
        tmp_lrd_table_name = gen_tmp_name(name="lrd")
        tmp_lof_table_name = gen_tmp_name(name="lof")
        try:
            if not (index):
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
                    CREATE TABLE {self.name} AS 
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
        finally:
            drop(f"v_temp_schema.{tmp_main_table_name}", method="table")
            drop(f"v_temp_schema.{tmp_distance_table_name}", method="table")
            drop(f"v_temp_schema.{tmp_lrd_table_name}", method="table")
            drop(f"v_temp_schema.{tmp_lof_table_name}", method="table")
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

    def predict(self):
        """
	Creates a vDataFrame of the model.

	Returns
	-------
	vDataFrame
 		the vDataFrame including the prediction.
		"""
        return vDataFrame(self.name)
