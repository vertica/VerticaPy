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
from verticapy import vDataFrame
from verticapy.learn.plot import *
from verticapy.utilities import *
from verticapy.toolbox import *
from verticapy.connections.connect import read_auto_connect
from verticapy.errors import *
from verticapy.learn.metrics import *

# ---#
def binary_classification_score(model, method: str = "accuracy", cutoff: float = 0.5):
    """
---------------------------------------------------------------------------
Computes the model score.

Parameters
----------
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

cutoff: float, optional
	Cutoff for which the tested category will be accepted as prediction. 

Returns
-------
float
	score
	"""
    check_types(
        [("cutoff", cutoff, [int, float], False), ("method", method, [str], False)]
    )
    if method in ("accuracy", "acc"):
        return accuracy_score(
            model.y, model.deploySQL(cutoff), model.test_relation, model.cursor
        )
    elif method == "auc":
        return auc(model.y, model.deploySQL(), model.test_relation, model.cursor)
    elif method == "prc_auc":
        return prc_auc(model.y, model.deploySQL(), model.test_relation, model.cursor)
    elif method in ("best_cutoff", "best_threshold"):
        return roc_curve(
            model.y,
            model.deploySQL(),
            model.test_relation,
            model.cursor,
            best_threshold=True,
        )
    elif method in ("recall", "tpr"):
        return recall_score(
            model.y, model.deploySQL(cutoff), model.test_relation, model.cursor
        )
    elif method in ("precision", "ppv"):
        return precision_score(
            model.y, model.deploySQL(cutoff), model.test_relation, model.cursor
        )
    elif method in ("specificity", "tnr"):
        return specificity_score(
            model.y, model.deploySQL(cutoff), model.test_relation, model.cursor
        )
    elif method in ("negative_predictive_value", "npv"):
        return precision_score(
            model.y, model.deploySQL(cutoff), model.test_relation, model.cursor
        )
    elif method in ("log_loss", "logloss"):
        return log_loss(model.y, model.deploySQL(), model.test_relation, model.cursor)
    elif method == "f1":
        return f1_score(
            model.y, model.deploySQL(cutoff), model.test_relation, model.cursor
        )
    elif method == "mcc":
        return matthews_corrcoef(
            model.y, model.deploySQL(cutoff), model.test_relation, model.cursor
        )
    elif method in ("bm", "informedness"):
        return informedness(
            model.y, model.deploySQL(cutoff), model.test_relation, model.cursor
        )
    elif method in ("mk", "markedness"):
        return markedness(
            model.y, model.deploySQL(cutoff), model.test_relation, model.cursor
        )
    elif method in ("csi", "critical_success_index"):
        return critical_success_index(
            model.y, model.deploySQL(cutoff), model.test_relation, model.cursor
        )
    else:
        raise ParameterError(
            "The parameter 'method' must be in accuracy|auc|prc_auc|best_cutoff|recall|precision|log_loss|negative_predictive_value|specificity|mcc|informedness|markedness|critical_success_index"
        )


# ---#
def classification_report_binary(model, cutoff: float = 0.5):
    """
---------------------------------------------------------------------------
Computes a classification report using multiple metrics to evaluate the model
(AUC, accuracy, PRC AUC, F1...). 

Parameters
----------
cutoff: float, optional
	Probability cutoff.

Returns
-------
tablesample
		An object containing the result. For more information, see
		utilities.tablesample.
	"""
    check_types([("cutoff", cutoff, [int, float], False)])
    if cutoff > 1 or cutoff < 0:
        cutoff = model.score(method="best_cutoff")
    return classification_report(
        model.y,
        [model.deploySQL(), model.deploySQL(cutoff)],
        model.test_relation,
        model.cursor,
    )


# ---#
def classification_report_multiclass(model, cutoff=[], labels: list = []):
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
        [
            ("cutoff", cutoff, [int, float, list], False),
            ("labels", labels, [list], False),
        ]
    )
    if not (labels):
        labels = model.classes
    return classification_report(cutoff=cutoff, estimator=model, labels=labels)


# ---#
def confusion_matrix_binary(model, cutoff: float = 0.5):
    """
---------------------------------------------------------------------------
Computes the model confusion matrix.

Parameters
----------
cutoff: float, optional
	Probability cutoff.

Returns
-------
tablesample
		An object containing the result. For more information, see
		utilities.tablesample.
	"""
    check_types([("cutoff", cutoff, [int, float], False)])
    return confusion_matrix(
        model.y, model.deploySQL(cutoff), model.test_relation, model.cursor
    )


# ---#
def confusion_matrix_multiclass(model, pos_label=None, cutoff: float = -1):
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
    check_types([("cutoff", cutoff, [int, float], False)])
    pos_label = (
        model.classes[1]
        if (pos_label == None and len(model.classes) == 2)
        else pos_label
    )
    if pos_label in model.classes and cutoff <= 1 and cutoff >= 0:
        return confusion_matrix(
            model.y,
            model.deploySQL(pos_label, cutoff),
            model.test_relation,
            model.cursor,
            pos_label=pos_label,
        )
    else:
        return multilabel_confusion_matrix(
            model.y, model.deploySQL(), model.test_relation, model.classes, model.cursor
        )


# ---#
def deploySQL(model, X: list = []):
    """
---------------------------------------------------------------------------
Returns the SQL code needed to deploy the model. 

Parameters
----------
X: list, optional
	List of the columns used to deploy the model. If empty, the model
	predictors will be used.

Returns
-------
str
	the SQL code needed to deploy the model.
	"""
    check_types([("X", X, [list], False)])
    X = [str_column(elem) for elem in X]
    fun = get_model_fun(model)[1]
    sql = "{}({} USING PARAMETERS model_name = '{}', match_by_pos = 'true')"
    return sql.format(fun, ", ".join(model.X if not (X) else X), model.name)


# ---#
def deployInverseSQL(model, X: list = []):
    """
---------------------------------------------------------------------------
Returns the SQL code needed to deploy the inverse model. 

Parameters
----------
X: list, optional
	List of the columns used to deploy the model. If empty, the model
	predictors will be used.

Returns
-------
str
	the SQL code needed to deploy the inverse model.
	"""
    check_types([("X", X, [list], False)])
    X = [str_column(elem) for elem in X]
    fun = get_model_fun(model)[2]
    sql = "{}({} USING PARAMETERS model_name = '{}', match_by_pos = 'true')"
    return sql.format(fun, ", ".join(model.X if not (X) else X), model.name)


# ---#
def deploySQL_binary(model, cutoff: float = -1, X: list = []):
    """
---------------------------------------------------------------------------
Returns the SQL code needed to deploy the model. 

Parameters
----------
cutoff: float, optional
	Probability cutoff. If this number is not between 0 and 1, the method 
	will return the probability to be of class 1.
X: list, optional
	List of the columns used to deploy the model. If empty, the model
	predictors will be used.

Returns
-------
str
	the SQL code needed to deploy the model.
	"""
    check_types([("cutoff", cutoff, [int, float], False), ("X", X, [list], False)])
    X = [str_column(elem) for elem in X]
    fun = get_model_fun(model)[1]
    sql = "{}({} USING PARAMETERS model_name = '{}', type = 'probability', match_by_pos = 'true')"
    if cutoff <= 1 and cutoff >= 0:
        sql = "(CASE WHEN {} > {} THEN 1 ELSE 0 END)".format(sql, cutoff)
    return sql.format(fun, ", ".join(model.X if not (X) else X), model.name)


# ---#
def deploySQL_multiclass(
    model, pos_label=None, cutoff: float = -1, allSQL: bool = False, X: list = []
):
    """
---------------------------------------------------------------------------
Returns the SQL code needed to deploy the model. 

Parameters
----------
pos_label: int/float/str, optional
	Label to consider as positive. All the other classes will be merged and
	considered as negative in case of multi classification.
cutoff: float, optional
	Cutoff for which the tested category will be accepted as prediction. If 
	the cutoff is not between 0 and 1, a probability will be returned.
allSQL: bool, optional
	If set to True, the output will be a list of the different SQL codes 
	needed to deploy the different categories score.
X: list, optional
	List of the columns used to deploy the model. If empty, the model
	predictors will be used.

Returns
-------
str / list
	the SQL code needed to deploy the model.
	"""
    check_types(
        [
            ("cutoff", cutoff, [int, float], False),
            ("allSQL", allSQL, [bool], False),
            ("X", X, [list], False),
        ]
    )
    X = [str_column(elem) for elem in X]
    fun = get_model_fun(model)[1]
    if allSQL:
        sql = "{}({} USING PARAMETERS model_name = '{}', class = '{}', type = 'probability', match_by_pos = 'true')".format(
            fun, ", ".join(model.X if not (X) else X), model.name, "{}"
        )
        sql = [
            sql,
            "{}({} USING PARAMETERS model_name = '{}', match_by_pos = 'true')".format(
                fun, ", ".join(model.X if not (X) else X), model.name
            ),
        ]
    else:
        if pos_label in model.classes and cutoff <= 1 and cutoff >= 0:
            sql = "{}({} USING PARAMETERS model_name = '{}', class = '{}', type = 'probability', match_by_pos = 'true')".format(
                fun, ", ".join(model.X if not (X) else X), model.name, pos_label
            )
            if len(model.classes) > 2:
                sql = "(CASE WHEN {} >= {} THEN '{}' WHEN {} IS NULL THEN NULL ELSE 'Non-{}' END)".format(
                    sql, cutoff, pos_label, sql, pos_label
                )
            else:
                non_pos_label = (
                    model.classes[0]
                    if (model.classes[0] != pos_label)
                    else model.classes[1]
                )
                sql = "(CASE WHEN {} >= {} THEN '{}' WHEN {} IS NULL THEN NULL ELSE '{}' END)".format(
                    sql, cutoff, pos_label, sql, non_pos_label
                )
        elif pos_label in model.classes:
            sql = "{}({} USING PARAMETERS model_name = '{}', class = '{}', type = 'probability', match_by_pos = 'true')".format(
                fun, ", ".join(model.X if not (X) else X), model.name, pos_label
            )
        else:
            sql = "{}({} USING PARAMETERS model_name = '{}', match_by_pos = 'true')".format(
                fun, ", ".join(model.X if not (X) else X), model.name
            )
    return sql


# ---#
def deploySQL_decomposition(
    model,
    n_components: int = 0,
    cutoff: float = 1,
    key_columns: list = [],
    X: list = [],
):
    """
---------------------------------------------------------------------------
Returns the SQL code needed to deploy the model. 

Parameters
----------
n_components: int, optional
	Number of components to return. If set to 0, all the components will be
	deployed.
cutoff: float, optional
	Specifies the minimum accumulated explained variance. Components are taken 
	until the accumulated explained variance reaches this value.
key_columns: list, optional
	Predictors used during the algorithm computation which will be deployed
	with the principal components.
X: list, optional
	List of the columns used to deploy the model. If empty, the model
	predictors will be used.

Returns
-------
str
	the SQL code needed to deploy the model.
	"""
    check_types(
        [
            ("n_components", n_components, [int, float], False),
            ("cutoff", cutoff, [int, float], False),
            ("key_columns", key_columns, [list], False),
            ("X", X, [list], False),
        ]
    )
    X = [str_column(elem) for elem in X]
    fun = get_model_fun(model)[1]
    sql = "{}({} USING PARAMETERS model_name = '{}', match_by_pos = 'true'"
    if key_columns:
        sql += ", key_columns = '{}'".format(
            ", ".join([str_column(item) for item in key_columns])
        )
    if n_components:
        sql += ", num_components = {}".format(n_components)
    else:
        sql += ", cutoff = {}".format(cutoff)
    sql += ")"
    return sql.format(fun, ", ".join(model.X if not (X) else X), model.name)


# ---#
def deployInverseSQL_decomposition(model, key_columns: list = [], X: list = []):
    """
---------------------------------------------------------------------------
Returns the SQL code needed to deploy the inverse model. 

Parameters
----------
key_columns: list, optional
	Predictors used during the algorithm computation which will be deployed
	with the principal components.
X: list, optional
	List of the columns used to deploy the model. If empty, the model
	predictors will be used.

Returns
-------
str
	the SQL code needed to deploy the inverse model.
	"""
    check_types([("key_columns", key_columns, [list], False), ("X", X, [list], False)])
    X = [str_column(elem) for elem in X]
    fun = get_model_fun(model)[2]
    sql = "{}({} USING PARAMETERS model_name = '{}', match_by_pos = 'true'"
    if key_columns:
        sql += ", key_columns = '{}'".format(
            ", ".join([str_column(item) for item in key_columns])
        )
    sql += ")"
    return sql.format(fun, ", ".join(model.X if not (X) else X), model.name)


# ---#
def drop(model):
    """
---------------------------------------------------------------------------
Drops the model from the Vertica DB.
	"""
    drop_model(model.name, model.cursor, print_info=False)


# ---#
def export_graphviz(model, tree_id: int = 0):
    """
---------------------------------------------------------------------------
Converts the input tree to graphviz.

Parameters
----------
tree_id: int, optional
	Unique tree identifier. It is an integer between 0 and n_estimators - 1

Returns
-------
str
		graphviz formatted tree.
	"""
    check_types([("tree_id", tree_id, [int, float], False)])
    query = "SELECT READ_TREE ( USING PARAMETERS model_name = '{}', tree_id = {}, format = 'graphviz');".format(
        model.name, tree_id
    )
    model.cursor.execute(query)
    return model.cursor.fetchone()[1]


# ---#
def features_importance(model):
    """
	---------------------------------------------------------------------------
	Computes the model features importance.

	Returns
	-------
	tablesample
		An object containing the result. For more information, see
		utilities.tablesample.
	"""
    if model.type in ("RandomForestClassifier", "RandomForestRegressor"):
        query = "SELECT predictor_name AS predictor, ROUND(100 * importance_value / SUM(importance_value) OVER (), 2) AS importance, SIGN(importance_value) AS sign FROM (SELECT RF_PREDICTOR_IMPORTANCE ( USING PARAMETERS model_name = '{}')) x ORDER BY 2 DESC;".format(
            model.name
        )
        print_legend = False
    else:
        query = "SELECT predictor, ROUND(100 * importance / SUM(importance) OVER(), 2) AS importance, sign FROM "
        query += "(SELECT stat.predictor AS predictor, ABS(coefficient * (max - min)) AS importance, SIGN(coefficient) AS sign FROM "
        query += '(SELECT LOWER("column") AS predictor, min, max FROM (SELECT SUMMARIZE_NUMCOL({}) OVER() '.format(
            ", ".join(model.X)
        )
        query += " FROM {}) x) stat NATURAL JOIN (SELECT GET_MODEL_ATTRIBUTE (USING PARAMETERS model_name = '{}', ".format(
            model.input_relation, model.name
        )
        query += "attr_name = 'details')) coeff) importance_t ORDER BY 2 DESC;"
        print_legend = True
    model.cursor.execute(query)
    result = model.cursor.fetchall()
    coeff_importances, coeff_sign = {}, {}
    for elem in result:
        coeff_importances[elem[0]] = elem[1]
        coeff_sign[elem[0]] = elem[2]
    try:
        plot_importance(coeff_importances, coeff_sign, print_legend=print_legend)
    except:
        pass
    importances = {"index": ["importance"]}
    for elem in coeff_importances:
        importances[elem] = [coeff_importances[elem]]
    return tablesample(values=importances, table_info=False).transpose()


# ---#
def fit(model, input_relation: str, X: list, y: str, test_relation: str = ""):
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
	model
	"""
    check_types(
        [
            ("input_relation", input_relation, [str], False),
            ("X", X, [list], False),
            ("y", y, [str], False),
            ("test_relation", test_relation, [str], False),
        ]
    )
    model.input_relation = input_relation
    model.test_relation = test_relation if (test_relation) else input_relation
    model.X = [str_column(column) for column in X]
    model.y = str_column(y)
    parameters = vertica_param_dict(model)
    if (
        "regularization" in parameters
        and parameters["regularization"].lower() == "'enet'"
    ):
        alpha = parameters["alpha"]
        del parameters["alpha"]
    else:
        alpha = None
    if "mtry" in parameters:
        if parameters["mtry"] == "'auto'":
            parameters["mtry"] = int(len(model.X) / 3 + 1)
        elif model.parameters["mtry"] == "'max'":
            model.parameters["mtry"] = len(model.X)
    fun = get_model_fun(model)[0]
    query = "SELECT {}('{}', '{}', '{}', '{}' USING PARAMETERS "
    query = query.format(fun, model.name, input_relation, model.y, ", ".join(model.X))
    query += ", ".join(
        ["{} = {}".format(elem, parameters[elem]) for elem in parameters]
    )
    if alpha != None:
        query += ", alpha = {}".format(alpha)
    query += ")"
    model.cursor.execute(query)
    if model.type in (
        "LinearSVC",
        "LinearSVR",
        "LogisticRegression",
        "LinearRegression",
    ):
        model.coef = to_tablesample(
            query="SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'details')".format(
                model.name
            ),
            cursor=model.cursor,
        )
        model.coef.table_info = False
    elif model.type in ("RandomForestClassifier", "MultinomialNB"):
        model.cursor.execute(
            "SELECT DISTINCT {} FROM {} WHERE {} IS NOT NULL ORDER BY 1".format(
                model.y, input_relation, model.y
            )
        )
        classes = model.cursor.fetchall()
        model.classes = [item[0] for item in classes]
    return model


# ---#
def fit_unsupervised(model, input_relation: str, X: list):
    """
---------------------------------------------------------------------------
Trains the model.

Parameters
----------
input_relation: str
	Train relation.
X: list
	List of the predictors.

Returns
-------
object
	model
	"""
    check_types(
        [("input_relation", input_relation, [str], False), ("X", X, [list], False)]
    )
    model.input_relation = input_relation
    model.X = [str_column(column) for column in X]
    parameters = vertica_param_dict(model)
    if "num_components" in parameters and not (parameters["num_components"]):
        del parameters["num_components"]
    fun = get_model_fun(model)[0]
    query = "SELECT {}('{}', '{}', '{}'".format(
        fun, model.name, input_relation, ", ".join(model.X)
    )
    if model.type == "KMeans":
        query += ", {}".format(parameters["n_cluster"])
    elif model.type == "Normalizer":
        query += ", {}".format(parameters["method"])
        del parameters["method"]
    if model.type != "Normalizer":
        query += " USING PARAMETERS "
    if "init_method" in parameters and type(parameters["init_method"]) != str:
        schema = schema_relation(model.name)[0]
        name = "VERTICAPY_KMEANS_INITIAL"
        del parameters["init_method"]
        try:
            model.cursor.execute("DROP TABLE IF EXISTS {}.{}".format(schema, name))
        except:
            pass
        if len(model.parameters["init"]) != model.parameters["n_cluster"]:
            raise ParameterError(
                "'init' must be a list of 'n_cluster' = {} points".format(
                    model.parameters["n_cluster"]
                )
            )
        else:
            for item in model.parameters["init"]:
                if len(X) != len(item):
                    raise ParameterError(
                        "Each points of 'init' must be of size len(X) = {}".format(
                            len(model.X)
                        )
                    )
            query0 = []
            for i in range(len(model.parameters["init"])):
                line = []
                for j in range(len(model.parameters["init"][0])):
                    line += [str(model.parameters["init"][i][j]) + " AS " + X[j]]
                line = ",".join(line)
                query0 += ["SELECT " + line]
            query0 = " UNION ".join(query0)
            query0 = "CREATE TABLE {}.{} AS {}".format(schema, name, query0)
            model.cursor.execute(query0)
            query += "initial_centers_table = '{}.{}', ".format(schema, name)
    elif "init_method" in parameters:
        del parameters["init_method"]
        query += "init_method = '{}', ".format(model.parameters["init"])
    query += ", ".join(
        ["{} = {}".format(elem, parameters[elem]) for elem in parameters]
    )
    query += ")"
    model.cursor.execute(query)
    if model.type == "KMeans":
        try:
            model.cursor.execute("DROP TABLE IF EXISTS {}.{}".format(schema, name))
        except:
            pass
    if model.type in ("KMeans"):
        model.cluster_centers = to_tablesample(
            query="SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'centers')".format(
                model.name
            ),
            cursor=model.cursor,
        )
        model.cluster_centers.table_info = False
        query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'metrics')".format(
            model.name
        )
        model.cursor.execute(query)
        result = model.cursor.fetchone()[0]
        values = {
            "index": [
                "Between-Cluster Sum of Squares",
                "Total Sum of Squares",
                "Total Within-Cluster Sum of Squares",
                "Between-Cluster SS / Total SS",
                "converged",
            ]
        }
        values["value"] = [
            float(result.split("Between-Cluster Sum of Squares: ")[1].split("\n")[0]),
            float(result.split("Total Sum of Squares: ")[1].split("\n")[0]),
            float(
                result.split("Total Within-Cluster Sum of Squares: ")[1].split("\n")[0]
            ),
            float(result.split("Between-Cluster Sum of Squares: ")[1].split("\n")[0])
            / float(result.split("Total Sum of Squares: ")[1].split("\n")[0]),
            result.split("Converged: ")[1].split("\n")[0] == "True",
        ]
        model.metrics = tablesample(values, table_info=False)
    elif model.type in ("PCA"):
        model.components = to_tablesample(
            query="SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'principal_components')".format(
                model.name
            ),
            cursor=model.cursor,
        )
        model.components.table_info = False
        model.explained_variance = to_tablesample(
            query="SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'singular_values')".format(
                model.name
            ),
            cursor=model.cursor,
        )
        model.explained_variance.table_info = False
        model.mean = to_tablesample(
            query="SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'columns')".format(
                model.name
            ),
            cursor=model.cursor,
        )
        model.mean.table_info = False
    elif model.type in ("SVD"):
        model.singular_values = to_tablesample(
            query="SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'right_singular_vectors')".format(
                model.name
            ),
            cursor=model.cursor,
        )
        model.singular_values.table_info = False
        model.explained_variance = to_tablesample(
            query="SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'singular_values')".format(
                model.name
            ),
            cursor=model.cursor,
        )
        model.explained_variance.table_info = False
    elif model.type in ("Normalizer"):
        model.param = to_tablesample(
            query="SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'details')".format(
                model.name
            ),
            cursor=model.cursor,
        )
        model.param.table_info = False
    elif model.type == "OneHotEncoder":
        try:
            model.param = to_tablesample(
                query="SELECT category_name, category_level::varchar, category_level_index FROM (SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'integer_categories')) x UNION ALL SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'varchar_categories')".format(
                    model.name, model.name
                ),
                cursor=model.cursor,
            )
        except:
            try:
                model.param = to_tablesample(
                    query="SELECT category_name, category_level::varchar, category_level_index FROM (SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'integer_categories')) x".format(
                        model.name
                    ),
                    cursor=model.cursor,
                )
            except:
                model.param = to_tablesample(
                    query="SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'varchar_categories')".format(
                        model.name
                    ),
                    cursor=model.cursor,
                )
        model.param.table_info = False
    return model


# ---#
def get_model_summary(model):
    """
---------------------------------------------------------------------------
Returns the model summary.

Returns
-------
str
	model summary.
	"""
    model.cursor.execute(
        "SELECT GET_MODEL_SUMMARY(USING PARAMETERS model_name = '" + model.name + "')"
    )
    return model.cursor.fetchone()[0]


# ---#
def get_model_fun(model):
    """
---------------------------------------------------------------------------
Returns the Vertica associated functions.

Returns
-------
tuple
	(FIT,PREDICT,INVERSE)
	"""
    if model.type == "LinearRegression":
        return ("LINEAR_REG", "PREDICT_LINEAR_REG", "")
    elif model.type == "LogisticRegression":
        return ("LOGISTIC_REG", "PREDICT_LOGISTIC_REG", "")
    elif model.type == "LinearSVC":
        return ("SVM_CLASSIFIER", "PREDICT_SVM_CLASSIFIER", "")
    elif model.type == "LinearSVR":
        return ("SVM_REGRESSOR", "PREDICT_SVM_REGRESSOR", "")
    elif model.type == "RandomForestRegressor":
        return ("RF_REGRESSOR", "PREDICT_RF_REGRESSOR", "")
    elif model.type == "RandomForestClassifier":
        return ("RF_CLASSIFIER", "PREDICT_RF_CLASSIFIER", "")
    elif model.type == "MultinomialNB":
        return ("NAIVE_BAYES", "PREDICT_NAIVE_BAYES", "")
    elif model.type == "KMeans":
        return ("KMEANS", "APPLY_KMEANS", "")
    elif model.type == "PCA":
        return ("PCA", "APPLY_PCA", "APPLY_INVERSE_PCA")
    elif model.type == "SVD":
        return ("SVD", "APPLY_SVD", "APPLY_INVERSE_SVD")
    elif model.type == "Normalizer":
        return ("NORMALIZE_FIT", "APPLY_NORMALIZE", "REVERSE_NORMALIZE")
    elif model.type == "OneHotEncoder":
        return ("ONE_HOT_ENCODER_FIT", "APPLY_ONE_HOT_ENCODER", "")


# ---#
def get_model_repr(model):
    try:
        return get_model_summary(model)
    except:
        return "<{}>".format(model.type)


# ---#
def get_params(model):
    """
---------------------------------------------------------------------------
Returns the model Parameters.

Returns
-------
dict
	model parameters
	"""
    return model.parameters


# ---#
def get_tree(model, tree_id: int = 0):
    """
---------------------------------------------------------------------------
Returns a table with all the input tree information.

Parameters
----------
tree_id: int, optional
	Unique tree identifier. It is an integer between 0 and n_estimators - 1

Returns
-------
tablesample
		An object containing the result. For more information, see
		utilities.tablesample.
	"""
    check_types([("tree_id", tree_id, [int, float], False)])
    query = "SELECT READ_TREE ( USING PARAMETERS model_name = '{}', tree_id = {}, format = 'tabular');".format(
        model.name, tree_id
    )
    result = to_tablesample(query=query, cursor=model.cursor)
    result.table_info = False
    return result


# ---#
def inverse_transform(model, vdf=None, X: list = [], key_columns: list = []):
    """
---------------------------------------------------------------------------
Applies the Inverse Model on a vDataFrame.

Parameters
----------
vdf: vDataFrame, optional
	input vDataFrame.
X: list, optional
	List of the input vcolumns.
key_columns: list, optional
	Predictors to keep unchanged during the transformation.

Returns
-------
vDataFrame
	object result of the model transformation.
	"""
    check_types([("key_columns", key_columns, [list], False), ("X", X, [list], False)])
    if vdf:
        check_types(vdf=["vdf", vdf])
        X = vdf_columns_names(X, vdf)
        relation = vdf.__genSQL__()
    else:
        relation = model.input_relation
        X = [str_column(elem) for elem in X]
    main_relation = "(SELECT {} FROM {}) x".format(
        model.deployInverseSQL(key_columns, model.X if not (X) else X), relation
    )
    return vdf_from_relation(main_relation, "Inverse Transformation", model.cursor,)


# ---#
def inverse_transform_preprocessing(model, vdf=None, X: list = []):
    """
---------------------------------------------------------------------------
Creates a vDataFrame of the model.

Parameters
----------
vdf: vDataFrame, optional
	input vDataFrame.
X: list, optional
	List of the input vcolumns.

Returns
-------
vDataFrame
	object result of the model transformation.
	"""
    check_types([("X", X, [list], False)])
    if vdf:
        check_types(vdf=["vdf", vdf])
        X = vdf_columns_names(X, vdf)
        relation = vdf.__genSQL__()
    else:
        relation = model.input_relation
        X = [str_column(elem) for elem in X]
    return vdf_from_relation(
        "(SELECT {} FROM {}) x".format(
            model.deployInverseSQL(model.X if not (X) else X), relation
        ),
        model.name,
        model.cursor,
    )


# ---#
def lift_chart_binary(model):
    """
---------------------------------------------------------------------------
Draws the model Lift Chart.

Returns
-------
tablesample
		An object containing the result. For more information, see
		utilities.tablesample.
	"""
    return lift_chart(model.y, model.deploySQL(), model.test_relation, model.cursor)


# ---#
def lift_chart_multiclass(model, pos_label=None):
    """
---------------------------------------------------------------------------
Draws the model Lift Chart.

Parameters
----------
pos_label: int/float/str, optional
	To draw a lift chart, one of the response column class has to be the 
	positive one. The parameter 'pos_label' represents this class.

Returns
-------
tablesample
		An object containing the result. For more information, see
		utilities.tablesample.
	"""
    pos_label = (
        model.classes[1]
        if (pos_label == None and len(model.classes) == 2)
        else pos_label
    )
    if pos_label not in model.classes:
        raise ParameterError("'pos_label' must be one of the response column classes")
    return lift_chart(
        model.y,
        model.deploySQL(allSQL=True)[0].format(pos_label),
        model.test_relation,
        model.cursor,
        pos_label,
    )


# ---#
def multiclass_classification_score(
    model, method: str = "accuracy", pos_label=None, cutoff: float = -1
):
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
	If the parameter is not between 0 and 1, an automatic cutoff is 
	computed.
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
    check_types(
        [("cutoff", cutoff, [int, float], False), ("method", method, [str], False)]
    )
    pos_label = (
        model.classes[1]
        if (pos_label == None and len(model.classes) == 2)
        else pos_label
    )
    if (pos_label not in model.classes) and (method != "accuracy"):
        raise ParameterError("'pos_label' must be one of the response column classes")
    elif (cutoff >= 1 or cutoff <= 0) and (method != "accuracy"):
        cutoff = model.score("best_cutoff", pos_label, 0.5)
    if method in ("accuracy", "acc"):
        return accuracy_score(
            model.y,
            model.deploySQL(pos_label, cutoff),
            model.test_relation,
            model.cursor,
            pos_label,
        )
    elif method == "auc":
        return auc(
            "DECODE({}, '{}', 1, 0)".format(model.y, pos_label),
            model.deploySQL(allSQL=True)[0].format(pos_label),
            model.test_relation,
            model.cursor,
        )
    elif method == "prc_auc":
        return prc_auc(
            "DECODE({}, '{}', 1, 0)".format(model.y, pos_label),
            model.deploySQL(allSQL=True)[0].format(pos_label),
            model.test_relation,
            model.cursor,
        )
    elif method in ("best_cutoff", "best_threshold"):
        return roc_curve(
            "DECODE({}, '{}', 1, 0)".format(model.y, pos_label),
            model.deploySQL(allSQL=True)[0].format(pos_label),
            model.test_relation,
            model.cursor,
            best_threshold=True,
        )
    elif method in ("recall", "tpr"):
        return recall_score(
            model.y,
            model.deploySQL(pos_label, cutoff),
            model.test_relation,
            model.cursor,
        )
    elif method in ("precision", "ppv"):
        return precision_score(
            model.y,
            model.deploySQL(pos_label, cutoff),
            model.test_relation,
            model.cursor,
        )
    elif method in ("specificity", "tnr"):
        return specificity_score(
            model.y,
            model.deploySQL(pos_label, cutoff),
            model.test_relation,
            model.cursor,
        )
    elif method in ("negative_predictive_value", "npv"):
        return precision_score(
            model.y,
            model.deploySQL(pos_label, cutoff),
            model.test_relation,
            model.cursor,
        )
    elif method in ("log_loss", "logloss"):
        return log_loss(
            "DECODE({}, '{}', 1, 0)".format(model.y, pos_label),
            model.deploySQL(allSQL=True)[0].format(pos_label),
            model.test_relation,
            model.cursor,
        )
    elif method == "f1":
        return f1_score(
            model.y,
            model.deploySQL(pos_label, cutoff),
            model.test_relation,
            model.cursor,
        )
    elif method == "mcc":
        return matthews_corrcoef(
            model.y,
            model.deploySQL(pos_label, cutoff),
            model.test_relation,
            model.cursor,
        )
    elif method in ("bm", "informedness"):
        return informedness(
            model.y,
            model.deploySQL(pos_label, cutoff),
            model.test_relation,
            model.cursor,
        )
    elif method in ("mk", "markedness"):
        return markedness(
            model.y,
            model.deploySQL(pos_label, cutoff),
            model.test_relation,
            model.cursor,
        )
    elif method in ("csi", "critical_success_index"):
        return critical_success_index(
            model.y,
            model.deploySQL(pos_label, cutoff),
            model.test_relation,
            model.cursor,
        )
    else:
        raise ParameterError(
            "The parameter 'method' must be in accuracy|auc|prc_auc|best_cutoff|recall|precision|log_loss|negative_predictive_value|specificity|mcc|informedness|markedness|critical_success_index"
        )


# ---#
def plot_model(model, max_nb_points: int = 100):
    """
---------------------------------------------------------------------------
Draws the Model.

Parameters
----------
max_nb_points: int
	Maximum number of points to display.

Returns
-------
Figure
	Matplotlib Figure
	"""
    check_types([("max_nb_points", max_nb_points, [int, float], False)])
    if model.type in (
        "LinearRegression",
        "LogisticRegression",
        "LinearSVC",
        "LinearSVR",
    ):
        coefficients = model.coef.values["coefficient"]
        if model.type == "LogisticRegression":
            return logit_plot(
                model.X,
                model.y,
                model.input_relation,
                coefficients,
                model.cursor,
                max_nb_points,
            )
        elif model.type == "LinearSVC":
            return svm_classifier_plot(
                model.X,
                model.y,
                model.input_relation,
                coefficients,
                model.cursor,
                max_nb_points,
            )
        else:
            return regression_plot(
                model.X,
                model.y,
                model.input_relation,
                coefficients,
                model.cursor,
                max_nb_points,
            )
    elif model.type in ("KMeans"):
        vdf = vDataFrame(model.input_relation, model.cursor)
        model.predict(vdf, name="kmeans_cluster")
        if len(model.X) <= 3:
            return vdf.scatter(
                columns=model.X,
                catcol="kmeans_cluster",
                max_cardinality=100,
                max_nb_points=max_nb_points,
            )
        else:
            raise Exception("Clustering Plots are only available in 2D or 3D")


# ---#
def plot_model_voronoi(model):
    """
---------------------------------------------------------------------------
Draws the Voronoi Graph of the model.

Returns
-------
Figure
	Matplotlib Figure
	"""
    if len(model.X) == 2:
        from verticapy.learn.plot import voronoi_plot

        query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'centers')".format(
            model.name
        )
        model.cursor.execute(query)
        clusters = model.cursor.fetchall()
        return voronoi_plot(clusters=clusters, columns=model.X)
    else:
        raise Exception("Voronoi Plots are only available in 2D")


# ---#
def plot_rf_tree(model, tree_id: int = 0, pic_path: str = ""):
    """
---------------------------------------------------------------------------
Draws the input tree. The module anytree must be installed in the machine.

Parameters
----------
tree_id: int, optional
	Unique tree identifier. It is an integer between 0 and n_estimators - 1
pic_path: str, optional
	Absolute path to save the image of the tree.
	"""
    check_types(
        [
            ("tree_id", tree_id, [int, float], False),
            ("pic_path", pic_path, [str], False),
        ]
    )
    plot_tree(
        model.get_tree(tree_id=tree_id).values, metric="variance", pic_path=pic_path
    )


# ---#
def prc_curve_binary(model):
    """
---------------------------------------------------------------------------
Draws the model PRC curve.

Returns
-------
tablesample
		An object containing the result. For more information, see
		utilities.tablesample.
	"""
    return prc_curve(model.y, model.deploySQL(), model.test_relation, model.cursor)


# ---#
def prc_curve_multiclass(model, pos_label=None):
    """
---------------------------------------------------------------------------
Draws the model PRC curve.

Parameters
----------
pos_label: int/float/str, optional
	To draw the PRC curve, one of the response column class has to be the 
	positive one. The parameter 'pos_label' represents this class.

Returns
-------
tablesample
		An object containing the result. For more information, see
		utilities.tablesample.
	"""
    pos_label = (
        model.classes[1]
        if (pos_label == None and len(model.classes) == 2)
        else pos_label
    )
    if pos_label not in model.classes:
        raise ParameterError("'pos_label' must be one of the response column classes")
    return prc_curve(
        model.y,
        model.deploySQL(allSQL=True)[0].format(pos_label),
        model.test_relation,
        model.cursor,
        pos_label,
    )


# ---#
def predict(model, vdf, X: list = [], name: str = "", inplace: bool = True):
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
inplace: bool, optional
	If set to True, the prediction will be added to the vDataFrame.

Returns
-------
vDataFrame
	the input object.
	"""
    check_types(
        [("name", name, [str], False), ("X", X, [list], False)], vdf=["vdf", vdf]
    )
    X = [str_column(elem) for elem in X]
    name = (
        "{}_".format(model.type) + "".join(ch for ch in model.name if ch.isalnum())
        if not (name)
        else name
    )
    if inplace:
        return vdf.eval(name, model.deploySQL(X=X))
    else:
        return vdf.copy().eval(name, model.deploySQL(X=X))


# ---#
def predict_binary(
    model, vdf, X: list = [], name: str = "", cutoff: float = -1, inplace: bool = True
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
	Probability cutoff.
inplace: bool, optional
	If set to True, the prediction will be added to the vDataFrame.

Returns
-------
vDataFrame
	the input object.
	"""
    check_types(
        [
            ("name", name, [str], False),
            ("cutoff", cutoff, [int, float], False),
            ("X", X, [list], False),
        ],
        vdf=["vdf", vdf],
    )
    X = [str_column(elem) for elem in X]
    name = (
        "{}_".format(model.type) + "".join(ch for ch in model.name if ch.isalnum())
        if not (name)
        else name
    )
    if inplace:
        return vdf.eval(name, model.deploySQL(cutoff=cutoff, X=X))
    else:
        return vdf.copy().eval(name, model.deploySQL(cutoff=cutoff, X=X))


# ---#
def predict_multiclass(
    model,
    vdf,
    X: list = [],
    name: str = "",
    cutoff: float = -1,
    pos_label=None,
    inplace: bool = True,
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
	Cutoff for which the tested category will be accepted as prediction. 
	If the parameter is not between 0 and 1, the class probability will
	be returned.
pos_label: int/float/str, optional
	Class label.
inplace: bool, optional
	If set to True, the prediction will be added to the vDataFrame.

Returns
-------
vDataFrame
	the input object.
	"""
    check_types(
        [
            ("name", name, [str], False),
            ("cutoff", cutoff, [int, float], False),
            ("X", X, [list], False),
        ],
        vdf=["vdf", vdf],
    )
    X = [str_column(elem) for elem in X]
    name = (
        "{}_".format(model.type) + "".join(ch for ch in model.name if ch.isalnum())
        if not (name)
        else name
    )
    if len(model.classes) == 2 and pos_label == None:
        pos_label = model.classes[1]
    if inplace:
        return vdf.eval(name, model.deploySQL(pos_label=pos_label, cutoff=cutoff, X=X))
    else:
        return vdf.copy().eval(
            name, model.deploySQL(pos_label=pos_label, cutoff=cutoff, X=X)
        )


# ---#
def regression_metrics_report(model):
    """
---------------------------------------------------------------------------
Computes a regression report using multiple metrics to evaluate the model
(r2, mse, max error...). 

Returns
-------
tablesample
	An object containing the result. For more information, see
	utilities.tablesample.
	"""
    return regression_report(
        model.y, model.deploySQL(), model.test_relation, model.cursor
    )


# ---#
def regression_score(model, method: str = "r2"):
    """
---------------------------------------------------------------------------
Computes the model score.

Parameters
----------
method: str, optional
	The method to use to compute the score.
		max    : Max Error
		mae    : Mean Absolute Error
		median : Median Absolute Error
		mse    : Mean Squared Error
		msle   : Mean Squared Log Error
		r2     : R squared coefficient
		var    : Explained Variance 

Returns
-------
float
	score
	"""
    check_types([("method", method, [str], False)])
    if model.category.lower() == "regressor":
        if method in ("r2", "rsquared"):
            return r2_score(
                model.y, model.deploySQL(), model.test_relation, model.cursor
            )
        elif method in ("mae", "mean_absolute_error"):
            return mean_absolute_error(
                model.y, model.deploySQL(), model.test_relation, model.cursor
            )
        elif method in ("mse", "mean_squared_error"):
            return mean_squared_error(
                model.y, model.deploySQL(), model.test_relation, model.cursor
            )
        elif method in ("msle", "mean_squared_log_error"):
            return mean_squared_log_error(
                model.y, model.deploySQL(), model.test_relation, model.cursor
            )
        elif method in ("max", "max_error"):
            return max_error(
                model.y, model.deploySQL(), model.test_relation, model.cursor
            )
        elif method in ("median", "median_absolute_error"):
            return median_absolute_error(
                model.y, model.deploySQL(), model.test_relation, model.cursor
            )
        elif method in ("var", "explained_variance"):
            return explained_variance(
                model.y, model.deploySQL(), model.test_relation, model.cursor
            )
        else:
            raise ParameterError(
                "The parameter 'method' must be in r2|mae|mse|msle|max|median|var"
            )


# ---#
def roc_curve_binary(model):
    """
---------------------------------------------------------------------------
Draws the model ROC curve.

Returns
-------
tablesample
		An object containing the result. For more information, see
		utilities.tablesample.
	"""
    return roc_curve(model.y, model.deploySQL(), model.test_relation, model.cursor)


# ---#
def roc_curve_multiclass(model, pos_label=None):
    """
---------------------------------------------------------------------------
Draws the model ROC curve.

Parameters
----------
pos_label: int/float/str, optional
	To draw the ROC curve, one of the response column class has to be the 
	positive one. The parameter 'pos_label' represents this class.

Returns
-------
tablesample
		An object containing the result. For more information, see
		utilities.tablesample.
	"""
    pos_label = (
        model.classes[1]
        if (pos_label == None and len(model.classes) == 2)
        else pos_label
    )
    if pos_label not in model.classes:
        raise ParameterError("'pos_label' must be one of the response column classes")
    return roc_curve(
        model.y,
        model.deploySQL(allSQL=True)[0].format(pos_label),
        model.test_relation,
        model.cursor,
        pos_label,
    )


# ---#
def set_params(model, parameters: dict = {}):
    """
---------------------------------------------------------------------------
Sets the parameters of the model.

Parameters
----------
parameters: dict, optional
	New parameters.
	"""
    check_types([("parameters", parameters, [dict], False)])
    for param in parameters:
        model.parameters[param] = parameters[param]


# ---#
def transform(
    model,
    vdf=None,
    X: list = [],
    n_components: int = 0,
    cutoff: float = 1,
    key_columns: list = [],
):
    """
---------------------------------------------------------------------------
Applies the model on a vDataFrame.

Parameters
----------
vdf: vDataFrame, optional
	Input vDataFrame.
X: list, optional
	List of the input vcolumns.
n_components: int, optional
	Number of components to return. If set to 0, all the components will 
	be deployed.
cutoff: float, optional
	Specifies the minimum accumulated explained variance. Components are 
	taken until the accumulated explained variance reaches this value.
key_columns: list, optional
	Predictors to keep unchanged during the transformation.

Returns
-------
vDataFrame
	object result of the model transformation.
	"""
    check_types(
        [
            ("n_components", n_components, [int, float], False),
            ("cutoff", cutoff, [int, float], False),
            ("key_columns", key_columns, [list], False),
            ("X", X, [list], False),
        ]
    )
    if vdf:
        check_types(vdf=["vdf", vdf])
        X = vdf_columns_names(X, vdf)
        relation = vdf.__genSQL__()
    else:
        relation = model.input_relation
        X = [str_column(elem) for elem in X]
    main_relation = "(SELECT {} FROM {}) x".format(
        model.deploySQL(n_components, cutoff, key_columns, model.X if not (X) else X),
        relation,
    )
    return vdf_from_relation(main_relation, "Transformation", model.cursor,)


# ---#
def transform_preprocessing(model, vdf=None, X: list = []):
    """
---------------------------------------------------------------------------
Applies the model on a vDataFrame.

Parameters
----------
vdf: vDataFrame, optional
	Input vDataFrame.
X: list, optional
	List of the input vcolumns.

Returns
-------
vDataFrame
	object result of the model transformation.
	"""
    check_types([("X", X, [list], False)])
    if vdf:
        check_types(vdf=["vdf", vdf])
        X = vdf_columns_names(X, vdf)
        relation = vdf.__genSQL__()
    else:
        relation = model.input_relation
        X = [str_column(elem) for elem in X]
    return vdf_from_relation(
        "(SELECT {} FROM {}) x".format(
            model.deploySQL(model.X if not (X) else X), relation
        ),
        model.name,
        model.cursor,
    )


# ---#
def vertica_param_name(param: str):
    """
---------------------------------------------------------------------------
Returns the Vertica Param name.

Parameters
----------
param: str
	VerticaPy Model Parameter

Returns
-------
str
	Vertica Param Name.
	"""
    if param.lower() == "solver":
        return "optimizer"
    elif param.lower() == "tol":
        return "epsilon"
    elif param.lower() == "max_iter":
        return "max_iterations"
    elif param.lower() == "penalty":
        return "regularization"
    elif param.lower() == "C":
        return "lambda"
    elif param.lower() == "l1_ratio":
        return "alpha"
    elif param.lower() == "n_estimators":
        return "ntree"
    elif param.lower() == "max_features":
        return "mtry"
    elif param.lower() == "sample":
        return "sampling_size"
    elif param.lower() == "max_leaf_nodes":
        return "max_breadth"
    elif param.lower() == "min_samples_leaf":
        return "min_samples_leaf"
    elif param.lower() == "n_components":
        return "num_components"
    elif param.lower() == "init":
        return "init_method"
    else:
        return param


# ---#
def vertica_param_dict(model):
    """
---------------------------------------------------------------------------
Returns the Vertica Param dict.

Returns
-------
dict
	Vertica Param Dictionary.
	"""
    parameters = {}
    for param in model.parameters:
        if model.type in ("LinearSVC", "LinearSVR") and param == "C":
            parameters[param] = model.parameters[param]
        elif param == "max_leaf_nodes":
            parameters[vertica_param_name(param)] = int(model.parameters[param])
        elif param == "class_weight":
            parameters[param] = "'{}'".format(
                ", ".join([str(item) for item in model.parameters[param]])
            )
        elif type(model.parameters[param]) in (str, dict):
            parameters[vertica_param_name(param)] = "'{}'".format(
                model.parameters[param]
            )
        else:
            parameters[vertica_param_name(param)] = model.parameters[param]
    return parameters
