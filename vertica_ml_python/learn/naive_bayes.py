# (c) Copyright [2018] Micro Focus or one of its affiliates. 
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
# AUTHOR: BADR OUALI
#
############################################################################################################ 
#  __ __   ___ ____  ______ ____   __  ____      ___ ___ _          ____  __ __ ______ __ __  ___  ____    #
# |  |  | /  _|    \|      |    | /  ]/    |    |   |   | |        |    \|  |  |      |  |  |/   \|    \   #
# |  |  |/  [_|  D  |      ||  | /  /|  o  |    | _   _ | |        |  o  |  |  |      |  |  |     |  _  |  #
# |  |  |    _|    /|_|  |_||  |/  / |     |    |  \_/  | |___     |   _/|  ~  |_|  |_|  _  |  O  |  |  |  #
# |  :  |   [_|    \  |  |  |  /   \_|  _  |    |   |   |     |    |  |  |___, | |  | |  |  |     |  |  |  #
#  \   /|     |  .  \ |  |  |  \     |  |  |    |   |   |     |    |  |  |     | |  | |  |  |     |  |  |  #
#   \_/ |_____|__|\_| |__| |____\____|__|__|    |___|___|_____|    |__|  |____/  |__| |__|__|\___/|__|__|  #
#                                                                                                          #
############################################################################################################
# Vertica-ML-Python allows user to create Virtual Dataframe. vDataframes simplify   #
# data exploration,   data cleaning   and   machine   learning   in    Vertica.     #
# It is an object which keeps in it all the actions that the user wants to achieve  # 
# and execute them when they are needed.    										#
#																					#
# The purpose is to bring the logic to the data and not the opposite                #
#####################################################################################
#
# Libraries
from vertica_ml_python import drop_model
from vertica_ml_python import tablesample
from vertica_ml_python import to_tablesample

from vertica_ml_python.learn.metrics import accuracy_score
from vertica_ml_python.learn.metrics import auc
from vertica_ml_python.learn.metrics import prc_auc
from vertica_ml_python.learn.metrics import log_loss
from vertica_ml_python.learn.metrics import classification_report
from vertica_ml_python.learn.metrics import confusion_matrix
from vertica_ml_python.learn.metrics import critical_success_index
from vertica_ml_python.learn.metrics import f1_score
from vertica_ml_python.learn.metrics import informedness
from vertica_ml_python.learn.metrics import markedness
from vertica_ml_python.learn.metrics import matthews_corrcoef
from vertica_ml_python.learn.metrics import multilabel_confusion_matrix
from vertica_ml_python.learn.metrics import negative_predictive_score
from vertica_ml_python.learn.metrics import precision_score
from vertica_ml_python.learn.metrics import recall_score
from vertica_ml_python.learn.metrics import specificity_score

from vertica_ml_python.learn.plot import lift_chart
from vertica_ml_python.learn.plot import roc_curve
from vertica_ml_python.learn.plot import prc_curve

from vertica_ml_python.utilities import str_column

#
class MultinomialNB:
	#
	def  __init__(self,
				  name: str,
				  cursor,
				  alpha: float = 1.0):
		self.type = "classifier"
		self.cursor = cursor
		self.name = name
		self.alpha = alpha
	# 
	def __repr__(self):
		try:
			self.cursor.execute("SELECT GET_MODEL_SUMMARY(USING PARAMETERS model_name = '" + self.name + "')")
			return (self.cursor.fetchone()[0])
		except:
			return "<MultinomialNB>"
	#
	#
	#
	# METHODS
	# 
	#
	def add_to_vdf(self,
				   vdf,
				   name: str = "",
				   cutoff: float = 0.5):
		name = "MultinomialNB_" + self.name if not (name) else name
		pos_label = self.classes[1] if (len(self.classes) == 2) else None
		return (vdf.eval(name, self.deploySQL(pos_label, cutoff)))
	#
	def classification_report(self, cutoff: float = 0.5, labels = []):
		labels = self.classes if not(labels) else labels
		return (classification_report(cutoff = cutoff, estimator = self, labels = labels))
	#
	def confusion_matrix(self, pos_label = None, cutoff: float = 0.5):
		pos_label = self.classes[1] if (pos_label == None and len(self.classes) == 2) else pos_label
		if (pos_label in self.classes and cutoff < 1 and cutoff > 0):
			return (confusion_matrix(self.y, self.deploySQL(pos_label, cutoff), self.test_relation, self.cursor, pos_label = pos_label))
		else:
			return (multilabel_confusion_matrix(self.y, self.deploySQL(), self.test_relation, self.cursor, self.classes))
	#
	def deploySQL(self, pos_label = None, cutoff: float = -1, allSQL: bool = False):
		if (allSQL):
			sql = "PREDICT_NAIVE_BAYES({} USING PARAMETERS model_name = '{}', class = '{}', type = 'probability', match_by_pos = 'true')".format(", ".join(self.X), self.name, "{}")
			sql = [sql, "PREDICT_NAIVE_BAYES({} USING PARAMETERS model_name = '{}', match_by_pos = 'true')".format(", ".join(self.X), self.name)]
		else:
			if (pos_label in self.classes and cutoff <= 1 and cutoff >= 0):
				sql = "PREDICT_NAIVE_BAYES({} USING PARAMETERS model_name = '{}', class = '{}', type = 'probability', match_by_pos = 'true')".format(", ".join(self.X), self.name, pos_label)
				if (len(self.classes) > 2):
					sql = "(CASE WHEN {} >= {} THEN '{}' WHEN {} IS NULL THEN NULL ELSE 'Non-{}' END)".format(sql, cutoff, pos_label, sql, pos_label)
				else:
					non_pos_label = self.classes[0] if (self.classes[0] != pos_label) else self.classes[1]
					sql = "(CASE WHEN {} >= {} THEN '{}' WHEN {} IS NULL THEN NULL ELSE '{}' END)".format(sql, cutoff, pos_label, sql, non_pos_label)
			elif (pos_label in self.classes):
				sql = "PREDICT_NAIVE_BAYES({} USING PARAMETERS model_name = '{}', class = '{}', type = 'probability', match_by_pos = 'true')".format(", ".join(self.X), self.name, pos_label)
			else:
				sql = "PREDICT_NAIVE_BAYES({} USING PARAMETERS model_name = '{}', match_by_pos = 'true')".format(", ".join(self.X), self.name)
		return (sql)
	#
	def deploy_to_DB(self, name: str, view: bool = True, cutoff = -1, all_classes: bool = False):
		relation = "TABLE" if not(view) else "VIEW"
		sql = "CREATE {} {} AS SELECT {}, {} FROM {}".format(relation, name, ", ".join(self.X), "{}", self.test_relation)
		if (all_classes):
			predict = []
			for elem in self.classes:
				if elem not in (self.classes):
					raise ValueError("All the elements of 'pos_label' must be in the estimator classes")
				alias = '"{}_{}"'.format(self.y.replace('"', ''), elem) 
				predict += ["{} AS {}".format(self.deploySQL(elem), alias)]
			predict += ["{} AS {}".format(self.deploySQL(), self.y)]
		else:
			if (len(self.classes) == 2):
				predict = ["{} AS {}".format(self.deploySQL(self.classes[1], cutoff), self.y)]
			else:
				predict = ["{} AS {}".format(self.deploySQL(), self.y)]
		self.cursor.execute(sql.format(", ".join(predict)))
	#
	def drop(self):
		drop_model(self.name, self.cursor, print_info = False)
	#
	def fit(self,
			input_relation: str, 
			X: list, 
			y: str,
			test_relation: str = ""):
		self.input_relation = input_relation
		self.test_relation = test_relation if (test_relation) else input_relation
		self.X = [str_column(column) for column in X]
		self.y = str_column(y)
		query = "SELECT NAIVE_BAYES('{}', '{}', '{}', '{}' USING PARAMETERS alpha = {})".format(self.name, input_relation, self.y, ", ".join(self.X), self.alpha)
		self.cursor.execute(query)
		self.cursor.execute("SELECT DISTINCT {} FROM {} WHERE {} IS NOT NULL ORDER BY 1".format(self.y, input_relation, self.y))
		classes = self.cursor.fetchall()
		self.classes = [item[0] for item in classes]
		return (self)
	#
	def lift_chart(self, pos_label = None):
		pos_label = self.classes[1] if (pos_label == None and len(self.classes) == 2) else pos_label
		if (pos_label not in self.classes):
			raise ValueError("'pos_label' must be one of the response column classes")
		return (lift_chart(self.y, self.deploySQL(allSQL = True)[0].format(pos_label), self.test_relation, self.cursor, pos_label))
	#
	def prc_curve(self, pos_label = None):
		pos_label = self.classes[1] if (pos_label == None and len(self.classes) == 2) else pos_label
		if (pos_label not in self.classes):
			raise ValueError("'pos_label' must be one of the response column classes")
		return (prc_curve(self.y, self.deploySQL(allSQL = True)[0].format(pos_label), self.test_relation, self.cursor, pos_label))
	#
	def roc_curve(self, pos_label = None):
		pos_label = self.classes[1] if (pos_label == None and len(self.classes) == 2) else pos_label
		if (pos_label not in self.classes):
			raise ValueError("'pos_label' must be one of the response column classes")
		return (roc_curve(self.y, self.deploySQL(allSQL = True)[0].format(pos_label), self.test_relation, self.cursor, pos_label))
	#
	def score(self, pos_label = None, cutoff: float = 0.5, method: str = "accuracy"):
		pos_label = self.classes[1] if (pos_label == None and len(self.classes) == 2) else pos_label
		if (pos_label not in self.classes):
			raise ValueError("'pos_label' must be one of the response column classes")
		elif (cutoff >= 1 or cutoff <= 0):
			raise ValueError("'cutoff' must be in ]0;1[")
		if (method in ("accuracy", "acc")):
			return (accuracy_score(self.y, self.deploySQL(pos_label, cutoff), self.test_relation, self.cursor))
		elif (method == "auc"):
			return auc("DECODE({}, '{}', 1, 0)".format(self.y, pos_label), self.deploySQL(allSQL = True)[0].format(pos_label), self.test_relation, self.cursor)
		elif (method == "prc_auc"):
			return prc_auc("DECODE({}, '{}', 1, 0)".format(self.y, pos_label), self.deploySQL(allSQL = True)[0].format(pos_label), self.test_relation, self.cursor)
		elif (method in ("best_cutoff", "best_threshold")):
			return (roc_curve("DECODE({}, '{}', 1, 0)".format(self.y, pos_label), self.deploySQL(allSQL = True)[0].format(pos_label), self.test_relation, self.cursor, best_threshold = True))
		elif (method in ("recall", "tpr")):
			return (recall_score(self.y, self.deploySQL(pos_label, cutoff), self.test_relation, self.cursor))
		elif (method in ("precision", "ppv")):
			return (precision_score(self.y, self.deploySQL(pos_label, cutoff), self.test_relation, self.cursor))
		elif (method in ("specificity", "tnr")):
			return (specificity_score(self.y, self.deploySQL(pos_label, cutoff), self.test_relation, self.cursor))
		elif (method in ("negative_predictive_value", "npv")):
			return (precision_score(self.y, self.deploySQL(pos_label, cutoff), self.test_relation, self.cursor))
		elif (method in ("log_loss", "logloss")):
			return (log_loss("DECODE({}, '{}', 1, 0)".format(self.y, pos_label), self.deploySQL(allSQL = True)[0].format(pos_label), self.test_relation, self.cursor))
		elif (method == "f1"):
			return (f1_score(self.y, self.deploySQL(pos_label, cutoff), self.test_relation, self.cursor))
		elif (method == "mcc"):
			return (matthews_corrcoef(self.y, self.deploySQL(pos_label, cutoff), self.test_relation, self.cursor))
		elif (method in ("bm", "informedness")):
			return (informedness(self.y, self.deploySQL(pos_label, cutoff), self.test_relation, self.cursor))
		elif (method in ("mk", "markedness")):
			return (markedness(self.y, self.deploySQL(pos_label, cutoff), self.test_relation, self.cursor))
		elif (method in ("csi", "critical_success_index")):
			return (critical_success_index(self.y, self.deploySQL(pos_label, cutoff), self.test_relation, self.cursor))
		else:
			raise ValueError("The parameter 'method' must be in accuracy|auc|prc_auc|best_cutoff|recall|precision|log_loss|negative_predictive_value|specificity|mcc|informedness|markedness|critical_success_index")