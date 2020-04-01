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
import random, os, math
#
def category_from_type(ctype: str = ""):
	check_types([("ctype", ctype, [str], False)])
	ctype = ctype.lower()
	if (ctype != ""):
		if (ctype[0:4] == "date") or (ctype[0:4] == "time") or (ctype == "smalldatetime") or (ctype[0:8] == "interval"):
			return "date"
		elif ((ctype[0:3] == "int") or (ctype[0:4] == "bool")  or (ctype in ("tinyint", "smallint", "bigint"))):
			return "int"
		elif ((ctype[0:3] == "num") or (ctype[0:5] == "float") or (ctype[0:7] == "decimal") or (ctype == "money") or (ctype[0:6] == "double") or (ctype[0:4] == "real")):
			return "float"
		elif (("binary" in ctype) or ("byte" in ctype) or (ctype == "raw")):
			return "binary"
		elif (ctype[0:3] == "geo"):
			return "spatial"
		elif ("uuid" in ctype):
			return "uuid"
		else:
			return "text"
	else:
		return "undefined"
#
def check_types(types_list: list):
	for elem in types_list:
		if (elem[3]):
			if (elem[1] not in elem[2]):
				raise TypeError("Parameter '{}' must be in [{}], found '{}'".format(elem[0], "|".join(elem[2]), elem[1]))
		else:
			if (type(elem[1]) not in elem[2]):
				if (len(elem[2]) == 1):
					raise TypeError("Parameter '{}' must be of type {}, found type {}".format(elem[0], elem[2][0], type(elem[1])))
				else:
					raise TypeError("Parameter '{}' must be one of the following types {}, found type {}".format(elem[0], elem[2], type(elem[1])))
#
def column_check_ambiguous(column: str, columns: list):
	column = column.replace('"', '').lower()
	for col in columns:
		col = col.replace('"', '').lower()
		if (column == col):
			return True
	return False
#
def columns_check(columns: list, vdf, columns_nb = None):
	vdf_columns = vdf.get_columns()
	if (columns_nb != None and len(columns) not in columns_nb):
		raise Exception("The number of Virtual Columns expected is {}, found {}".format("|".join([str(elem) for elem in columns_nb]), len(columns)))
	for column in columns:
		if not(column_check_ambiguous(column, vdf_columns)):
			raise ValueError("The Virtual Column '{}' doesn't exist".format(column.lower().replace('"', '')))
#
def convert_special_type(category: str, convert_date: bool = True, column: str = '{}'):
	if (category == "binary"):
		return 'TO_HEX({})'.format(column)
	elif (category == "spatial"):
		return 'ST_AsText({})'.format(column)
	elif (category == "date") and (convert_date):
		return '{}::varchar'.format(column)
	else:
		return column
#
def drop_model(name: str, 
			   cursor, 
			   print_info: bool = True):
	check_types([("name", name, [str], False)])
	cursor.execute("SELECT 1;")
	try:
		query = "DROP MODEL {};".format(name)
		cursor.execute(query)
		if (print_info):
			print("The model {} was successfully dropped.".format(name))
	except:
		print("\u26A0 Warning: The model {} doesn't exist !".format(name))
# 
def drop_table(name: str, 
			   cursor, 
			   print_info: bool = True):
	check_types([("name", name, [str], False)])
	cursor.execute("SELECT 1;")
	try:
		query="DROP TABLE {};".format(name)
		cursor.execute(query)
		if (print_info):
			print("The table {} was successfully dropped.".format(name))
	except:
		print("\u26A0 Warning: The table {} doesn't exist !".format(name))
# 
def drop_text_index(name: str, 
			   		cursor, 
			   		print_info: bool = True):
	check_types([("name", name, [str], False)])
	cursor.execute("SELECT 1;")
	try:
		query="DROP TEXT INDEX {};".format(name)
		cursor.execute(query)
		if (print_info):
			print("The index {} was successfully dropped.".format(name))
	except:
		print("\u26A0 Warning: The table {} doesn't exist !".format(name))
# 
def drop_view(name: str,
			  cursor,
			  print_info: bool = True):
	check_types([("name", name, [str], False)])
	cursor.execute("SELECT 1;")
	try:
		query="DROP VIEW {};".format(name)
		cursor.execute(query)
		if (print_info):
			print("The view {} was successfully dropped.".format(name))
	except:
		print("\u26A0 Warning: The view {} doesn't exist !".format(name))
# 
def vHelp():
	import vertica_ml_python
	try:
		from IPython.core.display import HTML, display, Markdown
	except:
		pass
	path  = os.path.dirname(vertica_ml_python.__file__)
	img1  = "<center><img src='https://raw.githubusercontent.com/vertica/Vertica-ML-Python/master/img/logo.png' width=\"180px\"></center>"
	img2  = "############################################################################################################\n"
	img2 += "#  __ __   ___ ____  ______ ____   __  ____      ___ ___ _          ____  __ __ ______ __ __  ___  ____    #\n"
	img2 += "# |  |  | /  _|    \|      |    | /  ]/    |    |   |   | |        |    \|  |  |      |  |  |/   \|    \   #\n"
	img2 += "# |  |  |/  [_|  D  |      ||  | /  /|  o  |    | _   _ | |        |  o  |  |  |      |  |  |     |  _  |  #\n"
	img2 += "# |  |  |    _|    /|_|  |_||  |/  / |     |    |  \_/  | |___     |   _/|  ~  |_|  |_|  _  |  O  |  |  |  #\n"
	img2 += "# |  :  |   [_|    \  |  |  |  /   \_|  _  |    |   |   |     |    |  |  |___, | |  | |  |  |     |  |  |  #\n"
	img2 += "#  \   /|     |  .  \ |  |  |  \     |  |  |    |   |   |     |    |  |  |     | |  | |  |  |     |  |  |  #\n"
	img2 += "#   \_/ |_____|__|\_| |__| |____\____|__|__|    |___|___|_____|    |__|  |____/  |__| |__|__|\___/|__|__|  #\n"
	img2 += "#                                                                                                          #\n"
	img2 += "############################################################################################################"
	message  = img1 if (isnotebook()) else img2
	message += "\n\n&#128226; Welcome to the <b>VERTICA ML PYTHON</b> help Module. You are about to use a new fantastic way to analyze your data !\n\nYou can learn quickly how to set up a connection, how to create a Virtual Dataframe and much more.\n\nWhat do you want to know?\n - <b>[Enter  0]</b> Do you want to know why you should use this library ?\n - <b>[Enter  1]</b> Do you want to know how to connect to your Vertica Database using Python and to Create a Virtual Dataframe ?\n - <b>[Enter  2]</b> Do you want to know if your Vertica Version is compatible with the API ?\n - <b>[Enter  3]</b> You don't have data to play with and you want to load an available dataset ?\n - <b>[Enter  4]</b> Do you want to know other modules which can make your Data Science experience more complete ?\n - <b>[Enter  5]</b> Do you want to look at a quick example ?\n - <b>[Enter  6]</b> Do you want to look at the different functions available ?\n - <b>[Enter  7]</b> Do you want to get a link to the VERTICA ML PYTHON wiki ?\n - <b>[Enter  8]</b> Do you want to know how to display the Virtual Dataframe SQL code generation and the time elapsed to run the query ?\n - <b>[Enter  9]</b> Do you want to know how to load your own dataset inside Vertica ?\n - <b>[Enter 10]</b> Do you want to know how you writing direct SQL queries in Jupyter ?\n - <b>[Enter -1]</b> Exit"
	display(Markdown(message)) if (isnotebook()) else print(message)
	try:
		response = int(input())
	except:
		print("The choice is incorrect.\nPlease enter a number between 0 and 10.")
		try:
			response = int(input())
		except:
			print("The choice is still incorrect.\nRerun the help function when you need help.")
			return
	if (response == 0):
		message = "# Vertica-ML-Python\nNowadays, The 'Big Data' (Tb of data) is one of the main topics in the Data Science World. Data Scientists are now very important for any organisation. Becoming Data-Driven is mandatory to survive. Vertica is the first real analytic columnar Database and is still the fastest in the market. However, SQL is not enough flexible to be very popular for Data Scientists. Python flexibility is priceless and provides to any user a very nice experience. The level of abstraction is so high that it is enough to think about a function to notice that it already exists. Many Data Science APIs were created during the last 15 years and were directly adopted by the Data Science community (examples: pandas and scikit-learn). However, Python is only working in-memory for a single node process. Even if some famous highly distributed programming languages exist to face this challenge, they are still in-memory and most of the time they can not process on all the data. Besides, moving the data can become very expensive. Data Scientists must also find a way to deploy their data preparation and their models. We are far away from easiness and the entire process can become time expensive. \nThe idea behind VERTICA ML PYTHON is simple: Combining the Scalability of VERTICA with the Flexibility of Python to give to the community what they need *Bringing the logic to the data and not the opposite*. This version 1.0 is the work of 3 years of new ideas and improvement.\nMain Advantages:\n - easy Data Exploration.\n - easy Data Preparation.\n - easy Data Modeling.\n - easy Model Evaluation.\n - easy Model Deployment.\n - most of what pandas.Dataframe can do, vertica_ml_python.vDataframe can do (and even much more)\n - easy ML model creation and evaluation.\n - many scikit functions and algorithms are available (and scalable!).\n\n&#9888; Please read the Vertica ML Python Documentation. If you do not have time just read below.\n\n&#9888; The previous API is really nothing compare to the new version and many methods and functions totally changed. Consider this API as a totally new one.\nIf you have any feedback about the library, please contact me: <a href=\"mailto:badr.ouali@microfocus.com\">badr.ouali@microfocus.com</a>"
	elif (response == 1):
		message = "## Connection to the Database\nThis step is useless if <b>vertica-python</b> or <b>pyodbc</b> is already installed and you have a DSN in your machine. With this configuration, you do not need to manually create a cursor. It is possible to create a vDataframe using directly the DSN (<b>dsn</b> parameter of the vDataframe).\n### ODBC\nTo connect to the database, the user can use an ODBC connection to the Vertica database. <b>vertica-python</b> and <b>pyodbc</b> provide a cursor that will point to the database. It will be used by the <b>vertica-ml-python</b> to create all the different objects.\n```python\n#\n# vertica_python\n#\nimport vertica_python\n# Connection using all the DSN information\nconn_info = {'host': \"10.211.55.14\", 'port': 5433, 'user': \"dbadmin\", 'password': \"XxX\", 'database': \"testdb\"}\ncur = vertica_python.connect(** conn_info).cursor()\n# Connection using directly the DSN\nfrom vertica_ml_python.utilities import to_vertica_python_format # This function will parse the odbc.ini file\ndsn = \"VerticaDSN\"\ncur = vertica_python.connect(** to_vertica_python_format(dsn)).cursor()\n#\n# pyodbc\n#\nimport pyodbc\n# Connection using all the DSN information\ndriver = \"/Library/Vertica/ODBC/lib/libverticaodbc.dylib\"\nserver = \"10.211.55.14\"\ndatabase = \"testdb\"\nport = \"5433\"\nuid = \"dbadmin\"\npwd = \"XxX\"\ndsn = (\"DRIVER={}; SERVER={}; DATABASE={}; PORT={}; UID={}; PWD={};\").format(driver, server, database, port, uid, pwd)\ncur = pyodbc.connect(dsn).cursor()\n# Connection using directly the DSN\ndsn = (\"DSN=VerticaDSN\")\ncur = pyodbc.connect(dsn).cursor()\n```\n### JDBC\nThe user can also use a JDBC connection to the Vertica Database. \n```python\nimport jaydebeapi\n# Vertica Server Details\ndatabase = \"testdb\"\nhostname = \"10.211.55.14\"\nport = \"5433\"\nuid = \"dbadmin\"\npwd = \"XxX\"\n# Vertica JDBC class name\njdbc_driver_name = \"com.vertica.jdbc.Driver\"\n# Vertica JDBC driver path\njdbc_driver_loc = \"/Library/Vertica/JDBC/vertica-jdbc-9.3.1-0.jar\"\n# JDBC connection string\nconnection_string = 'jdbc:vertica://' + hostname + ':' + port + '/' + database\nurl = '{}:user={};password={}'.format(connection_string, uid, pwd)\nconn = jaydebeapi.connect(jdbc_driver_name, connection_string, {'user': uid, 'password': pwd}, jars = jdbc_driver_loc)\ncur = conn.cursor()\n```\nHappy Playing ! &#128540;\n"
	elif (response == 2):
		message = "## Vertica Version\n - If your Vertica version is greater or equal to 9.1, everything is well adapted.\n - If your Vertica version is greater or equal to 8.0, some algorithms may not work.\n - If your Vertica version is greater or equal to 7.0, only some algorithms will be available.\n - For other Vertica version, the Virtual Dataframe may work but no ML algorithms will be available."
	elif (response == 3):
		message = "In VERTICA ML PYTHON many datasets (titanic, iris, smart_meters, amazon, winequality) are already available to be ingested in your Vertica Database.\n\nTo ingest a dataset you can use the associated load function.\n\n<b>Example:</b>\n\n```python\nfrom vertica_python.learn.datasets import load_titanic\nvdf = load_titanic(db_cursor)\n```"
	elif (response == 4):
		message = "Some module will help VERTICA ML PYTHON to get more rendering capabilities:\n - <b>matplotlib</b> will help you to get rendering capabilities\n - <b>numpy</b> to enjoy 3D plot\n - <b>sqlparse</b> to indent correctly the SQL of the sql_on_off method\n - <b>anytree</b> to be able to plot trees"
	elif (response == 5):
		message = "## Quick Start\nInstall the library using the <b>pip</b> command:\n```\nroot@ubuntu:~$ pip3 install vertica_ml_python\n```\nInstall <b>vertica_python</b> or <b>pyodbc</b> to build a DB cursor:\n```shell\nroot@ubuntu:~$ pip3 install vertica_python\n```\nCreate a vertica cursor\n```python\nfrom vertica_ml_python.utilities import vertica_cursor\ncur = vertica_cursor(\"VerticaDSN\")\n```\nCreate the Virtual Dataframe of your relation:\n```python\nfrom vertica_ml_python import vDataframe\nvdf = vDataframe(\"my_relation\", cursor = cur)\n```\nIf you don't have data to play, you can easily load well known datasets\n```python\nfrom vertica_ml_python.learn.datasets import load_titanic\nvdf = load_titanic(cursor = cur)\n```\nYou can now play with the data...\n```python\nvdf.describe()\n# Output\n               min       25%        50%        75%   \nage           0.33      21.0       28.0       39.0   \nbody           1.0     79.25      160.5      257.5   \nfare           0.0    7.8958    14.4542    31.3875   \nparch          0.0       0.0        0.0        0.0   \npclass         1.0       1.0        3.0        3.0   \nsibsp          0.0       0.0        0.0        1.0   \nsurvived       0.0       0.0        0.0        1.0   \n                   max    unique  \nage               80.0        96  \nbody             328.0       118  \nfare          512.3292       277  \nparch              9.0         8  \npclass             3.0         3  \nsibsp              8.0         7  \nsurvived           1.0         2 \n```\nYou can also print the SQL code generation using the <b>sql_on_off</b> method.\n```python\nvdf.sql_on_off()\nvdf.describe()\n# Output\n## Compute the descriptive statistics of all the numerical columns ##\nSELECT SUMMARIZE_NUMCOL(\"age\",\"body\",\"survived\",\"pclass\",\"parch\",\"fare\",\"sibsp\") OVER ()\nFROM\n  (SELECT \"age\" AS \"age\",\n          \"body\" AS \"body\",\n          \"survived\" AS \"survived\",\n          \"ticket\" AS \"ticket\",\n          \"home.dest\" AS \"home.dest\",\n          \"cabin\" AS \"cabin\",\n          \"sex\" AS \"sex\",\n          \"pclass\" AS \"pclass\",\n          \"embarked\" AS \"embarked\",\n          \"parch\" AS \"parch\",\n          \"fare\" AS \"fare\",\n          \"name\" AS \"name\",\n          \"boat\" AS \"boat\",\n          \"sibsp\" AS \"sibsp\"\n   FROM public.titanic) final_table\n```\nWith Vertica ML Python, it is now possible to solve a ML problem with four lines of code (two if we don't consider the libraries loading).\n```python\nfrom vertica_ml_python.learn.model_selection import cross_validate\nfrom vertica_ml_python.learn.ensemble import RandomForestClassifier\n# Data Preparation\nvdf[\"sex\"].label_encode()[\"boat\"].fillna(method = \"0ifnull\")[\"name\"].str_extract(' ([A-Za-z]+)\.').eval(\"family_size\", expr = \"parch + sibsp + 1\").drop(columns = [\"cabin\", \"body\", \"ticket\", \"home.dest\"])[\"fare\"].fill_outliers().fillna().to_db(\"titanic_clean\")\n# Model Evaluation\ncross_validate(RandomForestClassifier(\"rf_titanic\", cur, max_leaf_nodes = 100, n_estimators = 30), \"titanic_clean\", [\"age\", \"family_size\", \"sex\", \"pclass\", \"fare\", \"boat\"], \"survived\", cutoff = 0.35)\n# Output\n                           auc               prc_auc   \n1-fold      0.9877114427860691    0.9530465915039339   \n2-fold      0.9965555014605642    0.7676485351425721   \n3-fold      0.9927239216549301    0.6419135521132449   \navg             0.992330288634        0.787536226253   \nstd           0.00362128464093         0.12779562393   \n                     accuracy              log_loss   \n1-fold      0.971291866028708    0.0502052541223871   \n2-fold      0.983253588516746    0.0298167751798457   \n3-fold      0.964824120603015    0.0392745694400433   \navg            0.973123191716       0.0397655329141   \nstd           0.0076344236729      0.00833079837099   \n                     precision                recall   \n1-fold                    0.96                  0.96   \n2-fold      0.9556962025316456                   1.0   \n3-fold      0.9647887323943662    0.9383561643835616   \navg             0.960161644975        0.966118721461   \nstd           0.00371376912311        0.025535200301   \n                      f1-score                   mcc   \n1-fold      0.9687259282082884    0.9376119402985075   \n2-fold      0.9867172675521821    0.9646971010878469   \n3-fold      0.9588020287309097    0.9240569687684576   \navg              0.97141507483        0.942122003385   \nstd            0.0115538960753       0.0168949813163   \n                  informedness            markedness   \n1-fold      0.9376119402985075    0.9376119402985075   \n2-fold      0.9737827715355807    0.9556962025316456   \n3-fold      0.9185148945422918    0.9296324823943662   \navg             0.943303202125        0.940980208408   \nstd            0.0229190954261       0.0109037699717   \n                           csi  \n1-fold      0.9230769230769231  \n2-fold      0.9556962025316456  \n3-fold      0.9072847682119205  \navg             0.928685964607  \nstd            0.0201579224026\n```\nHappy Playing ! &#128540;"
	elif (response == 6):
		message = "Please go to <a href='https://github.com/vertica/Vertica-ML-Python/blob/master/FEATURES.md'>https://github.com/vertica/Vertica-ML-Python/blob/master/FEATURES.md</a>"
	elif (response == 7):
		message = "Please go to <a href='https://github.com/vertica/Vertica-ML-Python/wiki'>https://github.com/vertica/Vertica-ML-Python/wiki</a>"
	elif (response == 8):
		message = "You can Display the SQL Code generation of the Virtual Dataframe using the <b>sql_on_off</b> method. You can also Display the query elapsed time using the <b>time_on_off</b> method.\nIt is also possible to print the current Virtual Dataframe relation using the <b>current_relation</b> method.\n"
	elif (response == 9):
		message = "VERTICA ML PYTHON allows you many ways to ingest data file. It is using Vertica Flex Tables to identify the columns types and store the data inside Vertica. These functions will also return the associated Virtual Dataframe.\n\nLet's load the data from the 'data.csv' file.\n\n\n```python\nfrom vertica_ml_python import read_csv\nvdf = read_csv('data.csv', db_cursor)\n```\n\nThe same for json. Let's now consider the file 'data.json'.\n\n\n```python\nfrom vertica_ml_python import read_json\nvdf = read_json('data.json', db_cursor)\n```\n\n"
	elif (response == 10):
		message = "SQL Alchemy and SQL Magic offer you a nice way to interact with Vertica. To install the modules, run the following commands in your Terminal: \n```\npip install pyodbc\npip install sqlalchemy-vertica[pyodbc,vertica-python]\npip install ipython-sql\n```\n\nWhen these modules are installed, you have a nice way to interact with Jupyter.\n```python\n# Creating a Connection\n%load_ext sql\n%sql vertica+pyodbc://VerticaDSN\n# You can run your sql code using the %sql or %%sql command\n%sql SELECT * FROM my_relation;\n```"
	elif (response == -1):
		message = "Thank you for using the VERTICA ML PYTHON help."
	elif (response == 666):
		message = "Thank you so much for using this library. My only purpose is to solve real Big Data problems in the context of Data Science. I worked years to be able to create this API and give you a real way to analyse your data.\n\nYour devoted Data Scientist: <i>Badr Ouali</i>"
	else:
		message = "The choice is incorrect.\nPlease enter a number between 0 and 10."
	display(Markdown(message)) if (isnotebook()) else print(message)
# 
def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False # Terminal running IPython
        else:
            return False # Other type (?)
    except NameError:
        return False # Probably standard Python interpreter
#
def load_model(name: str, cursor, test_relation: str = ""):
	check_types([("name", name, [str], False), ("test_relation", test_relation, [str], False)])
	try:
		cursor.execute("SELECT GET_MODEL_ATTRIBUTE (USING PARAMETERS model_name = '" + name + "', attr_name = 'call_string')")
		info = cursor.fetchone()[0].replace('\n', ' ')
	except:
		try:
			cursor.execute("SELECT GET_MODEL_SUMMARY (USING PARAMETERS model_name = '" + name + "')")
			info = cursor.fetchone()[0].replace('\n', ' ')
			info = "kmeans(" + info.split("kmeans(")[1]
		except:
			from vertica_ml_python.learn.preprocessing import Normalizer
			model = Normalizer(name, cursor)
			model.param = to_tablesample(query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'details')".format(name.replace("'", "''")), cursor = cursor)
			model.param.table_info = False
			model.X = ['"' + item + '"' for item in model.param.values["column_name"]]
			if ("avg" in model.param.values):
				model.method = "zscore" 
			elif ("max" in model.param.values):
				model.method = "minmax" 
			else:
				model.method = "robust_zscore"
			return model
	try:
		info = info.split("SELECT ")[1].split("(")
	except:
		info = info.split("(")
	model_type = info[0].lower()
	info = info[1].split(")")[0].replace(" ", '').split("USINGPARAMETERS")
	if (model_type == "svm_classifier"):
		parameters = "".join(info[1].split("class_weights=")[1].split("'"))
		parameters = parameters[3:len(parameters)].split(",")
		del parameters[0]
		parameters += ["class_weights=" + info[1].split("class_weights=")[1].split("'")[1]]
	elif (model_type != "svd"):
		parameters = info[1].split(",")
	if (model_type != "svd"):
		parameters = [item.split("=") for item in parameters]
		parameters_dict = {}
		for item in parameters:
			parameters_dict[item[0]] = item[1]
	info = info[0]
	if (model_type == "rf_regressor"):
		from vertica_ml_python.learn.ensemble import RandomForestRegressor
		model = RandomForestRegressor(name, cursor, int(parameters_dict['ntree']), int(parameters_dict['mtry']), int(parameters_dict['max_breadth']), float(parameters_dict['sampling_size']), int(parameters_dict['max_depth']), int(parameters_dict['min_leaf_size']), float(parameters_dict['min_info_gain']), int(parameters_dict['nbins']))
	elif (model_type == "rf_classifier"):
		from vertica_ml_python.learn.ensemble import RandomForestClassifier
		model = RandomForestClassifier(name, cursor, int(parameters_dict['ntree']), int(parameters_dict['mtry']), int(parameters_dict['max_breadth']), float(parameters_dict['sampling_size']), int(parameters_dict['max_depth']), int(parameters_dict['min_leaf_size']), float(parameters_dict['min_info_gain']), int(parameters_dict['nbins']))
	elif (model_type == "logistic_reg"):
		from vertica_ml_python.learn.linear_model import LogisticRegression
		model = LogisticRegression(name, cursor, parameters_dict['regularization'], float(parameters_dict['epsilon']), float(parameters_dict['lambda']), int(parameters_dict['max_iterations']), parameters_dict['optimizer'], float(parameters_dict['alpha']))
	elif (model_type == "linear_reg"):
		from vertica_ml_python.learn.linear_model import ElasticNet
		model = ElasticNet(name, cursor, parameters_dict['regularization'], float(parameters_dict['epsilon']), float(parameters_dict['lambda']), int(parameters_dict['max_iterations']), parameters_dict['optimizer'], float(parameters_dict['alpha']))
	elif (model_type == "naive_bayes"):
		from vertica_ml_python.learn.naive_bayes import MultinomialNB
		model = MultinomialNB(name, cursor, float(parameters_dict['alpha']))
	elif (model_type == "svm_regressor"):
		from vertica_ml_python.learn.svm import LinearSVR
		model = LinearSVR(name, cursor, float(parameters_dict['epsilon']), float(parameters_dict['C']), True, float(parameters_dict['intercept_scaling']), parameters_dict['intercept_mode'], float(parameters_dict['error_tolerance']), int(parameters_dict['max_iterations']))
	elif (model_type == "svm_classifier"):
		from vertica_ml_python.learn.svm import LinearSVC
		model = LinearSVC(name, cursor, float(parameters_dict['epsilon']), float(parameters_dict['C']), True, float(parameters_dict['intercept_scaling']), parameters_dict['intercept_mode'], [float(item) for item in parameters_dict['class_weights'].split(",")], int(parameters_dict['max_iterations']))
	elif (model_type == "kmeans"):
		from vertica_ml_python.learn.cluster import KMeans
		model = KMeans(name, cursor, -1, parameters_dict['init_method'], int(parameters_dict['max_iterations']), float(parameters_dict['epsilon']))
	elif (model_type == "pca"):
		from vertica_ml_python.learn.decomposition import PCA
		model = PCA(name, cursor, 0, bool(parameters_dict['scale']))
	elif (model_type == "svd"):
		from vertica_ml_python.learn.decomposition import SVD
		model = SVD(name, cursor)
	elif (model_type == "one_hot_encoder_fit"):
		from vertica_ml_python.learn.preprocessing import OneHotEncoder
		model = OneHotEncoder(name, cursor)
	model.input_relation = info.split(",")[1].replace("'", '').replace('\\', '')
	model.test_relation = test_relation if (test_relation) else model.input_relation
	if (model_type not in ("kmeans", "pca", "svd", "one_hot_encoder_fit")):
		model.X = info.split(",")[3:len(info.split(","))]
		model.X = [item.replace("'", '').replace('\\', '') for item in model.X]
		model.y = info.split(",")[2].replace("'", '').replace('\\', '')
	elif (model_type in ("pca")):
		model.X = info.split(",")[2:len(info.split(","))]
		model.X = [item.replace("'", '').replace('\\', '') for item in model.X]
		model.components = to_tablesample(query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'principal_components')".format(name.replace("'", "''")), cursor = cursor)
		model.components.table_info = False
		model.explained_variance = to_tablesample(query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'singular_values')".format(name.replace("'", "''")), cursor = cursor)
		model.explained_variance.table_info = False
		model.mean = to_tablesample(query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'columns')".format(name.replace("'", "''")), cursor = cursor)
		model.mean.table_info = False
	elif (model_type in ("svd")):
		model.X = info.split(",")[2:len(info.split(","))]
		model.X = [item.replace("'", '').replace('\\', '') for item in model.X]
		model.singular_values = to_tablesample(query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'right_singular_vectors')".format(name.replace("'", "''")), cursor = cursor)
		model.singular_values.table_info = False
		model.explained_variance = to_tablesample(query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'singular_values')".format(name.replace("'", "''")), cursor = cursor)
		model.explained_variance.table_info = False
	elif (model_type in ("one_hot_encoder_fit")):
		model.X = info.split(",")[2:len(info.split(","))]
		model.X = [item.replace("'", '').replace('\\', '') for item in model.X]
		model.param = to_tablesample(query = "SELECT category_name, category_level::varchar, category_level_index FROM (SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'integer_categories')) x UNION ALL SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'varchar_categories')".format(name.replace("'", "''"), name.replace("'", "''")), cursor = cursor)
		model.param.table_info = False
	else:
		model.X = info.split(",")[2:len(info.split(",")) - 1]
		model.X = [item.replace("'", '').replace('\\', '') for item in model.X]
		model.n_cluster = int(info.split(",")[-1])
		model.cluster_centers = to_tablesample(query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'centers')".format(name.replace("'", "''")), cursor = cursor)
		model.cluster_centers.table_info = False
		query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'metrics')".format(name.replace("'", "''"))
		cursor.execute(query)
		result = cursor.fetchone()[0]
		values = {"index": ["Between-Cluster Sum of Squares", "Total Sum of Squares", "Total Within-Cluster Sum of Squares", "Between-Cluster SS / Total SS", "converged"]}
		values["value"] = [float(result.split("Between-Cluster Sum of Squares: ")[1].split("\n")[0]), float(result.split("Total Sum of Squares: ")[1].split("\n")[0]), float(result.split("Total Within-Cluster Sum of Squares: ")[1].split("\n")[0]), float(result.split("Between-Cluster Sum of Squares: ")[1].split("\n")[0]) / float(result.split("Total Sum of Squares: ")[1].split("\n")[0]), result.split("Converged: ")[1].split("\n")[0] == "True"] 
		model.metrics = tablesample(values, table_info = False)
	if (model.type == "classifier"):
		cursor.execute("SELECT DISTINCT {} FROM {} WHERE {} IS NOT NULL ORDER BY 1".format(model.y, model.input_relation, model.y))
		classes = cursor.fetchall()
		model.classes = [item[0] for item in classes]
	return (model)
# 
def pandas_to_vertica(df, cur, name: str, schema: str = "public", insert: bool = False):
	check_types([("name", name, [str], False), ("schema", schema, [str], False), ("insert", insert, [bool], False)])
	path = "vertica_ml_python_{}.csv".format(elem for elem in name if elem.isalpha())
	try:
		df.to_csv(path, index = False)
		read_csv(path, cur, table_name = name, schema = schema, insert = insert)
		os.remove(path)
	except:
		os.remove(path)
		raise
# 
def print_table(data_columns, is_finished = True, offset = 0, repeat_first_column = False, first_element = ""):
	data_columns_rep = [] + data_columns
	if (repeat_first_column):
		del data_columns_rep[0]
		columns_ljust_val = min(len(max([str(item) for item in data_columns[0]], key = len)) + 4, 40)
	else:
		columns_ljust_val = len(str(len(data_columns[0]))) + 2
	try:
		import shutil
		screen_columns = shutil.get_terminal_size().columns
	except:
		screen_rows, screen_columns = os.popen('stty size', 'r').read().split()
	formatted_text = ""
	rjust_val = []
	for idx in range(0,len(data_columns_rep)):
		rjust_val += [min(len(max([str(item) for item in data_columns_rep[idx]], key = len)) + 2, 40)]
	total_column_len = len(data_columns_rep[0])
	while (rjust_val != []):
		columns_to_print = [data_columns_rep[0]]
		columns_rjust_val = [rjust_val[0]]
		max_screen_size = int(screen_columns) - 14 - int(rjust_val[0])
		del data_columns_rep[0]
		del rjust_val[0]
		while ((max_screen_size > 0) and (rjust_val != [])):
			columns_to_print += [data_columns_rep[0]]
			columns_rjust_val += [rjust_val[0]]
			max_screen_size = max_screen_size-7-int(rjust_val[0])
			del data_columns_rep[0]
			del rjust_val[0]
		if (repeat_first_column):
			columns_to_print = [data_columns[0]] + columns_to_print
		else:
			columns_to_print=[[i - 1 + offset for i in range(0,total_column_len)]] + columns_to_print
		columns_to_print[0][0] = first_element
		columns_rjust_val = [columns_ljust_val]+columns_rjust_val
		column_count = len(columns_to_print)
		for i in range(0,total_column_len):
			for k in range(0,column_count):
				val = columns_to_print[k][i]
				if len(str(val)) > 40:
					val = str(val)[0:37] + "..."
				if (k == 0):
					formatted_text += str(val).ljust(columns_rjust_val[k])
				else:
					formatted_text += str(val).rjust(columns_rjust_val[k])+"  "
			if ((rjust_val != [])):
				formatted_text += " \\\\"
			formatted_text += "\n"	
		if (not(is_finished) and (i == total_column_len-1)):
			for k in range(0,column_count):
				if (k==0):
					formatted_text += "...".ljust(columns_rjust_val[k])
				else:
					formatted_text += "...".rjust(columns_rjust_val[k])+"  "
			if (rjust_val != []):
				formatted_text += " \\\\"
			formatted_text += "\n"
	try:	
		if (isnotebook()):
			from IPython.core.display import HTML, display
			if not(repeat_first_column):
				data_columns=[[""] + list(range(0 + offset, len(data_columns[0]) - 1 + offset))] + data_columns
			m = len(data_columns)
			n = len(data_columns[0])
			html_table = "<table style=\"border-collapse: collapse; border: 2px solid white\">"
			for i in range(n):
				html_table += "<tr style=\"{border: 1px solid white;}\">"
				for j in range(m):
					if (j == 0):
						html_table += "<td style=\"border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white\"><b>" + str(data_columns[j][i]) + "</b></td>"
					elif (i == 0):
						html_table += "<td style=\"font-size:1.02em;background-color:#214579;color:white\"><b>" + str(data_columns[j][i]) + "</b></td>"
					else:
						html_table += "<td style=\"border: 1px solid white;\">" + str(data_columns[j][i]) + "</td>"
				html_table += "</tr>"
			if not(is_finished):
				html_table += "<tr>"
				for j in range(m):
					if (j == 0):
						html_table += "<td style=\"border-top: 1px solid white;background-color:#214579;color:white\"></td>"
					else:
						html_table += "<td style=\"border: 1px solid white;\">...</td>"
				html_table += "</tr>"
			html_table += "</table>"
			display(HTML(html_table))
			return "<object>  "
		else:
			return formatted_text
	except:
		return formatted_text
#
def pjson(path: str, cursor):
	flex_name = "VERTICA_ML_PYTHON_" + str(random.randint(0, 10000000)) + "_FLEX"
	cursor.execute("CREATE FLEX LOCAL TEMP TABLE {}(x int) ON COMMIT PRESERVE ROWS;".format(flex_name))
	cursor.execute("COPY {} FROM LOCAL '{}' PARSER FJSONPARSER();".format(flex_name, path.replace("'", "''")))
	cursor.execute("SELECT compute_flextable_keys('{}');".format(flex_name))
	cursor.execute("SELECT key_name, data_type_guess FROM {}_keys".format(flex_name))
	result = cursor.fetchall()
	dtype = {}
	for column_dtype in result:
		dtype[column_dtype[0]] = column_dtype[1]
	cursor.execute("DROP TABLE " + flex_name)
	return (dtype)
#
def pcsv(path: str, 
		 cursor, 
		 delimiter: str = ',', 
		 header: bool = True,
		 header_names: list = [],
		 null: str = '', 
		 enclosed_by: str = '"',
		 escape: str = '\\'):
	flex_name = "VERTICA_ML_PYTHON_" + str(random.randint(0, 10000000)) + "_FLEX"
	cursor.execute("CREATE FLEX LOCAL TEMP TABLE {}(x int) ON COMMIT PRESERVE ROWS;".format(flex_name))
	header_names = '' if not(header_names) else "header_names = '{}',".format(delimiter.join(header_names))
	try:
		with open(path, "r") as fs:
			cursor.copy("COPY {} FROM STDIN PARSER FCSVPARSER(type = 'traditional', delimiter = '{}', header = {}, {} enclosed_by = '{}', escape = '{}') NULL '{}';".format(flex_name, delimiter, header, header_names, enclosed_by, escape, null), fs)
	except:
		cursor.execute("COPY {} FROM LOCAL '{}' PARSER FCSVPARSER(type = 'traditional', delimiter = '{}', header = {}, {} enclosed_by = '{}', escape = '{}') NULL '{}';".format(flex_name, path, delimiter, header, header_names, enclosed_by, escape, null))
	cursor.execute("SELECT compute_flextable_keys('{}');".format(flex_name))
	cursor.execute("SELECT key_name, data_type_guess FROM {}_keys".format(flex_name))
	result = cursor.fetchall()
	dtype = {}
	for column_dtype in result:
		try:
			query = 'SELECT (CASE WHEN "{}"=\'{}\' THEN NULL ELSE "{}" END)::{} AS "{}" FROM {} WHERE "{}" IS NOT NULL LIMIT 1000'.format(column_dtype[0], null, column_dtype[0], column_dtype[1], column_dtype[0], flex_name, column_dtype[0])
			cursor.execute(query)
			dtype[column_dtype[0]] = column_dtype[1]
		except:
			dtype[column_dtype[0]] = "Varchar(100)"
	cursor.execute("DROP TABLE " + flex_name)
	return (dtype)
# 
def read_csv(path: str, 
			 cursor, 
			 schema: str = 'public', 
			 table_name: str = '', 
			 delimiter: str = ',', 
			 header: bool = True,
			 header_names: list = [],
			 null: str = '', 
			 enclosed_by: str = '"', 
			 escape: str = '\\', 
			 genSQL: bool = False,
			 parse_n_lines: int = -1,
			 insert: bool = False):
	check_types([("schema", schema, [str], False), ("table_name", table_name, [str], False), ("delimiter", delimiter, [str], False), ("header", header, [bool], False), ("header_names", header_names, [list], False), ("null", null, [str], False), ("enclosed_by", enclosed_by, [str], False), ("escape", escape, [str], False), ("genSQL", genSQL, [bool], False), ("parse_n_lines", parse_n_lines, [int, float], False), ("insert", insert, [bool], False)])
	path, delimiter, header_names, null, enclosed_by, escape = path.replace("'", "''"), delimiter.replace("'", "''"), [str(elem).replace("'", "''") for elem in header_names], null.replace("'", "''"), enclosed_by.replace("'", "''"), escape.replace("'", "''")
	file = path.split("/")[-1]
	file_extension = file[-3:len(file)]
	if (file_extension != 'csv'):
		raise ValueError("The file extension is incorrect !")
	table_name = table_name if (table_name) else path.split("/")[-1].split(".csv")[0]
	query = "SELECT column_name FROM columns WHERE table_name = '{}' AND table_schema = '{}'".format(table_name.replace("'", "''"), schema.replace("'", "''"))
	cursor.execute(query)
	result = cursor.fetchall()
	if ((result != []) and not(insert)):
		raise Exception("The table \"{}\".\"{}\" already exists !".format(schema, table_name))
	elif ((result == []) and (insert)):
		raise Exception("The table \"{}\".\"{}\" doesn't exist !".format(schema, table_name))
	else:
		input_relation = '"{}"."{}"'.format(schema, table_name)
		f = open(path,'r')
		file_header = f.readline().replace('\n', '').replace('"', '').split(delimiter)
		f.close()
		if ((header_names == []) and (header)):
			header_names = file_header
		elif (len(file_header) > len(header_names)):
			header_names += ["ucol{}".format(i + len(header_names)) for i in range(len(file_header) - len(header_names))]
		if ((parse_n_lines > 0) and not(insert)):
			f = open(path,'r')
			f2 = open(path[0:-4] + "VERTICA_ML_PYTHON_COPY.csv",'w')
			for i in range(parse_n_lines + int(header)):
				line = f.readline()
				f2.write(line)
			f.close()
			f2.close()
			path_test = path[0:-4] + "VERTICA_ML_PYTHON_COPY.csv"
		else:
			path_test = path
		query1 = ""
		if not(insert):
			dtype = pcsv(path_test, cursor, delimiter, header, header_names, null, enclosed_by, escape)
			if (parse_n_lines > 0):
				os.remove(path[0:-4] + "VERTICA_ML_PYTHON_COPY.csv")
			query1  = "CREATE TABLE {}({});".format(input_relation, ", ".join(['"{}" {}'.format(column, dtype[column]) for column in header_names]))
		skip   = " SKIP 1" if (header) else ""
		query2 = "COPY {}({}) FROM {} DELIMITER '{}' NULL '{}' ENCLOSED BY '{}' ESCAPE AS '{}'{};".format(input_relation, ", ".join(['"' + column + '"' for column in header_names]), "{}", delimiter, null, enclosed_by, escape, skip)
		if (genSQL):
			print(query1 + "\n" + query)
		else:
			if (query1):
				cursor.execute(query1)
			if ("vertica_python" in str(type(cursor))):
				with open(path, "r") as fs:
					cursor.copy(query2.format('STDIN'), fs)
			else:
				cursor.execute(query2.format("LOCAL '{}'".format(path)))
			if (query1):
				print("The table {} has been successfully created.".format(input_relation))
			from vertica_ml_python import vDataframe
			return vDataframe(table_name, cursor, schema = schema)
# 
def read_json(path: str, 
			  cursor, 
			  schema: str = 'public', 
			  table_name: str = '',
			  usecols: list = [],
			  new_name: dict = {},
			  insert: bool = False):
	check_types([("schema", schema, [str], False), ("table_name", table_name, [str], False), ("usecols", usecols, [list], False), ("new_name", new_name, [dict], False), ("insert", insert, [bool], False)])
	file = path.split("/")[-1]
	file_extension = file[-4:len(file)]
	if (file_extension != 'json'):
		raise ValueError("The file extension is incorrect !")
	table_name = table_name if (table_name) else path.split("/")[-1].split(".json")[0]
	query = "SELECT column_name, data_type FROM columns WHERE table_name = '{}' AND table_schema = '{}'".format(table_name.replace("'", "''"), schema.replace("'", "''"))
	cursor.execute(query)
	column_name = cursor.fetchall()
	if ((column_name != []) and not(insert)):
		raise Exception("The table \"{}\".\"{}\" already exists !".format(schema, table_name))
	elif ((column_name == []) and (insert)):
		raise Exception("The table \"{}\".\"{}\" doesn't exist !".format(schema, table_name))
	else:
		input_relation, flex_name = '"{}"."{}"'.format(schema, table_name), "VERTICA_ML_PYTHON_" + str(random.randint(0, 10000000)) + "_FLEX"
		cursor.execute("CREATE FLEX LOCAL TEMP TABLE {}(x int) ON COMMIT PRESERVE ROWS;".format(flex_name))
		if ("vertica_python" in str(type(cursor))):
			with open(path, "r") as fs:
				cursor.copy("COPY {} FROM STDIN PARSER FJSONPARSER();".format(flex_name), fs)
		else:
			cursor.execute("COPY {} FROM LOCAL '{}' PARSER FJSONPARSER();".format(flex_name, path.replace("'", "''")))
		cursor.execute("SELECT compute_flextable_keys('{}');".format(flex_name))
		cursor.execute("SELECT key_name, data_type_guess FROM {}_keys".format(flex_name))
		result = cursor.fetchall()
		dtype = {}
		for column_dtype in result:
			try:
				cursor.execute('SELECT "{}"::{} FROM {} LIMIT 1000'.format(column_dtype[0], column_dtype[1], flex_name))
				dtype[column_dtype[0]] = column_dtype[1]
			except:
				dtype[column_dtype[0]] = "Varchar(100)"
		if not(insert):
			cols = [column for column in dtype] if not(usecols) else [column for column in usecols]
			for i, column in enumerate(cols):
				cols[i] = '"{}"::{} AS "{}"'.format(column.replace('"', ''), dtype[column], new_name[column]) if (column in new_name) else '"{}"::{}'.format(column.replace('"', ''), dtype[column])
			cursor.execute("CREATE TABLE {} AS SELECT {} FROM {}".format(input_relation, ", ".join(cols), flex_name))
			print("The table {} has been successfully created.".format(input_relation))
		else:
			column_name_dtype = {}
			for elem in column_name:
				column_name_dtype[elem[0]] = elem[1]
			final_cols = {}
			for column in column_name_dtype:
				final_cols[column] = None
			for column in column_name_dtype:
				if column in dtype:
					final_cols[column] = column
				else:
					for col in new_name:
						if (new_name[col] == column):
							final_cols[column] = col
			final_transformation = []
			for column in final_cols:
				final_transformation += ['NULL AS "{}"'.format(column)] if (final_cols[column] == None) else ['"{}"::{} AS "{}"'.format(final_cols[column], column_name_dtype[column], column)]
			cursor.execute("INSERT INTO {} SELECT {} FROM {}".format(input_relation, ", ".join(final_transformation), flex_name))
		cursor.execute("DROP TABLE {}".format(flex_name))
		from vertica_ml_python import vDataframe
		return vDataframe(table_name, cursor, schema = schema)
# 
def read_vdf(path: str, cursor):
	check_types([("path", path, [str], False)])
	file = open(path, "r")
	save =  "from vertica_ml_python import vDataframe\nfrom vertica_ml_python.vcolumn import vColumn\n" + "".join(file.readlines())
	file.close()
	vdf = {}
	exec(save, globals(), vdf)
	vdf = vdf["vdf_save"]
	vdf.VERTICA_ML_PYTHON_VARIABLES["cursor"] = cursor
	return (vdf)
#
def schema_relation(relation: str):
	quote_nb = relation.count('"')
	if (quote_nb not in (0, 2, 4)):
		raise ValueError("The format of the input relation is incorrect.")
	if (quote_nb == 4):
		schema_input_relation = relation.split('"')[1], relation.split('"')[3]
	elif (quote_nb == 4):
		schema_input_relation = relation.split('"')[1], relation.split('"')[2][1:] if (relation.split('"')[0] == '') else relation.split('"')[0][0:-1], relation.split('"')[1]
	else:
		schema_input_relation = relation.split(".")
	if (len(schema_input_relation) == 1):
		schema, relation = "public", relation 
	else: 
		schema, relation = schema_input_relation[0], schema_input_relation[1]
	return (schema, relation)
#
def str_column(column: str):
	return ('"{}"'.format(column.replace('"', '')))
#
class tablesample:
	# Initialization
	def  __init__(self, 
				  values: dict = {}, 
				  dtype: dict = {}, 
				  name: str = "Sample", 
				  count: int = 0, 
				  offset: int = 0, 
				  table_info: bool = True):
		check_types([("values", values, [dict], False), ("dtype", dtype, [dict], False), ("name", name, [str], False), ("count", count, [int], False), ("offset", offset, [int], False), ("table_info", table_info, [bool], False)])
		self.values = values
		self.dtype = dtype
		self.count = count
		self.offset = offset
		self.table_info = table_info
		self.name = name
		for column in values:
			if column not in dtype:
				self.dtype[column] = "undefined"
	# Representation
	def __repr__(self):
		if (len(self.values) == 0):
			return ""
		data_columns = [[column] + self.values[column] for column in self.values]
		formatted_text = print_table(data_columns, is_finished = (self.count <= len(data_columns[0]) + self.offset), offset = self.offset, repeat_first_column = ("index" in self.values))
		if (self.table_info):
			if (len(self.values) == 1):
				column = list(self.values.keys())[0]
				formatted_text += "Name: {}, Number of rows: {}, dtype: {}".format(column, max(len(data_columns[0]) - 1, self.count), self.dtype[column]) 
			else:
				formatted_text += "Name: {}, Number of rows: {}, Number of columns: {}".format(self.name, max(len(data_columns[0]) - 1, self.count), len(data_columns)) 
		else:
			formatted_text = formatted_text[0:-2]
		return formatted_text
	# 
	def transpose(self):
		index = [column for column in self.values]
		first_item = list(self.values.keys())[0]
		columns =[[] for i in range(len(self.values[first_item]))]
		for column in self.values:
			for idx, item in enumerate(self.values[column]):
				columns[idx] += [item]
		columns = [index] + columns
		values = {}
		for item in columns:
			values[item[0]] = item[1:len(item)]
		self.values = values
		return (self)
	#
	def to_pandas(self):
		import pandas as pd
		if ("index" in self.values):
			df = pd.DataFrame(data = self.values, index = self.values["index"])
			return df.drop(columns = ['index'])
		else:
			return pd.DataFrame(data = self.values)
	def to_sql(self):
		sql = []
		n = len(self.values[list(self.values.keys())[0]])
		for i in range(n):
			row = [] 
			for column in self.values:
				val = self.values[column][i]
				if (type(val) == str):
					val = "'" + val.replace("'", "''") + "'"
				elif (val == None):
					val = "NULL"
				elif (math.isnan(val)):
					val = "NULL"
				row += ["{} AS {}".format(val, '"' + column.replace('"', '') + '"')]
			sql += ["(SELECT {})".format(", ".join(row))]
		sql = " UNION ALL ".join(sql)
		return (sql)
	def to_vdf(self, cursor = None, dsn: str = ""):
		check_types([("dsn", dsn, [str], False)])
		from vertica_ml_python import vdf_from_relation
		relation = "({}) sql_relation".format(self.to_sql())
		return (vdf_from_relation(relation, cursor = cursor, dsn = dsn)) 
#
def to_tablesample(query: str, cursor, name = "Sample"):
	check_types([("query", query, [str], False), ("name", name, [str], False)])
	cursor.execute(query)
	result = cursor.fetchall()
	columns = [column[0] for column in cursor.description]
	data_columns = [[item] for item in columns]
	data = [item for item in result]
	for row in data:
		for idx, val in enumerate(row):
			data_columns[idx] += [val]
	values = {}
	for column in data_columns:
		values[column[0]] = column[1:len(column)]
	return tablesample(values = values, name = name)
#
def vertica_cursor(dsn: str):
	check_types([("dsn", dsn, [str], False)])
	try:
		import vertica_python
		success = 1
	except:
		try:
			print("Failed to import vertica_python, try to import pyodbc")
			import pyodbc
			success = 0
		except:
			raise Exception("Neither pyodbc or vertica_python is installed in your computer\nPlease install one of the Library using the 'pip install' command to be able to create a Vertica Cursor")
	if not(success):
		cursor = pyodbc.connect("DSN=" + dsn, autocommit = True).cursor()
	else:
		cursor = vertica_python.connect(** to_vertica_python_format(dsn), autocommit = True).cursor()
	return (cursor)
def read_dsn(dsn: str):
	check_types([("dsn", dsn, [str], False)])
	f = open(os.environ['ODBCINI'], "r")
	odbc = f.read()
	f.close()
	if ("[{}]".format(dsn) not in odbc):
		raise ValueError("The DSN '{}' doesn't exist".format(dsn))
	odbc = odbc.split("[{}]\n".format(dsn))[1].split("\n\n")[0].split("\n")
	dsn = {}
	for elem in odbc:
		info = elem.replace(' ','').split('=')
		dsn[info[0].lower()] = info[1]
	return (dsn)
def to_vertica_python_format(dsn: str):
	check_types([("dsn", dsn, [str], False)])
	dsn = read_dsn(dsn)
	conn_info = {'host': dsn["servername"], 'port': 5433, 'user': dsn["uid"], 'password': dsn["pwd"], 'database': dsn["database"]}
	return (conn_info)
#
def vdf_columns_names(columns: list, vdf):
	vdf_columns = vdf.get_columns()
	columns_names = []
	for column in columns:
		for vdf_column in vdf_columns:
			if (str_column(column).lower() == str_column(vdf_column).lower()):
				columns_names += [str_column(vdf_column)]
	return (columns_names)
#
def vdf_from_relation(relation: str, name: str = "VDF", cursor = None, dsn: str = "", schema: str = "public"):
	check_types([("relation", relation, [str], False), ("name", name, [str], False), ("dsn", dsn, [str], False), ("schema", schema, [str], False)])
	name = ''.join(ch for ch in name if ch.isalnum())
	from vertica_ml_python import vDataframe
	vdf = vDataframe("", empty = True)
	vdf.VERTICA_ML_PYTHON_VARIABLES["dsn"] = dsn
	if (cursor == None):
		from vertica_ml_python import vertica_cursor
		cursor = vertica_cursor(dsn)
	vdf.VERTICA_ML_PYTHON_VARIABLES["input_relation"] = name
	vdf.VERTICA_ML_PYTHON_VARIABLES["main_relation"] = relation
	vdf.VERTICA_ML_PYTHON_VARIABLES["schema"] = schema
	vdf.VERTICA_ML_PYTHON_VARIABLES["cursor"] = cursor
	vdf.VERTICA_ML_PYTHON_VARIABLES["query_on"] = False
	vdf.VERTICA_ML_PYTHON_VARIABLES["time_on"] = False
	vdf.VERTICA_ML_PYTHON_VARIABLES["where"] = []
	vdf.VERTICA_ML_PYTHON_VARIABLES["order_by"] = ['' for i in range(100)]
	vdf.VERTICA_ML_PYTHON_VARIABLES["exclude_columns"] = []
	vdf.VERTICA_ML_PYTHON_VARIABLES["history"] = []
	vdf.VERTICA_ML_PYTHON_VARIABLES["saving"] = []
	cursor.execute("DROP TABLE IF EXISTS v_temp_schema.VERTICA_ML_PYTHON_{}_TEST;".format(name))
	cursor.execute("CREATE LOCAL TEMPORARY TABLE VERTICA_ML_PYTHON_{}_TEST ON COMMIT PRESERVE ROWS AS SELECT * FROM {} LIMIT 10;".format(name, relation))
	cursor.execute("SELECT column_name, data_type FROM columns WHERE table_name = 'VERTICA_ML_PYTHON_{}_TEST' AND table_schema = 'v_temp_schema'".format(name))
	result = cursor.fetchall()
	cursor.execute("DROP TABLE IF EXISTS v_temp_schema.VERTICA_ML_PYTHON_{}_TEST;".format(name))
	vdf.VERTICA_ML_PYTHON_VARIABLES["columns"] = ['"' + item[0] + '"' for item in result]
	for column, ctype in result:
		if ('"' in column):
			print("\u26A0 Warning: A double quote \" was found in the column {}, its alias was changed using underscores '_' to {}".format(column, column.replace('"', '_')))
		from vertica_ml_python.vcolumn import vColumn
		new_vColumn = vColumn('"{}"'.format(column.replace('"', '_')), parent = vdf, transformations = [('"{}"'.format(column.replace('"', '""')), ctype, category_from_type(ctype))])
		setattr(vdf, '"{}"'.format(column.replace('"', '_')), new_vColumn)
		setattr(vdf, column.replace('"', '_'), new_vColumn)
	return (vdf)