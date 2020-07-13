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
# VerticaPy allows user to create vDataFrames (Virtual Dataframes). 
# vDataFrames simplify data exploration, data cleaning and MACHINE LEARNING     
# in VERTICA. It is an object which keeps in it all the actions that the user 
# wants to achieve and execute them when they are needed.    										
#																					
# The purpose is to bring the logic to the data and not the opposite !
#
# 
# Modules
#
# Standard Python Modules
import os
# VerticaPy Modules
from verticapy.utilities import check_types
import verticapy
#
#---#
def available_auto_connection():
	"""
---------------------------------------------------------------------------
Displays all the available auto connections.

Returns
-------
list
	all the available auto connections.

See Also
--------
new_auto_connection : Saves a connection to automatically create DB cursors.
	"""
	path = os.path.dirname(verticapy.__file__) + "/connections/all/"
	all_connections = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
	all_connections = [elem.replace(".vertica", "") for elem in all_connections if ".vertica" in elem]
	if (len(all_connections) == 1):
		print("The only available connection is {}".format(all_connections[0]))
	elif (all_connections):
		print("The available connections are the following: {}".format(", ".join(all_connections)))
	else:
		print("No connections yet available. Use the new_auto_connection function to create your first one.")
	return(all_connections)
#---#
def change_auto_connection(name: str = "VML"):
	"""
---------------------------------------------------------------------------
Changes the current auto connection.

Parameters
----------
name: str, optional
	Name of the new auto connection.

See Also
--------
new_auto_connection : Saves a connection to automatically create DB cursors.
read_auto_connect   : Automatically creates a connection.
vertica_cursor      : Creates a Vertica Database cursor using the input method.
	"""
	try:
		path = os.path.dirname(verticapy.__file__) + "/connections/all/{}.vertica".format(name)
		file = open(path, "r")
		file.close()
	except:
		available_auto_connection()
		raise ValueError("The input name is incorrect. The connection '{}' has never been created.\nUse the new_auto_connection function to create a new connection.".format(name))
	path = os.path.dirname(verticapy.__file__) + "/connections/auto_connection"
	file = open(path, "w+")
	file.write(name)
	file.close()
#---#
def new_auto_connection(dsn: dict, 
					    method: str = "auto",
					    name: str = "VML"):
	"""
---------------------------------------------------------------------------
Saves a connection to automatically create DB cursors. It will create a 
file which will be used to automatically set up a connection when 
it is needed. It helps you to avoid redundant cursors creation.

Parameters
----------
dsn: dict
	Dictionnary containing the information to set up the connection.
		database : Database Name
		driver   : ODBC driver (only for pyodbc)
		host     : Server ID
		password : User Password
		port     : Database Port (optional, default: 5433)
		user     : User ID (optional, default: dbadmin)
method: str, optional
	Method used to save the connection.
	auto           : uses vertica_python if vertica_python installed, 
		otherwise pyodbc, otherwise jaydebeapi.
	pyodbc         : ODBC.
	jaydebeapi     : JDBC.
	vertica_python : Vertica Python Native Client (recommended).
name: str, optional
	Name of the auto connection.

See Also
--------
change_auto_connection : Changes the current auto creation.
read_auto_connect      : Automatically creates a connection.
vertica_cursor         : Creates a Vertica Database cursor using the input method.
	"""
	check_types([
		("dsn", dsn, [dict], False),
		("method", method, ["auto", "pyodbc", "jaydebeapi", "vertica_python"], True)])
	if ("port" not in dsn):
		print("\u26A0 Warning: No port found in the 'dsn' dictionary. The default port is 5433.")
		dsn["port"] = 5433
	if ("user" not in dsn):
		print("\u26A0 Warning: No user found in the 'dsn' dictionary. The default user is 'dbadmin'.")
		dsn["user"] = "dbadmin"
	if (method == "auto"):
		try:
			import vertica_python
			method = "vertica_python"
		except:
			try:
				import pyodbc
				method = "pyodbc"
			except:
				try:
					import jaydebeapi
					method = "jaydebeapi"
				except:
					raise Exception("Neither pyodbc, vertica_python or jaydebeapi is installed in your computer\nPlease install one of the Library using the 'pip install' command to be able to create a Vertica Cursor")
	if ("driver" not in dsn and method == "pyodbc"):
		raise ValueError("If the method is pyodbc, the Vertica ODBC driver location must be in the 'dsn' dictionary.")
	if ("password" not in dsn) or ("database" not in dsn) or ("host" not in dsn):
		raise ValueError('The dictionary \'dsn\' is incomplete. It must include all the needed credentitals to set up the connection.\nExample: dsn = { "host": "10.211.55.14", "port": "5433", "database": "testdb", "password": "XxX", "user": "dbadmin"}"')
	path = os.path.dirname(verticapy.__file__) + "/connections/all/{}.vertica".format(name)
	if (method == "pyodbc"):
		import pyodbc
		dsn = ("DRIVER={}; SERVER={}; DATABASE={}; PORT={}; UID={}; PWD={};").format(dsn["driver"], dsn["host"], dsn["database"], dsn["port"], dsn["user"], dsn["password"])
		conn = pyodbc.connect(dsn)
		conn.close()
		file = open(path, "w+")
		file.write("pyodbc\n{}".format(dsn))
		file.close()
	elif (method == "jaydebeapi"):
		import jaydebeapi
		jdbc_driver_name = "com.vertica.jdbc.Driver"
		jdbc_driver_loc = os.path.dirname(verticapy.__file__) + "/connections/vertica-jdbc-9.3.1-0.jar"
		connection_string = 'jdbc:vertica://{}:{}/{}'.format(dsn["host"], dsn["port"], dsn["database"])
		url = '{}:user={};password={}'.format(connection_string, dsn["user"], dsn["password"])
		conn = jaydebeapi.connect(jdbc_driver_name, connection_string, {'user': dsn["user"], 'password': dsn["password"]}, jars = jdbc_driver_loc)
		conn.close()
		file = open(path, "w+")
		file.write("jaydebeapi\n{}\n{}\n{}".format(connection_string, dsn["user"], dsn["password"]))
		file.close()
	elif (method == "vertica_python"):
		import vertica_python
		conn = vertica_python.connect(** dsn)
		conn.close()
		file = open(path, "w+")
		file.write("vertica_python\n{}\n{}\n{}\n{}\n{}".format(dsn["host"], dsn["port"], dsn["database"], dsn["user"], dsn["password"]))
		file.close()
#---#
def read_auto_connect():
	"""
---------------------------------------------------------------------------
Automatically creates a connection using the one created when using the 
function new_auto_connection.

Returns
-------
conn
	Database connection

See Also
--------
new_auto_connection : Saves a connection to automatically create DB cursors.
vertica_cursor      : Creates a Vertica Database cursor using the input method.
	"""
	path = os.path.dirname(verticapy.__file__) + "/connections"
	try:
		file = open(path + "/auto_connection", "r")
		name = file.read()
		path += "/all/{}.vertica".format(name)
		file = open(path, "r")
	except:
		raise Exception("No auto connection is available. To create an auto connection, use the new_auto_connection function of the verticapy.connections.connect module.")
	dsn = file.read()
	dsn = dsn.split("\n")
	if (dsn[0] == "vertica_python"):
		import vertica_python
		conn = vertica_python.connect(** {"host": dsn[1], "port": dsn[2], "database": dsn[3], "user": dsn[4], "password": dsn[5]}, autocommit = True)
	elif (dsn[0] == "pyodbc"):
		import pyodbc
		conn = pyodbc.connect(dsn[1], autocommit = True)
	elif (dsn[0] == "jaydebeapi"):
		import jaydebeapi
		jdbc_driver_name = "com.vertica.jdbc.Driver"
		jdbc_driver_loc = os.path.dirname(verticapy.__file__) + "/connections/vertica-jdbc-9.3.1-0.jar"
		conn = jaydebeapi.connect(jdbc_driver_name, dsn[1], {'user': dsn[2], 'password': dsn[3]}, jars = jdbc_driver_loc)
	else:
		raise Exception("The auto connection format is incorrect. To create a new auto connection, use the new_auto_connection function of the verticapy.connections.connect module.")
	return(conn)
#---#
def read_dsn(dsn: str):
	"""
---------------------------------------------------------------------------
Reads the DSN information from the ODBCINI environment variable.

Parameters
----------
dsn: str
	DSN name

Returns
-------
dict
	dictionary with all the credentials
	"""
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
#---#
def to_vertica_python_format(dsn: str):
	"""
---------------------------------------------------------------------------
Converts the ODBC dictionary obtained with the read_dsn method to the 
vertica_python format.

Parameters
----------
dsn: str
	DSN name

Returns
-------
dict
	dictionary with all the credentials
	"""
	check_types([("dsn", dsn, [str], False)])
	dsn = read_dsn(dsn)
	conn_info = {'host': dsn["servername"], 'port': 5433, 'user': dsn["uid"], 'password': dsn["pwd"], 'database': dsn["database"]}
	return (conn_info)
#---#
def vertica_cursor(dsn: str,
				   method: str = "auto"):
	"""
---------------------------------------------------------------------------
Reads the input DSN from the ODBCINI environment and creates a Vertica 
Database cursor using the input method.

Parameters
----------
dsn: str
	DSN name
method: str, optional
	Method used to create the connection.
	auto           : uses vertica_python if vertica_python installed, 
		otherwise pyodbc, otherwise jaydebeapi.
	pyodbc         : ODBC.
	jaydebeapi     : JDBC.
	vertica_python : Vertica Python Native Client (recommended).

Returns
-------
cursor
	Database cursor

See Also
--------
new_auto_connection : Saves a connection to automatically create DB cursors.
read_auto_connect   : Automatically creates a connection.
	"""
	check_types([
		("dsn", dsn, [str], False),
		("method", method, ["auto", "pyodbc", "jaydebeapi", "vertica_python"], True)])
	if (method == "auto"):
		try:
			import vertica_python
			method = "vertica_python"
		except:
			try:
				import pyodbc
				method = "pyodbc"
			except:
				try:
					import jaydebeapi
					method = "jaydebeapi"
				except:
					raise Exception("Neither pyodbc, vertica_python or jaydebeapi is installed in your computer\nPlease install one of the Library using the 'pip install' command to be able to create a Vertica Cursor")
	if (method == "pyodbc"):
		import pyodbc
		cursor = pyodbc.connect("DSN=" + dsn, autocommit = True).cursor()
	elif (method == "vertica_python"):
		import vertica_python
		cursor = vertica_python.connect(** to_vertica_python_format(dsn), autocommit = True).cursor()
	elif (method == "jaydebeapi"):
		import jaydebeapi
		dsn = to_vertica_python_format(dsn)
		jdbc_driver_name = "com.vertica.jdbc.Driver"
		jdbc_driver_loc = path = os.path.dirname(verticapy.__file__) + "/connections/vertica-jdbc-9.3.1-0.jar"
		connection_string = 'jdbc:vertica://{}:{}/{}'.format(dsn["host"], dsn["port"], dsn["database"])
		url = '{}:user={};password={}'.format(connection_string, dsn["user"], dsn["password"])
		conn = jaydebeapi.connect(jdbc_driver_name, connection_string, {'user': dsn["user"], 'password': dsn["password"]}, jars = jdbc_driver_loc)
		cursor = conn.cursor()
	return (cursor)