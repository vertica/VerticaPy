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
# Standard Python Modules
import os

# VerticaPy Modules
from verticapy.utilities import check_types
import verticapy
from verticapy.errors import *

# Vertica Modules
import vertica_python

#
# ---#
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
    all_connections = [
        f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
    ]
    all_connections = [
        elem.replace(".verticapy", "")
        for elem in all_connections
        if ".verticapy" in elem
    ]
    if len(all_connections) == 1:
        print("The only available connection is {}".format(all_connections[0]))
    elif all_connections:
        print(
            "The available connections are the following: {}".format(
                ", ".join(all_connections)
            )
        )
    else:
        print(
            "No connections yet available. Use the new_auto_connection function to create your first one."
        )
    return all_connections


# ---#
def change_auto_connection(name: str = "DSN"):
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
vertica_conn        : Creates a Vertica Database cursor using the input method.
	"""
    try:
        path = os.path.dirname(
            verticapy.__file__
        ) + "/connections/all/{}.verticapy".format(name)
        file = open(path, "r")
        file.close()
    except:
        available_auto_connection()
        raise NameError(
            "The input name is incorrect. The connection '{}' has never been created.\nUse the new_auto_connection function to create a new connection.".format(
                name
            )
        )
    path = os.path.dirname(verticapy.__file__) + "/connections/auto_connection"
    file = open(path, "w+")
    file.write(name)
    file.close()


# ---#
def new_auto_connection(dsn: dict, name: str = "DSN"):
    """
---------------------------------------------------------------------------
Saves a connection to automatically create DB cursors. This will create a 
used-as-needed file to automatically set up a connection, avoiding redundant 
cursors.

Parameters
----------
dsn: dict
	Dictionnary containing the information to set up the connection.
		database : Database Name
		host     : Server ID
		password : User Password
		port     : Database Port (optional, default: 5433)
		user     : User ID (optional, default: dbadmin)
name: str, optional
	Name of the auto connection.

See Also
--------
change_auto_connection : Changes the current auto creation.
read_auto_connect      : Automatically creates a connection.
vertica_conn           : Creates a Vertica Database connection.
	"""
    check_types([("dsn", dsn, [dict], False)])
    if "port" not in dsn:
        print(
            "\u26A0 Warning: No port found in the 'dsn' dictionary. The default port is 5433."
        )
        dsn["port"] = 5433
    if "user" not in dsn:
        print(
            "\u26A0 Warning: No user found in the 'dsn' dictionary. The default user is 'dbadmin'."
        )
        dsn["user"] = "dbadmin"
    if ("password" not in dsn) or ("database" not in dsn) or ("host" not in dsn):
        raise ParameterError(
            'The dictionary \'dsn\' is incomplete. It must include all the needed credentitals to set up the connection.\nExample: dsn = { "host": "10.211.55.14", "port": "5433", "database": "testdb", "password": "XxX", "user": "dbadmin"}"'
        )
    path = os.path.dirname(verticapy.__file__) + "/connections/all/{}.verticapy".format(
        name
    )
    file = open(path, "w+")
    file.write(
        "Connection - {}\nhost: {}\nport: {}\ndatabase: {}\nuser: {}\npassword: {}".format(
            name,
            dsn["host"],
            dsn["port"],
            dsn["database"],
            dsn["user"],
            dsn["password"],
        )
    )
    file.close()


# ---#
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
vertica_conn        : Creates a Vertica Database cursor using the input method.
	"""
    path = os.path.dirname(verticapy.__file__) + "/connections"
    try:
        file = open(path + "/auto_connection", "r")
        name = file.read()
        path += "/all/{}.verticapy".format(name)
        file = open(path, "r")
    except:
        raise NameError(
            "No auto connection is available. To create an auto connection, use the new_auto_connection function of the verticapy.connections.connect module."
        )
    try:
        dsn = file.read()
        dsn = dsn.split("\n")
        dsn[1] = dsn[1][6:]
        dsn[2] = dsn[2][6:]
        dsn[3] = dsn[3][10:]
        dsn[4] = dsn[4][6:]
        dsn[5] = dsn[5][10:]
    except:
        raise ParsingError(
            "The auto connection format seems to be incorrect. To create a new auto connection, use the new_auto_connection function of the verticapy.connections.connect module."
        )
    conn = vertica_python.connect(
        **{
            "host": dsn[1],
            "port": dsn[2],
            "database": dsn[3],
            "user": dsn[4],
            "password": dsn[5],
        }
    )
    return conn


# ---#
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
    f = open(os.environ["ODBCINI"], "r")
    odbc = f.read()
    f.close()
    if "[{}]".format(dsn) not in odbc:
        raise NameError("The DSN '{}' doesn't exist.".format(dsn))
    odbc = odbc.split("[{}]\n".format(dsn))[1].split("\n\n")[0].split("\n")
    dsn = {}
    for elem in odbc:
        info = elem.replace(" ", "").split("=")
        dsn[info[0].lower()] = info[1]
    return dsn


# ---#
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
    conn_info = {
        "host": dsn["servername"],
        "port": 5433,
        "user": dsn["uid"],
        "password": dsn["pwd"],
        "database": dsn["database"],
    }
    return conn_info


# ---#
def vertica_conn(dsn: str):
    """
---------------------------------------------------------------------------
Reads the input DSN from the ODBCINI environment and creates a Vertica 
Database connection.

Parameters
----------
dsn: str
	DSN name

Returns
-------
conn
	Database connection

See Also
--------
new_auto_connection : Saves a connection to automatically create DB cursors.
read_auto_connect   : Automatically creates a connection.
	"""
    check_types([("dsn", dsn, [str], False)])
    conn = vertica_python.connect(**to_vertica_python_format(dsn))
    return conn


def vertica_cursor(dsn: str):
    print(
        "\u26A0 Warning: This function has been deprecated, please use vertica_conn to create a DB connection."
    )
    return vertica_conn(dsn).cursor()
