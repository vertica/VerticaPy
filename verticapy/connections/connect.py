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
from configparser import ConfigParser

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
    path = os.path.dirname(verticapy.__file__) + "/connections/connections.verticapy"
    confparser = ConfigParser()
    confparser.optionxform = str
    try:
        confparser.read(path)
        confparser.remove_section("VERTICAPY_AUTO_CONNECTION")
    except:
        pass
    all_connections = confparser.sections()
    return all_connections


# ---#
def change_auto_connection(name: str):
    """
---------------------------------------------------------------------------
Changes the current auto connection.

Parameters
----------
name: str
	Name of the new auto connection.

See Also
--------
new_auto_connection : Saves a connection to automatically create DB cursors.
read_auto_connect   : Automatically creates a connection.
vertica_conn        : Creates a Vertica Database cursor using the input method.
	"""
    path = os.path.dirname(verticapy.__file__) + "/connections/connections.verticapy"
    confparser = ConfigParser()
    confparser.optionxform = str
    confparser.read(path)
    if confparser.has_section(name):
        confparser.remove_section("VERTICAPY_AUTO_CONNECTION")
        confparser.add_section("VERTICAPY_AUTO_CONNECTION")
        confparser.set("VERTICAPY_AUTO_CONNECTION", "name", name)
        f = open(path, "w+")
        confparser.write(f)
        f.close()
    else:
        raise NameError(
            "The input name is incorrect. The connection '{}' has never been created.\nUse the new_auto_connection function to create a new connection.".format(
                name
            )
        )


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
    check_types([("dsn", dsn, [dict],)])
    path = os.path.dirname(verticapy.__file__) + "/connections/connections.verticapy"
    confparser = ConfigParser()
    confparser.optionxform = str
    try:
        confparser.read(path)
    except:
        pass
    if confparser.has_section(name):
        confparser.remove_section(name)
    confparser.add_section(name)
    for elem in dsn:
        confparser.set(name, elem, dsn[elem])
    f = open(path, "w+")
    confparser.write(f)
    f.close()


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
    path = os.path.dirname(verticapy.__file__) + "/connections/connections.verticapy"
    confparser = ConfigParser()
    confparser.optionxform = str
    confparser.read(path)
    section = confparser.get("VERTICAPY_AUTO_CONNECTION", "name")
    return vertica_conn(section, path)


# ---#
def read_dsn(
    section: str, dsn: str = "",
):
    """
---------------------------------------------------------------------------
Reads the DSN information from the ODBCINI environment variable or the input
file.

Parameters
----------
section: str
    Name of the section in the configuration file.
dsn: str, optional
	Path to the file containing the credentials. If empty, the ODBCINI 
    environment variable will be used.

Returns
-------
dict
	dictionary with all the credentials
	"""
    check_types([("dsn", dsn, [str],), ("section", section, [str],)])
    confparser = ConfigParser()
    confparser.optionxform = str
    if not dsn:
        dsn = os.environ["ODBCINI"]
    confparser.read(dsn)
    if confparser.has_section(section):
        options = confparser.items(section)
        conn_info = {"port": 5433, "user": "dbadmin"}
        for elem in options:
            if elem[0].lower() == "servername":
                conn_info["host"] = elem[1]
            elif elem[0].lower() == "uid":
                conn_info["user"] = elem[1]
            elif elem[0].lower() == "port":
                try:
                    conn_info["port"] = int(elem[1])
                except:
                    conn_info["port"] = elem[1]
            elif elem[0].lower() == "pwd":
                conn_info["password"] = elem[1]
            elif elem[0].lower() == "kerberosservicename":
                conn_info["kerberos_service_name"] = elem[1]
            elif elem[0].lower() == "kerberoshostname":
                conn_info["kerberos_host_name"] = elem[1]
            else:
                conn_info[elem[0].lower()] = elem[1]
        return conn_info
    else:
        raise NameError("The DSN Section '{}' doesn't exist.".format(section))


# ---#
def vertica_conn(
    section: str, dsn: str = "",
):
    """
---------------------------------------------------------------------------
Reads the input DSN and creates a Vertica Database connection.

Parameters
----------
section: str
    Name of the section in the configuration file.
dsn: str, optional
    Path to the file containing the credentials. If empty, the ODBCINI 
    environment variable will be used.

Returns
-------
conn
	Database connection

See Also
--------
new_auto_connection : Saves a connection to automatically create DB cursors.
read_auto_connect   : Automatically creates a connection.
	"""
    check_types([("dsn", dsn, [str],)])
    conn = vertica_python.connect(**read_dsn(section, dsn))
    return conn
