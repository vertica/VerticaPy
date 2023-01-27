# (c) Copyright [2018-2023] Micro Focus or one of its affiliates.
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
# Standard Python Modules
import os
from configparser import ConfigParser

# VerticaPy Modules
import verticapy
from verticapy.decorators import (
    save_verticapy_logs,
    check_dtypes,
    check_minimum_version,
)
from verticapy.toolbox import is_special_symbol, get_special_symbols
from verticapy.errors import *

# Vertica Modules
import vertica_python

# Global Variables
VERTICAPY_AUTO_CONNECTION = "VERTICAPY_AUTO_CONNECTION"

#
# ---#
def available_connections():
    """
------------------------------------------------------------------------------------
Displays all the available connections.

Returns
-------
list
	all the available connections.
	"""
    path = get_connection_file()
    confparser = ConfigParser()
    confparser.optionxform = str
    confparser.read(path)
    if confparser.has_section(VERTICAPY_AUTO_CONNECTION):
        confparser.remove_section(VERTICAPY_AUTO_CONNECTION)
    all_connections = confparser.sections()
    return all_connections


available_auto_connection = available_connections
# ---#
def change_auto_connection(name: str):
    """
------------------------------------------------------------------------------------
Changes the current auto connection.

Parameters
----------
name: str
	Name of the new auto connection.
	"""
    path = get_connection_file()
    confparser = ConfigParser()
    confparser.optionxform = str
    confparser.read(path)

    if confparser.has_section(name):

        confparser.remove_section(VERTICAPY_AUTO_CONNECTION)
        confparser.add_section(VERTICAPY_AUTO_CONNECTION)
        confparser.set(VERTICAPY_AUTO_CONNECTION, "name", name)
        f = open(path, "w+")
        confparser.write(f)
        f.close()

    else:

        raise NameError(
            "The input name is incorrect. The connection "
            f"'{name}' has never been created.\nUse the "
            "new_connection function to create a new "
            "connection."
        )


# ---#
def close_connection():
    """
------------------------------------------------------------------------------------
Closes the connection to the database.
    """
    if verticapy.OPTIONS["connection"]["conn"] and not (
        verticapy.OPTIONS["connection"]["conn"].closed()
    ):
        verticapy.OPTIONS["connection"]["conn"].close()


# ---#
def connect(section: str, dsn: str = ""):
    """
------------------------------------------------------------------------------------
Connects to the database.

Parameters
----------
section: str
    Name of the section in the configuration file.
dsn: str, optional
    Path to the file containing the credentials. If empty, the 
    Connection File will be used.
    """
    prev_conn = verticapy.OPTIONS["connection"]["conn"]
    if not (dsn):
        dsn = get_connection_file()
    if prev_conn and not (prev_conn.closed()):
        prev_conn.close()
    try:
        verticapy.OPTIONS["connection"]["conn"] = vertica_connection(section, dsn)
        verticapy.OPTIONS["connection"]["dsn"] = dsn
        verticapy.OPTIONS["connection"]["section"] = section
    except Exception as e:
        if "The DSN Section" in str(e):
            raise ConnectionError(
                f"The connection '{section}' does not exist. To create "
                "a new connection, you use the 'new_connection' "
                "function with your credentials: {'database': ..., "
                "'host': ..., 'password': ..., 'user': ...}.\n"
                "To view available connections, use the "
                "the 'available_connections' function."
            )
        raise (e)


# ---#
def current_connection():
    """
------------------------------------------------------------------------------------
Returns the current database connection.
If the connection is closed, VerticaPy attempts to reconnect with the 
existing connection.

If the connection attempt fails, VerticaPy attempts to reconnect using 
stored credentials. If this also fails, VerticaPy attempts to connect using
an auto connection. Otherwise, VerticaPy attempts to connect to a 
VerticaLab Environment.
    """

    # Look if the connection does not exist or is closed
    if (
        not (verticapy.OPTIONS["connection"]["conn"])
        or verticapy.OPTIONS["connection"]["conn"].closed()
    ):

        # Connection using the existing credentials
        if (
            verticapy.OPTIONS["connection"]["section"]
            and verticapy.OPTIONS["connection"]["dsn"]
        ):
            connect(
                verticapy.OPTIONS["connection"]["section"],
                verticapy.OPTIONS["connection"]["dsn"],
            )

        else:

            try:
                # Connection using the Auto Connection
                read_auto_connect()

            except Exception as e:

                try:
                    # Connection to the VerticaLab environment
                    conn = verticalab_connection()
                    verticapy.OPTIONS["connection"]["conn"] = conn

                except:
                    raise (e)

    return verticapy.OPTIONS["connection"]["conn"]


# ---#
def current_cursor():
    """
------------------------------------------------------------------------------------
Returns the current database cursor.
    """
    return current_connection().cursor()


# ---#
@check_dtypes
def delete_connection(name: str):
    """
------------------------------------------------------------------------------------
Deletes a specified connection from the connection file.

Parameters
----------
name: str
    Name of the connection.

Returns
-------
bool
    True if the connection was deleted, False otherwise.
    """
    path = get_connection_file()
    confparser = ConfigParser()
    confparser.optionxform = str
    confparser.read(path)

    if confparser.has_section(name):

        confparser.remove_section(name)
        if confparser.has_section(VERTICAPY_AUTO_CONNECTION):
            name_auto = confparser.get(VERTICAPY_AUTO_CONNECTION, "name")
            if name_auto == name:
                confparser.remove_section(VERTICAPY_AUTO_CONNECTION)
        f = open(path, "w+")
        confparser.write(f)
        f.close()
        return True

    else:

        warnings.warn(f"The connection {name} does not exist.", Warning)
        return False


# ---#
def get_connection_file():
    """
------------------------------------------------------------------------------------
Gets (and creates, if necessary) the auto-connection file.
If the environment variable 'VERTICAPY_CONNECTIONS' is set, it is assumed 
to be the full path to the auto-connection file.
Otherwise, we reference "connections.verticapy" in the hidden ".verticapy" 
folder in the user's home directory.

Returns
-------
string
    the full path to the auto-connection file.
    """
    if "VERTICAPY_CONNECTIONS" in os.environ:
        return os.environ["VERTICAPY_CONNECTIONS"]
    path = os.path.join(os.path.expanduser("~"), ".vertica")
    os.makedirs(path, 0o700, exist_ok=True)
    path = os.path.join(path, "connections.verticapy")
    return path


# ---#
@check_dtypes
def new_connection(
    conn_info: dict,
    name: str = "vertica_connection",
    auto: bool = True,
    overwrite: bool = True,
):
    """
------------------------------------------------------------------------------------
Saves the new connection in the VerticaPy connection file.
The information is saved plaintext in the local machine.
The function 'get_connection_file' returns the associated connection file path.
If you want a temporary connection, you can use the 'set_connection' function.

Parameters
----------
conn_info: dict
	Dictionnary containing the information to set up the connection.
		database : Database Name.
		host     : Server ID.
		password : User Password.
		port     : Database Port (optional, default: 5433).
		user     : User ID (optional, default: dbadmin).
        ...
        env      : Bool to indicate whether the user and password are replaced 
                   by the associated environment variables. If True, VerticaPy 
                   reads the associated environment variables instead of 
                   writing and directly using the username and password. 
                   For example: {'user': 'ENV_USER', 'password': 'ENV_PASSWORD'}
                   This works only for the user and password. The real values 
                   of the other variables are stored plaintext in the VerticaPy 
                   connection file. Using the enviornment variables hides the 
                   username and password in cases where the local machine is 
                   shared.
name: str, optional
	Name of the connection.
auto: bool, optional
    If set to True, the connection will become the new auto-connection.
overwrite: bool, optional
    If set to True and the connection already exists, it will be 
    overwritten.
env: bool, optional
    If True, user and password are replaced by the associated environment 
    variables. VerticaPy reads the associated environment variables instead 
    of writing and directly using the username and password.
    For example: {'user': 'ENV_USER', 'password': 'ENV_PASSWORD'}  
	"""
    path = get_connection_file()
    confparser = ConfigParser()
    confparser.optionxform = str
    confparser.read(path)

    if confparser.has_section(name):

        if not (overwrite):
            raise ParserError(
                f"The section '{name}' already exists. You "
                "can overwrite it by setting the parameter "
                "'overwrite' to True."
            )
        confparser.remove_section(name)

    confparser.add_section(name)
    for c in conn_info:
        confparser.set(name, c, str(conn_info[c]))
    f = open(path, "w+")
    confparser.write(f)
    f.close()
    if auto:
        change_auto_connection(name)

    connect(name, path)


new_auto_connection = new_connection
# ---#
def read_auto_connect():
    """
------------------------------------------------------------------------------------
Automatically creates a connection using the auto-connection.
	"""
    path = get_connection_file()
    confparser = ConfigParser()
    confparser.optionxform = str
    confparser.read(path)
    if confparser.has_section(VERTICAPY_AUTO_CONNECTION):
        section = confparser.get(VERTICAPY_AUTO_CONNECTION, "name")
    else:
        raise ConnectionError(
            "No Auto Connection available. You can create one using "
            "the 'new_connection' function or set manually a connection"
            " using the 'set_connection' function."
        )
    connect(section, path)


# ---#
@check_dtypes
def read_dsn(section: str, dsn: str = ""):
    """
------------------------------------------------------------------------------------
Reads the DSN information from the VERTICAPY_CONNECTIONS environment 
variable or the input file.

Parameters
----------
section: str
    Name of the section in the configuration file.
dsn: str, optional
	Path to the file containing the credentials. If empty, the 
    VERTICAPY_CONNECTIONS environment variable will be used.

Returns
-------
dict
	dictionary with all the credentials.
	"""
    confparser = ConfigParser()
    confparser.optionxform = str

    if not dsn:
        dsn = get_connection_file()

    confparser.read(dsn)

    if confparser.has_section(section):

        options = confparser.items(section)
        conn_info = {
            "port": 5433,
            "user": "dbadmin",
            "session_label": f"verticapy-{verticapy.__version__}-{verticapy.OPTIONS['identifier']}",
        }

        env = False
        for option_name, option_val in options:
            if option_name.lower().startswith("env"):
                if option_val.lower() in ("true", "t", "yes", "y"):
                    env = True
                break

        for option_name, option_val in options:

            option_name = option_name.lower()

            if option_name in ("pwd", "password", "uid", "user") and env:
                if option_name == "pwd":
                    option_name = "password"
                elif option_name == "uid":
                    option_name = "user"
                if os.getenv(option_val) != None:
                    conn_info[option_name] = os.getenv(option_val)
                else:
                    raise ParameterError(
                        f"The '{option_name}' environment variable '{option_val}' does not exist "
                        "and the 'env' option is set to True.\n"
                        "Impossible to set up the final DSN.\nTips: You can manually set it "
                        "up by importing os and running the following command:\n"
                        f"os.environ['{option_name}'] = '******'"
                    )

            elif option_name in ("servername", "server"):
                conn_info["host"] = option_val

            elif option_name == "uid":
                conn_info["user"] = option_val

            elif (option_name in ("port", "connection_timeout")) and (
                option_val.isnumeric()
            ):
                conn_info[option_name] = int(option_val)

            elif option_name == "pwd":
                conn_info["password"] = option_val

            elif option_name == "backup_server_node":
                backup_server_node = {}
                exec(f"id_port = '{option_val}'", {}, backup_server_node)
                conn_info["backup_server_node"] = backup_server_node["id_port"]

            elif option_name == "kerberosservicename":
                conn_info["kerberos_service_name"] = option_val

            elif option_name == "kerberoshostname":
                conn_info["kerberos_host_name"] = option_val

            elif "vp_test_" in option_name:
                conn_info[option_name[8:]] = option_val

            elif option_name in (
                "ssl",
                "autocommit",
                "use_prepared_statements",
                "connection_load_balance",
                "disable_copy_local",
            ):
                option_val = option_val.lower()
                conn_info[option_name] = option_val in ("true", "t", "yes", "y")

            elif option_name != "session_label" and not (option_name.startswith("env")):
                conn_info[option_name] = option_val

        return conn_info

    else:

        raise NameError(f"The DSN Section '{section}' doesn't exist.")


# ---#
def set_connection(conn):
    """
------------------------------------------------------------------------------------
Saves a custom connection to the VerticaPy object. This allows you to 
specify, for example, a JDBC or ODBC connection. This should not be 
confused with a native VerticaPy connection created by the new_connection 
function.

Parameters
----------
conn: object
    Connection object.
    """
    try:
        conn.cursor().execute("SELECT /*+LABEL('connect.set_connection')*/ 1;")
        res = conn.cursor().fetchone()[0]
        assert res == 1
    except:
        ParameterError("The input connector is not working properly.")
    verticapy.OPTIONS["connection"]["conn"] = conn
    verticapy.OPTIONS["connection"]["dsn"] = None
    verticapy.OPTIONS["connection"]["section"] = None


# ---#
@check_dtypes
def set_external_connection(cid: str, rowset: int = 500, symbol: str = "$"):
    """
------------------------------------------------------------------------------------
Sets a Connection Identifier Database. It connects to an external
source using DBLINK. For more information, see:
https://github.com/vertica/dblink

Parameters
----------
cid: str
    Connection Identifier Database.
rowset: int, optional
    Number of rows retrieved from the remote database during each 
    SQLFetch() cycle.
symbol: str, optional
    One of the following:
    "$", "€", "£", "%", "@", "&", "§", "?", "!"
    A special character, to identify the connection. 
    For example, if the symbol is '$', you can call external tables 
    with the input cid by writing $$$QUERY$$$, where QUERY represents 
    a custom query.
    """
    assert is_special_symbol(symbol), ParameterError(
        f"Parameter 'symbol' must be a special char. One of the following: {', '.join(get_special_symbols())}"
    )
    if isinstance(cid, str) and isinstance(rowset, int):
        verticapy.OPTIONS["external_connection"][symbol] = {
            "cid": cid,
            "rowset": rowset,
        }
    else:
        raise ParameterError(
            "Could not set the external connection. Found a wrong type."
        )


# ---#
@check_dtypes
def vertica_connection(section: str, dsn: str = ""):
    """
------------------------------------------------------------------------------------
Reads the input DSN and creates a Vertica Database connection.

Parameters
----------
section: str
    Name of the section in the configuration file.
dsn: str, optional
    Path to the file containing the credentials. If empty, the 
    VERTICAPY_CONNECTIONS environment variable will be used.

Returns
-------
conn
	Database connection.
	"""
    return vertica_python.connect(**read_dsn(section, dsn))


# ---#
def verticalab_connection():
    """
------------------------------------------------------------------------------------
Returns the VerticaLab connection if possible.

Returns
-------
conn
    Database connection.
    """
    conn_info = {
        "host": "vertica-demo",
        "port": 5433,
        "user": "dbadmin",
        "password": "",
        "database": "demo",
        "backup_server_node": ["localhost"],
        "session_label": f"verticapy-{verticapy.__version__}-{verticapy.OPTIONS['identifier']}",
    }
    return vertica_python.connect(**conn_info)
