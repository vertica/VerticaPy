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
import vertica_python

from verticapy.connection._global import (
    _connection,
    SESSION_LABEL,
    VERTICAPY_AUTO_CONNECTION,
)
from verticapy.connection.read import read_dsn
from verticapy.connection.utils import get_confparser, get_connection_file
from verticapy.errors import ConnectionError, ParameterError


def auto_connect():
    """
Automatically creates a connection using the auto-connection.
    """
    confparser = get_confparser()

    if confparser.has_section(VERTICAPY_AUTO_CONNECTION):
        section = confparser.get(VERTICAPY_AUTO_CONNECTION, "name")
    else:
        raise ConnectionError(
            "No Auto Connection available. You can create one using "
            "the 'new_connection' function or set manually a connection"
            " using the 'set_connection' function."
        )
    connect(section)


read_auto_connect = auto_connect


def close_connection():
    """
Closes the connection to the database.
    """
    if _connection["conn"] and not (_connection["conn"].closed()):
        _connection["conn"].close()


def connect(section: str, dsn: str = ""):
    """
Connects to the database.

Parameters
----------
section: str
    Name of the section in the configuration file.
dsn: str, optional
    Path to the file containing the credentials. If empty, the 
    Connection File will be used.
    """
    global _connection
    prev_conn = _connection["conn"]
    if not (dsn):
        dsn = get_connection_file()
    if prev_conn and not (prev_conn.closed()):
        prev_conn.close()
    try:
        _connection["conn"] = vertica_connection(section, dsn)
        _connection["dsn"] = dsn
        _connection["section"] = section
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


def current_connection():
    """
Returns the current database connection.
If the connection is closed, VerticaPy attempts to reconnect with the 
existing connection.

If the connection attempt fails, VerticaPy attempts to reconnect using 
stored credentials. If this also fails, VerticaPy attempts to connect using
an auto connection. Otherwise, VerticaPy attempts to connect to a 
VerticaLab Environment.
    """
    global _connection
    # Look if the connection does not exist or is closed
    if not (_connection["conn"]) or _connection["conn"].closed():

        # Connection using the existing credentials
        if _connection["section"] and _connection["dsn"]:
            connect(
                _connection["section"], _connection["dsn"],
            )

        else:

            try:
                # Connection using the Auto Connection
                auto_connect()

            except Exception as e:

                try:
                    # Connection to the VerticaLab environment
                    conn = verticalab_connection()
                    _connection["conn"] = conn

                except:
                    raise (e)

    return _connection["conn"]


def current_cursor():
    """
Returns the current database cursor.
    """
    return current_connection().cursor()


def set_connection(conn):
    """
Saves a custom connection to the VerticaPy object. This allows you to 
specify, for example, a JDBC or ODBC connection. This should not be 
confused with a native VerticaPy connection created by the new_connection 
function.

Parameters
----------
conn: object
    Connection object.
    """
    global _connection
    try:
        conn.cursor().execute("SELECT /*+LABEL('connect.set_connection')*/ 1;")
        res = conn.cursor().fetchone()[0]
        assert res == 1
    except:
        ParameterError("The input connector is not working properly.")
    _connection["conn"] = conn
    _connection["dsn"] = None
    _connection["section"] = None


def vertica_connection(section: str, dsn: str = ""):
    """
Reads the input DSN and creates a Vertica Database connection.

Parameters
----------
section: str
    Name of the section in the configuration file.
dsn: str, optional
    Path to the file containing the credentials. If empty, the 
    VERTICAPY_CONNECTION environment variable will be used.

Returns
-------
conn
    Database connection.
    """
    return vertica_python.connect(**read_dsn(section, dsn))


def verticalab_connection():
    """
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
        "session_label": SESSION_LABEL,
    }
    return vertica_python.connect(**conn_info)
