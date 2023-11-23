"""
Copyright  (c)  2018-2024 Open Text  or  one  of its
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
import os
from typing import Optional

from verticapy._typing import NoneType
from verticapy.connection.global_connection import get_global_connection
from verticapy.connection.utils import get_confparser


def available_connections() -> list[str]:
    """
    Displays all available connections.

    Returns
    -------
    list
        all available connections.

    Example
    -------
    Displays all available connections:

    .. ipython:: python

        from verticapy.connection import available_connections

        available_connections()
    """
    gb_conn = get_global_connection()

    confparser = get_confparser()
    if confparser.has_section(gb_conn.vpy_auto_connection):
        confparser.remove_section(gb_conn.vpy_auto_connection)
    all_connections = confparser.sections()
    return all_connections


available_auto_connection = available_connections


def read_dsn(section: str, dsn: Optional[str] = None) -> dict:
    """
    Reads the DSN information from the VERTICAPY_
    CONNECTION environment variable or the input
    file.

    Parameters
    ----------
    section: str
        Name of the section in the configuration file
        that contains the credentials.
    dsn: str, optional
        Path to the file containing the credentials.
        If empty, the VERTICAPY_CONNECTION environment
        variable is used.

    Returns
    -------
    dict
        dictionary with all the credentials.

    Example
    -------
    Read the DSN information from the ODBCINI environment variable:

    .. code-block:: python

        from verticapy.connection import read_dsn

        dsn = read_dsn("VerticaDSN")
        dsn

    | ``{``
    | ``'database': 'testdb',``
    | ``'description': 'DSN for Vertica',``
    | ``'driver': '/Library/Vertica/ODBC/lib/libverticaodbc.dylib',``
    | ``'host': '10.211.55.14',``
    | ``'kerberos_host_name': 'badr',``
    | ``'password': 'XxX',``
    | ``'port': '5433',``
    | ``'user': 'dbadmin'``
    | ``}``

    Read the DSN information from a input file:

    .. code-block:: python

        dsn = read_dsn("vp_test_config",
               "/Users/Badr/Library/Python/3.6/lib/python/site-packages/verticapy/tests/verticaPy_test.conf")
        dsn

    | ``{``
    | ``'password': 'XxX',``
    | ``'port': 5433,``
    | ``'user': 'dbadmin',``
    | ``'vp_test_database': 'testdb',``
    | ``'vp_test_host': '10.211.55.14',``
    | ``'vp_test_log_dir': 'mylog/vp_tox_tests_log',``
    | ``'vp_test_password': 'XxX',``
    | ``'vp_test_port': '5433',``
    | ``'vp_test_user': 'dbadmin'``
    | ``}``
    """
    confparser = get_confparser(dsn)

    if confparser.has_section(section):
        options = confparser.items(section)

        gb_conn = get_global_connection()
        conn_info = {
            "port": 5433,
            "user": "dbadmin",
            "session_label": gb_conn.vpy_session_label,
            "unicode_error": "ignore",
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
                if not isinstance(os.getenv(option_val), NoneType):
                    conn_info[option_name] = os.getenv(option_val)
                else:
                    raise ValueError(
                        f"The '{option_name}' environment variable "
                        f"'{option_val}' does not exist and the 'env' "
                        "option is set to True.\nImpossible to set up "
                        "the final DSN.\nTips: You can manually set "
                        "it up by importing os and running the following "
                        f"command:\nos.environ['{option_name}'] = '******'"
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

            elif option_name != "session_label" and not option_name.startswith("env"):
                conn_info[option_name] = option_val

        return conn_info

    else:
        raise NameError(f"The DSN Section '{section}' doesn't exist.")
