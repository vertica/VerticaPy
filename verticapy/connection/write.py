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

from getpass import getpass
import warnings
import vertica_python

import verticapy._config.config as conf
from verticapy.connection.errors import ConnectionError, OAuthTokenRefreshError
from verticapy.connection.global_connection import get_global_connection
from verticapy.connection.oauth_manager import OAuthManager
from verticapy.connection.read import read_dsn
from verticapy.connection.utils import get_confparser, get_connection_file


def change_auto_connection(name: str) -> None:
    """
    Changes the current
    auto connection.

    Parameters
    ----------
    name: str
        Name of the new
        auto connection.

    Examples
    --------
    Create a new connection:

    .. code-block:: python

        from verticapy.connection import new_connection, change_auto_connection

        new_connection(
            {
                "host": "10.211.55.14",
                "port": "5433",
                "database": "testdb",
                "password": "XxX",
                "user": "dbadmin",
            },
            name = "my_auto_connection",
            auto = False,
        )

    Change the auto connection
    to "my_auto_connection":

    .. code-block:: python

        change_auto_connection("my_auto_connection")

    .. seealso::

        | :py:func:`~verticapy.connection.new_connection` :
            Creates a new VerticaPy connection.
    """
    gb_conn = get_global_connection()

    confparser = get_confparser()

    if confparser.has_section(name):
        confparser.remove_section(gb_conn.vpy_auto_connection)
        confparser.add_section(gb_conn.vpy_auto_connection)
        confparser.set(gb_conn.vpy_auto_connection, "name", name)
        path = get_connection_file()

        with open(path, "w+", encoding="utf-8") as f:
            confparser.write(f)

    else:
        raise NameError(
            "The input name is incorrect. The connection "
            f"'{name}' has never been created.\nUse the "
            "new_connection function to create a new "
            "connection."
        )


def delete_connection(name: str) -> bool:
    """
    Deletes a specified connection
    from the connection file.

    Parameters
    ----------
    name: str
        Name of the connection.

    Returns
    -------
    bool
        ``True`` if the connection
        was deleted, ``False``
        otherwise.

    Examples
    --------
    Create a connection named
    'My_New_Vertica_Connection':

    .. code-block:: python

        from verticapy.connection import new_connection

        new_connection(
            {
                "host": "10.20.110.10",
                "port": "5433",
                "database": "vertica_eon",
                "password": "vertica",
                "user": "dbadmin",
            },
            name = "My_New_Vertica_Connection",
        )

    Display all available
    connections:

    .. code-block:: python

        from verticapy.connection import available_connections

        available_connections()

    ``['VerticaDSN', 'My_New_Vertica_Connection']``

    Delete the 'My_New_Vertica_Connection'
    connection:

    .. code-block:: python

        from verticapy.connection import delete_connection

        delete_connection("My_New_Vertica_Connection")

    Confirm that the connection
    no longer appears in the
    available connections:

    .. code-block:: python

        available_connections()

    ``['VerticaDSN']``

    .. seealso::

        | :py:func:`~verticapy.connection.new_connection` :
            Creates a new VerticaPy connection.
    """
    gb_conn = get_global_connection()

    confparser = get_confparser()

    if confparser.has_section(name):
        confparser.remove_section(name)
        if confparser.has_section(gb_conn.vpy_auto_connection):
            name_auto = confparser.get(gb_conn.vpy_auto_connection, "name")
            if name_auto == name:
                confparser.remove_section(gb_conn.vpy_auto_connection)
        path = get_connection_file()

        with open(path, "w+", encoding="utf-8") as f:
            confparser.write(f)

        return True

    else:
        warnings.warn(f"The connection {name} does not exist.", Warning)

        return False


def new_connection(
    conn_info: dict,
    name: str = "vertica_connection",
    auto: bool = True,
    overwrite: bool = True,
    connect_attempt: bool = True,
    prompt: bool = False,
) -> None:
    """
    Saves the new connection in the VerticaPy
    connection file. The information is saved
    as plaintext in the local machine.
    The function
    :py:func:`~verticapy.connection.get_connection_file`
    returns the associated connection file
    path. If you want a temporary connection,
    you can use the
    :py:func:`~verticapy.connection.set_connection`
    function.

    Parameters
    ----------
    conn_info: dict
        ``dictionnary`` containing
        the information to set up
        the connection.

         - database:
            Database Name.
         - host:
            Server ID.
         - password:
            User Password.
         - port:
            Database Port (optional, default: 5433).
         - user:
            User ID (optional, default: dbadmin).

        ...

         - env:
            ``bool`` to indicate whether the user and
            password are replaced by the associated
            environment variables. If ``True``, VerticaPy
            reads the associated environment variables
            instead of writing and directly using the
            username and password.
            For example:
            ``{'user': 'ENV_USER', 'password': 'ENV_PASSWORD'}``

            This works only for the user and password.
            The real values of the other variables are
            stored plaintext in the VerticaPy connection
            file. Using the environment variables hides
            the username and password in cases where the
            local machine is shared.

    name: str, optional
        Name of the connection.
    auto: bool, optional
        If set to True, the connection
        will become the new auto-connection.
    overwrite: bool, optional
        If set to ``True`` and the connection
        already exists, the existing connection
        will be overwritten.
    connect_attempt: bool
        If set to False, it will not attempt
        to connect automatically.
    prompt: bool, optional
        If set to True, it will open a prompt
        tp ask for ``oauth_refresh_token`` as well as ``client_secret``.

    Examples
    --------
    Create a new connection to VerticaPy:

    .. note::

        If no errors are raised, the new connection was
        successful.

    .. code-block:: python

        from verticapy.connection import new_connection

        conn_info = {
            "host": "10.211.55.14",
            "port": "5433",
            "database": "testdb",
            "password": "XxX",
            "user": "dbadmin",
        }

        new_connection(conn_info, name = "VerticaDSN")

    .. seealso::

        | :py:func:`~verticapy.connection.get_connection_file` :
            Gets the VerticaPy connection file.
        | :py:func:`~verticapy.connection.set_connection` :
            Sets the VerticaPy connection.
    """
    doPrintInfo = conf.get_option("print_info")

    def _printInfo(info):
        if doPrintInfo:
            print(info)

    path = get_connection_file()
    confparser = get_confparser()

    if confparser.has_section(name):
        if not overwrite:
            raise ValueError(
                f"The section '{name}' already exists. You "
                "can overwrite it by setting the parameter "
                "'overwrite' to True."
            )
        confparser.remove_section(name)
    confparser.add_section(name)

    if prompt:
        if not (oauth_access_token := getpass("Input OAuth Access Token:")):
            _printInfo("Default value applied: Input left empty.")
        else:
            conn_info["oauth_access_token"] = oauth_access_token
        if not (oath_refresh_token := getpass("Input OAuth Refresh Token:")):
            _printInfo("Default value applied: Input left empty.")
        else:
            conn_info["oauth_refresh_token"] = oath_refresh_token
        if not (
            client_secret := getpass(
                "Input OAuth Client Secret: (OTCAD and public client users should leave this blank)"
            )
        ):
            _printInfo("Default value applied: Input left empty.")
        else:
            conn_info["oauth_config"]["client_secret"] = client_secret

    try:
        if conn_info.get("oauth_refresh_token", False):
            oauth_manager = OAuthManager(conn_info["oauth_refresh_token"])
            oauth_manager.set_config(conn_info["oauth_config"])
            conn_info["oauth_access_token"] = oauth_manager.do_token_refresh()
            conn_info["oauth_refresh_token"] = oauth_manager.refresh_token
    except OAuthTokenRefreshError as error:
        print("An error occured while refreshing your OAuth token")
        raise error

    for c in conn_info:
        confparser.set(name, c, str(conn_info[c]))

    with open(path, "w+", encoding="utf-8") as f:
        confparser.write(f)

    if auto:
        change_auto_connection(name)

    if connect_attempt:
        # To prevent auto-connection. Needed for re-prompts in case of errors.
        gb_conn = get_global_connection()
        try:
            gb_conn.set_connection(
                vertica_python.connect(**read_dsn(name, path)), name, path
            )
        except OAuthTokenRefreshError as e:
            print(
                "Access Denied: Your authentication credentials are incorrect or have expired. Please retry"
            )
            new_connection(
                conn_info=read_dsn(name, path), prompt=True, connect_attempt=False
            )
            try:
                gb_conn.set_connection(
                    vertica_python.connect(**read_dsn(name, path)), name, path
                )
                if doPrintInfo:
                    print("Connected Successfully!")
            except OAuthTokenRefreshError as error:
                print("Error persists:")
                raise error
        except ConnectionError as error:
            print(
                "A connection error occured. Common reasons may be an invalid host, port, or, if requiring "
                "OAuth and token refresh, this may be due to an incorrect or malformed token url."
            )
            raise error


new_auto_connection = new_connection
