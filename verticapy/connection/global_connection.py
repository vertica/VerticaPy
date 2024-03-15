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

import uuid
from typing import Literal, Optional
from vertica_python.vertica.connection import Connection

from verticapy import __version__

VERTICAPY_AUTO_CONNECTION: str = "VERTICAPY_AUTO_CONNECTION"
VERTICAPY_SESSION_IDENTIFIER: str = str(uuid.uuid1()).replace("-", "")
VERTICAPY_SESSION_LABEL: str = f"verticapy-{__version__}-{VERTICAPY_SESSION_IDENTIFIER}"


class GlobalConnection:
    """
    Main Class to store the
    Global Connection used
    by all VerticaPy objects.
    """

    # Properties.

    @property
    def special_symbols(self) -> list[str]:
        """
        List of special symbols that can be
        assigned to any external database.
        """
        return [
            "$",
            "€",
            "£",
            "%",
            "@",
            "&",
            "§",
            "%",
            "?",
            "!",
        ]

    @property
    def vpy_auto_connection(self) -> Literal[VERTICAPY_AUTO_CONNECTION]:
        return VERTICAPY_AUTO_CONNECTION

    @property
    def vpy_session_identifier(self) -> Literal[VERTICAPY_SESSION_IDENTIFIER]:
        return VERTICAPY_SESSION_IDENTIFIER

    @property
    def vpy_session_label(self) -> Literal[VERTICAPY_SESSION_LABEL]:
        return VERTICAPY_SESSION_LABEL

    # System Methods.

    def __init__(self) -> None:
        self._connection = {
            "conn": None,
            "section": None,
            "dsn": None,
        }
        self._external_connections = {}

    # Main Methods.

    def get_connection(self) -> Connection:
        """
        Returns the current connection.

        Examples
        --------
        The following code demonstrates
        the usage of the function.

        .. ipython:: python

            # Import the Global Connection.
            from verticapy.connection.global_connection import get_global_connection

            # Example
            get_global_connection().get_connection()

        .. note::

            These functions serve as utilities to
            construct others, simplifying the overall
            code.
        """
        return self._connection["conn"]

    def get_external_connections(self) -> dict:
        """
        Returns the external connection.

        Examples
        --------
        The following code demonstrates
        the usage of the function.

        .. ipython:: python

            # Import the Global Connection.
            from verticapy.connection.global_connection import get_global_connection

            # Example
            get_global_connection().get_external_connections()

        .. note::

            These functions serve as utilities to
            construct others, simplifying the overall
            code.
        """
        return self._external_connections

    def get_dsn(self) -> str:
        """
        Returns the current dsn.

        Examples
        --------
        The following code demonstrates
        the usage of the function.

        .. ipython:: python

            # Import the Global Connection.
            from verticapy.connection.global_connection import get_global_connection

            # Example
            get_global_connection().get_dsn()

        .. note::

            These functions serve as utilities to
            construct others, simplifying the overall
            code.
        """
        return self._connection["dsn"]

    def get_dsn_section(self) -> str:
        """
        Returns the current dsn section.

        Examples
        --------
        The following code demonstrates
        the usage of the function.

        .. ipython:: python

            # Import the Global Connection.
            from verticapy.connection.global_connection import get_global_connection

            # Example
            get_global_connection().get_dsn_section()

        .. note::

            These functions serve as utilities to
            construct others, simplifying the overall
            code.
        """
        return self._connection["section"]

    def set_connection(
        self,
        conn: Connection,
        section: Optional[str] = None,
        dsn: Optional[str] = None,
    ) -> None:
        """
        Returns the current dsn section.

        Examples
        --------
        The following code demonstrates
        the usage of the function.

        .. code-block:: python

            # Import the Global Connection.
            from verticapy.connection.global_connection import get_global_connection

            # Import the Vertica connection function
            from verticapy.connection import vertica_connection

            # Building a connection
            conn = vertica_connection("VerticaDSN")

            # Example
            get_global_connection().set_connection(conn)

        .. note::

            These functions serve as utilities to
            construct others, simplifying the overall
            code.
        """
        self._connection["conn"] = conn
        self._connection["section"] = section
        self._connection["dsn"] = dsn

    def set_external_connections(self, symbol: str, cid: str, rowset: int) -> None:
        """
        Returns the current dsn section.

        Examples
        --------
        The following code demonstrates
        the usage of the function.

        .. code-block:: python

            # Import the Global Connection.
            from verticapy.connection.global_connection import get_global_connection

            # Example
            get_global_connection().set_external_connections(
                symbol = '$',
                cid = 'pgrs_cid',
                rowset = 500,
            )

        .. note::

            These functions serve as utilities to
            construct others, simplifying the overall
            code.
        """
        if (
            isinstance(cid, str)
            and isinstance(rowset, int)
            and symbol in self.special_symbols
        ):
            self._external_connections[symbol] = {
                "cid": cid,
                "rowset": rowset,
            }

        else:
            raise ValueError(
                "Could not set the external connection. Found a wrong type."
            )


_global_connection: GlobalConnection = GlobalConnection()


def get_global_connection() -> GlobalConnection:
    """
    Returns the Global connection.

    Examples
    --------
    The following code demonstrates
    the usage of the function.

    .. ipython:: python

        # Import the function.
        from verticapy.connection.global_connection import get_global_connection

        # Example
        get_global_connection()

    .. note::

        These functions serve as utilities to
        construct others, simplifying the overall
        code.
    """
    return _global_connection
