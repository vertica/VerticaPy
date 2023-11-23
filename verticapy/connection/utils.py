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
from configparser import ConfigParser
from typing import Optional


def get_confparser(dsn: Optional[str] = None) -> ConfigParser:
    """
    Parses the input DSN and returns the linked
    Config Parser.

    Parameters
    ----------
    dsn: str, optional
        Path to the file containing the credentials.
        If empty, the VERTICAPY_CONNECTION environment
        variable is used.

    Examples
    --------
    Get the Config Parser using the VERTICAPY_CONNECTION environment variable:

    .. ipython:: python

        from verticapy.connection import get_confparser

        get_confparser()
    """
    if not dsn:
        dsn = get_connection_file()
    confparser = ConfigParser()
    confparser.optionxform = str
    confparser.read(dsn)
    return confparser


def get_connection_file() -> str:
    """
    Gets (and creates, if necessary) the auto
    -connection file. If the environment variable
    'VERTICAPY_CONNECTION' is set, it is assumed
    to be the full path to the auto-connection file.
    Otherwise, we reference "connections.verticapy"
    in the hidden ".verticapy" folder in the user's
    home directory.

    Returns
    -------
    string
        the full path to the auto-connection file.

    Example
    -------
    The connection file is used to store all connections:

    .. warning:: Connections are stored in plain text format. In shared environments, it's recommended to create connections manually for added security.

    .. code-block:: python

        from verticapy.connection import get_connection_file

        get_connection_file()

    ``'C:\\Users\\user\\.vertica\\connections.verticapy'``
    """
    if "VERTICAPY_CONNECTION" in os.environ:
        return os.environ["VERTICAPY_CONNECTION"]
    path = os.path.join(os.path.expanduser("~"), ".vertica")
    os.makedirs(path, 0o700, exist_ok=True)
    path = os.path.join(path, "connections.verticapy")
    return path
