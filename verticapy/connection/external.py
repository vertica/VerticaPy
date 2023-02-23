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
from typing import Literal

from verticapy._config.connection import _external_connections, SPECIAL_SYMBOLS


def set_external_connection(
    cid: str, rowset: int = 500, symbol: Literal[tuple(SPECIAL_SYMBOLS)] = "$"
):
    """
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
    global _external_connections
    if isinstance(cid, str) and isinstance(rowset, int) and symbol in SPECIAL_SYMBOLS:
        _external_connections[symbol] = {
            "cid": cid,
            "rowset": rowset,
        }
    else:
        raise ParameterError(
            "Could not set the external connection. Found a wrong type."
        )
