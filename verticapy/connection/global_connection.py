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
import uuid
from typing import Literal, Optional

from verticapy import __version__

VERTICAPY_AUTO_CONNECTION = "VERTICAPY_AUTO_CONNECTION"
VERTICAPY_SESSION_IDENTIFIER = str(uuid.uuid1()).replace("-", "")
VERTICAPY_SESSION_LABEL = f"verticapy-{__version__}-{VERTICAPY_SESSION_IDENTIFIER}"


class GlobalConnection:
    @property
    def _special_symbols(self) -> list[str]:
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
    def _vpy_auto_connection(self) -> Literal[VERTICAPY_AUTO_CONNECTION]:
        return VERTICAPY_AUTO_CONNECTION

    @property
    def _vpy_session_identifier(self) -> Literal[VERTICAPY_SESSION_IDENTIFIER]:
        return VERTICAPY_SESSION_IDENTIFIER

    @property
    def _vpy_session_label(self) -> Literal[VERTICAPY_SESSION_LABEL]:
        return VERTICAPY_SESSION_LABEL

    def __init__(self):
        self._connection = {
            "conn": None,
            "section": None,
            "dsn": None,
        }
        self._external_connections = {}

    def _get_connection(self):
        return self._connection["conn"]

    def _get_external_connections(self):
        return self._external_connections

    def _get_dsn(self):
        return self._connection["dsn"]

    def _get_dsn_section(self):
        return self._connection["section"]

    def _set_connection(
        self, conn, section: Optional[str] = None, dsn: Optional[str] = None,
    ):
        self._connection["conn"] = conn
        self._connection["section"] = section
        self._connection["dsn"] = dsn

    def _set_external_connections(self, symbol: str, cid: str, rowset: int):
        if (
            isinstance(cid, str)
            and isinstance(rowset, int)
            and symbol in self._special_symbols
        ):
            self._external_connections[symbol] = {
                "cid": cid,
                "rowset": rowset,
            }
        else:
            raise ParameterError(
                "Could not set the external connection. Found a wrong type."
            )


_global_connection = GlobalConnection()


def get_global_connection():
    return _global_connection
