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
import secrets
import string
from typing import Optional

from verticapy._utils._sql._format import quote_ident

from verticapy.connection.connect import current_cursor


def gen_name(L: list) -> str:
    """
    Generates a name using the input list.
    """
    return "_".join(
        [
            "".join(ch for ch in str(elem).lower() if ch.isalnum() or ch == "_")
            for elem in L
        ]
    )


def gen_col_name(n: int = 5) -> str:
    """
    Generate a name using n character
    """
    return "".join(secrets.choice(string.ascii_letters) for _ in range(n)).lower()


def gen_tmp_name(schema: Optional[str] = None, name: Optional[str] = None) -> str:
    """
    Generates a temporary name using the input schema
    and name.
    """
    current_cursor().execute("SELECT CURRENT_SESSION(), USERNAME();")
    current_session, username = current_cursor().fetchone()
    session_user = f"{current_session}_{username}"
    L = session_user.split("_")
    L[0] = "".join(filter(str.isalnum, L[0]))
    L[1] = "".join(filter(str.isalnum, L[1]))
    universal_unique_id = str(uuid.uuid1()).replace("-", "")
    name = f'"_verticapy_tmp_{name.lower()}_{L[0]}_{L[1]}_{universal_unique_id}_"'
    if schema:
        name = f"{quote_ident(schema)}.{name}"
    return name
