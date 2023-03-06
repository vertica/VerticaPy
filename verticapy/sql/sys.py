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
from verticapy._utils._sql._sys import _executeSQL


def current_session() -> int:
    """
    Returns the current DB session.
    """
    res = _executeSQL(
        query="SELECT /*+LABEL(current_session)*/ CURRENT_SESSION();",
        method="fetchfirstelem",
        print_time_sql=False,
    )
    return int(res.split(":")[1], base=16)


def username() -> str:
    """
    Returns the current DB username.
    """
    return _executeSQL(
        query="SELECT /*+LABEL(username)*/ USERNAME();",
        method="fetchfirstelem",
        print_time_sql=False,
    )
