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
from verticapy.core.str_sql.base import str_sql


def random():
    """
Returns a Random Number.

Returns
-------
str_sql
    SQL expression.
    """
    return str_sql("RANDOM()", "float")


def randomint(n: int):
    """
Returns a Random Number from 0 through n – 1.

Parameters
----------
n: int
    Integer Value.

Returns
-------
str_sql
    SQL expression.
    """
    return str_sql(f"RANDOMINT({n})", "int")


def seeded_random(random_state: int):
    """
Returns a Seeded Random Number using the input random state.

Parameters
----------
random_state: int
    Integer used to seed the randomness.

Returns
-------
str_sql
    SQL expression.
    """
    return str_sql(f"SEEDED_RANDOM({random_state})", "float")
