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
from typing import Optional

import verticapy._config.config as conf


def _current_random(rand_int: Optional[int] = None) -> str:
    """
    Returns the 'random' function to be used in the
    query. The returned function depends on the input
    parameter 'rand_int' and whether the random state has
    been changed.
    """
    random_state = conf.get_option("random_state")
    if isinstance(rand_int, int):
        if isinstance(random_state, int):
            random_func = f"FLOOR({rand_int} * SEEDED_RANDOM({random_state}))"
        else:
            random_func = f"RANDOMINT({rand_int})"
    else:
        if isinstance(random_state, int):
            random_func = f"SEEDED_RANDOM({random_state})"
        else:
            random_func = "RANDOM()"
    return random_func
