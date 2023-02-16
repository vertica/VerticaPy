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
from verticapy.connect.external import (
    EXTERNAL_CONNECTION,
    SPECIAL_SYMBOLS,
    set_external_connection,
)
from verticapy.connect.connect import (
    VERTICAPY_AUTO_CONNECTION,
    SESSION_IDENTIFIER,
    SESSION_LABEL,
    CONNECTION,
    auto_connect,
    close_connection,
    connect,
    current_connection,
    current_cursor,
    set_connection,
    vertica_connection,
    verticalab_connection,
)
from verticapy.connect.write import (
    change_auto_connection,
    delete_connection,
    new_connection,
)
from verticapy.connect.read import available_connections, read_dsn
