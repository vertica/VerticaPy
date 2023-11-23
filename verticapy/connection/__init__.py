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
from verticapy.connection.external import set_external_connection
from verticapy.connection.connect import (
    auto_connect,
    close_connection,
    connect,
    current_connection,
    current_cursor,
    set_connection,
    vertica_connection,
    verticapylab_connection,
)
from verticapy.connection.write import (
    change_auto_connection,
    delete_connection,
    new_connection,
)
from verticapy.connection.read import available_connections, read_dsn
from verticapy.connection.utils import get_connection_file, get_confparser
