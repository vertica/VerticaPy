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
import verticapy as vp
from verticapy.connection import *
from verticapy.connection.global_connection import get_global_connection


class TestConnect:
    def test_auto_connection(self, base):
        # test for read_dsn / new_connection / available_connections /
        #          auto_connect / change_auto_connection / delete_connection.
        # read_dsn
        gb_conn = get_global_connection()
        d = read_dsn(
            "vp_test_config",
            os.path.dirname(vp.__file__) + "/tests/verticaPy_test_tmp.conf",
        )
        assert int(d["port"]) > 0
        # new_auto_connection
        new_connection(d, "vp_test_config")
        # available_connections
        result = available_connections()
        assert "vp_test_config" in result
        # change_auto_connection
        change_auto_connection("vp_test_config")
        # auto_connect
        auto_connect()
        cur = gb_conn.get_connection().cursor()
        cur.execute("SELECT 1;")
        result2 = cur.fetchone()
        assert result2 == [1]
        # delete_connection
        assert delete_connection("vp_test_config")
        # connection label
        current_cursor().execute(
            "SELECT client_label FROM v_monitor.sessions WHERE client_label LIKE 'verticapy%' LIMIT 1;"
        )
        label = current_cursor().fetchone()[0].split("-")
        assert label[1] == vp.__version__.split("-")[0]
        # assert label[2] == str(gb_conn.vpy_session_identifier)

    def test_vertica_connection(self, base):
        cur = vertica_connection(
            "vp_test_config",
            os.path.dirname(vp.__file__) + "/tests/verticaPy_test_tmp.conf",
        ).cursor()
        cur.execute("SELECT 1;")
        result = cur.fetchone()
        assert result == [1]
