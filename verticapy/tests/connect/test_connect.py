# (c) Copyright [2018-2022] Micro Focus or one of its affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# VerticaPy
from verticapy.connect import *


class TestConnect:
    def test_auto_connection(self, base):
        # test for read_dsn / new_connection / available_connections /
        #          read_auto_connect / change_auto_connection / delete_connection.
        # read_dsn
        d = read_dsn(
            "vp_test_config",
            os.path.dirname(verticapy.__file__) + "/tests/verticaPy_test_tmp.conf",
        )
        assert int(d["port"]) > 0
        # new_auto_connection
        new_connection(d, "vp_test_config")
        # available_connections
        result = available_connections()
        assert "vp_test_config" in result
        # change_auto_connection
        change_auto_connection("vp_test_config")
        # read_auto_connect
        read_auto_connect()
        cur = verticapy.options["connection"]["conn"].cursor()
        cur.execute("SELECT 1;")
        result2 = cur.fetchone()
        assert result2 == [1]
        # delete_connection
        assert delete_connection("vp_test_config")

    def test_vertica_connection(self, base):
        cur = vertica_connection(
            "vp_test_config",
            os.path.dirname(verticapy.__file__) + "/tests/verticaPy_test_tmp.conf",
        ).cursor()
        cur.execute("SELECT 1;")
        result = cur.fetchone()
        assert result == [1]
