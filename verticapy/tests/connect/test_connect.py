# (c) Copyright [2018-2021] Micro Focus or one of its affiliates.
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

import pytest
from verticapy.connect import *


class TestConnect:
    def test_auto_connection(self, base):
        # test for read_dsn / new_auto_connection / available_auto_connection /
        #          read_auto_connect / change_auto_connection.
        # read_dsn
        d = read_dsn(
            "vp_test_config",
            os.path.dirname(verticapy.__file__) + "/tests/verticaPy_test_tmp.conf",
        )
        assert int(d["port"]) == 5433
        # new_auto_connection
        new_auto_connection(d, "VerticaDSN_test")
        # available_auto_connection
        result = available_auto_connection()
        assert "VerticaDSN_test" in result
        # change_auto_connection
        change_auto_connection("VerticaDSN_test")
        # read_auto_connect
        cur = read_auto_connect().cursor()
        cur.execute("SELECT 1;")
        result2 = cur.fetchone()
        assert result2 == [1]

    def test_vertica_conn(self, base):
        cur = vertica_conn(
            "vp_test_config",
            os.path.dirname(verticapy.__file__) + "/tests/verticaPy_test_tmp.conf",
        ).cursor()
        cur.execute("SELECT 1;")
        result = cur.fetchone()
        assert result == [1]
