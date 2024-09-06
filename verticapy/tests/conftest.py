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

import vertica_python, pytest, os, verticapy
from .base import VerticaPyTestBase
from configparser import ConfigParser


def create_conn_file():
    base_test = VerticaPyTestBase()
    base_test.setUp()

    if not (
        os.path.isfile(
            os.path.dirname(verticapy.__file__) + "/tests/verticaPy_test_tmp.conf"
        )
    ):
        path = os.path.dirname(verticapy.__file__) + "/tests/verticaPy_test_tmp.conf"
        confparser = ConfigParser()
        confparser.optionxform = str
        confparser.read(path)
        if confparser.has_section("vp_test_config"):
            confparser.remove_section("vp_test_config")
        confparser.add_section("vp_test_config")
        for elem in base_test.test_config:
            if elem != "log_level":
                confparser.set("vp_test_config", elem, str(base_test.test_config[elem]))
        f = open(path, "w+", encoding="utf-8")
        confparser.write(f)
        f.close()


def delete_conn_file():
    os.remove(os.path.dirname(verticapy.__file__) + "/tests/verticaPy_test_tmp.conf")


def get_version():
    base_class = VerticaPyTestBase()
    base_class.setUp()
    create_conn_file()
    verticapy.connect(
        "vp_test_config",
        os.path.dirname(verticapy.__file__) + "/tests/verticaPy_test_tmp.conf",
    )
    result = verticapy.vertica_version()
    base_class.tearDown()
    try:
        delete_conn_file()
    except:
        pass
    return result


@pytest.fixture(scope="session")
def base():
    base_class = VerticaPyTestBase()
    base_class.setUp()
    create_conn_file()
    verticapy.connect(
        "vp_test_config",
        os.path.dirname(verticapy.__file__) + "/tests/verticaPy_test_tmp.conf",
    )
    yield base_class
    base_class.tearDown()
    try:
        delete_conn_file()
    except:
        pass
    verticapy.close_connection()
