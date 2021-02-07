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
from verticapy import drop, set_option
from verticapy.geo import *

set_option("print_info", False)


@pytest.fixture(scope="module")
def cities_vd(base):
    from verticapy.datasets import load_cities

    cities = load_cities(cursor=base.cursor)
    yield cities
    with warnings.catch_warnings(record=True) as w:
        drop(
            name="public.cities", cursor=base.cursor,
        )


@pytest.fixture(scope="module")
def world_vd(base):
    from verticapy.datasets import load_world

    cities = load_world(cursor=base.cursor)
    yield cities
    with warnings.catch_warnings(record=True) as w:
        drop(
            name="public.world", cursor=base.cursor,
        )


class TestUtilities:
    def test_read_shp(self, cities_vd):
        with warnings.catch_warnings(record=True) as w:
            drop(
                name="public.cities_test",
                cursor=cities_vd._VERTICAPY_VARIABLES_["cursor"],
            )
        cities_vd.to_shp("cities_test", "/home/dbadmin/", shape="Point")
        vdf = read_shp(
            "/home/dbadmin/cities_test.shp", cities_vd._VERTICAPY_VARIABLES_["cursor"]
        )
        assert vdf.shape() == (202, 3)
        try:
            os.remove("/home/dbadmin/cities_test.shp")
            os.remove("/home/dbadmin/cities_test.shx")
            os.remove("/home/dbadmin/cities_test.dbf")
        except:
            pass
        with warnings.catch_warnings(record=True) as w:
            drop(
                name="public.cities_test",
                cursor=cities_vd._VERTICAPY_VARIABLES_["cursor"],
            )
