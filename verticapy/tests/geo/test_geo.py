# (c) Copyright [2018-2020] Micro Focus or one of its affiliates.
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
from verticapy import drop_table, set_option
from verticapy.geo import *

set_option("print_info", False)


@pytest.fixture(scope="module")
def cities_vd(base):
    from verticapy.learn.datasets import load_cities

    cities = load_cities(cursor=base.cursor)
    yield cities
    with warnings.catch_warnings(record=True) as w:
        drop_table(
            name="public.cities", cursor=base.cursor,
        )


@pytest.fixture(scope="module")
def world_vd(base):
    from verticapy.learn.datasets import load_world

    world = load_world(cursor=base.cursor)
    yield world
    with warnings.catch_warnings(record=True) as w:
        drop_table(
            name="public.world", cursor=base.cursor,
        )


class TestGeo:

    def test_create_index(self, world_vd):
        result = create_index(world_vd, "pop_est", "geometry", "world_polygons", True)
        assert result["polygons"][0] == 177
        assert result["min_x"][0] == pytest.approx(-180.0)
        assert result["min_y"][0] == pytest.approx(-90.0)
        assert result["max_x"][0] == pytest.approx(180.0)
        assert result["max_y"][0] == pytest.approx(83.64513)
        