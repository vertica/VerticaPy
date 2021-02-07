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

    world = load_world(cursor=base.cursor)
    yield world
    with warnings.catch_warnings(record=True) as w:
        drop(
            name="public.world", cursor=base.cursor,
        )


class TestGeo:
    def test_index_create_describe_rename_intersect(self, world_vd, cities_vd):
        world_copy = world_vd.copy()
        world_copy["id"] = "ROW_NUMBER() OVER (ORDER BY pop_est)"
        result = create_index(world_copy, "id", "geometry", "world_polygons", True)
        assert result["polygons"][0] == 177
        assert result["min_x"][0] == pytest.approx(-180.0)
        assert result["min_y"][0] == pytest.approx(-90.0)
        assert result["max_x"][0] == pytest.approx(180.0)
        assert result["max_y"][0] == pytest.approx(83.64513)
        rename_index(
            "world_polygons",
            "world_polygons_rename",
            world_vd._VERTICAPY_VARIABLES_["cursor"],
            True,
        )
        result2 = describe_index(
            "world_polygons_rename", world_vd._VERTICAPY_VARIABLES_["cursor"], True
        )
        assert result2.shape() == (177, 3)
        cities_copy = cities_vd.copy()
        cities_copy["id"] = "ROW_NUMBER() OVER (ORDER BY city)"
        result3 = intersect(cities_copy, "world_polygons_rename", "id", "geometry")
        assert result3.shape() == (172, 2)
        cities_copy["x"] = "ST_X(geometry)"
        cities_copy["y"] = "ST_Y(geometry)"
        result4 = intersect(cities_copy, "world_polygons_rename", "id", x="x", y="y")
        assert result4.shape() == (172, 2)
        drop(
            "world_polygons_rename",
            world_vd._VERTICAPY_VARIABLES_["cursor"],
            method="geo",
        )
