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
from verticapy.datasets import *

set_option("print_info", False)

class TestDatasets:
    def test_load_airline_passengers(self, base):
        result = load_airline_passengers(cursor = base.cursor)
        assert result.shape() == (144, 2)

    def test_load_amazon(self, base):
        result = load_amazon(cursor = base.cursor)
        assert result.shape() == (6454, 3)

    def test_load_cities(self, base):
        result = load_cities(cursor = base.cursor)
        assert result.shape() == (202, 2)

    def test_load_commodities(self, base):
        result = load_commodities(cursor = base.cursor)
        assert result.shape() == (416, 7)

    def test_load_iris(self, base):
        result = load_iris(cursor = base.cursor)
        assert result.shape() == (150, 5)

    def test_load_market(self, base):
        result = load_market(cursor = base.cursor)
        assert result.shape() == (314, 3)

    def test_load_smart_meters(self, base):
        result = load_smart_meters(cursor = base.cursor)
        assert result.shape() == (11844, 3)

    def test_load_titanic(self, base):
        result = load_titanic(cursor = base.cursor)
        assert result.shape() == (1234, 14)

    def test_load_winequality(self, base):
        result = load_winequality(cursor = base.cursor)
        assert result.shape() == (6497, 14)

    def test_load_world(self, base):
        result = load_world(cursor = base.cursor)
        assert result.shape() == (177, 4)