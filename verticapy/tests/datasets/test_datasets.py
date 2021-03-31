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
import datetime

set_option("print_info", False)

class TestDatasets:
    def test_gen_dataset(self, base):
        result = gen_dataset(features_ranges = {"name": {"type": str, "values": ["Badr", "Badr", "Raghu", "Waqas",]},
                                                "age": {"type": int, "range": [20, 40]},
                                                "distance": {"type": float, "range": [1000, 4000]},
                                                "date": {"type": datetime.date, "range": ["1993-11-03", 365]},
                                                "datetime": {"type": datetime.datetime, "range": ["1993-11-03", 365]},}, 
                             cursor = base.cursor)
        assert result["name"].mode() == 'Badr'
        assert result["date"].max() < datetime.date(1995, 11, 2)
        assert result["datetime"].max() < datetime.datetime(1995, 11, 2)
        assert result["distance"].max() < 4001
        assert result["age"].max() < 41

    def test_gen_meshgrid(self, base):
        result = gen_meshgrid(features_ranges = {"name": {"type": str, "values": ["Badr", "Badr", "Raghu", "Waqas",]},
                                                 "age": {"type": int, "range": [20, 40], "nbins": 10,},
                                                 "distance": {"type": float, "range": [1000, 4000], "nbins": 10,},
                                                 "date": {"type": datetime.date, "range": ["1993-11-03", 365], "nbins": 10,},
                                                 "datetime": {"type": datetime.datetime, "range": ["1993-11-03", 365], "nbins": 10,},}, 
                                     cursor = base.cursor)
        assert result.shape() == (58564, 5)
        assert result["date"].max() == datetime.date(1994, 11, 3)
        assert result["datetime"].max() == datetime.datetime(1994, 11, 3, 0, 0)
        assert result["distance"].max() == 4000.0
        assert result["age"].max() == 40.0

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