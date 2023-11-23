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

# Pytest
import pytest

# Standard Python Modules
import datetime

# VerticaPy
from verticapy import set_option
from verticapy.datasets import *

set_option("print_info", False)


class TestDatasets:
    def test_gen_dataset(self):
        result = gen_dataset(
            features_ranges={
                "name": {"type": str, "values": ["Badr", "Badr", "Raghu", "Waqas"]},
                "age": {"type": int, "range": [20, 40]},
                "distance": {"type": float, "range": [1000, 4000]},
                "date": {"type": datetime.date, "range": ["1993-11-03", 365]},
                "datetime": {
                    "type": datetime.datetime,
                    "range": ["1993-11-03", 365],
                },
            }
        )
        assert result["name"].mode() == "Badr"
        assert result["date"].max() < datetime.date(1995, 11, 2)
        assert result["datetime"].max() < datetime.datetime(1995, 11, 2)
        assert result["distance"].max() < 4001
        assert result["age"].max() < 41

    def test_gen_meshgrid(self):
        result = gen_meshgrid(
            features_ranges={
                "name": {"type": str, "values": ["Badr", "Badr", "Raghu", "Waqas"]},
                "age": {"type": int, "range": [20, 40], "nbins": 10},
                "distance": {"type": float, "range": [1000, 4000], "nbins": 10},
                "date": {
                    "type": datetime.date,
                    "range": ["1993-11-03", 365],
                    "nbins": 10,
                },
                "datetime": {
                    "type": datetime.datetime,
                    "range": ["1993-11-03", 365],
                    "nbins": 10,
                },
            }
        )
        assert result.shape() == (58564, 5)
        assert result["date"].max() == datetime.date(1994, 11, 3)
        assert result["datetime"].max() == datetime.datetime(1994, 11, 3, 0, 0)
        assert result["distance"].max() == 4000.0
        assert result["age"].max() == 40.0

    def test_load_airline_passengers(self):
        result = load_airline_passengers()
        assert result.shape() == (144, 2)

    def test_load_amazon(self):
        result = load_amazon()
        assert result.shape() == (6454, 3)

    def test_load_cities(self):
        result = load_cities()
        assert result.shape() == (202, 2)

    def test_load_commodities(self):
        result = load_commodities()
        assert result.shape() == (416, 7)

    def test_load_iris(self):
        result = load_iris()
        assert result.shape() == (150, 5)

    def test_load_market(self):
        result = load_market()
        assert result.shape() == (314, 3)

    def test_load_smart_meters(self):
        result = load_smart_meters()
        assert result.shape() == (11844, 3)

    def test_load_titanic(self):
        result = load_titanic()
        assert result.shape() == (1234, 14)

    def test_load_winequality(self):
        result = load_winequality()
        assert result.shape() == (6497, 14)

    def test_load_world(self):
        result = load_world()
        assert result.shape() == (177, 4)
