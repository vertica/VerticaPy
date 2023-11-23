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
from verticapy.datasets.generators import gen_dataset, gen_meshgrid
from verticapy.datasets.loaders import (
    load_airline_passengers,
    load_amazon,
    load_cities,
    load_commodities,
    load_gapminder,
    load_iris,
    load_laliga,
    load_market,
    load_pop_growth,
    load_smart_meters,
    load_titanic,
    load_winequality,
    load_world,
    load_africa_education,
)
from verticapy.datasets.tests_loaders import (
    load_dataset_cl,
    load_dataset_reg,
    load_dataset_num,
)
