"""
(c)  Copyright  [2018-2023]  OpenText  or one of its
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

# VerticaPy
import verticapy._config.config as conf

# Other Modules
import plotly


@pytest.fixture(scope="module")
def plotting_library_object():
    """
    Set default plotting object to highcharts
    """
    return plotly.graph_objs.Figure


@pytest.fixture(scope="session", autouse=True)
def load_plotlib():
    """
    Set default plotting library to highcharts
    """
    conf.set_option("plotting_lib", "plotly")
    yield
