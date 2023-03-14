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

# Standard Python Modules
import datetime, os, sys

# Other Modules
import plotly.express as px
import plotly

# VerticaPy
import verticapy
import verticapy._config.config as conf
from verticapy import drop, set_option
from verticapy.datasets import (
    load_titanic,
    load_amazon,
    load_commodities,
    load_iris,
    load_world,
    load_pop_growth,
    load_gapminder,
)

set_option("print_info", False)

@pytest.fixture(scope="module")
def titanic_vd():
    titanic = load_titanic()
    yield titanic
    drop(name="public.titanic")

@pytest.fixture(scope="module")
def load_plotly():
    conf.set_option("plotting_lib","plotly")
    yield
    conf.set_option("plotting_lib","matplotlib")

class TestvDFPlotPlotly:
    def test_vDF_hist(self, titanic_vd,load_plotly):
        # for plotly
        ## 1D bar charts
        assert(conf.get_option("plotting_lib")=="plotly")
        survived_values=titanic_vd.to_pandas()["survived"]
        test_fig=px.bar(
            x=[0,1], 
            y=[survived_values[survived_values==0].count(),survived_values[survived_values==1].count()]
            )
        test_fig=test_fig.update_xaxes(type='category')
        result=titanic_vd["survived"].bar()
        assert(type(test_fig)==plotly.graph_objs._figure.Figure)
        assert(type(result)==plotly.graph_objs._figure.Figure)
        assert(test_fig.data[0]['y'][0]/test_fig.data[0]['y'][1]==result.data[0]['y'][0]/result.data[0]['y'][1])
        assert(test_fig.data[0]['x'][0]==result.data[0]['x'][0])
        assert(test_fig.layout['xaxis']['type']=='category')
        result=titanic_vd["survived"].bar(xaxis_title="Custom X Axis Title")
        assert(result.layout['xaxis']['title']['text']=='Custom X Axis Title')
        result=titanic_vd["survived"].bar(yaxis_title="Custom Y Axis Title")
        assert(result.layout['yaxis']['title']['text']=='Custom Y Axis Title')