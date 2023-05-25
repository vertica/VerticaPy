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

# VerticaPy
from vertica_highcharts.highcharts.highcharts import Highchart
from verticapy import drop
from verticapy.learn.delphi import AutoML

# Other Modules
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def get_xaxis_label(obj):
    """
    Get x-axis label for given plotting object
    """
    if isinstance(obj, plt.Axes):
        return obj.get_xlabel()
    if isinstance(obj, go.Figure):
        return obj.layout["xaxis"]["title"]["text"]
    if isinstance(obj, Highchart):
        return obj.options["xAxis"].title.text
    return None


def get_yaxis_label(obj):
    """
    Get y-axis label for given plotting object
    """
    if isinstance(obj, plt.Axes):
        return obj.get_ylabel()
    if isinstance(obj, go.Figure):
        return obj.layout["yaxis"]["title"]["text"]
    if isinstance(obj, Highchart):
        return obj.options["yAxis"].title.text
    return None


def get_zaxis_label(obj):
    """
    Get z-axis label for given plotting object
    """
    if isinstance(obj, plt.Axes):
        return obj.get_zlabel()
    if isinstance(obj, go.Figure):
        return obj.layout["zaxis"]["title"]["text"]
    if isinstance(obj, Highchart):
        return obj.options["zAxis"].title.text
    return None


def get_width(obj):
    """
    Get width for given plotting object
    """
    if isinstance(obj, plt.Axes):
        return obj.get_figure().get_size_inches()[0]
    if isinstance(obj, go.Figure):
        return obj.layout["width"]
    if isinstance(obj, Highchart):
        return obj.options["chart"].width
    return None


def get_height(obj):
    """
    Get height for given plotting object
    """
    if isinstance(obj, plt.Axes):
        return obj.get_figure().get_size_inches()[1]
    if isinstance(obj, go.Figure):
        return obj.layout["height"]
    if isinstance(obj, Highchart):
        return obj.options["chart"].height
    return None


def get_title(obj):
    """
    Get title for given plotting object
    """
    if isinstance(obj, plt.Axes):
        return obj.get_title()
    if isinstance(obj, go.Figure):
        return obj.layout["title"]["text"]
    if isinstance(obj, Highchart):
        return obj.options["title"].text
    return None


# Expensive models
@pytest.fixture(name="champion_challenger_plot", scope="package")
def load_champion_Challenger_plot(schema_loader, dummy_dist_vd):
    COL_NAME_1 = "binary"
    COL_NAME_2 = "0"
    model = AutoML(f"{schema_loader}.model_automl", lmax=10, print_info=False)
    model.fit(
        dummy_dist_vd,
        [
            COL_NAME_1,
        ],
        COL_NAME_2,
    )
    yield model
    model.drop()
