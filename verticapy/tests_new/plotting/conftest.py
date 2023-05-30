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
# Standard Python Modules
import random
import tempfile
import string
from abc import abstractmethod

# Pytest
import pytest

# Other Modules
from scipy.special import erfinv
import numpy as np
import pandas as pd


# VerticaPy
import verticapy
from verticapy import drop
from verticapy.datasets import load_titanic, load_iris, load_amazon
from verticapy.learn.delphi import AutoML

# Other Modules
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from vertica_highcharts.highcharts.highcharts import Highchart

DUMMY_TEST_SIZE = 100


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


class BasicPlotTests:
    """
    Basic Tests for all plots
    """

    cols = []

    # @property
    @abstractmethod
    def create_plot(self):
        """
        Abstract method to create the plot
        """

    @property
    def result(self):
        """
        Create the plot
        """
        func, arg = self.create_plot()
        return func(**arg)

    def test_properties_output_type(self, plotting_library_object):
        """
        Test if correct object created
        """
        assert isinstance(self.result, plotting_library_object), "wrong object crated"

    def test_properties_xaxis_label(self):
        """
        Testing x-axis label
        """
        test_title = self.cols[0]
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_label(self):
        """
        Testing y-axis title
        """
        test_title = self.cols[1]
        assert get_yaxis_label(self.result) == test_title, "Y axis label incorrect"

    def test_properties_zaxis_label(self):
        """
        Testing y-axis title
        """
        if len(self.cols) > 2:
            test_title = self.cols[2]
            assert get_zaxis_label(self.result) == test_title, "Z axis label incorrect"
        else:
            pass

    def test_additional_options_custom_width_and_height(self):
        """
        Test custom width and height
        """
        func, arg = self.create_plot()
        custom_width = 300
        custom_height = 400
        result = func(width=custom_width, height=custom_height, **arg)
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"

