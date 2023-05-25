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


# Other Modules


# Verticapy
from verticapy.learn.tree import DecisionTreeRegressor
from verticapy.tests_new.plotting.conftest import (
    get_xaxis_label,
    get_yaxis_label,
)

# Testing variables
COL_NAME_1 = "0"
COL_NAME_2 = "1"


class TestPlotlyMachineLearningRegressionTreePlot:
    """
    Testing different attributes of Regression Tree plot
    """

    def __init__(self):
        self.x_col = None
        self.y_col = None

    @pytest.fixture(scope="class")
    def plot_result(self, dummy_dist_vd):
        """
        Create a Regression Tree plot
        """
        model = DecisionTreeRegressor(name="model_titanic")
        x_col = COL_NAME_1
        y_col = COL_NAME_2
        model.fit(dummy_dist_vd, x_col, y_col)
        return model.plot(), x_col, y_col

    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        """
        Get the plot results
        """
        self.result, self.x_col, self.y_col = plot_result

    def test_properties_output_type(self, plotting_library_object):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_xaxis_label(self):
        """
        Testing x-axis label
        """
        # Arrange
        test_title = self.x_col
        # Act
        # Assert
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_label(self):
        """
        Testing y-axis title
        """
        # Arrange
        test_title = self.y_col
        # Act
        # Assert
        assert get_yaxis_label(self.result) == test_title, "Y axis label incorrect"

    def test_properties_observations_label(self):
        """
        Test plot title
        """
        # Arrange
        test_title = "Observations"
        # Act
        # Assert
        assert self.result.data[0]["name"] == test_title, "X axis label incorrect"

    def test_properties_prediction_label(self):
        """
        Test plot title
        """
        # Arrange
        test_title = "Prediction"
        # Act
        # Assert
        assert self.result.data[1]["name"] == test_title, "Y axis label incorrect"

    def test_properties_hover_label(self):
        """
        Test hover labels
        """
        # Arrange
        test_title = f"{self.x_col}: %" "{x} <br>" f"{self.y_col}: %" "{y} <br>"
        # Act
        # Assert
        assert (
            self.result.data[0]["hovertemplate"] == test_title
        ), "Hover information incorrect"

    def test_properties_no_of_elements(self):
        """
        Test number of elements
        """
        # Arrange
        total_items = 2
        # Act
        # Assert
        assert len(self.result.data) == pytest.approx(
            total_items, abs=1
        ), "Some elements missing"

    def test_additional_options_custom_height(self, dummy_dist_vd):
        """
        Test custom width and height
        """
        # rrange
        custom_height = 650
        custom_width = 700
        model = DecisionTreeRegressor(name="model_titanic")
        model.fit(dummy_dist_vd, COL_NAME_1, COL_NAME_2)
        # Act
        result = model.plot(
            height=custom_height,
            width=custom_width,
        )
        # Assert
        assert (
            result.layout["height"] == custom_height
            and result.layout["width"] == custom_width
        ), "Custom height and width not working"
