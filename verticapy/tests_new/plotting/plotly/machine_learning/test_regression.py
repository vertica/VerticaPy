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
from verticapy.learn.linear_model import LinearRegression
from verticapy.tests_new.plotting.conftest import (
    get_xaxis_label,
    get_yaxis_label,
)

# Testing variables
COL_NAME_1 = "X"
COL_NAME_2 = "Y"


class TestPlotlyMachineLearningRegressionPlot:
    """
    Testing different attributes of Regression plot
    """

    @pytest.fixture(scope="class")
    def plot_result(self, dummy_scatter_vd):
        """
        Create a regression plot
        """
        model = LinearRegression("LR_churn")
        model.fit(dummy_scatter_vd, [COL_NAME_1], COL_NAME_2)
        return model.plot()

    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        """
        Get the plot results
        """
        self.result = plot_result

    def test_properties_output_type_for(self, plotting_library_object):
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
        test_title = COL_NAME_1
        # Act
        # Assert
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_label(self):
        """
        Testing y-axis title
        """
        # Arrange
        test_title = COL_NAME_2
        # Act
        # Assert
        assert get_yaxis_label(self.result) == test_title, "Y axis label incorrect"

    def test_properties_scatter_and_line_plot(self):
        """
        Test two items exist
        """
        # Arrange
        total_items = 2
        # Act
        # Assert
        assert len(self.result.data) == total_items, "Either line or scatter missing"

    def test_data_all_scatter_points(self, dummy_scatter_vd):
        """
        Test all datapoints
        """
        # Arrange
        no_of_points = len(dummy_scatter_vd)
        # Act
        # Assert
        assert (
            len(self.result.data[0]["x"]) == no_of_points
        ), "Discrepancy between points plotted and total number ofp oints"

    def test_additional_options_custom_height(self, dummy_scatter_vd):
        """
        Test custom width and height
        """
        # rrange
        custom_height = 650
        custom_width = 700
        # Act
        model = LinearRegression("LR_churn")
        model.fit(dummy_scatter_vd, ["X"], "Y")
        result = model.plot(height=custom_height, width=custom_width)
        # Assert
        assert (
            result.layout["height"] == custom_height
            and result.layout["width"] == custom_width
        ), "Custom height and width not working"
