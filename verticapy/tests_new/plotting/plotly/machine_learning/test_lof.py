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
from verticapy.learn.neighbors import LocalOutlierFactor
from verticapy.tests_new.plotting.conftest import (
    get_xaxis_label,
    get_yaxis_label,
    get_width,
    get_height,
)

# Testing variables
COL_NAME_1 = "X"
COL_NAME_2 = "Y"
COL_NAME_3 = "Z"


class TestPlotlyMachineLearningLOFPlot2D:
    """
    Testing different attributes of 2D LOF plot
    """

    @pytest.fixture(scope="class")
    def plot_result(self, dummy_scatter_vd):
        """
        Create a LOF plot
        """
        model = LocalOutlierFactor("lof_test")
        model.fit(dummy_scatter_vd, [COL_NAME_1, COL_NAME_2])
        return model.plot()

    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        """
        Get the plot results
        """
        self.result = plot_result

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
        test_title = COL_NAME_1
        # Act
        # Assert
        assert (
            self.result.layout["xaxis"]["title"]["text"] == test_title
        ), "X axis label incorrect"

    def test_properties_yaxis_label(self):
        """
        Testing y-axis title
        """
        # Arrange
        test_title = COL_NAME_2
        # Act
        # Assert
        assert (
            self.result.layout["yaxis"]["title"]["text"] == test_title
        ), "Y axis label incorrect"

    def test_properties_scatter_and_line_plot(self):
        """
        Test outline and scatter
        """
        # Arrange
        total_items = 2
        # Act
        # Assert
        assert len(self.result.data) == total_items, "Either outline or scatter missing"

    def test_properties_hoverinfo_for_2d(self):
        """
        Test hover info
        """
        # Arrange
        x_val = "{x}"
        y_val = "{y}"
        # Act
        # Assert
        assert (
            x_val in self.result.data[1]["hovertemplate"]
            and y_val in self.result.data[1]["hovertemplate"]
        ), "Hover information does not contain x or y"

    def test_additional_options_custom_height(self, dummy_scatter_vd):
        """
        Test custom width and height
        """
        # rrange
        custom_height = 600
        custom_width = 700
        # Act
        model = LocalOutlierFactor("lof_test")
        model.fit(dummy_scatter_vd, [COL_NAME_1, COL_NAME_2])
        result = model.plot(height=custom_height, width=custom_width)
        # Assert
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"


class TestPlotlyMachineLearningLOFPlot3D:
    """
    Testing different attributes of 3D LOF plot
    """

    @pytest.fixture(scope="class")
    def plot_result_2(self, dummy_scatter_vd):
        """
        Create a 3D LOF plot
        """
        model = LocalOutlierFactor("lof_test_3d")
        model.fit(dummy_scatter_vd, [COL_NAME_1, COL_NAME_2, COL_NAME_3])
        return model.plot()

    @pytest.fixture(autouse=True)
    def result(self, plot_result_2):
        """
        Get the plot results
        """
        self.result = plot_result_2

    def test_properties_output_type_for_3d(self, plotting_library_object):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_xaxis_label_for_3d(self):
        """
        Testing x-axis label
        """
        # Arrange
        test_title = COL_NAME_1
        # Act
        # Assert
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_label_for_3d(self):
        """
        Testing y-axis label
        """
        # Arrange
        test_title = COL_NAME_2
        # Act
        # Assert
        assert get_yaxis_label(self.result) == test_title, "X axis label incorrect"
