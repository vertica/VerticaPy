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
from verticapy.learn.model_selection import elbow

# Testing variables
COL_NAME_1 = "PetalLengthCm"
COL_NAME_2 = "PetalWidthCm"


class TestPlotlyMachineLearningElbowCurve:
    """
    Testing different attributes of Elbow Curve plot
    """

    @pytest.fixture(scope="class")
    def plot_result(self, iris_vd):
        """
        Create a elbow curve plot
        """
        return elbow(input_relation=iris_vd, X=[COL_NAME_1, COL_NAME_2])

    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        """
        Get the plot results
        """
        self.result = plot_result

    def test_properties_output_type_for(self, plotting_library_object):
        """
        Get the plot results
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
        test_title = "Number of Clusters"
        # Act
        # Assert - checking if correct object created
        assert (
            self.result.layout["xaxis"]["title"]["text"] == test_title
        ), "X axis label incorrect"

    def test_data_all_scatter_points(self):
        """
        Test if both line and markers are displayed
        """
        # Arrange
        mode = "markers+line"
        # Act
        # Assert - checking if correct object created
        assert set(self.result.data[0]["mode"]) == set(
            mode
        ), "Either lines or marker missing"

    @pytest.mark.slow
    @pytest.mark.notcritical
    def test_additional_options_custom_height(self, iris_vd):
        """
        Test custom width and height
        """
        # rrange
        custom_height = 650
        custom_width = 700
        # Act
        result = elbow(
            input_relation=iris_vd,
            X=[COL_NAME_1, COL_NAME_2],
            width=custom_width,
            height=custom_height,
        )
        # Assert - checking if correct object created
        assert (
            result.layout["height"] == custom_height
            and result.layout["width"] == custom_width
        ), "Custom height and width not working"
