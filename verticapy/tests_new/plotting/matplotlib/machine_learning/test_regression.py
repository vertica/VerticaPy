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
col_name_1 = "X"
col_name_2 = "Y"


@pytest.fixture(scope="class")
def plot_result(dummy_scatter_vd):
    model = LinearRegression("LR_churn")
    model.fit(dummy_scatter_vd, [col_name_1], col_name_2)
    return model.plot()


class TestMachineLearningRegressionPlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result

    def test_properties_output_type(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_xaxis_label(self):
        # Arrange
        test_title = col_name_1
        # Act
        # Assert
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_label(self):
        # Arrange
        test_title = col_name_2
        # Act
        # Assert
        assert get_yaxis_label(self.result) == test_title, "Y axis label incorrect"

    def test_data_all_scatter_points(self, dummy_scatter_vd):
        # Arrange
        no_of_points = 100
        # Act
        # Assert
        assert len(self.result.collections[0].get_offsets()) == len(
            dummy_scatter_vd
        ), "Discrepancy between points plotted and total number ofp oints"

    def test_additional_options_custom_height(self, dummy_scatter_vd):
        # rrange
        custom_height = 650
        custom_width = 700
        # Act
        model = LinearRegression("LR_churn")
        model.fit(dummy_scatter_vd, ["X"], "Y")
        result = model.plot(height=custom_height, width=custom_width)
        # Assert
        assert (
            result.get_figure().get_size_inches()[0] == custom_width
            and result.get_figure().get_size_inches()[1] == custom_height
        ), "Custom width or height not working"
