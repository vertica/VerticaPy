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
import numpy as np

# Testing variables
col_name = "check 2"


@pytest.fixture(scope="class")
def plot_result(dummy_dist_vd):
    def func(a, b):
        return b

    return dummy_dist_vd.contour(["0", "binary"], func)


class TestVDFContourPlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result

    def test_properties_output_type(self, plotly_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result) == plotly_figure_object, "Wrong object created"

    def test_properties_x_axis_title(
        self,
    ):
        # Arrange
        # Arrange
        # Act
        # Assert
        assert (
            self.result.layout["xaxis"]["title"]["text"] == "0"
        ), "X axis title incorrect"

    def test_properties_y_axis_title(
        self,
    ):
        # Arrange
        # Act
        # Assert
        assert (
            self.result.layout["yaxis"]["title"]["text"] == "binary"
        ), "Y axis title incorrect"

    def test_data_count_xaxis_default_bins(
        self,
    ):
        # Arrange
        # Act
        # Assert
        assert self.result.data[0]["x"].shape[0] == 100, "The default bins are not 100."

    def test_data_count_xaxis_custom_bins(self, load_plotly, dummy_dist_vd):
        # Arrange
        custom_bins = 1000

        def func(a, b):
            return b

        # Act
        result = dummy_dist_vd.contour(
            columns=["0", "binary"], nbins=custom_bins, func=func
        )
        # Assert
        assert (
            result.data[0]["x"].shape[0] == custom_bins
        ), "The custom bins option is not working."

    def test_data_x_axis_range(self, dummy_dist_vd):
        # Arrange
        x_min = dummy_dist_vd["0"].min()
        x_max = dummy_dist_vd["0"].max()
        custom_bins = 1000
        # Act
        # Assert
        assert (
            self.result.data[0]["x"].min() == x_min
            and self.result.data[0]["x"].max() == x_max
        ), "The range in data is not consistent with plot"

    def test_additional_options_custom_width_and_height(
        self, load_plotly, dummy_dist_vd
    ):
        # Arrange
        custom_width = 700
        custom_height = 700

        def func(a, b):
            return b

        # Act
        result = dummy_dist_vd.contour(
            ["0", "binary"], func, width=custom_width, height=custom_height
        )
        # Assert
        assert (
            result.layout["width"] == custom_width
            and result.layout["height"] == custom_height
        ), "Custom width not working"
