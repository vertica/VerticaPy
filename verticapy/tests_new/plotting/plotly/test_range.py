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
time_col = "date"
col_name_1 = "value"


@pytest.fixture(scope="class")
def plot_result(dummy_date_vd):
    return dummy_date_vd[col_name_1].range_plot(ts=time_col, plot_median=True)


class TestVDFRangeCurve:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result

    def test_properties_output_type(self, plotly_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result) == plotly_figure_object, "wrong object created"

    def test_properties_xaxis(
        self,
        load_plotly,
    ):
        # Arrange
        column_name = time_col
        # Act
        # Assert -
        assert (
            self.result.layout["xaxis"]["title"]["text"] == column_name
        ), "X axis label incorrect"

    def test_properties_xaxis(
        self,
    ):
        # Arrange
        column_name = col_name_1
        # Act
        # Assert -
        assert (
            self.result.layout["yaxis"]["title"]["text"] == column_name
        ), "Y axis label incorrect"

    def test_data_x_axis(self, dummy_date_vd):
        # Arrange
        test_set = set(dummy_date_vd.to_numpy()[:, 0])
        # Act
        result = dummy_date_vd[col_name_1].range_plot(ts=time_col)
        assert set(result.data[0]["x"]).issubset(
            test_set
        ), "There is descripancy between x axis values for the bounds"

    def test_data_x_axis_for_median(self, dummy_date_vd):
        # Arrange
        test_set = set(dummy_date_vd.to_numpy()[:, 0])
        # Act
        assert set(self.result.data[1]["x"]).issubset(
            test_set
        ), "There is descripancy between x axis values for the median"

    def test_additional_options_turn_off_median(self, load_plotly, dummy_date_vd):
        # Arrange
        # Act
        result = dummy_date_vd[col_name_1].range_plot(ts=time_col, plot_median=False)
        # Assert
        assert (
            len(result.data) == 1
        ), "Median is still showing even after it is turned off"

    def test_additional_options_turn_on_median(self, load_plotly, dummy_date_vd):
        # Arrange
        # Act
        # Assert
        assert (
            len(self.result.data) > 1
        ), "Median is still showing even after it is turned off"

    def test_additional_options_custom_width_and_height(
        self, load_plotly, dummy_date_vd
    ):
        # Arrange
        custom_width = 700
        custom_height = 700
        # Act
        result = dummy_date_vd[col_name_1].range_plot(
            ts=time_col, width=custom_width, height=custom_height
        )
        # Assert
        assert (
            result.layout["width"] == custom_width
            and result.layout["height"] == custom_height
        ), "Custom width or height not working"
