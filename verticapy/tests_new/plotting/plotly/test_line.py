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
import random

# Standard Python Modules


# Other Modules
import numpy as np

# Testing variables
time_col = "date"
col_name_1 = "values"
col_name_2 = "category"
cat_option = "A"


@pytest.fixture(scope="class")
def plot_result(dummy_line_data_vd):
    return dummy_line_data_vd[col_name_1].plot(ts=time_col, by=col_name_2)


@pytest.fixture(scope="class")
def plot_result_vDF(dummy_line_data_vd):
    return dummy_line_data_vd[dummy_line_data_vd[col_name_2] == cat_option].plot(
        ts=time_col, columns=col_name_1
    )


class TestVDFLinePlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result

    @pytest.fixture(autouse=True)
    def result_2(self, plot_result):
        self.vdf_result = plot_result

    def test_properties_output_type(self, plotly_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result) == plotly_figure_object, "wrong object created"

    def test_properties_output_type_for_vDataFrame(self, plotly_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert (
            type(self.vdf_result) == plotly_figure_object
        ), "wrong object created for vDataFrame"

    def test_properties_output_type_for_one_trace(
        self, dummy_line_data_vd, plotly_figure_object
    ):
        # Arrange
        # Act
        result = dummy_line_data_vd[dummy_line_data_vd[col_name_2] == cat_option][
            col_name_1
        ].plot(ts=time_col)
        # Assert - checking if correct object created
        assert type(result) == plotly_figure_object, "wrong object created"

    def test_properties_x_axis_title(
        self,
    ):
        # Arrange
        test_tile = "time"
        # Act
        # Assert - checking if correct object created
        assert (
            self.result.layout["xaxis"]["title"]["text"] == test_tile
        ), "X axis title incorrect"

    def test_properties_y_axis_title(
        self,
    ):
        # Arrange
        test_tile = col_name_1
        # Act
        # Assert - checking if correct object created
        assert (
            self.result.layout["yaxis"]["title"]["text"] == col_name_1
        ), "Y axis title incorrect"

    def test_data_count_of_all_values(self, dummy_line_data_vd):
        # Arrange
        total_count = dummy_line_data_vd.shape()[0]
        # Act
        assert (
            self.result.data[0]["x"].shape[0] + self.result.data[1]["x"].shape[0]
            == total_count
        ), "The total values in the plot are not equal to the values in the dataframe."

    def test_data_spot_check(self, dummy_line_data_vd):
        # Arrange
        # Act
        assert (
            str(
                dummy_line_data_vd[time_col][random.randint(0, len(dummy_line_data_vd))]
            )
            in self.result.data[0]["x"]
            or str(
                dummy_line_data_vd[time_col][random.randint(0, len(dummy_line_data_vd))]
            )
            in self.result.data[0]["x"]
        ), "Two time values that exist in the data do not exist in the plot"

    def test_additional_options_custom_width_and_height(self, dummy_line_data_vd):
        # Arrange
        custom_width = 400
        custom_height = 600
        # Act
        result = dummy_line_data_vd[col_name_1].plot(
            ts=time_col, by=col_name_2, width=custom_width, height=custom_height
        )
        # Assert - checking if correct object created
        assert (
            result.layout["width"] == custom_width
            and result.layout["height"] == custom_height
        ), "Custom width not working"

    def test_additional_options_marker_on(self, dummy_line_data_vd):
        # Arrange
        # Act
        result = dummy_line_data_vd[col_name_1].plot(
            ts=time_col, by=col_name_2, markers=True
        )
        # Assert - checking if correct object created
        assert set(result.data[0]["mode"]) == set(
            "lines+markers"
        ), "Markers not turned on"
