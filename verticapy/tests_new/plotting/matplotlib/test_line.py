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

# Vertica
from verticapy.tests_new.plotting.conftest import get_xaxis_label, get_yaxis_label

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


class TestMatplotlibLinePlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result

    @pytest.fixture(autouse=True)
    def result_2(self, plot_result_vDF):
        self.vdf_result = plot_result_vDF

    def test_properties_output_type(self, matplotlib_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, matplotlib_figure_object), "Wrong object created"

    def test_properties_output_type_for_vDataFrame(self, matplotlib_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(
            self.vdf_result, matplotlib_figure_object
        ), "Wrong object created"

    def test_properties_output_type_for_one_trace(
        self, dummy_line_data_vd, matplotlib_figure_object
    ):
        # Arrange
        # Act
        result = dummy_line_data_vd[dummy_line_data_vd[col_name_2] == cat_option][
            col_name_1
        ].plot(ts=time_col)
        # Assert - checking if correct object created
        assert isinstance(result, matplotlib_figure_object), "Wrong object created"

    def test_properties_x_axis_title(
        self,
    ):
        # Arrange
        test_tile = "date"
        # Act
        # Assert - checking if correct object created
        assert get_xaxis_label(self.result) == test_tile, "X axis title incorrect"

    def test_properties_y_axis_title(
        self,
    ):
        # Arrange
        test_tile = col_name_1
        # Act
        # Assert - checking if correct object created
        assert get_yaxis_label(self.result) == col_name_1, "Y axis title incorrect"

    def test_data_count_of_all_values(self, dummy_line_data_vd):
        # Arrange
        total_count = dummy_line_data_vd.shape()[0]
        # Act
        assert (
            sum(
                [
                    len(self.result.get_lines()[i].get_xdata())
                    for i in range(len(self.result.get_lines()))
                ]
            )
            == total_count
        ), "The total values in the plot are not equal to the values in the dataframe."

    def test_additional_options_custom_width_and_height(self, dummy_line_data_vd):
        # Arrange
        custom_width = 4
        custom_height = 6
        # Act
        result = dummy_line_data_vd[col_name_1].plot(
            ts=time_col, by=col_name_2, width=custom_width, height=custom_height
        )
        # Assert - checking if correct object created
        assert (
            result.get_figure().get_size_inches()[0] == custom_width
            and result.get_figure().get_size_inches()[1] == custom_height
        ), "Custom width or height not working"

    @pytest.mark.skip()
    @pytest.mark.parametrize("method", ["count", "density"])
    @pytest.mark.parametrize("max_cardinality", [3, 5])
    def test_properties_output_type_for_all_options(
        self, dummy_dist_vd, matplotlib_figure_object, max_cardinality, method
    ):
        # Arrange
        # Act
        result = dummy_line_data_vd[col_name_1].plot(
            ts=time_col, by=col_name_2, markers=True
        )
        # Assert - checking if correct object created
        assert isinstance(self.result, matplotlib_figure_object), "Wrong object created"
