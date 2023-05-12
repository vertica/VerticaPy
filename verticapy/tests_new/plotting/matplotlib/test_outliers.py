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


# Vertica
from verticapy.tests_new.plotting.conftest import get_xaxis_label, get_yaxis_label

# Testing variables
col_name_1 = "0"
col_name_2 = "1"


@pytest.fixture(scope="class")
def plot_result(dummy_dist_vd):
    return dummy_dist_vd.outliers_plot(columns=[col_name_1])


@pytest.fixture(scope="class")
def plot_result_2D(dummy_dist_vd):
    return dummy_dist_vd.outliers_plot(columns=[col_name_1, col_name_2])


@pytest.fixture(scope="class")
def plotting_library_object(matplotlib_figure_object):
    return matplotlib_figure_object


class TestMatplotlibOutliersPlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result

    def test_properties_output_type_for_1d(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_xaxis_for_1d(
        self,
    ):
        # Arrange
        test_title = col_name_1
        # Act
        # Assert -
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_data_all_scatter_points_for_1d(
        self,
        dummy_dist_vd,
    ):
        # Arrange
        total_points = len(dummy_dist_vd[col_name_1])
        # Act
        result = dummy_dist_vd.outliers_plot(columns=[col_name_1], max_nb_points=10000)
        plot_points_count = sum(
            [len(result.get_children()[i].get_offsets()) for i in range(10, 12)]
        )
        assert (
            plot_points_count == total_points
        ), "All points are not plotted for 1d plot"

    def test_additional_options_custom_width_and_height(self, dummy_dist_vd):
        # Arrange
        custom_width = 3
        custom_height = 3
        # Act
        result = dummy_dist_vd.outliers_plot(
            columns=[col_name_1, col_name_2], width=custom_width, height=custom_height
        )
        # Assert
        assert (
            result.get_figure().get_size_inches()[0] == custom_width
            and result.get_figure().get_size_inches()[1] == custom_height
        ), "Custom width or height not working"


class TestMatplotlibOutliersPlot2D:
    @pytest.fixture(autouse=True)
    def result(self, plot_result_2D):
        self.result = plot_result_2D

    def test_properties_output_type(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_xaxis_label(
        self,
    ):
        # Arrange
        test_title = col_name_1
        # Act
        # Assert -
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_label(
        self,
    ):
        # Arrange
        test_title = col_name_2
        # Act
        # Assert -
        assert get_yaxis_label(self.result) == test_title, "X axis label incorrect"
