# Pytest
import pytest
import random

# Standard Python Modules


# Other Modules
import numpy as np

# Testing variables
col_name_1 = "0"
col_name_2 = "1"


@pytest.fixture(scope="class")
def plot_result(dummy_dist_vd):
    return dummy_dist_vd.outliers_plot(columns=[col_name_1])


@pytest.fixture(scope="class")
def plot_result_2D(dummy_dist_vd):
    return dummy_dist_vd.outliers_plot(columns=[col_name_1, col_name_2])


class TestVDFOutliersPlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result

    @pytest.fixture(autouse=True)
    def result_2(self, plot_result_2D):
        self._2d_result = plot_result_2D

    def test_properties_output_type_for_1d(self, plotly_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result) == plotly_figure_object, "wrong object created"

    def test_properties_output_type_for_2d(self, plotly_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self._2d_result) == plotly_figure_object, "wrong object created"

    def test_properties_xaxis_for_1d(
        self,
    ):
        # Arrange
        column_name = col_name_1
        # Act
        # Assert -
        assert self.result.data[0]["x"][0] == column_name, "X axis label incorrect"

    def test_properties_xaxis_for_2d(
        self,
    ):
        # Arrange
        column_name = col_name_1
        # Act
        # Assert -
        assert (
            self._2d_result.layout["xaxis"]["title"]["text"] == column_name
        ), "X axis label incorrect"

    def test_properties_yaxis_for_2d(
        self,
    ):
        # Arrange
        column_name = col_name_2
        # Act
        # Assert -
        assert (
            self._2d_result.layout["yaxis"]["title"]["text"] == column_name
        ), "X axis label incorrect"

    def test_data_all_scatter_points_for_1d(
        self,
        dummy_dist_vd,
    ):
        # Arrange
        total_points = len(dummy_dist_vd[col_name_1])
        # Act
        result = dummy_dist_vd.outliers_plot(columns=[col_name_1], max_nb_points=10000)
        assert (
            result.data[0]["y"].shape[0] + result.data[1]["y"].shape[0] == total_points
        ), "All points are not plotted for 1d plot"

    def test_data_all_scatter_points_for_2d(self, dummy_dist_vd):
        # Arrange
        total_points = len(dummy_dist_vd[col_name_1])
        # Act
        result = dummy_dist_vd.outliers_plot(
            columns=[col_name_1, col_name_2], max_nb_points=10000
        )
        assert result.data[-1]["y"].shape[0] + result.data[-2]["y"].shape[
            0
        ] == pytest.approx(
            total_points, abs=1
        ), "All points are not plotted for 2d plot"

    def test_data_all_sinformation_plotted_for_2d(
        self,
    ):
        # Arrange
        total_elements = 4
        # Act
        assert (
            len(self._2d_result.data) == total_elements
        ), "The total number of elements plotted is not correct"

    def test_additional_options_custom_width_and_height(self, dummy_dist_vd):
        # Arrange
        custom_width = 700
        custom_height = 700
        # Act
        result = dummy_dist_vd.outliers_plot(
            columns=[col_name_1, col_name_2], width=custom_width, height=custom_height
        )
        # Assert
        assert result.layout["width"] == custom_width, "Custom width not working"
