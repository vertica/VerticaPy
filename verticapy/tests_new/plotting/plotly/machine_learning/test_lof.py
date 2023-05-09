# Pytest
import pytest

# Standard Python Modules


# Other Modules
import numpy as np

# Verticapy
from verticapy.learn.neighbors import LocalOutlierFactor

# Testing variables
col_name_1 = "X"
col_name_2 = "Y"
col_name_3 = "Z"


@pytest.fixture(scope="class")
def plot_result(load_plotly, dummy_scatter_vd):
    model = LocalOutlierFactor("lof_test")
    model.fit(dummy_scatter_vd, [col_name_1, col_name_2])
    return model.plot()


@pytest.fixture(scope="class")
def plot_result_2(load_plotly, dummy_scatter_vd):
    model = LocalOutlierFactor("lof_test_3d")
    model.fit(dummy_scatter_vd, [col_name_1, col_name_2, col_name_3])
    return model.plot()


class TestMachineLearningLOFPlot:
    @pytest.fixture(autouse=True)
    def result_2d(self, plot_result):
        self.result = plot_result

    @pytest.fixture(autouse=True)
    def result_3d(self, plot_result_2):
        self.result_3d = plot_result_2

    def test_properties_output_type_for_2d(self, plotly_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result) == plotly_figure_object, "wrong object crated"

    def test_properties_output_type_for_3d(self, plotly_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result_3d) == plotly_figure_object, "wrong object crated"

    def test_properties_xaxis_label(self):
        # Arrange
        test_title = col_name_1
        # Act
        # Assert
        assert (
            self.result.layout["xaxis"]["title"]["text"] == test_title
        ), "X axis label incorrect"

    def test_properties_yaxis_label(self):
        # Arrange
        test_title = col_name_2
        # Act
        # Assert
        assert (
            self.result.layout["yaxis"]["title"]["text"] == test_title
        ), "Y axis label incorrect"

    def test_properties_xaxis_label_for_3d(self):
        # Arrange
        test_title = col_name_1
        # Act
        # Assert
        assert (
            self.result_3d.layout["xaxis"]["title"]["text"] == test_title
        ), "X axis label incorrect"

    def test_properties_yaxis_label_for_3d(self):
        # Arrange
        test_title = col_name_2
        # Act
        # Assert
        assert (
            self.result_3d.layout["yaxis"]["title"]["text"] == test_title
        ), "Y axis label incorrect"

    def test_properties_scatter_and_line_plot(self):
        # Arrange
        total_items = 2
        # Act
        # Assert
        assert len(self.result.data) == total_items, "Either outline or scatter missing"

    def test_properties_hoverinfo_for_2d(self):
        # Arrange
        x = "{x}"
        y = "{y}"
        # Act
        # Assert
        assert (
            x in self.result.data[1]["hovertemplate"]
            and y in self.result.data[1]["hovertemplate"]
        ), "Hover information does not contain x or y"

    def test_properties_hoverinfo_for_3d(self):
        # Arrange
        x = "{x}"
        y = "{y}"
        z = "{z}"
        # Act
        # Assert
        assert (
            (x in self.result_3d.data[1]["hovertemplate"])
            and (y in self.result_3d.data[1]["hovertemplate"])
            and (z in self.result_3d.data[1]["hovertemplate"])
        ), "Hover information does not contain x, y or z"

    def test_additional_options_custom_height(self, load_plotly, dummy_scatter_vd):
        # rrange
        custom_height = 650
        custom_width = 700
        # Act
        model = LocalOutlierFactor("lof_test")
        model.fit(dummy_scatter_vd, [col_name_1, col_name_2])
        result = model.plot(height=custom_height, width=custom_width)
        # Assert
        assert (
            result.layout["height"] == custom_height
            and result.layout["width"] == custom_width
        ), "Custom height and width not working"
