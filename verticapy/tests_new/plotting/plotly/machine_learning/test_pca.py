# Pytest
import pytest

# Standard Python Modules


# Other Modules
import numpy as np

# Verticapy
from verticapy.learn.decomposition import PCA

# Testing variables
col_name_1 = "X"
col_name_2 = "Y"
col_name_3 = "Z"


@pytest.fixture(scope="class")
def plot_result(load_plotly, dummy_scatter_vd):
    model = PCA("pca_circle_test")
    model.drop()
    model.fit(dummy_scatter_vd[col_name_1, col_name_2, col_name_3])
    return model.plot_circle()


class TestMachineLearningPCACirclePlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result

    def test_properties_output_type(self, plotly_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result) == plotly_figure_object, "Wrong object crated"

    def test_properties_xaxis_label(self):
        # Arrange
        test_title = "Dim1"
        # Act
        # Assert
        assert (
            test_title in self.result.layout["xaxis"]["title"]["text"]
        ), "X axis label incorrect"

    def test_properties_yaxis_label(self):
        # Arrange
        test_title = "Dim2"
        # Act
        # Assert
        assert (
            test_title in self.result.layout["yaxis"]["title"]["text"]
        ), "Y axis label incorrect"

    def test_data_no_of_columns(self):
        # Arrange
        total_items = 3
        # Act
        # Assert
        assert len(self.result.data) == total_items, "Some columns missing"

    def test_additional_options_custom_height(self, dummy_scatter_vd):
        # rrange
        custom_height = 650
        custom_width = 700
        # Act
        model = PCA("pca_circle_test")
        model.drop()
        model.fit(dummy_scatter_vd[col_name_1, col_name_2, col_name_3])
        result = model.plot_circle(height=custom_height, width=custom_width)
        # Assert
        assert (
            result.layout["height"] == custom_height
            and result.layout["width"] == custom_width
        ), "Custom height and width not working"
