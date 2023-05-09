# Pytest
import pytest

# Standard Python Modules


# Other Modules
import numpy as np

# Verticapy
from verticapy.learn.cluster import KMeans

# Testing variables
col_name_1 = "PetalLengthCm"
col_name_2 = "PetalWidthCm"


@pytest.fixture(scope="class")
def plot_result(iris_vd):
    model = KMeans(name="test_KMeans_iris")
    model.fit(
        iris_vd,
        [col_name_1, col_name_2],
    )
    return model.plot_voronoi()


class TestMachineLearningLiftChart:
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

    def test_properties_no_of_elements(self):
        # Arrange
        total_items = 20
        # Act
        # Assert
        assert len(self.result.data) == pytest.approx(
            total_items, abs=2
        ), "Some elements missing"

    def test_additional_options_custom_height(self, load_plotly, iris_vd):
        # rrange
        custom_height = 650
        custom_width = 700
        model = KMeans(name="public.KMeans_iris")
        model.fit(
            iris_vd,
            [col_name_1, col_name_2],
        )
        # Act
        result = model.plot_voronoi(width=custom_width, height=custom_height)
        # Assert
        assert (
            result.layout["height"] == custom_height
            and result.layout["width"] == custom_width
        ), "Custom height and width not working"
