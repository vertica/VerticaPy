# Pytest
import pytest

# Standard Python Modules


# Other Modules


# Verticapy
from verticapy.learn.model_selection import elbow

# Testing variables
col_name_1 = "PetalLengthCm"
col_name_2 = "PetalWidthCm"


@pytest.fixture(scope="class")
def plot_result(iris_vd):
    return elbow(input_relation=iris_vd, X=[col_name_1, col_name_2])


class TestMachineLearningElbowCurve:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result

    def test_properties_output_type_for(self, plotly_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result) == plotly_figure_object, "wrong object crated"

    def test_properties_xaxis_label(self):
        # Arrange
        test_title = "Number of Clusters"
        # Act
        # Assert - checking if correct object created
        assert (
            self.result.layout["xaxis"]["title"]["text"] == test_title
        ), "X axis label incorrect"

    def test_data_all_scatter_points(self):
        # Arrange
        mode = "markers+line"
        # Act
        # Assert - checking if correct object created
        assert set(self.result.data[0]["mode"]) == set(
            mode
        ), "Either lines or marker missing"

    @pytest.mark.slow
    @pytest.mark.notcritical
    def test_additional_options_custom_height(self, load_plotly, iris_vd):
        # rrange
        custom_height = 650
        custom_width = 700
        # Act
        result = elbow(
            input_relation=iris_vd,
            X=[col_name_1, col_name_2],
            width=custom_width,
            height=custom_height,
        )
        # Assert - checking if correct object created
        assert (
            result.layout["height"] == custom_height
            and result.layout["width"] == custom_width
        ), "Custom height and width not working"
