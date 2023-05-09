# Pytest
import pytest

# Standard Python Modules


# Other Modules
import numpy as np

# Verticapy
from verticapy.learn.ensemble import RandomForestClassifier

# Testing variables
col_name_1 = "PetalLengthCm"
col_name_2 = "PetalWidthCm"
col_name_3 = "SepalWidthCm"
col_name_4 = "SepalLengthCm"
by_col = "Species"


@pytest.fixture(scope="class")
def plot_result(iris_vd):
    model = RandomForestClassifier("importance_test")
    model.fit(
        iris_vd,
        [col_name_1, col_name_2, col_name_3, col_name_4],
        by_col,
    )
    return model.features_importance(), model


class TestMachineLearningImportanceBarChart:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result[0]

    def test_properties_output_type_for_2d(self, plotly_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result) == plotly_figure_object, "wrong object crated"

    def test_properties_xaxis_label(self):
        # Arrange
        test_title = "Importance (%)"
        # Act
        # Assert
        assert (
            self.result.layout["xaxis"]["title"]["text"] == test_title
        ), "X axis label incorrect"

    def test_properties_yaxis_label(self):
        # Arrange
        test_title = "Features"
        # Act
        # Assert
        assert (
            self.result.layout["yaxis"]["title"]["text"] == test_title
        ), "Y axis label incorrect"

    def test_data_no_of_columns(self):
        # Arrange
        total_items = 4
        # Act
        # Assert
        assert len(self.result.data[0]["x"]) == total_items, "Some columns missing"

    def test_additional_options_custom_height(self, plot_result):
        # rrange
        custom_height = 650
        custom_width = 700
        # Act
        result = plot_result[1].features_importance(
            height=custom_height, width=custom_width
        )
        # Assert
        assert (
            result.layout["height"] == custom_height
            and result.layout["width"] == custom_width
        ), "Custom height and width not working"
