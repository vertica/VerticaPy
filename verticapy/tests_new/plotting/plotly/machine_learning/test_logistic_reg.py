# Pytest
import pytest

# Standard Python Modules


# Other Modules
import numpy as np

# Verticapy
from verticapy.learn.linear_model import LogisticRegression

# Testing variables
col_name_1 = "fare"
col_name_2 = "survived"
col_name_3 = "age"


@pytest.fixture(scope="class")
def plot_result(titanic_vd):
    model = LogisticRegression("log_reg_test")
    model.fit(titanic_vd, [col_name_1], col_name_2)
    return model.plot()


@pytest.fixture(scope="class")
def plot_result_2(titanic_vd):
    model = LogisticRegression("log_reg_test")
    model.fit(titanic_vd, [col_name_1, col_name_3], col_name_2)
    return model.plot()


class TestMachineLearningLogisticRegressionPlot:
    @pytest.fixture(autouse=True)
    def result_2d(self, plot_result):
        self.result_2d = plot_result

    @pytest.fixture(autouse=True)
    def result_3d(self, plot_result_2):
        self.result_3d = plot_result_2

    def test_properties_output_type_for_2d(self, plotly_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result_2d) == plotly_figure_object, "wrong object crated"

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
            self.result_2d.layout["xaxis"]["title"]["text"] == test_title
        ), "X axis label incorrect"

    def test_properties_yaxis_label(self):
        # Arrange
        test_title = "P(survived = 1)"
        # Act
        # Assert
        assert (
            self.result_2d.layout["yaxis"]["title"]["text"] == test_title
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
        test_title = "P(survived = 1)"
        # Act
        # Assert
        assert (
            self.result_3d.layout["yaxis"]["title"]["text"] == test_title
        ), "Y axis label incorrect"

    def test_properties_two_scatter_and_line_plot(self):
        # Arrange
        total_items = 3
        # Act
        # Assert
        assert (
            len(self.result_2d.data) == total_items
        ), "Either line or the two scatter plots are missing"

    def test_additional_options_custom_height(self, load_plotly, titanic_vd):
        # rrange
        custom_height = 650
        custom_width = 700
        # Act
        model = LogisticRegression("log_reg_test")
        model.fit(titanic_vd, [col_name_1], col_name_2)
        result = model.plot(height=custom_height, width=custom_width)
        # Assert
        assert (
            result.layout["height"] == custom_height
            and result.layout["width"] == custom_width
        ), "Custom height and width not working"
