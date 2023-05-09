# Pytest
import pytest

# Standard Python Modules


# Other Modules


# Verticapy
from verticapy.learn.linear_model import LinearRegression

# Testing variables
col_name_1 = "X"
col_name_2 = "Y"


@pytest.fixture(scope="class")
def plot_result(dummy_scatter_vd):
    model = LinearRegression("LR_churn")
    model.fit(dummy_scatter_vd, [col_name_1], col_name_2)
    return model.plot()


class TestMachineLearningRegressionPlot:
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

    def test_properties_scatter_and_line_plot(self):
        # Arrange
        total_items = 2
        # Act
        # Assert
        assert len(self.result.data) == total_items, "Either line or scatter missing"

    def test_data_all_scatter_points(self):
        # Arrange
        no_of_points = 100
        # Act
        # Assert
        assert (
            len(self.result.data[0]["x"]) == no_of_points
        ), "Discrepancy between points plotted and total number ofp oints"

    def test_additional_options_custom_height(self, load_plotly, dummy_scatter_vd):
        # rrange
        custom_height = 650
        custom_width = 700
        # Act
        model = LinearRegression("LR_churn")
        model.fit(dummy_scatter_vd, ["X"], "Y")
        result = model.plot(height=custom_height, width=custom_width)
        # Assert
        assert (
            result.layout["height"] == custom_height
            and result.layout["width"] == custom_width
        ), "Custom height and width not working"
