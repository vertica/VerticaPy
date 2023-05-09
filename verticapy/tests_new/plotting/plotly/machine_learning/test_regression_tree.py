# Pytest
import pytest

# Standard Python Modules


# Other Modules


# Verticapy
from verticapy.learn.tree import DecisionTreeRegressor

# Testing variables
col_name_1 = "0"
col_name_2 = "1"


@pytest.fixture(scope="class")
def plot_result(dummy_dist_vd):
    model = DecisionTreeRegressor(name="model_titanic")
    x_col = col_name_1
    y_col = col_name_2
    model.fit(dummy_dist_vd, x_col, y_col)
    return model.plot(), x_col, y_col


class TestMachineLearningRegressionTreePlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result, self.x_col, self.y_col = plot_result

    def test_properties_output_type(self, plotly_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result) == plotly_figure_object, "Wrong object crated"

    def test_properties_xaxis_label(self):
        # Arrange
        test_title = self.x_col
        # Act
        # Assert
        assert (
            self.result.layout["xaxis"]["title"]["text"] == test_title
        ), "X axis label incorrect"

    def test_properties_yaxis_label(self):
        # Arrange
        test_title = self.y_col
        # Act
        # Assert
        assert (
            self.result.layout["yaxis"]["title"]["text"] == test_title
        ), "Y axis label incorrect"

    def test_properties_observations_label(self):
        # Arrange
        test_title = "Observations"
        # Act
        # Assert
        assert self.result.data[0]["name"] == test_title, "Y axis label incorrect"

    def test_properties_prediction_label(self):
        # Arrange
        test_title = "Prediction"
        # Act
        # Assert
        assert self.result.data[1]["name"] == test_title, "Y axis label incorrect"

    def test_properties_hover_label(self):
        # Arrange
        test_title = f"{self.x_col}: %" "{x} <br>" f"{self.y_col}: %" "{y} <br>"
        # Act
        # Assert
        assert (
            self.result.data[0]["hovertemplate"] == test_title
        ), "Hover information incorrect"

    def test_properties_no_of_elements(self):
        # Arrange
        total_items = 2
        # Act
        # Assert
        assert len(self.result.data) == pytest.approx(
            total_items, abs=1
        ), "Some elements missing"

    def test_additional_options_custom_height(self, dummy_dist_vd):
        # rrange
        custom_height = 650
        custom_width = 700
        model = DecisionTreeRegressor(name="model_titanic")
        model.fit(dummy_dist_vd, col_name_1, col_name_2)
        # Act
        result = model.plot(
            height=custom_height,
            width=custom_width,
        )
        # Assert
        assert (
            result.layout["height"] == custom_height
            and result.layout["width"] == custom_width
        ), "Custom height and width not working"
