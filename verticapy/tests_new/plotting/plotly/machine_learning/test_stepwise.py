# Pytest
import pytest

# Standard Python Modules


# Other Modules


# Verticapy
from verticapy.learn.model_selection import stepwise
from verticapy.learn.linear_model import LogisticRegression

# Testing variables
col_name_1 = "age"
col_name_2 = "fare"
col_name_3 = "parch"
col_name_4 = "pclass"
by_col = "survived"


@pytest.fixture(scope="class")
def plot_result(titanic_vd):
    model = LogisticRegression(
        name="test_LR_titanic", tol=1e-4, max_iter=100, solver="Newton"
    )
    stepwise_result = stepwise(
        model,
        input_relation=titanic_vd,
        X=[
            col_name_1,
            col_name_2,
            col_name_3,
            col_name_4,
        ],
        y=by_col,
        direction="backward",
    )
    return stepwise_result.step_wise_


class TestMachineLearningStepwisePlot:
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
        test_title = "n_features"
        # Act
        # Assert
        assert (
            self.result.layout["xaxis"]["title"]["text"] == test_title
        ), "X axis label incorrect"

    def test_properties_yaxis_label(self):
        # Arrange
        test_title = "bic"
        # Act
        # Assert
        assert (
            self.result.layout["yaxis"]["title"]["text"] == test_title
        ), "Y axis label incorrect"

    def test_properties_no_of_elements(self):
        # Arrange
        total_items = 8
        # Act
        # Assert
        assert len(self.result.data) == pytest.approx(
            total_items, abs=1
        ), "Some elements missing"

    def test_data_start_and_end(self):
        # Arrange
        start = "Start"
        end = "End"
        # Act
        # Assert
        assert start in [
            self.result.data[i]["name"] for i in range(len(self.result.data))
        ] and end in [
            self.result.data[i]["name"] for i in range(len(self.result.data))
        ], "Some elements missing"

    def test_additional_options_custom_height(self, load_plotly, titanic_vd):
        # rrange
        custom_height = 650
        custom_width = 700
        model = LogisticRegression(
            name="test_LR_titanic", tol=1e-4, max_iter=100, solver="Newton"
        )
        # Act
        stepwise_result = stepwise(
            model,
            input_relation=titanic_vd,
            X=[
                "age",
                "fare",
                "parch",
                "pclass",
            ],
            y="survived",
            direction="backward",
            height=custom_height,
            width=custom_width,
        )
        result = stepwise_result.step_wise_
        # Assert
        assert (
            result.layout["height"] == custom_height
            and result.layout["width"] == custom_width
        ), "Custom height and width not working"
