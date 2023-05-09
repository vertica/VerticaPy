# Pytest
import pytest

# Standard Python Modules


# Other Modules
import numpy as np

# Verticapy
from verticapy.learn.ensemble import RandomForestClassifier
from verticapy.learn.model_selection import lift_chart, prc_curve

# Testing variables
col_name_1 = "0"
col_name_2 = "1"
by_col = "binary"
pos_label = "0"


@pytest.fixture(scope="class")
def plot_result_roc(load_plotly, dummy_dist_vd):
    model = RandomForestClassifier("roc_plot_test")
    model.drop()
    model.fit(dummy_dist_vd, [col_name_1, col_name_2], by_col)
    return model.roc_curve()


@pytest.fixture(scope="class")
def plot_result_cutoff(load_plotly, dummy_dist_vd):
    model = RandomForestClassifier("roc_plot_test_2")
    model.drop()
    model.fit(dummy_dist_vd, [col_name_1, col_name_2], by_col)
    return model.cutoff_curve(pos_label=pos_label)


@pytest.fixture(scope="class")
def plot_result_prc(load_plotly, dummy_probability_data):
    return prc_curve("y_true", "y_score", dummy_probability_data)


@pytest.fixture(scope="class")
def plot_result_lift_chart(load_plotly, dummy_probability_data):
    return lift_chart("y_true", "y_score", dummy_probability_data)


class TestMachineLearningROCCurve:
    @pytest.fixture(autouse=True)
    def result(self, plot_result_roc):
        self.result = plot_result_roc

    def test_properties_output_type(self, plotly_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result) == plotly_figure_object, "Wrong object crated"

    def test_properties_title(self):
        # Arrange
        test_title = "ROC Curve"
        # Act
        # Assert
        assert self.result.layout["title"]["text"] == test_title, "Plot Title Incorrect"

    def test_properties_xaxis_label(self):
        # Arrange
        test_title = "False Positive Rate (1-Specificity)"
        # Act
        # Assert
        assert (
            self.result.layout["xaxis"]["title"]["text"] == test_title
        ), "X axis label incorrect"

    def test_properties_yaxis_label(self):
        # Arrange
        test_title = "True Positive Rate (Sensitivity)"
        # Act
        # Assert
        assert (
            self.result.layout["yaxis"]["title"]["text"] == test_title
        ), "Y axis label incorrect"

    def test_properties_no_of_elements(self):
        # Arrange
        total_items = 2
        # Act
        # Assert
        assert len(self.result.data) == total_items, "Some elements missing"

    def test_additional_options_custom_height_and_width(self, dummy_dist_vd):
        # rrange
        custom_height = 650
        custom_width = 700
        # Act
        model = RandomForestClassifier("roc_plot_test")
        model.drop()
        model.fit(dummy_dist_vd, [col_name_1, col_name_2], by_col)
        result = model.cutoff_curve(
            pos_label=pos_label, width=custom_width, height=custom_height
        )
        # Assert
        assert (
            result.layout["height"] == custom_height
            and result.layout["width"] == custom_width
        ), "Custom height and width not working"


class TestMachineLearningCutoffCurve:
    @pytest.fixture(autouse=True)
    def result(self, plot_result_cutoff):
        self.result = plot_result_cutoff

    def test_properties_output_type(self, plotly_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result) == plotly_figure_object, "Wrong object crated"

    def test_properties_xaxis_label(self):
        # Arrange
        test_title = "Decision Boundary"
        # Act
        # Assert
        assert (
            self.result.layout["xaxis"]["title"]["text"] == test_title
        ), "X axis label incorrect"

    def test_properties_yaxis_label(self):
        # Arrange
        test_title = "Values"
        # Act
        # Assert
        assert (
            self.result.layout["yaxis"]["title"]["text"] == test_title
        ), "Y axis label incorrect"

    def test_properties_no_of_elements(self):
        # Arrange
        total_items = 2
        # Act
        # Assert
        assert len(self.result.data) == total_items, "Some elements missing"

    def test_additional_options_custom_height(self, dummy_dist_vd):
        # rrange
        custom_height = 650
        custom_width = 700
        # Act
        model = RandomForestClassifier("cutoff_curve_plot_test")
        model.drop()
        model.fit(dummy_dist_vd, [col_name_1, col_name_2], by_col)
        result = model.cutoff_curve(
            pos_label=pos_label, width=custom_width, height=custom_height
        )
        # Assert
        assert (
            result.layout["height"] == custom_height
            and result.layout["width"] == custom_width
        ), "Custom height and width not working"


class TestMachineLearningPRCCurve:
    @pytest.fixture(autouse=True)
    def result(self, plot_result_prc):
        self.result = plot_result_prc

    def test_properties_output_type(self, plotly_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result) == plotly_figure_object, "Wrong object crated"

    def test_properties_xaxis_label(self):
        # Arrange
        test_title = "Recall"
        # Act
        # Assert
        assert (
            self.result.layout["xaxis"]["title"]["text"] == test_title
        ), "X axis label incorrect"

    def test_properties_yaxis_label(self):
        # Arrange
        test_title = "Precision"
        # Act
        # Assert
        assert (
            self.result.layout["yaxis"]["title"]["text"] == test_title
        ), "Y axis label incorrect"

    def test_properties_no_of_elements(self):
        # Arrange
        total_items = 1
        # Act
        # Assert
        assert len(self.result.data) == total_items, "Some elements missing"

    def test_additional_options_custom_height(self, dummy_probability_data):
        # rrange
        custom_height = 650
        custom_width = 700
        # Act
        result = prc_curve(
            "y_true",
            "y_score",
            dummy_probability_data,
            width=custom_width,
            height=custom_height,
        )
        # Assert
        assert (
            result.layout["height"] == custom_height
            and result.layout["width"] == custom_width
        ), "Custom height and width not working"


class TestMachineLearningLiftChart:
    @pytest.fixture(autouse=True)
    def result(self, plot_result_lift_chart):
        self.result = plot_result_lift_chart

    def test_properties_output_type(self, plotly_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result) == plotly_figure_object, "Wrong object crated"

    def test_properties_title(self):
        # Arrange
        test_title = "Lift Table"
        # Act
        # Assert
        assert self.result.layout["title"]["text"] == test_title, "Plot title incorrect"

    def test_properties_xaxis_label(self):
        # Arrange
        test_title = "Cumulative Data Fraction"
        # Act
        # Assert
        assert (
            self.result.layout["xaxis"]["title"]["text"] == test_title
        ), "X axis label incorrect"

    def test_properties_yaxis_label(self):
        # Arrange
        test_title = "Values"
        # Act
        # Assert
        assert (
            self.result.layout["yaxis"]["title"]["text"] == test_title
        ), "Y axis label incorrect"

    def test_properties_no_of_elements(self):
        # Arrange
        total_items = 2
        # Act
        # Assert
        assert len(self.result.data) == total_items, "Some elements missing"

    def test_additional_options_custom_height(
        self, load_plotly, dummy_probability_data
    ):
        # rrange
        custom_height = 650
        custom_width = 700
        # Act
        result = prc_curve(
            "y_true",
            "y_score",
            dummy_probability_data,
            width=custom_width,
            height=custom_height,
        )
        # Assert
        assert (
            result.layout["height"] == custom_height
            and result.layout["width"] == custom_width
        ), "Custom height and width not working"
