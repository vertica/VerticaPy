# Pytest
import pytest

# Standard Python Modules


# Other Modules
import numpy as np

# Vertica
from ..conftest import get_xaxis_label, get_yaxis_label

# Testing variables
col_name = "check 2"
col_name_2 = "check 1"


@pytest.fixture(scope="class")
def plot_result(dummy_vd):
    return dummy_vd[col_name].bar()


class TestMatplotlibBarPlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result

    def test_properties_output_type(self, matplotlib_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, matplotlib_figure_object), "wrong object crated"

    def test_properties_xaxis_label(self):
        # Arrange
        test_title = col_name
        # Act
        # Assert - checking x axis label
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_label(self):
        # Arrange
        test_title = "density"
        # Act
        # Assert - checking y axis label
        assert get_yaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_xaxis_category(self):
        # Arrange
        # Act
        # Assert
        assert self.result.xaxis.get_scale() == "linear"

    def test_additional_options_custom_width_and_height(self, dummy_vd):
        # Arrange
        custom_width = 3
        custom_height = 4
        # Act
        result = dummy_vd[col_name].bar(
            width=custom_width,
            height=custom_height,
        )
        # Assert - checking if correct object created
        assert (
            result.get_figure().get_size_inches()[0] == custom_width
            and result.get_figure().get_size_inches()[1] == custom_height
        ), "Custom width or height not working"

    def test_additional_options_kind_stack(self, dummy_vd, matplotlib_figure_object):
        # Arrange
        kind = "stacked"
        # Act
        result3 = dummy_vd.bar(
            columns=[col_name],
            method="avg",
            of=col_name_2,
            kind=kind,
        )
        # Assert
        assert isinstance(self.result, matplotlib_figure_object), "wrong object crated"
