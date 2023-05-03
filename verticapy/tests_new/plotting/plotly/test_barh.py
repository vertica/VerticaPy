# Pytest
import pytest

# Standard Python Modules


# Other Modules
import numpy as np

# Testing variables
col_name = "check 2"


@pytest.fixture(scope="class")
def plot_result(dummy_vd):
    return dummy_vd[col_name].barh()


class TestBarhPlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result

    def test_properties_output_type_for(self, plotly_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result) == plotly_figure_object, "wrong object crated"

    def test_data_sum_equals_one(self):
        # Arrange
        # Act
        # Assert - Comparing total adds up to 1
        assert sum(self.result.data[0]["x"]) == 1

    def test_data_ratios(self, dummy_vd):
        ### Checking if the density was plotted correctly
        nums = dummy_vd.to_pandas()[col_name].value_counts()
        total = len(dummy_vd)
        assert set(self.result.data[0]["x"]).issubset(
            set([nums["A"] / total, nums["B"] / total, nums["C"] / total])
        )

    def test_properties_xaxis_label(self):
        # Arrange
        test_title = "density"
        # Act
        # Assert - checking x axis label
        assert (
            self.result.layout["xaxis"]["title"]["text"] == test_title
        ), "X axis label incorrect"

    def test_properties_yaxis_label(self):
        # Arrange
        test_title = col_name
        # Act
        # Assert - checking y axis label
        assert (
            self.result.layout["yaxis"]["title"]["text"] == test_title
        ), "X axis label incorrect"

    def test_xaxis_category(self):
        # Arrange
        # Act
        # Assert
        assert self.result.layout["yaxis"]["type"] == "category"

    def test_all_categories_crated(self):
        assert set(self.result.data[0]["y"]).issubset(set(["A", "B", "C"]))

    def test_additional_options_custom_width_and_height(self, dummy_vd):
        # Arrange
        custom_width = 300
        custom_height = 400
        # Act
        result = dummy_vd[col_name].barh(
            width=custom_width,
            height=custom_height,
        )
        # Assert - checking if correct object created
        assert (
            result.layout["width"] == custom_width
            and result.layout["height"] == custom_height
        ), "Custom width or height not working"

    def test_additional_options_custom_x_axis_title(self, dummy_vd):
        # Arrange
        # Act
        result = dummy_vd[col_name].barh(xaxis_title="Custom X Axis Title")
        # Assert
        assert result.layout["xaxis"]["title"]["text"] == "Custom X Axis Title"

    def test_additional_options_custom_y_axis_title(self, dummy_vd):
        # Arrange
        # Act
        result = dummy_vd[col_name].barh(yaxis_title="Custom Y Axis Title")
        # Assert
        assert result.layout["yaxis"]["title"]["text"] == "Custom Y Axis Title"
