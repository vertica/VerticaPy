# Pytest
import pytest


@pytest.fixture(scope="class")
def acf_plot_result(amazon_vd):
    return amazon_vd.acf(
        ts="date",
        column="number",
        p=12,
        by=["state"],
        unit="month",
        method="spearman",
    )


class TestACFPlot:
    @pytest.fixture(autouse=True)
    def result(self, acf_plot_result):
        self.result = acf_plot_result

    def test_properties_output_type_for(self, plotly_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result) == plotly_figure_object, "wrong object crated"

    def test_properties_xaxis_label(self):
        # Arrange
        test_title = "Lag"
        # Act
        # Assert - checking x axis label
        assert (
            self.result.layout["xaxis"]["title"]["text"] == test_title
        ), "X axis label incorrect"

    def test_properties_scatter_points_and_confidence(self):
        # Arrange
        total_elements = 3
        # Act
        # Assert - checking if all plotting objects were created
        assert (
            len(self.result.data) == total_elements
        ), "Some elements of plot are missing"

    def test_properties_vertical_lines_for_custom_lag(self, load_plotly, amazon_vd):
        # Arrange
        lag_number = 24
        # Act
        result = amazon_vd.acf(
            ts="date",
            column="number",
            p=lag_number - 1,
            by=["state"],
            unit="month",
            method="spearman",
        )
        # Assert
        assert (
            len(result.layout["shapes"]) == lag_number
        ), "Number of vertical lines inconsistent"

    def test_properties_mode_lines(self, load_plotly, amazon_vd):
        # Arrange
        mode = "lines+markers"
        # Act
        result = amazon_vd.acf(
            ts="date",
            column="number",
            p=12,
            by=["state"],
            unit="month",
            method="spearman",
            kind="line",
        )
        # Assert
        assert result.data[-1]["mode"] == mode, "Number of vertical lines inconsistent"

    def test_data_all_scatter_points(self, acf_plot_result):
        # Arrange
        lag_number = 13
        # Act
        # Assert
        assert (
            len(acf_plot_result.data[0]["x"]) == lag_number
        ), "Number of lag points inconsistent"

    def test_additional_options_custom_width_and_height(self, load_plotly, amazon_vd):
        # Arrange
        custom_width = 700
        custom_height = 700
        # Act
        result = amazon_vd.acf(
            ts="date",
            column="number",
            p=12,
            by=["state"],
            unit="month",
            method="spearman",
            width=custom_width,
            height=custom_height,
        )
        # Assert - checking if correct object created
        assert (
            result.layout["width"] == custom_width
            and result.layout["height"] == custom_height
        ), "Custom width or height not working"
