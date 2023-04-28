# Pytest
import pytest

# Standard Python Modules


# Other Modules
import numpy as np

# Testing variables
col_name_1 = "X"
col_name_2 = "Y"
col_name_3 = "Z"
col_name_4 = "Category"
all_categories = ["A", "B", "C"]


@pytest.fixture(scope="class")
def plot_result(dummy_scatter_vd):
    return dummy_scatter_vd.scatter([col_name_1, col_name_2])


@pytest.fixture(scope="class")
def plot_result_2(dummy_scatter_vd):
    return dummy_scatter_vd.scatter([col_name_1, col_name_2, col_name_3])


@pytest.fixture(scope="class")
def plot_result_3(dummy_scatter_vd):
    result = dummy_scatter_vd.scatter(
        [
            col_name_1,
            col_name_2,
            col_name_3,
        ],
        by=col_name_4,
    )
    return result


class TestVDFScatter2DPlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result

    def test_properties_output_type(self, plotly_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result) == plotly_figure_object, "wrong object crated"

    def test_properties_xaxis_title(
        self,
    ):
        # Arrange
        # Act
        # Assert
        assert (
            self.result.layout["xaxis"]["title"]["text"] == col_name_1
        ), "X-axis title issue"

    def test_properties_yaxis_title(
        self,
    ):
        # Arrange
        # Act
        # Assert
        assert (
            self.result.layout["yaxis"]["title"]["text"] == col_name_2
        ), "Y-axis title issue"

    def test_properties_all_unique_values_for_by(self, dummy_scatter_vd):
        # Arrange
        # Act
        result = dummy_scatter_vd.scatter(
            [
                col_name_2,
                col_name_3,
            ],
            by=col_name_4,
        )
        # Assert
        assert set(
            [result.data[0]["name"], result.data[1]["name"], result.data[2]["name"]]
        ).issubset(set(all_categories)), "Some unique values were not found in the plot"

    def test_properties_colors_for_by(self, dummy_scatter_vd):
        # Arrange
        # Act
        result = dummy_scatter_vd.scatter(
            [
                col_name_2,
                col_name_3,
            ],
            by=col_name_4,
        )
        assert (
            len(
                set(
                    [
                        result.data[0]["marker"]["color"],
                        result.data[1]["marker"]["color"],
                        result.data[2]["marker"]["color"],
                    ]
                )
            )
            == 3
        ), "Colors are not unique for three different cat_col parameter"

    def test_data_total_number_of_points(self, dummy_scatter_vd):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert len(self.result.data[0]["x"]) == len(
            dummy_scatter_vd
        ), "Number of points not consistent with data"
        assert len(self.result.data[0]["y"]) == len(
            dummy_scatter_vd
        ), "Number of points not consistent with data"

    def test_data_random_point_from_plot_in_data(self, dummy_scatter_vd):
        # Arrange
        # Act
        # Assert -
        len_of_data = len(
            dummy_scatter_vd.search(
                conditions=[
                    f"{col_name_1} ={self.result.data[0]['x'][0]} and {col_name_2}={self.result.data[0]['y'][0]}"
                ],
                usecols=[col_name_1, col_name_2],
            )
        )
        assert len_of_data > 0, "A wrong point was plotted"

    def test_additional_options_custom_width_and_height(self, dummy_scatter_vd):
        # Arrange
        custom_width = 300
        custom_height = 400
        # Act
        result = dummy_scatter_vd.scatter(
            [col_name_1, col_name_2],
            width=custom_width,
            height=custom_height,
        )
        # Assert - checking if correct object created
        assert (
            result.layout["width"] == custom_width
            and result.layout["height"] == custom_height
        ), "Custom width or height not working"


class TestVDFScatter3DPlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result_2):
        self.result = plot_result_2

    def test_properties_output_type(self, plotly_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result) == plotly_figure_object, "wrong object crated"

    def test_properties_xaxis_title_3D_plot(
        self,
    ):
        # Arrange
        # Act
        # Assert
        assert (
            self.result.layout["scene"]["xaxis"]["title"]["text"] == col_name_1
        ), "X-axis title issue in 3D plot"

    def test_properties_yaxis_title_3D_plot(
        self,
    ):
        # Arrange
        # Act
        # Assert
        assert (
            self.result.layout["scene"]["yaxis"]["title"]["text"] == col_name_2
        ), "Y-axis title issue in 3D plot"

    def test_properties_zaxis_title_3D_plot(
        self,
    ):
        # Arrange
        # Act
        # Assert
        assert (
            self.result.layout["scene"]["zaxis"]["title"]["text"] == col_name_3
        ), "Z-axis title issue in 3D plot"

    def test_properties_all_unique_values_for_by_3D_plot(self, plot_result_3):
        # Arrange
        # Act
        result = plot_result_3
        # Assert
        assert set(
            [
                result.data[0]["name"],
                result.data[1]["name"],
                result.data[2]["name"],
            ]
        ).issubset(
            set(all_categories)
        ), "Some unique values were not found in the 3D plot"

    def test_data_total_number_of_points_3D_plot(self, dummy_scatter_vd):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert len(self.result.data[0]["x"]) == len(
            dummy_scatter_vd
        ), "Number of points not consistent with data"
        assert len(self.result.data[0]["y"]) == len(
            dummy_scatter_vd
        ), "Number of points not consistent with data"
        assert len(self.result.data[0]["z"]) == len(
            dummy_scatter_vd
        ), "Number of points not consistent with data"

    def test_properties_colors_for_by_3D_plot(self, plot_result_3):
        # Arrange
        # Act
        result = plot_result_3
        # Assert
        assert (
            len(
                set(
                    [
                        result.data[0]["marker"]["color"],
                        result.data[1]["marker"]["color"],
                        result.data[2]["marker"]["color"],
                    ]
                )
            )
            == 3
        ), "Colors are not unique for three different cat_col parameter"
