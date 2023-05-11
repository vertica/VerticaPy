"""
(c)  Copyright  [2018-2023]  OpenText  or one of its
affiliates.  Licensed  under  the   Apache  License,
Version 2.0 (the  "License"); You  may  not use this
file except in compliance with the License.

You may obtain a copy of the License at:
http://www.apache.org/licenses/LICENSE-2.0

Unless  required  by applicable  law or  agreed to in
writing, software  distributed  under the  License is
distributed on an  "AS IS" BASIS,  WITHOUT WARRANTIES
OR CONDITIONS OF ANY KIND, either express or implied.
See the  License for the specific  language governing
permissions and limitations under the License.
"""
# Pytest
import pytest

# Standard Python Modules


# Other Modules
import numpy as np

# Testing variables
col_name_1 = "PetalLengthCm"
col_name_2 = "SepalLengthCm"
pivot_col_1 = "survived"
pivot_col_2 = "pclass"


@pytest.fixture(scope="class")
def plot_result(iris_vd):
    return iris_vd.heatmap([col_name_1, col_name_2])


@pytest.fixture(scope="class")
def plot_result_pivot(titanic_vd):
    return titanic_vd.pivot_table([pivot_col_1, pivot_col_2])


class TestVDFHeatMap:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result

    @pytest.fixture(autouse=True)
    def result_2(self, plot_result_pivot):
        self.pivot_result = plot_result_pivot

    def test_properties_output_type(self, plotly_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result) == plotly_figure_object, "wrong object crated"

    def test_properties_output_type_for_pivot_table(self, plotly_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert (
            type(self.pivot_result) == plotly_figure_object
        ), "wrong object crated for pivot table"

    def test_properties_output_type_for_corr(
        self, dummy_scatter_vd, plotly_figure_object
    ):
        # Arrange
        # Act
        result = dummy_scatter_vd.corr(method="spearman")
        # Assert - checking if correct object created
        assert (
            type(result) == plotly_figure_object
        ), "wrong object crated for corr() plot"

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

    # ToDo Remove double quotes after the labels are fixed
    def test_properties_yaxis_labels_for_categorical_data(self, titanic_vd):
        # Arrange
        expected_labels = (
            '"survived"',
            '"pclass"',
            '"fare"',
            '"parch"',
            '"age"',
            '"sibsp"',
            '"body"',
        )
        # Act
        result = titanic_vd.corr(method="pearson", focus="survived")
        # Assert
        assert result.data[0]["y"] == expected_labels, "Y-axis labels incorrect"

    def test_data_matrix_shape(self, iris_vd):
        # Arrange
        expected_shape = (9, 6)
        # Act
        # Assert
        assert (
            self.result.data[0]["z"].shape == expected_shape
        ), "Incorrect shape of output matrix"

    def test_data_matrix_shape_for_pivot_table(
        self,
    ):
        # Arrange
        expected_shape = (3, 2)
        # Act
        # Assert
        assert (
            self.pivot_result.data[0]["z"].shape == expected_shape
        ), "Incorrect shape of output matrix"

    def test_data_x_range(self, iris_vd):
        # Arrange
        upper_bound = iris_vd[col_name_1].max()
        lower_bound = iris_vd[col_name_1].min()
        # Act
        x_array = np.array(self.result.data[0]["x"], dtype=float)
        # Assert
        assert np.all(
            (x_array[1:] >= lower_bound) & (x_array[:-1] <= upper_bound)
        ), "X-axis Values outside of data range"

    def test_data_y_range(self, iris_vd):
        # Arrange
        upper_bound = iris_vd[col_name_2].max()
        lower_bound = iris_vd[col_name_2].min()
        # Act
        y_array = np.array(self.result.data[0]["y"], dtype=float)
        # Assert
        assert np.all(
            (y_array[:-1] >= lower_bound) & (y_array[1:] <= upper_bound)
        ), "X-axis Values outside of data range"

    def test_additional_options_custom_width_height(self, iris_vd):
        # Arrange
        custom_width = 400
        custom_height = 700
        # Act
        result = iris_vd.heatmap(
            [col_name_1, col_name_2], width=custom_width, height=custom_height
        )
        # Assert
        assert (
            result.layout["width"] == custom_width
            and result.layout["height"] == custom_height
        )
