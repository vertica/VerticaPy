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

# Vertica
from verticapy.tests_new.plotting.conftest import get_xaxis_label, get_yaxis_label

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


class TestMatplotlibHeatMap:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result

    @pytest.fixture(autouse=True)
    def result_2(self, plot_result_pivot):
        self.pivot_result = plot_result_pivot

    def test_properties_output_type(self, matplotlib_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, matplotlib_figure_object), "Wrong object created"

    def test_properties_output_type_for_pivot_table(self, matplotlib_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(
            self.pivot_result, matplotlib_figure_object
        ), "Wrong object created"

    def test_properties_output_type_for_corr(
        self, dummy_scatter_vd, matplotlib_figure_object
    ):
        # Arrange
        # Act
        result = dummy_scatter_vd.corr(method="spearman")
        # Assert - checking if correct object created
        assert isinstance(result, matplotlib_figure_object), "wrong object crated"

    def test_properties_xaxis_title(self):
        # Arrange
        test_title = col_name_1
        # Act
        # Assert - checking x axis label
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_title(self):
        # Arrange
        test_title = col_name_2
        # Act
        # Assert - checking y axis label
        assert get_yaxis_label(self.result) == test_title, "X axis label incorrect"

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
        yaxis_labels = [
            result.get_yticklabels()[i].get_text()
            for i in range(len(result.get_yticklabels()))
        ]
        # Assert
        assert set(yaxis_labels).issubset(expected_labels), "Y-axis labels incorrect"

    def test_additional_options_custom_width_and_height(self, iris_vd):
        # Arrange
        custom_width = 3
        custom_height = 4
        # Act
        result = iris_vd.heatmap(
            [col_name_1, col_name_2], width=custom_width, height=custom_height
        )
        # Assert
        assert (
            result.get_figure().get_size_inches()[0] == custom_width
            and result.get_figure().get_size_inches()[1] == custom_height
        ), "Custom width or height not working"

    @pytest.mark.parametrize("method", ["count", "density"])
    def test_properties_output_type_for_all_options(
        self, iris_vd, matplotlib_figure_object, method
    ):
        # Arrange
        # Act
        result = iris_vd.heatmap(
            [col_name_1, col_name_2],
            method=method,
        )
        # Assert - checking if correct object created
        assert isinstance(self.result, matplotlib_figure_object), "Wrong object created"
