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


# Vertica
from verticapy.tests_new.plotting.conftest import (
    get_xaxis_label,
    get_yaxis_label,
    get_width,
    get_height,
)

# Testing variables
COL_NAME_1 = "PetalLengthCm"
COL_NAME_2 = "SepalLengthCm"
PIVOT_COL_1 = "survived"
PIVOT_COL_2 = "pclass"


class TestMatplotlibVDFHeatMap:
    """
    Testing different attributes of Heatmap plot on a vDataFrame
    """

    @pytest.fixture(scope="class")
    def plot_result(self, iris_vd):
        """
        Create a heatmap plot
        """
        return iris_vd.heatmap([COL_NAME_1, COL_NAME_2])

    @pytest.fixture(scope="class")
    def plot_result_pivot(self, titanic_vd):
        """
        Create a heatmap plot using pivot table
        """
        return titanic_vd.pivot_table([PIVOT_COL_1, PIVOT_COL_2])

    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        """
        Get the plot results
        """
        self.result = plot_result

    @pytest.fixture(autouse=True)
    def pivot_result(self, plot_result_pivot):
        """
        Get the plot results
        """
        self.pivot_result = plot_result_pivot

    def test_properties_output_type(self, matplotlib_figure_object):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, matplotlib_figure_object), "Wrong object created"

    def test_properties_output_type_for_pivot_table(self, matplotlib_figure_object):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(
            self.pivot_result, matplotlib_figure_object
        ), "Wrong object created"

    def test_properties_output_type_for_corr(
        self, dummy_scatter_vd, matplotlib_figure_object
    ):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        result = dummy_scatter_vd.corr(method="spearman")
        # Assert - checking if correct object created
        assert isinstance(result, matplotlib_figure_object), "wrong object crated"

    def test_properties_xaxis_title(self):
        """
        Testing x-axis title
        """
        # Arrange
        test_title = COL_NAME_1
        # Act
        # Assert - checking x axis label
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_title(self):
        """
        Testing y-axis title
        """
        # Arrange
        test_title = COL_NAME_2
        # Act
        # Assert - checking y axis label
        assert get_yaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_labels_for_categorical_data(self, titanic_vd):
        """
        Test labels for Y-axis
        """
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
        """
        Testing custom width and height
        """
        # Arrange
        custom_width = 3
        custom_height = 4
        # Act
        result = iris_vd.heatmap(
            [COL_NAME_1, COL_NAME_2], width=custom_width, height=custom_height
        )
        # Assert
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"

    @pytest.mark.parametrize("method", ["count", "density"])
    def test_properties_output_type_for_all_options(
        self, iris_vd, matplotlib_figure_object, method
    ):
        """
        Test different method types
        """
        # Arrange
        # Act
        result = iris_vd.heatmap(
            [COL_NAME_1, COL_NAME_2],
            method=method,
        )
        # Assert - checking if correct object created
        assert isinstance(result, matplotlib_figure_object), "Wrong object created"
