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
from ..conftest import BasicPlotTests


# Testing variables
COL_NAME_1 = "PetalLengthCm"
COL_NAME_2 = "SepalLengthCm"
PIVOT_COL_1 = "survived"
PIVOT_COL_2 = "pclass"


class TestHighchartsVDFPivotHeatMap(BasicPlotTests):
    """
    Testing different attributes of Heatmap plot on a vDataFrame
    """

    @pytest.fixture(autouse=True)
    def data(self, titanic_vd):
        """
        Load test data
        """
        self.data = titanic_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [PIVOT_COL_1, PIVOT_COL_2]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data.pivot_table,
            {"columns": [PIVOT_COL_1, PIVOT_COL_2]},
        )

    def test_properties_output_type_for_corr(
        self, dummy_scatter_vd, plotting_library_object
    ):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        result = dummy_scatter_vd.corr(method="spearman")
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "wrong object crated"

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
        yaxis_labels = result.options["yAxis"].categories
        # Assert
        assert set(yaxis_labels).issubset(expected_labels), "Y-axis labels incorrect"


@pytest.mark.skip("Error in highcharts need to be fixed")
class TestHighchartsVDFHeatMap(BasicPlotTests):
    """
    Testing different attributes of Heatmap plot on a vDataFrame
    """

    @pytest.fixture(autouse=True)
    def data(self, iris_vd):
        """
        Load test data
        """
        self.data = iris_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [COL_NAME_1, COL_NAME_2]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data.heatmap,
            {"columns": [COL_NAME_1, COL_NAME_2]},
        )

    @pytest.mark.parametrize("method", ["count", "density"])
    def test_properties_output_type_for_all_options(
        self, iris_vd, plotting_library_object, method
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
        assert isinstance(result, plotting_library_object), "Wrong object created"
