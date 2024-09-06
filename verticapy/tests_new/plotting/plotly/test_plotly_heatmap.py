"""
Copyright  (c)  2018-2024 Open Text  or  one  of its
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
import numpy as np

# Vertica
from verticapy.tests_new.plotting.base_test_files import VDFPivotHeatMap, VDFHeatMap


class TestPlotlyVDFPivotHeatMap(VDFPivotHeatMap):
    """
    Testing different attributes of Heatmap plot on a vDataFrame
    """

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
        # Assert
        assert result.data[0]["y"] == expected_labels, "Y-axis labels incorrect"

    def test_data_matrix_shape_for_pivot_table(
        self,
    ):
        """
        Test shape of matrix
        """
        # Arrange
        expected_shape = (3, 2)
        # Act
        # Assert
        assert (
            self.result.data[0]["z"].shape == expected_shape
        ), "Incorrect shape of output matrix"


@pytest.mark.skip("Error in Plotly need to be fixed")
class TestPlotlyVDFHeatMap(VDFHeatMap):
    """
    Testing different attributes of Heatmap plot on a vDataFrame
    """

    def test_data_matrix_shape(
        self,
    ):
        """
        Test shape of matrix
        """
        # Arrange
        expected_shape = (9, 6)
        # Act
        # Assert
        assert (
            self.result.data[0]["z"].shape == expected_shape
        ), "Incorrect shape of output matrix"

    def test_data_x_range(self, iris_vd):
        """
        Test x-axis range
        """
        # Arrange
        upper_bound = iris_vd[self.COL_NAME_1].max()
        lower_bound = iris_vd[self.COL_NAME_1].min()
        # Act
        x_array = np.array(self.result.data[0]["x"], dtype=float)
        # Assert
        assert np.all(
            (x_array[1:] >= lower_bound) & (x_array[:-1] <= upper_bound)
        ), "X-axis Values outside of data range"

    def test_data_y_range(self, iris_vd):
        """
        Test y-axis range
        """
        # Arrange
        upper_bound = iris_vd[self.COL_NAME_2].max()
        lower_bound = iris_vd[self.COL_NAME_2].min()
        # Act
        y_array = np.array(self.result.data[0]["y"], dtype=float)
        # Assert
        assert np.all(
            (y_array[:-1] >= lower_bound) & (y_array[1:] <= upper_bound)
        ), "X-axis Values outside of data range"
