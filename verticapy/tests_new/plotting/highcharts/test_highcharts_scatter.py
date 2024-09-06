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
# Vertica
from verticapy.tests_new.plotting.base_test_files import (
    ScatterVDF2DPlot,
    ScatterVDF3DPlot,
)


class TestHighchartsScatterVDF2DPlot(ScatterVDF2DPlot):
    """
    Testing different attributes of 2D scatter plot on a vDataFrame
    """

    def test_properties_all_unique_values_for_by(self, dummy_scatter_vd):
        """
        Test if all unique valies are inside the plot
        """
        # Arrange
        # Act
        result = dummy_scatter_vd.scatter(
            [
                self.COL_NAME_2,
                self.COL_NAME_3,
            ],
            by=self.COL_NAME_4,
        )
        # Assert
        assert len(result.data_temp) == len(
            self.all_categories
        ), "Some unique values were not found in the plot"

    def test_data_total_number_of_points(self, dummy_scatter_vd):
        """
        Test if all datapoints were plotted
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert sum(len(item.data) for item in self.result.data_temp) == len(
            dummy_scatter_vd
        ), "Number of points not consistent with data"


class TestHighchartsScatterVDF3DPlot(ScatterVDF3DPlot):
    """
    Testing different attributes of 3D scatter plot on a vDataFrame
    """

    def test_properties_all_unique_values_for_by_3d_plot(
        self,
    ):
        """
        Test if all unique values plotted
        """
        # Arrange
        # Act
        # Assert
        assert len(self.result.data_temp) == len(
            self.all_categories
        ), "Some unique values were not found in the plot"

    def test_data_total_number_of_points_3d_plot(self, dummy_scatter_vd):
        """
        Test if all datapoints were plotted
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert sum(len(item.data) for item in self.result.data_temp) == len(
            dummy_scatter_vd
        ), "Number of points not consistent with data"
