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
COL_NAME_1 = "X"
COL_NAME_2 = "Y"
COL_NAME_3 = "Z"
COL_NAME_4 = "Category"
all_categories = ["A", "B", "C"]


class TestHighchartsScatterVDF2DPlot(BasicPlotTests):
    """
    Testing different attributes of 2D scatter plot on a vDataFrame
    """

    @pytest.fixture(autouse=True)
    def data(self, dummy_scatter_vd):
        """
        Load test data
        """
        self.data = dummy_scatter_vd

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
            self.data.scatter,
            {"columns": [COL_NAME_1, COL_NAME_2]},
        )

    def test_properties_all_unique_values_for_by(self, dummy_scatter_vd):
        """
        Test if all unique valies are inside the plot
        """
        # Arrange
        # Act
        result = dummy_scatter_vd.scatter(
            [
                COL_NAME_2,
                COL_NAME_3,
            ],
            by=COL_NAME_4,
        )
        # Assert
        assert len(result.data_temp) == len(
            all_categories
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

    @pytest.mark.parametrize("attributes", [[COL_NAME_3, 50, 2], [None, 1000, 4]])
    def test_properties_output_type_for_all_options(
        self,
        dummy_scatter_vd,
        plotting_library_object,
        attributes,
    ):
        """
        Test different sizes, number of points and max_cardinality
        """
        # Arrange
        # Act
        size, max_nb_points, max_cardinality = attributes
        result = dummy_scatter_vd.scatter(
            [COL_NAME_1, COL_NAME_2],
            size=size,
            max_nb_points=max_nb_points,
            max_cardinality=max_cardinality,
        )
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"


class TestHighchartsScatterVDF3DPlot(BasicPlotTests):
    """
    Testing different attributes of 3D scatter plot on a vDataFrame
    """

    @pytest.fixture(autouse=True)
    def data(self, dummy_scatter_vd):
        """
        Load test data
        """
        self.data = dummy_scatter_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [COL_NAME_1, COL_NAME_2, COL_NAME_3]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data.scatter,
            {"columns": [COL_NAME_1, COL_NAME_2, COL_NAME_3], "by": COL_NAME_4},
        )

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
            all_categories
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
