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
    get_zaxis_label,
    get_width,
    get_height,
)


# Testing variables
COL_NAME_1 = "X"
COL_NAME_2 = "Y"
COL_NAME_3 = "Z"
COL_NAME_4 = "Category"
all_categories = ["A", "B", "C"]


class TestHighchartsScatterVDF2DPlot:
    """
    Testing different attributes of 2D scatter plot on a vDataFrame
    """

    @pytest.fixture(scope="class")
    def plot_result(self, dummy_scatter_vd):
        """
        Create a 2D scatter plot for vDataFrame
        """
        return dummy_scatter_vd.scatter([COL_NAME_1, COL_NAME_2])

    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        """
        Get the plot results
        """
        self.result = plot_result

    def test_properties_output_type(self, plotting_library_object):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_xaxis_title(
        self,
    ):
        """
        Testing x-axis label
        """
        # Arrange
        test_title = COL_NAME_1
        # Act
        # Assert
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_title(
        self,
    ):
        """
        Testing y-axis label
        """
        # Arrange
        test_title = COL_NAME_2
        # Act
        # Assert
        assert get_yaxis_label(self.result) == test_title, "Y axis label incorrect"

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

    def test_additional_options_custom_width_and_height(self, dummy_scatter_vd):
        """
        Testing custom width and height
        """
        # Arrange
        custom_width = 3
        custom_height = 4
        # Act
        result = dummy_scatter_vd.scatter(
            [COL_NAME_1, COL_NAME_2],
            width=custom_width,
            height=custom_height,
        )
        # Assert - checking if correct object created
        # Assert
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"

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


class TestHighchartsScatterVDF3DPlot:
    """
    Testing different attributes of 3D scatter plot on a vDataFrame
    """

    @pytest.fixture(scope="class")
    def plot_result_2(self, dummy_scatter_vd):
        """
        Create a 3D scatter plot for vDataFrame
        """
        return dummy_scatter_vd.scatter([COL_NAME_1, COL_NAME_2, COL_NAME_3])

    @pytest.fixture(scope="class")
    def plot_result_3(self, dummy_scatter_vd):
        """
        Create a 3D scatter plot for vDataFrame using "by" option
        """
        result = dummy_scatter_vd.scatter(
            [
                COL_NAME_1,
                COL_NAME_2,
                COL_NAME_3,
            ],
            by=COL_NAME_4,
        )
        return result

    @pytest.fixture(autouse=True)
    def result(self, plot_result_2):
        """
        Get the plot results
        """
        self.result = plot_result_2

    def test_properties_output_type(self, plotting_library_object):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_xaxis_title_3d_plot(
        self,
    ):
        """
        Testing x-axis label
        """
        # Arrange
        test_title = COL_NAME_1
        # Act
        # Assert
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_title_3d_plot(
        self,
    ):
        """
        Testing y-axis label
        """
        # Arrange
        test_title = COL_NAME_2
        # Act
        # Assert
        assert (
            get_yaxis_label(self.result) == test_title
        ), "Y axis label incorrect in 3D plot"

    def test_properties_zaxis_title_3d_plot(
        self,
    ):
        """
        Testing z-axis label
        """
        # Arrange
        test_title = COL_NAME_3
        # Act
        # Assert
        assert (
            get_zaxis_label(self.result) == test_title
        ), "Z axis label incorrect in 3D plot"

    def test_properties_all_unique_values_for_by_3d_plot(self, plot_result_3):
        """
        Test if all unique values plotted
        """
        # Arrange
        # Act
        result = plot_result_3
        # Assert
        # Assert
        assert len(result.data_temp) == len(
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
