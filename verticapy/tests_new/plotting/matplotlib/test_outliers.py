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
COL_NAME_1 = "0"
COL_NAME_2 = "1"


class TestMatplotlibVDFOutliersPlot:
    """
    Testing different attributes of outliers plot on a vDataColumn
    """

    @pytest.fixture(scope="class")
    def plot_result(self, dummy_dist_vd):
        """
        Create an outlier plot for vDataColumn
        """
        return dummy_dist_vd.outliers_plot(columns=[COL_NAME_1])

    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        """
        Get the plot results
        """
        self.result = plot_result

    def test_properties_output_type_for_1d(self, plotting_library_object):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_xaxis_for_1d(
        self,
    ):
        """
        Testing x-axis label
        """
        # Arrange
        test_title = COL_NAME_1
        # Act
        # Assert -
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_data_all_scatter_points_for_1d(
        self,
        dummy_dist_vd,
    ):
        """
        Testing to make sure all poitns are plotted
        """
        # Arrange
        total_points = len(dummy_dist_vd[COL_NAME_1])
        # Act
        result = dummy_dist_vd.outliers_plot(columns=[COL_NAME_1], max_nb_points=10000)
        plot_points_count = sum(
            len(result.get_children()[i].get_offsets()) for i in range(10, 12)
        )
        assert (
            plot_points_count == total_points
        ), "All points are not plotted for 1d plot"

    def test_additional_options_custom_width_and_height(self, dummy_dist_vd):
        """
        Testing custom width and height
        """
        # Arrange
        custom_width = 3
        custom_height = 3
        # Act
        result = dummy_dist_vd.outliers_plot(
            columns=[COL_NAME_1, COL_NAME_2], width=custom_width, height=custom_height
        )
        # Assert
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"


class TestMatplotlibOutliersPlot2D:
    """
    Testing different attributes of outliers plot on a vDataFrame
    """

    @pytest.fixture(scope="class")
    def plot_result_2d(self, dummy_dist_vd):
        """
        Create an outlier plot for vDataFrame
        """
        return dummy_dist_vd.outliers_plot(columns=[COL_NAME_1, COL_NAME_2])

    @pytest.fixture(autouse=True)
    def result(self, plot_result_2d):
        """
        Get the plot results
        """
        self.result = plot_result_2d

    def test_properties_output_type(self, plotting_library_object):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_xaxis_label(
        self,
    ):
        """
        Testing x-axis label
        """
        # Arrange
        test_title = COL_NAME_1
        # Act
        # Assert -
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_label(
        self,
    ):
        """
        Testing y-axis label
        """
        # Arrange
        test_title = COL_NAME_2
        # Act
        # Assert -
        assert get_yaxis_label(self.result) == test_title, "X axis label incorrect"
