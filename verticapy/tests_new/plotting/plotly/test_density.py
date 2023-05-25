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

# Vertica
from verticapy.tests_new.plotting.conftest import (
    get_xaxis_label,
    get_yaxis_label,
    get_width,
    get_height,
)

# Standard Python Modules


# Other Modules


# Testing variables
COL_NAME = "0"
BY_COL = "binary"


class TestPlotlyVDCDensityPlot:
    """
    Testing different attributes of Density plot on a vDataColumn
    """

    @pytest.fixture(scope="class")
    def plot_result(self, dummy_dist_vd):
        """
        Create a density plot for vDataColumn
        """
        return dummy_dist_vd[COL_NAME].density()

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

    def test_properties_x_axis_title(
        self,
    ):
        """
        Testing x-axis title
        """
        # Arrange
        test_title = f'"{COL_NAME}"'
        # Act
        # Assert -
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_y_axis_title(
        self,
    ):
        """
        Testing y-axis title
        """
        # Arrange
        test_title = "density"
        # Act
        # Assert
        assert get_yaxis_label(self.result) == test_title, "Y axis label incorrect"

    def test_data_x_axis_range(self, dummy_dist_vd):
        """
        Test x-axis range
        """
        # Arrange
        x_min = dummy_dist_vd["0"].min()
        x_max = dummy_dist_vd["0"].max()

        # Act
        assert pytest.approx(self.result.data[0]["x"].min(), 4) == pytest.approx(
            x_min, 4
        ) and pytest.approx(self.result.data[0]["x"].max(), 4) == pytest.approx(
            x_max, 4
        ), "The range in data is not consistent with plot"

    def test_additional_options_custom_width(self, dummy_dist_vd):
        """
        Testing custom width and height
        """
        # Arrange
        custom_width = 700
        custom_height = 700
        # Act
        result = dummy_dist_vd["0"].density(width=custom_width, height=custom_height)
        # Assert
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"

    @pytest.mark.parametrize("nbins", [10, 20])
    @pytest.mark.parametrize("kernel", ["logistic", "sigmoid", "silverman"])
    def test_properties_output_type_for_all_options(
        self, dummy_dist_vd, plotting_library_object, nbins, kernel
    ):
        """
        Test different bin sizes and kernel types
        """
        # Arrange
        # Act
        result = dummy_dist_vd[COL_NAME].density(kernel=kernel, nbins=nbins)
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"


@pytest.mark.skip("Error in this highchart plot")
class TestHighchartVDCDensityMultiPlot:
    """
    Testing different attributes of Multiple Density plots on a vDataColumn
    """

    @pytest.fixture(scope="class")
    def plot_result_multiplot(self, dummy_dist_vd):
        """
        Create a multi-density plot for vDataColumn
        """
        return dummy_dist_vd[COL_NAME].density(by=BY_COL)

    @pytest.fixture(autouse=True)
    def result(self, plot_result_multiplot):
        """
        Get the plot results
        """
        self.result = plot_result_multiplot

    def test_properties_output_type_for_multiplot(self, plotting_library_object):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "wrong object crated"

    def test_properties_multiple_plots_produced_for_multiplot(
        self,
    ):
        """
        Test if two plots created
        """
        # Arrange
        number_of_plots = 2
        # Act
        # Assert
        assert (
            len(self.result.lines) == number_of_plots
        ), "Two plots not produced for two classes"


class TestPlotlyVDFDensityPlot:
    """
    Testing different attributes of Density plot on a vDataFrame
    """

    @pytest.fixture(scope="class")
    def plot_result_vdf(self, dummy_dist_vd):
        """
        Create a density plot for vDataFrame
        """
        return dummy_dist_vd.density(columns=[COL_NAME])

    @pytest.fixture(autouse=True)
    def result(self, plot_result_vdf):
        """
        Get the plot results
        """
        self.result = plot_result_vdf

    def test_properties_output_type(self, plotting_library_object):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_xaxis_title(self):
        """
        Testing x-axis title
        """
        # Arrange
        test_title = f'"{COL_NAME}"'
        # Act
        # Assert - checking x axis label
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_title(self):
        """
        Testing y-axis title
        """
        # Arrange
        test_title = "density"
        # Act
        # Assert - checking y axis label
        assert get_yaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_additional_options_custom_width_and_height(self, dummy_dist_vd):
        """
        Testing custom width and height
        """
        # Arrange
        custom_width = 300
        custom_height = 400
        # Act
        result = dummy_dist_vd.density(
            [COL_NAME], width=custom_width, height=custom_height
        )
        # Assert
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"

    @pytest.mark.parametrize("nbins", [10, 20])
    @pytest.mark.parametrize("kernel", ["logistic", "sigmoid", "silverman"])
    def test_properties_output_type_for_all_options(
        self, dummy_dist_vd, plotting_library_object, nbins, kernel
    ):
        """
        Test different bin sizes and kernel types
        """
        # Arrange
        # Act
        result = dummy_dist_vd["0"].density(kernel=kernel, nbins=nbins)
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"
