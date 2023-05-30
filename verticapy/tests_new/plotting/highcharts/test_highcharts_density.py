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
COL_NAME = "0"
BY_COL = "binary"


class TestHighchartsVDCDensityPlot(BasicPlotTests):
    """
    Testing different attributes of Density plot on a vDataColumn
    """

    @pytest.fixture(autouse=True)
    def data(self, dummy_dist_vd):
        """
        Load test data
        """
        self.data = dummy_dist_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [COL_NAME, "density"]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data[COL_NAME].density,
            {},
        )

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
class TestHighchartVDCDensityMultiPlot(BasicPlotTests):
    """
    Testing different attributes of Multiple Density plots on a vDataColumn
    """

    @pytest.fixture(autouse=True)
    def data(self, dummy_dist_vd):
        """
        Load test data
        """
        self.data = dummy_dist_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [COL_NAME, "density"]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data[COL_NAME].density,
            {"by": BY_COL},
        )

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


class TestHighchartsVDFDensityPlot:
    """
    Testing different attributes of Density plot on a vDataFrame
    """

    @pytest.fixture(autouse=True)
    def data(self, dummy_dist_vd):
        """
        Load test data
        """
        self.data = dummy_dist_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [COL_NAME, "density"]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data.density,
            {"columns": COL_NAME},
        )

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
