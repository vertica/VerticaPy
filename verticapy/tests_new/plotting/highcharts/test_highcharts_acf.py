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


# Vertica
from verticapy.tests_new.plotting.conftest import (
    get_xaxis_label,
    get_width,
    get_height,
)

# Other Modules


class TestHighchartsVDFACFPlot:
    """
    Testing different attributes of ACF plot on a vDataFrame
    """

    @pytest.fixture(scope="class")
    def acf_plot_fixture(self, amazon_vd):
        """
        Create an ACF plot
        """
        return amazon_vd.acf(
            ts="date",
            column="number",
            p=12,
            by=["state"],
            unit="month",
            method="spearman",
        )

    @pytest.fixture(autouse=True)
    def result(self, acf_plot_fixture):
        """
        Get the plot results
        """
        self.result = acf_plot_fixture

    def test_properties_output_type(self, plotting_library_object):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "wrong object crated"

    def test_properties_xaxis_label(self):
        """
        Testing x-axis label
        """
        # Arrange
        test_title = "lag"
        # Act
        # Assert - checking x axis label
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_vertical_lines_for_custom_lag(self, amazon_vd):
        """
        Test to check number of vertical lines
        """
        # Arrange
        lag_number = 24
        # Act
        result = amazon_vd.acf(
            ts="date",
            column="number",
            p=lag_number - 1,
            by=["state"],
            unit="month",
            method="spearman",
        )
        # Assert
        assert (
            len(result.data_temp[0].data) == lag_number
        ), "Number of vertical lines inconsistent"

    def test_data_all_scatter_points(
        self,
    ):
        """
        Test to check if all points plotted
        """
        # Arrange
        lag_number = 13
        # Act
        # Assert
        assert (
            len(self.result.data_temp[0].data) == lag_number
        ), "Number of lag points inconsistent"

    def test_additional_options_custom_width_and_height(self, amazon_vd):
        """
        Test if custom width and height working
        """
        # Arrange
        custom_width = 700
        custom_height = 700
        # Act
        result = amazon_vd.acf(
            ts="date",
            column="number",
            p=12,
            by=["state"],
            unit="month",
            method="spearman",
            width=custom_width,
            height=custom_height,
        )
        # Assert - checking if correct object created
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"
