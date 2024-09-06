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

# Standard Python Modules


# Other Modules


# Vertica
from verticapy.tests_new.plotting.base_test_files import (
    get_xaxis_label,
    get_yaxis_label,
    get_width,
    get_height,
)

# Testing variables
COL_NAME_1 = "age"
COL_NAME_2 = "fare"
COL_OF = "survived"


class TestMatplotlibVDFHexbinPlot:
    """
    Testing different attributes of Hexbin plot on a vDataFrame
    """

    @pytest.fixture(scope="class")
    def plot_result(self, titanic_vd):
        """
        Create hexbin for vDataFrame
        """
        return titanic_vd.hexbin(columns=[COL_NAME_1, COL_NAME_2])

    @pytest.fixture(scope="class")
    def plot_result_of(self, titanic_vd):
        """
        Create hexbin for vDataFrame using OF option
        """
        return titanic_vd.hexbin(
            columns=[COL_NAME_1, COL_NAME_2], method="avg", of=COL_OF
        )

    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        """
        Get the plot results
        """
        self.result = plot_result

    @pytest.fixture(autouse=True)
    def pivot_result(self, plot_result_of):
        """
        Get the plot results
        """
        self.pivot_result = plot_result_of

    def test_properties_output_type(self, plotting_library_object):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_output_type_for_pivot_table(self, plotting_library_object):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(
            self.pivot_result, plotting_library_object
        ), "Wrong object created"

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

    def test_additional_options_custom_width_and_height(self, titanic_vd):
        """
        Testing custom width and height
        """
        # Arrange
        custom_width = 3
        custom_height = 4
        # Act
        result = titanic_vd.hexbin(
            columns=[COL_NAME_1, COL_NAME_2], width=custom_width, height=custom_height
        )
        # Assert
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"

    @pytest.mark.parametrize("method", ["count", "density", "max"])
    def test_properties_output_type_for_all_options(
        self, titanic_vd, plotting_library_object, method
    ):
        """
        Test different methods
        """
        # Arrange
        # Act
        result = titanic_vd.hexbin(
            columns=[COL_NAME_1, COL_NAME_2], method=method, of=COL_OF
        )
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"
