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


# Testing variables
COL_NAME_1 = "cats"
BY_COL = "binary"


class TestPlotlyVDFSpiderPlot:
    """
    Testing different attributes of Spider plot on a vDataColumn
    """

    @pytest.fixture(scope="class")
    def plot_result(self, dummy_dist_vd):
        """
        Create a spider plot for vDataColumn
        """
        return dummy_dist_vd[COL_NAME_1].spider()

    @pytest.fixture(scope="class")
    def plot_result_2(self, dummy_dist_vd):
        """
        Create a spider plot for vDataColumn using "by" parameter
        """
        return dummy_dist_vd[COL_NAME_1].spider(by=BY_COL)

    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        """
        Get the plot results
        """
        self.result = plot_result

    @pytest.fixture(autouse=True)
    def by_result(self, plot_result_2):
        """
        Get the plot results
        """
        self.by_result = plot_result_2

    def test_properties_output_type(self, plotting_library_object):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_output_type_for_multiplot(
        self,
        plotting_library_object,
    ):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(
            self.by_result, plotting_library_object
        ), "Wrong object created"

    def test_properties_title(
        self,
    ):
        """
        Test plot title
        """
        # Arrange
        column_name = COL_NAME_1
        # Act
        # Assert -
        assert self.result.layout["title"]["text"] == column_name, "Title incorrect"

    def test_properties_method_title_at_bottom(
        self,
    ):
        """
        Test method title
        """
        # Arrange
        method_text = "(Method: Density)"
        # Act
        # Assert -
        assert (
            self.result.layout["annotations"][0]["text"] == method_text
        ), "Method title incorrect"

    def test_properties_multiple_plots_produced_for_multiplot(
        self,
    ):
        """
        Test if multiple plots produced
        """
        # Arrange
        number_of_plots = 2
        # Act
        # Assert
        assert (
            len(self.by_result.data) == number_of_plots
        ), "Two traces not produced for two classes of binary"

    def test_data_all_categories(self, dummy_dist_vd):
        """
        Test all categories
        """
        # Arrange
        no_of_category = dummy_dist_vd["cats"].nunique()
        # Act
        assert (
            self.result.data[0]["r"].shape[0] == no_of_category
        ), "The number of categories in the data differ from the plot"

    def test_additional_options_custom_width(self, dummy_dist_vd):
        """
        Test custom width and height
        """
        # Arrange
        custom_width = 700
        custom_height = 600
        # Act
        result = dummy_dist_vd["cats"].spider(width=custom_width, height=custom_height)
        # Assert
        assert (
            result.layout["width"] == custom_width
            and result.layout["height"] == custom_height
        ), "Custom width or height not working"
