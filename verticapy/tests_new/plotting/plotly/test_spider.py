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
import numpy as np

# Testing variables
col_name_1 = "cats"
by_col = "binary"


@pytest.fixture(scope="class")
def plot_result(dummy_dist_vd):
    return dummy_dist_vd[col_name_1].spider()


@pytest.fixture(scope="class")
def plot_result_2(dummy_dist_vd):
    return dummy_dist_vd[col_name_1].spider(by=by_col)


class TestVDFSpiderPlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result

    @pytest.fixture(autouse=True)
    def result_2(self, plot_result_2):
        self.by_result = plot_result_2

    def test_properties_output_type(self, plotly_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result) == plotly_figure_object, "wrong object created"

    def test_properties_output_type_for_multiplot(
        self, plotly_figure_object, dummy_dist_vd
    ):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.by_result) == plotly_figure_object, "wrong object created"

    def test_properties_title(self, load_plotly, dummy_dist_vd):
        # Arrange
        column_name = col_name_1
        # Act
        # Assert -
        assert self.result.layout["title"]["text"] == column_name, "Title incorrect"

    def test_properties_method_title_at_bottom(self, load_plotly, dummy_dist_vd):
        # Arrange
        method_text = "(Method: Density)"
        # Act
        # Assert -
        assert (
            self.result.layout["annotations"][0]["text"] == method_text
        ), "Method title incorrect"

    def test_properties_multiple_plots_produced_for_multiplot(
        self, load_plotly, dummy_dist_vd
    ):
        # Arrange
        number_of_plots = 2
        # Act
        # Assert
        assert (
            len(self.by_result.data) == number_of_plots
        ), "Two traces not produced for two classes of binary"

    def test_data_all_categories(self, dummy_dist_vd):
        # Arrange
        no_of_category = dummy_dist_vd["cats"].nunique()
        # Act
        assert (
            self.result.data[0]["r"].shape[0] == no_of_category
        ), "The number of categories in the data differ from the plot"

    def test_additional_options_custom_width(self, load_plotly, dummy_dist_vd):
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
