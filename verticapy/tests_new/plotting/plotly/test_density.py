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
col_name = "0"
by_col = "binary"


@pytest.fixture(scope="class")
def plot_result(dummy_dist_vd):
    return dummy_dist_vd[col_name].density()


@pytest.fixture(scope="class")
def plot_result_multiplot(dummy_dist_vd):
    return dummy_dist_vd[col_name].density(by=by_col)


class TestVDFDensityPlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result

    @pytest.fixture(autouse=True)
    def result_2(self, plot_result_multiplot):
        self.multi_plot_result = plot_result_multiplot

    def test_properties_output_type(self, plotly_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result) == plotly_figure_object, "wrong object created"

    def test_properties_output_type_for_multiplot(self, plotly_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert (
            type(self.multi_plot_result) == plotly_figure_object
        ), "wrong object crated"

    # ToDO - Change below after quotation bug fixed
    def test_properties_x_axis_title(
        self,
    ):
        # Arrange
        # Act
        # Assert -
        assert (
            self.result.layout["xaxis"]["title"]["text"] == '"0"'
        ), "X axis title incorrect"

    def test_properties_y_axis_title(
        self,
    ):
        # Arrange
        # Act
        # Assert
        assert (
            self.result.layout["yaxis"]["title"]["text"] == "density"
        ), "Y axis title incorrect"

    def test_properties_multiple_plots_produced_for_multiplot(
        self,
    ):
        # Arrange
        number_of_plots = 2
        # Act
        # Assert
        assert (
            len(self.multi_plot_result.data) == number_of_plots
        ), "Two plots not produced for two classes"

    def test_data_x_axis_range(self, dummy_dist_vd):
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
        # Arrange
        custom_width = 700
        custom_height = 700
        # Act
        result = dummy_dist_vd["0"].density(width=custom_width, height=custom_height)
        # Assert
        assert (
            result.layout["width"] == custom_width
            and result.layout["height"] == custom_height
        ), "Custom width or height not working"
