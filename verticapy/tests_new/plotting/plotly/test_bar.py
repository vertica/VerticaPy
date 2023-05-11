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
col_name = "check 2"


@pytest.fixture(scope="class")
def plot_result(dummy_vd):
    return dummy_vd[col_name].bar()


class TestBarPlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result

    def test_properties_output_type_for(self, plotly_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result) == plotly_figure_object, "wrong object crated"

    def test_data_ratios(self, dummy_vd):
        ### Checking if the density was plotted correctly
        nums = dummy_vd.to_pandas()[col_name].value_counts()
        total = len(dummy_vd)
        assert set(self.result.data[0]["y"]).issubset(
            set([nums["A"] / total, nums["B"] / total, nums["C"] / total])
        )

    def test_properties_xaxis_label(self):
        # Arrange
        test_title = col_name
        # Act
        # Assert - checking x axis label
        assert (
            self.result.layout["xaxis"]["title"]["text"] == test_title
        ), "X axis label incorrect"

    def test_properties_yaxis_label(self):
        # Arrange
        test_title = "density"
        # Act
        # Assert - checking y axis label
        assert (
            self.result.layout["yaxis"]["title"]["text"] == test_title
        ), "X axis label incorrect"

    def test_xaxis_category(self):
        # Arrange
        # Act
        # Assert
        assert self.result.layout["xaxis"]["type"] == "category"

    def test_additional_options_custom_width_and_height(self, dummy_vd):
        # Arrange
        custom_width = 300
        custom_height = 400
        # Act
        result = dummy_vd[col_name].bar(
            width=custom_width,
            height=custom_height,
        )
        # Assert - checking if correct object created
        assert (
            result.layout["width"] == custom_width
            and result.layout["height"] == custom_height
        ), "Custom width or height not working"

    def test_additional_options_custom_x_axis_title(self, dummy_vd):
        # Arrange
        # Act
        result = dummy_vd[col_name].bar(xaxis_title="Custom X Axis Title")
        # Assert
        assert result.layout["xaxis"]["title"]["text"] == "Custom X Axis Title"

    def test_additional_options_custom_y_axis_title(self, dummy_vd):
        # Arrange
        # Act
        result = dummy_vd[col_name].bar(yaxis_title="Custom Y Axis Title")
        # Assert
        assert result.layout["yaxis"]["title"]["text"] == "Custom Y Axis Title"
