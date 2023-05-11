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
col_name_1 = "binary"


@pytest.fixture(scope="class")
def plot_result(dummy_dist_vd):
    return dummy_dist_vd[col_name_1].hist()


class TestHistogram:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result

    def test_properties_output_type(self, plotly_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result) == plotly_figure_object, "Wrong object crated"

    def test_properties_xaxis_label(self):
        # Arrange
        test_title = col_name_1
        # Act
        # Assert
        assert (
            self.result.layout["xaxis"]["title"]["text"] == test_title
        ), "X axis label incorrect"

    def test_properties_yaxis_label(self):
        # Arrange
        test_title = "density"
        # Act
        # Assert
        assert (
            self.result.layout["yaxis"]["title"]["text"] == test_title
        ), "Y axis label incorrect"

    def test_properties_no_of_elements(self):
        # Arrange
        total_items = 1
        # Act
        # Assert
        assert len(self.result.data) == pytest.approx(
            total_items, abs=1
        ), "Some elements missing"

    def test_additional_options_custom_height(self, dummy_dist_vd):
        # rrange
        custom_height = 650
        custom_width = 700
        # Act

        result = dummy_dist_vd[col_name_1].hist(
            height=custom_height,
            width=custom_width,
        )
        # Assert
        assert (
            result.layout["height"] == custom_height
            and result.layout["width"] == custom_width
        ), "Custom height and width not working"
