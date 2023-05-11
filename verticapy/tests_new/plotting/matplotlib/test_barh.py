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

# Vertica
from verticapy.tests_new.plotting.conftest import get_xaxis_label, get_yaxis_label

# Testing variables
col_name = "check 2"


@pytest.fixture(scope="class")
def plot_result(dummy_vd):
    return dummy_vd[col_name].barh()


class TestMatplotlibBarhPlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result

    def test_properties_output_type(self, matplotlib_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, matplotlib_figure_object), "wrong object crated"

    def test_data_sum_equals_one(self):
        # Arrange
        # Act
        # Assert - Comparing total adds up to 1
        assert sum([self.result.patches[i].get_width() for i in range(0, 3)]) == 1

    def test_data_ratios(self, dummy_vd):
        ### Checking if the density was plotted correctly
        nums = dummy_vd.to_pandas()[col_name].value_counts()
        total = len(dummy_vd)
        assert set([self.result.patches[i].get_width() for i in range(0, 3)]).issubset(
            set([nums["A"] / total, nums["B"] / total, nums["C"] / total])
        )

    def test_properties_xaxis_label(self):
        # Arrange
        test_title = "density"
        # Act
        # Assert - checking x axis label
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_label(self):
        # Arrange
        test_title = col_name
        # Act
        # Assert - checking y axis label
        assert get_yaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_xaxis_category(self):
        # Arrange
        # Act
        # Assert
        assert self.result.yaxis.get_scale() == "linear"

    def test_all_categories_created(self):
        assert set(
            [self.result.get_yticklabels()[i].get_text() for i in range(3)]
        ).issubset(set(["A", "B", "C"]))

    def test_additional_options_custom_width_and_height(self, dummy_vd):
        # Arrange
        custom_width = 3
        custom_height = 4
        # Act
        result = dummy_vd[col_name].bar(
            width=custom_width,
            height=custom_height,
        )
        # Assert - checking if correct object created
        assert (
            result.get_figure().get_size_inches()[0] == custom_width
            and result.get_figure().get_size_inches()[1] == custom_height
        ), "Custom width or height not working"
