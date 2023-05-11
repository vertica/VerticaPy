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


@pytest.fixture(scope="class")
def acf_plot_result(amazon_vd):
    return amazon_vd.acf(
        ts="date",
        column="number",
        p=12,
        by=["state"],
        unit="month",
        method="spearman",
    )


class TestACFPlot:
    @pytest.fixture(autouse=True)
    def result(self, acf_plot_result):
        self.result = acf_plot_result

    def test_properties_output_type(self, matplotlib_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, matplotlib_figure_object), "wrong object crated"

    def test_properties_xaxis_label(self):
        # Arrange
        test_title = "lag"
        # Act
        # Assert - checking x axis label
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_vertical_lines_for_custom_lag(self, amazon_vd):
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
            len(result.get_lines()[0].get_xdata()) - 2 == lag_number
        ), "Number of vertical lines inconsistent"

    def test_data_all_scatter_points(self, acf_plot_result):
        # Arrange
        lag_number = 13
        # Act
        # Assert
        assert (
            len(self.result.get_lines()[0].get_xdata()) - 2 == lag_number
        ), "Number of lag points inconsistent"

    def test_additional_options_custom_width_and_height(self, amazon_vd):
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
            result.get_figure().get_size_inches()[0] == custom_width
            and result.get_figure().get_size_inches()[1] == custom_height
        ), "Custom width or height not working"
