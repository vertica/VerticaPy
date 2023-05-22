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
from vertica_highcharts.highstock.highstock import Highstock

# Testing variables
COL_NAME_1 = "values"
TIME_COL = "date"
COL_OF = "survived"
BY_COL = "category"


class TestHighChartsVDCCandlestick:
    """
    Testing different attributes of Candlestick plot on a vDataColumn
    """

    @pytest.fixture(scope="class")
    def plot_result(self, dummy_line_data_vd):
        """
        Create a candlestick plot for vDataColumn
        """
        return dummy_line_data_vd[COL_NAME_1].candlestick(ts=TIME_COL)

    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        """
        Get the plot results
        """
        self.result = plot_result

    def test_properties_output_type(self):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, Highstock), "Wrong object created"

    def test_additional_options_custom_width_and_height(self, dummy_line_data_vd):
        """
        Testing custom width and height
        """
        # Arrange
        custom_width = 3
        custom_height = 4
        # Act
        result = dummy_line_data_vd[COL_NAME_1].candlestick(
            ts=TIME_COL, width=custom_width, height=custom_height
        )
        # Assert
        assert (
            result.options["chart"].width == custom_width
            and result.options["chart"].height == custom_height
        ), "Custom width or height not working"

    @pytest.mark.parametrize(
        "method, start_date", [("count", 1910), ("density", 1920), ("max", 1920)]
    )
    def test_properties_output_type_for_all_options(
        self, dummy_line_data_vd, method, start_date
    ):
        """
        Test "method" and "start date" parameters
        """
        # Arrange
        # Act
        result = dummy_line_data_vd[COL_NAME_1].candlestick(
            ts=TIME_COL, method=method, start_date=start_date
        )
        # Assert - checking if correct object created
        assert isinstance(result, Highstock), "Wrong object created"
