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

# Vertica
from verticapy.tests_new.plotting.base_test_files import VDCCandlestick
from vertica_highcharts.highstock.highstock import Highstock


# Testing variables
COL_NAME_1 = "values"
TIME_COL = "date"
COL_OF = "survived"
BY_COL = "category"


class TestHighChartsVDCCandlestick(VDCCandlestick):
    """
    Testing different attributes of Candlestick plot on a vDataColumn
    """

    def test_properties_output_type(self, plotting_library_object):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, Highstock), "Wrong object created"

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

    @pytest.mark.skip(reason="The plot does not have custom width and height yet")
    def test_additional_options_custom_width_and_height(
        self,
    ):
        """
        Testing custom width and height
        """
