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


# Vertica
from ..conftest import BasicPlotTests

# Other Modules


class TestHighchartsVDFACFPlot(BasicPlotTests):
    """
    Testing different attributes of ACF plot on a vDataFrame
    """

    @pytest.fixture(autouse=True)
    def data(self, amazon_vd):
        """
        Load test data
        """
        self.data = amazon_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return ["lag", "value"]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data.acf,
            {
                "ts": "date",
                "column": "number",
                "p": 12,
                "by": ["state"],
                "unit": "month",
                "method": "spearman",
            },
        )

    def test_properties_vertical_lines_for_custom_lag(self, amazon_vd):
        """
        Test to check number of vertical lines
        """
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
            len(result.data_temp[0].data) == lag_number
        ), "Number of vertical lines inconsistent"

    def test_data_all_scatter_points(
        self,
    ):
        """
        Test to check if all points plotted
        """
        # Arrange
        lag_number = 13
        # Act
        # Assert
        assert (
            len(self.result.data_temp[0].data) == lag_number
        ), "Number of lag points inconsistent"
