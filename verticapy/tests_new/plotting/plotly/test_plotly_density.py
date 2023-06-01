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


# Vertica
from verticapy.tests_new.plotting.base_test_files import (
    VDCDensityPlot,
    VDCDensityMultiPlot,
    VDFDensityPlot,
)


class TestPlotlyVDCDensityPlot(VDCDensityPlot):
    """
    Testing different attributes of Density plot on a vDataColumn
    """

    def test_data_x_axis_range(self, dummy_dist_vd):
        """
        Test x-axis range
        """
        # Arrange
        x_min = dummy_dist_vd["0"].min()
        x_max = dummy_dist_vd["0"].max()

        # Act
        assert pytest.approx(self.result.data[0]["x"].min(), 4) == pytest.approx(
            x_min, 4
        ) and pytest.approx(self.result.data[0]["x"].max(), 4) == pytest.approx(
            x_max, 4
        ), "The range in data is not consistent with plot"

    @pytest.mark.skip(reason="Need to remove extra quootation marks")
    def test_properties_xaxis_label(self):
        """
        Testing x-axis title
        """


class TestplotlyVDCDensityMultiPlot(VDCDensityMultiPlot):
    """
    Testing different attributes of Multiple Density plots on a vDataColumn
    """

    @pytest.mark.skip(reason="Need to remove extra quootation marks")
    def test_properties_xaxis_label(self):
        """
        Testing x-axis title
        """


class TestPlotlyVDFDensityPlot(VDFDensityPlot):
    """
    Testing different attributes of Density plot on a vDataFrame
    """

    @pytest.mark.skip(reason="Need to remove extra quootation marks")
    def test_properties_xaxis_label(self):
        """
        Testing x-axis title
        """
