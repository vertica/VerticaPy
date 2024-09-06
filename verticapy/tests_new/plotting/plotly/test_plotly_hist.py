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
from verticapy.tests_new.plotting.base_test_files import (
    VDCHistogramPlot,
    VDFHistogramPlot,
    VDFHistogramMultiPlot,
)


class TestPlotlyVDCHistogramPlot(VDCHistogramPlot):
    """
    Testing different attributes of Histogram plot on a vDataColumn
    """

    def test_properties_no_of_elements(self):
        """
        Test if all elements plotted
        """
        # Arrange
        total_items = 1
        # Act
        # Assert
        assert len(self.result.data) == pytest.approx(
            total_items, abs=1
        ), "Some elements missing"


class TestPlotlyVDFHistogramPlot(VDFHistogramPlot):
    """
    Testing different attributes of Histogram plot on a vDataFrame
    """


class TestPlotlyVDFHistogramMultiPlot(VDFHistogramMultiPlot):
    """
    Testing different attributes of Multi-Histogram plot on a vDataFrame
    """
