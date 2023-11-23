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
# Vertica
from verticapy.tests_new.plotting.base_test_files import (
    VDCHistogramPlot,
    VDFHistogramPlot,
    VDFHistogramMultiPlot,
)


class TestMatplotlibVDCHistogramPlot(VDCHistogramPlot):
    """
    Testing different attributes of Histogram plot on a vDataColumn
    """


class TestMatplotlibVDFHistogramPlot(VDFHistogramPlot):
    """
    Testing different attributes of Histogram plot on a vDataFrame
    """


class TestMatplotlibVDFHistogramMultiPlot(VDFHistogramMultiPlot):
    """
    Testing different attributes of Multi-Histogram plot on a vDataFrame
    """

    def test_properties_xaxis_label(self):
        """
        Testing x-axis label
        """
        assert self.result.get_xlabel() == "", "X axis label incorrect"
