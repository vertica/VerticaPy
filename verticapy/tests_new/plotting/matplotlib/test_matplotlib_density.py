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
    VDCDensityPlot,
    VDCDensityMultiPlot,
    VDFDensityPlot,
)


class TestMatplotlibVDCDensityPlot(VDCDensityPlot):
    """
    Testing different attributes of Density plot on a vDataColumn
    """


class TestHighchartVDCDensityMultiPlot(VDCDensityMultiPlot):
    """
    Testing different attributes of Multiple Density plots on a vDataColumn
    """

    def test_properties_multiple_plots_produced_for_multiplot(
        self,
    ):
        """
        Test if two plots created
        """
        # Arrange
        number_of_plots = 2
        # Act
        # Assert
        assert (
            len(self.result.lines) == number_of_plots
        ), "Two plots not produced for two classes"


class TestHighchartsVDFDensityPlot(VDFDensityPlot):
    """
    Testing different attributes of Density plot on a vDataFrame
    """
