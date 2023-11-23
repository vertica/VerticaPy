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
from verticapy.tests_new.plotting.base_test_files import VDCPiePlot, NestedVDFPiePlot


class TestHighchartsVDCPiePlot(VDCPiePlot):
    """
    Testing different attributes of Pie plot on a vDataColumn
    """

    def test_plot_type_wedges(
        self,
    ):
        """
        Test if multiple sections of pie plot is created
        """
        # Arrange
        # Act
        # Assert - check value corresponding to 0s
        assert len(self.result.data_temp[0].data) > 1


class TestHighchartsNestedVDFPiePlot(NestedVDFPiePlot):
    """
    Testing different attributes of Pie plot on a vDataFrame
    """

    def test_plot_type_wedges(
        self,
    ):
        """
        Test if nested plots are produced
        """
        # Arrange
        all_elements_count = sum(len(item.data) for item in self.result.data_temp)
        # Act
        # Assert - check value corresponding to 0s
        assert all_elements_count > 2
