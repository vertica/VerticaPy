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


class TestMatplotlibVDCPiePlot(VDCPiePlot):
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
        assert len(self.result.patches) > 1

    def test_properties_labels(self, dummy_vd):
        """
        Test if all unique values grouped
        """
        # Arrange
        # Act
        # Assert - check value corresponding to 0s
        assert set(
            self.result.get_legend().get_texts()[i].get_text()
            for i in range(len(self.result.get_legend().get_texts()))
        ) == set(dummy_vd.to_pandas()[self.COL_NAME].unique())


class TestMatplotlibNestedVDFPiePlot(NestedVDFPiePlot):
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
        # Act
        # Assert - check value corresponding to 0s
        assert len(self.result.patches) > 2
