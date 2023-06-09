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
# Vertica

from verticapy.tests_new.plotting.base_test_files import VDCLinePlot, VDFLinePlot


class TestMatplotlibVDCLinePlot(VDCLinePlot):
    """
    Testing different attributes of Line plot on a vDataColumn
    """

    def test_data_count_of_all_values(self, dummy_line_data_vd):
        """
        Testing total points
        """
        # Arrange
        total_count = dummy_line_data_vd.shape()[0]
        # Act
        assert (
            sum(len(line.get_xdata()) for line in self.result.get_lines())
            == total_count
        ), "The total values in the plot are not equal to the values in the dataframe."


class TestMatplotlibVDFLinePlot(VDFLinePlot):
    """
    Testing different attributes of Line plot on a vDataFrame
    """

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [self.TIME_COL, ""]
