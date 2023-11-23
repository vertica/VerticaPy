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

from verticapy.tests_new.plotting.base_test_files import VDCLinePlot, VDFLinePlot


class TestHighchartsVDCLinePlot(VDCLinePlot):
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
            len(self.result.data_temp[0].data[0]) * len(self.result.data_temp[0].data)
            == total_count
        ), "The total values in the plot are not equal to the values in the dataframe."


class TestHighchartsVDFLinePlot(VDFLinePlot):
    """
    Testing different attributes of Line plot on a vDataFrame
    """
