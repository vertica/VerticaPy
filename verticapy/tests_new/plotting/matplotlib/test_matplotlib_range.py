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
from verticapy.tests_new.plotting.base_test_files import VDCRangeCurve, VDFRangeCurve


class TestMatplotlibVDCRangeCurve(VDCRangeCurve):
    """
    Testing different attributes of range curve plot on a vDataColumn
    """

    def test_data_x_axis(self, dummy_date_vd):
        """
        Test x-ticks
        """
        # Arrange
        test_set = set(dummy_date_vd.to_numpy()[:, 0])
        # Act
        assert test_set.issubset(self.result.get_xticks())


class TestMatplotlibVDFRangeCurve(VDFRangeCurve):
    """
    Testing different attributes of range curve plot on a vDataFrame
    """

    def test_data_x_axis(self, dummy_date_vd):
        """
        Test all unique values
        """
        # Arrange
        test_set = set(dummy_date_vd.to_numpy()[:, 0])
        # Act
        # Assert
        assert test_set.issubset(self.result.get_xticks())
