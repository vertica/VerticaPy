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


class TestPlotlyVDCRangeCurve(VDCRangeCurve):
    """
    Testing different attributes of range curve plot on a vDataColumn
    """

    def test_data_x_axis(self, dummy_date_vd):
        """
        Test data of x-axis
        """
        # Arrange
        test_set = set(dummy_date_vd.to_numpy()[:, 0])
        # Act
        result = dummy_date_vd[self.COL_NAME_1].range_plot(ts=self.TIME_COL)
        assert set(result.data[0]["x"]).issubset(
            test_set
        ), "There is descripancy between x axis values for the bounds"

    def test_data_x_axis_for_median(self, dummy_date_vd):
        """
        Test median value
        """
        # Arrange
        test_set = set(dummy_date_vd.to_numpy()[:, 0])
        # Act
        assert set(self.result.data[1]["x"]).issubset(
            test_set
        ), "There is descripancy between x axis values for the median"

    def test_additional_options_turn_off_median(self, dummy_date_vd):
        """
        Test when median is turned off
        """
        # Arrange
        # Act
        result = dummy_date_vd[self.COL_NAME_1].range_plot(
            ts=self.TIME_COL, plot_median=False
        )
        # Assert
        assert (
            len(result.data) == 1
        ), "Median is still showing even after it is turned off"

    def test_additional_options_turn_on_median(
        self,
    ):
        """
        Test when median is turned on
        """
        # Arrange
        # Act
        # Assert
        assert (
            len(self.result.data) > 1
        ), "Median is still showing even after it is turned off"


class TestPlotlyVDFRangeCurve(VDFRangeCurve):
    """
    Testing different attributes of range curve plot on a vDataFrame
    """

    def test_data_x_axis(self, dummy_date_vd):
        """
        Test data for x-axis
        """
        # Arrange
        test_set = set(dummy_date_vd.to_numpy()[:, 0])
        # Act
        assert set(self.result.data[0]["x"]).issubset(
            test_set
        ), "There is descripancy between x axis values for the bounds"
