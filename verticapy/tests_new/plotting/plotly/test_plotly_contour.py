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
from verticapy.tests_new.plotting.base_test_files import VDFContourPlot


class TestPlotlyVDFContourPlot(VDFContourPlot):
    """
    Testing different attributes of Contour plot on a vDataFrame
    """

    def test_data_count_xaxis_default_bins(
        self,
    ):
        """
        Testing default bins
        """
        # Arrange
        # Act
        # Assert
        assert self.result.data[0]["x"].shape[0] == 100, "The default bins are not 100."

    def test_data_count_xaxis_custom_bins(self, dummy_dist_vd):
        """
        Test different bin sizes
        """
        # Arrange
        custom_bins = 200

        def func(param_a, param_b):
            return param_b + param_a * 0

        # Act
        result = dummy_dist_vd.contour(
            columns=[self.COL_NAME_1, self.COL_NAME_2], nbins=custom_bins, func=func
        )
        # Assert
        assert (
            result.data[0]["x"].shape[0] == custom_bins
        ), "The custom bins option is not working."

    def test_data_x_axis_range(self, dummy_dist_vd):
        """
        Test x-axis range
        """
        # Arrange
        x_min = dummy_dist_vd[self.COL_NAME_1].min()
        x_max = dummy_dist_vd[self.COL_NAME_1].max()
        # Act
        # Assert
        assert (
            self.result.data[0]["x"].min() == x_min
            and self.result.data[0]["x"].max() == x_max
        ), "The range in data is not consistent with plot"
