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
from verticapy.tests_new.plotting.base_test_files import OutliersPlot2D, OutliersPlot


class TestHighchartsOutliersPlot(OutliersPlot):
    """
    Testing different attributes of outliers plot on a vDataColumn
    """

    def test_data_all_scatter_points_for_1d(
        self,
        dummy_dist_vd,
    ):
        """
        Testing to make sure all poitns are plotted
        """
        # Arrange
        total_points = len(dummy_dist_vd[self.COL_NAME_1])
        # Act
        result = dummy_dist_vd.outliers_plot(
            columns=[self.COL_NAME_1], max_nb_points=10000
        )
        plot_points_count = sum(len(result.data_temp[i].data) for i in range(1, 3))
        assert (
            plot_points_count == total_points
        ), "All points are not plotted for 1d plot"


class TestHighchartsOutliersPlot2D(OutliersPlot2D):
    """
    Testing different attributes of outliers plot on a vDataFrame
    """
