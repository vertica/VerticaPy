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
# Pytest
import pytest

# Vertica
from verticapy.tests_new.plotting.base_test_files import OutliersPlot2D, OutliersPlot


class TestPlotlyOutliersPlot(OutliersPlot):
    """
    Testing different attributes of outliers plot on a vDataColumn
    """

    @pytest.mark.skip(reason="Y axis label not applied to plotly plot")
    def test_properties_yaxis_label(self):
        """
        Testing y-axis title
        """

    def test_properties_xaxis_label(self):
        """
        Testing x-axis title
        """
        # Arrange
        column_name = self.COL_NAME_1
        # Act
        # Assert -
        assert self.result.data[0]["x"][0] == column_name, "X axis label incorrect"

    def test_data_all_scatter_points_for_1d(
        self,
        dummy_dist_vd,
    ):
        """
        Test if all points are plotted
        """
        # Arrange
        total_points = len(dummy_dist_vd[self.COL_NAME_1])
        # Act
        result = dummy_dist_vd.outliers_plot(
            columns=[self.COL_NAME_1], max_nb_points=10000
        )
        plot_points_count = sum(data["y"].shape[0] for data in result.data)
        assert (
            plot_points_count == total_points
        ), "All points are not plotted for 1d plot"


class TestPlotlyOutliersPlot2D(OutliersPlot2D):
    """
    Testing different attributes of outliers plot on a vDataFrame
    """

    def test_data_all_scatter_points_for_2d(self, dummy_dist_vd):
        """
        Test if all points ar eplotted
        """
        # Arrange
        total_points = len(dummy_dist_vd[self.COL_NAME_1])
        # Act
        result = dummy_dist_vd.outliers_plot(
            columns=[self.COL_NAME_1, self.COL_NAME_2], max_nb_points=10000
        )
        assert result.data[-1]["y"].shape[0] + result.data[-2]["y"].shape[
            0
        ] == pytest.approx(
            total_points, abs=1
        ), "All points are not plotted for 2d plot"

    def test_data_all_information_plotted_for_2d(
        self,
    ):
        """
        Test if all four elements plotted
        """
        # Arrange
        total_elements = 4
        # Act
        assert (
            len(self.result.data) == total_elements
        ), "The total number of elements plotted is not correct"
