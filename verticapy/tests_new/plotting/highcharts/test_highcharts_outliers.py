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
# Pytest
import pytest

# Standard Python Modules


# Other Modules


# Vertica
from ..conftest import BasicPlotTests


# Testing variables
COL_NAME_1 = "0"
COL_NAME_2 = "1"


class TestHighchartsOutliersPlot(BasicPlotTests):
    """
    Testing different attributes of outliers plot on a vDataColumn
    """

    @pytest.fixture(autouse=True)
    def data(self, dummy_dist_vd):
        """
        Load test data
        """
        self.data = dummy_dist_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [COL_NAME_1, ""]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data.outliers_plot,
            {"columns": COL_NAME_1},
        )

    def test_data_all_scatter_points_for_1d(
        self,
        dummy_dist_vd,
    ):
        """
        Testing to make sure all poitns are plotted
        """
        # Arrange
        total_points = len(dummy_dist_vd[COL_NAME_1])
        # Act
        result = dummy_dist_vd.outliers_plot(columns=[COL_NAME_1], max_nb_points=10000)
        plot_points_count = sum(len(result.data_temp[i].data) for i in range(1, 3))
        assert (
            plot_points_count == total_points
        ), "All points are not plotted for 1d plot"


class TestHighchartsOutliersPlot2D(BasicPlotTests):
    """
    Testing different attributes of outliers plot on a vDataFrame
    """

    @pytest.fixture(autouse=True)
    def data(self, dummy_dist_vd):
        """
        Load test data
        """
        self.data = dummy_dist_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [COL_NAME_1, COL_NAME_2]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data.outliers_plot,
            {"columns": [COL_NAME_1, COL_NAME_2]},
        )
