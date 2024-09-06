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
from verticapy.tests_new.plotting.base_test_files import (
    ScatterVDF2DPlot,
    ScatterVDF3DPlot,
)


class TestPlotlyScatterVDF2DPlot(ScatterVDF2DPlot):
    """
    Testing different attributes of 2D scatter plot on a vDataFrame
    """

    def test_properties_all_unique_values_for_by(self, dummy_scatter_vd):
        """
        Test if all unique valies are inside the plot
        """
        # Arrange
        # Act
        result = dummy_scatter_vd.scatter(
            [
                self.COL_NAME_2,
                self.COL_NAME_3,
            ],
            by=self.COL_NAME_4,
        )
        # Assert
        assert set(
            [result.data[0]["name"], result.data[1]["name"], result.data[2]["name"]]
        ).issubset(
            set(self.all_categories)
        ), "Some unique values were not found in the plot"

    def test_properties_colors_for_by(self, dummy_scatter_vd):
        """
        Test if there are different colors for each category
        """
        # Arrange
        # Act
        result = dummy_scatter_vd.scatter(
            [
                self.COL_NAME_2,
                self.COL_NAME_3,
            ],
            by=self.COL_NAME_4,
        )
        assert (
            len(
                set(
                    [
                        result.data[0]["marker"]["color"],
                        result.data[1]["marker"]["color"],
                        result.data[2]["marker"]["color"],
                    ]
                )
            )
            == 3
        ), "Colors are not unique for three different cat_col parameter"

    def test_data_total_number_of_points(self, dummy_scatter_vd):
        """
        Test if all datapoints were plotted
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert len(self.result.data[0]["x"]) == len(
            dummy_scatter_vd
        ), "Number of points not consistent with data"
        assert len(self.result.data[0]["y"]) == len(
            dummy_scatter_vd
        ), "Number of points not consistent with data"

    def test_data_random_point_from_plot_in_data(self, dummy_scatter_vd):
        """
        Test spot check
        """
        # Arrange
        # Act
        # Assert -
        len_of_data = len(
            dummy_scatter_vd.search(
                conditions=[
                    f"{self.COL_NAME_1} = \
                    {self.result.data[0]['x'][0]} and \
                    {self.COL_NAME_2}={self.result.data[0]['y'][0]}"
                ],
                usecols=[self.COL_NAME_1, self.COL_NAME_2],
            )
        )
        assert len_of_data > 0, "A wrong point was plotted"


class TestPlotlyScatterVDF3DPlot(ScatterVDF3DPlot):
    """
    Testing different attributes of 3D scatter plot on a vDataFrame
    """

    def test_properties_xaxis_label(
        self,
    ):
        """
        Testing x-axis label
        """
        # Arrange
        # Act
        # Assert
        assert (
            self.result.layout["scene"]["xaxis"]["title"]["text"] == self.COL_NAME_1
        ), "X-axis title issue in 3D plot"

    def test_properties_yaxis_label(
        self,
    ):
        """
        Testing y-axis label
        """
        # Arrange
        # Act
        # Assert
        assert (
            self.result.layout["scene"]["yaxis"]["title"]["text"] == self.COL_NAME_2
        ), "Y-axis title issue in 3D plot"

    def test_properties_zaxis_label(
        self,
    ):
        """
        Testing z-axis label
        """
        # Arrange
        # Act
        # Assert
        assert (
            self.result.layout["scene"]["zaxis"]["title"]["text"] == self.COL_NAME_3
        ), "Z-axis title issue in 3D plot"

    def test_properties_all_unique_values_for_by_3d_plot(self):
        """
        Test if all unique values plotted
        """
        # Assert
        assert set(
            [
                self.result.data[0]["name"],
                self.result.data[1]["name"],
                self.result.data[2]["name"],
            ]
        ).issubset(
            set(self.all_categories)
        ), "Some unique values were not found in the 3D plot"

    def test_properties_colors_for_by_3d_plot(self):
        """
        Test if unique colors for each plot
        """
        # Assert
        assert (
            len(
                set(
                    [
                        self.result.data[0]["marker"]["color"],
                        self.result.data[1]["marker"]["color"],
                        self.result.data[2]["marker"]["color"],
                    ]
                )
            )
            == 3
        ), "Colors are not unique for three different cat_col parameter"

    @pytest.mark.skip(reason="Custom width not apt for 3D graphs")
    def test_additional_options_custom_width_and_height(self):
        """
        Testing custom width and height
        """
