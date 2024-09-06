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
from verticapy.tests_new.plotting.base_test_files import (
    VDCBarPlot,
    VDFBarPlot,
    VDFBarPlot2D,
)


class TestPlotlyVDCBarPlot(VDCBarPlot):
    """
    Testing different attributes of Bar plot on a vDataColumn
    """

    def test_data_ratios(self, dummy_vd):
        """
        Test data ratio plotted for bar chart
        """
        ### Checking if the density was plotted correctly
        nums = dummy_vd.to_pandas()[self.COL_NAME].value_counts()
        total = len(dummy_vd)
        assert set(self.result.data[0]["y"]).issubset(
            set([nums["A"] / total, nums["B"] / total, nums["C"] / total])
        )

    def test_all_categories_crated(self):
        """
        Test all categories
        """
        assert set(self.result.data[0]["x"]).issubset(set(["A", "B", "C"]))

    def test_xaxis_category(self):
        """
        Test type of x-axis
        """
        # Arrange
        # Act
        # Assert
        assert self.result.layout["xaxis"]["type"] == "category"

    def test_additional_options_custom_x_axis_title(self, dummy_vd):
        """
        Test custom x-axis title
        """
        # Arrange
        # Act
        result = dummy_vd[self.COL_NAME].bar(xaxis_title="Custom X Axis Title")
        # Assert
        assert result.layout["xaxis"]["title"]["text"] == "Custom X Axis Title"

    def test_additional_options_custom_y_axis_title(self, dummy_vd):
        """
        Test custom y-axis title
        """
        # Arrange
        # Act
        result = dummy_vd[self.COL_NAME].bar(yaxis_title="Custom Y Axis Title")
        # Assert
        assert result.layout["yaxis"]["title"]["text"] == "Custom Y Axis Title"


class TestPlotlyVDFBarPlot(VDFBarPlot):
    """
    Testing different attributes of Bar plot on a vDataFrame
    """

    def test_data_ratios(self, dummy_dist_vd):
        """
        Test data ratio
        """
        ### Checking if the density was plotted correctly
        nums = dummy_dist_vd.to_pandas()[self.COL_NAME_VDF_1].value_counts()
        total = len(dummy_dist_vd)
        assert set(self.result.data[0]["y"]).issubset(
            set([nums["A"] / total, nums["B"] / total, nums["C"] / total])
        )


class TestPlotlyVDFBarPlot2D(VDFBarPlot2D):
    """
    Testing different attributes of 2D Bar plot on a vDataFrame
    """

    def test_stacked_bar_type(self, dummy_dist_vd):
        """
        Test bar type
        """
        result = dummy_dist_vd.bar(
            [self.COL_NAME_VDF_1, self.COL_NAME_VDF_2], kind="stacked"
        )
        assert result.layout["barmode"] == "stack"
