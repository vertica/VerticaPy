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
    VDCBarhPlot,
    VDFBarhPlot,
    VDFBarhPlot2D,
    get_yaxis_label,
    get_xaxis_label,
)


class TestPlotlyVDCBarhPlot(VDCBarhPlot):
    """
    Testing different attributes of HHorizontal Bar plot on a vDataColumn
    """

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return ["density", self.COL_NAME]

    def test_data_sum_equals_one(self):
        """
        Test if all ratios add up to 1
        """
        # Arrange
        # Act
        # Assert - Comparing total adds up to 1
        assert sum(self.result.data[0]["x"]) == 1

    def test_xaxis_category(self):
        """
        Test type of x-axis
        """
        # Arrange
        # Act
        # Assert
        assert self.result.layout["yaxis"]["type"] == "category"

    def test_data_ratios(self, dummy_vd):
        """
        Test data ratio plotted
        """
        ### Checking if the density was plotted correctly
        nums = dummy_vd.to_pandas()[self.COL_NAME].value_counts()
        total = len(dummy_vd)
        assert set(self.result.data[0]["x"]).issubset(
            set([nums["A"] / total, nums["B"] / total, nums["C"] / total])
        )

    def test_all_categories_crated(self):
        """
        Test all categories
        """
        assert set(self.result.data[0]["y"]).issubset(set(["A", "B", "C"]))

    def test_additional_options_custom_x_axis_title(self, dummy_vd):
        """
        Test custom x-axis title
        """
        # Arrange
        # Act
        result = dummy_vd[self.COL_NAME].barh(xaxis_title="Custom X Axis Title")
        # Assert
        assert (
            get_xaxis_label(result) == "Custom X Axis Title"
        ), "Custom X axis title not working"

    def test_additional_options_custom_y_axis_title(self, dummy_vd):
        """
        Test custom y-axis title
        """
        # Arrange
        # Act
        result = dummy_vd[self.COL_NAME].barh(yaxis_title="Custom Y Axis Title")
        # Assert
        assert (
            get_yaxis_label(result) == "Custom Y Axis Title"
        ), "Custom Y axis title not working"


class TestPlotlyVDFBarhPlot(VDFBarhPlot):
    """
    Testing different attributes of HHorizontal Bar plot on a vDataFrame
    """

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return ["density", self.COL_NAME_VDF_1]

    def test_data_ratios(self, dummy_dist_vd):
        """
        Test all ratios
        """
        ### Checking if the density was plotted correctly
        nums = dummy_dist_vd.to_pandas()[self.COL_NAME_VDF_1].value_counts()
        total = len(dummy_dist_vd)
        assert set(self.result.data[0]["x"]).issubset(
            set([nums["A"] / total, nums["B"] / total, nums["C"] / total])
        )

    def test_all_categories_created(self):
        """
        Test if all categories exist in plot
        """
        assert set(self.result.data[0]["y"]).issubset(set(["A", "B", "C"]))


class TestPlotlyVDFBarhPlot2D(VDFBarhPlot2D):
    """
    Testing different attributes of HHorizontal Bar plot on a vDataFrame
    """

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return ["density", self.COL_NAME_VDF_1]

    def test_stacked_bar_type(self, dummy_dist_vd):
        """
        Test bar type
        """
        result = dummy_dist_vd.barh(
            [self.COL_NAME_VDF_1, self.COL_NAME_VDF_2], kind="stacked"
        )
        assert result.layout["barmode"] == "stack"

    def test_fully_stacked_bar_type(self, dummy_dist_vd):
        """
        Test bar type
        """
        result = dummy_dist_vd.barh(
            [self.COL_NAME_VDF_1, self.COL_NAME_VDF_2], kind="fully"
        )
        assert result.layout["barmode"] == "stack"

    def test_fully_stacked_value(self, dummy_dist_vd):
        """
        Test sum of data equal to 1 for every bar
        """
        result = dummy_dist_vd.barh(
            [self.COL_NAME_VDF_1, self.COL_NAME_VDF_2], kind="fully"
        )
        for i in range(len(result.data[0]["x"])):
            assert result.data[0]["x"][i] + result.data[1]["x"][i]

    def test_negative_bar_type(self, dummy_dist_vd):
        """
        Test bar type
        """
        result = dummy_dist_vd.barh(
            [self.COL_NAME_VDF_1, self.COL_NAME_VDF_2], kind="pyramid"
        )
        assert result.layout["barmode"] == "relative"
