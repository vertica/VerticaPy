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

# Vertica
from verticapy.tests_new.plotting.conftest import (
    get_xaxis_label,
    get_yaxis_label,
    get_width,
    get_height,
)

# Other Modules


# Testing variables
COL_NAME = "check 2"
COL_NAME_VDF_1 = "cats"
COL_NAME_VDF_OF = "0"


class TestPlotlyVDCBarPlot:
    """
    Testing different attributes of Bar plot on a vDataColumn
    """

    @pytest.fixture(scope="class")
    def plot_result(self, dummy_vd):
        """
        Create a bar plot for vDataColumn
        """
        return dummy_vd[COL_NAME].bar()

    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        """
        Get the plot results
        """
        self.result = plot_result

    def test_data_ratios(self, dummy_vd):
        """
        Test data ratio plotted for bar chart
        """
        ### Checking if the density was plotted correctly
        nums = dummy_vd.to_pandas()[COL_NAME].value_counts()
        total = len(dummy_vd)
        assert set(self.result.data[0]["y"]).issubset(
            set([nums["A"] / total, nums["B"] / total, nums["C"] / total])
        )

    def test_properties_xaxis_label(self):
        """
        Testing x-axis label
        """
        # Arrange
        test_title = COL_NAME
        # Act
        # Assert - checking x axis label
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_label(self):
        """
        Testing y-axis title
        """
        # Arrange
        test_title = "density"
        # Act
        # Assert - checking y axis label
        assert get_yaxis_label(self.result) == test_title, "Y axis label incorrect"

    def test_xaxis_category(self):
        """
        Test type of x-axis
        """
        # Arrange
        # Act
        # Assert
        assert self.result.layout["xaxis"]["type"] == "category"

    def test_additional_options_custom_width_and_height(self, dummy_vd):
        """
        Testing custom width and height
        """
        # Arrange
        custom_width = 300
        custom_height = 400
        # Act
        result = dummy_vd[COL_NAME].bar(
            width=custom_width,
            height=custom_height,
        )
        # Assert - checking if correct object created
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"

    def test_additional_options_custom_x_axis_title(self, dummy_vd):
        """
        Test custom x-axis title
        """
        # Arrange
        # Act
        result = dummy_vd[COL_NAME].bar(xaxis_title="Custom X Axis Title")
        # Assert
        assert result.layout["xaxis"]["title"]["text"] == "Custom X Axis Title"

    def test_additional_options_custom_y_axis_title(self, dummy_vd):
        """
        Test custom y-axis title
        """
        # Arrange
        # Act
        result = dummy_vd[COL_NAME].bar(yaxis_title="Custom Y Axis Title")
        # Assert
        assert result.layout["yaxis"]["title"]["text"] == "Custom Y Axis Title"


class TestPlotlyVDFBarPlot:
    """
    Testing different attributes of Bar plot on a vDataFrame
    """

    @pytest.fixture(scope="class")
    def plot_result_vdf(self, dummy_dist_vd):
        """
        Create a bar plot for vDataFrame
        """
        return dummy_dist_vd.bar(columns=[COL_NAME_VDF_1])

    @pytest.fixture(autouse=True)
    def result(self, plot_result_vdf):
        """
        Get the plot results
        """
        self.result = plot_result_vdf

    def test_properties_output_type(self, plotting_library_object):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "wrong object crated"

    def test_data_ratios(self, dummy_dist_vd):
        """
        Test data ratio
        """
        ### Checking if the density was plotted correctly
        nums = dummy_dist_vd.to_pandas()[COL_NAME_VDF_1].value_counts()
        total = len(dummy_dist_vd)
        assert set(self.result.data[0]["y"]).issubset(
            set([nums["A"] / total, nums["B"] / total, nums["C"] / total])
        )

    def test_properties_xaxis_label(self):
        """
        Testing x-axis label
        """
        # Arrange
        test_title = COL_NAME_VDF_1
        # Act
        # Assert - checking x axis label
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_label(self):
        """
        Testing y-axis title
        """
        # Arrange
        test_title = "density"
        # Act
        # Assert - checking y axis label
        assert get_yaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_additional_options_custom_width_and_height(self, dummy_dist_vd):
        """
        Test custom width and height
        """
        # Arrange
        custom_width = 300
        custom_height = 400
        # Act
        result = dummy_dist_vd.bar(
            columns=[COL_NAME_VDF_1],
            width=custom_width,
            height=custom_height,
        )
        # Assert - checking if correct object created
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"

    @pytest.mark.parametrize(
        "of_col, method", [(COL_NAME_VDF_OF, "min"), (COL_NAME_VDF_OF, "max")]
    )
    def test_properties_output_type_for_all_options(
        self,
        dummy_dist_vd,
        plotting_library_object,
        of_col,
        method,
    ):
        """
        Test of and method combination
        """
        # Arrange
        # Act
        result = dummy_dist_vd[COL_NAME_VDF_1].bar(
            of=of_col,
            method=method,
        )
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"
