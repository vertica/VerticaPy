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
from verticapy.tests_new.plotting.conftest import get_xaxis_label, get_yaxis_label

# Testing variables
TIME_COL = "date"
COL_NAME_1 = "values"
COL_NAME_2 = "category"
CAT_OPTION = "A"


class TestMatplotlibVDCLinePlot:
    """
    Testing different attributes of Line plot on a vDataColumn
    """

    @pytest.fixture(scope="class")
    def plot_result(self, dummy_line_data_vd):
        """
        Create a line plot for vDataColumn
        """
        return dummy_line_data_vd[COL_NAME_1].plot(ts=TIME_COL, by=COL_NAME_2)

    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        """
        Get the plot results
        """
        self.result = plot_result

    def test_properties_output_type(self, matplotlib_figure_object):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, matplotlib_figure_object), "Wrong object created"

    def test_properties_output_type_for_one_trace(
        self, dummy_line_data_vd, matplotlib_figure_object
    ):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        result = dummy_line_data_vd[dummy_line_data_vd[COL_NAME_2] == CAT_OPTION][
            COL_NAME_1
        ].plot(ts=TIME_COL)
        # Assert - checking if correct object created
        assert isinstance(result, matplotlib_figure_object), "Wrong object created"

    def test_properties_x_axis_title(
        self,
    ):
        """
        Testing x-axis label
        """
        # Arrange
        test_tile = "date"
        # Act
        # Assert - checking if correct object created
        assert get_xaxis_label(self.result) == test_tile, "X axis title incorrect"

    def test_properties_y_axis_title(
        self,
    ):
        """
        Testing y-axis label
        """
        # Arrange
        test_tile = COL_NAME_1
        # Act
        # Assert - checking if correct object created
        assert get_yaxis_label(self.result) == test_tile, "Y axis title incorrect"

    def test_data_count_of_all_values(self, dummy_line_data_vd):
        """
        Testing total points
        """
        # Arrange
        total_count = dummy_line_data_vd.shape()[0]
        # Act
        assert (
            sum(len(line.get_xdata()) for line in self.result.get_lines())
            == total_count
        ), "The total values in the plot are not equal to the values in the dataframe."

    def test_additional_options_custom_width_and_height(self, dummy_line_data_vd):
        """
        Testing custom width and height
        """
        # Arrange
        custom_width = 4
        custom_height = 6
        # Act
        result = dummy_line_data_vd[COL_NAME_1].plot(
            ts=TIME_COL, by=COL_NAME_2, width=custom_width, height=custom_height
        )
        # Assert - checking if correct object created
        assert (
            result.get_figure().get_size_inches()[0] == custom_width
            and result.get_figure().get_size_inches()[1] == custom_height
        ), "Custom width or height not working"

    @pytest.mark.parametrize("kind", ["spline", "area", "step"])
    @pytest.mark.parametrize("start_date", ["1930"])
    def test_properties_output_type_for_all_options(
        self, dummy_line_data_vd, matplotlib_figure_object, start_date, kind
    ):
        """
        Testing different kinds and start date
        """
        # Arrange
        # Act
        result = dummy_line_data_vd[COL_NAME_1].plot(
            ts=TIME_COL, kind=kind, start_date=start_date
        )
        # Assert - checking if correct object created
        assert isinstance(result, matplotlib_figure_object), "Wrong object created"


class TestMatplotlibVDFLinePlot:
    """
    Testing different attributes of Line plot on a vDataFrame
    """

    @pytest.fixture(scope="class")
    def plot_result_vdf(self, dummy_line_data_vd):
        """
        Create a line plot for vDataFrame
        """
        return dummy_line_data_vd[dummy_line_data_vd[COL_NAME_2] == CAT_OPTION].plot(
            ts=TIME_COL, columns=COL_NAME_1
        )

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
        assert isinstance(self.result, plotting_library_object), "Wrong object created"
