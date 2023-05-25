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
# Standard Python Modules
import random

# Pytest
import pytest


# Vertica
from verticapy.tests_new.plotting.conftest import (
    get_xaxis_label,
    get_yaxis_label,
    get_width,
    get_height,
)

# Other Modules


# Testing variables
TIME_COL = "date"
COL_NAME_1 = "values"
COL_NAME_2 = "category"
CAT_OPTION = "A"


class TestPlotlyVDCLinePlot:
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

    def test_properties_output_type(self, plotting_library_object):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_output_type_for_one_trace(
        self, dummy_line_data_vd, plotting_library_object
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
        assert isinstance(result, plotting_library_object), "Wrong object created"

    def test_properties_x_axis_title(
        self,
    ):
        """
        Testing x-axis label
        """
        # Arrange
        test_tile = "time"
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
            self.result.data[0]["x"].shape[0] + self.result.data[1]["x"].shape[0]
            == total_count
        ), "The total values in the plot are not equal to the values in the dataframe."

    def test_data_spot_check(self, dummy_line_data_vd):
        """
        Spot check one data point
        """
        # Arrange
        # Act
        assert (
            str(
                dummy_line_data_vd[TIME_COL][
                    random.randint(0, len(dummy_line_data_vd)) - 1
                ]
            )
            in self.result.data[0]["x"]
        ), "Two time values that exist in the data do not exist in the plot"

    def test_additional_options_custom_width_and_height(self, dummy_line_data_vd):
        """
        Testing custom width and height
        """
        # Arrange
        custom_width = 400
        custom_height = 600
        # Act
        result = dummy_line_data_vd[COL_NAME_1].plot(
            ts=TIME_COL, by=COL_NAME_2, width=custom_width, height=custom_height
        )
        # Assert - checking if correct object created
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"

    def test_additional_options_marker_on(self, dummy_line_data_vd):
        """
        Test marker option
        """
        # Arrange
        # Act
        result = dummy_line_data_vd[COL_NAME_1].plot(
            ts=TIME_COL, by=COL_NAME_2, markers=True
        )
        # Assert - checking if correct object created
        assert set(result.data[0]["mode"]) == set(
            "lines+markers"
        ), "Markers not turned on"

    @pytest.mark.parametrize(
        "kind, start_date", [("spline", "1930"), ("area", None), ("step", None)]
    )
    # @pytest.mark.parametrize("start_date", ["1930"])
    def test_properties_output_type_for_all_options(
        self, dummy_line_data_vd, plotting_library_object, start_date, kind
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
        assert isinstance(result, plotting_library_object), "Wrong object created"


class TestHighchartsVDFLinePlot:
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
