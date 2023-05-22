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
from verticapy.tests_new.plotting.conftest import (
    get_xaxis_label,
    get_yaxis_label,
    get_width,
    get_height,
)

# Testing variables
TIME_COL = "date"
COL_NAME_1 = "value"


class TestHighchartsVDCRangeCurve:
    """
    Testing different attributes of range curve plot on a vDataColumn
    """

    @pytest.fixture(scope="class")
    def plot_result(self, dummy_date_vd):
        """
        Create a range curve plot for vDataColumn
        """
        return dummy_date_vd[COL_NAME_1].range_plot(ts=TIME_COL, plot_median=True)

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

    def test_properties_xaxis_label(
        self,
    ):
        """
        Testing x-axis label
        """
        # Arrange
        test_title = TIME_COL
        # Act
        # Assert -
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_label(
        self,
    ):
        """
        Testing y-axis label
        """
        # Arrange
        test_title = COL_NAME_1
        # Act
        # Assert -
        assert get_yaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_data_x_axis(self, dummy_date_vd):
        """
        Test all unique values
        """
        # Arrange
        test_set = set(dummy_date_vd.to_numpy()[:, 0])
        # Act
        assert test_set.issubset(
            [
                self.result.data_temp[0].data[i][0]
                for i in range(len(self.result.data_temp[0].data))
            ]
        )

    def test_additional_options_custom_width_and_height(self, dummy_date_vd):
        """
        Test custom width and height
        """
        # Arrange
        custom_width = 700
        custom_height = 700
        # Act
        result = dummy_date_vd[COL_NAME_1].range_plot(
            ts=TIME_COL, width=custom_width, height=custom_height
        )
        # Assert
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"

    @pytest.mark.parametrize(
        "plot_median, date_range",
        [("True", [1920, None]), ("False", [None, 1950])],
    )
    def test_properties_output_type_for_all_options(
        self,
        dummy_date_vd,
        plotting_library_object,
        plot_median,
        date_range,
    ):
        """
        Test different values for median, start date, and end date
        """
        # Arrange
        # Act
        result = dummy_date_vd[COL_NAME_1].range_plot(
            ts=TIME_COL,
            plot_median=plot_median,
            start_date=date_range[0],
            end_date=date_range[1],
        )
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"


class TestHighchartsVDFRangeCurve:
    """
    Testing different attributes of range curve plot on a vDataFrame
    """

    @pytest.fixture(scope="class")
    def plot_result_vdf(self, dummy_date_vd):
        """
        Create a range curve plot for vDataFrame
        """
        return dummy_date_vd.range_plot(
            columns=[COL_NAME_1], ts=TIME_COL, plot_median=True
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

    def test_properties_xaxis_label(
        self,
    ):
        """
        Testing x-axis label
        """
        # Arrange
        test_title = TIME_COL
        # Act
        # Assert -
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_label(
        self,
    ):
        """
        Testing y-axis label
        """
        # Arrange
        test_title = COL_NAME_1
        # Act
        # Assert -
        assert get_yaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_data_x_axis(self, dummy_date_vd):
        """
        Test all unique values
        """
        # Arrange
        test_set = set(dummy_date_vd.to_numpy()[:, 0])
        # Act
        assert test_set.issubset(
            [
                self.result.data_temp[0].data[i][0]
                for i in range(len(self.result.data_temp[0].data))
            ]
        )

    def test_additional_options_custom_width_and_height(self, dummy_date_vd):
        """
        Test custom width and height
        """
        # Arrange
        custom_width = 700
        custom_height = 700
        # Act
        result = dummy_date_vd.range_plot(
            columns=[COL_NAME_1], ts=TIME_COL, width=custom_width, height=custom_height
        )
        # Assert
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"

    @pytest.mark.parametrize(
        "plot_median, date_range",
        [("True", [1920, None]), ("False", [None, 1950])],
    )
    def test_properties_output_type_for_all_options(
        self,
        dummy_date_vd,
        plotting_library_object,
        plot_median,
        date_range,
    ):
        """
        Tes different values for median, start date, and end date
        """
        # Arrange
        # Act
        result = dummy_date_vd.range_plot(
            columns=[COL_NAME_1],
            ts=TIME_COL,
            plot_median=plot_median,
            start_date=date_range[0],
            end_date=date_range[1],
        )
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"
