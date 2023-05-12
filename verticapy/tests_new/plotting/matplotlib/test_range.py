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
time_col = "date"
col_name_1 = "value"


@pytest.fixture(scope="class")
def plotting_library_object(matplotlib_figure_object):
    return matplotlib_figure_object


@pytest.fixture(scope="class")
def plot_result(dummy_date_vd):
    return dummy_date_vd[col_name_1].range_plot(ts=time_col, plot_median=True)


class TestMatplotlibRangeCurve:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result

    def test_properties_output_type(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_xaxis(
        self,
    ):
        # Arrange
        test_title = time_col
        # Act
        # Assert -
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis(
        self,
    ):
        # Arrange
        test_title = col_name_1
        # Act
        # Assert -
        assert get_yaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_data_x_axis(self, dummy_date_vd):
        # Arrange
        test_set = set(dummy_date_vd.to_numpy()[:, 0])
        # Act
        assert test_set.issubset(self.result.get_xticks())

    def test_additional_options_custom_width_and_height(self, dummy_date_vd):
        # Arrange
        custom_width = 700
        custom_height = 700
        # Act
        result = dummy_date_vd[col_name_1].range_plot(
            ts=time_col, width=custom_width, height=custom_height
        )
        # Assert
        assert (
            result.get_figure().get_size_inches()[0] == custom_width
            and result.get_figure().get_size_inches()[1] == custom_height
        ), "Custom width or height not working"

    @pytest.mark.parametrize("plot_median", ["True", "False"])
    @pytest.mark.parametrize("start_date", [1920])
    @pytest.mark.parametrize("end_date", [1950])
    def test_properties_output_type_for_all_options(
        self,
        dummy_date_vd,
        plotting_library_object,
        plot_median,
        start_date,
        end_date,
    ):
        # Arrange
        # Act
        result = dummy_date_vd[col_name_1].range_plot(
            ts=time_col,
            plot_median=plot_median,
            start_date=start_date,
            end_date=end_date,
        )
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"
