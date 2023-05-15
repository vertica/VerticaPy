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


# Verticapy
from verticapy.learn.svm import LinearSVC
from verticapy.tests_new.plotting.conftest import (
    get_xaxis_label,
)

# Testing variables
col_name_1 = "X"
col_name_2 = "Y"
col_name_3 = "Z"
by_col = "Category"


@pytest.fixture(scope="class")
def plot_result(dummy_pred_data_vd):
    model = LinearSVC(name="public.SVC")
    model.fit(dummy_pred_data_vd, [col_name_1], by_col)
    return model.plot()


@pytest.fixture(scope="class")
def plot_result_2d(dummy_pred_data_vd):
    model = LinearSVC(name="public.SVC")
    model.fit(dummy_pred_data_vd, [col_name_1, col_name_2], by_col)
    return model.plot()


@pytest.fixture(scope="class")
def plot_result_3d(dummy_pred_data_vd):
    model = LinearSVC(name="public.SVC")
    model.fit(
        dummy_pred_data_vd,
        [col_name_1, col_name_2, col_name_3],
        by_col,
    )
    return model.plot()


class TestMachineLearningSVMClassifierPlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result, plot_result_2d, plot_result_3d):
        self.result = plot_result
        self.result_2d = plot_result_2d
        self.result_3d = plot_result_3d

    def test_properties_output_type_for_1d(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_output_typefor_2d(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_output_type_for_3d(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_xaxis_label(self):
        # Arrange
        test_title = col_name_1
        # Act
        # Assert
        assert get_xaxis_label(self.result) == test_title, "Y axis label incorrect"

    def test_additional_options_custom_height(self, dummy_pred_data_vd):
        # rrange
        custom_height = 6
        custom_width = 3
        model = LinearSVC(name="public.SVC")
        model.fit(dummy_pred_data_vd, [col_name_1], by_col)
        # Act
        result = model.plot(width=custom_width, height=custom_height)
        # Assert
        assert (
            result.get_figure().get_size_inches()[0] == custom_width
            and result.get_figure().get_size_inches()[1] == custom_height
        ), "Custom width or height not working"

    def test_additional_options_custom_height_for_2d(self, dummy_pred_data_vd):
        # rrange
        custom_height = 6
        custom_width = 7
        model = LinearSVC(name="public.SVC")
        model.fit(dummy_pred_data_vd, [col_name_1, col_name_2], by_col)
        # Act
        result = model.plot(width=custom_width, height=custom_height)
        # Assert
        assert (
            result.get_figure().get_size_inches()[0] == custom_width
            and result.get_figure().get_size_inches()[1] == custom_height
        ), "Custom width or height not working"
