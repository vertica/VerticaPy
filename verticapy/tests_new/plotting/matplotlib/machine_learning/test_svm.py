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
COL_NAME_1 = "X"
COL_NAME_2 = "Y"
COL_NAME_3 = "Z"
BY_COL = "Category"


class TestMatplotlibMachineLearningSVMClassifierPlot:
    """
    Testing different attributes of SVM classifier plot
    """

    def __init__(self):
        self.result_2d = None
        self.result_3d = None

    @pytest.fixture(scope="class")
    def plot_result(self, dummy_pred_data_vd):
        """
        Create 1D SVM classifier plot
        """
        model = LinearSVC(name="public.SVC")
        model.fit(dummy_pred_data_vd, [COL_NAME_1], BY_COL)
        return model.plot()

    @pytest.fixture(scope="class")
    def plot_result_2d(self, dummy_pred_data_vd):
        """
        Create 2D SVM classifier plot
        """
        model = LinearSVC(name="public.SVC")
        model.fit(dummy_pred_data_vd, [COL_NAME_1, COL_NAME_2], BY_COL)
        return model.plot()

    @pytest.fixture(scope="class")
    def plot_result_3d(self, dummy_pred_data_vd):
        """
        Create 3D SVM classifier plot
        """
        model = LinearSVC(name="public.SVC")
        model.fit(
            dummy_pred_data_vd,
            [COL_NAME_1, COL_NAME_2, COL_NAME_3],
            BY_COL,
        )
        return model.plot()

    @pytest.fixture(autouse=True)
    def result(self, plot_result, plot_result_2d, plot_result_3d):
        """
        Get the plot results
        """
        self.result = plot_result
        self.result_2d = plot_result_2d
        self.result_3d = plot_result_3d

    def test_properties_output_type_for_1d(self, plotting_library_object):
        """
        Test if correct object created for 1D plot
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_output_typefor_2d(self, plotting_library_object):
        """
        Test if correct object created for 2D plot
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_output_type_for_3d(self, plotting_library_object):
        """
        Test if correct object created for 3D plot
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_xaxis_label(self):
        """
        Testing x-axis label
        """
        # Arrange
        test_title = COL_NAME_1
        # Act
        # Assert
        assert get_xaxis_label(self.result) == test_title, "Y axis label incorrect"

    def test_additional_options_custom_height(self, dummy_pred_data_vd):
        """
        Test custom width and height
        """
        # Arrange
        custom_height = 6
        custom_width = 3
        model = LinearSVC(name="public.SVC_4")
        model.fit(dummy_pred_data_vd, [COL_NAME_1], BY_COL)
        # Act
        result = model.plot(width=custom_width, height=custom_height)
        # Assert
        assert (
            result.get_figure().get_size_inches()[0] == custom_width
            and result.get_figure().get_size_inches()[1] == custom_height
        ), "Custom width or height not working"

    def test_additional_options_custom_height_for_2d(self, dummy_pred_data_vd):
        """
        Test custom width and height for 2D
        """
        # Arrange
        custom_height = 6
        custom_width = 7
        model = LinearSVC(name="public.SVC_5")
        model.fit(dummy_pred_data_vd, [COL_NAME_1, COL_NAME_2], BY_COL)
        # Act
        result = model.plot(width=custom_width, height=custom_height)
        # Assert
        assert (
            result.get_figure().get_size_inches()[0] == custom_width
            and result.get_figure().get_size_inches()[1] == custom_height
        ), "Custom width or height not working"
