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
from verticapy.learn.neighbors import LocalOutlierFactor
from verticapy.tests_new.plotting.conftest import (
    get_xaxis_label,
    get_yaxis_label,
)

# Testing variables
col_name_1 = "X"
col_name_2 = "Y"
col_name_3 = "Z"


@pytest.fixture(scope="class")
def plotting_library_object(matplotlib_figure_object):
    return matplotlib_figure_object


@pytest.fixture(scope="class")
def plot_result(dummy_scatter_vd):
    model = LocalOutlierFactor("lof_test")
    model.fit(dummy_scatter_vd, [col_name_1, col_name_2])
    return model.plot()


@pytest.fixture(scope="class")
def plot_result_2(dummy_scatter_vd):
    model = LocalOutlierFactor("lof_test_3d")
    model.fit(dummy_scatter_vd, [col_name_1, col_name_2, col_name_3])
    return model.plot()


class TestMatplotlibMachineLearningLOFPlot2D:
    @pytest.fixture(autouse=True)
    def result_2d(self, plot_result):
        self.result = plot_result

    def test_properties_output_type(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_xaxis_label(self):
        # Arrange
        test_title = col_name_1
        # Act
        # Assert
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_label(self):
        # Arrange
        test_title = col_name_2
        # Act
        # Assert
        assert get_yaxis_label(self.result) == test_title, "Y axis label incorrect"

    def test_additional_options_custom_height(self, dummy_scatter_vd):
        # rrange
        custom_height = 6
        custom_width = 7
        # Act
        model = LocalOutlierFactor("lof_test")
        model.fit(dummy_scatter_vd, [col_name_1, col_name_2])
        result = model.plot(height=custom_height, width=custom_width)
        # Assert
        assert (
            result.get_figure().get_size_inches()[0] == custom_width
            and result.get_figure().get_size_inches()[1] == custom_height
        ), "Custom width or height not working"


class TestMatplotlibMachineLearningLOFPlot3D:
    @pytest.fixture(autouse=True)
    def result_3d(self, plot_result_2):
        self.result = plot_result_2

    def test_properties_output_type_for_3d(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_xaxis_label_for_3d(self):
        # Arrange
        test_title = col_name_1
        # Act
        # Assert
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_label_for_3d(self):
        # Arrange
        test_title = col_name_2
        # Act
        # Assert
        assert get_yaxis_label(self.result) == test_title, "X axis label incorrect"
