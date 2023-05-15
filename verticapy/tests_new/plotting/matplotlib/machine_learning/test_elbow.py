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
from verticapy.learn.model_selection import elbow
from verticapy.tests_new.plotting.conftest import get_xaxis_label

# Testing variables
col_name_1 = "PetalLengthCm"
col_name_2 = "PetalWidthCm"


@pytest.fixture(scope="class")
def plot_result(iris_vd):
    return elbow(input_relation=iris_vd, X=[col_name_1, col_name_2])


class TestMachineLearningElbowCurve:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result

    def test_properties_output_type(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_xaxis_label(self):
        # Arrange
        test_title = "Number of Clusters"
        # Act
        # Assert - checking if correct object created
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    @pytest.mark.slow
    @pytest.mark.notcritical
    def test_additional_options_custom_height(self, iris_vd):
        # rrange
        custom_height = 3
        custom_width = 3
        # Act
        result = elbow(
            input_relation=iris_vd,
            X=[col_name_1, col_name_2],
            width=custom_width,
            height=custom_height,
        )
        # Assert - checking if correct object created
        assert (
            result.get_figure().get_size_inches()[0] == custom_width
            and result.get_figure().get_size_inches()[1] == custom_height
        ), "Custom width or height not working"
