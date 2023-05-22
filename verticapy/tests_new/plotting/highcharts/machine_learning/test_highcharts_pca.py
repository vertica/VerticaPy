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
from verticapy.learn.decomposition import PCA
from verticapy.tests_new.plotting.conftest import (
    get_xaxis_label,
    get_yaxis_label,
    get_width,
    get_height,
)

# Testing variables
COL_NAME_1 = "X"
COL_NAME_2 = "Y"
COL_NAME_3 = "Z"


class TestHighchartsMachineLearningPCACirclePlot:
    """
    Testing different attributes of PCA circle plot
    """

    @pytest.fixture(scope="class")
    def plot_result(self, dummy_scatter_vd):
        """
        Create a PCA circle plot
        """
        model = PCA("pca_circle_test")
        model.drop()
        model.fit(dummy_scatter_vd[COL_NAME_1, COL_NAME_2, COL_NAME_3])
        return model.plot_circle()

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

    def test_properties_xaxis_label(self):
        """
        Testing x-axis label
        """
        # Arrange
        test_title = "Dim1"
        # Act
        # Assert
        assert test_title in get_xaxis_label(self.result), "X axis label incorrect"

    def test_properties_yaxis_label(self):
        """
        Testing y-axis title
        """
        # Arrange
        test_title = "Dim2"
        # Act
        # Assert
        assert test_title in get_yaxis_label(self.result), "Y axis label incorrect"

    def test_additional_options_custom_height(self, dummy_scatter_vd):
        """
        Test custom width and height
        """
        # rrange
        custom_height = 6
        custom_width = 7
        # Act
        model = PCA("pca_circle_test")
        model.drop()
        model.fit(dummy_scatter_vd[COL_NAME_1, COL_NAME_2, COL_NAME_3])
        result = model.plot_circle(height=custom_height, width=custom_width)
        # Assert
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"
