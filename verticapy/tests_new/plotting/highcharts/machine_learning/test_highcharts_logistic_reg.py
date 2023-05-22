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
from verticapy.learn.linear_model import LogisticRegression
from verticapy.tests_new.plotting.conftest import (
    get_xaxis_label,
    get_yaxis_label,
    get_zaxis_label,
    get_width,
    get_height,
)

# Testing variables
COL_NAME_1 = "fare"
COL_NAME_2 = "survived"
COL_NAME_3 = "age"


class TestHighchartsMachineLearningLogisticRegressionPlot2D:
    """
    Testing different attributes of 2D Logisti Regression plot
    """

    @pytest.fixture(scope="class")
    def plot_result(self, titanic_vd):
        """
        Create a logistic regression plot
        """
        model = LogisticRegression("log_reg_test")
        model.fit(titanic_vd, [COL_NAME_1], COL_NAME_2)
        return model.plot()

    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        """
        Get the plot results
        """
        self.result = plot_result

    def test_properties_output_type_for_2d(self, plotting_library_object):
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
        test_title = COL_NAME_1
        # Act
        # Assert
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_label(self):
        """
        Testing y-axis title
        """
        # Arrange
        test_title = f"p({COL_NAME_2}=1)"
        # Act
        # Assert
        assert get_yaxis_label(self.result) == test_title, "Y axis label incorrect"

    def test_additional_options_custom_height(self, titanic_vd):
        """
        Test custom width and height
        """
        # rrange
        custom_height = 6
        custom_width = 7
        # Act
        model = LogisticRegression("log_reg_test")
        model.fit(titanic_vd, [COL_NAME_1], COL_NAME_2)
        result = model.plot(height=custom_height, width=custom_width)
        # Assert
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"

    @pytest.mark.parametrize("max_nb_points", [50])
    def test_properties_output_type_for_all_options(
        self,
        titanic_vd,
        plotting_library_object,
        max_nb_points,
    ):
        """
        Test different number of maximum points
        """
        # Arrange
        # Act
        model = LogisticRegression("log_reg_test")
        model.fit(titanic_vd, [COL_NAME_1], COL_NAME_2)
        result = model.plot(
            max_nb_points=max_nb_points,
        )
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"


@pytest.mark.skip(reason="Currently highchart only supports 2D plot")
class TestHighchartsMachineLearningLogisticRegressionPlot3D:
    """
    Testing different attributes of 3D Logisti Regression plot
    """

    @pytest.fixture(scope="class")
    def plot_result_2(self, titanic_vd):
        """
        Create a 3D logistic regression plot
        """
        model = LogisticRegression("log_reg_test")
        model.fit(titanic_vd, [COL_NAME_1, COL_NAME_3], COL_NAME_2)
        return model.plot()

    @pytest.fixture(autouse=True)
    def result(self, plot_result_2):
        """
        Get the plot results
        """
        self.result = plot_result_2

    def test_properties_output_type(self, plotting_library_object):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_xaxis_label_for_3d(self):
        """
        Testing x-axis label
        """
        # Arrange
        test_title = COL_NAME_1
        # Act
        # Assert
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_label_for_3d(self):
        """
        Testing y-axis label
        """
        # Arrange
        test_title = COL_NAME_3
        # Act
        # Assert
        assert get_yaxis_label(self.result) == test_title, "Y axis label incorrect"

    def test_properties_zaxis_label_for_3d(self):
        """
        Testing z-axis label
        """
        # Arrange
        test_title = COL_NAME_2
        # Act
        # Assert
        assert get_zaxis_label(self.result) == test_title, "X axis label incorrect"
