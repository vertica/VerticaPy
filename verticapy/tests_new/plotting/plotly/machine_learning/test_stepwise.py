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
from verticapy.learn.model_selection import stepwise
from verticapy.learn.linear_model import LogisticRegression
from verticapy.tests_new.plotting.conftest import (
    get_xaxis_label,
    get_yaxis_label,
)

# Testing variables
COL_NAME_1 = "age"
COL_NAME_2 = "fare"
COL_NAME_3 = "parch"
COL_NAME_4 = "pclass"
BY_COL = "survived"


class TestMachineLearningStepwisePlot:
    """
    Testing different attributes of Stepwise plot
    """

    @pytest.fixture(scope="class")
    def plot_result(self, titanic_vd):
        """
        Create a stepwise regression plot
        """
        model = LogisticRegression(
            name="test_LR_titanic", tol=1e-4, max_iter=100, solver="Newton"
        )
        stepwise_result = stepwise(
            model,
            input_relation=titanic_vd,
            X=[
                COL_NAME_1,
                COL_NAME_2,
                COL_NAME_3,
                COL_NAME_4,
            ],
            y=BY_COL,
            direction="backward",
        )
        return stepwise_result.step_wise_

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
        test_title = "n_features"
        # Act
        # Assert
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_label(self):
        """
        Testing y-axis title
        """
        # Arrange
        test_title = "bic"
        # Act
        # Assert
        assert get_yaxis_label(self.result) == test_title, "Y axis label incorrect"

    def test_properties_no_of_elements(self):
        """
        Test all objects
        """
        # Arrange
        total_items = 8
        # Act
        # Assert
        assert len(self.result.data) == pytest.approx(
            total_items, abs=1
        ), "Some elements missing"

    def test_data_start_and_end(self):
        """
        Test start and end objects
        """
        # Arrange
        start = "Start"
        end = "End"
        # Act
        # Assert
        assert start in [
            self.result.data[i]["name"] for i in range(len(self.result.data))
        ] and end in [
            self.result.data[i]["name"] for i in range(len(self.result.data))
        ], "Some elements missing"

    def test_additional_options_custom_height(self, titanic_vd):
        """
        Test custom width and height
        """
        # rrange
        custom_height = 650
        custom_width = 700
        model = LogisticRegression(
            name="test_LR_titanic", tol=1e-4, max_iter=100, solver="Newton"
        )
        # Act
        stepwise_result = stepwise(
            model,
            input_relation=titanic_vd,
            X=[
                "age",
                "fare",
                "parch",
                "pclass",
            ],
            y="survived",
            direction="backward",
            height=custom_height,
            width=custom_width,
        )
        result = stepwise_result.step_wise_
        # Assert
        assert (
            result.layout["height"] == custom_height
            and result.layout["width"] == custom_width
        ), "Custom height and width not working"
