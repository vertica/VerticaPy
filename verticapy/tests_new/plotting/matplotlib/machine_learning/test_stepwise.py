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
col_name_1 = "age"
col_name_2 = "fare"
col_name_3 = "parch"
col_name_4 = "pclass"
by_col = "survived"


@pytest.fixture(scope="class")
def plot_result(titanic_vd):
    model = LogisticRegression(
        name="test_LR_titanic", tol=1e-4, max_iter=100, solver="Newton"
    )
    stepwise_result = stepwise(
        model,
        input_relation=titanic_vd,
        X=[
            col_name_1,
            col_name_2,
            col_name_3,
            col_name_4,
        ],
        y=by_col,
        direction="backward",
    )
    return stepwise_result.step_wise_


class TestMachineLearningStepwisePlot:
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
        test_title = "n_features"
        # Act
        # Assert
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_label(self):
        # Arrange
        test_title = "bic"
        # Act
        # Assert
        assert get_yaxis_label(self.result) == test_title, "Y axis label incorrect"

    def test_additional_options_custom_height(self, titanic_vd):
        # rrange
        custom_height = 6
        custom_width = 7
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
            result.get_figure().get_size_inches()[0] == custom_width
            and result.get_figure().get_size_inches()[1] == custom_height
        ), "Custom width or height not working"
