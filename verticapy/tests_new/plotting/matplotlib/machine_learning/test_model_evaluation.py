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
import numpy as np

# Verticapy
from verticapy.learn.ensemble import RandomForestClassifier
from verticapy.learn.model_selection import lift_chart, prc_curve
from verticapy.tests_new.plotting.conftest import (
    get_xaxis_label,
    get_yaxis_label,
)

# Testing variables
col_name_1 = "0"
col_name_2 = "1"
by_col = "binary"
pos_label = "0"


@pytest.fixture(scope="class")
def plot_result_roc(dummy_dist_vd):
    model = RandomForestClassifier("roc_plot_test")
    model.drop()
    model.fit(dummy_dist_vd, [col_name_1, col_name_2], by_col)
    return model.roc_curve()


@pytest.fixture(scope="class")
def plot_result_cutoff(dummy_dist_vd):
    model = RandomForestClassifier("roc_plot_test_2")
    model.drop()
    model.fit(dummy_dist_vd, [col_name_1, col_name_2], by_col)
    return model.cutoff_curve(pos_label=pos_label)


@pytest.fixture(scope="class")
def plot_result_prc(dummy_probability_data):
    return prc_curve("y_true", "y_score", dummy_probability_data)


@pytest.fixture(scope="class")
def plot_result_lift_chart(dummy_probability_data):
    return lift_chart("y_true", "y_score", dummy_probability_data)


class TestMachineLearningROCCurve:
    @pytest.fixture(autouse=True)
    def result(self, plot_result_roc):
        self.result = plot_result_roc

    def test_properties_output_type(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_title(self):
        # Arrange
        test_title = "ROC Curve"
        # Act
        # Assert
        assert self.result.get_title() == test_title, "Plot Title Incorrect"

    def test_properties_xaxis_label(self):
        # Arrange
        test_title = "False Positive Rate (1-Specificity)"
        # Act
        # Assert
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_label(self):
        # Arrange
        test_title = "True Positive Rate (Sensitivity)"
        # Act
        # Assert
        assert get_yaxis_label(self.result) == test_title, "Y axis label incorrect"

    def test_additional_options_custom_height_and_width(self, dummy_dist_vd):
        # rrange
        custom_height = 3
        custom_width = 4
        # Act
        model = RandomForestClassifier("roc_plot_test")
        model.drop()
        model.fit(dummy_dist_vd, [col_name_1, col_name_2], by_col)
        result = model.cutoff_curve(
            pos_label=pos_label, width=custom_width, height=custom_height
        )
        # Assert
        assert (
            result.get_figure().get_size_inches()[0] == custom_width
            and result.get_figure().get_size_inches()[1] == custom_height
        ), "Custom width or height not working"


class TestMachineLearningCutoffCurve:
    @pytest.fixture(autouse=True)
    def result(self, plot_result_cutoff):
        self.result = plot_result_cutoff

    def test_properties_output_type(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_xaxis_label(self):
        # Arrange
        test_title = "Decision Boundary"
        # Act
        # Assert
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    @pytest.mark.skip(reason="Need to fix y axis")
    def test_properties_yaxis_label(self):
        # Arrange
        test_title = "Values"
        # Act
        # Assert
        assert get_yaxis_label(self.result) == test_title, "Y axis label incorrect"

    def test_additional_options_custom_height(self, dummy_dist_vd):
        # rrange
        custom_height = 2
        custom_width = 3
        # Act
        model = RandomForestClassifier("cutoff_curve_plot_test")
        model.drop()
        model.fit(dummy_dist_vd, [col_name_1, col_name_2], by_col)
        result = model.cutoff_curve(
            pos_label=pos_label, width=custom_width, height=custom_height
        )
        # Assert
        assert (
            result.get_figure().get_size_inches()[0] == custom_width
            and result.get_figure().get_size_inches()[1] == custom_height
        ), "Custom width or height not working"


class TestMachineLearningPRCCurve:
    @pytest.fixture(autouse=True)
    def result(self, plot_result_prc):
        self.result = plot_result_prc

    def test_properties_output_type(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_xaxis_label(self):
        # Arrange
        test_title = "Recall"
        # Act
        # Assert
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_label(self):
        # Arrange
        test_title = "Precision"
        # Act
        # Assert
        assert get_yaxis_label(self.result) == test_title, "Y axis label incorrect"

    def test_additional_options_custom_height(self, dummy_probability_data):
        # rrange
        custom_height = 650
        custom_width = 700
        # Act
        result = prc_curve(
            "y_true",
            "y_score",
            dummy_probability_data,
            width=custom_width,
            height=custom_height,
        )
        # Assert
        assert (
            result.get_figure().get_size_inches()[0] == custom_width
            and result.get_figure().get_size_inches()[1] == custom_height
        ), "Custom width or height not working"


class TestMachineLearningLiftChart:
    @pytest.fixture(autouse=True)
    def result(self, plot_result_lift_chart):
        self.result = plot_result_lift_chart

    def test_properties_output_type(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_title(self):
        # Arrange
        test_title = "Lift Table"
        # Act
        # Assert
        assert self.result.get_title() == test_title, "Plot title incorrect"

    def test_properties_xaxis_label(self):
        # Arrange
        test_title = "Cumulative Data Fraction"
        # Act
        # Assert
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    @pytest.mark.skip(reason="Need to fix y axis")
    def test_properties_yaxis_label(self):
        # Arrange
        test_title = "Values"
        # Act
        # Assert
        assert get_yaxis_label(self.result) == test_title, "Y axis label incorrect"

    def test_additional_options_custom_height(self, dummy_probability_data):
        # rrange
        custom_height = 6
        custom_width = 7
        # Act
        result = prc_curve(
            "y_true",
            "y_score",
            dummy_probability_data,
            width=custom_width,
            height=custom_height,
        )
        # Assert
        assert (
            result.get_figure().get_size_inches()[0] == custom_width
            and result.get_figure().get_size_inches()[1] == custom_height
        ), "Custom width or height not working"
