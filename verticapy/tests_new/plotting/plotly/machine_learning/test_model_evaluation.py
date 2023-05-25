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
from verticapy.learn.ensemble import RandomForestClassifier
from verticapy.learn.model_selection import lift_chart, prc_curve
from verticapy.tests_new.plotting.conftest import (
    get_width,
    get_height,
)

# Testing variables
COL_NAME_1 = "0"
COL_NAME_2 = "1"
BY_COL = "binary"
POS_LABEL = "0"


class TestPlotlyMachineLearningROCCurve:
    """
    Testing different attributes of ROC plot
    """

    @pytest.fixture(scope="class")
    def plot_result_roc(self, dummy_dist_vd):
        """
        Create an ROC plot
        """
        model = RandomForestClassifier("roc_plot_test")
        model.drop()
        model.fit(dummy_dist_vd, [COL_NAME_1, COL_NAME_2], BY_COL)
        return model.roc_curve()

    @pytest.fixture(autouse=True)
    def result(self, plot_result_roc):
        """
        Get the plot results
        """
        self.result = plot_result_roc

    def test_properties_output_type(self, plotting_library_object):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_title(self):
        """
        Test plot title
        """
        # Arrange
        test_title = "ROC Curve"
        # Act
        # Assert
        assert self.result.layout["title"]["text"] == test_title, "Plot Title Incorrect"

    def test_properties_xaxis_label(self):
        """
        Testing x-axis label
        """
        # Arrange
        test_title = "False Positive Rate (1-Specificity)"
        # Act
        # Assert
        assert (
            self.result.layout["xaxis"]["title"]["text"] == test_title
        ), "X axis label incorrect"

    def test_properties_yaxis_label(self):
        """
        Testing y-axis title
        """
        # Arrange
        test_title = "True Positive Rate (Sensitivity)"
        # Act
        # Assert
        assert (
            self.result.layout["yaxis"]["title"]["text"] == test_title
        ), "Y axis label incorrect"

    def test_properties_no_of_elements(self):
        """
        Test if both elements plotted
        """
        # Arrange
        total_items = 2
        # Act
        # Assert
        assert len(self.result.data) == total_items, "Some elements missing"

    def test_additional_options_custom_height_and_width(self, dummy_dist_vd):
        """
        Test custom width and height
        """
        # rrange
        custom_height = 650
        custom_width = 700
        # Act
        model = RandomForestClassifier("roc_plot_test")
        model.drop()
        model.fit(dummy_dist_vd, [COL_NAME_1, COL_NAME_2], BY_COL)
        result = model.cutoff_curve(
            pos_label=POS_LABEL, width=custom_width, height=custom_height
        )
        # Assert
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"


class TestPlotlyMachineLearningCutoffCurve:
    """
    Testing different attributes of Curve plot
    """

    @pytest.fixture(scope="class")
    def plot_result_cutoff(self, dummy_dist_vd):
        """
        Create cutoff curve
        """
        model = RandomForestClassifier("roc_plot_test_2")
        model.drop()
        model.fit(dummy_dist_vd, [COL_NAME_1, COL_NAME_2], BY_COL)
        return model.cutoff_curve(pos_label=POS_LABEL)

    @pytest.fixture(autouse=True)
    def result(self, plot_result_cutoff):
        """
        Get the plot results
        """
        self.result = plot_result_cutoff

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
        test_title = "Decision Boundary"
        # Act
        # Assert
        assert (
            self.result.layout["xaxis"]["title"]["text"] == test_title
        ), "X axis label incorrect"

    def test_properties_yaxis_label(self):
        """
        Testing y-axis title
        """
        # Arrange
        test_title = "Values"
        # Act
        # Assert
        assert (
            self.result.layout["yaxis"]["title"]["text"] == test_title
        ), "Y axis label incorrect"

    def test_properties_no_of_elements(self):
        """
        Test if both elements plotted
        """
        # Arrange
        total_items = 2
        # Act
        # Assert
        assert len(self.result.data) == total_items, "Some elements missing"

    def test_additional_options_custom_height(self, dummy_dist_vd):
        """
        Test custom width and height
        """
        # rrange
        custom_height = 650
        custom_width = 700
        # Act
        model = RandomForestClassifier("cutoff_curve_plot_test")
        model.drop()
        model.fit(dummy_dist_vd, [COL_NAME_1, COL_NAME_2], BY_COL)
        result = model.cutoff_curve(
            pos_label=POS_LABEL, width=custom_width, height=custom_height
        )
        # Assert
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"


class TestPlotlyMachineLearningPRCCurve:
    """
    Testing different attributes of PRC plot
    """

    @pytest.fixture(scope="class")
    def plot_result_prc(self, dummy_probability_data):
        """
        Create a PRC plot
        """
        return prc_curve("y_true", "y_score", dummy_probability_data)

    @pytest.fixture(autouse=True)
    def result(self, plot_result_prc):
        """
        Get the plot results
        """
        self.result = plot_result_prc

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
        test_title = "Recall"
        # Act
        # Assert
        assert (
            self.result.layout["xaxis"]["title"]["text"] == test_title
        ), "X axis label incorrect"

    def test_properties_yaxis_label(self):
        """
        Testing y-axis title
        """
        # Arrange
        test_title = "Precision"
        # Act
        # Assert
        assert (
            self.result.layout["yaxis"]["title"]["text"] == test_title
        ), "Y axis label incorrect"

    def test_properties_no_of_elements(self):
        """
        Test if only element plotted
        """
        # Arrange
        total_items = 1
        # Act
        # Assert
        assert len(self.result.data) == total_items, "Some elements missing"

    def test_additional_options_custom_height(self, dummy_probability_data):
        """
        Test custom width and height
        """
        # Arrange
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
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"


class TestPlotlyMachineLearningLiftChart:
    """
    Testing different attributes of Lift Chart plot
    """

    @pytest.fixture(scope="class")
    def plot_result_lift_chart(self, dummy_probability_data):
        """
        Create a lift chart
        """
        return lift_chart("y_true", "y_score", dummy_probability_data)

    @pytest.fixture(autouse=True)
    def result(self, plot_result_lift_chart):
        """
        Get the plot results
        """
        self.result = plot_result_lift_chart

    def test_properties_output_type(self, plotting_library_object):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_title(self):
        """
        Test plot title
        """
        # Arrange
        test_title = "Lift Table"
        # Act
        # Assert
        assert self.result.layout["title"]["text"] == test_title, "Plot title incorrect"

    def test_properties_xaxis_label(self):
        """
        Testing x-axis label
        """
        # Arrange
        test_title = "Cumulative Data Fraction"
        # Act
        # Assert
        assert (
            self.result.layout["xaxis"]["title"]["text"] == test_title
        ), "X axis label incorrect"

    def test_properties_yaxis_label(self):
        """
        Testing y-axis title
        """
        # Arrange
        test_title = "Values"
        # Act
        # Assert
        assert (
            self.result.layout["yaxis"]["title"]["text"] == test_title
        ), "Y axis label incorrect"

    def test_properties_no_of_elements(self):
        """
        Test if both elements plotted
        """
        # Arrange
        total_items = 2
        # Act
        # Assert
        assert len(self.result.data) == total_items, "Some elements missing"

    def test_additional_options_custom_height(self, dummy_probability_data):
        """
        Test custom width and height
        """
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
            result.layout["height"] == custom_height
            and result.layout["width"] == custom_width
        ), "Custom height and width not working"
