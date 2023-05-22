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
    get_xaxis_label,
    get_yaxis_label,
    get_width,
    get_height,
    get_title,
)

# Testing variables
COL_NAME_1 = "0"
COL_NAME_2 = "1"
BY_COL = "binary"
POS_LABEL = "0"


class TestHighchartsMachineLearningROCPlot:
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
        assert get_title(self.result) == test_title, "Plot Title Incorrect"

    def test_properties_xaxis_label(self):
        """
        Testing x-axis label
        """
        # Arrange
        test_title = "False Positive Rate (1-Specificity)"
        # Act
        # Assert
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_label(self):
        """
        Testing y-axis title
        """
        # Arrange
        test_title = "True Positive Rate (Sensitivity)"
        # Act
        # Assert
        assert get_yaxis_label(self.result) == test_title, "Y axis label incorrect"

    def test_additional_options_custom_height_and_width(self, dummy_dist_vd):
        """
        Test custom width and height
        """
        # rrange
        custom_height = 3
        custom_width = 4
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


class TestHighchartsMachineLearningCutoffCurvePlot:
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
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    @pytest.mark.skip(reason="Cannot extract y axis value from highchart")
    def test_properties_yaxis_label(self):
        """
        Testing y-axis title
        """
        # Arrange
        test_title = "Values"
        # Act
        # Assert
        assert get_yaxis_label(self.result) == test_title, "Y axis label incorrect"

    def test_additional_options_custom_height(self, dummy_dist_vd):
        """
        Test custom width and height
        """
        # rrange
        custom_height = 2
        custom_width = 3
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


class TestHighchartsMachineLearningPRCPlot:
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
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_label(self):
        """
        Testing y-axis title
        """
        # Arrange
        test_title = "Precision"
        # Act
        # Assert
        assert get_yaxis_label(self.result) == test_title, "Y axis label incorrect"

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
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"


class TestHighchartsMachineLearningLiftChartPlot:
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
        assert get_title(self.result) == test_title, "Plot Title Incorrect"

    def test_properties_xaxis_label(self):
        """
        Testing x-axis label
        """
        # Arrange
        test_title = "Cumulative Data Fraction"
        # Act
        # Assert
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    @pytest.mark.skip(reason="Need to fix y axis")
    def test_properties_yaxis_label(self):
        """
        Testing y-axis title
        """
        # Arrange
        test_title = "Values"
        # Act
        # Assert
        assert get_yaxis_label(self.result) == test_title, "Y axis label incorrect"

    def test_additional_options_custom_height(self, dummy_probability_data):
        """
        Test custom width and height
        """
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
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"
