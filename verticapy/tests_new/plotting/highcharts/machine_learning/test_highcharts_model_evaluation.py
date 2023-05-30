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


# Verticapy
from verticapy.tests_new.plotting.conftest import BasicPlotTests
from verticapy.learn.ensemble import RandomForestClassifier
from verticapy.learn.model_selection import lift_chart, prc_curve
from verticapy.tests_new.plotting.conftest import get_title

# Testing variables
COL_NAME_1 = "0"
COL_NAME_2 = "1"
BY_COL = "binary"
POS_LABEL = "0"


@pytest.fixture(name="model_result", scope="module")
def load_model(schema_loader, dummy_dist_vd):
    """
    Load the Random Forest Classifier model
    """
    model = RandomForestClassifier(f"{schema_loader}.random_forest_plot_test")
    model.drop()
    model.fit(dummy_dist_vd, [COL_NAME_1, COL_NAME_2], BY_COL)
    yield model
    model.drop()


class TestHighchartsMachineLearningROCPlot(BasicPlotTests):
    """
    Testing different attributes of ROC plot
    """

    @pytest.fixture(autouse=True)
    def model(self, model_result):
        """
        Load test model
        """
        self.model = model_result

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [
            "False Positive Rate (1-Specificity)",
            "True Positive Rate (Sensitivity)",
        ]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.model.roc_curve,
            {},
        )

    def test_properties_title(self):
        """
        Test plot title
        """
        # Arrange
        test_title = "ROC Curve"
        # Act
        # Assert
        assert get_title(self.result) == test_title, "Plot Title Incorrect"


class TestHighchartsMachineLearningCutoffCurvePlot(BasicPlotTests):
    """
    Testing different attributes of Curve plot
    """

    @pytest.fixture(autouse=True)
    def model(self, model_result):
        """
        Load test model
        """
        self.model = model_result

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return ["Decision Boundary", "Values"]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.model.cutoff_curve,
            {"pos_label": POS_LABEL},
        )

    @pytest.mark.skip(reason="Cannot extract y axis value from highchart")
    def test_properties_yaxis_label(self):
        """
        Testing y-axis title
        """


class TestHighchartsMachineLearningPRCPlot(BasicPlotTests):
    """
    Testing different attributes of PRC plot
    """

    @pytest.fixture(autouse=True)
    def data(self, dummy_probability_data):
        """
        Load test model
        """
        self.data = dummy_probability_data

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return ["Recall", "Precision"]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            prc_curve,
            {"y_true": "y_true", "y_score": "y_score", "input_relation": self.data},
        )


class TestHighchartsMachineLearningLiftChartPlot(BasicPlotTests):
    """
    Testing different attributes of Lift Chart plot
    """

    @pytest.fixture(autouse=True)
    def data(self, dummy_probability_data):
        """
        Load test model
        """
        self.data = dummy_probability_data

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return ["Cumulative Data Fraction", "Values"]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            lift_chart,
            {"y_true": "y_true", "y_score": "y_score", "input_relation": self.data},
        )

    @pytest.mark.skip(reason="Need to fix y axis")
    def test_properties_yaxis_label(self):
        """
        Testing y-axis title
        """

    def test_properties_title(self):
        """
        Test plot title
        """
        # Arrange
        test_title = "Lift Table"
        # Act
        # Assert
        assert get_title(self.result) == test_title, "Plot Title Incorrect"
