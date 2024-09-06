"""
Copyright  (c)  2018-2024 Open Text  or  one  of its
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

# Vertica
from verticapy.tests_new.plotting.base_test_files import LearningRegressionTreePlot


class TestPlotlyMachineLearningRegressionTreePlot(LearningRegressionTreePlot):
    """
    Testing different attributes of Regression Tree plot
    """

    def test_properties_observations_label(self):
        """
        Test plot title
        """
        # Arrange
        test_title = "Observations"
        # Act
        # Assert
        assert self.result.data[0]["name"] == test_title, "X axis label incorrect"

    def test_properties_prediction_label(self):
        """
        Test plot title
        """
        # Arrange
        test_title = "Prediction"
        # Act
        # Assert
        assert self.result.data[1]["name"] == test_title, "Y axis label incorrect"

    def test_properties_hover_label(self):
        """
        Test hover labels
        """
        # Arrange
        test_title = (
            f"{self.COL_NAME_1}: %" "{x} <br>" f"{self.COL_NAME_2}: %" "{y} <br>"
        )
        # Act
        # Assert
        assert (
            self.result.data[0]["hovertemplate"] == test_title
        ), "Hover information incorrect"

    def test_properties_no_of_elements(self):
        """
        Test number of elements
        """
        # Arrange
        total_items = 2
        # Act
        # Assert
        assert len(self.result.data) == pytest.approx(
            total_items, abs=1
        ), "Some elements missing"
