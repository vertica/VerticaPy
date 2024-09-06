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

# Verticapy
from verticapy.tests_new.plotting.base_test_files import (
    ROCPlot,
    CutoffCurvePlot,
    PRCPlot,
    LiftChartPlot,
)


class TestPlotlyMachineLearningROCPlot(ROCPlot):
    """
    Testing different attributes of ROC plot
    """

    def test_properties_no_of_elements(self):
        """
        Test if both elements plotted
        """
        # Arrange
        total_items = 2
        # Act
        # Assert
        assert len(self.result.data) == total_items, "Some elements missing"


class TestPlotlyMachineLearningCutoffCurvePlot(CutoffCurvePlot):
    """
    Testing different attributes of Curve plot
    """

    def test_properties_no_of_elements(self):
        """
        Test if both elements plotted
        """
        # Arrange
        total_items = 2
        # Act
        # Assert
        assert len(self.result.data) == total_items, "Some elements missing"


class TestPlotlyMachineLearningPRCPlot(PRCPlot):
    """
    Testing different attributes of PRC plot
    """

    def test_properties_no_of_elements(self):
        """
        Test if only element plotted
        """
        # Arrange
        total_items = 1
        # Act
        # Assert
        assert len(self.result.data) == total_items, "Some elements missing"


class TestPlotlyMachineLearningLiftChartPlot(LiftChartPlot):
    """
    Testing different attributes of Lift Chart plot
    """

    def test_properties_no_of_elements(self):
        """
        Test if both elements plotted
        """
        # Arrange
        total_items = 2
        # Act
        # Assert
        assert len(self.result.data) == total_items, "Some elements missing"
