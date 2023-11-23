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
    LogisticRegressionPlot2D,
    LogisticRegressionPlot3D,
)


class TestPlotlyMachineLearningLogisticRegressionPlot2D(LogisticRegressionPlot2D):
    """
    Testing different attributes of 2D Logisti Regression plot
    """

    def test_properties_two_scatter_and_line_plot(self):
        """
        Test if two scatter plots and one line is plotted
        """
        # Arrange
        total_items = 3
        # Act
        # Assert
        assert (
            len(self.result.data) == total_items
        ), "Either line or the two scatter plots are missing"


class TestPlotlyMachineLearningLogisticRegressionPlot3D(LogisticRegressionPlot3D):
    """
    Testing different attributes of 3D Logisti Regression plot
    """

    def test_properties_zaxis_label(self):
        """
        Testing y-axis title
        """
        assert self.result.layout["scene"]["yaxis"]["title"]["text"] == self.COL_NAME_3
