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
from verticapy.tests_new.plotting.base_test_files import ImportanceBarChartPlot


class TestMatplotlibMachineLearningImportanceBarChart(ImportanceBarChartPlot):
    """
    Testing different attributes of Importance Bar Chart plot
    """

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return ["Importance (%)", "Features"]

    def test_properties_xaxis_label(self):
        """
        Testing x-axis title
        """

    def test_data_no_of_columns(self):
        """
        Test if four columns are produced
        """
        # Arrange
        total_items = 4
        # Act
        # Assert
        assert len(self.result.containers[0]) == total_items, "Some columns missing"
