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
# Verticapy
from verticapy.tests_new.plotting.base_test_files import LearningRegressionPlot


class TestPlotlyMachineLearningRegressionPlot(LearningRegressionPlot):
    """
    Testing different attributes of Regression plot
    """

    def test_properties_scatter_and_line_plot(self):
        """
        Test two items exist
        """
        # Arrange
        total_items = 2
        # Act
        # Assert
        assert len(self.result.data) == total_items, "Either line or scatter missing"

    def test_data_all_scatter_points(self, dummy_scatter_vd):
        """
        Test all datapoints
        """
        # Arrange
        no_of_points = len(dummy_scatter_vd)
        # Act
        # Assert
        assert (
            len(self.result.data[0]["x"]) == no_of_points
        ), "Discrepancy between points plotted and total number of points"
