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
from verticapy.tests_new.plotting.base_test_files import StepwisePlot


class TestPlotlyMachineLearningStepwisePlot(StepwisePlot):
    """
    Testing different attributes of Stepwise plot
    """

    def test_properties_no_of_elements(self):
        """
        Test all objects
        """
        # Arrange
        total_items = 8
        # Act
        # Assert
        assert len(self.result.data) == pytest.approx(
            total_items, abs=1
        ), "Some elements missing"

    def test_data_start_and_end(self):
        """
        Test start and end objects
        """
        # Arrange
        start = "Start"
        end = "End"
        # Act
        # Assert
        assert start in [
            self.result.data[i]["name"] for i in range(len(self.result.data))
        ] and end in [
            self.result.data[i]["name"] for i in range(len(self.result.data))
        ], "Some elements missing"
