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
from verticapy.tests_new.exp.conftest import BasicPlotTests

# Verticapy
from verticapy.learn.ensemble import RandomForestClassifier


# Testing variables
COL_NAME_1 = "PetalLengthCm"
COL_NAME_2 = "PetalWidthCm"
COL_NAME_3 = "SepalWidthCm"
COL_NAME_4 = "SepalLengthCm"
BY_COL = "Species"


class TestHighchartsMachineLearningImportanceBarChartPlot(BasicPlotTests):
    """
    Testing different attributes of Importance Bar Chart plot
    """

    @pytest.fixture(autouse=True)
    def model(self, iris_vd, schema_loader):
        """
        Load test model
        """
        model = RandomForestClassifier(f"{schema_loader}.importance_test")
        model.fit(
            iris_vd,
            [COL_NAME_1, COL_NAME_2, COL_NAME_3, COL_NAME_4],
            BY_COL,
        )
        self.model = model
        yield
        model.drop()

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return ["Features", "Importance (%)"]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.model.features_importance,
            {},
        )

    def test_data_no_of_columns(self):
        """
        Test if four columns are produced
        """
        # Arrange
        total_items = 4
        # Act
        # Assert
        assert len(self.result.data_temp[0].data) == total_items, "Some columns missing"
