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
from verticapy.tests_new.plotting.conftest import (
    get_xaxis_label,
    get_yaxis_label,
    get_width,
    get_height,
)

# Testing variables
COL_NAME_1 = "PetalLengthCm"
COL_NAME_2 = "PetalWidthCm"
COL_NAME_3 = "SepalWidthCm"
COL_NAME_4 = "SepalLengthCm"
BY_COL = "Species"


class TestHighchartsMachineLearningImportanceBarChartPlot:
    """
    Testing different attributes of Importance Bar Chart plot
    """

    @pytest.fixture(scope="class")
    def plot_result(self, schema_loader, iris_vd):
        """
        Create an Importance Bar chart plot using RandomForest Classifier
        """
        model = RandomForestClassifier(f"{schema_loader}.importance_test")
        model.fit(
            iris_vd,
            [COL_NAME_1, COL_NAME_2, COL_NAME_3, COL_NAME_4],
            BY_COL,
        )
        yield model.features_importance(), model
        model.drop()

    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        """
        Get the plot results
        """
        self.result = plot_result[0]

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
        test_title = "Importance (%)"
        # Act
        # Assert
        assert get_yaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_label(self):
        """
        Testing y-axis title
        """
        # Arrange
        test_title = "Features"
        # Act
        # Assert
        assert get_xaxis_label(self.result) == test_title, "Y axis label incorrect"

    def test_data_no_of_columns(self):
        """
        Test if four columns are produced
        """
        # Arrange
        total_items = 4
        # Act
        # Assert
        assert len(self.result.data_temp[0].data) == total_items, "Some columns missing"

    def test_additional_options_custom_height(self, plot_result):
        """
        Test custom width and height
        """
        # rrange
        custom_height = 600
        custom_width = 700
        # Act
        result = plot_result[1].features_importance(
            height=custom_height, width=custom_width
        )
        # Assert
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"
