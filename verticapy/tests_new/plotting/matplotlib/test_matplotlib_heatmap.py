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
# Vertica
from verticapy.tests_new.plotting.base_test_files import VDFPivotHeatMap, VDFHeatMap


class TestMatplotlibVDFPivotHeatMap(VDFPivotHeatMap):
    """
    Testing different attributes of Heatmap plot on a vDataFrame
    """

    def test_properties_yaxis_labels_for_categorical_data(self, titanic_vd):
        """
        Test labels for Y-axis
        """
        # Arrange
        expected_labels = (
            '"survived"',
            '"pclass"',
            '"fare"',
            '"parch"',
            '"age"',
            '"sibsp"',
            '"body"',
        )
        # Act
        result = titanic_vd.corr(method="pearson", focus="survived")
        yaxis_labels = [
            result.get_yticklabels()[i].get_text()
            for i in range(len(result.get_yticklabels()))
        ]
        # Assert
        assert set(yaxis_labels).issubset(expected_labels), "Y-axis labels incorrect"


class TestMatplotlibVDFHeatMap(VDFHeatMap):
    """
    Testing different attributes of Heatmap plot on a vDataFrame
    """
