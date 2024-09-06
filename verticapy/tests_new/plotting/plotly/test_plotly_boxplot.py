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
import pytest

# Vertica
from verticapy.tests_new.plotting.base_test_files import (
    VDCBoxPlot,
    VDCParitionBoxPlot,
    VDFBoxPlot,
)


class TestPlotlyVDCBoxPlot(VDCBoxPlot):
    """
    Testing different attributes of Box plot on a vDataColumn
    """

    def test_properties_orientation(self):
        """
        Test orientation of plot
        """
        # Arrange
        # Act
        # Assert
        assert self.result.data[0]["orientation"] == "h", "Orientation is not correct"

    def test_properties_bound_hover_labels(self):
        """
        Test hover labels
        """
        # Arrange
        # Act

        name_list = []
        for i, _ in enumerate(self.result.data):
            name_list.append(self.result.data[i]["hovertemplate"].split(":")[0])
        test_list = ["Lower", "Median", "Upper"]
        is_subset = all(elem in name_list for elem in test_list)
        # Assert
        assert is_subset, "Hover label error"

    def test_properties_quartile_labels(self):
        """
        Test quartile labels
        """
        # Arrange
        # Act

        name_list = []
        for i, _ in enumerate(self.result.data):
            name_list.append(self.result.data[i]["hovertemplate"].split("%")[0])
        test_list = ["25.0", "75.0"]
        is_subset = all(elem in name_list for elem in test_list)
        # Assert
        assert is_subset, "Hover label error for quantiles"

    def test_properties_quartile_labels_for_custom_q1(self, dummy_dist_vd):
        """
        Test quartile labels for custom q1
        """
        # Arrange
        # Act
        result = dummy_dist_vd["0"].boxplot(q=[0.2, 0.7])
        name_list = []
        for i, _ in enumerate(result.data):
            name_list.append(result.data[i]["hovertemplate"].split("%")[0])
        test_list = ["20.0", "70.0"]
        is_subset = all(elem in name_list for elem in test_list)
        # Assert
        assert is_subset, "Hover label error for quantiles"

    def test_properties_lower_hover_box_max_value_is_equal_to_minimum_of_q1(self):
        """
        Test overlap between hover boxes
        """
        # Arrange
        # Act
        # Assert
        assert self.result.data[1]["base"][0] + self.result.data[1]["x"][
            0
        ] == pytest.approx(
            self.result.data[2]["base"][0], abs=1e-2
        ), "Hover boxes may overlap"

    def test_properties_q1_hover_box_max_value_is_equal_to_minimum_of_median(self):
        """
        Test overlap between hover boxes
        """
        # Arrange
        # Act
        # Assert
        assert self.result.data[2]["base"][0] + self.result.data[2]["x"][
            0
        ] == pytest.approx(
            self.result.data[3]["base"][0], abs=1e-2
        ), "Hover boxes may overlap"

    def test_data_median_value(self):
        """
        Test if median comptued correctly
        """
        # Arrange
        # Act
        # Assert
        assert self.result.data[0]["median"][0] == pytest.approx(
            50, 1
        ), "median not computed correctly"


class TestPlotlyParitionVDCBoxPlot(VDCParitionBoxPlot):
    """
    Testing different attributes of Box plot on a vDataColumn using "by" attribute
    """

    def test_properties_bound_hover_labels_for_partitioned_data(self):
        """
        Test if two hover labels created
        """
        # Arrange
        name_list = []
        for i, _ in enumerate(self.result.data):
            name_list.append(self.result.data[i]["hovertemplate"].split(":")[0])
        # Act
        # Assert
        assert name_list.count("25.0% ") == 2, "Hover label error"
        assert name_list.count("75.0% ") == 2, "Hover label error"
        assert name_list.count("Lower") == 2, "Hover label error"
        assert name_list.count("Median") == 2, "Hover label error"
        assert name_list.count("Upper") == 2, "Hover label error"

    def test_properties_quartile_labels_for_custom_q1_for_partitioned_data(
        self, dummy_dist_vd
    ):
        """
        Test quartile labels
        """
        # Arrange
        # Act
        result = dummy_dist_vd["0"].boxplot(by="binary", q=[0.2, 0.7])
        name_list = []
        for i, _ in enumerate(result.data):
            name_list.append(result.data[i]["hovertemplate"].split("%")[0])
        # Assert
        assert name_list.count("20.0") == 2, "Hover label error for quantiles"
        assert name_list.count("70.0") == 2, "Hover label error for quantiles"

    def test_properties_lower_hover_box_max_value_is_equal_to_minimum_of_q1_for_partitioned_data(
        self,
    ):
        """
        Test hover labels
        """
        # Arrange
        # Act
        # Assert
        assert (
            self.result.data[4]["base"][0] + self.result.data[4]["y"][0]
            == self.result.data[5]["base"][0]
        ), "Hover boxes may overlap"

    def test_data_median_value_for_partitioned_data_for_x_is_0(self):
        """
        Test median value for x = 0
        """
        # Arrange
        # Act
        # Assert
        assert self.result.data[0]["median"][0] == pytest.approx(
            50, 1
        ), "Median not computed correctly for binary=0"

    def test_data_median_value_for_partitioned_data_for_x_is_1(self):
        """
        Test median value for x = 1
        """
        # Arrange
        # Act
        # Assert
        assert self.result.data[1]["median"][0] == pytest.approx(
            50, 1
        ), "Median not computed correctly for binary=1"


class TestPlotlyVDFBoxPlot(VDFBoxPlot):
    """
    Testing different attributes of Box plot on a vDataFrame
    """

    @pytest.mark.skip(reason="The plot does not have label on y-axis yet")
    def test_properties_yaxis_label(self):
        """
        Testing x-axis title
        """
