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
import numpy as np

# Testing variables
col_name_1 = "0"
col_name_2 = "binary"


@pytest.fixture(scope="class")
def plot_result(dummy_dist_vd):
    return dummy_dist_vd[col_name_1].boxplot()


@pytest.fixture(scope="class")
def plot_result_2(dummy_dist_vd):
    return dummy_dist_vd[col_name_1].boxplot(by=col_name_2)


class TestBoxPlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result

    def test_properties_output_type(self, plotly_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result) == plotly_figure_object, "wrong object crated"

    def test_properties_xaxis_title(self):
        # Arrange
        # Act
        # Assert
        assert self.result.layout["xaxis"]["title"]["text"] == "0", "X-axis title issue"

    def test_properties_yaxis_title(self):
        # Arrange
        # Act
        # Assert
        assert (
            self.result.layout["yaxis"]["title"]["text"] is None
        ), "Y-axis title issue"

    def test_properties_orientation(self):
        # Arrange
        # Act
        # Assert
        assert self.result.data[0]["orientation"] == "h", "Orientation is not correct"

    def test_properties_bound_hover_labels(self):
        # Arrange
        # Act

        name_list = []
        for i in range(len(self.result.data)):
            name_list.append(self.result.data[i]["hovertemplate"].split(":")[0])
        test_list = ["Lower", "Median", "Upper"]
        is_subset = all(elem in name_list for elem in test_list)
        # Assert
        assert is_subset, "Hover label error"

    def test_properties_quartile_labels(self):
        # Arrange
        # Act

        name_list = []
        for i in range(len(self.result.data)):
            name_list.append(self.result.data[i]["hovertemplate"].split("%")[0])
        test_list = ["25.0", "75.0"]
        is_subset = all(elem in name_list for elem in test_list)
        # Assert
        assert is_subset, "Hover label error for quantiles"

    def test_properties_quartile_labels_for_custom_q1(self, dummy_dist_vd):
        # Arrange
        # Act
        result = dummy_dist_vd["0"].boxplot(q=[0.2, 0.7])
        name_list = []
        for i in range(len(result.data)):
            name_list.append(result.data[i]["hovertemplate"].split("%")[0])
        test_list = ["20.0", "70.0"]
        is_subset = all(elem in name_list for elem in test_list)
        # Assert
        assert is_subset, "Hover label error for quantiles"

    def test_properties_lower_hover_box_max_value_is_equal_to_minimum_of_q1(self):
        # Arrange
        # Act
        # Assert
        assert self.result.data[1]["base"][0] + self.result.data[1]["x"][
            0
        ] == pytest.approx(
            self.result.data[2]["base"][0], abs=1e-2
        ), "Hover boxes may overlap"

    def test_properties_q1_hover_box_max_value_is_equal_to_minimum_of_median(self):
        # Arrange
        # Act
        # Assert
        assert self.result.data[2]["base"][0] + self.result.data[2]["x"][
            0
        ] == pytest.approx(
            self.result.data[3]["base"][0], abs=1e-2
        ), "Hover boxes may overlap"

    def test_data_median_value(self):
        # Arrange
        # Act
        # Assert
        assert self.result.data[0]["median"][0] == pytest.approx(
            50, 1
        ), "median not computed correctly"

    def test_data_maximum_point_value(self, dummy_dist_vd):
        # Arrange
        # Act
        # Assert
        assert dummy_dist_vd.max()["max"][0] == pytest.approx(
            max(self.result.data[0]["x"][0]), 2
        ), "Maximum value not in plot"


class TestParitionBoxPlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result_2):
        self.result = plot_result_2

    def test_properties_bound_hover_labels_for_partitioned_data(self):
        # Arrange
        name_list = []
        for i in range(len(self.result.data)):
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
        # Arrange
        # Act
        result = dummy_dist_vd["0"].boxplot(by="binary", q=[0.2, 0.7])
        name_list = []
        for i in range(len(result.data)):
            name_list.append(result.data[i]["hovertemplate"].split("%")[0])
        # Assert
        assert name_list.count("20.0") == 2, "Hover label error for quantiles"
        assert name_list.count("70.0") == 2, "Hover label error for quantiles"

    def test_properties_lower_hover_box_max_value_is_equal_to_minimum_of_q1_for_partitioned_data(
        self,
    ):
        # Arrange
        # Act
        # Assert
        assert (
            self.result.data[4]["base"][0] + self.result.data[4]["y"][0]
            == self.result.data[5]["base"][0]
        ), "Hover boxes may overlap"

    def test_data_median_value_for_partitioned_data_for_x_is_0(self):
        # Arrange
        # Act
        # Assert
        assert self.result.data[0]["median"][0] == pytest.approx(
            50, 1
        ), "Median not computed correctly for binary=0"

    def test_data_median_value_for_partitioned_data_for_x_is_0(self):
        # Arrange
        # Act
        # Assert
        assert self.result.data[1]["median"][0] == pytest.approx(
            50, 1
        ), "Median not computed correctly for binary=1"
