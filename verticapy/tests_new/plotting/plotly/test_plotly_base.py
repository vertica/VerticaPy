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
from verticapy.plotting._plotly.base import PlotlyBase

# Standard Python Modules
import numpy as np
import plotly.graph_objects as go


@pytest.fixture(name="dummy_data_parents_children", scope="class")
def load_dummy_data_parents_children():
    """
    Create a dummy nested dataset
    """
    parents = ["0", "1"]
    children = ["A", "B", "C"]
    all_counts = [4, 8, 12, 6, 12, 18]
    data_dict = {}
    for i, parent in enumerate(parents):
        for j, child in enumerate(children):
            count = all_counts[i * len(children) + j]
            data_dict[(parent, child)] = count

    data_dict[("", parents[0])] = sum(
        data_dict[(parents[0], child)] for child in children
    )
    data_dict[("", parents[1])] = sum(
        data_dict[(parents[1], child)] for child in children
    )

    data_shaped = np.array(
        [
            [parents[0], parents[0], parents[0], parents[1], parents[1], parents[1]],
            [
                children[0],
                children[1],
                children[2],
                children[0],
                children[1],
                children[2],
            ],
            [data_dict[(parent, child)] for parent, child in data_dict if parent != ""],
        ],
        dtype="<U21",
    )
    return data_shaped, data_dict


class TestConvertLabelsAndGetCounts:
    """
    Test _convert_labels_and_get_counts function in base file
    """

    @pytest.fixture(autouse=True)
    def result(self, dummy_data_parents_children):
        """
        Get the plot results
        """
        func = PlotlyBase()
        self.result = func._convert_labels_and_get_counts(
            dummy_data_parents_children[0]
        )

    def test_counts(self, dummy_data_parents_children):
        """
        Test counts
        """
        # Arrange
        reference = dummy_data_parents_children[1]
        # Act
        result = {}
        for i in range(len(self.result[0])):
            child_value = self.result[1][i]
            parent_value = self.result[2][i]
            count = self.result[3][i]
            key = (parent_value, child_value)
            result[key] = count
        # Assert
        assert result == reference

    def test_unique_ids(self):
        """
        Test if unique ids created for each element
        """
        # Arrange
        # Act
        assert len(self.result[0]) == len(set(self.result[0]))


class TestGetFig:
    """
    Test _get_fig function in base file
    """

    def test_input_fig(self):
        """
        Test for an input plotly object
        """
        # Arrange
        func = PlotlyBase()
        test_fig = go.Figure([go.Bar(x=[0, 1], y=[5, 10])])
        # Act
        result = func._get_fig(test_fig)
        # Assert
        assert result == test_fig

    def test_input_none(self):
        """
        Test for a None object
        """
        # Arrange
        func = PlotlyBase()
        # Act
        result = func._get_fig(None)
        # Assert
        assert result == go.Figure()


def test_convert_labels_for_heatmap_result():
    """
    Test _convert_labels_for_heatmap function in base file
    """
    # Arrange
    func = PlotlyBase()
    test = ["[0.0;5]", "[5;10]", "[10;15]"]
    mid_points = ["2.5", "7.5", "12.5"]
    # Act
    result = func._convert_labels_for_heatmap(test)
    # Assert
    assert result == mid_points


test_data_max_2_decimal = np.array([[0, 1.0, 0.2], [0.1, 0.0, 0.12]])
test_data_max_4_decimal = np.array([[0, 1.0, 0.2], [0.1, 0.0, 0.1234]])
test_data_max_8_decimal = np.array([[0, 1.0, 0.2], [0.1, 0.0, 0.12345678]])


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (test_data_max_2_decimal, 2),
        (test_data_max_4_decimal, 4),
        (test_data_max_8_decimal, 8),
    ],
)
def test__get_max_decimal_point_result(test_input, expected):
    """
    Test _get_max_decimal_point function in base file
    """
    # Arrange
    func = PlotlyBase()
    # Act
    result = func._get_max_decimal_point(test_input)
    # Assert
    assert result == expected
