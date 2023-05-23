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


# Vertica
from verticapy.tests_new.plotting.conftest import (
    get_xaxis_label,
    get_yaxis_label,
    get_width,
    get_height,
)

# Testing variables
COL_NAME_1 = "binary"
COL_OF = "0"


class TestMatplotlibVDCHistogramPlot:
    """
    Testing different attributes of Histogram plot on a vDataColumn
    """

    @pytest.fixture(scope="class")
    def plot_result(self, dummy_dist_vd):
        """
        Create a histogram plot for vDataColumn
        """
        return dummy_dist_vd[COL_NAME_1].hist()

    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        """
        Get the plot results
        """
        self.result = plot_result

    def test_properties_output_type(self, matplotlib_figure_object):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, matplotlib_figure_object), "Wrong object created"

    def test_properties_xaxis_title(self):
        """
        Testing x-axis title
        """
        # Arrange
        test_title = COL_NAME_1
        # Act
        # Assert - checking x axis label
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_title(self):
        """
        Testing y-axis title
        """
        # Arrange
        test_title = "density"
        # Act
        # Assert - checking y axis label
        assert get_yaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_additional_options_custom_height(self, dummy_dist_vd):
        """
        Test custom width and height
        """
        # rrange
        custom_height = 6
        custom_width = 7
        # Act
        result = dummy_dist_vd[COL_NAME_1].hist(
            height=custom_height,
            width=custom_width,
        )
        # Assert
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"

    @pytest.mark.parametrize("method", ["count", "density"])
    @pytest.mark.parametrize("max_cardinality", [3, 5])
    def test_properties_output_type_for_all_options(
        self, dummy_dist_vd, matplotlib_figure_object, max_cardinality, method
    ):
        """
        Test different method types and number of max_cardinality
        """
        # Arrange
        # Act
        result = dummy_dist_vd[COL_NAME_1].hist(
            method=method, max_cardinality=max_cardinality
        )
        # Assert - checking if correct object created
        assert isinstance(result, matplotlib_figure_object), "Wrong object created"


class TestMatplotlibVDFHistogramPlot:
    """
    Testing different attributes of Histogram plot on a vDataFrame
    """

    @pytest.fixture(scope="class")
    def plot_result_vdf(self, dummy_dist_vd):
        """
        Create a histogram plot for vDataFrame
        """
        return dummy_dist_vd.hist(columns=[COL_NAME_1])

    @pytest.fixture(autouse=True)
    def result(self, plot_result_vdf):
        """
        Get the plot results
        """
        self.result = plot_result_vdf

    def test_properties_output_type(self, plotting_library_object):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_xaxis_title(self):
        """
        Testing x-axis title
        """
        # Arrange
        test_title = COL_NAME_1
        # Act
        # Assert - checking x axis label
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_title(self):
        """
        Testing y-axis title
        """
        # Arrange
        test_title = "density"
        # Act
        # Assert - checking y axis label
        assert get_yaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_additional_options_custom_height_and_width(self, dummy_dist_vd):
        """
        Test custom width and height
        """
        # rrange
        custom_height = 6
        custom_width = 7
        # Act
        result = dummy_dist_vd.hist(
            columns=[COL_NAME_1],
            height=custom_height,
            width=custom_width,
        )
        # Assert
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"

    @pytest.mark.skip(reason="Bug needs to be fixed for VDF hist with max cardinality")
    @pytest.mark.parametrize("method", ["min", "max"])
    @pytest.mark.parametrize("max_cardinality", [3, 5])
    def test_properties_output_type_for_all_options(
        self, dummy_dist_vd, plotting_library_object, max_cardinality, method
    ):
        """
        Test different method types and number of max_cardinality
        """
        # Arrange
        # Act
        result = dummy_dist_vd.hist(
            columns=[COL_NAME_1],
            of=COL_OF,
            method=method,
            max_cardinality=max_cardinality,
        )
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"
