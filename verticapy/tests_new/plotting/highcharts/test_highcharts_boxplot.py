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
from verticapy.tests_new.plotting.conftest import get_xaxis_label

# Testing variables
COL_NAME_1 = "0"
COL_NAME_2 = "binary"


class TestHighchartsVDCBoxPlot:
    """
    Testing different attributes of Box plot on a vDataColumn
    """

    @pytest.fixture(scope="class")
    def plot_result(self, dummy_dist_vd):
        """
        Create a box plot for vDataColumn
        """
        return dummy_dist_vd[COL_NAME_1].boxplot()

    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        """
        Get the plot results
        """
        self.result = plot_result

    def test_properties_output_type(self, plotting_library_object):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "wrong object crated"

    @pytest.mark.skip(reason="The plot does not have label on x-axis yet")
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
        test_title = "0"
        # Act
        # Assert - checking y axis label
        assert (
            self.result.options["xAxis"].categories[0] == test_title
        ), "X axis label incorrect"


class TestHighchartsParitionVDCBoxPlot:
    """
    Testing different attributes of Box plot on a vDataColumn using "by" attribute
    """

    @pytest.fixture(scope="class")
    def plot_result_2(self, dummy_dist_vd):
        """
        Create a box plot using "by" attribute for vDataColumn
        """
        return dummy_dist_vd[COL_NAME_1].boxplot(by=COL_NAME_2)

    @pytest.fixture(autouse=True)
    def result(self, plot_result_2):
        """
        Get the plot results
        """
        self.result = plot_result_2

    def test_properties_output_type(self, plotting_library_object):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "wrong object crated"


class TestHighchartsVDFBoxPlot:
    """
    Testing different attributes of Box plot on a vDataFrame
    """

    @pytest.fixture(scope="class")
    def plot_result_vdf(self, dummy_dist_vd):
        """
        Create a box plot for vDataFrame
        """
        return dummy_dist_vd.boxplot(columns=[COL_NAME_1])

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
        assert isinstance(self.result, plotting_library_object), "wrong object crated"

    @pytest.mark.skip(reason="The plot does not have label on x-axis yet")
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
        test_title = "0"
        # Act
        # Assert - checking y axis label
        assert (
            self.result.options["xAxis"].categories[0] == test_title
        ), "X axis label incorrect"
