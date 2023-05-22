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


# Testing variables
COL_NAME = "check 1"
COL_NAME_2 = "check 2"
all_elements = {"1", "0", "A", "B", "C"}


class TestHighchartsVDCPiePlot:
    """
    Testing different attributes of Pie plot on a vDataColumn
    """

    @pytest.fixture(scope="class")
    def plot_result(self, dummy_vd):
        """
        Create a pie plot for vDataColumn
        """
        return dummy_vd[COL_NAME].pie()

    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        """
        Get the plot results
        """
        self.result = plot_result

    def test_properties_output_type_for(self, plotting_library_object):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_plot_type_wedges(
        self,
    ):
        """
        Test if multiple sections of pie plot is created
        """
        # Arrange
        # Act
        # Assert - check value corresponding to 0s
        assert len(self.result.data_temp[0].data) > 1

    @pytest.mark.parametrize("kind, max_cardinality", [("donut", 2), ("rose", 4)])
    def test_properties_output_type_for_all_options(
        self,
        dummy_vd,
        plotting_library_object,
        kind,
        max_cardinality,
    ):
        """
        Test different kind and max-cardinality options
        """
        # Arrange
        # Act
        result = dummy_vd[COL_NAME].pie(kind=kind, max_cardinality=max_cardinality)
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"


class TestHighchartsNestedVDFPiePlot:
    """
    Testing different attributes of Pie plot on a vDataFrame
    """

    @pytest.fixture(scope="class")
    def plot_result_2(self, dummy_vd):
        """
        Create a pie plot for vDataFrame
        """
        return dummy_vd.pie([COL_NAME, COL_NAME_2])

    @pytest.fixture(autouse=True)
    def result(self, plot_result_2):
        """
        Get the plot results
        """
        self.result = plot_result_2

    def test_properties_type(self, plotting_library_object):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_plot_type_wedges(
        self,
    ):
        """
        Test if nested plots are produced
        """
        # Arrange
        all_elements_count = sum(len(item.data) for item in self.result.data_temp)
        # Act
        # Assert - check value corresponding to 0s
        assert all_elements_count > 2
