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
from ..conftest import BasicPlotTests


# Testing variables
COL_NAME_1 = "binary"
COL_OF = "0"


@pytest.mark.skip(reason="Hist not available in Highcharts currently")
class TestHighchartsVDCHistogramPlot(BasicPlotTests):
    """
    Testing different attributes of Histogram plot on a vDataColumn
    """

    @pytest.fixture(autouse=True)
    def data(self, dummy_dist_vd):
        """
        Load test data
        """
        self.data = dummy_dist_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [COL_NAME_1, "density"]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data[COL_NAME_1].hist,
            {},
        )

    @pytest.mark.parametrize("method", ["count", "density"])
    @pytest.mark.parametrize("max_cardinality", [3, 5])
    def test_properties_output_type_for_all_options(
        self, dummy_dist_vd, plotting_library_object, max_cardinality, method
    ):
        """
        Test different method types and number of max_cardinality
        """
        # Arrange
        # Act
        result = dummy_dist_vd[COL_NAME_1].hist(
            of=COL_OF, method=method, max_cardinality=max_cardinality
        )
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"


@pytest.mark.skip(reason="Hist not available in Highcharts currently")
class TestHighchartsVDFHistogramPlot(BasicPlotTests):
    """
    Testing different attributes of Histogram plot on a vDataFrame
    """

    @pytest.fixture(autouse=True)
    def data(self, dummy_dist_vd):
        """
        Load test data
        """
        self.data = dummy_dist_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [COL_NAME_1, "density"]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data.hist,
            {"columns": COL_NAME_1},
        )

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
