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
COL_NAME = "check 2"
COL_NAME_2 = "check 1"
COL_NAME_VDF_1 = "cats"
COL_NAME_VDF_OF = "0"


class TestHighchartsVDCBarhPlot(BasicPlotTests):
    """
    Testing different attributes of HHorizontal Bar plot on a vDataColumn
    """

    @pytest.fixture(autouse=True)
    def data(self, dummy_vd):
        """
        Load test data
        """
        self.data = dummy_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [COL_NAME, "density"]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data[COL_NAME].barh,
            {},
        )

    def test_data_ratios(self, dummy_vd):
        """
        Test data ratio plotted
        """
        ### Checking if the density was plotted correctly
        nums = dummy_vd.to_pandas()[COL_NAME].value_counts()
        total = len(dummy_vd)
        assert set(self.result.data_temp[0].data).issubset(
            set([nums["A"] / total, nums["B"] / total, nums["C"] / total])
        )

    def test_all_categories_created(self):
        """
        Test all categories
        """
        assert set(self.result.options["xAxis"].categories).issubset(
            set(["A", "B", "C"])
        )

    def test_additional_options_bargap(self, dummy_vd):
        """
        Test bargap option
        """
        # Arrange
        # Act
        result = dummy_vd[COL_NAME].barh(
            bargap=0.5,
        )
        # Assert - checking if correct object created
        assert result.data_temp[0].pointPadding == 0.25, "Custom bargap not working"

    # @pytest.mark.parametrize("max_cardinality", [1, 2])
    @pytest.mark.parametrize(
        "max_cardinality, method", [(1, "mean"), (1, "max"), (2, "sum")]
    )
    def test_properties_output_type_for_all_options(
        self,
        dummy_vd,
        method,
        plotting_library_object,
        max_cardinality,
    ):
        """
        Test max_cardinality and method combination
        """
        # Arrange
        # Act
        result = dummy_vd[COL_NAME].barh(
            method=method,
            of=COL_NAME_2,
            max_cardinality=max_cardinality,
        )
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"


class TestHighchartsVDFBarhPlot(BasicPlotTests):
    """
    Testing different attributes of HHorizontal Bar plot on a vDataFrame
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
        return [COL_NAME_VDF_1, "density"]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data.barh,
            {"columns": COL_NAME_VDF_1},
        )

    def test_data_ratios(self, dummy_dist_vd):
        """
        Test data plotted
        """
        ### Checking if the density was plotted correctly
        nums = dummy_dist_vd.to_pandas()[COL_NAME_VDF_1].value_counts()
        total = len(dummy_dist_vd)
        assert set(self.result.data_temp[0].data).issubset(
            set([nums["A"] / total, nums["B"] / total, nums["C"] / total])
        )

    def test_all_categories_created(self):
        """
        Test all categories
        """
        assert set(self.result.options["xAxis"].categories).issubset(
            set(["A", "B", "C"])
        )

    @pytest.mark.parametrize(
        "of_col, method", [(COL_NAME_VDF_OF, "min"), (COL_NAME_VDF_OF, "max")]
    )
    def test_properties_output_type_for_all_options(
        self,
        dummy_dist_vd,
        plotting_library_object,
        of_col,
        method,
    ):
        """
        Test of and method combination
        """
        # Arrange
        # Act
        result = dummy_dist_vd[COL_NAME_VDF_1].barh(
            of=of_col,
            method=method,
        )
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"
