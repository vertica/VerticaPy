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
COL_NAME_1 = "0"
COL_NAME_2 = "binary"


class TestHighchartsVDFContourPlot(BasicPlotTests):
    """
    Testing different attributes of Contour plot on a vDataFrame
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
        return [COL_NAME_1, COL_NAME_2]

    def create_plot(self):
        """
        Create the plot
        """

        def func(param_a, param_b):
            """
            Arbitrary custom function for testing
            """
            return param_b + param_a * 0

        return (
            self.data.contour,
            {"columns": [COL_NAME_1, COL_NAME_2], "func": func},
        )

    @pytest.mark.parametrize("nbins", [10, 20])
    def test_properties_output_type_for_all_options(
        self, dummy_dist_vd, plotting_library_object, nbins
    ):
        """
        Test different bin sizes
        """

        # Arrange
        def func(param_a, param_b):
            return param_b + param_a * 0

        # Act
        result = dummy_dist_vd.contour([COL_NAME_1, COL_NAME_2], func, nbins=nbins)
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"
