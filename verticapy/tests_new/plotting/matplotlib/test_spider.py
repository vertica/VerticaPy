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
col_name_1 = "cats"
by_col = "binary"

# Vertica
from verticapy.tests_new.plotting.conftest import get_xaxis_label, get_yaxis_label


@pytest.fixture(scope="class")
def plotting_library_object(matplotlib_figure_object):
    return matplotlib_figure_object


@pytest.fixture(scope="class")
def plot_result(dummy_dist_vd):
    return dummy_dist_vd[col_name_1].spider()


@pytest.fixture(scope="class")
def plot_result_2(dummy_dist_vd):
    return dummy_dist_vd[col_name_1].spider(by=by_col)


class TestVDFSpiderPlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result

    @pytest.fixture(autouse=True)
    def result_2(self, plot_result_2):
        self.by_result = plot_result_2

    def test_properties_output_type(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_output_type_for_multiplot(
        self, plotting_library_object, dummy_dist_vd
    ):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(
            self.by_result, plotting_library_object
        ), "Wrong object created"

    def test_properties_multiple_plots_produced_for_multiplot(self, dummy_dist_vd):
        # Arrange
        number_of_plots = 2
        # Act
        # Assert
        assert (
            len(
                self.by_result.get_subplotspec()
                .get_topmost_subplotspec()
                .get_gridspec()
                .get_geometry()
            )
            == number_of_plots
        ), "Two traces not produced for two classes of binary"
