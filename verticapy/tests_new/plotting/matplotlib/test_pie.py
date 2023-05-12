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
from verticapy.tests_new.plotting.conftest import get_xaxis_label, get_yaxis_label


# Testing variables
col_name = "check 1"
col_name_2 = "check 2"
all_elements = {"1", "0", "A", "B", "C"}
parents = ["0", "1"]
children = ["A", "B", "C"]


@pytest.fixture(scope="class")
def plot_result(dummy_vd):
    return dummy_vd[col_name].pie()


@pytest.fixture(scope="class")
def plot_result_2(dummy_vd):
    return dummy_vd.pie([col_name, col_name_2])


@pytest.fixture(scope="class")
def plotting_library_object(matplotlib_figure_object):
    return matplotlib_figure_object


class TestMatplotlibSimplePiePlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result

    def test_properties_output_type_for(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_plot_type_wedges(self, dummy_vd):
        # Arrange
        # Act
        # Assert - check value corresponding to 0s
        assert len(self.result.patches) > 1

    def test_properties_labels(self, dummy_vd):
        # Arrange
        # Act
        # Assert - check value corresponding to 0s
        assert set(
            [
                self.result.get_legend().get_texts()[i].get_text()
                for i in range(len(self.result.get_legend().get_texts()))
            ]
        ) == set(dummy_vd.to_pandas()[col_name].unique())

    @pytest.mark.parametrize("kind", ["donut", "rose"])
    @pytest.mark.parametrize("max_cardinality", [2, 4])
    def test_properties_output_type_for_all_options(
        self,
        dummy_vd,
        plotting_library_object,
        kind,
        max_cardinality,
    ):
        # Arrange
        # Act
        result = dummy_vd[col_name].pie(kind=kind, max_cardinality=max_cardinality)
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"


class TestMatplotlibNestedPiePlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result_2):
        self.result = plot_result_2

    def test_properties_type(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_plot_type_wedges(self, dummy_vd):
        # Arrange
        # Act
        # Assert - check value corresponding to 0s
        assert len(self.result.patches) > 2
