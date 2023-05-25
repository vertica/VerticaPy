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


# Testing variables
COL_NAME = "check 1"
COL_NAME_2 = "check 2"
all_elements = {"1", "0", "A", "B", "C"}
parents = ["0", "1"]
children = ["A", "B", "C"]


class TestPlotlyVDCPiePlot:
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

    def test_data_0_values(self, dummy_vd):
        """
        Test data for "0" category values
        """
        # Arrange
        # Act
        # Assert - check value corresponding to 0s
        assert (
            self.result.data[0]["values"][0]
            == dummy_vd[dummy_vd[COL_NAME] == 0][COL_NAME].count()
            / dummy_vd[COL_NAME].count()
        )

    def test_data_1_values(self, dummy_vd):
        """
        Test data for "1" category values
        """
        # Arrange
        # Act
        # Assert - check value corresponding to 0s
        assert (
            self.result.data[0]["values"][1]
            == dummy_vd[dummy_vd[COL_NAME] == 1][COL_NAME].count()
            / dummy_vd[COL_NAME].count()
        )

    def test_properties_labels(self, dummy_vd):
        """
        Test plot labels
        """
        # Arrange
        # Act
        # Assert - check value corresponding to 0s
        assert set(self.result.data[0]["labels"]) == set(
            dummy_vd.to_pandas()[COL_NAME].unique()
        )

    def test_all_categories_crated(self):
        """
        Test title
        """
        # Arrange
        # Act
        # Assert - Check Title
        assert self.result.layout["title"]["text"] == COL_NAME

    @pytest.mark.parametrize(
        "attr, max_cardinality", [(["donut", True], 2), (["rose", False], 4)]
    )
    def test_properties_output_type_for_all_options(
        self, dummy_vd, plotting_library_object, attr, max_cardinality
    ):
        """
        Test different kind, exploded and max-cardinality options
        """
        kind = attr[0]
        exploded = attr[1]
        # Arrange
        # Act
        result = dummy_vd[COL_NAME].pie(
            kind=kind, max_cardinality=max_cardinality, exploded=exploded
        )
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"


class TestPLotlyNestedVDFPiePlot:
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

    def test_properties_branch_values(self):
        """
        Test if the branch values are covering all
        """
        # Arrange
        # Act
        # Assert - checking if the branch values are covering all
        assert self.result.data[0]["branchvalues"] == "total"

    def test_data_all_labels_for_nested(self):
        """
        Test if all labels are plotted
        """
        # Arrange
        # Act
        # Assert - checking if all the labels exist
        assert set(self.result.data[0]["labels"]) == all_elements

    @pytest.mark.parametrize("values", children)
    def test_data_check_parent_of_a(self, values):
        """
        Test all parents of "A"
        """
        # Arrange
        # Act
        # Assert - checking the parent of 'A' which is an element of column "check 2"
        assert self.result.data[0]["parents"][
            self.result.data[0]["labels"].index(values)
        ] in [
            "0",
            "1",
        ]

    def test_data_check_parent_of_0(self):
        """
        Test parent of "0"
        """
        # Arrange
        # Act
        # Assert - checking the parent of '0' which is an element of column "check 1"
        assert self.result.data[0]["parents"][
            self.result.data[0]["labels"].index("0")
        ] in [""]

    def test_data_add_up_all_0s_from_children(self, dummy_vd):
        """
        Test if children of 0 all add up to original count
        """
        # Arrange
        # Act
        zero_indices = [
            i for i, x in enumerate(self.result.data[0]["parents"]) if x == "0"
        ]
        # Assert - checking if if all the children elements of 0 add up to its count
        assert sum(list(self.result.data[0]["values"])[i] for i in zero_indices) == len(
            dummy_vd[dummy_vd[COL_NAME] == 0]
        )
