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
TIME_COL = "date"
COL_NAME_1 = "values"
COL_NAME_2 = "category"
CAT_OPTION = "A"


class TestHighchartsVDCLinePlot(BasicPlotTests):
    """
    Testing different attributes of Line plot on a vDataColumn
    """

    @pytest.fixture(autouse=True)
    def data(self, dummy_line_data_vd):
        """
        Load test data
        """
        self.data = dummy_line_data_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return ["date", COL_NAME_1]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data[COL_NAME_1].plot,
            {"ts": TIME_COL, "by": COL_NAME_2},
        )

    @pytest.mark.skip(reason="The plot does not have label on y-axis yet")
    def test_properties_yaxis_label(self):
        """
        Testing y-axis title
        """

    def test_properties_output_type_for_one_trace(
        self, dummy_line_data_vd, plotting_library_object
    ):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        result = dummy_line_data_vd[dummy_line_data_vd[COL_NAME_2] == CAT_OPTION][
            COL_NAME_1
        ].plot(ts=TIME_COL)
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"

    def test_data_count_of_all_values(self, dummy_line_data_vd):
        """
        Testing total points
        """
        # Arrange
        total_count = dummy_line_data_vd.shape()[0]
        # Act
        assert (
            len(self.result.data_temp[0].data[0]) * len(self.result.data_temp[0].data)
            == total_count
        ), "The total values in the plot are not equal to the values in the dataframe."

    @pytest.mark.parametrize("kind", ["spline", "area", "step"])
    @pytest.mark.parametrize("start_date", ["1930"])
    def test_properties_output_type_for_all_options(
        self, dummy_line_data_vd, plotting_library_object, start_date, kind
    ):
        """
        Testing different kinds and start date
        """
        # Arrange
        # Act
        result = dummy_line_data_vd[COL_NAME_1].plot(
            ts=TIME_COL, kind=kind, start_date=start_date
        )
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"


class TestHighchartsVDFLinePlot(BasicPlotTests):
    """
    Testing different attributes of Line plot on a vDataFrame
    """

    @pytest.fixture(autouse=True)
    def data(self, dummy_line_data_vd):
        """
        Load test data
        """
        self.data = dummy_line_data_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return ["date", COL_NAME_1]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data[self.data[COL_NAME_2] == CAT_OPTION].plot,
            {"ts": TIME_COL, "columns": COL_NAME_1},
        )
