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
COL_NAME_1 = "value"


class TestHighchartsVDCRangeCurve(BasicPlotTests):
    """
    Testing different attributes of range curve plot on a vDataColumn
    """

    @pytest.fixture(autouse=True)
    def data(self, dummy_date_vd):
        """
        Load test data
        """
        self.data = dummy_date_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [TIME_COL, COL_NAME_1]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data[COL_NAME_1].range_plot,
            {"ts": TIME_COL, "plot_median": True},
        )

    @pytest.mark.parametrize(
        "plot_median, date_range",
        [("True", [1920, None]), ("False", [None, 1950])],
    )
    def test_properties_output_type_for_all_options(
        self,
        dummy_date_vd,
        plotting_library_object,
        plot_median,
        date_range,
    ):
        """
        Test different values for median, start date, and end date
        """
        # Arrange
        # Act
        result = dummy_date_vd[COL_NAME_1].range_plot(
            ts=TIME_COL,
            plot_median=plot_median,
            start_date=date_range[0],
            end_date=date_range[1],
        )
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"


class TestHighchartsVDFRangeCurve(BasicPlotTests):
    """
    Testing different attributes of range curve plot on a vDataFrame
    """

    @pytest.fixture(autouse=True)
    def data(self, dummy_date_vd):
        """
        Load test data
        """
        self.data = dummy_date_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [TIME_COL, COL_NAME_1]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data.range_plot,
            {"columns": [COL_NAME_1], "ts": TIME_COL, "plot_median": True},
        )

    @pytest.mark.parametrize(
        "plot_median, date_range",
        [("True", [1920, None]), ("False", [None, 1950])],
    )
    def test_properties_output_type_for_all_options(
        self,
        dummy_date_vd,
        plotting_library_object,
        plot_median,
        date_range,
    ):
        """
        Tes different values for median, start date, and end date
        """
        # Arrange
        # Act
        result = dummy_date_vd.range_plot(
            columns=[COL_NAME_1],
            ts=TIME_COL,
            plot_median=plot_median,
            start_date=date_range[0],
            end_date=date_range[1],
        )
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"
