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

# Vertica
from ..conftest import BasicPlotTests


# Testing variables
COL_NAME_1 = "0"
COL_NAME_2 = "binary"


class TestHighchartsVDCBoxPlot(BasicPlotTests):
    """
    Testing different attributes of Box plot on a vDataColumn
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
        return [COL_NAME_1, None]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data[COL_NAME_1].boxplot,
            {},
        )

    @pytest.mark.skip(reason="The plot does not have label on x-axis yet")
    def test_properties_xaxis_label(self):
        """
        Testing x-axis title
        """


class TestHighchartsParitionVDCBoxPlot(BasicPlotTests):
    """
    Testing different attributes of Box plot on a vDataColumn using "by" attribute
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
            self.data[COL_NAME_1].boxplot,
            {"by": COL_NAME_2},
        )

    @pytest.mark.skip(reason="The plot does not have label on x-axis yet")
    def test_properties_xaxis_label(self):
        """
        Testing x-axis title
        """

    @pytest.mark.skip(reason="The plot does not have label on y-axis yet")
    def test_properties_yaxis_label(self):
        """
        Testing x-axis title
        """


class TestHighchartsVDFBoxPlot(BasicPlotTests):
    """
    Testing different attributes of Box plot on a vDataFrame
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
        return [None, COL_NAME_1]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data.boxplot,
            {"columns": COL_NAME_1},
        )

    @pytest.mark.skip(reason="The plot does not have label on x-axis yet")
    def test_properties_xaxis_title(self):
        """
        Testing x-axis title
        """

    def test_properties_yaxis_label(self):
        """
        Testing y-axis title
        """
        # Arrange
        test_title = COL_NAME_1
        # Act
        # Assert - checking y axis label
        assert (
            self.result.options["xAxis"].categories[0] == test_title
        ), "X axis label incorrect"
