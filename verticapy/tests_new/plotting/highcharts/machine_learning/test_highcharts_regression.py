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


# Verticapy
from verticapy.learn.linear_model import LinearRegression
from verticapy.tests_new.plotting.conftest import BasicPlotTests


# Testing variables
COL_NAME_1 = "X"
COL_NAME_2 = "Y"


class TestHighchartsMachineLearningRegressionPlot(BasicPlotTests):
    """
    Testing different attributes of Regression plot
    """

    @pytest.fixture(autouse=True)
    def model(self, schema_loader, dummy_scatter_vd):
        """
        Load test model
        """
        model = LinearRegression(f"{schema_loader}.LR_churn")
        model.fit(dummy_scatter_vd, [COL_NAME_1], COL_NAME_2)
        self.model = model
        yield
        model.drop()

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [
            COL_NAME_1,
            COL_NAME_2,
        ]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.model.plot,
            {},
        )

    def test_data_all_scatter_points(self, dummy_scatter_vd):
        """
        Test if all points are plotted
        """
        # Arrange
        # Act
        # Assert
        assert len(self.result.data_temp[1].data) == len(
            dummy_scatter_vd
        ), "Discrepancy between points plotted and total number ofp oints"
