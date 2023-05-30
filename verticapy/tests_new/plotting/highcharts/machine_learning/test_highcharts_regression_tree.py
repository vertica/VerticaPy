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
from verticapy.learn.tree import DecisionTreeRegressor
from verticapy.tests_new.plotting.conftest import BasicPlotTests

# Testing variables
COL_NAME_1 = "0"
COL_NAME_2 = "1"


class TestHighchartsMachineLearningRegressionTreePlot(BasicPlotTests):
    """
    Testing different attributes of Regression Tree plot
    """

    @pytest.fixture(autouse=True)
    def model(self, schema_loader, dummy_dist_vd):
        """
        Load test model
        """
        model = DecisionTreeRegressor(name=f"{schema_loader}.model_titanic")
        x_col = COL_NAME_1
        y_col = COL_NAME_2
        model.fit(dummy_dist_vd, x_col, y_col)
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
