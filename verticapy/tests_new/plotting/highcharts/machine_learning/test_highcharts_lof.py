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
from verticapy.tests_new.plotting.conftest import BasicPlotTests
from verticapy.learn.neighbors import LocalOutlierFactor


# Testing variables
COL_NAME_1 = "X"
COL_NAME_2 = "Y"
COL_NAME_3 = "Z"


class TestHighchartsMachineLearningLOFPlot2D(BasicPlotTests):
    """
    Testing different attributes of 2D LOF plot
    """

    @pytest.fixture(autouse=True)
    def model(self, dummy_scatter_vd, schema_loader):
        """
        Load test model
        """
        model = LocalOutlierFactor(f"{schema_loader}.lof_test")
        model.fit(dummy_scatter_vd, [COL_NAME_1, COL_NAME_2])
        self.model = model
        yield
        model.drop()

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
        return (
            self.model.plot,
            {},
        )


@pytest.mark.skip(reason="Currently highchart only supports 2D plot")
class TestHighchartsMachineLearningLOFPlot3D(BasicPlotTests):
    """
    Testing different attributes of 3D LOF plot
    """

    @pytest.fixture(autouse=True)
    def model(self, dummy_scatter_vd, schema_loader):
        """
        Load test model
        """
        model = LocalOutlierFactor(f"{schema_loader}.lof_test")
        model.fit(dummy_scatter_vd, [COL_NAME_1, COL_NAME_2, COL_NAME_3])
        self.model = model
        yield
        model.drop()

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [COL_NAME_1, COL_NAME_2, COL_NAME_3]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.model.plot,
            {},
        )
