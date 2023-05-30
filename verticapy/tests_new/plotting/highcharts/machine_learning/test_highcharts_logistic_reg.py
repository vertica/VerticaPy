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
from verticapy.tests_new.exp.conftest import BasicPlotTests

# Verticapy
from verticapy.learn.linear_model import LogisticRegression


# Testing variables
COL_NAME_1 = "fare"
COL_NAME_2 = "survived"
COL_NAME_3 = "age"


class TestHighchartsMachineLearningLogisticRegressionPlot2D(BasicPlotTests):
    """
    Testing different attributes of 2D Logisti Regression plot
    """

    @pytest.fixture(autouse=True)
    def model(self, titanic_vd, schema_loader):
        """
        Load test model
        """
        model = LogisticRegression(f"{schema_loader}.lof_test")
        model.fit(titanic_vd, [COL_NAME_1], COL_NAME_2)
        self.model = model
        yield
        model.drop()

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [COL_NAME_1, f"p({COL_NAME_2}=1)"]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.model.plot,
            {},
        )

    @pytest.mark.parametrize("max_nb_points", [50])
    def test_properties_output_type_for_all_options(
        self,
        plotting_library_object,
        max_nb_points,
    ):
        """
        Test different number of maximum points
        """
        # Arrange
        # Act
        result = self.model.plot(
            max_nb_points=max_nb_points,
        )
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"


@pytest.mark.skip(reason="Currently highchart only supports 2D plot")
class TestHighchartsMachineLearningLogisticRegressionPlot3D(BasicPlotTests):
    """
    Testing different attributes of 3D Logisti Regression plot
    """

    @pytest.fixture(autouse=True)
    def model(self, titanic_vd, schema_loader):
        """
        Load test model
        """
        model = LogisticRegression(f"{schema_loader}.lof_test")
        model.fit(titanic_vd, [COL_NAME_1, COL_NAME_3], COL_NAME_2)
        self.model = model
        yield
        model.drop()

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [COL_NAME_1, f"p({COL_NAME_2}=1)", COL_NAME_2]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.model.plot,
            {},
        )
