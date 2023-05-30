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
from verticapy.tests_new.exp.conftest import BasicPlotTests

# Other Modules


# Verticapy
from verticapy.learn.svm import LinearSVC

# Testing variables
COL_NAME_1 = "X"
COL_NAME_2 = "Y"
COL_NAME_3 = "Z"
BY_COL = "Category"


class TestHighchartsMachineLearningSVMClassifier1DPlot(BasicPlotTests):
    """
    Testing different attributes of SVM classifier plot
    """

    @pytest.fixture(autouse=True)
    def model(self, schema_loader, dummy_pred_data_vd):
        """
        Load test model
        """
        model = LinearSVC(name=f"{schema_loader}.SVC")
        model.fit(dummy_pred_data_vd, [COL_NAME_1], BY_COL)
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
            BY_COL,
        ]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.model.plot,
            {},
        )


class TestHighchartsMachineLearningSVMClassifier2DPlot(BasicPlotTests):
    """
    Testing different attributes of SVM classifier plot
    """

    @pytest.fixture(autouse=True)
    def model(self, schema_loader, dummy_pred_data_vd):
        """
        Load test model
        """
        model = LinearSVC(name=f"{schema_loader}.SVC")
        model.fit(dummy_pred_data_vd, [COL_NAME_1, COL_NAME_2], BY_COL)
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


@pytest.mark.skip(reason="3d plot not supported in highcharts")
class TestHighchartsMachineLearningSVMClassifier3DPlot(BasicPlotTests):
    """
    Testing different attributes of SVM classifier plot
    """

    @pytest.fixture(autouse=True)
    def model(self, schema_loader, dummy_pred_data_vd):
        """
        Load test model
        """
        model = LinearSVC(name=f"{schema_loader}.SVC")
        model.fit(
            dummy_pred_data_vd,
            [COL_NAME_1, COL_NAME_2, COL_NAME_3],
            BY_COL,
        )
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
