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
from verticapy.learn.decomposition import PCA
from verticapy.tests_new.plotting.conftest import (
    get_xaxis_label,
    get_yaxis_label,
)

# Testing variables
COL_NAME_1 = "X"
COL_NAME_2 = "Y"
COL_NAME_3 = "Z"


class TestHighchartsMachineLearningPCACirclePlot(BasicPlotTests):
    """
    Testing different attributes of PCA circle plot
    """

    @pytest.fixture(autouse=True)
    def model(self, schema_loader, dummy_scatter_vd):
        """
        Load test model
        """
        model = PCA(f"{schema_loader}.pca_circle_test")
        model.drop()
        model.fit(dummy_scatter_vd[COL_NAME_1, COL_NAME_2, COL_NAME_3])
        self.model = model
        yield
        model.drop()

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [
            "Dim1 (80%)",
            "Dim2 (20%)",
        ]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.model.plot_circle,
            {},
        )

    def test_properties_xaxis_label(self):
        """
        Testing x-axis label
        """
        # Arrange
        test_title = "Dim1"
        # Act
        # Assert
        assert test_title in get_xaxis_label(self.result), "X axis label incorrect"

    def test_properties_yaxis_label(self):
        """
        Testing y-axis title
        """
        # Arrange
        test_title = "Dim2"
        # Act
        # Assert
        assert test_title in get_yaxis_label(self.result), "Y axis label incorrect"
