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
import numpy as np

# Verticapy
from verticapy.learn.delphi import AutoML

# Testing variables
col_name_1 = "binary"
col_name_2 = "0"


@pytest.fixture(scope="class")
def plot_result(dummy_dist_vd):
    model = AutoML("model_automl", lmax=10, print_info=False)
    model.fit(
        dummy_dist_vd,
        [
            col_name_1,
        ],
        col_name_2,
    )
    return model.plot()


class TestMachineLearningChampionChallengerPlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result

    def test_properties_output_type(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"
