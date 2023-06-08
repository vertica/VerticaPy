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

# VerticaPy
from verticapy.learn.delphi import AutoML
from verticapy.learn.ensemble import RandomForestClassifier


# Expensive models
@pytest.fixture(name="champion_challenger_plot", scope="package")
def load_champion_Challenger_plot(schema_loader, dummy_dist_vd):
    COL_NAME_1 = "binary"
    COL_NAME_2 = "0"
    model = AutoML(f"{schema_loader}.model_automl", lmax=10, print_info=False)
    model.fit(
        dummy_dist_vd,
        [
            COL_NAME_1,
        ],
        COL_NAME_2,
    )
    yield model
    model.drop()


@pytest.fixture(name="randon_forest_model_result", scope="module")
def load_random_forest_model(schema_loader, dummy_dist_vd):
    """
    Load the Random Forest Classifier model
    """
    COL_NAME_1 = "0"
    COL_NAME_2 = "1"
    BY_COL = "binary"
    model = RandomForestClassifier(f"{schema_loader}.random_forest_plot_test")
    model.drop()
    model.fit(dummy_dist_vd, [COL_NAME_1, COL_NAME_2], BY_COL)
    yield model
    model.drop()
