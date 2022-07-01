# (c) Copyright [2018-2022] Micro Focus or one of its affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Pytest
import pytest

# VerticaPy
from verticapy import drop, set_option
from verticapy.connect import current_cursor
from verticapy.datasets import load_titanic
from verticapy.learn.decomposition import SVD
from verticapy.learn.pcalg import PC

set_option("print_info", False)


@pytest.fixture(scope="module")
def titanic_vd():
    titanic = load_titanic()
    titanic["cabin"].drop()
    titanic["embarked"].drop()
    titanic["home.dest"].drop()
    titanic["body"].drop()
    titanic["ticket"].drop()
    titanic["name"].drop()
    titanic["sex"].label_encode()
    titanic["family_size"] = titanic["parch"] + titanic["sibsp"] + 1
    titanic["fare"].fill_outliers(method = "winsorize", alpha = 0.03)
    titanic["age"].fillna(method = "mean", by = ["pclass", "sex"])
    titanic["boat"].fillna(method='0ifnull');
    yield titanic
    drop(name="public.titanic",)


@pytest.fixture(scope="module")
def model(winequality_vd):
    model_class = PC()
    #model_class.fit("public.winequality", ["citric_acid", "residual_sugar", "alcohol"])
    yield model_class
    #model_class.drop()


class TestSVD:
    def test_repr(self, titanic_vd):
        #assert "SVD" in model.__repr__()
        model_repr = PC()
        res = PC.build_skeleton(titanic_vd)
        assert len(res) == 2
