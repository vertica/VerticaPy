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
from verticapy.learn.causal_model import PC

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
    titanic["boat"].fillna(method='0ifnull')
    yield titanic
    drop(name="public.titanic",)


@pytest.fixture(scope="module")
def model(titanic_vd):
    model_class = PC(titanic_vd)
    yield model_class
    model_class.drop()    

class TestPC:
    def test_model(self, model):
        model.reformat(max_unique=5, method='same_freq')
        res = model.build_skeleton(method='stable', max_cond_vars=2, significance_level=0.01)
        assert len(res) == 2
        (skeleton, sepsets) = res   
        temp = {'pclass': {'age', 'fare'},
                'survived': {'boat', 'sex'},
                'sex': {'age', 'fare', 'survived'},
                'age': {'fare', 'parch', 'pclass', 'sex', 'sibsp'},
                'sibsp': {'age', 'family_size', 'parch'},
                'parch': {'age', 'family_size', 'sibsp'},
                'fare': {'age', 'pclass', 'sex'},
                'boat': {'survived'},
                'family_size': {'parch', 'sibsp'}}
        assert skeleton == temp
        temp = set([
                frozenset({'parch', 'pclass'}),
                frozenset({'age', 'survived'}),
                frozenset({'age', 'boat'}),
                frozenset({'boat', 'sibsp'}),
                frozenset({'fare', 'sibsp'}),
                frozenset({'parch', 'survived'}),
                frozenset({'family_size', 'survived'}),
                frozenset({'fare', 'survived'}),
                frozenset({'sibsp', 'survived'}),
                frozenset({'boat', 'sex'}),
                frozenset({'age', 'family_size'}),
                frozenset({'boat', 'parch'}),
                frozenset({'parch', 'sex'}),
                frozenset({'boat', 'family_size'}),
                frozenset({'boat', 'pclass'}),
                frozenset({'pclass', 'survived'}),
                frozenset({'family_size', 'pclass'}),
                frozenset({'sex', 'sibsp'}),
                frozenset({'fare', 'parch'}),
                frozenset({'family_size', 'fare'}),
                frozenset({'pclass', 'sex'}),
                frozenset({'family_size', 'sex'}),
                frozenset({'boat', 'fare'}),
                frozenset({'pclass', 'sibsp'})])
        assert set(sepsets.keys()) == temp
        assert model.pdag == None
        model.fit(method='stable', max_cond_vars=2, significance_level=0.01)
        assert model.skeleton == skeleton
        assert model.sepsets == sepsets
        res = model.build_pdag()
        assert model.pdag != None
        assert model.pdag == res
        temp = {'pclass': {'age', 'fare'},
                'survived': {'boat', 'sex'},
                'sex': set(),
                'age': set(),
                'sibsp': {'parch'},
                'parch': {'age', 'family_size', 'sibsp'},
                'fare': {'age', 'pclass', 'sex'},
                'boat': {'survived'},
                'family_size': {'parch', 'sibsp'}}
        assert res == temp
        
        
