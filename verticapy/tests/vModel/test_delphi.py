# (c) Copyright [2018-2021] Micro Focus or one of its affiliates.
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

import pytest, warnings, os, verticapy
from verticapy import set_option, current_cursor
from verticapy.learn.delphi import *

import matplotlib.pyplot as plt

set_option("print_info", False)
set_option("random_state", 0)


@pytest.fixture(scope="module")
def amazon_vd():
    from verticapy.datasets import load_amazon

    amazon = load_amazon()
    yield amazon
    with warnings.catch_warnings(record=True) as w:
        drop(name="public.amazon",)

@pytest.fixture(scope="module")
def titanic_vd():
    from verticapy.datasets import load_titanic

    titanic = load_titanic()
    yield titanic
    with warnings.catch_warnings(record=True) as w:
        drop(name="public.titanic",)


@pytest.fixture(scope="module")
def winequality_vd():
    from verticapy.datasets import load_winequality

    winequality = load_winequality()
    yield winequality
    with warnings.catch_warnings(record=True) as w:
        drop(name="public.winequality",)

class TestDelphi:

    def test_AutoML(self, titanic_vd):
        model = AutoML("AutoML_test_ml", )
        model.drop()
        model.fit(titanic_vd, y="survived")
        assert model.model_grid_["avg_score"][0] < 0.1
        assert len(model.plot().get_default_bbox_extra_artists()) < 30
        plt.close("all")
        assert len(model.plot("stepwise").get_default_bbox_extra_artists()) < 200
        plt.close("all")
        model.drop()

    def test_AutoDataPrep(self, titanic_vd, amazon_vd):
        model = AutoDataPrep("AutoML_test_dp", )
        model.drop()
        model.fit(titanic_vd)
        assert model.final_relation_.shape() == (1234, 56)
        model.drop()
        model2 = AutoDataPrep("AutoML_test_dp", num_method="same_freq")
        model2.drop()
        model2.fit(titanic_vd)
        assert model2.final_relation_.shape() == (1234, 101)
        model2.drop()
        model3 = AutoDataPrep("AutoML_test_dp", num_method="same_width", na_method="drop", apply_pca=True)
        model3.drop()
        model3.fit(titanic_vd)
        assert model3.final_relation_.shape() == (112, 122)
        model3.drop()
        model4 = AutoDataPrep("AutoML_test_dp", )
        model4.drop()
        model4.fit(amazon_vd)
        assert model4.final_relation_.shape() == (6318, 3)
        model4.drop()

    def test_AutoClustering(self, titanic_vd):
        model = AutoClustering("AutoML_test_cluster", )
        model.drop()
        model.fit(titanic_vd,)
        assert model.model_.parameters["n_cluster"] < 100
        model.drop()

