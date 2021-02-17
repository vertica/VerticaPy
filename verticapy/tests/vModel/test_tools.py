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
from verticapy import vDataFrame, set_option, vertica_conn
from verticapy.learn.model_selection import *
from verticapy.learn.linear_model import *
from verticapy.learn.naive_bayes import *
from verticapy.learn.ensemble import *
from verticapy.learn.tree import *
from verticapy.learn.svm import *
from verticapy.learn.cluster import *
from verticapy.learn.neighbors import *
from verticapy.learn.decomposition import *
from verticapy.learn.preprocessing import *
from verticapy.learn.tsa import *

import matplotlib.pyplot as plt

set_option("print_info", False)
set_option("random_state", 0)

@pytest.fixture(scope="module")
def titanic_vd(base):
    from verticapy.datasets import load_titanic

    titanic = load_titanic(cursor=base.cursor)
    yield titanic
    with warnings.catch_warnings(record=True) as w:
        drop(
            name="public.titanic", cursor=base.cursor,
        )


class TestTools:
    def test_does_model_exist(self, base, titanic_vd):
        model = LinearRegression("model_test", cursor=base.cursor,)
        model.drop()
        assert does_model_exist("model_test", base.cursor) == False
        model.fit(titanic_vd, ["age", "fare"], "survived",)
        assert does_model_exist("model_test", base.cursor) == True
        assert does_model_exist("model_test", base.cursor, return_model_type=True).lower() == "linear_regression"
        model.drop()

    def test_load_model(self, base, titanic_vd):
        try:
            create_verticapy_schema(iris_vd._VERTICAPY_VARIABLES_["cursor"])
        except:
            pass
        # VAR
        model = VAR("model_test", cursor=base.cursor, max_iter=100)
        model.drop()
        model.fit(titanic_vd, ["fare"], "age",)
        result = load_model("model_test", base.cursor)
        assert isinstance(result, VAR) and result.get_params()["max_iter"] == 100
        model.drop()
        # SARIMAX
        model = SARIMAX("model_test", cursor=base.cursor, max_iter=100)
        model.drop()
        model.fit(titanic_vd, "fare", "age",)
        result = load_model("model_test", base.cursor)
        assert isinstance(result, SARIMAX) and result.get_params()["max_iter"] == 100
        model.drop()
        # Normalizer
        model = Normalizer("model_test", cursor=base.cursor, method='minmax')
        model.drop()
        model.fit(titanic_vd, ["age", "fare"],)
        result = load_model("model_test", base.cursor)
        assert isinstance(result, Normalizer) and result.get_params()["method"] == 'minmax'
        model.drop()
        # Normalizer
        model = Normalizer("model_test", cursor=base.cursor, method='minmax')
        model.drop()
        model.fit(titanic_vd, ["age", "fare"],)
        result = load_model("model_test", base.cursor)
        assert isinstance(result, Normalizer) and result.get_params()["method"] == 'minmax'
        model.drop()
        # OneHotEncoder
        model = OneHotEncoder("model_test", cursor=base.cursor,)
        model.drop()
        model.fit(titanic_vd, ["sex", "embarked"],)
        result = load_model("model_test", base.cursor)
        assert isinstance(result, OneHotEncoder)
        model.drop()
        # LOF
        model = LocalOutlierFactor("model_test", cursor=base.cursor, p=3)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"],)
        result = load_model("model_test", base.cursor)
        assert isinstance(result, LocalOutlierFactor) and result.get_params()["p"] == 3
        model.drop()
        # DBSCAN
        model = DBSCAN("model_test", cursor=base.cursor, p=3)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"],)
        result = load_model("model_test", base.cursor)
        assert isinstance(result, DBSCAN) and result.get_params()["p"] == 3
        model.drop()
        # PCA
        model = PCA("model_test", cursor=base.cursor,)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"],)
        result = load_model("model_test", base.cursor)
        assert isinstance(result, PCA)
        model.drop()
        # SVD
        model = SVD("model_test", cursor=base.cursor,)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"],)
        result = load_model("model_test", base.cursor)
        assert isinstance(result, SVD)
        model.drop()
        # LinearRegression
        model = LinearRegression("model_test", cursor=base.cursor, tol=1e-88)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived",)
        result = load_model("model_test", base.cursor)
        assert result.get_params()["tol"] == 1e-88
        assert isinstance(result, LinearRegression) and result.get_params()["penalty"] == 'none'
        model.drop()
        # ElasticNet
        model = ElasticNet("model_test", cursor=base.cursor, tol=1e-88)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived",)
        result = load_model("model_test", base.cursor)
        assert result.get_params()["tol"] == 1e-88
        assert isinstance(result, ElasticNet) and result.get_params()["penalty"] == 'enet'
        model.drop()
        # Lasso
        model = Lasso("model_test", cursor=base.cursor, tol=1e-88)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived",)
        result = load_model("model_test", base.cursor)
        assert result.get_params()["tol"] == 1e-88
        assert isinstance(result, Lasso) and result.get_params()["penalty"] == 'l1'
        model.drop()
        # Ridge
        model = Ridge("model_test", cursor=base.cursor, tol=1e-88)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived",)
        result = load_model("model_test", base.cursor)
        assert result.get_params()["tol"] == 1e-88
        assert isinstance(result, Ridge) and result.get_params()["penalty"] == 'l2'
        model.drop()
        # LogisticRegression
        model = LogisticRegression("model_test", cursor=base.cursor, tol=1e-88, penalty="enet", solver="cgd")
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived",)
        result = load_model("model_test", base.cursor)
        assert result.get_params()["tol"] == 1e-88
        assert result.get_params()["penalty"] == 'enet'
        assert isinstance(result, LogisticRegression) and result.get_params()["solver"] == 'cgd'
        model.drop()
        # DummyTreeClassifier
        model = DummyTreeClassifier("model_test", cursor=base.cursor,)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived",)
        result = load_model("model_test", base.cursor)
        assert isinstance(result, RandomForestClassifier)
        model.drop()
        # DummyTreeRegressor
        model = DummyTreeRegressor("model_test", cursor=base.cursor,)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived",)
        result = load_model("model_test", base.cursor)
        assert isinstance(result, RandomForestRegressor)
        model.drop()
        # DecisionTreeClassifier
        model = DecisionTreeClassifier("model_test", cursor=base.cursor, max_depth=3)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived",)
        result = load_model("model_test", base.cursor)
        assert isinstance(result, RandomForestClassifier) and result.get_params()["max_depth"] == 3
        model.drop()
        # DecisionTreeRegressor
        model = DecisionTreeRegressor("model_test", cursor=base.cursor, max_depth=3)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived",)
        result = load_model("model_test", base.cursor)
        assert isinstance(result, RandomForestRegressor) and result.get_params()["max_depth"] == 3
        model.drop()
        # RandomForestClassifier
        model = RandomForestClassifier("model_test", cursor=base.cursor, n_estimators=33)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived",)
        result = load_model("model_test", base.cursor)
        assert isinstance(result, RandomForestClassifier) and result.get_params()["n_estimators"] == 33
        model.drop()
        # RandomForestRegressor
        model = RandomForestRegressor("model_test", cursor=base.cursor, n_estimators=33)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived",)
        result = load_model("model_test", base.cursor)
        assert isinstance(result, RandomForestRegressor) and result.get_params()["n_estimators"] == 33
        model.drop()
        # XGBoostClassifier
        model = XGBoostClassifier("model_test", cursor=base.cursor, max_ntree=12)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived",)
        result = load_model("model_test", base.cursor)
        assert isinstance(result, XGBoostClassifier) and result.get_params()["max_ntree"] == 12
        model.drop()
        # XGBoostRegressor
        model = XGBoostRegressor("model_test", cursor=base.cursor, max_ntree=12)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived",)
        result = load_model("model_test", base.cursor)
        assert isinstance(result, XGBoostRegressor) and result.get_params()["max_ntree"] == 12
        model.drop()
        # NaiveBayes
        model = NaiveBayes("model_test", cursor=base.cursor, alpha=0.5)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived",)
        result = load_model("model_test", base.cursor)
        assert isinstance(result, NaiveBayes) and result.get_params()["alpha"] == 0.5
        model.drop()
        # LinearSVC
        model = LinearSVC("model_test", cursor=base.cursor, tol=1e-4)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived",)
        result = load_model("model_test", base.cursor)
        assert isinstance(result, LinearSVC) and result.get_params()["tol"] == 1e-4
        model.drop()
        # LinearSVR
        model = LinearSVR("model_test", cursor=base.cursor, tol=1e-4)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived",)
        result = load_model("model_test", base.cursor)
        assert isinstance(result, LinearSVR) and result.get_params()["tol"] == 1e-4
        model.drop()
        # KMeans
        model = KMeans("model_test", cursor=base.cursor, tol=1e-4)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"],)
        result = load_model("model_test", base.cursor)
        assert isinstance(result, KMeans) and result.get_params()["tol"] == 1e-4
        model.drop()
        # BisectingKMeans
        model = BisectingKMeans("model_test", cursor=base.cursor, tol=1e-4)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"],)
        result = load_model("model_test", base.cursor)
        assert isinstance(result, BisectingKMeans) and result.get_params()["tol"] == 1e-4
        model.drop()
        # KNeighborsClassifier
        model = KNeighborsClassifier("model_test", cursor=base.cursor, p=4)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived",)
        result = load_model("model_test", base.cursor)
        assert isinstance(result, KNeighborsClassifier) and result.get_params()["p"] == 4
        model.drop()
        # KNeighborsRegressor
        model = KNeighborsRegressor("model_test", cursor=base.cursor, p=4)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived",)
        result = load_model("model_test", base.cursor)
        assert isinstance(result, KNeighborsRegressor) and result.get_params()["p"] == 4
        model.drop()
        # NearestCentroid
        model = NearestCentroid("model_test", cursor=base.cursor, p=4)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived",)
        result = load_model("model_test", base.cursor)
        assert isinstance(result, NearestCentroid) and result.get_params()["p"] == 4
        model.drop()
        # KernelDensity - BUG
        #model = KernelDensity("model_test", cursor=base.cursor, p=2)
        #model.drop()
        #model.fit(titanic_vd, ["age",],)
        #result = load_model("model_test", base.cursor)
        #assert isinstance(result, KernelDensity) and result.get_params()["p"] == 2
        #model.drop()


    