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

# Other Modules
import matplotlib.pyplot as plt

# VerticaPy
from verticapy import set_option
from verticapy.connect import current_cursor
from verticapy.datasets import load_titanic
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
from verticapy.learn.tools import *

set_option("print_info", False)
set_option("random_state", 0)


@pytest.fixture(scope="module")
def titanic_vd():
    titanic = load_titanic()
    yield titanic
    drop(name="public.titanic")


class TestTools:
    def test_does_model_exist(self, titanic_vd):
        current_cursor().execute("CREATE SCHEMA IF NOT EXISTS load_model_test")
        model = LinearRegression("load_model_test.model_test")
        model.drop()
        assert does_model_exist("load_model_test.model_test") == False
        model.fit(titanic_vd, ["age", "fare"], "survived")
        assert does_model_exist("load_model_test.model_test") == True
        assert (
            does_model_exist(
                "load_model_test.model_test", return_model_type=True
            ).lower()
            == "linear_regression"
        )
        model.drop()
        current_cursor().execute("DROP SCHEMA load_model_test CASCADE")

    def test_load_model(self, titanic_vd):
        create_verticapy_schema()
        current_cursor().execute("CREATE SCHEMA IF NOT EXISTS load_model_test")
        # VAR
        model = VAR("load_model_test.model_test", max_iter=100)
        model.drop()
        model.fit(titanic_vd, ["fare"], "age")
        result = load_model("load_model_test.model_test")
        assert isinstance(result, VAR) and result.get_params()["max_iter"] == 100
        model.drop()
        # SARIMAX
        model = SARIMAX("load_model_test.model_test", max_iter=100)
        model.drop()
        model.fit(titanic_vd, "fare", "age")
        result = load_model("load_model_test.model_test")
        assert isinstance(result, SARIMAX) and result.get_params()["max_iter"] == 100
        model.drop()
        # Normalizer
        model = Normalizer("load_model_test.model_test", method="minmax")
        model.drop()
        model.fit(titanic_vd, ["age", "fare"])
        result = load_model("load_model_test.model_test")
        assert (
            isinstance(result, Normalizer) and result.get_params()["method"] == "minmax"
        )
        model.drop()
        # Normalizer
        model = Normalizer("load_model_test.model_test", method="minmax")
        model.drop()
        model.fit(titanic_vd, ["age", "fare"])
        result = load_model("load_model_test.model_test")
        assert (
            isinstance(result, Normalizer) and result.get_params()["method"] == "minmax"
        )
        model.drop()
        # OneHotEncoder
        model = OneHotEncoder("load_model_test.model_test")
        model.drop()
        model.fit(titanic_vd, ["sex", "embarked"])
        result = load_model("load_model_test.model_test")
        assert isinstance(result, OneHotEncoder)
        model.drop()
        # LOF
        model = LocalOutlierFactor("load_model_test.model_test", p=3)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"])
        result = load_model("load_model_test.model_test")
        assert isinstance(result, LocalOutlierFactor) and result.get_params()["p"] == 3
        model.drop()
        # DBSCAN
        model = DBSCAN("load_model_test.model_test", p=3)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"])
        result = load_model("load_model_test.model_test")
        assert isinstance(result, DBSCAN) and result.get_params()["p"] == 3
        model.drop()
        # PCA
        model = PCA("load_model_test.model_test")
        model.drop()
        model.fit(titanic_vd, ["age", "fare"])
        result = load_model("load_model_test.model_test")
        assert isinstance(result, PCA)
        model.drop()
        # SVD
        model = SVD("load_model_test.model_test")
        model.drop()
        model.fit(titanic_vd, ["age", "fare"])
        result = load_model("load_model_test.model_test")
        assert isinstance(result, SVD)
        model.drop()
        # LinearRegression
        model = LinearRegression("load_model_test.model_test", tol=1e-88)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived")
        result = load_model("load_model_test.model_test")
        assert result.get_params()["tol"] == 1e-88
        assert (
            isinstance(result, LinearRegression)
            and result.get_params()["penalty"] == "none"
        )
        model.drop()
        # ElasticNet
        model = ElasticNet("load_model_test.model_test", tol=1e-88)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived")
        result = load_model("load_model_test.model_test")
        assert result.get_params()["tol"] == 1e-88
        assert (
            isinstance(result, ElasticNet) and result.get_params()["penalty"] == "enet"
        )
        model.drop()
        # Lasso
        model = Lasso("load_model_test.model_test", tol=1e-88)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived")
        result = load_model("load_model_test.model_test")
        assert result.get_params()["tol"] == 1e-88
        assert isinstance(result, Lasso) and result.get_params()["penalty"] == "l1"
        model.drop()
        # Ridge
        model = Ridge("load_model_test.model_test", tol=1e-88)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived")
        result = load_model("load_model_test.model_test")
        assert result.get_params()["tol"] == 1e-88
        assert isinstance(result, Ridge) and result.get_params()["penalty"] == "l2"
        model.drop()
        # LogisticRegression
        model = LogisticRegression(
            "load_model_test.model_test", tol=1e-88, penalty="enet", solver="cgd"
        )
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived")
        result = load_model("load_model_test.model_test")
        assert result.get_params()["tol"] == 1e-88
        assert result.get_params()["penalty"] == "enet"
        assert (
            isinstance(result, LogisticRegression)
            and result.get_params()["solver"] == "cgd"
        )
        model.drop()
        # DummyTreeClassifier
        model = DummyTreeClassifier("load_model_test.model_test")
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived")
        result = load_model("load_model_test.model_test")
        assert isinstance(result, RandomForestClassifier)
        model.drop()
        # DummyTreeRegressor
        model = DummyTreeRegressor("load_model_test.model_test")
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived")
        result = load_model("load_model_test.model_test")
        assert isinstance(result, RandomForestRegressor)
        model.drop()
        # DecisionTreeClassifier
        model = DecisionTreeClassifier("load_model_test.model_test", max_depth=3)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived")
        result = load_model("load_model_test.model_test")
        assert (
            isinstance(result, RandomForestClassifier)
            and result.get_params()["max_depth"] == 3
        )
        model.drop()
        # DecisionTreeRegressor
        model = DecisionTreeRegressor("load_model_test.model_test", max_depth=3)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived")
        result = load_model("load_model_test.model_test")
        assert (
            isinstance(result, RandomForestRegressor)
            and result.get_params()["max_depth"] == 3
        )
        model.drop()
        # RandomForestClassifier
        model = RandomForestClassifier("load_model_test.model_test", n_estimators=33)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived")
        result = load_model("load_model_test.model_test")
        assert (
            isinstance(result, RandomForestClassifier)
            and result.get_params()["n_estimators"] == 33
        )
        model.drop()
        # RandomForestRegressor
        model = RandomForestRegressor("load_model_test.model_test", n_estimators=33)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived")
        result = load_model("load_model_test.model_test")
        assert (
            isinstance(result, RandomForestRegressor)
            and result.get_params()["n_estimators"] == 33
        )
        model.drop()
        # XGBoostClassifier
        model = XGBoostClassifier("load_model_test.model_test", max_ntree=12)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived")
        result = load_model("load_model_test.model_test")
        assert (
            isinstance(result, XGBoostClassifier)
            and result.get_params()["max_ntree"] == 12
        )
        model.drop()
        # XGBoostRegressor
        model = XGBoostRegressor("load_model_test.model_test", max_ntree=12)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived")
        result = load_model("load_model_test.model_test")
        assert (
            isinstance(result, XGBoostRegressor)
            and result.get_params()["max_ntree"] == 12
        )
        model.drop()
        # NaiveBayes
        model = NaiveBayes("load_model_test.model_test", alpha=0.5)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived")
        result = load_model("load_model_test.model_test")
        assert isinstance(result, NaiveBayes) and result.get_params()["alpha"] == 0.5
        model.drop()
        # LinearSVC
        model = LinearSVC("load_model_test.model_test", tol=1e-4)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived")
        result = load_model("load_model_test.model_test")
        assert isinstance(result, LinearSVC) and result.get_params()["tol"] == 1e-4
        model.drop()
        # LinearSVR
        model = LinearSVR("load_model_test.model_test", tol=1e-4)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived")
        result = load_model("load_model_test.model_test")
        assert isinstance(result, LinearSVR) and result.get_params()["tol"] == 1e-4
        model.drop()
        # KMeans
        model = KMeans("load_model_test.model_test", tol=1e-4)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"])
        result = load_model("load_model_test.model_test")
        assert isinstance(result, KMeans) and result.get_params()["tol"] == 1e-4
        model.drop()
        # BisectingKMeans
        model = BisectingKMeans("load_model_test.model_test", tol=1e-4)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"])
        result = load_model("load_model_test.model_test")
        assert (
            isinstance(result, BisectingKMeans) and result.get_params()["tol"] == 1e-4
        )
        model.drop()
        # KNeighborsClassifier
        model = KNeighborsClassifier("load_model_test.model_test", p=4)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived")
        result = load_model("load_model_test.model_test")
        assert (
            isinstance(result, KNeighborsClassifier) and result.get_params()["p"] == 4
        )
        model.drop()
        # KNeighborsRegressor
        model = KNeighborsRegressor("load_model_test.model_test", p=4)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived")
        result = load_model("load_model_test.model_test")
        assert isinstance(result, KNeighborsRegressor) and result.get_params()["p"] == 4
        model.drop()
        # NearestCentroid
        model = NearestCentroid("load_model_test.model_test", p=4)
        model.drop()
        model.fit(titanic_vd, ["age", "fare"], "survived")
        result = load_model("load_model_test.model_test")
        assert isinstance(result, NearestCentroid) and result.get_params()["p"] == 4
        model.drop()
        # KernelDensity - BUG
        # model = KernelDensity("model_test", p=2)
        # model.drop()
        # model.fit(titanic_vd, ["age"])
        # result = load_model("model_test")
        # assert isinstance(result, KernelDensity) and result.get_params()["p"] == 2
        # model.drop()
        current_cursor().execute("DROP SCHEMA load_model_test CASCADE")

    def test_matrix_rotation(self):
        result = matrix_rotation([[0.5, 0.6], [0.1, 0.2]])
        assert result[0][0] == pytest.approx(0.01539405)
        assert result[0][1] == pytest.approx(0.78087324)
        assert result[1][0] == pytest.approx(0.05549495)
        assert result[1][1] == pytest.approx(0.21661097)
        result = matrix_rotation([[0.5, 0.6], [0.1, 0.2]], gamma=0.0)
        assert result[0][0] == pytest.approx(0.0010429389547800816)
        assert result[0][1] == pytest.approx(0.78102427)
        assert result[1][0] == pytest.approx(-0.05092405)
        assert result[1][1] == pytest.approx(0.21773089)
