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

import matplotlib.pyplot as plt

set_option("print_info", False)
set_option("random_state", 0)


@pytest.fixture(scope="module")
def amazon_vd(base):
    from verticapy.datasets import load_amazon

    amazon = load_amazon(cursor=base.cursor)
    yield amazon
    with warnings.catch_warnings(record=True) as w:
        drop(
            name="public.amazon", cursor=base.cursor,
        )

@pytest.fixture(scope="module")
def titanic_vd(base):
    from verticapy.datasets import load_titanic

    titanic = load_titanic(cursor=base.cursor)
    yield titanic
    with warnings.catch_warnings(record=True) as w:
        drop(
            name="public.titanic", cursor=base.cursor,
        )


@pytest.fixture(scope="module")
def winequality_vd(base):
    from verticapy.datasets import load_winequality

    winequality = load_winequality(cursor=base.cursor)
    yield winequality
    with warnings.catch_warnings(record=True) as w:
        drop(
            name="public.winequality", cursor=base.cursor,
        )


class TestModelSelection:
    def test_best_k(self, winequality_vd):
        result = best_k(
            "public.winequality",
            ["residual_sugar", "alcohol"],
            cursor=winequality_vd._VERTICAPY_VARIABLES_["cursor"],
            n_cluster=(1, 5),
            init="kmeanspp",
            elbow_score_stop=0.8,
        )
        assert result in [3, 4]
        result = best_k(
            winequality_vd,
            ["residual_sugar", "alcohol"],
            n_cluster=(1, 5),
            init="random",
            elbow_score_stop=0.8,
        )
        assert result in [3, 4]

    def test_cross_validate(self, winequality_vd):
        result = cross_validate(
            LinearRegression(
                "model_test", cursor=winequality_vd._VERTICAPY_VARIABLES_["cursor"],
            ),
            winequality_vd,
            ["residual_sugar", "alcohol"],
            "quality",
            "r2",
            cv=3,
            training_score=True,
        )
        assert result[0]["r2"][3] == pytest.approx(0.21464568751357532, 5e-1)
        assert result[1]["r2"][3] == pytest.approx(0.207040342625429, 5e-1)
        result2 = cross_validate(
            LogisticRegression(
                "model_test", cursor=winequality_vd._VERTICAPY_VARIABLES_["cursor"],
            ),
            "public.winequality",
            ["residual_sugar", "alcohol"],
            "good",
            "auc",
            cv=3,
            training_score=True,
        )
        assert result2[0]["auc"][3] == pytest.approx(0.7604040062168419, 5e-1)
        assert result2[1]["auc"][3] == pytest.approx(0.7749948214599245, 5e-1)
        result3 = cross_validate(
            NaiveBayes(
                "model_test", cursor=winequality_vd._VERTICAPY_VARIABLES_["cursor"],
            ),
            "public.winequality",
            ["residual_sugar", "alcohol"],
            "quality",
            "auc",
            cv=3,
            training_score=True,
            pos_label=7,
        )
        assert result3[0]["auc"][3] == pytest.approx(0.7405650946597986, 5e-1)
        assert result3[1]["auc"][3] == pytest.approx(0.7386519406866139, 5e-1)

    def test_enet_search_cv(self, titanic_vd):
        result = enet_search_cv(titanic_vd, ["age", "fare"], "survived", cursor=titanic_vd._VERTICAPY_VARIABLES_["cursor"], small=True,)
        assert len(result["parameters"]) == 19

    def test_bayesian_search_cv(self, titanic_vd):
        model = LinearRegression("LR_bs_test", titanic_vd._VERTICAPY_VARIABLES_["cursor"])
        model.drop()
        result = bayesian_search_cv(model, titanic_vd, ["age", "fare"], "survived", cursor=titanic_vd._VERTICAPY_VARIABLES_["cursor"])
        assert len(result["parameters"]) == 25
        model = NaiveBayes("NB_bs_test", titanic_vd._VERTICAPY_VARIABLES_["cursor"])
        model.drop()
        result = bayesian_search_cv(model, titanic_vd, ["age", "fare"], "embarked", pos_label="C", cursor=titanic_vd._VERTICAPY_VARIABLES_["cursor"], lmax=4)
        assert len(result["parameters"]) == 14

    def test_randomized_features_search_cv(self, titanic_vd):
        model = LogisticRegression("Logit_fs_test", titanic_vd._VERTICAPY_VARIABLES_["cursor"])
        model.drop()
        result = randomized_features_search_cv(model, titanic_vd, ["age", "fare", "pclass",], "survived",)
        assert len(result["features"]) == 7

    def test_elbow(self, winequality_vd):
        result = elbow(
            "public.winequality",
            ["residual_sugar", "alcohol"],
            cursor=winequality_vd._VERTICAPY_VARIABLES_["cursor"],
            n_cluster=(1, 5),
            init="kmeanspp",
        )
        plt.close("all")
        assert result["Within-Cluster SS"][0] == pytest.approx(0.0)
        assert len(result["Within-Cluster SS"]) == 4
        result2 = elbow(
            winequality_vd,
            ["residual_sugar", "alcohol"],
            cursor=winequality_vd._VERTICAPY_VARIABLES_["cursor"],
            n_cluster=(1, 5),
            init="kmeanspp",
        )
        assert result2["Within-Cluster SS"][0] == pytest.approx(0.0)
        assert len(result2["Within-Cluster SS"]) == 4
        plt.close("all")

    def test_gen_params_grid(self,):
        assert len(gen_params_grid(LogisticRegression("model_test",), lmax=3)) == 3
        assert len(gen_params_grid(LinearSVC("model_test",), lmax=3)) == 3
        assert len(gen_params_grid(LinearSVR("model_test",), lmax=3)) == 3
        assert len(gen_params_grid(ElasticNet("model_test",), lmax=3)) == 3
        assert len(gen_params_grid(Lasso("model_test",), lmax=3)) == 3
        assert len(gen_params_grid(Ridge("model_test",), lmax=3)) == 3
        assert len(gen_params_grid(LinearRegression("model_test",), lmax=3)) == 3
        assert len(gen_params_grid(NaiveBayes("model_test",), lmax=3)) == 3
        assert len(gen_params_grid(RandomForestClassifier("model_test",), lmax=3, nbins=3)) == 3
        assert len(gen_params_grid(RandomForestRegressor("model_test",), lmax=3, nbins=3)) == 3
        assert len(gen_params_grid(XGBoostClassifier("model_test",), lmax=3, nbins=3)) == 3
        assert len(gen_params_grid(XGBoostRegressor("model_test",), lmax=3, nbins=3)) == 3
        assert len(gen_params_grid(DecisionTreeRegressor("model_test",), lmax=3, nbins=3)) == 3
        assert len(gen_params_grid(DecisionTreeClassifier("model_test",), lmax=3, nbins=3)) == 3
        assert len(gen_params_grid(DummyTreeClassifier("model_test",), lmax=3)) == 0
        assert len(gen_params_grid(DummyTreeRegressor("model_test",), lmax=3)) == 0
        assert len(gen_params_grid(KNeighborsClassifier("model_test",), lmax=3, nbins=3)) == 3
        assert len(gen_params_grid(KNeighborsRegressor("model_test",), lmax=3, nbins=3)) == 3
        assert len(gen_params_grid(NearestCentroid("model_test",), lmax=3, nbins=3)) == 3
        assert len(gen_params_grid(KMeans("model_test",), lmax=3, nbins=3)) == 3
        assert len(gen_params_grid(BisectingKMeans("model_test",), lmax=3, nbins=3)) == 3
        assert len(gen_params_grid(DBSCAN("model_test",), lmax=3, nbins=3)) == 3
        assert len(gen_params_grid(LocalOutlierFactor("model_test",), lmax=3, nbins=3)) == 3

    def test_grid_search_cv(self, winequality_vd):
        result = grid_search_cv(
            LogisticRegression(
                "model_test", cursor=winequality_vd._VERTICAPY_VARIABLES_["cursor"],
            ),
            {"solver": ["Newton", "BFGS", "CGD"], "tol": [0.1, 0.01]},
            winequality_vd,
            ["residual_sugar", "alcohol"],
            "good",
            "auc",
            cv=3,
        )
        assert len(result.values) == 6
        assert len(result["parameters"]) == 6

    def test_lift_chart(self, winequality_vd):
        model = LogisticRegression(
            "model_test", cursor=winequality_vd._VERTICAPY_VARIABLES_["cursor"],
        )
        model.drop()
        model.fit("public.winequality", ["residual_sugar", "alcohol"], "good")
        data = winequality_vd.copy()
        data = model.predict(data, name="prediction")
        result = lift_chart(
            "good",
            "prediction",
            data,
            cursor=winequality_vd._VERTICAPY_VARIABLES_["cursor"],
            pos_label=1,
            nbins=30,
        )
        assert result["lift"][0] == pytest.approx(3.53927343297811)
        assert len(result["lift"]) == 31
        model.drop()
        plt.close("all")

    def test_parameter_grid(self,):
        assert parameter_grid({"param1": [1, 2, 3], "param2": ["a", "b", "c"]}) == [{'param1': 1, 'param2': 'a'},
                                                                                    {'param1': 1, 'param2': 'b'},
                                                                                    {'param1': 1, 'param2': 'c'},
                                                                                    {'param1': 2, 'param2': 'a'},
                                                                                    {'param1': 2, 'param2': 'b'},
                                                                                    {'param1': 2, 'param2': 'c'},
                                                                                    {'param1': 3, 'param2': 'a'},
                                                                                    {'param1': 3, 'param2': 'b'},
                                                                                    {'param1': 3, 'param2': 'c'}]

    def test_plot_acf_pacf(self, amazon_vd):
        result = plot_acf_pacf(amazon_vd, ts="date", by=["state"], column="number", p=3)
        plt.close("all")
        assert result["acf"] == [
            pytest.approx(1.0),
            pytest.approx(0.673),
            pytest.approx(0.349),
            pytest.approx(0.165),
        ]
        assert result["pacf"] == [
            pytest.approx(1.0),
            pytest.approx(0.672667529541858),
            pytest.approx(-0.188727403801382),
            pytest.approx(0.022206688265849),
        ]

    def test_prc_curve(self, winequality_vd):
        model = LogisticRegression(
            "model_test", cursor=winequality_vd._VERTICAPY_VARIABLES_["cursor"],
        )
        model.drop()
        model.fit("public.winequality", ["residual_sugar", "alcohol"], "good")
        data = winequality_vd.copy()
        data = model.predict(data, name="prediction")
        result = prc_curve(
            "good",
            "prediction",
            data,
            cursor=winequality_vd._VERTICAPY_VARIABLES_["cursor"],
            pos_label=1,
            nbins=30,
        )
        assert result["precision"][1] == pytest.approx(0.196552254886871)
        assert len(result["precision"]) == 30
        model.drop()
        plt.close("all")

    def test_randomized_search_cv(self, winequality_vd):
        result = randomized_search_cv(
            LogisticRegression(
                "model_test", cursor=winequality_vd._VERTICAPY_VARIABLES_["cursor"],
            ),
            winequality_vd,
            ["residual_sugar", "alcohol"],
            "good",
            "auc",
            cv=3,
            lmax=4,
            print_info=False,
        )
        assert len(result.values) == 6
        assert len(result["parameters"]) == 4

    def test_roc_curve(self, winequality_vd):
        model = LogisticRegression(
            "model_test", cursor=winequality_vd._VERTICAPY_VARIABLES_["cursor"],
        )
        model.drop()
        model.fit("public.winequality", ["residual_sugar", "alcohol"], "good")
        data = winequality_vd.copy()
        data = model.predict(data, name="prediction")
        result = roc_curve(
            "good",
            "prediction",
            data,
            cursor=winequality_vd._VERTICAPY_VARIABLES_["cursor"],
            pos_label=1,
            nbins=30,
        )
        assert result["true_positive"][2] == pytest.approx(0.945967110415035)
        assert len(result["true_positive"]) == 31
        model.drop()
        plt.close("all")

    def test_validation_curve(self, winequality_vd):
        result = validation_curve(
            LogisticRegression(
                "model_test", cursor=winequality_vd._VERTICAPY_VARIABLES_["cursor"],
            ),
            "tol",
            [0.1, 0.01, 0.001],
            winequality_vd,
            ["residual_sugar", "alcohol"],
            "good",
            "auc",
            cv=3,
            ax=None,
        )
        plt.close("all")
        assert len(result["tol"]) == 3
        assert len(result["test_score"]) == 3
        assert len(result.values) == 7

    def test_learning_curve(self, winequality_vd):
        for elem in ["efficiency", "performance", "scalability"]:
            result = learning_curve(
                LogisticRegression(
                    "model_test", cursor=winequality_vd._VERTICAPY_VARIABLES_["cursor"],
                ),
                winequality_vd,
                ["residual_sugar", "alcohol"],
                "good",
                [0.1, 0.33, 0.55,],
                elem,
                "auc",
                cv=3,
                ax=None,
            )
            plt.close("all")
            assert len(result["n"]) == 3

    def test_stepwise(self, titanic_vd):
        titanic = titanic_vd.copy()
        titanic["boat"].fillna(method="0ifnull")
        model = LogisticRegression("Logit_stepwise_test", titanic_vd._VERTICAPY_VARIABLES_["cursor"])
        model.drop()
        result = stepwise(model, titanic, ["age", "fare", "boat", "pclass",], "survived", "bic", "backward", 100, 3, True, "pearson", True, True,)
        assert result["importance"][-1] == pytest.approx(91.17990063924101, 1e-2)
        assert result["importance"][-4] == pytest.approx(8.820099360758984, 1e-2)
        plt.close("all")
        result = stepwise(model, titanic, ["age", "fare", "boat", "pclass",], "survived", "aic", "forward", 100, 3, True, "spearman", True, True,)
        assert result["importance"][-1] == pytest.approx(9.058315001826145, 1e-2)
        assert result["importance"][-4] == pytest.approx(90.94168499817386, 1e-2)
        plt.close("all")
        model = LinearRegression("LR_stepwise_test", titanic_vd._VERTICAPY_VARIABLES_["cursor"])
        model.drop()
        assert result["importance"][-1] == pytest.approx(9.058315001826145, 1e-2)
        assert result["importance"][-4] == pytest.approx(90.94168499817386, 1e-2)
        plt.close("all")
