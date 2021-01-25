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

import pytest, warnings
from verticapy import vDataFrame
from verticapy.learn.model_selection import *
from verticapy.learn.linear_model import *
from verticapy.learn.naive_bayes import NaiveBayes
import matplotlib.pyplot as plt

from verticapy import set_option

set_option("print_info", False)
set_option("random_state", 0)


@pytest.fixture(scope="module")
def amazon_vd(base):
    from verticapy.learn.datasets import load_amazon

    amazon = load_amazon(cursor=base.cursor)
    yield amazon
    with warnings.catch_warnings(record=True) as w:
        drop_table(
            name="public.amazon", cursor=base.cursor,
        )

@pytest.fixture(scope="module")
def winequality_vd(base):
    from verticapy.learn.datasets import load_winequality

    winequality = load_winequality(cursor=base.cursor)
    yield winequality
    with warnings.catch_warnings(record=True) as w:
        drop_table(
            name="public.winequality", cursor=base.cursor,
        )

class TestModelSelection:

    def test_best_k(self, winequality_vd):
        result = best_k("public.winequality",
                       ["residual_sugar", "alcohol"],
                       cursor=winequality_vd._VERTICAPY_VARIABLES_["cursor"],
                       n_cluster=(1,5),
                       init="kmeanspp",
                       elbow_score_stop=0.8)
        assert result in [3, 4]
        result = best_k(winequality_vd,
                       ["residual_sugar", "alcohol"],
                       n_cluster=(1,5),
                       init="random",
                       elbow_score_stop=0.8)
        assert result in [3, 4]

    def test_cross_validate(self, winequality_vd):
        result = cross_validate(LinearRegression("model_test", cursor=winequality_vd._VERTICAPY_VARIABLES_["cursor"],),
                                winequality_vd,
                                ["residual_sugar", "alcohol"],
                                "quality",
                                "r2",
                                cv=3,
                                training_score=True,)
        assert result[0]["r2"][3] == pytest.approx(0.21464568751357532, 5e-1)
        assert result[1]["r2"][3] == pytest.approx(0.207040342625429, 5e-1)
        result2 = cross_validate(LogisticRegression("model_test", cursor=winequality_vd._VERTICAPY_VARIABLES_["cursor"],),
                                 "public.winequality",
                                 ["residual_sugar", "alcohol"],
                                 "good",
                                 "auc",
                                 cv=3,
                                 training_score=True,)
        assert result2[0]["auc"][3] == pytest.approx(0.7604040062168419, 5e-1)
        assert result2[1]["auc"][3] == pytest.approx(0.7749948214599245, 5e-1)
        result3 = cross_validate(NaiveBayes("model_test", cursor=winequality_vd._VERTICAPY_VARIABLES_["cursor"],),
                                 "public.winequality",
                                 ["residual_sugar", "alcohol"],
                                 "quality",
                                 "auc",
                                 cv=3,
                                 training_score=True,
                                 pos_label=7,)
        assert result3[0]["auc"][3] == pytest.approx(0.7405650946597986, 5e-1)
        assert result3[1]["auc"][3] == pytest.approx(0.7386519406866139, 5e-1)


    def test_elbow(self, winequality_vd):
        result = elbow("public.winequality",
                       ["residual_sugar", "alcohol"],
                       cursor=winequality_vd._VERTICAPY_VARIABLES_["cursor"],
                       n_cluster=(1,5),
                       init="kmeanspp")
        assert result["Within-Cluster SS"][0] == pytest.approx(0.0)
        assert len(result["Within-Cluster SS"]) == 4
        result2 = elbow(winequality_vd,
                       ["residual_sugar", "alcohol"],
                       cursor=winequality_vd._VERTICAPY_VARIABLES_["cursor"],
                       n_cluster=(1,5),
                       init="kmeanspp")
        assert result2["Within-Cluster SS"][0] == pytest.approx(0.0)
        assert len(result2["Within-Cluster SS"]) == 4
        plt.close()

    def test_grid_search_cv(self, winequality_vd):
        result = grid_search_cv(
            LogisticRegression("model_test", cursor=winequality_vd._VERTICAPY_VARIABLES_["cursor"],),
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
        model = LogisticRegression("model_test", cursor=winequality_vd._VERTICAPY_VARIABLES_["cursor"],)
        model.drop()
        model.fit("public.winequality", 
                  ["residual_sugar", "alcohol"],
                  "good")
        data = winequality_vd.copy()
        data = model.predict(data, name = "prediction")
        result = lift_chart(
            "good",
            "prediction",
            data,
            cursor=winequality_vd._VERTICAPY_VARIABLES_["cursor"],
            pos_label = 1,
            nbins = 30,
        )
        assert result["lift"][0] == pytest.approx(3.53927343297811)
        assert len(result["lift"]) == 31
        model.drop()
        plt.close()

    def test_prc_curve(self, winequality_vd):
        model = LogisticRegression("model_test", cursor=winequality_vd._VERTICAPY_VARIABLES_["cursor"],)
        model.drop()
        model.fit("public.winequality", 
                  ["residual_sugar", "alcohol"],
                  "good")
        data = winequality_vd.copy()
        data = model.predict(data, name = "prediction")
        result = prc_curve(
            "good",
            "prediction",
            data,
            cursor=winequality_vd._VERTICAPY_VARIABLES_["cursor"],
            pos_label = 1,
            nbins = 30,
        )
        assert result["precision"][1] == pytest.approx(0.196552254886871)
        assert len(result["precision"]) == 30
        model.drop()
        plt.close()

    def test_roc_curve(self, winequality_vd):
        model = LogisticRegression("model_test", cursor=winequality_vd._VERTICAPY_VARIABLES_["cursor"],)
        model.drop()
        model.fit("public.winequality", 
                  ["residual_sugar", "alcohol"],
                  "good")
        data = winequality_vd.copy()
        data = model.predict(data, name = "prediction")
        result = roc_curve(
            "good",
            "prediction",
            data,
            cursor=winequality_vd._VERTICAPY_VARIABLES_["cursor"],
            pos_label = 1,
            nbins = 30,
        )
        assert result["true_positive"][2] == pytest.approx(0.945967110415035)
        assert len(result["true_positive"]) == 31
        model.drop()
        plt.close()

    def test_validation_curve(self, winequality_vd):
        result = validation_curve(
            LogisticRegression("model_test", cursor=winequality_vd._VERTICAPY_VARIABLES_["cursor"],),
            "tol",
            [0.1, 0.01, 0.001],
            winequality_vd,
            ["residual_sugar", "alcohol"],
            "good",
            "auc",
            cv=3,
            ax=None,
        )
        assert len(result["tol"]) == 3
        assert len(result["test_score"]) == 3
        assert len(result.values) == 7
        plt.close()
