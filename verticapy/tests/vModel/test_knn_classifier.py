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

import pytest, warnings, sys, os, verticapy
from verticapy.learn.neighbors import KNeighborsClassifier
from verticapy import drop, set_option, vertica_conn, create_verticapy_schema
import matplotlib.pyplot as plt

set_option("print_info", False)


@pytest.fixture(scope="module")
def titanic_vd(base):
    from verticapy.datasets import load_titanic

    titanic = load_titanic(cursor=base.cursor)
    yield titanic
    with warnings.catch_warnings(record=True) as w:
        drop(name="public.titanic", cursor=base.cursor)


@pytest.fixture(scope="module")
def model(base, titanic_vd):
    try:
        create_verticapy_schema(base.cursor)
    except:
        pass
    model_class = KNeighborsClassifier("knn_model_test", cursor=base.cursor)
    model_class.drop()
    model_class.fit(
        "public.titanic", ["age", "fare",], "survived"
    )
    yield model_class
    model_class.drop()


class TestKNeighborsClassifier:
    def test_repr(self, model):
        assert "Additional Info" in model.__repr__()
        model_repr = KNeighborsClassifier("model_repr", model.cursor)
        model_repr.drop()
        assert model_repr.__repr__() == "<KNeighborsClassifier>"

    def test_contour(self, base, titanic_vd):
        model_test = KNeighborsClassifier("model_contour", cursor=base.cursor)
        model_test.drop()
        model_test.fit(
            titanic_vd,
            ["age", "fare",],
            "survived",
        )
        result = model_test.contour()
        assert len(result.get_default_bbox_extra_artists()) == 34
        model_test.drop()

    def test_lift_chart(self, model):
        lift_ch = model.lift_chart(nbins=1000)

        assert lift_ch["decision_boundary"][300] == pytest.approx(0.3)
        assert lift_ch["positive_prediction_ratio"][300] == pytest.approx(0.353846153846154)
        assert lift_ch["lift"][300] == pytest.approx(1.81819061441703)
        assert lift_ch["decision_boundary"][900] == pytest.approx(0.9)
        assert lift_ch["positive_prediction_ratio"][900] == pytest.approx(1.0)
        assert lift_ch["lift"][900] == pytest.approx(1.0)
        plt.close("all")

    def test_roc_curve(self, model):
        roc_curve = model.roc_curve(nbins=1000)

        assert roc_curve["threshold"][100] == pytest.approx(0.1)
        assert roc_curve["false_positive"][100] == pytest.approx(1.0)
        assert roc_curve["true_positive"][100] == pytest.approx(1.0)
        assert roc_curve["threshold"][700] == pytest.approx(0.7)
        assert roc_curve["false_positive"][700] == pytest.approx(0.0491803278688525)
        assert roc_curve["true_positive"][700] == pytest.approx(0.353846153846154)
        plt.close("all")

    def test_prc_curve(self, model):
        prc_curve = model.prc_curve(nbins=1000)

        assert prc_curve["threshold"][100] == pytest.approx(0.099)
        assert prc_curve["recall"][100] == pytest.approx(1.0)
        assert prc_curve["precision"][100] == pytest.approx(0.477356181150551)
        assert prc_curve["threshold"][700] == pytest.approx(0.699)
        assert prc_curve["recall"][700] == pytest.approx(0.353846153846154)
        assert prc_curve["precision"][700] == pytest.approx(0.867924528301887)
        plt.close("all")

    def test_cutoff_curve(self, model):
        cutoff_curve = model.cutoff_curve(nbins=1000)

        assert cutoff_curve["threshold"][100] == pytest.approx(0.1)
        assert cutoff_curve["false_positive"][100] == pytest.approx(1.0)
        assert cutoff_curve["true_positive"][100] == pytest.approx(1.0)
        assert cutoff_curve["threshold"][700] == pytest.approx(0.7)
        assert cutoff_curve["false_positive"][700] == pytest.approx(0.0491803278688525)
        assert cutoff_curve["true_positive"][700] == pytest.approx(0.353846153846154)
        plt.close("all")

    def test_deploySQL(self, model):
        expected_sql = '(SELECT row_id, "age", "fare", "survived", predict_neighbors, COUNT(*) / 5 AS proba_predict FROM (SELECT x."age", x."fare", x."survived", ROW_NUMBER() OVER(PARTITION BY x."age", x."fare", row_id ORDER BY POWER(POWER(ABS(x."age" - y."age"), 2) + POWER(ABS(x."fare" - y."fare"), 2), 1 / 2)) AS ordered_distance, y."survived" AS predict_neighbors, row_id FROM (SELECT *, ROW_NUMBER() OVER() AS row_id FROM public.titanic WHERE "age" IS NOT NULL AND "fare" IS NOT NULL) x CROSS JOIN (SELECT * FROM public.titanic WHERE "age" IS NOT NULL AND "fare" IS NOT NULL) y) z WHERE ordered_distance <= 5 GROUP BY "age", "fare", "survived", row_id, predict_neighbors) kneighbors_table'
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_drop(self, base):
        model_test = KNeighborsClassifier("model_test_drop", cursor=base.cursor)
        model_test.drop()
        model_test.fit("public.titanic", ["age",], "survived")
        base.cursor.execute(
            "SELECT model_name FROM verticapy.models WHERE model_name IN ('model_test_drop', '\"model_test_drop\"')"
        )
        assert base.cursor.fetchone()[0] in ('model_test_drop', '"model_test_drop"')

        model_test.drop()
        base.cursor.execute(
            "SELECT model_name FROM verticapy.models WHERE model_name IN ('model_test_drop', '\"model_test_drop\"')"
        )
        assert base.cursor.fetchone() is None

    def test_get_params(self, model):
        assert model.get_params() == {'n_neighbors': 5, 'p': 2}

    def test_get_predicts(self, titanic_vd, model):
        titanic_copy = titanic_vd.copy()
        titanic_copy = model.predict(
            titanic_copy,
            X=["age", "fare",],
            name="predicted_quality",
        )

        assert titanic_copy["predicted_quality"].mean() == pytest.approx(
            0.461199510403917, abs=1e-6
        )

    def test_classification_report(self, model):
        cls_rep1 = model.classification_report().transpose()

        assert cls_rep1["auc"][0] == pytest.approx(0.7529724373986667)
        assert cls_rep1["prc_auc"][0] == pytest.approx(0.7776321621297582)
        assert cls_rep1["accuracy"][0] == pytest.approx(0.6658506731946144)
        assert cls_rep1["log_loss"][0] == pytest.approx(0.248241359319007)
        assert cls_rep1["precision"][0] == pytest.approx(0.8679245283018868)
        assert cls_rep1["recall"][0] == pytest.approx(0.35384615384615387)
        assert cls_rep1["f1_score"][0] == pytest.approx(0.5157548811134738)
        assert cls_rep1["mcc"][0] == pytest.approx(0.38437795748893316)
        assert cls_rep1["informedness"][0] == pytest.approx(0.3046658259773014)
        assert cls_rep1["markedness"][0] == pytest.approx(0.4849458048976314)
        assert cls_rep1["csi"][0] == pytest.approx(0.3357664233576642)
        assert cls_rep1["cutoff"][0] == pytest.approx(0.6)

    def test_score(self, model):
        assert model.score(cutoff=0.9, method="accuracy") == pytest.approx(0.5691554467564259)
        assert model.score(cutoff=0.1, method="accuracy") == pytest.approx(0.4773561811505508)
        assert model.score(method="best_cutoff") == pytest.approx(0.6)
        assert model.score(method="bm") == pytest.approx(0.0)
        assert model.score(method="csi") == pytest.approx(0.4773561811505508)
        assert model.score(method="f1") == pytest.approx(0.6462303231151615)
        assert model.score(method="logloss") == pytest.approx(0.248241359319007)
        assert model.score(method="mcc") == pytest.approx(0)
        assert model.score(method="mk") == pytest.approx(-0.5226438188494492)
        assert model.score(method="npv") == pytest.approx(0.4773561811505508)
        assert model.score(method="prc_auc") == pytest.approx(0.7776321621297582)
        assert model.score(method="precision") == pytest.approx(0.4773561811505508)
        assert model.score(method="specificity") == pytest.approx(0.0)

    def test_set_cursor(self, model):
        cur = vertica_conn(
            "vp_test_config",
            os.path.dirname(verticapy.__file__) + "/tests/verticaPy_test_tmp.conf",
        ).cursor()
        model.set_cursor(cur)
        model.cursor.execute("SELECT 1;")
        result = model.cursor.fetchone()
        assert result[0] == 1

    def test_set_params(self, model):
        model.set_params({"p": 1})

        assert model.get_params()["p"] == 1

    def test_model_from_vDF(self, base, titanic_vd):
        model_test = KNeighborsClassifier("knn_from_vDF", cursor=base.cursor)
        model_test.drop()
        model_test.fit(titanic_vd, ["age"], "survived")
        assert model_test.score(cutoff=0.9, method="accuracy") == pytest.approx(0.5890710382513661)
        model_test.drop()
