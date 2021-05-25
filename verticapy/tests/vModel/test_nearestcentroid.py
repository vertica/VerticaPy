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
from verticapy.learn.neighbors import NearestCentroid
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
    model_class = NearestCentroid("nc_model_test", cursor=base.cursor)
    model_class.drop()
    model_class.fit(
        "public.titanic", ["age", "fare",], "survived"
    )
    yield model_class
    model_class.drop()


class TestNearestCentroid:
    def test_repr(self, model):
        assert "Additional Info" in model.__repr__()
        model_repr = NearestCentroid("model_repr", model.cursor)
        model_repr.drop()
        assert model_repr.__repr__() == "<NearestCentroid>"

    def test_contour(self, base, titanic_vd):
        model_test = NearestCentroid("model_contour", cursor=base.cursor)
        model_test.drop()
        model_test.fit(
            titanic_vd,
            ["age", "fare",],
            "survived",
        )
        result = model_test.contour()
        assert len(result.get_default_bbox_extra_artists()) == 40
        model_test.drop()

    def test_lift_chart(self, model):
        lift_ch = model.lift_chart(nbins=1000)

        assert lift_ch["decision_boundary"][300] == pytest.approx(0.3)
        assert lift_ch["positive_prediction_ratio"][300] == pytest.approx(0.176470588235294)
        assert lift_ch["lift"][300] == pytest.approx(1.09170624771648)
        assert lift_ch["decision_boundary"][900] == pytest.approx(0.9)
        assert lift_ch["positive_prediction_ratio"][900] == pytest.approx(0.989769820971867)
        assert lift_ch["lift"][900] == pytest.approx(0.998795077698054)
        plt.close("all")

    def test_roc_curve(self, model):
        roc_curve = model.roc_curve(nbins=1000)

        assert roc_curve["threshold"][100] == pytest.approx(0.1)
        assert roc_curve["false_positive"][100] == pytest.approx(0.991735537190083)
        assert roc_curve["true_positive"][100] == pytest.approx(0.989769820971867)
        assert roc_curve["threshold"][700] == pytest.approx(0.7)
        assert roc_curve["false_positive"][700] == pytest.approx(0.152066115702479)
        assert roc_curve["true_positive"][700] == pytest.approx(0.176470588235294)
        plt.close("all")

    def test_cutoff_curve(self, model):
        cutoff_curve = model.cutoff_curve(nbins=1000)

        assert cutoff_curve["threshold"][100] == pytest.approx(0.1)
        assert cutoff_curve["false_positive"][100] == pytest.approx(0.991735537190083)
        assert cutoff_curve["true_positive"][100] == pytest.approx(0.989769820971867)
        assert cutoff_curve["threshold"][700] == pytest.approx(0.7)
        assert cutoff_curve["false_positive"][700] == pytest.approx(0.152066115702479)
        assert cutoff_curve["true_positive"][700] == pytest.approx(0.176470588235294)
        plt.close("all")

    def test_deploySQL(self, model):
        expected_sql = '(SELECT "age", "fare", "survived", predict_neighbors, (1 - DECODE(distance, 0, 0, (distance / SUM(distance) OVER (PARTITION BY "age", "fare")))) / 1 AS proba_predict, ordered_distance FROM (SELECT x."age", x."fare", ROW_NUMBER() OVER(PARTITION BY x."age", x."fare", row_id ORDER BY POWER(POWER(ABS(x."age" - y."age"), 2) + POWER(ABS(x."fare" - y."fare"), 2), 1 / 2)) AS ordered_distance, POWER(POWER(ABS(x."age" - y."age"), 2) + POWER(ABS(x."fare" - y."fare"), 2), 1 / 2) AS distance, y."survived" AS predict_neighbors, x."survived" FROM (SELECT *, ROW_NUMBER() OVER() AS row_id FROM public.titanic WHERE "age" IS NOT NULL AND "fare" IS NOT NULL) x CROSS JOIN ((SELECT 30.6420462046205 AS "age", 23.4255950191571 AS "fare", 0 AS "survived") UNION ALL (SELECT 29.3936572890026 AS "age", 52.3002593333333 AS "fare", 1 AS "survived")) y) nc_distance_table) neighbors_table'
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_drop(self, base):
        model_test = NearestCentroid("model_test_drop", cursor=base.cursor)
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
        assert model.get_params() == {'p': 2}

    def test_get_predicts(self, titanic_vd, model):
        titanic_copy = titanic_vd.copy()
        titanic_copy = model.predict(
            titanic_copy,
            X=["age", "fare",],
            name="predicted_quality",
        )

        assert titanic_copy["predicted_quality"].mean() == pytest.approx(
            0.475221125041987, abs=1e-6
        )

    def test_classification_report(self, model):
        cls_rep1 = model.classification_report().transpose()

        assert cls_rep1["auc"][0] == pytest.approx(0.5481072055124611)
        assert cls_rep1["prc_auc"][0] == pytest.approx(0.43249617661669015)
        assert cls_rep1["accuracy"][0] == pytest.approx(0.5441767068273092)
        assert cls_rep1["log_loss"][0] == pytest.approx(0.325324671069203)
        assert cls_rep1["precision"][0] == pytest.approx(0.4463373083475298)
        assert cls_rep1["recall"][0] == pytest.approx(0.670076726342711)
        assert cls_rep1["f1_score"][0] == pytest.approx(0.547483115041606)
        assert cls_rep1["mcc"][0] == pytest.approx(0.13190665097885387)
        assert cls_rep1["informedness"][0] == pytest.approx(0.1328866436980829)
        assert cls_rep1["markedness"][0] == pytest.approx(0.1309338853646449)
        assert cls_rep1["csi"][0] == pytest.approx(0.3659217877094972)
        assert cls_rep1["cutoff"][0] == pytest.approx(0.366666666666667)

    def test_score(self, model):
        assert model.score(cutoff=0.9, method="accuracy") == pytest.approx(0.6114457831325302)
        assert model.score(cutoff=0.1, method="accuracy") == pytest.approx(0.39357429718875503)
        assert model.score(method="best_cutoff") == pytest.approx(0.366666666666667)
        assert model.score(method="bm") == pytest.approx(0.0)
        assert model.score(method="csi") == pytest.approx(0.392570281124498)
        assert model.score(method="f1") == pytest.approx(0.5638067772170151)
        assert model.score(method="logloss") == pytest.approx(0.325324671069203)
        assert model.score(method="mcc") == pytest.approx(0)
        assert model.score(method="mk") == pytest.approx(-0.607429718875502)
        assert model.score(method="npv") == pytest.approx(0.392570281124498)
        assert model.score(method="prc_auc") == pytest.approx(0.43249617661669015)
        assert model.score(method="precision") == pytest.approx(0.392570281124498)
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

    def test_to_sklearn(self, model):
        md = model.to_sklearn()
        assert 0 == pytest.approx(
            md.predict([[5.006, 3.418,]])[0]
        )

    def test_to_python(self, model):
        assert 0 == pytest.approx(
            model.to_python(return_str=False,)([[5.006, 3.418,]])[0]
        )
        assert model.to_python(return_str=False, return_distance_clusters=True,)([[5.006, 3.418,]])[0][0] in (pytest.approx(32.519389961314424), pytest.approx(45.6436412237776))

    def test_to_sql(self, model):
        model.cursor.execute(
            "SELECT {}::float".format(
                model.to_sql([3.0, 11.0])
            )
        )
        prediction = model.cursor.fetchone()
        assert prediction[0] == pytest.approx(0.38207751614202)

    def test_model_from_vDF(self, base, titanic_vd):
        model_test = NearestCentroid("nc_from_vDF", cursor=base.cursor)
        model_test.fit(titanic_vd, ["age"], "survived")
        assert model_test.score(cutoff=0.9, method="accuracy") == pytest.approx(0.4312938816449348)
        model_test.drop()
