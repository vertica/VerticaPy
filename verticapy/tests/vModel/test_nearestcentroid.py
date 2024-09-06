"""
Copyright  (c)  2018-2024 Open Text  or  one  of its
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

# Other Modules
import matplotlib.pyplot as plt

# VerticaPy
from verticapy import (
    drop,
    set_option,
)
from verticapy.connection import current_cursor
from verticapy.datasets import load_titanic
from verticapy.learn.neighbors import NearestCentroid

# Matplotlib skip
import matplotlib

matplotlib_version = matplotlib.__version__
skip_plt = pytest.mark.skipif(
    matplotlib_version > "3.5.2",
    reason="Test skipped on matplotlib version greater than 3.5.2",
)

set_option("print_info", False)


@pytest.fixture(scope="module")
def titanic_vd():
    titanic = load_titanic()
    yield titanic
    drop(
        name="public.titanic",
    )


@pytest.fixture(scope="module")
def model(titanic_vd):
    model_class = NearestCentroid(
        "nc_model_test",
    )
    model_class.drop()
    model_class.fit("public.titanic", ["age", "fare"], "survived")
    yield model_class
    model_class.drop()


class TestNearestCentroid:
    def test_repr(self, model):
        assert model.__repr__() == "<NearestCentroid>"

    @skip_plt
    def test_contour(self, titanic_vd):
        model_test = NearestCentroid(
            "model_contour",
        )
        model_test.drop()
        model_test.fit(
            titanic_vd,
            ["age", "fare"],
            "survived",
        )
        result = model_test.contour()
        assert len(result.get_default_bbox_extra_artists()) == 40
        model_test.drop()

    def test_lift_chart(self, model):
        lift_ch = model.lift_chart(nbins=1000, show=False)

        assert lift_ch["decision_boundary"][300] == pytest.approx(0.3)
        assert lift_ch["positive_prediction_ratio"][300] == pytest.approx(
            0.0895140664961637
        )
        assert lift_ch["lift"][300] == pytest.approx(1.89693638787615)
        assert lift_ch["decision_boundary"][900] == pytest.approx(0.9)
        assert lift_ch["positive_prediction_ratio"][900] == pytest.approx(
            0.979539641943734
        )
        assert lift_ch["lift"][900] == pytest.approx(0.998589031091053)
        plt.close("all")

    def test_roc_curve(self, model):
        roc_curve = model.roc_curve(nbins=1000, show=False)

        assert roc_curve["threshold"][100] == pytest.approx(0.1)
        assert roc_curve["false_positive"][100] == pytest.approx(0.981818181818182)
        assert roc_curve["true_positive"][100] == pytest.approx(0.979539641943734)
        assert roc_curve["threshold"][700] == pytest.approx(0.7)
        assert roc_curve["false_positive"][700] == pytest.approx(0.0198347107438017)
        assert roc_curve["true_positive"][700] == pytest.approx(0.0895140664961637)
        plt.close("all")

    def test_cutoff_curve(self, model):
        cutoff_curve = model.cutoff_curve(nbins=1000, show=False)

        assert cutoff_curve["threshold"][100] == pytest.approx(0.1)
        assert cutoff_curve["false_positive"][100] == pytest.approx(0.981818181818182)
        assert cutoff_curve["true_positive"][100] == pytest.approx(0.979539641943734)
        assert cutoff_curve["threshold"][700] == pytest.approx(0.7)
        assert cutoff_curve["false_positive"][700] == pytest.approx(0.0198347107438017)
        assert cutoff_curve["true_positive"][700] == pytest.approx(0.0895140664961637)
        plt.close("all")

    def test_deploySQL(self, model):
        expected_sql = 'CASE WHEN "age" IS NULL OR "fare" IS NULL THEN NULL WHEN POWER(POWER("age" - 29.3936572890026, 2) + POWER("fare" - 52.3002593333333, 2), 1 / 2) <= POWER(POWER("age" - 30.6420462046205, 2) + POWER("fare" - 23.4255950191571, 2), 1 / 2) THEN 1 ELSE 0 END'
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_to_memmodel(self, titanic_vd, model):
        titanic = titanic_vd.copy()
        mmodel = model.to_memmodel()
        res = mmodel.predict([[11.0, 1993.0], [1.0, 1999.0]])
        res_py = model.to_python()([[11.0, 1993.0], [1.0, 1999.0]])
        assert res[0] == res_py[0]
        assert res[1] == res_py[1]
        res = mmodel.predict_proba([[11.0, 1993.0], [1.0, 1999.0]])
        res_py = model.to_python(return_proba=True)([[11.0, 1993.0], [1.0, 1999.0]])
        assert res[0][0] == res_py[0][0]
        assert res[0][1] == res_py[0][1]
        assert res[1][0] == res_py[1][0]
        assert res[1][1] == res_py[1][1]
        titanic["prediction_sql"] = mmodel.predict_sql(["age", "fare"])
        titanic["prediction_proba_sql_0"] = mmodel.predict_proba_sql(["age", "fare"])[0]
        titanic["prediction_proba_sql_1"] = mmodel.predict_proba_sql(["age", "fare"])[1]
        titanic = model.predict(titanic, name="prediction_vertica_sql", cutoff=0.5)
        titanic = model.predict_proba(
            titanic,
            name="prediction_proba_vertica_sql_0",
            pos_label=model.classes_[0],
        )
        titanic = model.predict_proba(
            titanic,
            name="prediction_proba_vertica_sql_1",
            pos_label=model.classes_[1],
        )
        score = titanic.score(
            "prediction_sql", "prediction_vertica_sql", metric="accuracy"
        )
        print(titanic[["prediction_sql", "prediction_vertica_sql"]])
        print(titanic.current_relation())
        assert score == pytest.approx(1.0)
        score = titanic.score(
            "prediction_proba_sql_0", "prediction_proba_vertica_sql_0", metric="r2"
        )
        assert score == pytest.approx(1.0)
        score = titanic.score(
            "prediction_proba_sql_1", "prediction_proba_vertica_sql_1", metric="r2"
        )
        assert score == pytest.approx(1.0)

    def test_get_params(self, model):
        assert model.get_params() == {"p": 2}

    def test_predict(self, titanic_vd, model):
        titanic_copy = titanic_vd.copy()

        titanic_copy = model.predict(
            titanic_copy,
            X=["age", "fare"],
            name="predicted_quality",
            inplace=False,
        )
        assert titanic_copy["predicted_quality"].mean() == pytest.approx(
            0.245983935742972, abs=1e-6
        )

    def test_predict_proba(self, titanic_vd, model):
        titanic_copy = titanic_vd.copy()

        titanic_copy = model.predict_proba(
            titanic_copy,
            X=["age", "fare"],
            name="prob_quality",
            inplace=False,
            pos_label=1,
        )
        assert titanic_copy["prob_quality"].mean() == pytest.approx(
            0.371179022830741, abs=1e-6
        )

    def test_classification_report(self, model):
        cls_rep1 = model.classification_report().transpose()

        assert cls_rep1["auc"][0] == pytest.approx(0.6325400012682033)
        assert cls_rep1["prc_auc"][0] == pytest.approx(0.5442487908406839)
        assert cls_rep1["accuracy"][0] == pytest.approx(0.6746987951807228)
        assert cls_rep1["log_loss"][0] == pytest.approx(0.282873255537287)
        assert cls_rep1["precision"][0] == pytest.approx(0.636734693877551)
        assert cls_rep1["recall"][0] == pytest.approx(0.3989769820971867)
        assert cls_rep1["f1_score"][0] == pytest.approx(0.49056603773584906)
        assert cls_rep1["mcc"][0] == pytest.approx(0.28558718217018486)
        assert cls_rep1["informedness"][0] == pytest.approx(0.25186954408065776)
        assert cls_rep1["markedness"][0] == pytest.approx(0.32381858202668545)
        assert cls_rep1["csi"][0] == pytest.approx(0.325)

    def test_score(self, model):
        assert model.score(cutoff=0.9, metric="accuracy") == pytest.approx(
            0.607429718875502
        )
        assert model.score(cutoff=0.1, metric="accuracy") == pytest.approx(
            0.39558232931726905
        )
        assert model.score(metric="best_cutoff") == pytest.approx(0.352)
        assert model.score(metric="bm") == pytest.approx(0.25186954408065776)
        assert model.score(metric="csi") == pytest.approx(0.325)
        assert model.score(metric="f1") == pytest.approx(0.49056603773584906)
        assert model.score(metric="logloss") == pytest.approx(0.282873255537287)
        assert model.score(metric="mcc") == pytest.approx(0.28558718217018486)
        assert model.score(metric="mk") == pytest.approx(0.32381858202668545)
        assert model.score(metric="npv") == pytest.approx(0.6870838881491345)
        assert model.score(metric="prc_auc") == pytest.approx(0.5442487908406839)
        assert model.score(metric="precision") == pytest.approx(0.636734693877551)
        assert model.score(metric="specificity") == pytest.approx(0.8528925619834711)

    def test_set_params(self, model):
        model.set_params({"p": 1})
        assert model.get_params()["p"] == 1
        model.set_params({"p": 2})

    def test_to_python(self, model):
        assert 0 == pytest.approx(model.to_python()([[5.006, 3.418]])[0])
        assert model.to_python(return_distance_clusters=True)([[5.006, 3.418]])[0][
            0
        ] in (
            pytest.approx(32.519389961314424),
            pytest.approx(45.6436412237776),
        )

    def test_to_sql(self, model):
        current_cursor().execute("SELECT {}::int".format(model.to_sql([3.0, 11.0])))
        prediction = current_cursor().fetchone()
        assert prediction[0] == 0

    def test_model_from_vDF(self, titanic_vd):
        model_test = NearestCentroid(
            "nc_from_vDF",
        )
        model_test.drop()
        model_test.fit(titanic_vd, ["age"], "survived")
        assert model_test.score(cutoff=0.9, metric="accuracy") == pytest.approx(
            0.6078234704112337
        )
        model_test.drop()

    def test_optional_name(self):
        model = NearestCentroid()
        assert model.model_name is not None
