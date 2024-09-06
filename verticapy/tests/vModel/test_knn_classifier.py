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
from verticapy.learn.neighbors import KNeighborsClassifier

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
    model_class = KNeighborsClassifier(
        "knn_model_test",
    )
    model_class.drop()
    model_class.fit("public.titanic", ["age", "fare"], "survived")
    yield model_class
    model_class.drop()


class TestKNeighborsClassifier:
    def test_repr(self, model):
        assert model.__repr__() == "<KNeighborsClassifier>"

    def test_get_attributes(self, model):
        m_att = model.get_attributes()
        assert m_att == ["classes_", "n_neighbors_", "p_"]
        m_att = model.get_attributes("n_neighbors")
        assert m_att == model.parameters["n_neighbors"]
        m_att = model.get_attributes("p")
        assert m_att == model.parameters["p"]
        m_att = model.get_attributes("classes")
        assert m_att[1] == model.classes_[1]

    @skip_plt
    def test_contour(self, titanic_vd):
        model_test = KNeighborsClassifier(
            "model_contour",
        )
        model_test.drop()
        model_test.fit(
            titanic_vd,
            ["age", "fare"],
            "survived",
        )
        result = model_test.contour()
        assert len(result.get_default_bbox_extra_artists()) == 34
        model_test.drop()

    def test_lift_chart(self, model):
        lift_ch = model.lift_chart(nbins=1000, show=False)

        assert lift_ch["decision_boundary"][300] == pytest.approx(0.3)
        assert lift_ch["positive_prediction_ratio"][300] == pytest.approx(
            0.353846153846154
        )
        assert lift_ch["lift"][300] == pytest.approx(1.81819061441703)
        assert lift_ch["decision_boundary"][900] == pytest.approx(0.9)
        assert lift_ch["positive_prediction_ratio"][900] == pytest.approx(1.0)
        assert lift_ch["lift"][900] == pytest.approx(1.0)
        plt.close("all")

    def test_roc_curve(self, model):
        roc_curve = model.roc_curve(nbins=1000, show=False)

        assert roc_curve["threshold"][100] == pytest.approx(0.1)
        assert roc_curve["false_positive"][100] == pytest.approx(1.0)
        assert roc_curve["true_positive"][100] == pytest.approx(1.0)
        assert roc_curve["threshold"][700] == pytest.approx(0.7)
        assert roc_curve["false_positive"][700] == pytest.approx(0.0491803278688525)
        assert roc_curve["true_positive"][700] == pytest.approx(0.353846153846154)
        plt.close("all")

    def test_prc_curve(self, model):
        prc_curve = model.prc_curve(nbins=1000, show=False)

        assert prc_curve["threshold"][100] == pytest.approx(0.099)
        assert prc_curve["recall"][100] == pytest.approx(1.0)
        assert prc_curve["precision"][100] == pytest.approx(0.477356181150551)
        assert prc_curve["threshold"][700] == pytest.approx(0.699)
        assert prc_curve["recall"][700] == pytest.approx(0.353846153846154)
        assert prc_curve["precision"][700] == pytest.approx(0.867924528301887)
        plt.close("all")

    def test_cutoff_curve(self, model):
        cutoff_curve = model.cutoff_curve(nbins=1000, show=False)

        assert cutoff_curve["threshold"][100] == pytest.approx(0.1)
        assert cutoff_curve["false_positive"][100] == pytest.approx(1.0)
        assert cutoff_curve["true_positive"][100] == pytest.approx(1.0)
        assert cutoff_curve["threshold"][700] == pytest.approx(0.7)
        assert cutoff_curve["false_positive"][700] == pytest.approx(0.0491803278688525)
        assert cutoff_curve["true_positive"][700] == pytest.approx(0.353846153846154)
        plt.close("all")

    def test_deploySQL(self, model):
        expected_sql = '(SELECT row_id, "age", "fare", "survived", predict_neighbors, COUNT(*) / 5 AS proba_predict FROM ( SELECT x."age", x."fare", x."survived", ROW_NUMBER() OVER(PARTITION BY x."age", x."fare", row_id ORDER BY POWER(POWER(ABS(x."age" - y."age"), 2) + POWER(ABS(x."fare" - y."fare"), 2), 1 / 2)) AS ordered_distance, y."survived" AS predict_neighbors, row_id FROM (SELECT *, ROW_NUMBER() OVER() AS row_id FROM public.titanic WHERE "age" IS NOT NULL AND "fare" IS NOT NULL) x CROSS JOIN (SELECT * FROM public.titanic WHERE "age" IS NOT NULL AND "fare" IS NOT NULL) y) z WHERE ordered_distance <= 5 GROUP BY "age", "fare", "survived", row_id, predict_neighbors) kneighbors_table'
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_get_params(self, model):
        assert model.get_params() == {"n_neighbors": 5, "p": 2}

    def test_predict(self, titanic_vd, model):
        titanic_copy = titanic_vd.copy()

        titanic_copy = model.predict(
            titanic_copy,
            X=["age", "fare"],
            name="predicted_quality",
            inplace=False,
        )

        assert titanic_copy["predicted_quality"].mean() == pytest.approx(
            0.381884944920441, abs=1e-6
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
            0.378313253012048, abs=1e-6
        )

    def test_classification_report(self, model):
        cls_rep1 = model.classification_report().transpose()

        assert cls_rep1["auc"][0] == pytest.approx(0.696400048039392)
        assert cls_rep1["prc_auc"][0] == pytest.approx(0.7591081348272292)
        assert cls_rep1["accuracy"][0] == pytest.approx(0.7013463892288861)
        assert cls_rep1["log_loss"][0] == pytest.approx(26.8788249694002)
        assert cls_rep1["precision"][0] == pytest.approx(0.7339743589743589)
        assert cls_rep1["recall"][0] == pytest.approx(0.5871794871794872)
        assert cls_rep1["f1_score"][0] == pytest.approx(0.6524216524216524)
        assert cls_rep1["mcc"][0] == pytest.approx(0.40382652359985155)
        assert cls_rep1["informedness"][0] == pytest.approx(0.39280009607878474)
        assert cls_rep1["markedness"][0] == pytest.approx(0.41516247778624016)
        assert cls_rep1["csi"][0] == pytest.approx(0.48414376321353064)

    def test_score(self, model):
        assert model.score(cutoff=0.9, metric="accuracy") == pytest.approx(
            0.5691554467564259
        )
        assert model.score(cutoff=0.1, metric="accuracy") == pytest.approx(
            0.4773561811505508
        )
        assert model.score(metric="best_cutoff") == pytest.approx(0.999)
        assert model.score(metric="bm") == pytest.approx(0.39280009607878474)
        assert model.score(metric="csi") == pytest.approx(0.48414376321353064)
        assert model.score(metric="f1") == pytest.approx(0.6524216524216524)
        assert model.score(metric="logloss") == pytest.approx(26.8788249694002)
        assert model.score(metric="mcc") == pytest.approx(0.40382652359985155)
        assert model.score(metric="mk") == pytest.approx(0.41516247778624016)
        assert model.score(metric="npv") == pytest.approx(0.6811881188118812)
        assert model.score(metric="prc_auc") == pytest.approx(0.7591081348272292)
        assert model.score(metric="precision") == pytest.approx(0.7339743589743589)
        assert model.score(metric="specificity") == pytest.approx(0.8056206088992974)

    def test_set_params(self, model):
        model.set_params({"p": 1})

        assert model.get_params()["p"] == 1

    def test_model_from_vDF(self, titanic_vd):
        model_test = KNeighborsClassifier(
            "knn_from_vDF",
        )
        model_test.drop()
        model_test.fit(titanic_vd, ["age"], "survived")
        assert model_test.score(cutoff=0.9, metric="accuracy") == pytest.approx(
            0.5890710382513661
        )
        model_test.drop()
