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
from verticapy.datasets import load_winequality
from verticapy.learn.decomposition import PCA

set_option("print_info", False)


@pytest.fixture(scope="module")
def winequality_vd():
    winequality = load_winequality()
    yield winequality
    drop(name="public.winequality",)


@pytest.fixture(scope="module")
def model(winequality_vd):
    model_class = PCA("pca_model_test",)
    model_class.drop()
    model_class.fit("public.winequality", ["citric_acid", "residual_sugar", "alcohol"])
    yield model_class
    model_class.drop()


class TestPCA:
    def test_repr(self, model):
        assert "index|     name     |  mean  |   sd" in model.__repr__()
        model_repr = PCA("model_repr")
        model_repr.drop()
        assert model_repr.__repr__() == "<PCA>"

    def test_deploySQL(self, model):
        expected_sql = 'APPLY_PCA("citric_acid", "residual_sugar", "alcohol" USING PARAMETERS model_name = \'pca_model_test\', match_by_pos = \'true\', cutoff = 1)'
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_deployInverseSQL(self, model):
        expected_sql = 'APPLY_INVERSE_PCA("citric_acid", "residual_sugar", "alcohol" USING PARAMETERS model_name = \'pca_model_test\', match_by_pos = \'true\')'
        result_sql = model.deployInverseSQL()

        assert result_sql == expected_sql

    def test_drop(self):
        current_cursor().execute("DROP MODEL IF EXISTS pca_model_test_drop")
        model_test = PCA("pca_model_test_drop",)
        model_test.fit("public.winequality", ["alcohol", "quality"])

        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'pca_model_test_drop'"
        )
        assert current_cursor().fetchone()[0] == "pca_model_test_drop"

        model_test.drop()
        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'pca_model_test_drop'"
        )
        assert current_cursor().fetchone() is None

    def test_get_attr(self, model):
        m_att = model.get_attr()

        assert m_att["attr_name"] == [
            "columns",
            "singular_values",
            "principal_components",
            "counters",
            "call_string",
        ]
        assert m_att["attr_fields"] == [
            "index, name, mean, sd",
            "index, value, explained_variance, accumulated_explained_variance",
            "index, PC1, PC2, PC3",
            "counter_name, counter_value",
            "call_string",
        ]
        assert m_att["#_of_rows"] == [3, 3, 3, 3, 1]

        m_att_details = model.get_attr(attr_name="principal_components")

        assert m_att_details["PC1"][0] == pytest.approx(0.00430584055130197, abs=1e-6)
        assert m_att_details["PC1"][1] == pytest.approx(0.995483456627961, abs=1e-6)
        assert m_att_details["PC1"][2] == pytest.approx(-0.0948374784417728, abs=1e-6)
        assert m_att_details["PC2"][0] == pytest.approx(0.00623540848865928, abs=1e-6)
        assert m_att_details["PC2"][1] == pytest.approx(0.0948097859779201, abs=1e-6)
        assert m_att_details["PC2"][2] == pytest.approx(0.995475878243064, abs=1e-6)
        assert m_att_details["PC3"][0] == pytest.approx(0.999971289296911, abs=1e-6)
        assert m_att_details["PC3"][1] == pytest.approx(-0.0048777108225008, abs=1e-6)
        assert m_att_details["PC3"][2] == pytest.approx(-0.00579901017465387, abs=1e-6)

    def test_get_params(self, model):
        assert model.get_params() == {
            "method": "lapack",
            "n_components": 0,
            "scale": False,
        }

    def test_plot(self, model):
        result = model.plot()
        assert len(result.get_default_bbox_extra_artists()) == 8
        result = model.plot(dimensions=(2, 3))
        assert len(result.get_default_bbox_extra_artists()) == 8

    def test_plot_scree(self, model):
        result = model.plot_scree()
        assert len(result.get_default_bbox_extra_artists()) == 14

    def test_plot_circle(self, model):
        result = model.plot_circle()
        assert len(result.get_default_bbox_extra_artists()) == 16
        result = model.plot_circle(dimensions=(2, 3))
        assert len(result.get_default_bbox_extra_artists()) == 16

    def test_to_python(self, model):
        current_cursor().execute(
            "SELECT APPLY_PCA(citric_acid, residual_sugar, alcohol USING PARAMETERS model_name = '{}', match_by_pos=True) FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                model.name
            )
        )
        prediction = current_cursor().fetchone()
        assert prediction == pytest.approx(
            model.to_python(return_str=False)([[3.0, 11.0, 93.0]])[0]
        )

    def test_to_sql(self, model):
        current_cursor().execute(
            "SELECT APPLY_PCA(citric_acid, residual_sugar, alcohol USING PARAMETERS model_name = '{}', match_by_pos=True) FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                model.name
            )
        )
        prediction = [float(elem) for elem in current_cursor().fetchone()]
        current_cursor().execute(
            "SELECT {} FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                ", ".join(model.to_sql())
            )
        )
        prediction2 = [float(elem) for elem in current_cursor().fetchone()]
        assert prediction == pytest.approx(prediction2)

    def test_to_memmodel(self, model):
        current_cursor().execute(
            "SELECT APPLY_PCA(citric_acid, residual_sugar, alcohol USING PARAMETERS model_name = '{}', match_by_pos=True) FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                model.name
            )
        )
        prediction = [float(elem) for elem in current_cursor().fetchone()]
        current_cursor().execute(
            "SELECT {} FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                ", ".join(
                    model.to_memmodel().transform_sql(
                        ["citric_acid", "residual_sugar", "alcohol"]
                    )
                )
            )
        )
        prediction2 = [float(elem) for elem in current_cursor().fetchone()]
        assert prediction == pytest.approx(prediction2)
        prediction3 = model.to_memmodel().transform([[3.0, 11.0, 93.0]])
        assert prediction == pytest.approx(list(prediction3[0]))

    def test_get_transform(self, winequality_vd, model):
        winequality_trans = model.transform(
            winequality_vd, X=["citric_acid", "residual_sugar", "alcohol"]
        )
        assert winequality_trans["col1"].mean() == pytest.approx(0.0, abs=1e-6)
        assert winequality_trans["col2"].mean() == pytest.approx(0.0, abs=1e-6)
        assert winequality_trans["col3"].mean() == pytest.approx(0.0, abs=1e-6)

    def test_get_inverse_transform(self, winequality_vd, model):
        winequality_trans = model.transform(
            winequality_vd, X=["citric_acid", "residual_sugar", "alcohol"]
        )
        winequality_trans = model.inverse_transform(
            winequality_trans, X=["col1", "col2", "col3"]
        )
        assert winequality_trans["citric_acid"].mean() == pytest.approx(
            winequality_vd["citric_acid"].mean(), abs=1e-6
        )
        assert winequality_trans["residual_sugar"].mean() == pytest.approx(
            winequality_vd["residual_sugar"].mean(), abs=1e-6
        )
        assert winequality_trans["alcohol"].mean() == pytest.approx(
            winequality_vd["alcohol"].mean(), abs=1e-6
        )

    def test_pca_score(self, model):
        result = model.score()
        assert result["Score"][0] == pytest.approx(0.0, abs=1e-6)
        assert result["Score"][1] == pytest.approx(0.0, abs=1e-6)
        assert result["Score"][2] == pytest.approx(0.0, abs=1e-6)

    def test_set_params(self, model):
        model.set_params({"n_components": 3})
        assert model.get_params()["n_components"] == 3

    def test_model_from_vDF(self, winequality_vd):
        current_cursor().execute("DROP MODEL IF EXISTS pca_vDF")
        model_test = PCA("pca_vDF",)
        model_test.fit(winequality_vd, ["alcohol", "quality"])
        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'pca_vDF'"
        )
        assert current_cursor().fetchone()[0] == "pca_vDF"
        model_test.drop()
