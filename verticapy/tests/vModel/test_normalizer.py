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

# VerticaPy
from verticapy import drop, set_option
from verticapy.connection import current_cursor
from verticapy.datasets import load_winequality
from verticapy.learn.preprocessing import (
    Scaler,
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
)

set_option("print_info", False)


@pytest.fixture(scope="module")
def winequality_vd():
    winequality = load_winequality()
    yield winequality
    drop(
        name="public.winequality",
    )


@pytest.fixture(scope="module")
def model(winequality_vd):
    current_cursor().execute("DROP MODEL IF EXISTS norm_model_test")
    model_class = Scaler(
        "norm_model_test",
    )
    model_class.fit("public.winequality", ["citric_acid", "residual_sugar", "alcohol"])
    yield model_class
    model_class.drop()


class TestScaler:
    def test_repr(self, model):
        assert model.__repr__() == "<Scaler>"

    def test_Scaler_subclasses(self):
        result = StandardScaler("model_test")
        assert result.parameters["method"] == "zscore"
        result = RobustScaler("model_test")
        assert result.parameters["method"] == "robust_zscore"
        result = MinMaxScaler("model_test")
        assert result.parameters["method"] == "minmax"

    def test_deploySQL(self, model):
        expected_sql = 'APPLY_NORMALIZE("citric_acid", "residual_sugar", "alcohol" USING PARAMETERS model_name = \'norm_model_test\', match_by_pos = \'true\')'
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_deployInverseSQL(self, model):
        expected_sql = 'REVERSE_NORMALIZE("citric_acid", "residual_sugar", "alcohol" USING PARAMETERS model_name = \'norm_model_test\', match_by_pos = \'true\')'
        result_sql = model.deployInverseSQL()

        assert result_sql == expected_sql

    def test_drop(self):
        current_cursor().execute("DROP MODEL IF EXISTS norm_model_test_drop")
        model_test = Scaler(
            "norm_model_test_drop",
        )
        model_test.fit("public.winequality", ["alcohol", "quality"])

        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'norm_model_test_drop'"
        )
        assert current_cursor().fetchone()[0] == "norm_model_test_drop"

        model_test.drop()
        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'norm_model_test_drop'"
        )
        assert current_cursor().fetchone() is None

    def test_get_vertica_attributes(self, model):
        m_att = model.get_vertica_attributes()

        assert m_att["attr_name"] == [
            "details",
        ]
        assert m_att["attr_fields"] == [
            "column_name, avg, std_dev",
        ]
        assert m_att["#_of_rows"] == [3]

        m_att_details = model.get_vertica_attributes(attr_name="details")

        assert m_att_details["column_name"] == [
            "citric_acid",
            "residual_sugar",
            "alcohol",
        ]
        assert m_att_details["avg"][0] == pytest.approx(0.318633215330152, abs=1e-6)
        assert m_att_details["avg"][1] == pytest.approx(5.44323533938741, abs=1e-6)
        assert m_att_details["avg"][2] == pytest.approx(10.4918008311528, abs=1e-6)
        assert m_att_details["std_dev"][0] == pytest.approx(0.145317864897592, abs=1e-6)
        assert m_att_details["std_dev"][1] == pytest.approx(4.75780374314742, abs=1e-6)
        assert m_att_details["std_dev"][2] == pytest.approx(1.192711748871)

    def test_get_params(self, model):
        assert model.get_params() == {"method": "zscore"}

    def test_to_python(self, model):
        # Zscore
        current_cursor().execute(
            "SELECT APPLY_NORMALIZE(citric_acid, residual_sugar, alcohol USING PARAMETERS model_name = '{}', match_by_pos=True) FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                model.model_name
            )
        )
        prediction = current_cursor().fetchone()[0]
        assert prediction == pytest.approx(model.to_python()([[3.0, 11.0, 93.0]])[0][0])
        # Minmax
        model2 = Scaler("norm_model_test2", method="minmax")
        model2.drop()
        model2.fit("public.winequality", ["citric_acid", "residual_sugar", "alcohol"])
        current_cursor().execute(
            "SELECT APPLY_NORMALIZE(citric_acid, residual_sugar, alcohol USING PARAMETERS model_name = '{}', match_by_pos=True) FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                model2.model_name
            )
        )
        prediction = current_cursor().fetchone()[0]
        assert prediction == pytest.approx(
            model2.to_python()([[3.0, 11.0, 93.0]])[0][0]
        )
        model2.drop()
        # Robust Zscore
        model3 = Scaler("norm_model_test2", method="robust_zscore")
        model3.drop()
        model3.fit("public.winequality", ["citric_acid", "residual_sugar", "alcohol"])
        current_cursor().execute(
            "SELECT APPLY_NORMALIZE(citric_acid, residual_sugar, alcohol USING PARAMETERS model_name = '{}', match_by_pos=True) FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                model3.model_name
            )
        )
        prediction = current_cursor().fetchone()[0]
        assert prediction == pytest.approx(
            model3.to_python()([[3.0, 11.0, 93.0]])[0][0]
        )
        model3.drop()

    def test_to_sql(self, model):
        # Zscore
        current_cursor().execute(
            "SELECT APPLY_NORMALIZE(citric_acid, residual_sugar, alcohol USING PARAMETERS model_name = '{}', match_by_pos=True) FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                model.model_name
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
        # Minmax
        model2 = Scaler("norm_model_test2", method="minmax")
        model2.drop()
        model2.fit("public.winequality", ["citric_acid", "residual_sugar", "alcohol"])
        current_cursor().execute(
            "SELECT APPLY_NORMALIZE(citric_acid, residual_sugar, alcohol USING PARAMETERS model_name = '{}', match_by_pos=True) FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                model2.model_name
            )
        )
        prediction = [float(elem) for elem in current_cursor().fetchone()]
        current_cursor().execute(
            "SELECT {} FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                ", ".join(model2.to_sql())
            )
        )
        prediction2 = [float(elem) for elem in current_cursor().fetchone()]
        assert prediction == pytest.approx(prediction2)
        model2.drop()
        # Robust Zscore
        model3 = Scaler("norm_model_test2", method="robust_zscore")
        model3.drop()
        model3.fit("public.winequality", ["citric_acid", "residual_sugar", "alcohol"])
        current_cursor().execute(
            "SELECT APPLY_NORMALIZE(citric_acid, residual_sugar, alcohol USING PARAMETERS model_name = '{}', match_by_pos=True) FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                model3.model_name
            )
        )
        prediction = [float(elem) for elem in current_cursor().fetchone()]
        current_cursor().execute(
            "SELECT {} FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                ", ".join(model3.to_sql())
            )
        )
        prediction2 = [float(elem) for elem in current_cursor().fetchone()]
        assert prediction == pytest.approx(prediction2)
        model3.drop()

    def test_to_memmodel(self, model):
        # Zscore
        current_cursor().execute(
            "SELECT APPLY_NORMALIZE(citric_acid, residual_sugar, alcohol USING PARAMETERS model_name = '{}', match_by_pos=True) FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                model.model_name
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
        assert prediction[0] == pytest.approx(prediction3[0][0])
        assert prediction[1] == pytest.approx(prediction3[0][1])
        assert prediction[2] == pytest.approx(prediction3[0][2])
        # Minmax
        model2 = Scaler("norm_model_test2", method="minmax")
        model2.drop()
        model2.fit("public.winequality", ["citric_acid", "residual_sugar", "alcohol"])
        current_cursor().execute(
            "SELECT APPLY_NORMALIZE(citric_acid, residual_sugar, alcohol USING PARAMETERS model_name = '{}', match_by_pos=True) FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                model2.model_name
            )
        )
        prediction = [float(elem) for elem in current_cursor().fetchone()]
        current_cursor().execute(
            "SELECT {} FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                ", ".join(
                    model2.to_memmodel().transform_sql(
                        ["citric_acid", "residual_sugar", "alcohol"]
                    )
                )
            )
        )
        prediction2 = [float(elem) for elem in current_cursor().fetchone()]
        assert prediction == pytest.approx(prediction2)
        prediction3 = model2.to_memmodel().transform([[3.0, 11.0, 93.0]])
        assert prediction[0] == pytest.approx(prediction3[0][0])
        assert prediction[1] == pytest.approx(prediction3[0][1])
        assert prediction[2] == pytest.approx(prediction3[0][2])
        model2.drop()
        # Robust Zscore
        model3 = Scaler("norm_model_test2", method="robust_zscore")
        model3.drop()
        model3.fit("public.winequality", ["citric_acid", "residual_sugar", "alcohol"])
        current_cursor().execute(
            "SELECT APPLY_NORMALIZE(citric_acid, residual_sugar, alcohol USING PARAMETERS model_name = '{}', match_by_pos=True) FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                model3.model_name
            )
        )
        prediction = [float(elem) for elem in current_cursor().fetchone()]
        current_cursor().execute(
            "SELECT {} FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                ", ".join(
                    model3.to_memmodel().transform_sql(
                        ["citric_acid", "residual_sugar", "alcohol"]
                    )
                )
            )
        )
        prediction2 = [float(elem) for elem in current_cursor().fetchone()]
        assert prediction == pytest.approx(prediction2)
        prediction3 = model3.to_memmodel().transform([[3.0, 11.0, 93.0]])
        assert prediction[0] == pytest.approx(prediction3[0][0])
        assert prediction[1] == pytest.approx(prediction3[0][1])
        assert prediction[2] == pytest.approx(prediction3[0][2])
        model3.drop()

    def test_get_transform(self, winequality_vd, model):
        # Zscore
        winequality_trans = model.transform(
            winequality_vd, X=["citric_acid", "residual_sugar", "alcohol"]
        )
        assert winequality_trans["citric_acid"].mean() == pytest.approx(0.0, abs=1e-6)
        assert winequality_trans["residual_sugar"].mean() == pytest.approx(
            0.0, abs=1e-6
        )
        assert winequality_trans["alcohol"].mean() == pytest.approx(0.0, abs=1e-6)
        # Minmax
        model2 = Scaler("norm_model_test2", method="minmax")
        model2.drop()
        model2.fit("public.winequality", ["citric_acid", "residual_sugar", "alcohol"])
        winequality_trans = model2.transform(
            winequality_vd, X=["citric_acid", "residual_sugar", "alcohol"]
        )
        assert winequality_trans["citric_acid"].min() == pytest.approx(0.0, abs=1e-6)
        assert winequality_trans["residual_sugar"].max() == pytest.approx(1.0, abs=1e-6)
        assert winequality_trans["alcohol"].min() == pytest.approx(0.0, abs=1e-6)
        model2.drop()
        # Robust Zscore
        model3 = Scaler("norm_model_test2", method="robust_zscore")
        model3.drop()
        model3.fit("public.winequality", ["citric_acid", "residual_sugar", "alcohol"])
        winequality_trans = model3.transform(
            winequality_vd, X=["citric_acid", "residual_sugar", "alcohol"]
        )
        assert winequality_trans["citric_acid"].median() == pytest.approx(0.0, abs=1e-6)
        assert winequality_trans["residual_sugar"].median() == pytest.approx(
            0.0, abs=1e-6
        )
        assert winequality_trans["alcohol"].median() == pytest.approx(0.0, abs=1e-6)
        model3.drop()

    def test_get_inverse_transform(self, winequality_vd, model):
        winequality_trans = model.inverse_transform(
            winequality_vd, X=["citric_acid", "residual_sugar", "alcohol"]
        )
        assert winequality_trans["citric_acid"].mean() == pytest.approx(
            0.364936313867385, abs=1e-6
        )
        assert winequality_trans["residual_sugar"].mean() == pytest.approx(
            31.3410808119571, abs=1e-6
        )
        assert winequality_trans["alcohol"].mean() == pytest.approx(
            23.0054949492833, abs=1e-6
        )

    def test_set_params(self, model):
        model.set_params({"method": "robust_zscore"})
        assert model.get_params()["method"] == "robust_zscore"
        model.set_params({"method": "zscore"})
        assert model.get_params()["method"] == "zscore"

    def test_model_from_vDF(self, winequality_vd):
        current_cursor().execute("DROP MODEL IF EXISTS norm_vDF")
        model_test = Scaler(
            "norm_vDF",
        )
        model_test.fit(winequality_vd, ["alcohol", "quality"])
        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'norm_vDF'"
        )
        assert current_cursor().fetchone()[0] == "norm_vDF"
        model_test.drop()

    def test_optional_name(self):
        model = Scaler()
        assert model.model_name is not None
        model = StandardScaler()
        assert model.model_name is not None
        model = RobustScaler()
        assert model.model_name is not None
        model = MinMaxScaler()
        assert model.model_name is not None
