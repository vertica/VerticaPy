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
from verticapy import drop, set_option
from verticapy.datasets import load_titanic, load_amazon

set_option("print_info", False)


@pytest.fixture(scope="module")
def titanic_vd():
    titanic = load_titanic()
    yield titanic
    drop(name="public.titanic")


@pytest.fixture(scope="module")
def amazon_vd():
    amazon = load_amazon()
    yield amazon
    drop(name="public.amazon")


class TestvDFCorrelation:
    def test_vDF_chaid(self, titanic_vd):
        result = titanic_vd.chaid("survived", ["age", "fare", "sex"])
        tree = result.tree_
        assert tree["chi2"] == pytest.approx(345.12775126385327)
        assert tree["children"]["female"]["chi2"] == pytest.approx(10.472532457814179)
        assert tree["children"]["female"]["children"][127.6]["chi2"] == pytest.approx(
            3.479525088868805
        )
        assert tree["children"]["female"]["children"][127.6]["children"][19.0][
            "prediction"
        ][0] == pytest.approx(0.325581395348837)
        assert tree["children"]["female"]["children"][127.6]["children"][38.0][
            "prediction"
        ][1] == pytest.approx(0.720496894409938)
        assert not tree["split_is_numerical"]
        assert tree["split_predictor"] == '"sex"'
        assert tree["split_predictor_idx"] == 2
        pred = result.predict([[3.0, 11.0, "male"], [11.0, 1.0, "female"]])
        assert pred[0] == 0
        assert pred[1] == 1
        pred = result.predict_proba([[3.0, 11.0, "male"], [11.0, 1.0, "female"]])
        assert pred[0][0] == pytest.approx(0.75968992)
        assert pred[0][1] == pytest.approx(0.24031008)
        assert pred[1][0] == pytest.approx(0.3255814)
        assert pred[1][1] == pytest.approx(0.6744186)

    def test_vDF_pivot_table_chi2(self, titanic_vd):
        result = titanic_vd.pivot_table_chi2("survived")
        assert result["chi2"][0] == pytest.approx(345.12775126385327)
        assert result["chi2"][1] == pytest.approx(139.2026595859198)
        assert result["chi2"][2] == pytest.approx(103.48373526643627)
        assert result["chi2"][3] == pytest.approx(53.47413727076916)
        assert result["chi2"][4] == pytest.approx(44.074789072247576)
        assert result["chi2"][5] == pytest.approx(42.65927519250441)
        assert result["chi2"][6] == pytest.approx(34.54227539207669)
        assert result["chi2"][7] == pytest.approx(-2.6201263381153694e-14)
        assert result["p_value"][0] == pytest.approx(4.8771014178794746e-77)
        assert result["p_value"][1] == pytest.approx(6.466448884215497e-32)
        assert result["p_value"][2] == pytest.approx(5.922792773022131e-31)
        assert result["p_value"][3] == pytest.approx(2.9880194205753303e-09)
        assert result["p_value"][4] == pytest.approx(7.143900900120299e-08)
        assert result["p_value"][5] == pytest.approx(5.4532585750903e-10)
        assert result["p_value"][6] == pytest.approx(0.0028558258758971957)
        set_option("random_state", 0)
        result = titanic_vd.pivot_table_chi2("survived", method="smart")
        assert result["chi2"][0] == pytest.approx(345.12775126385327)
        assert result["chi2"][1] == pytest.approx(187.75090682844288)
        assert result["chi2"][2] == pytest.approx(139.2026595859198)
        assert result["chi2"][3] == pytest.approx(53.474137270768885)
        assert result["chi2"][4] == pytest.approx(44.074789072247576)
        assert result["chi2"][5] == pytest.approx(42.65927519250441)
        assert result["chi2"][6] == pytest.approx(38.109904618027755)
        assert result["chi2"][7] == pytest.approx(0.0)
