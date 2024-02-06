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

import pytest
from verticapy.machine_learning.memmodel.naive_bayes import NaiveBayes


@pytest.fixture(scope="module")
def memmodel_nb():
    attributes = [
        {
            "type": "gaussian",
            "C": {"mu": 63.9878308300395, "sigma_sq": 7281.87598377196},
            "Q": {"mu": 13.0217386792453, "sigma_sq": 211.626862330204},
            "S": {"mu": 27.6928120412844, "sigma_sq": 1428.57067393938},
        },
        {
            "type": "multinomial",
            "C": 0.771666666666667,
            "Q": 0.910714285714286,
            "S": 0.878216123499142,
        },
        {
            "type": "bernoulli",
            "C": 0.771666666666667,
            "Q": 0.910714285714286,
            "S": 0.878216123499142,
        },
        {
            "type": "categorical",
            "C": {
                "female": 0.407843137254902,
                "male": 0.592156862745098,
            },
            "Q": {
                "female": 0.416666666666667,
                "male": 0.583333333333333,
            },
            "S": {
                "female": 0.406666666666667,
                "male": 0.593333333333333,
            },
        },
    ]
    prior = [0.8, 0.1, 0.1]
    classes = ["C", "Q", "S"]
    memmodel_nb = NaiveBayes(attributes, prior, classes)
    yield memmodel_nb


class TestNaiveBayes:
    """
    test class - TestNaiveBayes
    """

    @pytest.mark.parametrize(
        "age, pclass, survived, sex, expected_output",
        [
            (40.0, 1, True, "male", "C"),
            (60.0, 3, True, "male", "C"),
            (15.0, 2, False, "female", "Q"),
        ],
    )
    def test_predict(self, memmodel_nb, age, pclass, survived, sex, expected_output):
        """
        test function - predict
        """
        data = [
            [age, pclass, survived, sex],
        ]
        prediction = memmodel_nb.predict(data)
        assert prediction == expected_output

    @pytest.mark.parametrize(
        "age, pclass, survived, sex, expected_output",
        [
            (40.0, 1, True, "male", [0.64564673, 0.12105224, 0.23330103]),
            (60.0, 3, True, "male", [0.74783083, 0.00570541, 0.24646376]),
            (15.0, 2, False, "female", [0.34471925, 0.49592024, 0.15936051]),
        ],
    )
    def test_predict_proba(
        self, memmodel_nb, age, pclass, survived, sex, expected_output
    ):
        """
        test function - predict_proba
        """
        data = [
            [age, pclass, survived, sex],
        ]
        prediction = memmodel_nb.predict_proba(data)
        assert prediction[0][0] == pytest.approx(expected_output[0])
        assert prediction[0][1] == pytest.approx(expected_output[1])
        assert prediction[0][2] == pytest.approx(expected_output[2])

    def test_predict_sql(self, memmodel_nb):
        """
        test function - predict_proba_sql
        """
        cnames = ["age", "pclass", "survived", "sex"]
        pred_sql = memmodel_nb.predict_sql(cnames)
        assert (
            pred_sql
            == "CASE WHEN age IS NULL OR pclass IS NULL OR survived IS NULL OR sex IS NULL THEN NULL WHEN 0.010555023401917874 * EXP(- POWER(age - 27.6928120412844, 2) / 2857.14134787876) * POWER(0.878216123499142, pclass) * (CASE WHEN survived THEN 0.878216123499142 ELSE 0.12178387650085798 END) * DECODE(sex, 'female', 0.406666666666667, 'male', 0.593333333333333) * 0.1 >= 0.004675073323276673 * EXP(- POWER(age - 63.9878308300395, 2) / 14563.75196754392) * POWER(0.771666666666667, pclass) * (CASE WHEN survived THEN 0.771666666666667 ELSE 0.22833333333333306 END) * DECODE(sex, 'female', 0.407843137254902, 'male', 0.592156862745098) * 0.8 AND 0.010555023401917874 * EXP(- POWER(age - 27.6928120412844, 2) / 2857.14134787876) * POWER(0.878216123499142, pclass) * (CASE WHEN survived THEN 0.878216123499142 ELSE 0.12178387650085798 END) * DECODE(sex, 'female', 0.406666666666667, 'male', 0.593333333333333) * 0.1 >= 0.027423612860412977 * EXP(- POWER(age - 13.0217386792453, 2) / 423.253724660408) * POWER(0.910714285714286, pclass) * (CASE WHEN survived THEN 0.910714285714286 ELSE 0.08928571428571397 END) * DECODE(sex, 'female', 0.416666666666667, 'male', 0.583333333333333) * 0.1 THEN 'S' WHEN 0.027423612860412977 * EXP(- POWER(age - 13.0217386792453, 2) / 423.253724660408) * POWER(0.910714285714286, pclass) * (CASE WHEN survived THEN 0.910714285714286 ELSE 0.08928571428571397 END) * DECODE(sex, 'female', 0.416666666666667, 'male', 0.583333333333333) * 0.1 >= 0.004675073323276673 * EXP(- POWER(age - 63.9878308300395, 2) / 14563.75196754392) * POWER(0.771666666666667, pclass) * (CASE WHEN survived THEN 0.771666666666667 ELSE 0.22833333333333306 END) * DECODE(sex, 'female', 0.407843137254902, 'male', 0.592156862745098) * 0.8 THEN 'Q' ELSE 'C' END"
        )

    def test_predict_proba_sql(self, memmodel_nb):
        """
        test function - predict_proba_sql
        """
        cnames = ["age", "pclass", "survived", "sex"]
        pred_proba_sql = memmodel_nb.predict_proba_sql(cnames)
        assert (
            pred_proba_sql[0]
            == "(0.004675073323276673 * EXP(- POWER(age - 63.9878308300395, 2) / 14563.75196754392) * POWER(0.771666666666667, pclass) * (CASE WHEN survived THEN 0.771666666666667 ELSE 0.22833333333333306 END) * DECODE(sex, 'female', 0.407843137254902, 'male', 0.592156862745098) * 0.8) / (0.004675073323276673 * EXP(- POWER(age - 63.9878308300395, 2) / 14563.75196754392) * POWER(0.771666666666667, pclass) * (CASE WHEN survived THEN 0.771666666666667 ELSE 0.22833333333333306 END) * DECODE(sex, 'female', 0.407843137254902, 'male', 0.592156862745098) * 0.8 + 0.027423612860412977 * EXP(- POWER(age - 13.0217386792453, 2) / 423.253724660408) * POWER(0.910714285714286, pclass) * (CASE WHEN survived THEN 0.910714285714286 ELSE 0.08928571428571397 END) * DECODE(sex, 'female', 0.416666666666667, 'male', 0.583333333333333) * 0.1 + 0.010555023401917874 * EXP(- POWER(age - 27.6928120412844, 2) / 2857.14134787876) * POWER(0.878216123499142, pclass) * (CASE WHEN survived THEN 0.878216123499142 ELSE 0.12178387650085798 END) * DECODE(sex, 'female', 0.406666666666667, 'male', 0.593333333333333) * 0.1)"
        )
        assert (
            pred_proba_sql[1]
            == "(0.027423612860412977 * EXP(- POWER(age - 13.0217386792453, 2) / 423.253724660408) * POWER(0.910714285714286, pclass) * (CASE WHEN survived THEN 0.910714285714286 ELSE 0.08928571428571397 END) * DECODE(sex, 'female', 0.416666666666667, 'male', 0.583333333333333) * 0.1) / (0.004675073323276673 * EXP(- POWER(age - 63.9878308300395, 2) / 14563.75196754392) * POWER(0.771666666666667, pclass) * (CASE WHEN survived THEN 0.771666666666667 ELSE 0.22833333333333306 END) * DECODE(sex, 'female', 0.407843137254902, 'male', 0.592156862745098) * 0.8 + 0.027423612860412977 * EXP(- POWER(age - 13.0217386792453, 2) / 423.253724660408) * POWER(0.910714285714286, pclass) * (CASE WHEN survived THEN 0.910714285714286 ELSE 0.08928571428571397 END) * DECODE(sex, 'female', 0.416666666666667, 'male', 0.583333333333333) * 0.1 + 0.010555023401917874 * EXP(- POWER(age - 27.6928120412844, 2) / 2857.14134787876) * POWER(0.878216123499142, pclass) * (CASE WHEN survived THEN 0.878216123499142 ELSE 0.12178387650085798 END) * DECODE(sex, 'female', 0.406666666666667, 'male', 0.593333333333333) * 0.1)"
        )
        assert (
            pred_proba_sql[2]
            == "(0.010555023401917874 * EXP(- POWER(age - 27.6928120412844, 2) / 2857.14134787876) * POWER(0.878216123499142, pclass) * (CASE WHEN survived THEN 0.878216123499142 ELSE 0.12178387650085798 END) * DECODE(sex, 'female', 0.406666666666667, 'male', 0.593333333333333) * 0.1) / (0.004675073323276673 * EXP(- POWER(age - 63.9878308300395, 2) / 14563.75196754392) * POWER(0.771666666666667, pclass) * (CASE WHEN survived THEN 0.771666666666667 ELSE 0.22833333333333306 END) * DECODE(sex, 'female', 0.407843137254902, 'male', 0.592156862745098) * 0.8 + 0.027423612860412977 * EXP(- POWER(age - 13.0217386792453, 2) / 423.253724660408) * POWER(0.910714285714286, pclass) * (CASE WHEN survived THEN 0.910714285714286 ELSE 0.08928571428571397 END) * DECODE(sex, 'female', 0.416666666666667, 'male', 0.583333333333333) * 0.1 + 0.010555023401917874 * EXP(- POWER(age - 27.6928120412844, 2) / 2857.14134787876) * POWER(0.878216123499142, pclass) * (CASE WHEN survived THEN 0.878216123499142 ELSE 0.12178387650085798 END) * DECODE(sex, 'female', 0.406666666666667, 'male', 0.593333333333333) * 0.1)"
        )

    def test_get_attributes(self, memmodel_nb):
        """
        test function - get_attributes
        """
        attributes = memmodel_nb.get_attributes()
        assert attributes["prior"][0] == 0.8
        assert attributes["prior"][1] == 0.1
        assert attributes["prior"][2] == 0.1
        assert attributes["classes"][0] == "C"
        assert attributes["classes"][1] == "Q"
        assert attributes["classes"][2] == "S"
        assert attributes["attributes"][0]["type"] == "gaussian"
        assert attributes["attributes"][1]["type"] == "multinomial"
        assert attributes["attributes"][2]["type"] == "bernoulli"
        assert attributes["attributes"][3]["type"] == "categorical"

    def test_object_type(self, memmodel_nb):
        """
        test function - object_type
        """
        assert memmodel_nb.object_type == "NaiveBayes"
