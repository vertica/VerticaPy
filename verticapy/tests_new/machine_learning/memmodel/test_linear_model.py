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
from verticapy.machine_learning.memmodel.linear_model import LinearModel
from verticapy.machine_learning.memmodel.linear_model import LinearModelClassifier


@pytest.fixture(scope="module")
def memmodel_lm():
    coefficients = [0.5, 1.2]
    intercept = 2.0
    memmodel_lm = LinearModel(coefficients, intercept)
    yield memmodel_lm


@pytest.fixture(scope="module")
def memmodel_lmc():
    coefficients = [0.5, 1.2]
    intercept = 2.0
    memmodel_lmc = LinearModelClassifier(coefficients, intercept)
    yield memmodel_lmc


class TestLinearModel:
    """
    test class - TestLinearModel
    """

    @pytest.mark.parametrize(
        "col1, col2, expected_output",
        [
            (1.0, 0.3, 2.86),
            (2.0, -0.6, 2.28),
        ],
    )
    def test_predict(self, memmodel_lm, col1, col2, expected_output):
        """
        test function - predict
        """
        data = [
            [col1, col2],
        ]
        prediction = memmodel_lm.predict(data)
        assert prediction[0] == pytest.approx(expected_output)

    # @pytest.mark.parametrize(
    # "col1, col2, expected_output",
    # [
    # (40.0, 1, True, "male", [0.64564673, 0.12105224, 0.23330103]),
    # (60.0, 3, True, "male", [0.74783083, 0.00570541, 0.24646376]),
    # (15.0, 2, False, "female", [0.34471925, 0.49592024, 0.15936051]),
    # ],
    # )
    # def test_predict_proba(
    # self, memmodel_lm, col1, col2, survived, sex, expected_output
    # ):
    # """
    # test function - predict_proba
    # """
    # data = [
    # [col1, col2, survived, sex],
    # ]
    # prediction = memmodel_lm.predict_proba(data)
    # assert prediction[0][0] == pytest.approx(expected_output[0])
    # assert prediction[0][1] == pytest.approx(expected_output[1])
    # assert prediction[0][2] == pytest.approx(expected_output[2])

    def test_predict_sql(self, memmodel_lm):
        """
        test function - predict_sql
        """
        cnames = ["col1", "col2"]
        pred_sql = memmodel_lm.predict_sql(cnames)
        assert pred_sql == "2.0 + 0.5 * col1 + 1.2 * col2"

    # def test_predict_proba_sql(self, memmodel_lm):
    # """
    # test function - predict_proba_sql
    # """
    # cnames = ["col1", "col2", "survived", "sex"]
    # pred_proba_sql = memmodel_lm.predict_proba_sql(cnames)
    # assert (
    # pred_proba_sql[0]
    # == "(0.004675073323276673 * EXP(- POWER(col1 - 63.9878308300395, 2) / 14563.75196754392) * POWER(0.771666666666667, col2) * (CASE WHEN survived THEN 0.771666666666667 ELSE 0.22833333333333306 END) * DECODE(sex, 'female', 0.407843137254902, 'male', 0.592156862745098) * 0.8) / (0.004675073323276673 * EXP(- POWER(col1 - 63.9878308300395, 2) / 14563.75196754392) * POWER(0.771666666666667, col2) * (CASE WHEN survived THEN 0.771666666666667 ELSE 0.22833333333333306 END) * DECODE(sex, 'female', 0.407843137254902, 'male', 0.592156862745098) * 0.8 + 0.027423612860412977 * EXP(- POWER(col1 - 13.0217386792453, 2) / 423.253724660408) * POWER(0.910714285714286, col2) * (CASE WHEN survived THEN 0.910714285714286 ELSE 0.08928571428571397 END) * DECODE(sex, 'female', 0.416666666666667, 'male', 0.583333333333333) * 0.1 + 0.010555023401917874 * EXP(- POWER(col1 - 27.6928120412844, 2) / 2857.14134787876) * POWER(0.878216123499142, col2) * (CASE WHEN survived THEN 0.878216123499142 ELSE 0.12178387650085798 END) * DECODE(sex, 'female', 0.406666666666667, 'male', 0.593333333333333) * 0.1)"
    # )
    # assert (
    # pred_proba_sql[1]
    # == "(0.027423612860412977 * EXP(- POWER(col1 - 13.0217386792453, 2) / 423.253724660408) * POWER(0.910714285714286, col2) * (CASE WHEN survived THEN 0.910714285714286 ELSE 0.08928571428571397 END) * DECODE(sex, 'female', 0.416666666666667, 'male', 0.583333333333333) * 0.1) / (0.004675073323276673 * EXP(- POWER(col1 - 63.9878308300395, 2) / 14563.75196754392) * POWER(0.771666666666667, col2) * (CASE WHEN survived THEN 0.771666666666667 ELSE 0.22833333333333306 END) * DECODE(sex, 'female', 0.407843137254902, 'male', 0.592156862745098) * 0.8 + 0.027423612860412977 * EXP(- POWER(col1 - 13.0217386792453, 2) / 423.253724660408) * POWER(0.910714285714286, col2) * (CASE WHEN survived THEN 0.910714285714286 ELSE 0.08928571428571397 END) * DECODE(sex, 'female', 0.416666666666667, 'male', 0.583333333333333) * 0.1 + 0.010555023401917874 * EXP(- POWER(col1 - 27.6928120412844, 2) / 2857.14134787876) * POWER(0.878216123499142, col2) * (CASE WHEN survived THEN 0.878216123499142 ELSE 0.12178387650085798 END) * DECODE(sex, 'female', 0.406666666666667, 'male', 0.593333333333333) * 0.1)"
    # )
    # assert (
    # pred_proba_sql[2]
    # == "(0.010555023401917874 * EXP(- POWER(col1 - 27.6928120412844, 2) / 2857.14134787876) * POWER(0.878216123499142, col2) * (CASE WHEN survived THEN 0.878216123499142 ELSE 0.12178387650085798 END) * DECODE(sex, 'female', 0.406666666666667, 'male', 0.593333333333333) * 0.1) / (0.004675073323276673 * EXP(- POWER(col1 - 63.9878308300395, 2) / 14563.75196754392) * POWER(0.771666666666667, col2) * (CASE WHEN survived THEN 0.771666666666667 ELSE 0.22833333333333306 END) * DECODE(sex, 'female', 0.407843137254902, 'male', 0.592156862745098) * 0.8 + 0.027423612860412977 * EXP(- POWER(col1 - 13.0217386792453, 2) / 423.253724660408) * POWER(0.910714285714286, col2) * (CASE WHEN survived THEN 0.910714285714286 ELSE 0.08928571428571397 END) * DECODE(sex, 'female', 0.416666666666667, 'male', 0.583333333333333) * 0.1 + 0.010555023401917874 * EXP(- POWER(col1 - 27.6928120412844, 2) / 2857.14134787876) * POWER(0.878216123499142, col2) * (CASE WHEN survived THEN 0.878216123499142 ELSE 0.12178387650085798 END) * DECODE(sex, 'female', 0.406666666666667, 'male', 0.593333333333333) * 0.1)"
    # )

    def test_get_attributes(self, memmodel_lm):
        """
        test function - get_attributes
        """
        attributes = memmodel_lm.get_attributes()
        assert attributes["coef"][0] == 0.5
        assert attributes["coef"][1] == 1.2
        assert attributes["intercept"] == 2.0

    def test_set_attributes(self, memmodel_lm):
        memmodel_lm.set_attributes(**{"coef": [-0.4, 0.5]})
        attributes = memmodel_lm.get_attributes()
        assert attributes["coef"][0] == -0.4
        assert attributes["coef"][1] == 0.5
        assert attributes["intercept"] == 2.0

    def test_object_type(self, memmodel_lm):
        """
        test function - object_type
        """
        assert memmodel_lm.object_type == "LinearModel"


class TestLinearModelClassifier:
    """
    test class - TestLinearModelClassifier
    """

    @pytest.mark.parametrize(
        "col1, col2, expected_output",
        [
            (1.0, 0.3, 1),
            (2.0, -0.6, 1),
        ],
    )
    def test_predict(self, memmodel_lmc, col1, col2, expected_output):
        """
        test function - predict
        """
        data = [
            [col1, col2],
        ]
        prediction = memmodel_lmc.predict(data)
        assert prediction[0] == pytest.approx(expected_output)

    @pytest.mark.parametrize(
        "col1, col2, expected_output",
        [
            (1.0, 0.3, [0.0541667, 0.9458333]),
            (-0.5, -0.8, [0.31216867, 0.68783133]),
        ],
    )
    def test_predict_proba(self, memmodel_lmc, col1, col2, expected_output):
        """
        test function - predict_proba
        """
        data = [
            [col1, col2],
        ]
        prediction = memmodel_lmc.predict_proba(data)
        assert prediction[0][0] == pytest.approx(expected_output[0])
        assert prediction[0][1] == pytest.approx(expected_output[1])

    def test_predict_sql(self, memmodel_lmc):
        """
        test function - predict_sql
        """
        cnames = ["col1", "col2"]
        pred_sql = memmodel_lmc.predict_sql(cnames)
        assert (
            pred_sql
            == "((1 / (1 + EXP(- (2.0 + 0.5 * col1 + 1.2 * col2)))) > 0.5)::int"
        )

    def test_predict_proba_sql(self, memmodel_lmc):
        """
        test function - predict_proba_sql
        """
        cnames = ["col1", "col2"]
        pred_proba_sql = memmodel_lmc.predict_proba_sql(cnames)
        assert (
            pred_proba_sql[0]
            == "1 - (1 / (1 + EXP(- (2.0 + 0.5 * col1 + 1.2 * col2))))"
        )
        assert pred_proba_sql[1] == "1 / (1 + EXP(- (2.0 + 0.5 * col1 + 1.2 * col2)))"

    def test_get_attributes(self, memmodel_lmc):
        """
        test function - get_attributes
        """
        attributes = memmodel_lmc.get_attributes()
        assert attributes["coef"][0] == 0.5
        assert attributes["coef"][1] == 1.2
        assert attributes["intercept"] == 2.0

    def test_set_attributes(self, memmodel_lmc):
        memmodel_lmc.set_attributes(**{"coef": [-0.4, 0.5]})
        attributes = memmodel_lmc.get_attributes()
        assert attributes["coef"][0] == -0.4
        assert attributes["coef"][1] == 0.5
        assert attributes["intercept"] == 2.0

    def test_object_type(self, memmodel_lmc):
        """
        test function - object_type
        """
        assert memmodel_lmc.object_type == "LinearModelClassifier"
