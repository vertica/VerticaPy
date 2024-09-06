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
from verticapy.machine_learning.memmodel.tree import BinaryTreeRegressor
from verticapy.machine_learning.memmodel.tree import BinaryTreeClassifier
from verticapy.machine_learning.memmodel.ensemble import RandomForestRegressor
from verticapy.machine_learning.memmodel.ensemble import RandomForestClassifier
from verticapy.machine_learning.memmodel.ensemble import XGBRegressor
from verticapy.machine_learning.memmodel.ensemble import XGBClassifier


@pytest.fixture(scope="module")
def memmodel_rfr():
    model1 = BinaryTreeRegressor(
        children_left=[1, 3, None, None, None],
        children_right=[2, 4, None, None, None],
        feature=[0, 1, None, None, None],
        threshold=["female", 30, None, None, None],
        value=[None, None, 3.0, 11.0, 23.5],
    )
    model2 = BinaryTreeRegressor(
        children_left=[1, 3, None, None, None],
        children_right=[2, 4, None, None, None],
        feature=[0, 1, None, None, None],
        threshold=["female", 30, None, None, None],
        value=[None, None, -3, 12, 56],
    )
    model3 = BinaryTreeRegressor(
        children_left=[1, 3, None, None, None],
        children_right=[2, 4, None, None, None],
        feature=[0, 1, None, None, None],
        threshold=["female", 30, None, None, None],
        value=[None, None, 1, 3, 6],
    )
    memmodel_rfr = RandomForestRegressor(trees=[model1, model2, model3])
    yield memmodel_rfr


@pytest.fixture(scope="module")
def memmodel_rfc():
    model1 = BinaryTreeClassifier(
        children_left=[1, 3, None, None, None],
        children_right=[2, 4, None, None, None],
        feature=[0, 1, None, None, None],
        threshold=["female", 30, None, None, None],
        value=[None, None, [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]],
        classes=["a", "b", "c"],
    )
    model2 = BinaryTreeClassifier(
        children_left=[1, 3, None, None, None],
        children_right=[2, 4, None, None, None],
        feature=[0, 1, None, None, None],
        threshold=["female", 30, None, None, None],
        value=[None, None, [0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.2, 0.2, 0.6]],
        classes=["a", "b", "c"],
    )
    model3 = BinaryTreeClassifier(
        children_left=[1, 3, None, None, None],
        children_right=[2, 4, None, None, None],
        feature=[0, 1, None, None, None],
        threshold=["female", 30, None, None, None],
        value=[None, None, [0.4, 0.4, 0.2], [0.2, 0.2, 0.6], [0.2, 0.5, 0.3]],
        classes=["a", "b", "c"],
    )
    memmodel_rfc = RandomForestClassifier(
        trees=[model1, model2, model3],
        classes=["a", "b", "c"],
    )
    yield memmodel_rfc


@pytest.fixture(scope="module")
def memmodel_xgbr():
    model1 = BinaryTreeRegressor(
        children_left=[1, 3, None, None, None],
        children_right=[2, 4, None, None, None],
        feature=[0, 1, None, None, None],
        threshold=["female", 30, None, None, None],
        value=[None, None, 3.0, 11.0, 23.5],
    )
    model2 = BinaryTreeRegressor(
        children_left=[1, 3, None, None, None],
        children_right=[2, 4, None, None, None],
        feature=[0, 1, None, None, None],
        threshold=["female", 30, None, None, None],
        value=[None, None, -3, 12, 56],
    )
    model3 = BinaryTreeRegressor(
        children_left=[1, 3, None, None, None],
        children_right=[2, 4, None, None, None],
        feature=[0, 1, None, None, None],
        threshold=["female", 30, None, None, None],
        value=[None, None, 1, 3, 6],
    )
    memmodel_xgbr = XGBRegressor(
        trees=[model1, model2, model3],
        mean=2.5,
        eta=0.9,
    )
    yield memmodel_xgbr


@pytest.fixture(scope="module")
def memmodel_xgbc():
    model1 = BinaryTreeClassifier(
        children_left=[1, 3, None, None, None],
        children_right=[2, 4, None, None, None],
        feature=[0, 1, None, None, None],
        threshold=["female", 30, None, None, None],
        value=[None, None, [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]],
        classes=["a", "b", "c"],
    )
    model2 = BinaryTreeClassifier(
        children_left=[1, 3, None, None, None],
        children_right=[2, 4, None, None, None],
        feature=[0, 1, None, None, None],
        threshold=["female", 30, None, None, None],
        value=[None, None, [0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.2, 0.2, 0.6]],
        classes=["a", "b", "c"],
    )
    model3 = BinaryTreeClassifier(
        children_left=[1, 3, None, None, None],
        children_right=[2, 4, None, None, None],
        feature=[0, 1, None, None, None],
        threshold=["female", 30, None, None, None],
        value=[None, None, [0.4, 0.4, 0.2], [0.2, 0.2, 0.6], [0.2, 0.5, 0.3]],
        classes=["a", "b", "c"],
    )
    memmodel_xgbc = XGBClassifier(
        trees=[model1, model2, model3],
        classes=["a", "b", "c"],
        logodds=[0.1, 0.12, 0.15],
        learning_rate=0.1,
    )
    yield memmodel_xgbc


class TestRandomForestRegressor:
    """
    test class - TestRandomForestRegressor
    """

    @pytest.mark.parametrize(
        "sex, fare, expected_output",
        [
            ("male", 100, 0.33333333),
            ("female", 20, 8.66666667),
            ("female", 50, 28.5),
        ],
    )
    def test_predict(self, memmodel_rfr, sex, fare, expected_output):
        """
        test function - predict
        """
        data = [
            [sex, fare],
        ]
        prediction = memmodel_rfr.predict(data)
        assert prediction[0] == pytest.approx(expected_output)

    def test_predict_sql(self, memmodel_rfr):
        """
        test function - predict_sql
        """
        cnames = ["sex", "fare"]
        pred_sql = memmodel_rfr.predict_sql(cnames)
        assert (
            pred_sql
            == "((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 11.0 ELSE 23.5 END) ELSE 3.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 12 ELSE 56 END) ELSE -3 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 3 ELSE 6 END) ELSE 1 END)) / 3"
        )

    def test_get_attributes(self, memmodel_rfr):
        """
        test function - get_attributes
        """
        attributes = memmodel_rfr.get_attributes()["trees"][0].get_attributes()
        assert attributes["children_left"][0] == 1
        assert attributes["children_left"][1] == 3
        assert attributes["children_right"][0] == 2
        assert attributes["children_right"][1] == 4
        assert attributes["feature"][0] == 0
        assert attributes["feature"][1] == 1
        assert attributes["threshold"][0] == "female"
        assert attributes["threshold"][1] == 30
        assert attributes["value"][2] == 3
        assert attributes["value"][3] == 11

    def test_object_type(self, memmodel_rfr):
        """
        test function - object_type
        """
        assert memmodel_rfr.object_type == "RandomForestRegressor"


class TestRandomForestClassifier:
    """
    test class - TestRandomForestClassifier
    """

    @pytest.mark.parametrize(
        "sex, fare, expected_output",
        [
            ("male", 100, "a"),
            ("female", 20, "b"),
            ("female", 50, "c"),
        ],
    )
    def test_predict(self, memmodel_rfc, sex, fare, expected_output):
        """
        test function - predict
        """
        data = [
            [sex, fare],
        ]
        prediction = memmodel_rfc.predict(data)
        assert prediction[0] == expected_output

    @pytest.mark.parametrize(
        "sex, fare, expected_output",
        [
            ("male", 100, [1, 0, 0]),
            ("female", 20, [0, 0.66666667, 0.33333333]),
            ("female", 50, [0, 0.33333333, 0.66666667]),
        ],
    )
    def test_predict_proba(self, memmodel_rfc, sex, fare, expected_output):
        """
        test function - predict_proba
        """
        data = [
            [sex, fare],
        ]
        prediction = memmodel_rfc.predict_proba(data)
        assert prediction[0][0] == pytest.approx(expected_output[0])
        assert prediction[0][1] == pytest.approx(expected_output[1])
        assert prediction[0][2] == pytest.approx(expected_output[2])

    def test_predict_sql(self, memmodel_rfc):
        """
        test function - predict_sql
        """
        cnames = ["sex", "fare"]
        pred_sql = memmodel_rfc.predict_sql(cnames)
        assert (
            pred_sql
            == "CASE WHEN sex IS NULL OR fare IS NULL THEN NULL WHEN ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 1.0 END) ELSE 0.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 1.0 END) ELSE 0.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 1.0 ELSE 0.0 END) ELSE 0.0 END)) / 3 >= ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 0.0 END) ELSE 1.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 0.0 END) ELSE 1.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 0.0 END) ELSE 1.0 END)) / 3 AND ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 1.0 END) ELSE 0.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 1.0 END) ELSE 0.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 1.0 ELSE 0.0 END) ELSE 0.0 END)) / 3 >= ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 1.0 ELSE 0.0 END) ELSE 0.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 1.0 ELSE 0.0 END) ELSE 0.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 1.0 END) ELSE 0.0 END)) / 3 THEN 'c' WHEN ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 1.0 ELSE 0.0 END) ELSE 0.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 1.0 ELSE 0.0 END) ELSE 0.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 1.0 END) ELSE 0.0 END)) / 3 >= ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 0.0 END) ELSE 1.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 0.0 END) ELSE 1.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 0.0 END) ELSE 1.0 END)) / 3 THEN 'b' ELSE 'a' END"
        )

    def test_predict_proba_sql(self, memmodel_rfc):
        """
        test function - predict_proba_sql
        """
        cnames = ["sex", "fare"]
        pred_proba_sql = memmodel_rfc.predict_proba_sql(cnames)
        assert (
            pred_proba_sql[0]
            == "((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 0.0 END) ELSE 1.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 0.0 END) ELSE 1.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 0.0 END) ELSE 1.0 END)) / 3"
        )
        assert (
            pred_proba_sql[1]
            == "((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 1.0 ELSE 0.0 END) ELSE 0.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 1.0 ELSE 0.0 END) ELSE 0.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 1.0 END) ELSE 0.0 END)) / 3"
        )
        assert (
            pred_proba_sql[2]
            == "((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 1.0 END) ELSE 0.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 1.0 END) ELSE 0.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 1.0 ELSE 0.0 END) ELSE 0.0 END)) / 3"
        )

    def test_get_attributes(self, memmodel_rfc):
        """
        test function - get_attributes
        """
        attributes = memmodel_rfc.get_attributes()["trees"][0].get_attributes()
        assert attributes["children_left"][0] == 1
        assert attributes["children_left"][1] == 3
        assert attributes["children_right"][0] == 2
        assert attributes["children_right"][1] == 4
        assert attributes["feature"][0] == 0
        assert attributes["feature"][1] == 1
        assert attributes["threshold"][0] == "female"
        assert attributes["threshold"][1] == 30
        assert attributes["value"][2][0] == 0.8
        assert attributes["value"][3][0] == 0.1

    def test_object_type(self, memmodel_rfc):
        """
        test function - object_type
        """
        assert memmodel_rfc.object_type == "RandomForestClassifier"


class TestXGBRegressor:
    """
    test class - TestXGBRegressor
    """

    @pytest.mark.parametrize(
        "sex, fare, expected_output",
        [
            ("male", 100, 3.4),
            ("female", 20, 25.9),
            ("female", 50, 79.45),
        ],
    )
    def test_predict(self, memmodel_xgbr, sex, fare, expected_output):
        """
        test function - predict
        """
        data = [
            [sex, fare],
        ]
        prediction = memmodel_xgbr.predict(data)
        assert prediction[0] == pytest.approx(expected_output)

    def test_predict_sql(self, memmodel_xgbr):
        """
        test function - predict_sql
        """
        cnames = ["sex", "fare"]
        pred_sql = memmodel_xgbr.predict_sql(cnames)
        assert (
            pred_sql
            == "((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 11.0 ELSE 23.5 END) ELSE 3.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 12 ELSE 56 END) ELSE -3 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 3 ELSE 6 END) ELSE 1 END)) * 0.9 + 2.5"
        )

    def test_get_attributes(self, memmodel_xgbr):
        """
        test function - get_attributes
        """
        attributes = memmodel_xgbr.get_attributes()["trees"][0].get_attributes()
        assert attributes["children_left"][0] == 1
        assert attributes["children_left"][1] == 3
        assert attributes["children_right"][0] == 2
        assert attributes["children_right"][1] == 4
        assert attributes["feature"][0] == 0
        assert attributes["feature"][1] == 1
        assert attributes["threshold"][0] == "female"
        assert attributes["threshold"][1] == 30
        assert attributes["value"][2] == 3
        assert attributes["value"][3] == 11
        attributes = memmodel_xgbr.get_attributes()
        assert attributes["eta"] == 0.9
        assert attributes["mean"] == 2.5

    def test_object_type(self, memmodel_xgbr):
        """
        test function - object_type
        """
        assert memmodel_xgbr.object_type == "XGBRegressor"


class TestXGBClassifier:
    """
    test class - TestXGBClassifier
    """

    @pytest.mark.parametrize(
        "sex, fare, expected_output",
        [
            ("male", 100, "a"),
            ("female", 20, "b"),
            ("female", 50, "c"),
        ],
    )
    def test_predict(self, memmodel_xgbc, sex, fare, expected_output):
        """
        test function - predict
        """
        data = [
            [sex, fare],
        ]
        prediction = memmodel_xgbc.predict(data)
        assert prediction[0] == expected_output

    @pytest.mark.parametrize(
        "sex, fare, expected_output",
        [
            ("male", 100, [0.34318847, 0.32840576, 0.32840576]),
            ("female", 20, [0.32393829, 0.34024456, 0.33581715]),
            ("female", 50, [0.32394919, 0.33138502, 0.34466579]),
        ],
    )
    def test_predict_proba(self, memmodel_xgbc, sex, fare, expected_output):
        """
        test function - predict_proba
        """
        data = [
            [sex, fare],
        ]
        prediction = memmodel_xgbc.predict_proba(data)
        assert prediction[0][0] == pytest.approx(expected_output[0])
        assert prediction[0][1] == pytest.approx(expected_output[1])
        assert prediction[0][2] == pytest.approx(expected_output[2])

    def test_predict_sql(self, memmodel_xgbc):
        """
        test function - predict_sql
        """
        cnames = ["sex", "fare"]
        pred_sql = memmodel_xgbc.predict_sql(cnames)
        assert (
            pred_sql
            == "CASE WHEN sex IS NULL OR fare IS NULL THEN NULL WHEN (1 / (1 + EXP(- (0.15 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.6 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.6 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.3 END) ELSE 0.2 END)))))) / ((1 / (1 + EXP(- (0.1 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.2 END) ELSE 0.8 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.3 ELSE 0.2 END) ELSE 0.7 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.2 END) ELSE 0.4 END)))))) + (1 / (1 + EXP(- (0.12 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.8 ELSE 0.2 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.5 ELSE 0.2 END) ELSE 0.2 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.5 END) ELSE 0.4 END)))))) + (1 / (1 + EXP(- (0.15 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.6 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.6 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.3 END) ELSE 0.2 END))))))) >= (1 / (1 + EXP(- (0.1 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.2 END) ELSE 0.8 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.3 ELSE 0.2 END) ELSE 0.7 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.2 END) ELSE 0.4 END)))))) / ((1 / (1 + EXP(- (0.1 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.2 END) ELSE 0.8 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.3 ELSE 0.2 END) ELSE 0.7 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.2 END) ELSE 0.4 END)))))) + (1 / (1 + EXP(- (0.12 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.8 ELSE 0.2 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.5 ELSE 0.2 END) ELSE 0.2 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.5 END) ELSE 0.4 END)))))) + (1 / (1 + EXP(- (0.15 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.6 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.6 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.3 END) ELSE 0.2 END))))))) AND (1 / (1 + EXP(- (0.15 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.6 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.6 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.3 END) ELSE 0.2 END)))))) / ((1 / (1 + EXP(- (0.1 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.2 END) ELSE 0.8 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.3 ELSE 0.2 END) ELSE 0.7 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.2 END) ELSE 0.4 END)))))) + (1 / (1 + EXP(- (0.12 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.8 ELSE 0.2 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.5 ELSE 0.2 END) ELSE 0.2 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.5 END) ELSE 0.4 END)))))) + (1 / (1 + EXP(- (0.15 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.6 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.6 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.3 END) ELSE 0.2 END))))))) >= (1 / (1 + EXP(- (0.12 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.8 ELSE 0.2 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.5 ELSE 0.2 END) ELSE 0.2 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.5 END) ELSE 0.4 END)))))) / ((1 / (1 + EXP(- (0.1 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.2 END) ELSE 0.8 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.3 ELSE 0.2 END) ELSE 0.7 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.2 END) ELSE 0.4 END)))))) + (1 / (1 + EXP(- (0.12 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.8 ELSE 0.2 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.5 ELSE 0.2 END) ELSE 0.2 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.5 END) ELSE 0.4 END)))))) + (1 / (1 + EXP(- (0.15 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.6 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.6 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.3 END) ELSE 0.2 END))))))) THEN 'c' WHEN (1 / (1 + EXP(- (0.12 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.8 ELSE 0.2 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.5 ELSE 0.2 END) ELSE 0.2 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.5 END) ELSE 0.4 END)))))) / ((1 / (1 + EXP(- (0.1 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.2 END) ELSE 0.8 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.3 ELSE 0.2 END) ELSE 0.7 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.2 END) ELSE 0.4 END)))))) + (1 / (1 + EXP(- (0.12 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.8 ELSE 0.2 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.5 ELSE 0.2 END) ELSE 0.2 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.5 END) ELSE 0.4 END)))))) + (1 / (1 + EXP(- (0.15 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.6 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.6 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.3 END) ELSE 0.2 END))))))) >= (1 / (1 + EXP(- (0.1 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.2 END) ELSE 0.8 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.3 ELSE 0.2 END) ELSE 0.7 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.2 END) ELSE 0.4 END)))))) / ((1 / (1 + EXP(- (0.1 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.2 END) ELSE 0.8 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.3 ELSE 0.2 END) ELSE 0.7 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.2 END) ELSE 0.4 END)))))) + (1 / (1 + EXP(- (0.12 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.8 ELSE 0.2 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.5 ELSE 0.2 END) ELSE 0.2 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.5 END) ELSE 0.4 END)))))) + (1 / (1 + EXP(- (0.15 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.6 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.6 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.3 END) ELSE 0.2 END))))))) THEN 'b' ELSE 'a' END"
        )

    def test_predict_proba_sql(self, memmodel_xgbc):
        """
        test function - predict_proba_sql
        """
        cnames = ["sex", "fare"]
        pred_proba_sql = memmodel_xgbc.predict_proba_sql(cnames)
        assert (
            pred_proba_sql[0]
            == "(1 / (1 + EXP(- (0.1 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.2 END) ELSE 0.8 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.3 ELSE 0.2 END) ELSE 0.7 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.2 END) ELSE 0.4 END)))))) / ((1 / (1 + EXP(- (0.1 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.2 END) ELSE 0.8 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.3 ELSE 0.2 END) ELSE 0.7 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.2 END) ELSE 0.4 END)))))) + (1 / (1 + EXP(- (0.12 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.8 ELSE 0.2 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.5 ELSE 0.2 END) ELSE 0.2 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.5 END) ELSE 0.4 END)))))) + (1 / (1 + EXP(- (0.15 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.6 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.6 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.3 END) ELSE 0.2 END)))))))"
        )
        assert (
            pred_proba_sql[1]
            == "(1 / (1 + EXP(- (0.12 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.8 ELSE 0.2 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.5 ELSE 0.2 END) ELSE 0.2 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.5 END) ELSE 0.4 END)))))) / ((1 / (1 + EXP(- (0.1 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.2 END) ELSE 0.8 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.3 ELSE 0.2 END) ELSE 0.7 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.2 END) ELSE 0.4 END)))))) + (1 / (1 + EXP(- (0.12 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.8 ELSE 0.2 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.5 ELSE 0.2 END) ELSE 0.2 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.5 END) ELSE 0.4 END)))))) + (1 / (1 + EXP(- (0.15 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.6 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.6 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.3 END) ELSE 0.2 END)))))))"
        )
        assert (
            pred_proba_sql[2]
            == "(1 / (1 + EXP(- (0.15 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.6 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.6 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.3 END) ELSE 0.2 END)))))) / ((1 / (1 + EXP(- (0.1 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.2 END) ELSE 0.8 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.3 ELSE 0.2 END) ELSE 0.7 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.2 END) ELSE 0.4 END)))))) + (1 / (1 + EXP(- (0.12 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.8 ELSE 0.2 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.5 ELSE 0.2 END) ELSE 0.2 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.5 END) ELSE 0.4 END)))))) + (1 / (1 + EXP(- (0.15 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.6 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.6 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.3 END) ELSE 0.2 END)))))))"
        )

    def test_get_attributes(self, memmodel_xgbc):
        """
        test function - get_attributes
        """
        attributes = memmodel_xgbc.get_attributes()["trees"][0].get_attributes()
        assert attributes["children_left"][0] == 1
        assert attributes["children_left"][1] == 3
        assert attributes["children_right"][0] == 2
        assert attributes["children_right"][1] == 4
        assert attributes["feature"][0] == 0
        assert attributes["feature"][1] == 1
        assert attributes["threshold"][0] == "female"
        assert attributes["threshold"][1] == 30
        assert attributes["value"][2][0] == 0.8
        assert attributes["value"][3][0] == 0.1

    def test_object_type(self, memmodel_xgbc):
        """
        test function - object_type
        """
        assert memmodel_xgbc.object_type == "XGBClassifier"
