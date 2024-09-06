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
from verticapy.machine_learning.memmodel.tree import NonBinaryTree
from verticapy import drop
from verticapy.datasets import load_titanic


@pytest.fixture(scope="module")
def memmodel_btr():
    children_left = [1, 3, None, None, None]
    children_right = [2, 4, None, None, None]
    feature = [0, 1, None, None, None]
    threshold = ["female", 30, None, None, None]
    value = [None, None, 3, 11, 1993]

    memmodel_btr = BinaryTreeRegressor(
        children_left=children_left,
        children_right=children_right,
        feature=feature,
        threshold=threshold,
        value=value,
    )

    yield memmodel_btr


@pytest.fixture(scope="module")
def memmodel_btc():
    children_left = [1, 3, None, None, None]
    children_right = [2, 4, None, None, None]
    feature = [0, 1, None, None, None]
    threshold = ["female", 30, None, None, None]
    value = [
        None,
        None,
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6],
    ]
    classes = ["a", "b", "c"]

    memmodel_btc = BinaryTreeClassifier(
        children_left=children_left,
        children_right=children_right,
        feature=feature,
        threshold=threshold,
        value=value,
        classes=classes,
    )
    yield memmodel_btc


@pytest.fixture(scope="module")
def titanic_vd():
    titanic = load_titanic()
    yield titanic
    drop(name="public.titanic")


@pytest.fixture(scope="module")
def memmodel_nbt(titanic_vd):
    nonbinary_tree = titanic_vd.chaid("survived", ["sex", "fare"]).tree_
    classes = ["a", "b"]
    memmodel_nbt = NonBinaryTree(nonbinary_tree, classes)
    yield memmodel_nbt


class TestBinaryTreeRegressor:
    """
    test class - TestBinaryTreeRegressor
    """

    @pytest.mark.parametrize(
        "sex, fare, expected_output",
        [
            ("male", 100, 3),
            ("female", 20, 11),
            ("female", 50, 1993),
        ],
    )
    def test_predict(self, memmodel_btr, sex, fare, expected_output):
        """
        test function - predict
        """
        data = [
            [sex, fare],
        ]
        prediction = memmodel_btr.predict(data)
        assert prediction[0] == pytest.approx(expected_output)

    def test_predict_sql(self, memmodel_btr):
        """
        test function - predict_sql
        """
        cnames = ["sex", "fare"]
        pred_sql = memmodel_btr.predict_sql(cnames)
        assert (
            pred_sql
            == "(CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 11 ELSE 1993 END) ELSE 3 END)"
        )

    def test_to_graphviz(self, memmodel_btr):
        """
        test function - test_to_graphviz
        """
        cnames = ["sex", "fare"]
        graphviz_code = memmodel_btr.to_graphviz(cnames)
        print(graphviz_code)
        assert 'digraph Tree {\ngraph [bgcolor="#FFFFFFDD"];' in graphviz_code

    def test_get_attributes(self, memmodel_btr):
        """
        test function - get_attributes
        """
        attributes = memmodel_btr.get_attributes()
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

    def test_object_type(self, memmodel_btr):
        """
        test function - object_type
        """
        assert memmodel_btr.object_type == "BinaryTreeRegressor"


class TestBinaryTreeClassifier:
    """
    test class - TestBinaryTreeClassifier
    """

    @pytest.mark.parametrize(
        "sex, fare, expected_output",
        [
            ("male", 100, "a"),
            ("female", 20, "b"),
            ("female", 50, "c"),
        ],
    )
    def test_predict(self, memmodel_btc, sex, fare, expected_output):
        """
        test function - predict
        """
        data = [
            [sex, fare],
        ]
        prediction = memmodel_btc.predict(data)
        assert prediction[0] == expected_output

    @pytest.mark.parametrize(
        "sex, fare, expected_output",
        [
            ("male", 100, [0.8, 0.1, 0.1]),
            ("female", 20, [0.1, 0.8, 0.1]),
            ("female", 50, [0.2, 0.2, 0.6]),
        ],
    )
    def test_predict_proba(self, memmodel_btc, sex, fare, expected_output):
        """
        test function - predict_proba
        """
        data = [
            [sex, fare],
        ]
        prediction = memmodel_btc.predict_proba(data)
        assert prediction[0][0] == pytest.approx(expected_output[0])
        assert prediction[0][1] == pytest.approx(expected_output[1])
        assert prediction[0][2] == pytest.approx(expected_output[2])

    def test_predict_sql(self, memmodel_btc):
        """
        test function - predict_sql
        """
        cnames = ["sex", "fare"]
        pred_sql = memmodel_btc.predict_sql(cnames)
        assert (
            pred_sql
            == "(CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 'b' ELSE 'c' END) ELSE 'a' END)"
        )

    def test_predict_proba_sql(self, memmodel_btc):
        """
        test function - predict_proba_sql
        """
        cnames = ["sex", "fare"]
        pred_proba_sql = memmodel_btc.predict_proba_sql(cnames)
        assert (
            pred_proba_sql[0]
            == "(CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.2 END) ELSE 0.8 END)"
        )
        assert (
            pred_proba_sql[1]
            == "(CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.8 ELSE 0.2 END) ELSE 0.1 END)"
        )
        assert (
            pred_proba_sql[2]
            == "(CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.6 END) ELSE 0.1 END)"
        )

    def test_to_graphviz(self, memmodel_btc):
        """
        test function - test_to_graphviz
        """
        cnames = ["sex", "fare"]
        graphviz_code = memmodel_btc.to_graphviz(cnames)
        assert 'digraph Tree {\ngraph [bgcolor="#FFFFFFDD"];\n0' in graphviz_code

    def test_get_attributes(self, memmodel_btc):
        """
        test function - get_attributes
        """
        attributes = memmodel_btc.get_attributes()
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
        assert attributes["classes"][0] == "a"
        assert attributes["classes"][1] == "b"
        assert attributes["classes"][2] == "c"

    def test_object_type(self, memmodel_btc):
        """
        test function - object_type
        """
        assert memmodel_btc.object_type == "BinaryTreeClassifier"


class TestNonBinaryTree:
    """
    test class - TestNonBinaryTree
    """

    @pytest.mark.parametrize(
        "sex, fare, expected_output",
        [
            ("male", 100, "a"),
            ("female", 20, "b"),
            ("female", 50, "b"),
        ],
    )
    def test_predict(self, memmodel_nbt, sex, fare, expected_output):
        """
        test function - predict
        """
        data = [
            [sex, fare],
        ]
        prediction = memmodel_nbt.predict(data)
        assert prediction[0] == expected_output

    @pytest.mark.parametrize(
        "sex, fare, expected_output",
        [
            ("male", 100, [0.82129278, 0.17870722]),
            ("female", 20, [0.3042328, 0.6957672]),
            ("female", 50, [0.3042328, 0.6957672]),
        ],
    )
    def test_predict_proba(self, memmodel_nbt, sex, fare, expected_output):
        """
        test function - predict_proba
        """
        data = [
            [sex, fare],
        ]
        prediction = memmodel_nbt.predict_proba(data)
        assert prediction[0][0] == pytest.approx(expected_output[0])
        assert prediction[0][1] == pytest.approx(expected_output[1])

    def test_predict_sql(self, memmodel_nbt):
        """
        test function - predict_sql
        """
        cnames = ["sex", "fare"]
        pred_sql = memmodel_nbt.predict_sql(cnames)
        assert (
            pred_sql
            == "(CASE WHEN sex = 'female' THEN (CASE WHEN fare <= 127.6 THEN 'b' WHEN fare <= 255.2 THEN 'b' WHEN fare <= 382.8 THEN 'b' WHEN fare <= 638.0 THEN 'b' ELSE NULL END) WHEN sex = 'male' THEN (CASE WHEN fare <= 129.36 THEN 'a' WHEN fare <= 258.72 THEN 'a' WHEN fare <= 388.08 THEN 'a' WHEN fare <= 517.44 THEN 'b' ELSE NULL END) ELSE NULL END)"
        )

    def test_predict_proba_sql(self, memmodel_nbt):
        """
        test function - predict_proba_sql
        """
        cnames = ["sex", "fare"]
        pred_proba_sql = memmodel_nbt.predict_proba_sql(cnames)
        assert (
            pred_proba_sql[0]
            == "(CASE WHEN sex = 'female' THEN (CASE WHEN fare <= 127.6 THEN 0.304232804232804 WHEN fare <= 255.2 THEN 0.09375 WHEN fare <= 382.8 THEN 0.0 WHEN fare <= 638.0 THEN 0.0 ELSE NULL END) WHEN sex = 'male' THEN (CASE WHEN fare <= 129.36 THEN 0.821292775665399 WHEN fare <= 258.72 THEN 0.777777777777778 WHEN fare <= 388.08 THEN 0.75 WHEN fare <= 517.44 THEN 0.0 ELSE NULL END) ELSE NULL END)"
        )
        assert (
            pred_proba_sql[1]
            == "(CASE WHEN sex = 'female' THEN (CASE WHEN fare <= 127.6 THEN 0.695767195767196 WHEN fare <= 255.2 THEN 0.90625 WHEN fare <= 382.8 THEN 1.0 WHEN fare <= 638.0 THEN 1.0 ELSE NULL END) WHEN sex = 'male' THEN (CASE WHEN fare <= 129.36 THEN 0.178707224334601 WHEN fare <= 258.72 THEN 0.222222222222222 WHEN fare <= 388.08 THEN 0.25 WHEN fare <= 517.44 THEN 1.0 ELSE NULL END) ELSE NULL END)"
        )

    def test_to_graphviz(self, memmodel_nbt):
        """
        test function - test_to_graphviz
        """
        cnames = ["sex", "fare"]
        graphviz_code = memmodel_nbt.to_graphviz(cnames)
        assert ' 2[label="<= 127.6"' in graphviz_code

    def test_get_attributes(self, memmodel_nbt):
        """
        test function - get_attributes
        """
        attributes = memmodel_nbt.get_attributes()
        assert attributes["tree"]["chi2"] == pytest.approx(345.12775126385327)
        assert not attributes["tree"]["is_leaf"]
        assert not attributes["tree"]["split_is_numerical"]
        assert attributes["tree"]["split_predictor"] == '"sex"'
        assert attributes["tree"]["split_predictor_idx"] == 0
        assert attributes["tree"]["children"]["female"]["chi2"] == pytest.approx(
            10.472532457814179
        )
        assert attributes["classes"][0] == "a"
        assert attributes["classes"][1] == "b"

    def test_object_type(self, memmodel_nbt):
        """
        test function - object_type
        """
        assert memmodel_nbt.object_type == "NonBinaryTree"
