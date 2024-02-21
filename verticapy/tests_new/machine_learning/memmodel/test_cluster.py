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
from verticapy.machine_learning.memmodel.cluster import KMeans
from verticapy.machine_learning.memmodel.cluster import KPrototypes
from verticapy.machine_learning.memmodel.cluster import BisectingKMeans
from verticapy.machine_learning.memmodel.cluster import NearestCentroid


@pytest.fixture(scope="module")
def memmodel_km():
    clusters = [
        [0.5, 0.6],
        [1, 2],
        [100, 200],
    ]
    p = 2
    memmodel_km = KMeans(clusters, p)
    yield memmodel_km


@pytest.fixture(scope="module")
def memmodel_kp():
    clusters = [
        [0.5, "high"],
        [1, "low"],
        [100, "high"],
    ]
    p = 2
    gamma = 1.0
    is_categorical = [0, 1]
    memmodel_kp = KPrototypes(clusters, p, gamma, is_categorical)
    yield memmodel_kp


@pytest.fixture(scope="module")
def memmodel_bkm():
    clusters = [
        [0.5, 0.6],
        [1, 2],
        [100, 200],
        [10, 700],
        [-100, -200],
    ]
    children_left = [1, 3, None, None, None]
    children_right = [2, 4, None, None, None]
    memmodel_bkm = BisectingKMeans(clusters, children_left, children_right)
    yield memmodel_bkm


@pytest.fixture(scope="module")
def memmodel_nc():
    clusters = [[0.5, 0.6], [1, 2], [100, 200]]
    p = 2
    classes = ["class_a", "class_b", "class_c"]
    memmodel_nc = NearestCentroid(clusters, classes, p)
    yield memmodel_nc


class TestKMeans:
    """
    test class - TestKMeans
    """

    @pytest.mark.parametrize(
        "col1, col2, expected_output",
        [
            (2, 3, 1),
            (100, 150, 2),
        ],
    )
    def test_predict(self, memmodel_km, col1, col2, expected_output):
        """
        test function - predict
        """
        data = [
            [col1, col2],
        ]
        prediction = memmodel_km.predict(data)
        assert prediction[0] == expected_output

    @pytest.mark.parametrize(
        "col1, col2, expected_output",
        [
            (2, 3, [0.33177263, 0.66395985, 0.00426752]),
            (100, 150, [0.17863144, 0.1800781, 0.64129046]),
        ],
    )
    def test_predict_proba(self, memmodel_km, col1, col2, expected_output):
        """
        test function - predict_proba
        """
        data = [
            [col1, col2],
        ]
        prediction = memmodel_km.predict_proba(data)
        assert prediction[0][0] == pytest.approx(expected_output[0])
        assert prediction[0][1] == pytest.approx(expected_output[1])
        assert prediction[0][2] == pytest.approx(expected_output[2])

    @pytest.mark.parametrize(
        "col1, col2, expected_output",
        [
            (2, 3, [2.83019434, 1.41421356, 220.02954347]),
            (100, 150, [179.50100278, 178.05897899, 50]),
        ],
    )
    def test_transform(self, memmodel_km, col1, col2, expected_output):
        """
        test function - transform
        """
        data = [
            [col1, col2],
        ]
        prediction = memmodel_km.transform(data)
        assert prediction[0][0] == pytest.approx(expected_output[0])
        assert prediction[0][1] == pytest.approx(expected_output[1])
        assert prediction[0][2] == pytest.approx(expected_output[2])

    def test_predict_sql(self, memmodel_km):
        """
        test function - predict_sql
        """
        cnames = ["col1", "col2"]
        pred_sql = memmodel_km.predict_sql(cnames)
        assert (
            pred_sql
            == "CASE WHEN col1 IS NULL OR col2 IS NULL THEN NULL WHEN POWER(POWER(col1 - 100.0, 2) + POWER(col2 - 200.0, 2), 1 / 2) <= POWER(POWER(col1 - 0.5, 2) + POWER(col2 - 0.6, 2), 1 / 2) AND POWER(POWER(col1 - 100.0, 2) + POWER(col2 - 200.0, 2), 1 / 2) <= POWER(POWER(col1 - 1.0, 2) + POWER(col2 - 2.0, 2), 1 / 2) THEN 2 WHEN POWER(POWER(col1 - 1.0, 2) + POWER(col2 - 2.0, 2), 1 / 2) <= POWER(POWER(col1 - 0.5, 2) + POWER(col2 - 0.6, 2), 1 / 2) THEN 1 ELSE 0 END"
        )

    def test_predict_proba_sql(self, memmodel_km):
        """
        test function - predict_proba_sql
        """
        cnames = ["col1", "col2"]
        pred_proba_sql = memmodel_km.predict_proba_sql(cnames)
        assert (
            pred_proba_sql[0]
            == "(CASE WHEN POWER(POWER(col1 - 0.5, 2) + POWER(col2 - 0.6, 2), 1 / 2) = 0 THEN 1.0 ELSE 1 / (POWER(POWER(col1 - 0.5, 2) + POWER(col2 - 0.6, 2), 1 / 2)) / (1 / (POWER(POWER(col1 - 0.5, 2) + POWER(col2 - 0.6, 2), 1 / 2)) + 1 / (POWER(POWER(col1 - 1.0, 2) + POWER(col2 - 2.0, 2), 1 / 2)) + 1 / (POWER(POWER(col1 - 100.0, 2) + POWER(col2 - 200.0, 2), 1 / 2))) END)"
        )
        assert (
            pred_proba_sql[1]
            == "(CASE WHEN POWER(POWER(col1 - 1.0, 2) + POWER(col2 - 2.0, 2), 1 / 2) = 0 THEN 1.0 ELSE 1 / (POWER(POWER(col1 - 1.0, 2) + POWER(col2 - 2.0, 2), 1 / 2)) / (1 / (POWER(POWER(col1 - 0.5, 2) + POWER(col2 - 0.6, 2), 1 / 2)) + 1 / (POWER(POWER(col1 - 1.0, 2) + POWER(col2 - 2.0, 2), 1 / 2)) + 1 / (POWER(POWER(col1 - 100.0, 2) + POWER(col2 - 200.0, 2), 1 / 2))) END)"
        )
        assert (
            pred_proba_sql[2]
            == "(CASE WHEN POWER(POWER(col1 - 100.0, 2) + POWER(col2 - 200.0, 2), 1 / 2) = 0 THEN 1.0 ELSE 1 / (POWER(POWER(col1 - 100.0, 2) + POWER(col2 - 200.0, 2), 1 / 2)) / (1 / (POWER(POWER(col1 - 0.5, 2) + POWER(col2 - 0.6, 2), 1 / 2)) + 1 / (POWER(POWER(col1 - 1.0, 2) + POWER(col2 - 2.0, 2), 1 / 2)) + 1 / (POWER(POWER(col1 - 100.0, 2) + POWER(col2 - 200.0, 2), 1 / 2))) END)"
        )

    def test_transform_sql(self, memmodel_km):
        """
        test function - transform_sql
        """
        cnames = ["col1", "col2"]
        transform_sql = memmodel_km.transform_sql(cnames)
        assert (
            transform_sql[0]
            == "POWER(POWER(col1 - 0.5, 2) + POWER(col2 - 0.6, 2), 1 / 2)"
        )
        assert (
            transform_sql[1]
            == "POWER(POWER(col1 - 1.0, 2) + POWER(col2 - 2.0, 2), 1 / 2)"
        )
        assert (
            transform_sql[2]
            == "POWER(POWER(col1 - 100.0, 2) + POWER(col2 - 200.0, 2), 1 / 2)"
        )

    def test_get_attributes(self, memmodel_km):
        """
        test function - get_attributes
        """
        attributes = memmodel_km.get_attributes()
        assert attributes["clusters"][0][0] == 0.5
        assert attributes["clusters"][0][1] == 0.6
        assert attributes["clusters"][1][0] == 1
        assert attributes["clusters"][1][1] == 2
        assert attributes["p"] == 2

    def test_object_type(self, memmodel_km):
        """
        test function - object_type
        """
        assert memmodel_km.object_type == "KMeans"


class TestKPrototypes:
    """
    test class - TestKPrototypes
    """

    @pytest.mark.parametrize(
        "col1, col2, expected_output",
        [
            (2, "low", 1),
            (90, "high", 2),
        ],
    )
    def test_predict(self, memmodel_kp, col1, col2, expected_output):
        """
        test function - predict
        """
        data = [
            [col1, col2],
        ]
        prediction = memmodel_kp.predict(data)
        assert prediction[0] == expected_output

    @pytest.mark.parametrize(
        "col1, col2, expected_output",
        [
            (2, "low", [2.35275386e-01, 7.64645005e-01, 7.96090583e-05]),
            (90, "high", [0.01217824, 0.01231391, 0.97550785]),
        ],
    )
    def test_predict_proba(self, memmodel_kp, col1, col2, expected_output):
        """
        test function - predict_proba
        """
        data = [
            [col1, col2],
        ]
        prediction = memmodel_kp.predict_proba(data)
        assert prediction[0][0] == pytest.approx(expected_output[0])
        assert prediction[0][1] == pytest.approx(expected_output[1])
        assert prediction[0][2] == pytest.approx(expected_output[2])

    @pytest.mark.parametrize(
        "col1, col2, expected_output",
        [
            (2, "low", [3.250e00, 1.000e00, 9.605e03]),
            (90, "high", [8010.25, 7922, 100]),
        ],
    )
    def test_transform(self, memmodel_kp, col1, col2, expected_output):
        """
        test function - transform
        """
        data = [
            [col1, col2],
        ]
        prediction = memmodel_kp.transform(data)
        assert prediction[0][0] == pytest.approx(expected_output[0])
        assert prediction[0][1] == pytest.approx(expected_output[1])
        assert prediction[0][2] == pytest.approx(expected_output[2])

    def test_predict_sql(self, memmodel_kp):
        """
        test function - predict_sql
        """
        cnames = ["col1", "col2"]
        pred_sql = memmodel_kp.predict_sql(cnames)
        assert (
            pred_sql
            == "CASE WHEN col1 IS NULL OR col2 IS NULL THEN NULL WHEN POWER(POWER(col1 - 100, 2), 1 / 2) + 1.0 * (ABS((col2 = 'high')::int - 1)) <= POWER(POWER(col1 - 0.5, 2), 1 / 2) + 1.0 * (ABS((col2 = 'high')::int - 1)) AND POWER(POWER(col1 - 100, 2), 1 / 2) + 1.0 * (ABS((col2 = 'high')::int - 1)) <= POWER(POWER(col1 - 1, 2), 1 / 2) + 1.0 * (ABS((col2 = 'low')::int - 1)) THEN 2 WHEN POWER(POWER(col1 - 1, 2), 1 / 2) + 1.0 * (ABS((col2 = 'low')::int - 1)) <= POWER(POWER(col1 - 0.5, 2), 1 / 2) + 1.0 * (ABS((col2 = 'high')::int - 1)) THEN 1 ELSE 0 END"
        )

    def test_predict_proba_sql(self, memmodel_kp):
        """
        test function - predict_proba_sql
        """
        cnames = ["col1", "col2"]
        pred_proba_sql = memmodel_kp.predict_proba_sql(cnames)
        assert (
            pred_proba_sql[0]
            == "(CASE WHEN POWER(POWER(col1 - 0.5, 2), 1 / 2) + 1.0 * (ABS((col2 = 'high')::int - 1)) = 0 THEN 1.0 ELSE 1 / (POWER(POWER(col1 - 0.5, 2), 1 / 2) + 1.0 * (ABS((col2 = 'high')::int - 1))) / (1 / (POWER(POWER(col1 - 0.5, 2), 1 / 2) + 1.0 * (ABS((col2 = 'high')::int - 1))) + 1 / (POWER(POWER(col1 - 1, 2), 1 / 2) + 1.0 * (ABS((col2 = 'low')::int - 1))) + 1 / (POWER(POWER(col1 - 100, 2), 1 / 2) + 1.0 * (ABS((col2 = 'high')::int - 1)))) END)"
        )
        assert (
            pred_proba_sql[1]
            == "(CASE WHEN POWER(POWER(col1 - 1, 2), 1 / 2) + 1.0 * (ABS((col2 = 'low')::int - 1)) = 0 THEN 1.0 ELSE 1 / (POWER(POWER(col1 - 1, 2), 1 / 2) + 1.0 * (ABS((col2 = 'low')::int - 1))) / (1 / (POWER(POWER(col1 - 0.5, 2), 1 / 2) + 1.0 * (ABS((col2 = 'high')::int - 1))) + 1 / (POWER(POWER(col1 - 1, 2), 1 / 2) + 1.0 * (ABS((col2 = 'low')::int - 1))) + 1 / (POWER(POWER(col1 - 100, 2), 1 / 2) + 1.0 * (ABS((col2 = 'high')::int - 1)))) END)"
        )
        assert (
            pred_proba_sql[2]
            == "(CASE WHEN POWER(POWER(col1 - 100, 2), 1 / 2) + 1.0 * (ABS((col2 = 'high')::int - 1)) = 0 THEN 1.0 ELSE 1 / (POWER(POWER(col1 - 100, 2), 1 / 2) + 1.0 * (ABS((col2 = 'high')::int - 1))) / (1 / (POWER(POWER(col1 - 0.5, 2), 1 / 2) + 1.0 * (ABS((col2 = 'high')::int - 1))) + 1 / (POWER(POWER(col1 - 1, 2), 1 / 2) + 1.0 * (ABS((col2 = 'low')::int - 1))) + 1 / (POWER(POWER(col1 - 100, 2), 1 / 2) + 1.0 * (ABS((col2 = 'high')::int - 1)))) END)"
        )

    def test_transform_sql(self, memmodel_kp):
        """
        test function - transform_sql
        """
        cnames = ["col1", "col2"]
        transform_sql = memmodel_kp.transform_sql(cnames)
        assert (
            transform_sql[0]
            == "POWER(POWER(col1 - 0.5, 2), 1 / 2) + 1.0 * (ABS((col2 = 'high')::int - 1))"
        )
        assert (
            transform_sql[1]
            == "POWER(POWER(col1 - 1, 2), 1 / 2) + 1.0 * (ABS((col2 = 'low')::int - 1))"
        )
        assert (
            transform_sql[2]
            == "POWER(POWER(col1 - 100, 2), 1 / 2) + 1.0 * (ABS((col2 = 'high')::int - 1))"
        )

    def test_get_attributes(self, memmodel_kp):
        """
        test function - get_attributes
        """
        attributes = memmodel_kp.get_attributes()
        assert attributes["clusters"][0][0] == "0.5"
        assert attributes["clusters"][0][1] == "high"
        assert attributes["clusters"][1][0] == "1"
        assert attributes["clusters"][1][1] == "low"
        assert attributes["p"] == 2
        assert attributes["gamma"] == 1.0
        assert attributes["is_categorical"][0] == 0
        assert attributes["is_categorical"][1] == 1

    def test_object_type(self, memmodel_kp):
        """
        test function - object_type
        """
        assert memmodel_kp.object_type == "KPrototypes"


class TestBisectingKMeans:
    """
    test class - TestBisectingKMeans
    """

    @pytest.mark.parametrize(
        "col1, col2, expected_output",
        [
            (2, 3, 4),
            (100, 150, 2),
        ],
    )
    def test_predict(self, memmodel_bkm, col1, col2, expected_output):
        """
        test function - predict
        """
        data = [
            [col1, col2],
        ]
        prediction = memmodel_bkm.predict(data)
        assert prediction[0] == expected_output

    @pytest.mark.parametrize(
        "col1, col2, expected_output",
        [
            (2, 3, [0.32996436, 0.66034105, 0.00424426, 0.001339744, 0.00411059]),
            (100, 150, [0.15709716, 0.15836942, 0.56398194, 0.05059813, 0.06995335]),
        ],
    )
    def test_predict_proba(self, memmodel_bkm, col1, col2, expected_output):
        """
        test function - predict_proba
        """
        data = [
            [col1, col2],
        ]
        prediction = memmodel_bkm.predict_proba(data)
        assert prediction[0][0] == pytest.approx(expected_output[0])
        assert prediction[0][1] == pytest.approx(expected_output[1])
        assert prediction[0][2] == pytest.approx(expected_output[2])
        assert prediction[0][3] == pytest.approx(expected_output[3])
        assert prediction[0][4] == pytest.approx(expected_output[4])

    @pytest.mark.parametrize(
        "col1, col2, expected_output",
        [
            (2, 3, [2.83019434, 1.41421356, 220.02954347, 697.04590954, 227.18494668]),
            (100, 150, [179.50100278, 178.05897899, 50, 557.31499172, 403.11288741]),
        ],
    )
    def test_transform(self, memmodel_bkm, col1, col2, expected_output):
        """
        test function - transform
        """
        data = [
            [col1, col2],
        ]
        prediction = memmodel_bkm.transform(data)
        assert prediction[0][0] == pytest.approx(expected_output[0])
        assert prediction[0][1] == pytest.approx(expected_output[1])
        assert prediction[0][2] == pytest.approx(expected_output[2])
        assert prediction[0][3] == pytest.approx(expected_output[3])
        assert prediction[0][4] == pytest.approx(expected_output[4])

    def test_predict_sql(self, memmodel_bkm):
        """
        test function - predict_sql
        """
        cnames = ["col1", "col2"]
        pred_sql = memmodel_bkm.predict_sql(cnames)
        assert (
            pred_sql
            == "(CASE WHEN col1 IS NULL OR col2 IS NULL THEN NULL ELSE (CASE WHEN POWER(POWER(col1 - 1.0, 2) + POWER(col2 - 2.0, 2), 1/2) < POWER(POWER(col1 - 100.0, 2) + POWER(col2 - 200.0, 2), 1/2) THEN (CASE WHEN POWER(POWER(col1 - 10.0, 2) + POWER(col2 - 700.0, 2), 1/2) < POWER(POWER(col1 - -100.0, 2) + POWER(col2 - -200.0, 2), 1/2) THEN 3 ELSE 4 END) ELSE 2 END) END)"
        )

    def test_predict_proba_sql(self, memmodel_bkm):
        """
        test function - predict_proba_sql
        """
        cnames = ["col1", "col2"]
        pred_proba_sql = memmodel_bkm.predict_proba_sql(cnames)
        assert (
            pred_proba_sql[0]
            == "(CASE WHEN POWER(POWER(col1 - 0.5, 2) + POWER(col2 - 0.6, 2), 1 / 2) = 0 THEN 1.0 ELSE 1 / (POWER(POWER(col1 - 0.5, 2) + POWER(col2 - 0.6, 2), 1 / 2)) / (1 / (POWER(POWER(col1 - 0.5, 2) + POWER(col2 - 0.6, 2), 1 / 2)) + 1 / (POWER(POWER(col1 - 1.0, 2) + POWER(col2 - 2.0, 2), 1 / 2)) + 1 / (POWER(POWER(col1 - 100.0, 2) + POWER(col2 - 200.0, 2), 1 / 2)) + 1 / (POWER(POWER(col1 - 10.0, 2) + POWER(col2 - 700.0, 2), 1 / 2)) + 1 / (POWER(POWER(col1 - -100.0, 2) + POWER(col2 - -200.0, 2), 1 / 2))) END)"
        )
        assert (
            pred_proba_sql[1]
            == "(CASE WHEN POWER(POWER(col1 - 1.0, 2) + POWER(col2 - 2.0, 2), 1 / 2) = 0 THEN 1.0 ELSE 1 / (POWER(POWER(col1 - 1.0, 2) + POWER(col2 - 2.0, 2), 1 / 2)) / (1 / (POWER(POWER(col1 - 0.5, 2) + POWER(col2 - 0.6, 2), 1 / 2)) + 1 / (POWER(POWER(col1 - 1.0, 2) + POWER(col2 - 2.0, 2), 1 / 2)) + 1 / (POWER(POWER(col1 - 100.0, 2) + POWER(col2 - 200.0, 2), 1 / 2)) + 1 / (POWER(POWER(col1 - 10.0, 2) + POWER(col2 - 700.0, 2), 1 / 2)) + 1 / (POWER(POWER(col1 - -100.0, 2) + POWER(col2 - -200.0, 2), 1 / 2))) END)"
        )
        assert (
            pred_proba_sql[2]
            == "(CASE WHEN POWER(POWER(col1 - 100.0, 2) + POWER(col2 - 200.0, 2), 1 / 2) = 0 THEN 1.0 ELSE 1 / (POWER(POWER(col1 - 100.0, 2) + POWER(col2 - 200.0, 2), 1 / 2)) / (1 / (POWER(POWER(col1 - 0.5, 2) + POWER(col2 - 0.6, 2), 1 / 2)) + 1 / (POWER(POWER(col1 - 1.0, 2) + POWER(col2 - 2.0, 2), 1 / 2)) + 1 / (POWER(POWER(col1 - 100.0, 2) + POWER(col2 - 200.0, 2), 1 / 2)) + 1 / (POWER(POWER(col1 - 10.0, 2) + POWER(col2 - 700.0, 2), 1 / 2)) + 1 / (POWER(POWER(col1 - -100.0, 2) + POWER(col2 - -200.0, 2), 1 / 2))) END)"
        )
        assert (
            pred_proba_sql[3]
            == "(CASE WHEN POWER(POWER(col1 - 10.0, 2) + POWER(col2 - 700.0, 2), 1 / 2) = 0 THEN 1.0 ELSE 1 / (POWER(POWER(col1 - 10.0, 2) + POWER(col2 - 700.0, 2), 1 / 2)) / (1 / (POWER(POWER(col1 - 0.5, 2) + POWER(col2 - 0.6, 2), 1 / 2)) + 1 / (POWER(POWER(col1 - 1.0, 2) + POWER(col2 - 2.0, 2), 1 / 2)) + 1 / (POWER(POWER(col1 - 100.0, 2) + POWER(col2 - 200.0, 2), 1 / 2)) + 1 / (POWER(POWER(col1 - 10.0, 2) + POWER(col2 - 700.0, 2), 1 / 2)) + 1 / (POWER(POWER(col1 - -100.0, 2) + POWER(col2 - -200.0, 2), 1 / 2))) END)"
        )
        assert (
            pred_proba_sql[4]
            == "(CASE WHEN POWER(POWER(col1 - -100.0, 2) + POWER(col2 - -200.0, 2), 1 / 2) = 0 THEN 1.0 ELSE 1 / (POWER(POWER(col1 - -100.0, 2) + POWER(col2 - -200.0, 2), 1 / 2)) / (1 / (POWER(POWER(col1 - 0.5, 2) + POWER(col2 - 0.6, 2), 1 / 2)) + 1 / (POWER(POWER(col1 - 1.0, 2) + POWER(col2 - 2.0, 2), 1 / 2)) + 1 / (POWER(POWER(col1 - 100.0, 2) + POWER(col2 - 200.0, 2), 1 / 2)) + 1 / (POWER(POWER(col1 - 10.0, 2) + POWER(col2 - 700.0, 2), 1 / 2)) + 1 / (POWER(POWER(col1 - -100.0, 2) + POWER(col2 - -200.0, 2), 1 / 2))) END)"
        )

    def test_transform_sql(self, memmodel_bkm):
        """
        test function - transform_sql
        """
        cnames = ["col1", "col2"]
        transform_sql = memmodel_bkm.transform_sql(cnames)
        assert (
            transform_sql[0]
            == "POWER(POWER(col1 - 0.5, 2) + POWER(col2 - 0.6, 2), 1 / 2)"
        )
        assert (
            transform_sql[1]
            == "POWER(POWER(col1 - 1.0, 2) + POWER(col2 - 2.0, 2), 1 / 2)"
        )
        assert (
            transform_sql[2]
            == "POWER(POWER(col1 - 100.0, 2) + POWER(col2 - 200.0, 2), 1 / 2)"
        )
        assert (
            transform_sql[3]
            == "POWER(POWER(col1 - 10.0, 2) + POWER(col2 - 700.0, 2), 1 / 2)"
        )
        assert (
            transform_sql[4]
            == "POWER(POWER(col1 - -100.0, 2) + POWER(col2 - -200.0, 2), 1 / 2)"
        )

    def test_get_attributes(self, memmodel_bkm):
        """
        test function - get_attributes
        """
        attributes = memmodel_bkm.get_attributes()
        assert attributes["clusters"][0][0] == 0.5
        assert attributes["clusters"][0][1] == 0.6
        assert attributes["clusters"][1][0] == 1
        assert attributes["clusters"][1][1] == 2
        assert attributes["children_left"][0] == 1
        assert attributes["children_right"][1] == 4

    def test_object_type(self, memmodel_bkm):
        """
        test function - object_type
        """
        assert memmodel_bkm.object_type == "BisectingKMeans"


class TestNearestCentroid:
    """
    test class - TestNearestCentroid
    """

    @pytest.mark.parametrize(
        "col1, col2, expected_output",
        [
            (2, 3, "class_b"),
            (100, 150, "class_c"),
        ],
    )
    def test_predict(self, memmodel_nc, col1, col2, expected_output):
        """
        test function - predict
        """
        data = [
            [col1, col2],
        ]
        prediction = memmodel_nc.predict(data)
        assert prediction[0] == expected_output

    @pytest.mark.parametrize(
        "col1, col2, expected_output",
        [
            (2, 3, [0.33177263, 0.66395985, 0.00426752]),
            (100, 150, [0.17863144, 0.1800781, 0.64129046]),
        ],
    )
    def test_predict_proba(self, memmodel_nc, col1, col2, expected_output):
        """
        test function - predict_proba
        """
        data = [
            [col1, col2],
        ]
        prediction = memmodel_nc.predict_proba(data)
        assert prediction[0][0] == pytest.approx(expected_output[0])
        assert prediction[0][1] == pytest.approx(expected_output[1])
        assert prediction[0][2] == pytest.approx(expected_output[2])

    @pytest.mark.parametrize(
        "col1, col2, expected_output",
        [
            (2, 3, [2.83019434, 1.41421356, 220.02954347]),
            (100, 150, [179.50100278, 178.05897899, 50]),
        ],
    )
    def test_transform(self, memmodel_nc, col1, col2, expected_output):
        """
        test function - transform
        """
        data = [
            [col1, col2],
        ]
        prediction = memmodel_nc.transform(data)
        assert prediction[0][0] == pytest.approx(expected_output[0])
        assert prediction[0][1] == pytest.approx(expected_output[1])
        assert prediction[0][2] == pytest.approx(expected_output[2])

    def test_predict_sql(self, memmodel_nc):
        """
        test function - predict_sql
        """
        cnames = ["col1", "col2"]
        pred_sql = memmodel_nc.predict_sql(cnames)
        assert (
            pred_sql
            == "CASE WHEN col1 IS NULL OR col2 IS NULL THEN NULL WHEN POWER(POWER(col1 - 100.0, 2) + POWER(col2 - 200.0, 2), 1 / 2) <= POWER(POWER(col1 - 0.5, 2) + POWER(col2 - 0.6, 2), 1 / 2) AND POWER(POWER(col1 - 100.0, 2) + POWER(col2 - 200.0, 2), 1 / 2) <= POWER(POWER(col1 - 1.0, 2) + POWER(col2 - 2.0, 2), 1 / 2) THEN 'class_c' WHEN POWER(POWER(col1 - 1.0, 2) + POWER(col2 - 2.0, 2), 1 / 2) <= POWER(POWER(col1 - 0.5, 2) + POWER(col2 - 0.6, 2), 1 / 2) THEN 'class_b' ELSE 'class_a' END"
        )

    def test_predict_proba_sql(self, memmodel_nc):
        """
        test function - predict_proba_sql
        """
        cnames = ["col1", "col2"]
        pred_proba_sql = memmodel_nc.predict_proba_sql(cnames)
        assert (
            pred_proba_sql[0]
            == "(CASE WHEN POWER(POWER(col1 - 0.5, 2) + POWER(col2 - 0.6, 2), 1 / 2) = 0 THEN 1.0 ELSE 1 / (POWER(POWER(col1 - 0.5, 2) + POWER(col2 - 0.6, 2), 1 / 2)) / (1 / (POWER(POWER(col1 - 0.5, 2) + POWER(col2 - 0.6, 2), 1 / 2)) + 1 / (POWER(POWER(col1 - 1.0, 2) + POWER(col2 - 2.0, 2), 1 / 2)) + 1 / (POWER(POWER(col1 - 100.0, 2) + POWER(col2 - 200.0, 2), 1 / 2))) END)"
        )
        assert (
            pred_proba_sql[1]
            == "(CASE WHEN POWER(POWER(col1 - 1.0, 2) + POWER(col2 - 2.0, 2), 1 / 2) = 0 THEN 1.0 ELSE 1 / (POWER(POWER(col1 - 1.0, 2) + POWER(col2 - 2.0, 2), 1 / 2)) / (1 / (POWER(POWER(col1 - 0.5, 2) + POWER(col2 - 0.6, 2), 1 / 2)) + 1 / (POWER(POWER(col1 - 1.0, 2) + POWER(col2 - 2.0, 2), 1 / 2)) + 1 / (POWER(POWER(col1 - 100.0, 2) + POWER(col2 - 200.0, 2), 1 / 2))) END)"
        )
        assert (
            pred_proba_sql[2]
            == "(CASE WHEN POWER(POWER(col1 - 100.0, 2) + POWER(col2 - 200.0, 2), 1 / 2) = 0 THEN 1.0 ELSE 1 / (POWER(POWER(col1 - 100.0, 2) + POWER(col2 - 200.0, 2), 1 / 2)) / (1 / (POWER(POWER(col1 - 0.5, 2) + POWER(col2 - 0.6, 2), 1 / 2)) + 1 / (POWER(POWER(col1 - 1.0, 2) + POWER(col2 - 2.0, 2), 1 / 2)) + 1 / (POWER(POWER(col1 - 100.0, 2) + POWER(col2 - 200.0, 2), 1 / 2))) END)"
        )

    def test_transform_sql(self, memmodel_nc):
        """
        test function - transform_sql
        """
        cnames = ["col1", "col2"]
        transform_sql = memmodel_nc.transform_sql(cnames)
        assert (
            transform_sql[0]
            == "POWER(POWER(col1 - 0.5, 2) + POWER(col2 - 0.6, 2), 1 / 2)"
        )
        assert (
            transform_sql[1]
            == "POWER(POWER(col1 - 1.0, 2) + POWER(col2 - 2.0, 2), 1 / 2)"
        )
        assert (
            transform_sql[2]
            == "POWER(POWER(col1 - 100.0, 2) + POWER(col2 - 200.0, 2), 1 / 2)"
        )

    def test_get_attributes(self, memmodel_nc):
        """
        test function - get_attributes
        """
        attributes = memmodel_nc.get_attributes()
        assert attributes["clusters"][0][0] == 0.5
        assert attributes["clusters"][0][1] == 0.6
        assert attributes["clusters"][1][0] == 1
        assert attributes["clusters"][1][1] == 2
        assert attributes["p"] == 2

    def test_object_type(self, memmodel_nc):
        """
        test function - object_type
        """
        assert memmodel_nc.object_type == "NearestCentroid"
