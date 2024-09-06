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
from verticapy.machine_learning.memmodel.decomposition import PCA
from verticapy.machine_learning.memmodel.decomposition import SVD


@pytest.fixture(scope="module")
def memmodel_pca():
    principal_components = [
        [0.4, 0.5],
        [0.3, 0.2],
    ]
    mean = [0.1, 0.3]
    memmodel_pca = PCA(principal_components, mean)
    yield memmodel_pca


@pytest.fixture(scope="module")
def memmodel_svd():
    vectors = [
        [0.4, 0.5],
        [0.3, 0.2],
    ]
    values = [0.1, 0.3]
    memmodel_svd = SVD(vectors, values)
    yield memmodel_svd


class TestPCA:
    """
    test class - TestPCA
    """

    @pytest.mark.parametrize(
        "col1, col2, expected_output",
        [
            (4, 5, [2.97, 2.89]),
            (3, 2, [1.67, 1.79]),
        ],
    )
    def test_transform(self, memmodel_pca, col1, col2, expected_output):
        """
        test function - transform
        """
        data = [
            [col1, col2],
        ]
        transformation = memmodel_pca.transform(data)
        assert transformation[0][0] == pytest.approx(expected_output[0])
        assert transformation[0][1] == pytest.approx(expected_output[1])

    def test_transform_sql(self, memmodel_pca):
        """
        test function - transform_sql
        """
        cnames = ["col1", "col2"]
        trf_sql = memmodel_pca.transform_sql(cnames)
        assert trf_sql[0] == "(col1 - 0.1) * 0.4 + (col2 - 0.3) * 0.3"
        assert trf_sql[1] == "(col1 - 0.1) * 0.5 + (col2 - 0.3) * 0.2"

    def test_get_attributes(self, memmodel_pca):
        """
        test function - get_attributes
        """
        attributes = memmodel_pca.get_attributes()
        assert attributes["principal_components"][0][0] == 0.4
        assert attributes["principal_components"][0][1] == 0.5
        assert attributes["principal_components"][1][0] == 0.3
        assert attributes["principal_components"][1][1] == 0.2
        assert attributes["mean"][0] == 0.1
        assert attributes["mean"][1] == 0.3

    @pytest.mark.parametrize(
        "gamma, expected_output",
        [
            (
                1.0,
                {
                    "principal_components": [
                        [0.07739603, 0.63561769],
                        [0.15004967, 0.3278492],
                    ],
                    "mean": [0.1, 0.3],
                },
            ),
            (
                0.0,
                {
                    "principal_components": [
                        [-0.01464650, 0.64014489],
                        [0.10143393, 0.345993],
                    ],
                    "mean": [0.1, 0.3],
                },
            ),
        ],
    )
    def test_rotate(self, memmodel_pca, gamma, expected_output):
        """
        test function - test_col_naming_indices
        """
        memmodel_pca.rotate(gamma)
        attributes = memmodel_pca.get_attributes()
        assert attributes["principal_components"][0][0] == pytest.approx(
            expected_output["principal_components"][0][0]
        )
        assert attributes["principal_components"][0][1] == pytest.approx(
            expected_output["principal_components"][0][1]
        )
        assert attributes["principal_components"][1][0] == pytest.approx(
            expected_output["principal_components"][1][0]
        )
        assert attributes["principal_components"][1][1] == pytest.approx(
            expected_output["principal_components"][1][1]
        )
        assert attributes["mean"][0] == pytest.approx(expected_output["mean"][0])
        assert attributes["mean"][1] == pytest.approx(expected_output["mean"][1])

    def test_object_type(self, memmodel_pca):
        """
        test function - object_type
        """
        assert memmodel_pca.object_type == "PCA"


class TestSVD:
    """
    test class - TestSVD
    """

    @pytest.mark.parametrize(
        "col1, col2, expected_output",
        [
            (0.3, 0.5, [2.7, 0.83333333]),
            (3, 2, [18, 6.33333333]),
        ],
    )
    def test_transform(self, memmodel_svd, col1, col2, expected_output):
        """
        test function - transform
        """
        data = [
            [col1, col2],
        ]
        transformation = memmodel_svd.transform(data)
        assert transformation[0][0] == pytest.approx(expected_output[0])
        assert transformation[0][1] == pytest.approx(expected_output[1])

    def test_transform_sql(self, memmodel_svd):
        """
        test function - transform_sql
        """
        cnames = ["col1", "col2"]
        trf_sql = memmodel_svd.transform_sql(cnames)
        assert trf_sql[0] == "col1 * 0.4 / 0.1 + col2 * 0.3 / 0.1"
        assert trf_sql[1] == "col1 * 0.5 / 0.3 + col2 * 0.2 / 0.3"

    def test_get_attributes(self, memmodel_svd):
        """
        test function - get_attributes
        """
        attributes = memmodel_svd.get_attributes()
        assert attributes["vectors"][0][0] == 0.4
        assert attributes["vectors"][0][1] == 0.5
        assert attributes["vectors"][1][0] == 0.3
        assert attributes["vectors"][1][1] == 0.2
        assert attributes["values"][0] == 0.1
        assert attributes["values"][1] == 0.3

    def test_object_type(self, memmodel_svd):
        """
        test function - object_type
        """
        assert memmodel_svd.object_type == "SVD"
