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
from verticapy.machine_learning.memmodel.preprocessing import StandardScaler
from verticapy.machine_learning.memmodel.preprocessing import MinMaxScaler
from verticapy.machine_learning.memmodel.preprocessing import OneHotEncoder


@pytest.fixture(scope="module")
def memmodel_sts():
    mean = [0.4, 0.1]
    std = [0.5, 0.2]
    memmodel_sts = StandardScaler(mean, std)
    yield memmodel_sts


@pytest.fixture(scope="module")
def memmodel_mms():
    min = [0.4, 0.1]
    max = [0.5, 0.2]
    memmodel_mms = MinMaxScaler(min, max)
    yield memmodel_mms


@pytest.fixture(scope="module")
def memmodel_ohe():
    memmodel_ohe = OneHotEncoder(
        categories=[["male", "female"], [1, 2, 3]],
        drop_first=False,
        column_naming=None,
    )
    yield memmodel_ohe


class TestStandardScaler:
    """
    test class - TestStandardScaler
    """

    @pytest.mark.parametrize(
        "col1, col2, expected_output",
        [
            (0.45, 0.17, [0.1, 0.35]),
            (2.01, -0.8, [3.22, -4.5]),
        ],
    )
    def test_transform(self, memmodel_sts, col1, col2, expected_output):
        """
        test function - transform
        """
        data = [
            [col1, col2],
        ]
        prediction = memmodel_sts.transform(data)
        assert prediction[0][0] == pytest.approx(expected_output[0])
        assert prediction[0][1] == pytest.approx(expected_output[1])

    def test_transform_sql(self, memmodel_sts):
        """
        test function - transform_sql
        """
        cnames = ["col1", "col2"]
        pred_sql = memmodel_sts.transform_sql(cnames)
        assert pred_sql[0] == "(col1 - 0.4) / 0.5"
        assert pred_sql[1] == "(col2 - 0.1) / 0.2"

    def test_get_attributes(self, memmodel_sts):
        """
        test function - get_attributes
        """
        attributes = memmodel_sts.get_attributes()
        assert attributes["sub"][0] == 0.4
        assert attributes["sub"][1] == 0.1
        assert attributes["den"][0] == 0.5
        assert attributes["den"][1] == 0.2

    def test_object_type(self, memmodel_sts):
        """
        test function - object_type
        """
        assert memmodel_sts.object_type == "StandardScaler"


class TestMinMaxScaler:
    """
    test class - TestMinMaxScaler
    """

    @pytest.mark.parametrize(
        "col1, col2, expected_output",
        [
            (0.45, 0.17, [0.5, 0.7]),
            (2.01, -0.8, [16.1, -9]),
        ],
    )
    def test_transform(self, memmodel_mms, col1, col2, expected_output):
        """
        test function - transform
        """
        data = [
            [col1, col2],
        ]
        prediction = memmodel_mms.transform(data)
        assert prediction[0][0] == pytest.approx(expected_output[0])
        assert prediction[0][1] == pytest.approx(expected_output[1])

    def test_transform_sql(self, memmodel_mms):
        """
        test function - transform_sql
        """
        cnames = ["col1", "col2"]
        pred_sql = memmodel_mms.transform_sql(cnames)
        assert pred_sql[0] == "(col1 - 0.4) / 0.09999999999999998"
        assert pred_sql[1] == "(col2 - 0.1) / 0.1"

    def test_get_attributes(self, memmodel_mms):
        """
        test function - get_attributes
        """
        attributes = memmodel_mms.get_attributes()
        assert attributes["sub"][0] == pytest.approx(0.4)
        assert attributes["sub"][1] == pytest.approx(0.1)
        assert attributes["den"][0] == pytest.approx(0.1)
        assert attributes["den"][1] == pytest.approx(0.1)

    def test_object_type(self, memmodel_mms):
        """
        test function - object_type
        """
        assert memmodel_mms.object_type == "MinMaxScaler"


class TestOneHotEncoder:
    """
    test class - TestOneHotEncoder
    """

    @pytest.mark.parametrize(
        "sex, pclass, expected_output",
        [
            ("male", 1, [1, 0, 1, 0, 0]),
            ("female", 3, [0, 1, 0, 0, 1]),
        ],
    )
    def test_transform(self, memmodel_ohe, sex, pclass, expected_output):
        """
        test function - transform
        """
        data = [
            [sex, pclass],
        ]
        prediction = memmodel_ohe.transform(data)
        assert prediction[0][0] == expected_output[0]
        assert prediction[0][1] == expected_output[1]
        assert prediction[0][2] == expected_output[2]
        assert prediction[0][3] == expected_output[3]
        assert prediction[0][4] == expected_output[4]

    def test_transform_sql(self, memmodel_ohe):
        """
        test function - transform_sql
        """
        cnames = ["sex", "pclass"]
        pred_sql = memmodel_ohe.transform_sql(cnames)
        assert pred_sql[0][0] == "(CASE WHEN sex = 'male' THEN 1 ELSE 0 END)"
        assert pred_sql[0][1] == "(CASE WHEN sex = 'female' THEN 1 ELSE 0 END)"
        assert pred_sql[1][0] == "(CASE WHEN pclass = 1 THEN 1 ELSE 0 END)"
        assert pred_sql[1][1] == "(CASE WHEN pclass = 2 THEN 1 ELSE 0 END)"
        assert pred_sql[1][2] == "(CASE WHEN pclass = 3 THEN 1 ELSE 0 END)"

    def test_get_attributes(self, memmodel_ohe):
        """
        test function - get_attributes
        """
        attributes = memmodel_ohe.get_attributes()
        assert attributes["categories"][0][0] == "male"
        assert attributes["categories"][0][1] == "female"
        assert attributes["categories"][1][0] == 1
        assert attributes["categories"][1][1] == 2
        assert attributes["categories"][1][2] == 3
        assert attributes["column_naming"] == None
        assert attributes["drop_first"] == False

    def test_col_naming_indices(self, memmodel_ohe):
        """
        test function - test_col_naming_indices
        """
        memmodel_ohe.set_attributes(**{"column_naming": "indices"})
        pred_sql = memmodel_ohe.transform_sql(["sex", "pclass"])
        assert (
            pred_sql[0][0] == "(CASE WHEN sex = 'male' THEN 1 ELSE 0 END) AS \"sex_0\""
        )

    def test_col_naming_values(self, memmodel_ohe):
        """
        test function - test_col_naming_values
        """
        memmodel_ohe.set_attributes(**{"column_naming": "values"})
        pred_sql = memmodel_ohe.transform_sql(["sex", "pclass"])
        assert (
            pred_sql[0][0]
            == "(CASE WHEN sex = 'male' THEN 1 ELSE 0 END) AS \"sex_male\""
        )
        assert (
            pred_sql[1][1] == '(CASE WHEN pclass = 2 THEN 1 ELSE 0 END) AS "pclass_2"'
        )

    def test_object_type(self, memmodel_ohe):
        """
        test function - object_type
        """
        assert memmodel_ohe.object_type == "OneHotEncoder"
