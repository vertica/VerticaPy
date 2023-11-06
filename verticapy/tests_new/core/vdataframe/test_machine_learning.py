"""
(c)  Copyright  [2018-2023]  OpenText  or one of its
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
import math
from itertools import chain
from sklearn import model_selection
import pytest
from verticapy.utilities import drop, TableSample


class TestMachineLearning:
    """
    test class for Machine Learning functions test
    """

    @pytest.mark.parametrize("use_gcd", [True, False])
    def test_add_duplicates(self, use_gcd):
        """
        test function - add_duplicates
        """
        cities = TableSample(
            {
                "cities": ["Boston", "New York City", "San Francisco"],
                "weight": [2, 4, 6],
            }
        ).to_vdf()

        result = (
            cities.add_duplicates("weight", use_gcd=use_gcd)
            .groupby("cities", "COUNT(*) AS cnt")
            .sort("cnt")
        )

        if use_gcd:
            cities["weight"] = cities["weight"] / math.gcd(
                *list(chain(*cities[["weight"]].to_list()))
            )

        assert cities.to_list() == result.to_list()

    # @pytest.mark.parametrize("columns, tcdt", [("age", "fare", "pclass", "boat", True), (None, False)])
    def test_cdt(self, titanic_vd_fun, columns, tcdt):
        """
        test function - cdt - complete disjunctive table
        """
        pass

    # @pytest.mark.parametrize("columns, tcdt", [("age", "fare", "pclass", "boat", True), (None, False)])
    def test_chaid(self, titanic_vd_fun, columns, tcdt):
        """
        test function - chaid
        """
        pass

    # @pytest.mark.parametrize("columns, tcdt", [("age", "fare", "pclass", "boat", True), (None, False)])
    def test_chaid_columns(self, titanic_vd_fun, columns, tcdt):
        """
        test function - chaid_columns
        """
        pass

    # @pytest.mark.parametrize("columns, tcdt", [("age", "fare", "pclass", "boat", True), (None, False)])
    def test_outliers(self, titanic_vd_fun, columns, tcdt):
        """
        test function - outliers
        """
        pass

    # @pytest.mark.parametrize("columns, tcdt", [("age", "fare", "pclass", "boat", True), (None, False)])
    def test_pivot_table_chi2(self, titanic_vd_fun, columns, tcdt):
        """
        test function - pivot_table_chi2
        """
        pass

    # @pytest.mark.parametrize("columns, tcdt", [("age", "fare", "pclass", "boat", True), (None, False)])
    def test_polynomial_comb(self, titanic_vd_fun, columns, tcdt):
        """
        test function - polynomial_comb
        """
        pass

    # @pytest.mark.parametrize("columns, tcdt", [("age", "fare", "pclass", "boat", True), (None, False)])
    def test_recommend(self, titanic_vd_fun, columns, tcdt):
        """
        test function - recommend
        """
        pass

    # @pytest.mark.parametrize("columns, tcdt", [("age", "fare", "pclass", "boat", True), (None, False)])
    def test_score(self, titanic_vd_fun, columns, tcdt):
        """
        test function - score
        """
        pass

    # @pytest.mark.parametrize("columns, tcdt", [("age", "fare", "pclass", "boat", True), (None, False)])
    def test_sessionize(self, titanic_vd_fun, columns, tcdt):
        """
        test function - sessionize
        """
        pass

    def test_train_test_split(self, titanic_vd_fun):
        """
        test function - train_test_split
        """
        titanic_pdf = titanic_vd_fun.to_pandas()
        vpy_train, vpy_test = titanic_vd_fun.train_test_split(
            test_size=0.25, order_by={"name": "asc"}, random_state=1
        )

        py_train, py_test = model_selection.train_test_split(
            titanic_pdf, test_size=0.25, random_state=1
        )

        assert len(vpy_train) == len(py_train) and len(vpy_test) == len(py_test)
