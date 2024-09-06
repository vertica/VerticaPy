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
from scipy.stats import median_abs_deviation
import pytest


class TestScaler:
    """
    test class for Scale function test
    """

    @pytest.mark.parametrize(
        "columns, method",
        [("age", "zscore"), ("age", "robust_zscore"), ("age", "minmax")],
    )
    def test_scale_vdf(self, titanic_vd, columns, method):
        """
        test function - scaling for vDataframe
        """
        titanic_pdf = titanic_vd.to_pandas()
        titanic_pdf[columns] = titanic_pdf[columns].astype(float)

        py_data = titanic_pdf[[columns]]
        if method == "zscore":
            vpy_res = titanic_vd.scale(columns=[columns], method=method)[columns].std()
            py_res = ((py_data - py_data.mean()) / py_data.std()).std()
        elif method == "robust_zscore":
            vpy_res = titanic_vd.scale(columns=[columns], method=method)[columns].std()
            py_res = (
                (py_data - py_data.median())
                / (1.4826 * median_abs_deviation(py_data, nan_policy="omit"))
            ).std()
        else:
            vpy_res = titanic_vd.scale(columns=[columns], method=method)[columns].mean()
            py_res = (
                (py_data - py_data.min()) / (py_data.max() - py_data.min())
            ).mean()

        print(
            f"method name: {method} \nVerticaPy Result: {vpy_res} \nPython Result :{py_res}\n"
        )

        assert vpy_res == pytest.approx(py_res)

    @pytest.mark.parametrize(
        "columns, method",
        [("age", "zscore"), ("age", "robust_zscore"), ("age", "minmax")],
    )
    @pytest.mark.parametrize("partition_by", ["pclass", None])
    def test_scale_vcolumn(self, titanic_vd_fun, partition_by, columns, method):
        """
        test function - scaling for vColumns
        """
        titanic_pdf = titanic_vd_fun.to_pandas()
        titanic_pdf[columns] = titanic_pdf[columns].astype(float)
        py_data = titanic_pdf[[columns]]
        vpy_data = titanic_vd_fun[columns]

        if method == "zscore":
            vpy_res = vpy_data.scale(method=method, by=partition_by)[columns].std()
            py_res = ((py_data - py_data.mean()) / py_data.std()).std()
        elif method == "robust_zscore":
            vpy_res = vpy_data.scale(method=method)[columns].std()
            # vpy_res = _res.std() if partition_by else _res["age"].std()
            py_res = (
                (py_data - py_data.median())
                / (1.4826 * median_abs_deviation(py_data, nan_policy="omit"))
            ).std()
        else:
            vpy_res = vpy_data.scale(method=method, by=partition_by)[columns].mean()
            py_res = (
                (py_data - py_data.min()) / (py_data.max() - py_data.min())
            ).mean()

        print(
            f"method name: {method} \nVerticaPy Result: {vpy_res} \nPython Result :{py_res}\n"
        )

        assert vpy_res == pytest.approx(py_res, rel=1e-00, abs=1e-02)
