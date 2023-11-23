"""
(c)  Copyright  [2018-2024]  OpenText  or one of its
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
from itertools import chain
import pytest
import statsmodels.api as sm
import numpy as np
import pandas as pd
from scipy.stats import (
    pearsonr,
    kendalltau,
    spearmanr,
    pointbiserialr,
    chi2_contingency,
    chi2,
)
import verticapy as vp
from verticapy.tests_new.core.vdataframe import REL_TOLERANCE, ABS_TOLERANCE


class TestCorrelation:
    """
    test class for correlation test
    """

    @staticmethod
    def calculate_cramer_corr(col1, col2):
        """
        test function - calculate_py_corr
        """
        _cm = pd.crosstab(col1, col2).to_numpy()
        _chi2 = chi2_contingency(_cm)[0]
        _n = _cm.sum()
        _r, _k = _cm.shape
        coeff = np.sqrt((_chi2 / _n) / min(_r - 1, _k - 1))
        test_stats = coeff * coeff * _n * min(_k, _r)  # need to check on this
        pvalue = chi2.sf(test_stats, (_k - 1) * (_r - 1))

        return coeff, pvalue

    @pytest.mark.parametrize(
        "column, ts, by, p, unit, method, confidence, alpha, show, kind, mround, chart",
        [
            (
                "number",
                "date",
                None,
                5,
                "rows",
                "pearson",
                True,
                0.95,
                False,
                "bar",
                3,
                None,
            ),
            (
                "number",
                "date",
                None,
                4,
                "rows",
                "pearson",
                True,
                0.45,
                False,
                "bar",
                3,
                None,
            ),
        ],
    )
    def test_acf(
        self,
        amazon_vd,
        column,
        ts,
        by,
        p,
        unit,
        method,
        confidence,
        alpha,
        show,
        kind,
        mround,
        chart,
    ):
        """
        test function - test_acf
        """
        amazon_pdf = amazon_vd.to_pandas()

        vpy_res = amazon_vd.acf(
            column=column,
            ts=ts,
            by=by,
            p=p,
            unit=unit,
            method=method,
            confidence=confidence,
            alpha=alpha,
            show=show,
            kind=kind,
            mround=mround,
            chart=chart,
        )

        vpy_acf, _ = vpy_res["value"], vpy_res["confidence"]
        py_acf, _ = sm.tsa.acf(amazon_pdf["number"], alpha=alpha, nlags=p)

        print(f"VerticaPy ACF Result: {vpy_acf} \nPython ACF Result :{py_acf}\n")

        assert vpy_acf == pytest.approx(py_acf, rel=REL_TOLERANCE, abs=1e-01)

    @pytest.mark.parametrize(
        "column, ts, by, p, unit, method, confidence, alpha, show, kind, chart",
        [
            (
                "number",
                "date",
                None,
                5,
                "rows",
                "pearson",
                True,
                0.95,
                False,
                "bar",
                None,
            ),
            (
                "number",
                "date",
                None,
                2,
                "rows",
                "pearson",
                True,
                0.55,
                False,
                "bar",
                None,
            ),
        ],
    )
    def test_pacf(
        self,
        amazon_vd,
        column,
        ts,
        by,
        p,
        unit,
        method,
        confidence,
        alpha,
        show,
        kind,
        chart,
    ):
        """
        test function - pacf
        """
        amazon_pdf = amazon_vd.to_pandas()

        vpy_res = amazon_vd.pacf(
            column=column,
            ts=ts,
            by=by,
            p=p,
            unit=unit,
            method=method,
            confidence=confidence,
            alpha=alpha,
            show=show,
            kind=kind,
            chart=chart,
        )
        vpy_pacf, _ = vpy_res["value"], vpy_res["confidence"]
        py_pacf, _ = sm.tsa.pacf(
            amazon_pdf["number"], alpha=alpha, nlags=p, method="yw"
        )

        print(f"VerticaPy PACF Result: {vpy_pacf} \nPython PACF Result :{py_pacf}\n")

        assert vpy_pacf == pytest.approx(py_pacf, rel=REL_TOLERANCE, abs=1e-01)

    @pytest.mark.parametrize(
        "columns, focus", [(None, None), (["pclass", "survived"], "survived")]
    )
    def test_cov(self, titanic_vd, columns, focus):
        """
        test function - covariance
        """
        titanic_pdf = titanic_vd.to_pandas()
        titanic_pdf["age"] = titanic_pdf["age"].astype(float)
        titanic_pdf["fare"] = titanic_pdf["fare"].astype(float)

        _vpy_res = (
            titanic_vd.cov(columns=columns, focus=focus, show=False)
            .transpose()
            .to_list()
        )

        if columns is None and focus is None:
            vpy_res = list(chain(*_vpy_res))

            _py_res = titanic_pdf.cov(numeric_only=True)
            py_res = list(chain(*_py_res.values))
        else:
            vpy_res = _vpy_res[0]
            py_res = [
                titanic_pdf[columns[1]].cov(other=titanic_pdf[focus]),
                titanic_pdf[columns[0]].cov(other=titanic_pdf[focus]),
            ]

        print(
            f"column name: {columns} \nVerticaPy Result: {vpy_res} \nPython Result :{py_res}\n"
        )
        assert vpy_res == pytest.approx(py_res, rel=1e-2)

    @pytest.mark.parametrize(
        "columns, focus",
        (
            (["pclass", "survived"], None),
            (["pclass", "survived"], "survived"),
        ),
    )
    @pytest.mark.parametrize(
        "method, py_func, _rel_tol, _abs_tol",
        [
            ("pearson", "pearsonr", REL_TOLERANCE, ABS_TOLERANCE),
            ("kendall", "kendalltau", REL_TOLERANCE, ABS_TOLERANCE),
            ("spearman", "spearmanr", REL_TOLERANCE, 1e-02),
            (
                "biserial",
                "pointbiserialr",
                REL_TOLERANCE,
                1e-03,
            ),  # getting nan in verticapy for float columns
            ("cramer", "cramer", REL_TOLERANCE, 1e-02),
        ],
    )
    def test_corr(
        self, titanic_vd, columns, method, focus, py_func, _rel_tol, _abs_tol
    ):
        """
        test function - correlation
        """
        vpy_res = titanic_vd.corr(
            columns=columns, method=method, focus=focus, show=False
        )

        if focus:
            vpy_res = vpy_res.transpose().to_list()[0][-1]

        # python
        titanic_pdf = titanic_vd.to_pandas()
        if method == "cramer":
            py_coeff, _ = self.calculate_cramer_corr(
                titanic_pdf[columns[0]], titanic_pdf[columns[1]]
            )
            py_res = py_coeff
        else:
            py_res = eval(
                f"{py_func}(titanic_pdf['{columns[0]}'], titanic_pdf['{columns[1]}']).statistic"
            )

        print(
            f"column name: {columns} \nVerticaPy Result: {vpy_res} \nPython Result :{py_res}\n"
        )
        assert vpy_res == pytest.approx(py_res, rel=_rel_tol, abs=_abs_tol)

    @pytest.mark.parametrize(
        "method, py_func, _rel_tol, _abs_tol",
        [
            ("pearson", "pearsonr", REL_TOLERANCE, ABS_TOLERANCE),
            ("kendall", "kendalltau", REL_TOLERANCE, 1e-8),
            ("spearman", "spearmanr", REL_TOLERANCE, 1e-00),
            (
                "biserial",
                "pointbiserialr",
                REL_TOLERANCE,
                ABS_TOLERANCE,
            ),  # getting nan in verticapy for float
            ("cramer", "cramer", REL_TOLERANCE, ABS_TOLERANCE),
        ],
    )
    def test_corr_pvalue(self, titanic_vd, method, py_func, _rel_tol, _abs_tol):
        """
        test function - correlation pvale
        """
        focus_columns = ["pclass", "survived"]
        titanic_pdf = titanic_vd.to_pandas().dropna(subset=focus_columns)

        vpy_res = titanic_vd.corr_pvalue(
            focus_columns[0], focus_columns[1], method=method
        )[1]

        if method == "cramer":
            _, py_pvalue = self.calculate_cramer_corr(
                titanic_pdf[focus_columns[0]],
                titanic_pdf[focus_columns[1]],
            )
            py_res = py_pvalue
        else:
            py_res = eval(
                f"{py_func}(titanic_pdf[focus_columns[0]], titanic_pdf[focus_columns[1]]).pvalue"
            )

        print(
            f"Correlation method name: {py_func} \nVerticaPy Result: {vpy_res} \nPython Result :{py_res}\n"
        )
        assert vpy_res == pytest.approx(py_res, rel=_rel_tol, abs=_abs_tol)

    @pytest.mark.parametrize(
        "method, expected",
        [
            (
                "avgx",
                [
                    [30.1524573721163, 30.1524573721163],
                    [2.20361083249749, 2.28444084278768],
                ],
            ),
            (
                "avgy",
                [
                    [30.1524573721163, 2.20361083249749],
                    [30.1524573721163, 2.28444084278768],
                ],
            ),
            ("count", [[997.0, 997.0], [997.0, 1234.0]]),
            ("r2", [[1.0, 0.160663600528957], [0.160663600528957, 1.0]]),
            ("beta", [[1.0, -0.0234385332617575], [-6.8546780950281, 1.0]]),
            (
                "sxx",
                [
                    [207544.507679438, 207544.507679438],
                    [709.667001003009, 875.161264181524],
                ],
            ),
            (
                "sxy",
                [
                    [207544.507679438, -4864.53884653961],
                    [-4864.53884653961, 875.161264181524],
                ],
            ),
            (
                "syy",
                [
                    [207544.507679438, 709.667001003009],
                    [207544.507679438, 875.161264181524],
                ],
            ),
        ],
    )
    @pytest.mark.parametrize(
        "columns",
        [
            ["age", "pclass"],
            # None,
        ],
    )
    def test_regr(self, titanic_vd, columns, method, expected):
        """
        test function - regression metrics
        """
        if columns:
            vpy_res = (
                titanic_vd.regr(columns=columns, method=method, show=False)
                .transpose()
                .to_list()
            )
            assert vpy_res[0] == pytest.approx(expected[0])
            assert vpy_res[1] == pytest.approx(expected[1])
        else:
            vpy_res = titanic_vd.regr(method=method, show=False)

        print(f"Correlation method name: {method} \nVerticaPy Result: {vpy_res} \n")

    @pytest.mark.parametrize(
        "input_type, columns",
        [
            ("vDataFrame", []),
            ("vDataFrame_column", ["sex", "pclass"]),
            ("vcolumn", ["sex"]),
        ],
    )
    def test_iv_woe(self, titanic_vd, input_type, columns):
        """
        test function - iv_woe
        """
        target = "survived"
        focus_columns = ["sex", "pclass", "parch", "embarked", "sibsp", "name"]

        # verticapy
        if input_type == "vDataFrame":
            res = titanic_vd.iv_woe(y=target, show=False).values
            vpy_res_map = dict(zip(res["index"], res["iv"]))
            vpy_res = [
                v for k, v in vpy_res_map.items() if k.replace('"', "") in focus_columns
            ]
        elif input_type == "vDataFrame_column":
            vpy_res = titanic_vd.iv_woe(columns=columns, y=target, show=False)["iv"]
        else:
            vpy_res = titanic_vd[columns[0]].iv_woe(y=target)["iv"][-1]

        # python
        py_res = []
        titanic_pdf = titanic_vd.to_pandas()

        for column in focus_columns if input_type == "vDataFrame" else columns:
            freq_data = pd.crosstab(
                titanic_pdf[column], titanic_pdf[target], normalize="columns"
            )
            _woe = np.log(
                freq_data[1].replace(0, np.nan) / freq_data[0].replace(0, np.nan)
            )
            _iv = np.sum((freq_data[1] - freq_data[0]) * _woe)
            py_res.append(_iv)

        py_res = py_res[0] if input_type == "vcolumn" else py_res

        print(f"VerticaPy Result: {vpy_res} \nPython Result :{py_res}\n")
        assert vpy_res == pytest.approx(py_res, abs=1e-03, rel=1e-03)
