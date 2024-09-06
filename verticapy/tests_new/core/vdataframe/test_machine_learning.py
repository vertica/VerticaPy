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
import pytest
import pandas as pd
from sklearn import model_selection
from scipy.stats import median_abs_deviation, chi2_contingency
from verticapy.utilities import TableSample
from verticapy import errors
from verticapy.tests_new.machine_learning.vertica.conftest import *


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

    @pytest.mark.parametrize(
        "columns, max_cardinality, nbins, tcdt, drop_transf_cols",
        [
            ("age", 20, 15, False, True),
            ("age", 20, 10, False, False),
            ("age", 25, 10, False, False),
            ("age", 20, 10, True, False),
        ],
    )
    def test_cdt(
        self, titanic_vd_fun, columns, max_cardinality, nbins, tcdt, drop_transf_cols
    ):
        """
        test function - cdt - complete disjunctive table
        """
        titanic_pdf = titanic_vd_fun.to_pandas()
        titanic_pdf[columns] = titanic_pdf[columns].astype("float")
        titanic_vd_fun_copy = titanic_vd_fun.copy()

        def _get_columns(df):
            col_names = [
                col.replace('"', "")
                for col in df.get_columns()
                if col.replace('"', "").startswith("age_")
            ]

            return col_names

        vpy_raw = titanic_vd_fun.cdt(
            columns=columns, drop_transf_cols=drop_transf_cols, tcdt=tcdt
        )
        vpy_res = vpy_raw.aggregate(
            columns=_get_columns(vpy_raw), func=["sum"]
        ).to_list()

        vpy_cdt_raw = (
            titanic_vd_fun[columns]
            .discretize(nbins=nbins)
            .one_hot_encode(
                columns,
                max_cardinality=max(max_cardinality, nbins) + 2,
                drop_first=False,
            )
        )
        if tcdt:
            for elem in _get_columns(vpy_cdt_raw):
                sum_cat = vpy_cdt_raw[elem].sum()
                vpy_cdt_raw[elem].apply(f"{{}} / {sum_cat} - 1")
        vpy_cdt_res = vpy_cdt_raw.aggregate(
            columns=_get_columns(vpy_cdt_raw), func=["sum"]
        ).to_list()

        assert (
            columns not in _get_columns(vpy_raw)
            if drop_transf_cols
            else vpy_res == vpy_cdt_res
        )

        # as there is no way to control h(interval  size) value in cdt function to compare result with pandas.
        # so first comparing results of vpy cdt with cdt(using discretize and one_hot_encode).

        h = 10
        vpy_cdt_h = (
            titanic_vd_fun_copy[columns]
            .discretize(h=h, nbins=nbins)
            .one_hot_encode(
                columns,
                max_cardinality=max(max_cardinality, nbins) + 2,
                drop_first=False,
            )
        )
        vpy_cdt_h_res = vpy_cdt_h.aggregate(
            columns=_get_columns(vpy_cdt_h), func=["sum"]
        ).to_list()

        # python
        bins = range(0, int(max(titanic_pdf[columns])) + 20, h)
        titanic_pdf[f"{columns}_cut"] = pd.cut(
            titanic_pdf[columns],
            bins=bins,
            right=False,
        )
        py_raw = pd.get_dummies(
            titanic_pdf, columns=[f"{columns}_cut"], drop_first=False
        )

        py_col_names = [col for col in list(py_raw.columns) if col.startswith("age_")]
        py_res = list(py_raw[py_col_names].sum().values)

        assert list(chain(*vpy_cdt_h_res)) == py_res

    @pytest.mark.parametrize(
        "response, columns, nbins, method, RFmodel_params",
        [
            ("survived", ["sex", "pclass"], 16, "same_width", None),
            ("survived", ["sex", "pclass"], 10, "smart", None),
        ],
    )
    def test_chaid(
        self, titanic_vd_fun, response, columns, nbins, method, RFmodel_params
    ):
        """
        test function - chaid
        """
        titanic_pdf = titanic_vd_fun.to_pandas()
        vpy_tree = titanic_vd_fun.chaid(
            response=response, columns=columns, nbins=nbins, method=method
        ).tree_
        vpy_col2_cats = vpy_tree["children"]["female"]["children"].keys()

        for col1_cat in titanic_pdf[columns[0]].unique():
            for col2_cat, vpy_col2_cat in zip(
                titanic_pdf[columns[1]].unique(), vpy_col2_cats
            ):
                subset_pdf = titanic_pdf.loc[titanic_pdf[columns[0]].isin([col1_cat])]

                sub_grp_pdf1 = subset_pdf.groupby([columns[1], response])[
                    columns[1]
                ].count()
                sub_grp_pdf2 = subset_pdf.groupby([columns[1]])[[columns[1]]].count()
                merge_pdf = pd.merge(
                    sub_grp_pdf1, sub_grp_pdf2, left_index=True, right_index=True
                )
                merge_pdf["prob"] = merge_pdf["pclass_x"] / merge_pdf["pclass_y"]

                contingency_table = pd.crosstab(
                    subset_pdf[response], subset_pdf[columns[1]]
                ).to_numpy()
                py_chi2, py_p_val, py_dof, _ = chi2_contingency(contingency_table)

                vpy_chi2_res = vpy_tree["children"][col1_cat]["chi2"]
                py_chi2_res = py_chi2

                vpy_prob_res = vpy_tree["children"][col1_cat]["children"][vpy_col2_cat][
                    "prediction"
                ]
                py_prob_res = merge_pdf[
                    merge_pdf.index.get_level_values(0) == col2_cat
                ]["prob"].tolist()

                assert vpy_chi2_res == pytest.approx(
                    py_chi2_res
                ) and vpy_prob_res == pytest.approx(py_prob_res)

    @pytest.mark.parametrize(
        "columns, max_cardinality, expected",
        [
            (["age", "pclass"], 20, ["age", "pclass"]),
            (
                None,
                None,
                [
                    "pclass",
                    "survived",
                    "sex",
                    "age",
                    "sibsp",
                    "parch",
                    "fare",
                    "embarked",
                    "body",
                ],
            ),
        ],
    )
    def test_chaid_columns(self, titanic_vd_fun, columns, max_cardinality, expected):
        """
        test function - chaid_columns
        """
        if columns:
            _vpy_res = titanic_vd_fun.chaid_columns(
                columns=columns, max_cardinality=max_cardinality
            )
        else:
            _vpy_res = titanic_vd_fun.chaid_columns()
        vpy_res = [col.replace('"', "") for col in _vpy_res]

        assert vpy_res == expected

    @pytest.mark.parametrize(
        "columns, name, threshold, robust",
        [
            ("age", "outliers", 2.5, False),
            ("age", "outliers", 2, True),
        ],
    )
    def test_outliers(self, titanic_vd_fun, columns, name, threshold, robust):
        """
        test function - outliers
        """
        titanic_pdf = titanic_vd_fun.to_pandas()
        titanic_pdf[columns] = titanic_pdf[columns].astype("float")

        titanic_vd_fun.outliers(
            columns=columns, name=name, threshold=threshold, robust=robust
        )
        vpy_res = list(chain(*titanic_vd_fun[[name]].to_list()))

        py_data = titanic_pdf[columns]
        if robust:
            titanic_pdf[name] = (
                (
                    abs(py_data - py_data.median())
                    / (1.4826 * median_abs_deviation(py_data, nan_policy="omit"))
                )
                > threshold
            ).astype(int)
        else:
            titanic_pdf[name] = (
                ((py_data - py_data.mean()) / py_data.std()) > threshold
            ).astype(int)
        py_res = titanic_pdf[name].tolist()

        assert vpy_res == py_res

    @pytest.mark.parametrize(
        "response, columns, nbins, method, RFmodel_params",
        [
            ("survived", "pclass", 16, "same_width", None),
            ("survived", "pclass", 10, "smart", None),
        ],
    )
    def test_pivot_table_chi2(
        self, titanic_vd_fun, response, columns, nbins, method, RFmodel_params
    ):
        """
        test function - pivot_table_chi2
        """
        titanic_pdf = titanic_vd_fun.to_pandas()
        (
            vpy_chi2,
            vpy_p_val,
            vpy_dof,
            vpy_categories,
            vpy_is_numerical,
        ) = titanic_vd_fun.pivot_table_chi2(
            response=response, columns=[columns]
        ).to_list()[
            0
        ]
        contingency_table = pd.crosstab(
            titanic_pdf[response], titanic_pdf[columns]
        ).to_numpy()
        py_chi2, py_p_val, py_dof, _ = chi2_contingency(contingency_table)

        assert (
            (vpy_chi2 == pytest.approx(py_chi2))
            and (vpy_p_val == pytest.approx(py_p_val))
            and (vpy_dof == pytest.approx(py_dof))
        )

    @pytest.mark.parametrize(
        "columns, r",
        [
            (["SepalLengthCm", "SepalWidthCm"], 2),
            (["SepalLengthCm", "SepalWidthCm", "PetalLengthCm"], 3),
        ],
    )
    def test_polynomial_comb(self, iris_vd_fun, columns, r):
        """
        test function - polynomial_comb
        """
        iris_pdf = iris_vd_fun.to_pandas()

        col_name = "_".join(columns)
        vpy_res = iris_vd_fun.polynomial_comb(columns=columns, r=r)

        if r == 3:
            iris_pdf[col_name] = (
                iris_pdf[columns[0]] * iris_pdf[columns[1]] * iris_pdf[columns[2]]
            )
        else:
            iris_pdf[col_name] = iris_pdf[columns[0]] * iris_pdf[columns[1]]

        assert vpy_res[col_name].sum() == float(iris_pdf[col_name].sum())

    def test_recommend(self, market_vd):
        """
        test function - recommend
        """
        # need to check on which library needs to use for comparison. spark uses alternating least squares (ALS)
        assert market_vd.recommend("Name", "Form").shape() == (126, 4)

    @pytest.mark.parametrize(
        "model_class, metric_name",
        [
            ("LinearRegression", "mse"),
            ("LinearRegression", "r2"),
        ],
    )
    def test_score(self, model_class, get_vpy_model, regression_metrics, metric_name):
        """
        test function - test_score
        """
        vpy_res = get_vpy_model(model_class).pred_vdf.score(
            y_true="quality", y_score="quality_pred", metric=metric_name
        )
        metrics_map = regression_metrics(model_class)
        py_res = metrics_map[metric_name]

        assert vpy_res == pytest.approx(py_res)

    @pytest.mark.parametrize(
        "session_threshold, name, expected",
        [
            ("1 time", "slot", "seems to be incorrect"),
            ("10 minutes", "slot", (11844, 4)),
            ("1 hour", "slot", (11844, 4)),
        ],
    )
    def test_sessionize(self, smart_meters_vd, session_threshold, name, expected):
        """
        test function - sessionize
        """
        smart_meters_copy = smart_meters_vd.copy()
        try:
            smart_meters_copy.sessionize(
                ts="time", by=["id"], session_threshold=session_threshold, name=name
            )
            assert smart_meters_copy.shape() == expected
        except errors.QueryError as exception_info:
            assert expected in exception_info.args[0]

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
