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
from itertools import chain

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder


class TestVDFEncoding:
    """
    test class for encoding function test for vDataFrame class
    """

    def test_case_when(self, titanic_vd_fun):
        """
        test function - case_when
        """
        titanic_vd_fun.case_when(
            "age_category",
            titanic_vd_fun["age"] < 12,
            "children",
            titanic_vd_fun["age"] < 18,
            "teenagers",
            titanic_vd_fun["age"] > 60,
            "seniors",
            titanic_vd_fun["age"] < 25,
            "young adults",
            "adults",
        )

        assert titanic_vd_fun["age_category"].distinct() == [
            "adults",
            "children",
            "seniors",
            "teenagers",
            "young adults",
        ]

    @pytest.mark.parametrize(
        "encode_column, prefix_sep, drop_first, use_numbers_as_suffix",
        [
            ("Species", "_", False, False),
            ("Species", "_", True, False),
            ("Species", "_", False, True),
        ],
    )
    @pytest.mark.parametrize(
        "columns, prefix",
        [
            (None, None),
            ("Species", None),
        ],
    )
    @pytest.mark.parametrize("method", ["one_hot_encode", "get_dummies"])
    def test_one_hot_encode(
        self,
        iris_vd_fun,
        columns,
        prefix,
        encode_column,
        prefix_sep,
        drop_first,
        use_numbers_as_suffix,
        method,
    ):
        """
        test function - one_hot_encode
        """
        iris_pdf = iris_vd_fun.to_pandas()
        unique_val = iris_pdf[encode_column].unique()

        idx = -(len(unique_val) - 1) if drop_first else -len(unique_val)

        _vpy_res = getattr(iris_vd_fun, method)(
            columns=columns,
            prefix_sep=prefix_sep,
            drop_first=drop_first,
            use_numbers_as_suffix=use_numbers_as_suffix,
        ).get_columns()[idx:]

        vpy_res = [v.replace('"', "") for v in _vpy_res]

        py_res = pd.get_dummies(
            iris_pdf,
            columns=[encode_column],
            prefix=prefix,
            prefix_sep=prefix_sep,
            drop_first=drop_first,
        ).columns.to_list()[idx:]

        if use_numbers_as_suffix:
            _py_res = []
            number_map = dict(zip(unique_val, range(len(unique_val))))

            for _res in py_res:
                for _k, _v in number_map.items():
                    if _res.endswith(_k):
                        _py_res.append(_res.replace(_k, str(_v)))

            py_res = _py_res

        if drop_first:
            _vpy_res, _py_res = [], []
            for _vpy, _py in zip(vpy_res, py_res):
                if _vpy.startswith(encode_column) and _py.startswith(encode_column):
                    _vpy_res.append(encode_column)
                    _py_res.append(encode_column)
            vpy_res, py_res = _vpy_res, _py_res

        assert vpy_res == pytest.approx(py_res)


class TestVDCEncoding:
    """
    test class for encoding function test for vDataColumn class
    """

    @pytest.mark.parametrize(
        "column, breaks, labels, include_lowest, right",
        [
            ("age", [0, 15, 80], None, True, True),
            ("parch", [0, 5, 10], ["small", "big"], True, True),
            ("fare", [0, 15, 800], None, True, False),
            ("fare", [0, 15, 800], None, False, False),
        ],
    )
    def test_cut(self, titanic_vd_fun, column, breaks, labels, include_lowest, right):
        """
        test function - cut
        """
        titanic_pdf = titanic_vd_fun.to_pandas()
        titanic_pdf[column] = titanic_pdf[column].astype(float)

        titanic_vd_fun[column].cut(breaks=breaks, labels=labels)
        vpy_res = titanic_vd_fun[column].distinct()

        _py_res = (
            pd.cut(
                titanic_pdf[column],
                bins=breaks,
                labels=labels,
                right=right,
                include_lowest=include_lowest,
            )
            .dropna()
            .unique()
            .categories.values
        )
        if labels:
            py_res = _py_res.tolist()
        else:
            py_res = _py_res.to_tuples().tolist()

        print(
            f"break: {breaks} \nVerticaPy Result: {vpy_res} \nPython Result :{py_res}\n"
        )

        assert len(vpy_res) == pytest.approx(len(py_res))

    def test_decode(self, titanic_vd_fun):
        """
        test function - decode
        """
        titanic_pdf = titanic_vd_fun.to_pandas()
        titanic_vd_fun["sex"].decode("male", 0, "female", 1, 2)

        titanic_pdf["sex_decode"] = titanic_pdf["sex"].apply(
            lambda x: 0 if x == "male" else (1 if x == "female" else 2)
        )

        assert (
            titanic_vd_fun["sex"].distinct().sort()
            == titanic_pdf["sex_decode"].unique().sort()
        )

    @pytest.mark.parametrize(
        "column, method, h, nbins, k, new_category, rf_model_params, response, expected",
        [
            ("age", "same_width", 10, None, None, None, None, None, None),
            ("age", "same_freq", None, 5, None, None, None, None, None),
            (
                "age",
                "smart",
                None,
                6,
                None,
                None,
                {"n_estimators": 100, "nbins": 100},
                "survived",
                6,
            ),
            ("age", "auto", 0, -1, None, None, None, None, 12),
            ("name", "topk", 0, -1, 6, "rare", None, None, 7),
        ],
    )
    def test_discretize(
        self,
        titanic_vd_fun,
        column,
        method,
        h,
        nbins,
        k,
        new_category,
        rf_model_params,
        response,
        expected,
    ):
        """
        test function - discretize
        """
        titanic_pdf = titanic_vd_fun.to_pandas()

        if method == "topk":
            titanic_vd_fun["name"].str_extract(r" ([A-Za-z])+\.")

        titanic_vd_fun[column].discretize(
            method=method,
            h=h,
            nbins=nbins,
            k=k,
            new_category=new_category,
            RFmodel_params=rf_model_params,
            response=response,
        )
        vpy_res = titanic_vd_fun[column].distinct()

        if method == "same_width":
            titanic_pdf[column] = titanic_pdf[column].astype(float)

            bins = range(0, int(max(titanic_pdf[column])) + 20, 10)
            py_res = (
                pd.cut(
                    titanic_pdf[column],
                    bins=bins,
                    right=False,
                )
                .dropna()
                .unique()
                .categories.values
            )
        elif method == "same_freq":
            titanic_pdf[column] = titanic_pdf[column].astype(float)

            py_res = (
                pd.qcut(titanic_pdf[column], q=nbins)
                .dropna()
                .unique()
                .categories.values
            )
        else:
            py_res = expected

        print(
            f"Method: {method} \nVerticaPy Result: {vpy_res} \nPython Result :{py_res}\n"
        )

        assert len(vpy_res) == pytest.approx(py_res if expected else len(py_res))

    @pytest.mark.parametrize(
        "encode_column, prefix_sep, drop_first, use_numbers_as_suffix",
        [
            ("Species", "_", False, False),
            ("Species", "_", True, False),
            ("Species", "_", False, True),
        ],
    )
    @pytest.mark.parametrize(
        "columns, prefix",
        [
            ("Species", None),
            ("Species", "dummy"),
        ],
    )
    @pytest.mark.parametrize("method", ["one_hot_encode", "get_dummies"])
    def test_one_hot_encode(
        self,
        iris_vd_fun,
        columns,
        prefix,
        encode_column,
        prefix_sep,
        drop_first,
        use_numbers_as_suffix,
        method,
    ):
        """
        test function - one_hot_encode
        """
        iris_pdf = iris_vd_fun.to_pandas()
        unique_val = iris_pdf[encode_column].unique()

        idx = -(len(unique_val) - 1) if drop_first else -len(unique_val)

        _vpy_res = getattr(iris_vd_fun[columns], method)(
            prefix=prefix,
            prefix_sep=prefix_sep,
            drop_first=drop_first,
            use_numbers_as_suffix=use_numbers_as_suffix,
        ).get_columns()[idx:]
        vpy_res = [v.replace('"', "") for v in _vpy_res]

        py_res = pd.get_dummies(
            iris_pdf,
            columns=[encode_column],
            prefix=prefix,
            prefix_sep=prefix_sep,
            drop_first=drop_first,
        ).columns.to_list()[idx:]

        if use_numbers_as_suffix:
            _py_res = []
            number_map = dict(zip(unique_val, range(len(unique_val))))

            for _res in py_res:
                for _k, _v in number_map.items():
                    if _res.endswith(_k):
                        _py_res.append(_res.replace(_k, str(_v)))

            py_res = _py_res

        if drop_first:
            _vpy_res, _py_res = [], []
            for _vpy, _py in zip(vpy_res, py_res):
                if _vpy.startswith(encode_column) and _py.startswith(encode_column):
                    _vpy_res.append(encode_column)
                    _py_res.append(encode_column)
            vpy_res, py_res = _vpy_res, _py_res

        assert vpy_res == pytest.approx(py_res)

    @pytest.mark.parametrize("column", ["embarked"])
    def test_label_encode(self, titanic_vd_fun, column):
        """
        test function - label_encode
        """
        titanic_pdf = titanic_vd_fun.to_pandas()

        vpy_res = titanic_vd_fun[column].label_encode()[column].distinct()

        la_encod = LabelEncoder()
        py_res = np.unique(la_encod.fit_transform(titanic_pdf[column]))

        assert vpy_res == pytest.approx(py_res)

    @pytest.mark.parametrize("target, response,", [("embarked", "survived")])
    def test_mean_encode(self, titanic_vd_fun, target, response):
        """
        test function - mean_encode
        """
        titanic_pdf = titanic_vd_fun.to_pandas()

        vpy_mean_val = titanic_vd_fun[target].mean_encode(response=response)
        vpy_res = vpy_mean_val[target].distinct()

        py_mean_val = (
            titanic_pdf.groupby([target], dropna=False)[[response]]
            .mean()
            .sort_values(by=response)
            .values.tolist()
        )
        py_res = list(chain(*py_mean_val))

        assert vpy_res == pytest.approx(py_res)
