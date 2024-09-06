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
from math import trunc
import pandas as pd
import pytest
from vertica_python.errors import QueryError


class TestFilter:
    """
    test class for filter functions test
    """

    def test_at_time(self, smart_meters_vd):
        """
        test function - at_time
        """
        smart_meters_pdf = smart_meters_vd.copy().to_pandas()
        smart_meters_pdf.set_index("time", inplace=True)

        vpy_res = smart_meters_vd.copy().at_time(ts="time", time="12:00")
        py_res = smart_meters_pdf.at_time(time="12:00")

        assert len(vpy_res) == len(py_res)

    @pytest.mark.parametrize(
        "column, method, x, order_by",
        [
            ("sex", "under", 0.4, None),
            ("sex", "over", 0.6, None),
            # ("sex", "under", 0.2, ["name"]),  # getting different data order
        ],
    )
    def test_balance(self, titanic_vd_fun, column, method, x, order_by):
        """
        test function - balance
        """
        titanic_vd_fun = titanic_vd_fun.sort({"age": "desc", "name": "asc"})

        vpy_res_map = dict(
            titanic_vd_fun.balance(column, method=method, x=x)
            .groupby(columns=[column], expr=[f"count({column})"])
            .to_list()
        )

        result_map = dict(
            titanic_vd_fun.groupby(columns=["sex"], expr=["count(sex)"]).to_list()
        )
        max_value, min_value = max(result_map.values()), min(result_map.values())
        max_key = [k for k, v in result_map.items() if v == max_value][0]
        min_key = [k for k, v in result_map.items() if v == min_value][0]

        if method == "over" and order_by is None:
            expected_result = max(min_value, trunc(max_value * x))
            result_map[min_key] = expected_result
        elif method == "under" and order_by is None:
            expected_result = max(min_value, trunc(max_value * (1 - x)))
            result_map[max_key] = expected_result
        else:
            titanic_copy = titanic_vd_fun.copy()
            result_map = titanic_copy.balance(
                column, method=method, x=x, order_by=order_by
            )[[column]][:5].to_list()

            vpy_res_map = titanic_vd_fun.balance(
                column, method=method, x=x, order_by=order_by
            )[[column]][:5].to_list()

        assert vpy_res_map == result_map

    @pytest.mark.parametrize(
        "column, start, end, inplace",
        [
            ("time", "2014-01-01", None, False),
            ("time", None, "2014-01-02", False),
            ("time", "2014-01-01", "2014-01-02", False),
            ("time", "2014-01-01", "2014-01-02", True),
            ("id", 2, 5, False),
            ("val", 0.148, 1.489, False),
        ],
    )
    def test_between(self, smart_meters_vd, column, start, end, inplace):
        """
        test function - between
        """
        smart_meters_vd_copy = smart_meters_vd.copy()
        smart_meters_pdf = smart_meters_vd.copy().to_pandas()

        if start and end is None and inplace is False:
            vpy_res = smart_meters_vd_copy.between(
                column=column, start=start, inplace=inplace
            )
            py_res = smart_meters_pdf[smart_meters_pdf[column] >= start]
        elif start is None and end and inplace is False:
            vpy_res = smart_meters_vd_copy.between(
                column=column, end=end, inplace=inplace
            )
            py_res = smart_meters_pdf[smart_meters_pdf[column] <= end]
        elif start and end and inplace is False:
            vpy_res = smart_meters_vd_copy.between(
                column=column, start=start, end=end, inplace=inplace
            )
            py_res = smart_meters_pdf[
                smart_meters_pdf[column].between(left=start, right=end)
            ]
        else:
            smart_meters_vd_copy.between(
                column=column, start=start, end=end, inplace=inplace
            )
            vpy_res = smart_meters_vd_copy
            py_res = smart_meters_pdf[
                smart_meters_pdf[column].between(left=start, right=end)
            ]

        assert len(vpy_res) == len(py_res)

    @pytest.mark.parametrize(
        "ts, start_time, end_time, inplace",
        [
            ("time", "12:00", "14:00", False),
            ("time", "12:00", "14:00", True),
        ],
    )
    def test_between_time(self, smart_meters_vd, ts, start_time, end_time, inplace):
        """
        test function - between_time
        """
        smart_meters_vd_copy = smart_meters_vd.copy()
        smart_meters_pdf = smart_meters_vd.copy().to_pandas()
        smart_meters_pdf.set_index(ts, inplace=True)

        if start_time and end_time and inplace is False:
            vpy_res = smart_meters_vd_copy.between_time(
                ts=ts, start_time=start_time, end_time=end_time, inplace=inplace
            )
        else:
            smart_meters_vd_copy.between_time(
                ts=ts, start_time=start_time, end_time=end_time, inplace=inplace
            )
            vpy_res = smart_meters_vd_copy
        py_res = smart_meters_pdf.between_time(start_time=start_time, end_time=end_time)

        assert len(vpy_res) == len(py_res)

    @pytest.mark.parametrize(
        "columns",
        [
            ["pclass", "boat", "embarked"],
            "age",
        ],
    )
    @pytest.mark.parametrize("function_type", ["vDataFrame", "vcolumn"])
    def test_drop(self, titanic_vd_fun, function_type, columns):
        """
        test function - drop
        """
        numer_of_columns = len(titanic_vd_fun.get_columns())

        if function_type == "vDataFrame":
            titanic_vd_fun.drop(columns=columns)
        else:
            titanic_vd_fun[columns].drop()

        assert (
            len(titanic_vd_fun.get_columns()) == numer_of_columns - 1
            if isinstance(columns, str)
            else len(columns)
        )

    @pytest.mark.parametrize(
        "columns",
        [
            None,
            ["free_sulfur_dioxide", "total_sulfur_dioxide", "density"],
            "residual_sugar",
        ],
    )
    def test_drop_duplicates(self, winequality_vpy_fun, columns):
        """
        test function - drop_duplicates
        """
        winequality_pdf = winequality_vpy_fun.to_pandas()
        record_count = len(winequality_pdf)

        vpy_res = winequality_vpy_fun.drop_duplicates(columns=columns).shape()[0]
        py_res = winequality_pdf.drop_duplicates(subset=columns).shape[0]

        assert vpy_res < record_count and vpy_res == py_res

    @pytest.mark.parametrize(
        "columns",
        [
            ["pclass", "boat", "embarked"],
            "ticket",
        ],
    )
    @pytest.mark.parametrize("function_type", ["vDataFrame", "vcolumn"])
    def test_dropna(self, titanic_vd_fun, function_type, columns):
        """
        test function - dropna
        """
        titanic_pdf = titanic_vd_fun.to_pandas()

        if function_type == "vDataFrame":
            titanic_vd_fun.dropna(columns=columns)
            titanic_pdf.dropna(subset=columns, inplace=True)
        else:
            titanic_vd_fun[columns].dropna()
            titanic_pdf[columns].dropna(inplace=True)

        assert len(titanic_vd_fun) == len(titanic_pdf)

    @pytest.mark.parametrize(
        "conditions, force_filter, raise_error",
        [
            ("sex = 'female' AND pclass = 1", True, False),
            ("cast(sex as int) AND pclass = 1", True, True),
            ("sex = 'female' AND pclass = 1", False, False),
        ],
    )
    def test_filter(self, titanic_vd_fun, conditions, force_filter, raise_error):
        """
        test function - filter
        """
        titanic_pdf = titanic_vd_fun.to_pandas()

        if not raise_error:
            titanic_vd_fun.filter(
                conditions=conditions,
                force_filter=force_filter,
                raise_error=raise_error,
            )
            titanic_subset_pdf = titanic_pdf[
                (titanic_pdf["sex"] == "female") & (titanic_pdf["pclass"] == 1)
            ]
            assert len(titanic_vd_fun) == len(titanic_subset_pdf)
        else:
            with pytest.raises(QueryError) as exception_info:
                titanic_vd_fun.filter(
                    conditions=conditions,
                    force_filter=force_filter,
                    raise_error=raise_error,
                )
            assert exception_info.match(
                'Could not convert "female" from column titanic.sex to an int8'
            )

    def test_first(self, smart_meters_vd):
        """
        test function - first
        """

        smart_meters_vd_copy = smart_meters_vd.copy()
        smart_meters_pdf = smart_meters_vd.copy().to_pandas()
        smart_meters_pdf.set_index("time", inplace=True)

        smart_meters_vd_copy.first(ts="time", offset="1 Day")

        min_datetime = smart_meters_pdf.index.min()
        py_res = smart_meters_pdf.loc[
            min_datetime : min_datetime + pd.DateOffset(days=1)
        ]

        assert len(smart_meters_vd_copy) == len(py_res)

    @pytest.mark.parametrize(
        "function_type, column, conditions",
        [
            ("vDataFrame", None, {"sex": ["female"], "survived": [1], "parch": [1]}),
            ("vcolumn", "sex", ["female"]),
        ],
    )
    def test_isin(self, titanic_vd_fun, function_type, column, conditions):
        """
        test function - isin
        """
        titanic_pdf = titanic_vd_fun.to_pandas()

        if function_type == "vDataFrame":
            vpy_res = titanic_vd_fun.isin(conditions)

            _py_res = titanic_pdf.isin(conditions)
            py_res = _py_res[
                (_py_res.sex == True)
                & (_py_res.survived == True)
                & (_py_res.parch == True)
            ]
        else:
            vpy_res = titanic_vd_fun[column].isin(conditions)

            _py_res = titanic_pdf[column].isin(conditions)
            print(_py_res)
            py_res = _py_res[(_py_res == True)]

        assert len(vpy_res) == len(py_res)

    def test_last(self, smart_meters_vd):
        """
        test function - last
        """
        smart_meters_vd_copy = smart_meters_vd.copy()
        smart_meters_pdf = smart_meters_vd.copy().to_pandas()
        smart_meters_pdf.set_index("time", inplace=True)

        smart_meters_vd_copy.last(ts="time", offset="1 Day")

        max_datetime = smart_meters_pdf.index.max()
        py_res = smart_meters_pdf.loc[
            max_datetime - pd.DateOffset(days=1) : max_datetime
        ]

        assert len(smart_meters_vd_copy) == len(py_res)

    @pytest.mark.parametrize(
        "columns, n, x, method, by",
        [
            (["name", "age"], 48, None, "random", None),
            # (["name", "age"], None, 0.5, "random", None),  # verticapy record counts are not consistency
            # (["name", "age"], 120, None, "systematic", None),  # need to implement
            # (["name", "age"], 120, None, "stratified", ["pclass", "sex"]), # need to implement
        ],
    )
    def test_sample(self, titanic_vd_fun, columns, n, x, method, by):
        """
        test function - sample
        """
        titanic_pdf = titanic_vd_fun.to_pandas()
        titanic_pdf[columns[1]] = titanic_pdf[columns[1]].astype(float)

        py_res = titanic_pdf.sample(n=n, frac=x, random_state=1)
        vpy_res = titanic_vd_fun.sample(n=n, x=x, method=method, by=by)

        assert len(vpy_res) == len(py_res)

    @pytest.mark.parametrize(
        "test_type, columns, conditions, usecols, expr, order_by",
        [
            (
                "single_condition",
                "age",
                "age BETWEEN 30 AND 70",
                ["name", "pclass", "boat", "embarked", "age", "family_size"],
                ["sibsp + parch + 1 AS family_size"],
                {"age": "desc", "family_size": "asc", "name": "asc"},
            ),
            (
                "multiple_condition",
                ["age", "pclass"],
                ["age BETWEEN 30 AND 64", "pclass = 1"],
                ["name", "pclass", "boat", "embarked", "age", "family_size"],
                ["sibsp + parch + 1 AS family_size"],
                {"age": "desc", "family_size": "asc", "name": "asc"},
            ),
        ],
    )
    def test_search(
        self, titanic_vd_fun, test_type, columns, conditions, usecols, expr, order_by
    ):
        """
        test function - search
        """
        titanic_pdf = titanic_vd_fun.to_pandas()
        titanic_pdf["age"] = titanic_pdf["age"].astype(float)

        if test_type == "multiple_condition":
            py_res = titanic_pdf.eval(expr="family_size=sibsp + parch + 1")[
                (titanic_pdf[columns[0]] >= 30)
                & (titanic_pdf[columns[0]] <= 64)
                & (titanic_pdf[columns[1]] == 1)
            ][usecols].sort_values(
                by=list(order_by.keys()), ascending=[False, True, True]
            )
        else:
            py_res = titanic_pdf.eval(expr="family_size=sibsp + parch + 1")[
                (titanic_pdf[columns] >= 30) & (titanic_pdf[columns] <= 70)
            ][usecols].sort_values(
                by=list(order_by.keys()), ascending=[False, True, True]
            )

        vpy_res = titanic_vd_fun.search(
            conditions=conditions, usecols=usecols, expr=expr, order_by=order_by
        )

        assert vpy_res[5] == py_res.iloc[5].tolist() and len(vpy_res) == len(py_res)

    @pytest.mark.parametrize(
        "column, threshold, use_threshold, alpha",
        [
            ("age", 3.0, True, 0.05),
            ("age", 3.0, False, 0.07),
        ],
    )
    def test_drop_outliers(
        self, titanic_vd_fun, column, threshold, use_threshold, alpha
    ):
        """
        test function - drop_outliers
        """
        titanic_pdf = titanic_vd_fun.to_pandas()
        titanic_pdf[column] = titanic_pdf[column].astype(float)

        vpy_res = titanic_vd_fun[column].drop_outliers(
            threshold=threshold, use_threshold=use_threshold, alpha=alpha
        )

        if use_threshold:
            mean, std = titanic_pdf[[column]].mean(), titanic_pdf[[column]].std()
            titanic_pdf[f"{column}_outliers"] = titanic_pdf[column].apply(
                lambda x: ((abs(x) - mean) / std) < 3.0
            )
            py_res = titanic_pdf[titanic_pdf[f"{column}_outliers"] == True]
        else:
            lower_limit, upper_limit = (
                titanic_pdf[[column]].quantile(alpha)[column],
                titanic_pdf[[column]].quantile(1 - alpha)[column],
            )
            py_res = titanic_pdf[
                (titanic_pdf[column] >= lower_limit)
                & (titanic_pdf[column] <= upper_limit)
            ]

        assert len(vpy_res) == len(py_res)
