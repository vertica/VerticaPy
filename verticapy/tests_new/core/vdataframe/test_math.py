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
import re
import datetime
import pandas as pd
import numpy as np  # pylint: disable=unused-import
from scipy.stats import (
    median_abs_deviation,
    jarque_bera,
)  # pylint: disable=unused-import
import pytest
from verticapy.core.tablesample.base import TableSample
from verticapy.tests_new.core.vdataframe import REL_TOLERANCE, ABS_TOLERANCE
from verticapy.tests_new import functions, AggregateFun


class TestMath:
    """
    test class for Text functions test
    """

    @pytest.mark.parametrize(
        "input_type, columns",
        [
            ("vDataFrame", "age"),
            ("vDataFrame_column", "age"),
            ("vDataFrame_column", ["age", "fare", "pclass", "survived"]),
            ("vcolumn", "age"),
            ("vcolumn", ["age", "fare", "pclass", "survived"]),
        ],
    )
    def test_abs(self, titanic_vd_fun, input_type, columns):
        """
        test function - absolute
        """
        focus_columns = [
            "pclass",
            "survived",
            "age",
            "sibsp",
            "parch",
            "fare",
            "body",
        ]

        titanic_vdf = titanic_vd_fun[focus_columns].normalize().fillna(0)
        titanic_pdf = titanic_vdf.to_pandas()
        titanic_pdf[columns] = titanic_pdf[columns].astype(float)

        if input_type == "vDataFrame":
            titanic_vdf.abs()
            vpy_res = titanic_vdf[columns].sum()
            py_res = titanic_pdf.abs()[columns].sum()
        elif input_type == "vDataFrame_column":
            if isinstance(columns, list) and len(columns) > 1:
                _vpy_res = titanic_vdf.abs(columns=columns)[columns].sum()
                vpy_res = dict(zip(_vpy_res["index"], _vpy_res["sum"]))['"age"']
                py_res = dict(titanic_pdf[columns].abs().sum())["age"]
            else:
                vpy_res = titanic_vdf.abs(columns=columns)[columns].sum()
                py_res = titanic_pdf[columns].abs().sum()
        else:
            if isinstance(columns, list) and len(columns) > 1:
                _vpy_res = titanic_vdf[columns].abs()[columns].sum()
                vpy_res = dict(zip(_vpy_res["index"], _vpy_res["sum"]))['"age"']
                py_res = dict(titanic_pdf[columns].abs().sum())["age"]
            else:
                vpy_res = titanic_vdf[columns].abs()[columns].sum()
                py_res = titanic_pdf[columns].abs().sum()

        print(
            f"Input Type: {input_type} \ncolumns: {columns} \nVerticaPy Result: {vpy_res} \nPython Result :{py_res}\n"
        )
        assert vpy_res == pytest.approx(py_res)

    @pytest.mark.parametrize(
        "func, columns, scalar",
        [
            ("add", "age", 2.643),
            ("div", "age", 2.12),
            ("mul", "age", 2),
            ("sub", "age", 2),
        ],
    )
    def test_binary_operator(self, titanic_vd_fun, func, columns, scalar):
        """
        test function - add
        """
        titanic_pdf = titanic_vd_fun.to_pandas()
        titanic_pdf[columns] = titanic_pdf[columns].astype(float)

        vpy_res = getattr(titanic_vd_fun[columns], func)(scalar)[columns].sum()

        py_res = getattr(titanic_pdf[columns], func)(scalar).sum()

        print(
            f"Function Name: {func} \nVerticaPy Result: {vpy_res} \nPython Result :{py_res}\n"
        )

        assert vpy_res == pytest.approx(py_res)

    @pytest.mark.parametrize(
        "columns, input_type, func, copy_name",
        [
            (
                ["age", "boat", "name"],
                "vDataFrame",
                {
                    "age": "COALESCE(age, AVG({}) OVER (PARTITION BY pclass, sex))",
                    "boat": "DECODE({}, NULL, 0, 1)",
                    "name": "REGEXP_SUBSTR({}, '([A-Za-z])+\.')",
                },
                None,
            ),
            (["age"], "vcolumn", "POWER({}, 2)", None),
            (["age"], "vcolumn", "POWER({}, 2)", "age_pow2"),
        ],
    )
    def test_apply(self, titanic_vd_fun, columns, input_type, func, copy_name):
        """
        test function - apply
        """
        titanic_pdf = titanic_vd_fun.to_pandas()
        titanic_pdf[columns[0]] = titanic_pdf[columns[0]].astype(float)

        if input_type == "vDataFrame":
            titanic_vd_fun.apply(func=func)

            vpy_res = [
                titanic_vd_fun[columns[0]].sum(),
                titanic_vd_fun[columns[1]].sum(),
                len(titanic_vd_fun[columns[2]].distinct()),
            ]

            titanic_pdf[columns[0]] = titanic_pdf.groupby(by=["pclass", "sex"])[
                columns[0]
            ].transform(lambda x: x.fillna(x.mean()))
            titanic_pdf[columns[1]] = titanic_pdf[columns[1]].apply(
                lambda x: 1 if x else 0
            )
            titanic_pdf[columns[2]] = titanic_pdf[columns[2]].apply(
                lambda x: re.search(r"([A-Za-z]+\.)", x)[0]
            )
            py_res = [
                titanic_pdf[columns[0]].sum(),
                titanic_pdf[columns[1]].sum(),
                len(titanic_pdf[columns[2]].unique()),
            ]
        else:
            apply_column_name = copy_name if copy_name else columns[0]
            titanic_vd_fun[columns[0]].apply(func=func, copy_name=copy_name)
            vpy_res = (
                titanic_vd_fun[copy_name].sum()
                if copy_name
                else titanic_vd_fun[columns[0]].sum()
            )

            titanic_pdf[apply_column_name] = titanic_pdf[columns[0]].apply(
                lambda x: x**2
            )
            py_res = titanic_pdf[apply_column_name].sum()

        print(f"VerticaPy Result: {vpy_res} \nPython Result :{py_res}\n")
        assert vpy_res == pytest.approx(py_res)

    @pytest.mark.parametrize(
        "columns, data, vpy_func, py_func",
        [
            ("age", None, "abs", "np.absolute(x)"),
            ("survived", None, "acos", "np.arccos(x)"),
            ("survived", None, "asin", "np.arcsin(x)"),
            ("survived", None, "atan", "np.arctan(x)"),
            ("album_cost", "sample_data", "avg", "np.mean(x)"),
            ("album_cost", "sample_data", "mean", "np.mean(x)"),
            ("age", None, "cbrt", "np.cbrt(x)"),
            ("age", None, "ceil", "np.ceil(x)"),
            ("album_cost", "sample_data", "contain", "1 if 2 in x else 0"),
            ("age", None, "cos", "np.cos(x)"),
            ("age", None, "cosh", "np.cosh(x)"),
            ("age", None, "cot", "np.cos(x)/np.sin(x)"),
            ("album_cost", "sample_data", "dim", "np.ndim(x)"),
            ("age", None, "exp", "np.exp(x)"),
            ("album_cost", "sample_data", "find", "1 if 2 in x else -1"),
            ("age", None, "floor", "np.floor(x)"),
            ("album_cost", "sample_data", "len", "np.size(x)"),
            ("album_cost", "sample_data", "length", "np.size(x)"),
            ("age", None, "ln", "np.log(x)"),
            (
                "age",
                None,
                "log",
                "np.log2(x, where=x != 0)",
            ),
            ("age", None, "log10", "np.log10(x)"),
            ("album_cost", "sample_data", "max", "np.max(x)"),
            ("album_cost", "sample_data", "min", "np.min(x)"),
            ("age", None, "mod", "np.mod(x, 2)"),
            ("age", None, "pow", "np.power(x, 2)"),
            ("age", None, "round", "np.round(x, 2)"),
            ("sign_num", "sample_data", "sign", "np.sign(x)"),
            ("age", None, "sin", "np.sin(x)"),
            ("age", None, "sinh", "np.sinh(x)"),
            ("age", None, "sqrt", "np.sqrt(x)"),
            ("album_cost", "sample_data", "sum", "np.sum(x)"),
            ("age", None, "tan", "np.tan(x)"),
            ("age", None, "tanh", "np.tanh(x)"),
        ],
    )
    def test_apply_fun(self, titanic_vd_fun, data, columns, vpy_func, py_func):
        """
        test function - apply_fun
        """
        titanic_pdf = titanic_vd_fun.to_pandas()

        sample_data = TableSample(
            values={
                "index": [0, 1, 2],
                "name": ["Bernard", "Fred", "Cassandra"],
                "fav_album": [
                    ["Inna", "Connect R"],
                    ["Majda Roumi"],
                    ["Beyonce", "Alicia Keys", "Dr Dre"],
                ],
                "album_cost": [
                    [65, 50, 90.11, 25, 71],
                    [40, 50, 90.11, 35],
                    [56, 50, 90.11, 55, 213],
                ],
                "sign_num": [0, -1, 2],
            }
        ).to_vdf()
        vpy_data = sample_data if data == "sample_data" else titanic_vd_fun

        sample_data_pdf = sample_data.to_pandas()
        py_data = sample_data_pdf if data == "sample_data" else titanic_pdf
        py_data[columns] = (
            py_data[columns].astype(float)
            if vpy_data[columns].isnum() and not vpy_data[columns].isarray()
            else py_data[columns]
        )

        vpy_data[columns].apply_fun(func=vpy_func)
        vpy_res = vpy_data[columns].sum()

        py_data[columns] = py_data[columns].apply(lambda x: eval(py_func))
        py_res = float(py_data[columns].sum())

        print(
            f"Function Name: {vpy_func}, \nVerticaPy Result: {vpy_res} \nPython Result :{py_res}\n"
        )
        assert vpy_res == pytest.approx(py_res)

    @pytest.mark.parametrize(
        "part, columns",
        [
            ("hour", "time"),
            ("minute", "time"),
            ("second", "time"),
            ("microsecond", "time"),
            ("day", "time"),
            ("month", "time"),
            ("year", "time"),
            ("quarter", "time"),
        ],
    )
    def test_date_part(self, smart_meters_vd, part, columns):
        """
        test function - date_part
        """
        smart_meters_copy = smart_meters_vd.copy()
        smart_meters_pdf = smart_meters_vd.to_pandas()

        vpy_res = smart_meters_copy[columns].date_part(part)[columns].sum()

        py_res = getattr(smart_meters_pdf[columns].dt, part).sum()

        print(
            f"Date Part: {part} \nVerticaPy Result: {vpy_res} \nPython Result :{py_res}\n"
        )

        assert vpy_res == pytest.approx(py_res)

    @pytest.mark.parametrize("col_type", (["complex", "string"]))
    def test_get_len(self, titanic_vd_fun, laliga_vd, col_type):
        """
        test function - get_len
        """
        titanic_pdf = titanic_vd_fun.to_pandas()
        laliga_pdf = laliga_vd.to_pandas()

        if col_type == "complex":
            vpy_res = laliga_vd["away_team"]["managers"][0]["name"].get_len().sum()
            py_res = (
                laliga_pdf["away_team"]
                .apply(lambda x: len(x["managers"][0]["name"]) if x["managers"] else 0)
                .sum()
            )
        else:
            vpy_res = titanic_vd_fun["name"].get_len().sum()
            py_res = titanic_pdf["name"].apply(len).sum()

        print(f"VerticaPy Result: {vpy_res} \nPython Result :{py_res}\n")

        assert vpy_res == pytest.approx(py_res)

    @pytest.mark.parametrize("column, n", ([("age", 4), ("fare", 2)]))
    def test_round(self, titanic_vd_fun, column, n):
        """
        test function - round
        """
        titanic_pdf = titanic_vd_fun.to_pandas()
        titanic_pdf[column] = titanic_pdf[column].astype(float)

        vpy_res = titanic_vd_fun[column].round(n)[column].sum()

        py_res = titanic_pdf[column].round(n).sum()

        print(f"VerticaPy Result: {vpy_res} \nPython Result :{py_res}\n")

        assert vpy_res == pytest.approx(py_res, rel=1e-04)

    @pytest.mark.parametrize(
        "length, unit, start, column, expected",
        (
            [
                (30, "minute", False, "time", datetime.datetime(2014, 1, 1, 1, 30)),
                (1, "hour", True, "time", datetime.datetime(2014, 1, 1, 1, 00)),
            ]
        ),
    )
    def test_slice(self, smart_meters_vd, length, unit, start, column, expected):
        """
        test function - slice
        """
        vpy_res = (
            smart_meters_vd[column]
            .slice(length=length, unit=unit, start=start)[column]
            .min()
        )

        print(f"VerticaPy Result: {vpy_res} \n")

        assert vpy_res == pytest.approx(expected)

    @pytest.mark.parametrize(
        "func, columns, by, order_by, name, offset, x_smoothing, add_count, _rel_tol, _abs_tol",
        [
            (
                "aad",
                "age",
                "pclass",
                None,
                "new_colm",
                1,
                0.5,
                True,
                REL_TOLERANCE,
                ABS_TOLERANCE,
            ),
            (
                "beta",
                ["age", "fare"],
                None,
                None,
                "new_colm",
                1,
                0.5,
                True,
                REL_TOLERANCE,
                ABS_TOLERANCE,
            ),
            (
                "count",
                "age",
                "pclass",
                None,
                "new_colm",
                1,
                0.5,
                True,
                REL_TOLERANCE,
                ABS_TOLERANCE,
            ),
            (
                "corr",
                ["age", "fare"],
                None,
                None,
                "new_colm",
                1,
                0.5,
                True,
                REL_TOLERANCE,
                ABS_TOLERANCE,
            ),
            (
                "cov",
                ["age", "fare"],
                None,
                None,
                "new_colm",
                1,
                0.5,
                True,
                REL_TOLERANCE,
                ABS_TOLERANCE,
            ),
            (
                "ema",
                "age",
                None,
                {"name": "asc", "ticket": "desc"},
                "new_colm",
                1,
                0.5,
                True,
                REL_TOLERANCE,
                ABS_TOLERANCE,
            ),  # Passed. vpy is returning nulls from the row when it gets 1st null
            (
                "first_value",
                "age",
                None,
                {"name": "asc", "ticket": "desc"},
                "new_colm",
                1,
                0.5,
                True,
                REL_TOLERANCE,
                ABS_TOLERANCE,
            ),
            (
                "iqr",
                "age",
                None,
                None,
                "new_colm",
                1,
                0.5,
                True,
                REL_TOLERANCE,
                ABS_TOLERANCE,
            ),
            (
                "dense_rank",
                None,
                None,
                {"pclass": "desc", "sex": "desc"},
                "new_colm",
                1,
                0.5,
                True,
                REL_TOLERANCE,
                ABS_TOLERANCE,
            ),
            (
                "kurtosis",
                "age",
                None,
                None,
                "new_colm",
                1,
                0.5,
                True,
                REL_TOLERANCE,
                ABS_TOLERANCE,
            ),
            ("jb", "age", None, None, "new_colm", 1, 0.5, True, 1e-02, ABS_TOLERANCE),
            (
                "lead",
                "age",
                None,
                {"name": "asc", "ticket": "desc"},
                "new_colm",
                5,
                0.5,
                True,
                REL_TOLERANCE,
                ABS_TOLERANCE,
            ),
            (
                "lag",
                "age",
                None,
                {"name": "asc", "ticket": "desc"},
                "new_colm",
                5,
                0.5,
                True,
                REL_TOLERANCE,
                ABS_TOLERANCE,
            ),
            (
                "last_value",
                "age",
                "home.dest",
                {"name": "asc", "ticket": "desc"},
                "new_colm",
                1,
                0.5,
                True,
                REL_TOLERANCE,
                ABS_TOLERANCE,
            ),
            (
                "mad",
                "age",
                None,
                None,
                "new_colm",
                1,
                0.5,
                True,
                REL_TOLERANCE,
                ABS_TOLERANCE,
            ),
            (
                "max",
                "age",
                None,
                None,
                "new_colm",
                1,
                0.5,
                True,
                REL_TOLERANCE,
                ABS_TOLERANCE,
            ),
            (
                "mean",
                "age",
                None,
                None,
                "new_colm",
                1,
                0.5,
                True,
                REL_TOLERANCE,
                ABS_TOLERANCE,
            ),
            (
                "median",
                "age",
                None,
                None,
                "new_colm",
                1,
                0.5,
                True,
                REL_TOLERANCE,
                ABS_TOLERANCE,
            ),
            (
                "min",
                "age",
                None,
                None,
                "new_colm",
                1,
                0.5,
                True,
                REL_TOLERANCE,
                ABS_TOLERANCE,
            ),
            (
                "mode",
                "embarked",
                None,
                None,
                "new_colm",
                1,
                0.5,
                True,
                REL_TOLERANCE,
                ABS_TOLERANCE,
            ),
            (
                "10%",
                "age",
                None,
                None,
                "new_colm",
                1,
                0.5,
                True,
                REL_TOLERANCE,
                ABS_TOLERANCE,
            ),
            (
                "pct_change",
                "age",
                None,
                {"name": "asc", "ticket": "desc"},
                "new_colm",
                1,
                0.5,
                True,
                REL_TOLERANCE,
                ABS_TOLERANCE,
            ),
            (
                "percent_rank",
                None,
                None,
                {"name": "asc", "ticket": "desc"},
                "new_colm",
                1,
                0.5,
                True,
                REL_TOLERANCE,
                ABS_TOLERANCE,
            ),
            (
                "prod",
                "body",
                "pclass",
                None,
                "new_colm",
                1,
                0.5,
                True,
                REL_TOLERANCE,
                ABS_TOLERANCE,
            ),
            (
                "range",
                "age",
                None,
                None,
                "new_colm",
                1,
                0.5,
                True,
                REL_TOLERANCE,
                ABS_TOLERANCE,
            ),
            (
                "rank",
                None,
                None,
                {"pclass": "desc", "sex": "desc"},
                "new_colm",
                1,
                0.5,
                True,
                REL_TOLERANCE,
                ABS_TOLERANCE,
            ),
            (
                "row_number",
                None,
                None,
                {"name": "asc", "ticket": "desc"},
                "new_colm",
                1,
                0.5,
                True,
                REL_TOLERANCE,
                ABS_TOLERANCE,
            ),
            (
                "sem",
                "age",
                None,
                None,
                "new_colm",
                1,
                0.5,
                True,
                REL_TOLERANCE,
                ABS_TOLERANCE,
            ),
            (
                "skewness",
                "age",
                None,
                None,
                "new_colm",
                1,
                0.5,
                True,
                REL_TOLERANCE,
                ABS_TOLERANCE,
            ),
            (
                "sum",
                "age",
                None,
                None,
                "new_colm",
                1,
                0.5,
                True,
                REL_TOLERANCE,
                ABS_TOLERANCE,
            ),
            (
                "std",
                "age",
                None,
                None,
                "new_colm",
                1,
                0.5,
                True,
                REL_TOLERANCE,
                ABS_TOLERANCE,
            ),
            (
                "unique",
                "pclass",
                None,
                None,
                "new_colm",
                1,
                0.5,
                True,
                REL_TOLERANCE,
                ABS_TOLERANCE,
            ),
            (
                "var",
                "age",
                None,
                None,
                "new_colm",
                1,
                0.5,
                True,
                REL_TOLERANCE,
                ABS_TOLERANCE,
            ),
        ],
    )
    def test_analytic(
        self,
        titanic_vd_fun,
        func,
        columns,
        by,
        order_by,
        name,
        offset,
        x_smoothing,
        add_count,  # pylint: disable=unused-argument
        _rel_tol,
        _abs_tol,
    ):
        """
        test function - analytic
        """
        titanic_pdf = titanic_vd_fun.to_pandas()
        titanic_pdf["age"] = titanic_pdf["age"].astype(float)
        titanic_pdf["fare"] = titanic_pdf["fare"].astype(float)

        vpy_func, py_func = (
            (AggregateFun(*functions[func]).vpy, AggregateFun(*functions[func]).py)
            if func in functions
            else (func, func)
        )

        if order_by:
            titanic_pdf.sort_values(
                by=list(order_by.keys()),
                ascending=[i == "asc" for i in list(order_by.values())],
                inplace=True,
            )

        if func in ["aad", "count"]:
            titanic_vd_fun.analytic(func=func, columns=columns, by=[by], name=name)
            vpy_res = titanic_vd_fun[name][0]

            py_grp_data = titanic_pdf.groupby([by])[columns]
            py_res = py_grp_data.transform(lambda py_data: eval(py_func))[0]
        elif func in ["ema", "first_value", "last_value", "pct_change"]:
            titanic_vd_fun.analytic(
                func=func, columns=columns, by=by, order_by=order_by, name=name
            )

            if func == "first_value":
                vpy_res = titanic_vd_fun[name].max()
                py_res = titanic_pdf[columns].iloc[0]
            elif func == "last_value":
                # vpy_res = titanic_vd_fun.analytic(func=func, columns=columns, by=by, order_by=order_by, name=name)[by].isin("Belfast, NI")[name]
                vpy_res = titanic_vd_fun[by].isin("Belfast, NI")[name]
                py_res = titanic_pdf.groupby(by).last(columns).loc["Belfast, NI"]
            elif func == "ema":
                vpy_res = titanic_vd_fun[:10][name].sum()
                py_res = (
                    titanic_pdf["age"]
                    .ewm(adjust=False, alpha=x_smoothing)
                    .mean()[:10]
                    .sum()
                )
            else:
                vpy_res = titanic_vd_fun[name].max()
                py_res = (
                    titanic_pdf[columns] / titanic_pdf[columns].shift(periods=1)
                ).max()
        elif func in ["dense_rank", "percent_rank", "rank", "row_number"]:
            titanic_vd_fun.analytic(func=func, order_by=order_by, name=name)
            vpy_res = titanic_vd_fun[name].max()

            if func in ["dense_rank", "percent_rank", "rank"]:
                col1, col2 = list(order_by.keys())[0], list(order_by.keys())[1]
                py_res = (
                    (titanic_pdf[col1].astype(str) + titanic_pdf[col2])
                    .rank(
                        method="dense" if func == "dense_rank" else "min",
                        ascending=False,
                        pct=func == "percent_rank",
                    )
                    .max()
                )
            else:
                titanic_pdf[func] = titanic_pdf.index + 1
                py_res = titanic_pdf[func].max()

        elif func in ["lead", "lag"]:
            titanic_vd_fun.analytic(
                func=func, columns=columns, order_by=order_by, offset=offset, name=name
            )
            vpy_res = titanic_vd_fun[name].sum()

            py_res = (
                titanic_pdf[columns].shift(offset if func == "lag" else -offset).sum()
            )
        else:
            titanic_vd_fun.analytic(func=func, columns=columns, name=name)
            vpy_res = titanic_vd_fun[name][0]

            if py_func in ["cov", "corr", "beta"]:
                py_cov = (titanic_pdf[columns[0]] * titanic_pdf[columns[1]]).mean() - (
                    titanic_pdf[columns[0]].mean() * titanic_pdf[columns[1]].mean()
                )

                if func in ["cov"]:
                    py_res = py_cov
                elif func in ["corr"]:
                    py_res = py_cov / (
                        titanic_pdf[columns[0]].std() * titanic_pdf[columns[1]].std()
                    )
                elif func == "beta":
                    py_var = titanic_pdf[columns[1]].var()
                    py_res = py_cov / py_var
            else:
                py_data = (
                    titanic_pdf[columns].to_frame()
                    if func in ["iqr", "10%", "mode"]
                    else titanic_pdf[columns]
                )
                py_res = eval(py_func)

        print(
            f"Function name: {vpy_func} \ncolumns: {columns} \nVerticaPy Result: {vpy_res} \nPython Result :{py_res}\n"
        )
        assert vpy_res == pytest.approx(
            py_res[0] if func == "quantile" else py_res, rel=_rel_tol, abs=_abs_tol
        )

    @pytest.mark.parametrize("column, func", [("sex", "DECODE({}, NULL, 0, 1)")])
    def test_applymap(self, titanic_vd_fun, column, func):
        """
        test function - applymap
        """
        titanic_pdf = titanic_vd_fun.to_pandas()
        vpy_res = titanic_vd_fun.applymap(func=func, numeric_only=False)[column].sum()
        py_res = titanic_pdf[column].map(lambda x: 0 if pd.isnull(x) else 1).sum()

        print(f"VerticaPy Result: {vpy_res} \nPython Result :{py_res}\n")
        assert vpy_res == pytest.approx(py_res)
