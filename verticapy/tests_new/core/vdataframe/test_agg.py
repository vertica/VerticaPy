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
from contextlib import nullcontext as does_not_raise
import pytest
import numpy as np
from scipy.stats import median_abs_deviation, jarque_bera
from verticapy.errors import MissingColumn
from verticapy.tests_new.core.vdataframe import REL_TOLERANCE, ABS_TOLERANCE
from verticapy.tests_new import functions, AggregateFun


class TestAgg:
    """
    test class for aggregate function test
    """

    @pytest.mark.parametrize(
        "dataset, columns, expr, rollup, having, raise_err, expected",
        [
            (
                "market_vd",
                ["Form", "Name"],
                ["AVG(Price) AS avg_price", "STDDEV(Price) AS std"],
                None,
                None,
                does_not_raise(),
                (159, 4),
            ),
            (
                "market_vd",
                ["Form", "Name"],
                ["AVG(Price) AS avg_price", "STDDEV(Price) AS std"],
                True,
                None,
                does_not_raise(),
                (197, 4),
            ),
            (
                "market_vd",
                ["Form", "Name"],
                ["AVG(Price) AS avg_price", "STDDEV(Price) AS std"],
                True,
                "AVG(Price) > 2",
                does_not_raise(),
                (73, 4),
            ),
            (
                "market_vd",
                ["Form", "Name"],
                ["AVG(Price) AS avg_price", "STDDEV(Price) AS std"],
                [False, True],
                None,
                does_not_raise(),
                (196, 4),
            ),
            (
                "titanic_vd",
                [("pclass", "sex"), "embarked"],
                ["AVG(survived) AS avg_survived"],
                [True, False],
                None,
                does_not_raise(),
                (33, 4),
            ),
            (
                "market_vd",
                ["For", "Name"],
                ["AVG(Price) AS avg_price", "STDDEV(Price) AS std"],
                None,
                None,
                pytest.raises(MissingColumn),
                "The Virtual Column 'For' doesn't exist.\nDid you mean '\"Form\"' ?",
            ),
        ],
    )
    def test_groupby(
        self, request, dataset, columns, expr, rollup, having, raise_err, expected
    ):
        """
        test function - groupby
        """
        data = request.getfixturevalue(dataset)
        # exc_info = None
        with raise_err:
            if rollup and having:
                vpy_res = data.groupby(
                    columns=columns, expr=expr, rollup=rollup, having=having
                ).shape()
            elif rollup and having is None:
                vpy_res = data.groupby(
                    columns=columns, expr=expr, rollup=rollup
                ).shape()
            elif having and rollup is None:
                vpy_res = data.groupby(
                    columns=columns,
                    expr=expr,
                ).shape()
            else:
                vpy_res = data.groupby(
                    columns=columns,
                    expr=expr,
                ).shape()

        assert (
            getattr(raise_err, "excinfo").match(expected)
            if hasattr(raise_err, "excinfo")
            else vpy_res == expected
        )

    @pytest.mark.parametrize(
        "vpy_func, py_func, _rel_tol, _abs_tols, expected",
        [
            ("aad", None, REL_TOLERANCE, ABS_TOLERANCE, None),
            (
                "approx_median",
                None,
                REL_TOLERANCE,
                ABS_TOLERANCE,
                [[3.0, 0.0, 28.0, 0.0, 0.0, 14.4542, 160.5], [28.0, 14.4542, 3.0, 0.0]],
            ),
            (
                "approx_10%",
                None,
                REL_TOLERANCE,
                ABS_TOLERANCE,
                [[1.0, 0.0, 14.5, 0.0, 0.0, 7.5892, 37.7], [14.5, 7.5892, 1.0, 0.0]],
            ),
            (
                "approx_90%",
                None,
                REL_TOLERANCE,
                ABS_TOLERANCE,
                [[3.0, 1.0, 50.0, 1.0, 1.0, 79.13, 297.3], [50.0, 79.13, 3.0, 1.0]],
            ),
            # ("approx_unique", None, None, REL_TOLERANCE, ABS_TOLERANCE, [[3.0, 2.0, 1233.0, 2.0, 96.0, 7.0, 8.0, 888.0, 275.0, 181.0, 3.0, 26.0, 118.0, 355.0], [96.0, 275.0, 3.0, 2.0]]),  # fail due to randomness in output
            ("count", "count", REL_TOLERANCE, ABS_TOLERANCE, None),
            ("cvar", None, REL_TOLERANCE, ABS_TOLERANCE, None),
            # ("dtype", None, REL_TOLERANCE, ABS_TOLERANCE, [['int', 'int', 'varchar(164)', 'varchar(20)', 'numeric(6,3)', 'int', 'int', 'varchar(36)', 'numeric(10,5)', 'varchar(30)', 'varchar(20)', 'varchar(100)', 'int', 'varchar(100)'], ['numeric(6,3)', 'numeric(10,5)', 'int', 'int']]),  # fail due to randomness in output
            ("iqr", None, REL_TOLERANCE, ABS_TOLERANCE, None),
            ("kurtosis", None, REL_TOLERANCE, ABS_TOLERANCE, None),
            ("jb", None, 1e-00, ABS_TOLERANCE, None),
            ("mad", None, REL_TOLERANCE, ABS_TOLERANCE, None),
            ("max", None, REL_TOLERANCE, ABS_TOLERANCE, None),
            ("mean", None, REL_TOLERANCE, ABS_TOLERANCE, None),
            ("median", None, REL_TOLERANCE, ABS_TOLERANCE, None),
            ("min", None, REL_TOLERANCE, ABS_TOLERANCE, None),
            ("mode", None, REL_TOLERANCE, ABS_TOLERANCE, None),
            ("percent", None, 1e-04, ABS_TOLERANCE, None),
            ("10%", None, REL_TOLERANCE, ABS_TOLERANCE, None),
            ("90%", None, REL_TOLERANCE, ABS_TOLERANCE, None),
            # ('prod', None, REL_TOLERANCE, ABS_TOLERANCE, None),  # fail getting inf for pclass column
            ("range", None, REL_TOLERANCE, ABS_TOLERANCE, None),
            ("sem", None, REL_TOLERANCE, ABS_TOLERANCE, None),
            ("skewness", None, REL_TOLERANCE, ABS_TOLERANCE, None),
            ("sum", None, REL_TOLERANCE, ABS_TOLERANCE, None),
            ("std", None, REL_TOLERANCE, ABS_TOLERANCE, None),
            ("top1", None, REL_TOLERANCE, ABS_TOLERANCE, None),
            ("top1_percent", None, 1e-04, ABS_TOLERANCE, None),
            ("unique", None, REL_TOLERANCE, ABS_TOLERANCE, None),
            ("var", None, REL_TOLERANCE, ABS_TOLERANCE, None),
        ],
    )
    @pytest.mark.parametrize(
        "input_type, columns",
        [
            ("vDataFrame", []),
            ("vDataFrame_column", ["age"]),
            ("vcolumn", ["age"]),
            ("vcolumn", ["age", "fare", "pclass", "survived"]),
        ],
    )
    @pytest.mark.parametrize("agg_func_type", ["agg", "aggregate"])
    def test_aggregate(
        self,
        titanic_vd,
        agg_func_type,
        vpy_func,
        py_func,
        columns,
        expected,
        _rel_tol,
        _abs_tols,
        input_type,
    ):
        """
        test function - aggregate
        """
        numeric_columns = [
            "pclass",
            "survived",
            "age",
            "sibsp",
            "parch",
            "fare",
            "body",
        ]

        titanic_pdf = titanic_vd.to_pandas()
        titanic_pdf["age"] = titanic_pdf["age"].astype(float)
        titanic_pdf["fare"] = titanic_pdf["fare"].astype(float)

        # VerticaPy
        if input_type == "vDataFrame":
            if vpy_func in ["top1", "top1_percent"]:
                vpy_res = (
                    getattr(titanic_vd[numeric_columns], agg_func_type)(func=vpy_func)
                    .transpose()
                    .to_list()[0]
                )
            elif vpy_func in ["aad"]:
                res = getattr(titanic_vd[numeric_columns], agg_func_type)(func=vpy_func)
                vpy_res_map = dict(zip(res["index"], res[vpy_func]))
                vpy_res = {k.replace('"', ""): v for k, v in vpy_res_map.items()}
            else:
                vpy_res = (
                    getattr(titanic_vd, agg_func_type)(func=vpy_func)
                    .transpose()
                    .to_list()[0]
                )
        elif input_type == "vDataFrame_column":
            vpy_res = (
                getattr(titanic_vd, agg_func_type)(func=vpy_func, columns=columns)
                .transpose()
                .to_list()[0]
            )
        else:
            vpy_res = (
                getattr(titanic_vd[columns], agg_func_type)(func=vpy_func)
                .transpose()
                .to_list()[0]
            )

        # Python
        py_res = []
        if py_func:
            if input_type == "vDataFrame":
                py_data = titanic_pdf
                py_res = getattr(titanic_pdf, agg_func_type)(func=py_func).tolist()
            else:
                py_res = getattr(titanic_pdf[columns], agg_func_type)(
                    func=py_func
                ).tolist()
        else:
            if expected:
                if input_type == "vDataFrame":
                    py_res = expected[0]
                else:
                    py_res = [expected[1][0]] if len(columns) == 1 else expected[1]
            else:
                _py_func = AggregateFun(*functions[vpy_func]).py
                if input_type == "vDataFrame" and vpy_func not in [
                    "top1",
                    "top1_percent",
                    "jb",
                ]:
                    py_data = (
                        titanic_pdf[numeric_columns]
                        if vpy_func in ["cvar", "jb", "mad"]
                        else titanic_pdf
                    )
                    if vpy_func in ["aad"]:
                        py_res = dict(eval(_py_func))
                    else:
                        py_res = eval(_py_func).tolist()
                else:
                    # py_res = eval(_py_func) if vpy_func in ["top2"] else eval(_py_func).tolist()
                    if vpy_func in ["top1", "top1_percent", "jb"]:
                        for column in columns if columns else numeric_columns:
                            py_data = titanic_pdf[column]
                            py_res.append(eval(f"{_py_func}"))
                        py_res = [None if np.isnan(v) else v for v in py_res]
                    else:
                        py_data = titanic_pdf[columns]
                        py_res = eval(_py_func).tolist()

        if vpy_func in ["mode"]:
            py_res = [None if np.isnan(v) else v for v in py_res[0]]

        print(
            f"Function name: {vpy_func} \nVerticaPy Result: {vpy_res} \nPython Result :{py_res}\n"
        )
        assert vpy_res == pytest.approx(py_res, rel=_rel_tol, abs=_abs_tols)

    @pytest.mark.parametrize(
        "func_name, _rel_tol, _abs_tol",
        [
            ("aad", REL_TOLERANCE, ABS_TOLERANCE),
            ("mean", REL_TOLERANCE, ABS_TOLERANCE),
            ("avg", REL_TOLERANCE, ABS_TOLERANCE),
            ("count", REL_TOLERANCE, ABS_TOLERANCE),
            ("kurt", REL_TOLERANCE, ABS_TOLERANCE),
            ("kurtosis", REL_TOLERANCE, ABS_TOLERANCE),
            ("mad", REL_TOLERANCE, ABS_TOLERANCE),
            ("max", REL_TOLERANCE, ABS_TOLERANCE),
            ("median", REL_TOLERANCE, ABS_TOLERANCE),
            ("min", REL_TOLERANCE, ABS_TOLERANCE),
            # ("prod", REL_TOLERANCE, ABS_TOLERANCE),  # fail getting inf for pclass column
            # # ("product", REL_TOLERANCE, ABS_TOLERANCE),  # fail getting inf for pclass column
            ("quantile", REL_TOLERANCE, ABS_TOLERANCE),
            ("sem", REL_TOLERANCE, ABS_TOLERANCE),
            ("skew", REL_TOLERANCE, ABS_TOLERANCE),
            ("skewness", REL_TOLERANCE, ABS_TOLERANCE),
            ("std", REL_TOLERANCE, ABS_TOLERANCE),
            ("stddev", REL_TOLERANCE, ABS_TOLERANCE),
            ("sum", REL_TOLERANCE, ABS_TOLERANCE),
            ("var", REL_TOLERANCE, ABS_TOLERANCE),
            ("variance", REL_TOLERANCE, ABS_TOLERANCE),
            ("nunique", REL_TOLERANCE, ABS_TOLERANCE),
        ],
    )
    @pytest.mark.parametrize(
        "function_type, columns",
        [
            ("vDataFrame", []),
            ("vDataFrame_columns", ["age"]),
            ("vcolumn", ["age", "fare", "pclass", "survived"]),
        ],
    )
    def test_vdf_vcol(
        self,
        titanic_vd,
        func_name,
        columns,
        _rel_tol,
        _abs_tol,
        function_type,
    ):
        """
        test function - VdataFrame and Vcolumn
        """
        numeric_columns = [
            "pclass",
            "survived",
            "age",
            "sibsp",
            "parch",
            "fare",
            "body",
        ]

        titanic_pdf = titanic_vd.to_pandas()
        titanic_pdf["age"] = titanic_pdf["age"].astype(float)
        titanic_pdf["fare"] = titanic_pdf["fare"].astype(float)
        # data = titanic_pdf[columns[0]]
        vpy_func, py_func = (
            AggregateFun(*functions[func_name]).vpy,
            AggregateFun(*functions[func_name]).py,
        )

        # VerticaPy
        vpy_func_name = (
            vpy_func
            if vpy_func.count(".") == 0
            else vpy_func.split(".")[1].split("(")[0]
        )

        if function_type == "vDataFrame":
            vpy_data = titanic_vd
            if vpy_func_name in ["aad"]:
                res = eval(vpy_func)
                vpy_res_map = dict(zip(res["index"], res[vpy_func_name]))
                vpy_res = {k.replace('"', ""): v for k, v in vpy_res_map.items()}
            else:
                vpy_res = eval(vpy_func).transpose().to_list()[0]
        elif function_type == "vDataFrame_columns":
            vpy_data = titanic_vd
            vpy_res = (
                eval(
                    vpy_func.replace(")", "columns=columns)")
                    if "()" in vpy_func
                    else vpy_func.replace(")", ", columns=columns)")
                )
                .transpose()
                .to_list()[0]
            )
        else:
            vpy_data = titanic_vd[columns]
            vpy_res = eval(vpy_func).transpose().to_list()[0]

        # Python
        if function_type == "vDataFrame":
            if vpy_func_name in ["mad", "sem"]:
                py_data = titanic_pdf[numeric_columns]
                py_res = eval(py_func).tolist()
            elif vpy_func_name in ["aad"]:
                py_data = titanic_pdf
                py_res = dict(eval(py_func))
            else:
                py_data = titanic_pdf
                py_res = eval(py_func).tolist()
        else:
            py_data = titanic_pdf[columns]
            py_res = eval(py_func).tolist()

        print(
            f"Function name: {vpy_func} \ncolumns: {columns} \nVerticaPy Result: {vpy_res} \nPython Result :{py_res}\n"
        )
        assert vpy_res == pytest.approx(
            py_res[0] if func_name == "quantile" else py_res, rel=_rel_tol, abs=_abs_tol
        )

    @pytest.mark.parametrize(
        "func_name, vpy_func, py_func",
        [
            (
                "all",
                "vpy_data.all(columns = columns)['bool_and'][0]",
                "py_data.all()",
            ),
            (
                "any",
                "vpy_data.any(columns = columns)['bool_or'][0]",
                "py_data.any()",
            ),
            (
                "count_percent",
                "vpy_data.count_percent(columns = columns, sort_result=True, desc=False)['percent'][0]",
                "py_data.notnull().mean()*100",
            ),
            (
                "duplicated",
                "vpy_data.duplicated(columns = columns, count=False, limit=30)['occurrence']",  # tablesample doesn't support sorting with null value
                "py_data.groupby(['survived']).size().reset_index(name='occurrence')['occurrence'].tolist()",
            ),
        ],
    )
    def test_vdf(self, titanic_vd, func_name, vpy_func, py_func):
        """
        test function - VdataFrame groupby
        """
        columns = ["survived"]
        titanic_pdf = titanic_vd.to_pandas()
        titanic_pdf["age"] = titanic_pdf["age"].astype(float)
        titanic_pdf["fare"] = titanic_pdf["fare"].astype(float)

        # VerticaPy
        vpy_data = titanic_vd
        vpy_res = eval(vpy_func)

        # Python
        py_data = (
            titanic_pdf[[columns[0]]]
            if func_name == "duplicated"
            else titanic_pdf[columns[0]]
        )
        py_res = eval(py_func)

        print(
            f"Function name: {vpy_func} \nVerticaPy Result: {vpy_res} \nPython Result :{py_res}\n"
        )
        assert vpy_res == pytest.approx(py_res, abs=1e-3)

    @pytest.mark.parametrize(
        "func_name",
        ["value_counts", "topk", "distinct"],
    )
    @pytest.mark.parametrize("columns", ["pclass"])
    def test_vcolumn(self, titanic_vd, columns, func_name):
        """
        test function - Vcolumn groupby
        """
        titanic_pdf = titanic_vd.to_pandas()

        vpy_func, py_func = (
            AggregateFun(*functions[func_name]).vpy,
            AggregateFun(*functions[func_name]).py,
        )

        # VerticaPy
        vpy_data = titanic_vd[columns]
        if func_name == "value_counts":
            vpy_res = eval(vpy_func).transpose().to_list()[0][4:]
        elif func_name == "distinct":
            vpy_res = eval(vpy_func)
        else:
            vpy_res = eval(vpy_func).transpose().to_list()[0]

        # Python
        py_data = titanic_pdf[columns]
        py_res = eval(py_func).tolist()

        print(
            f"Function name: {vpy_func} \nVerticaPy Result: {vpy_res} \nPython Result :{py_res}\n"
        )
        assert vpy_res == pytest.approx(py_res, abs=1e-3)
