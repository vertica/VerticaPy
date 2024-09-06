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
import itertools

from verticapy import drop
from verticapy.datasets import (
    load_amazon,
    load_iris,
    load_smart_meters,
    load_titanic,
)

from verticapy.pipeline._transform import transformation


class TestTransform:
    """
    test class for Transform tests
    """

    """
    Analytic Functions test
    - analytic 
    - interpolate
    - sessionize [NOT SUPPORTED]
    """

    @pytest.mark.parametrize(
        "func, columns, by, order_by, name",
        [
            ("aad", "age", "pclass", None, "new_colm"),
            ("beta", ["age", "fare"], None, None, "new_colm"),
            ("count", "age", "pclass", None, "new_colm"),
            ("corr", ["age", "fare"], None, None, "new_colm"),
            ("cov", ["age", "fare"], None, None, "new_colm"),
            (
                "ema",
                "age",
                None,
                {"name": "asc", "ticket": "desc"},
                "new_colm",
            ),  # Passed. vpy is returning nulls from the row when it gets 1st null
            ("first_value", "age", None, {"name": "asc", "ticket": "desc"}, "new_colm"),
            ("iqr", "age", None, None, "new_colm"),
            ("dense_rank", None, None, {"pclass": "desc", "sex": "desc"}, "new_colm"),
            ("kurtosis", "age", None, None, "new_colm"),
            ("jb", "age", None, None, "new_colm"),
            ("lead", "age", None, {"name": "asc", "ticket": "desc"}, "new_colm"),
            ("lag", "age", None, {"name": "asc", "ticket": "desc"}, "new_colm"),
            (
                "last_value",
                "age",
                "home.dest",
                {"name": "asc", "ticket": "desc"},
                "new_colm",
            ),
            ("mad", "age", None, None, "new_colm"),
            ("max", "age", None, None, "new_colm"),
            ("mean", "age", None, None, "new_colm"),
            ("median", "age", None, None, "new_colm"),
            ("min", "age", None, None, "new_colm"),
            ("mode", "embarked", None, None, "new_colm"),
            ("10%", "age", None, None, "new_colm"),
            ("pct_change", "age", None, {"name": "asc", "ticket": "desc"}, "new_colm"),
            ("percent_rank", None, None, {"name": "asc", "ticket": "desc"}, "new_colm"),
            ("prod", "body", "pclass", None, "new_colm"),
            ("range", "age", None, None, "new_colm"),
            ("rank", None, None, {"pclass": "desc", "sex": "desc"}, "new_colm"),
            ("row_number", None, None, {"name": "asc", "ticket": "desc"}, "new_colm"),
            ("sem", "age", None, None, "new_colm"),
            ("skewness", "age", None, None, "new_colm"),
            ("sum", "age", None, None, "new_colm"),
        ],
    )
    def test_analytic(self, func, columns, by, order_by, name):
        load_titanic()
        transform = {
            name: {
                "transform_method": {
                    "name": "analytic",
                    "params": {
                        "func": func,
                        "columns": columns,
                        # Add 'by' key iff by is not None
                        **({"by": [by]} if by is not None else {}),
                        "order_by": order_by,
                    },
                }
            }
        }
        # Column gets made in output column
        res = transformation(transform, "public.titanic")
        assert f'"{name}"' in res.get_columns()
        drop("public.titanic")

    @pytest.mark.parametrize(
        "ts, rule, method, by",
        [
            ("time", "1 hour", "bfill", "id"),
            ("time", "1 hour", "ffill", "id"),
            ("time", "1 hour", "linear", "id"),
        ],
    )
    def test_interpolate(self, ts, rule, method, by):
        load_smart_meters()
        transform = {
            "new_colm": {
                "sql": ts,
                "transform_method": {
                    "name": "interpolate",
                    "params": {
                        "ts": "new_colm",
                        "rule": rule,
                        "method": {"val": method},
                        "by": [by],
                    },
                },
            }
        }
        # Column gets made in output column
        res = transformation(transform, "public.smart_meters")
        assert '"new_colm"' in res.get_columns()
        drop("public.smart_meters")

    """
    Custom Features Creation test
    - case_when [NOT SUPPORTED]
    - eval
    """

    @pytest.mark.parametrize(
        "name, expr",
        [
            ("family_size", "parch + sibsp + 1"),
            ("missing_cabins", "CASE WHEN cabin IS NULL THEN 'missing' ELSE cabin END"),
        ],
    )
    def test_eval(self, name, expr):
        load_titanic()
        transform = {
            name: {"transform_method": {"name": "eval", "params": {"expr": expr}}}
        }
        res = transformation(transform, "public.titanic")
        assert f'"{name}"' in res.get_columns()
        drop("public.titanic")

    """
    Features Transformations test
    vDataFrame
    ----------
    - abs [NOT SUPPORTED]
    - apply [NOT SUPPORTED]
    - applymap [NOT SUPPORTED]
    - polynomial_comb
    - swap [NOT SUPPORTED]

    vDataColumn
    -----------
    - abs
    - apply
    - apply_fun
    - date_part
    - round
    - slice
    """

    @pytest.mark.parametrize(
        "columns, r",
        [
            (["SepalLengthCm", "SepalWidthCm"], 2),
            (["SepalLengthCm", "SepalWidthCm", "PetalLengthCm"], 3),
        ],
    )
    def test_polynomial_comb(self, columns, r):
        load_iris()
        transform = {
            "new_colm": {
                # This is a dummy to create the column
                "sql": "SepalLengthCm",
                "transform_method": {
                    "name": "polynomial_comb",
                    "params": {
                        "columns": columns,
                        "r": r,
                    },
                },
            }
        }
        res = transformation(transform, "public.iris")
        elements = columns
        combinations = itertools.product(elements, repeat=r)
        unique_combinations = set(
            tuple(sorted(c, key=elements.index)) for c in combinations
        )
        result = ["_".join(perm) for perm in unique_combinations]
        for col in result:
            assert f'"{col}"' in res.get_columns()
        drop("public.iris")

    @pytest.mark.parametrize(
        "columns",
        [
            ("age"),
            ("fare"),
            ("survived"),
        ],
    )
    def test_abs(self, columns):
        load_titanic()
        transform = {
            "new_colm": {
                "sql": columns,
                "transform_method": {
                    "name": "abs",
                },
            }
        }
        res = transformation(transform, "public.titanic")
        assert '"new_colm"' in res.get_columns()
        drop("public.titanic")

    @pytest.mark.parametrize(
        "columns, func, copy_name",
        [
            (["age"], "POWER({},2)", None),
            (["age"], "POWER({},2)", "age_pow2"),
        ],
    )
    def test_apply(self, columns, func, copy_name):
        load_titanic()
        transform = {
            "new_colm": {
                "sql": columns[0],
                "transform_method": {
                    "name": "apply",
                    "params": {
                        "func": func,
                        "copy_name": copy_name,
                    },
                },
            }
        }
        res = transformation(transform, "public.titanic")
        assert '"new_colm"' in res.get_columns()
        drop("public.titanic")

    @pytest.mark.parametrize(
        "columns, vpy_func",
        [
            ("age", "abs"),
            ("survived", "acos"),
            ("survived", "asin"),
            ("survived", "atan"),
            ("age", "cbrt"),
            ("age", "ceil"),
            ("age", "cos"),
            ("age", "cosh"),
            ("age", "cot"),
            ("age", "exp"),
            ("age", "floor"),
            ("age", "ln"),
            ("age", "log"),
            ("age", "log10"),
            ("age", "mod"),
            ("age", "pow"),
            ("age", "round"),
            ("age", "sin"),
            ("age", "sinh"),
            ("age", "sqrt"),
            ("age", "tan"),
            ("age", "tanh"),
        ],
    )
    def test_apply_fun(self, columns, vpy_func):
        load_titanic()
        transform = {
            "new_colm": {
                "sql": columns,
                "transform_method": {
                    "name": "apply_fun",
                    "params": {
                        "func": vpy_func,
                    },
                },
            }
        }
        res = transformation(transform, "public.titanic")
        assert '"new_colm"' in res.get_columns()
        drop("public.titanic")

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
    def test_date_part(self, part, columns):
        load_smart_meters()
        transform = {
            "new_colm": {
                "sql": columns,
                "transform_method": {
                    "name": "date_part",
                    "params": {
                        "field": part,
                    },
                },
            }
        }
        res = transformation(transform, "public.smart_meters")
        assert '"new_colm"' in res.get_columns()
        drop("public.smart_meters")

    @pytest.mark.parametrize(
        "column, n",
        [
            ("age", 4),
            ("fare", 2),
        ],
    )
    def test_round(self, column, n):
        load_titanic()
        transform = {
            "new_colm": {
                "sql": column,
                "transform_method": {
                    "name": "round",
                    "params": {
                        "n": n,
                    },
                },
            }
        }
        res = transformation(transform, "public.titanic")
        assert '"new_colm"' in res.get_columns()
        drop("public.titanic")

    @pytest.mark.parametrize(
        "length, unit, start, column",
        (
            [
                (30, "minute", False, "time"),
                (1, "hour", True, "time"),
            ]
        ),
    )
    def test_slice(self, length, unit, start, column):
        load_smart_meters()
        transform = {
            "new_colm": {
                "sql": column,
                "transform_method": {
                    "name": "slice",
                    "params": {
                        "length": length,
                        "unit": unit,
                        "start": start,
                    },
                },
            }
        }
        res = transformation(transform, "public.smart_meters")
        assert '"new_colm"' in res.get_columns()
        drop("public.smart_meters")

    """
    Moving Windows test
    vDataFrame
    ----------
    - cummax
    - cummin
    - cumprod
    - cumsum
    - rolling [NOT SUPPORTED]
    """

    @pytest.mark.parametrize(
        "func, columns, by, order_by, name",
        [
            ("cummax", "number", "state", "date", "cummax_num"),
            ("cummin", "number", "state", "date", "cummin_num"),
            ("cumprod", "number", "state", "date", "cumprod_num"),
            ("cumsum", "number", "state", "date", "cumsum_num"),
        ],
    )
    def test_cum_funcs(self, func, columns, by, order_by, name):
        load_amazon()
        transform = {
            name: {
                "transform_method": {
                    "name": func,
                    "params": {
                        "column": columns,
                        "by": [by],
                        "order_by": [order_by],
                    },
                }
            }
        }
        res = transformation(transform, "public.amazon")
        assert f'"{name}"' in res.get_columns()
        drop("public.amazon")

    """
    Moving Windows test
    vDataFrame
    ----------
    - regexp
    
    vDataColumn
    ----------
    - str_contains
    - str_count
    - str_extract
    - str_replace
    - str_slice
    """

    @pytest.mark.parametrize(
        "column, pattern, method, position, occurrence, replacement, return_position, name",
        [
            ("name", "son", "count", 1, None, None, None, "name_regex"),
            ("name", "Mrs", "ilike", None, 1, None, None, "name_ilike"),
            ("name", "Mrs.", "instr", 1, 1, None, 0, "name_instr"),
            ("name", "mrs", "like", None, 1, None, None, "name_like"),
            ("name", "Mrs", "not_ilike", None, 1, None, None, "name_not_ilike"),
            ("name", "Mrs.", "not_like", 1, 1, None, 0, "name_not_like"),
            ("name", "Mrs.", "replace", 1, 1, "Mr.", 0, "name_replace"),
            ("name", "([^,]+)", "substr", 1, 1, None, 0, "name_substr"),
        ],
    )
    def test_regexp(
        self,
        column,
        pattern,
        method,
        position,
        occurrence,
        replacement,
        return_position,
        name,
    ):
        load_titanic()
        transform = {
            name: {
                "transform_method": {
                    "name": "regexp",
                    "params": {
                        "column": column,
                        "pattern": pattern,
                        "method": method,
                        "position": position,
                        "occurrence": occurrence,
                        "replacement": replacement,
                        "return_position": return_position,
                    },
                }
            }
        }
        res = transformation(transform, "public.titanic")
        assert f'"{name}"' in res.get_columns()
        drop("public.titanic")

    @pytest.mark.parametrize(
        "func, column, pat",
        [
            ("str_contains", "name", r"([A-Za-z]+\.)"),
            ("str_count", "name", r"([A-Za-z]+\.)"),
            ("str_extract", "name", r"([A-Za-z]+\.)"),
        ],
    )
    def test_text(self, func, column, pat):
        load_titanic()
        transform = {
            "new_colm": {
                "sql": column,
                "transform_method": {"name": func, "params": {"pat": pat}},
            }
        }

        res = transformation(transform, "public.titanic")
        assert '"new_colm"' in res.get_columns()
        drop("public.titanic")

    @pytest.mark.parametrize(
        "column, to_replace, value", [("name", r" ([A-Za-z])+\.", "VERTICAPY")]
    )
    def test_str_replace(self, column, to_replace, value):
        load_titanic()
        transform = {
            "new_colm": {
                "sql": column,
                "transform_method": {
                    "name": "str_replace",
                    "params": {
                        "to_replace": to_replace,
                        "value": value,
                    },
                },
            }
        }
        res = transformation(transform, "public.titanic")
        assert '"new_colm"' in res.get_columns()
        drop("public.titanic")

    @pytest.mark.parametrize("column, start, end", [("name", 0, 3), ("name", 0, 4)])
    def test_str_slice(self, column, start, end):
        load_titanic()
        transform = {
            "new_colm": {
                "sql": column,
                "transform_method": {
                    "name": "str_slice",
                    "params": {
                        "start": start,
                        "step": end,
                    },
                },
            }
        }
        res = transformation(transform, "public.titanic")
        assert '"new_colm"' in res.get_columns()
        drop("public.titanic")

    """
    Binary Operator test
    vDataFrame
    ----------
    - add
    - div
    - mul
    - sub
    """

    @pytest.mark.parametrize(
        "func, columns, scalar",
        [
            ("add", "age", 2.643),
            ("div", "age", 2.12),
            ("mul", "age", 2),
            ("sub", "age", 2),
        ],
    )
    def test_binary_operator(self, func, columns, scalar):
        load_titanic()
        transform = {
            "new_colm": {
                "sql": columns,
                "transform_method": {
                    "name": func,
                    "params": {
                        "x": scalar,
                    },
                },
            }
        }
        res = transformation(transform, "public.titanic")
        assert '"new_colm"' in res.get_columns()
        drop("public.titanic")
