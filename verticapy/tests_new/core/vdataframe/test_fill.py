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
import datetime
import pandas as pd
import pytest


class TestFill:
    """
    test class for fill function test
    """

    @pytest.mark.parametrize(
        "column, val, method",
        [
            (
                "age",
                None,
                {"age": "mean"},
            ),
            (
                "boat",
                None,
                {"boat": "mode"},
            ),
            (
                "fare",
                None,
                {"fare": "median"},
            ),
            (
                "body",
                None,
                {"body": "0ifnull"},
            ),
            (
                "cabin",
                None,
                {"cabin": "auto"},
            ),
            # (
            #     "boat",
            #     {"boat": "No boat"},
            #     None,
            # ),
        ],
    )
    @pytest.mark.parametrize(
        "function_type, numeric_only, expr, by, order_by",
        [("vDataFrame", None, None, None, None), ("vcolumn", None, None, None, None)],
    )
    def test_fillna(
        self,
        titanic_vd,
        function_type,
        column,
        val,
        method,
        numeric_only,
        expr,
        by,
        order_by,
    ):
        """
        test function - fillna
        """
        titanic_pdf = titanic_vd.to_pandas()
        titanic_pdf["age"] = titanic_pdf["age"].astype(float)

        if function_type == "vDataFrame":
            vpy_res = titanic_vd.fillna(
                val=val, method=method, numeric_only=numeric_only
            )[column]
        else:
            vpy_res = titanic_vd[column].fillna(
                val=val, method=method[column], expr=expr, by=by, order_by=order_by
            )[column]

        if method[column] == "auto":
            method[column] = (
                "mean" if titanic_vd[column].dtype().startswith("numeric") else "mode"
            )

        if method[column] == "mode":
            titanic_pdf[column] = eval(
                f"titanic_pdf[column].fillna(titanic_pdf[column].{method[column]}()[0])"
            )
        elif method[column] == "0ifnull":
            titanic_pdf[[column]] = titanic_pdf[[column]].applymap(
                lambda x: 0 if pd.isnull(x) else 1
            )
        else:
            titanic_pdf[column] = eval(
                f"titanic_pdf[column].fillna(titanic_pdf[column].{method[column]}())"
            )
        py_res = titanic_pdf[column]

        if titanic_vd[column].dtype().startswith("numeric"):
            vpy_res = vpy_res.sum()
            py_res = py_res.sum()
        else:
            vpy_res = vpy_res.count()
            py_res = py_res.count()

        print(
            f"method name: {method[column]} \nVerticaPy Result: {vpy_res} \nPython Result :{py_res}\n"
        )

        assert vpy_res == pytest.approx(py_res)

    @pytest.mark.parametrize(
        "ts, rule, method, by, expected",
        [
            ("time", "1 hour", "bfill", "id", [1, 0, 0.277]),
            ("time", "1 hour", "ffill", "id", [1, 0, 0.029]),
            ("time", "1 hour", "linear", "id", [1, 0, 0.209363636363636]),
        ],
    )
    def test_interpolate(self, smart_meters_vd, ts, rule, method, by, expected):
        """
        test function - interpolate
        """
        vpy_res = smart_meters_vd.interpolate(
            ts=ts, rule=rule, method={"val": method}, by=[by]
        )
        vpy_res.sort({"id": "asc", "time": "asc"})

        assert (vpy_res["time"][3] - vpy_res["time"][2]) == datetime.timedelta(
            hours=expected[0]
        )
        assert vpy_res["id"][2] == expected[1]
        assert vpy_res["val"][2] == pytest.approx(expected[2])

    def test_clip(self, market_vd):
        """
        test function - clip
        """
        market_pdf = market_vd.to_pandas()
        vpy_res = market_vd["Price"].clip(lower=1.0, upper=4.0)["Price"].mean()
        py_res = market_pdf["Price"].clip(1.0, 4.0).mean()

        print(
            f"method name: 'Clip' \nVerticaPy Result: {vpy_res} \nPython Result :{py_res}\n"
        )

        assert vpy_res == pytest.approx(py_res)

    @pytest.mark.parametrize(
        "column, method, threshold, use_threshold, alpha",
        [
            ("Price", "null", 0.4, True, 0.05),
            ("Price", "winsorize", 0.4, True, 0.05),
            ("Price", "winsorize", None, False, 0.2),
            ("Price", "mean", 0.4, True, 0.05),
            ("Price", "mean", 0.4, True, 0.05),
        ],
    )
    def test_fill_outliers(
        self, market_vd, column, method, threshold, use_threshold, alpha
    ):
        """
        test function - fill_outliers
        """
        market_pdf = market_vd.to_pandas()
        py_data = market_pdf[[column]]

        if use_threshold:
            lower_limit = -threshold * py_data[column].std() + py_data[column].mean()
            upper_limit = threshold * py_data[column].std() + py_data[column].mean()
        else:
            lower_limit, upper_limit = (
                py_data.quantile(alpha)[column],
                py_data.quantile(1 - alpha)[column],
            )

        if method == "null":
            lower_limit_val = upper_limit_val = None
        elif method == "winsorize":
            lower_limit_val, upper_limit_val = lower_limit, upper_limit
        else:
            lower_limit_val, upper_limit_val = (
                py_data.loc[py_data[column] < lower_limit].mean()[column],
                py_data.loc[py_data[column] > upper_limit].mean()[column],
            )

        vpy_res = (
            market_vd[column]
            .fill_outliers(
                method=method,
                threshold=threshold,
                use_threshold=use_threshold,
                alpha=alpha,
            )[column]
            .mean()
        )

        py_res = py_data.apply(
            lambda x: lower_limit_val
            if x[column] < lower_limit
            else (upper_limit_val if x[column] > upper_limit else x[column]),
            axis=1,
        ).mean()

        print(
            f"method name: {'method'} \nVerticaPy Result: {vpy_res} \nPython Result :{py_res}\n"
        )

        assert vpy_res == pytest.approx(py_res)
