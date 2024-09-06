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
import numpy as np
import pytest
import verticapy.stats as st


class TestRolling:
    """
    test class for Text functions test
    """

    @pytest.mark.parametrize(
        "vpy_func, py_func, window, columns, by, order_by",
        [
            ("aad", "aad", (-1, 1), ["age"], None, {"name": "asc", "ticket": "desc"}),
            (
                "beta",
                "beta",
                (-1, 1),
                ["age", "fare"],
                None,
                {"name": "asc", "ticket": "desc"},
            ),  # few rows values are not matching with verticapy. comparing sample records
            (
                "count",
                "count",
                (-1, 1),
                ["age"],
                None,
                {"name": "asc", "ticket": "desc"},
            ),
            (
                "corr",
                "corr",
                (-1, 1),
                ["age", "fare"],
                None,
                {"name": "asc", "ticket": "desc"},
            ),  # all rows value for corr are not matching with verticapy. comparing sample records
            (
                "cov",
                "cov",
                (-1, 1),
                ["age", "fare"],
                None,
                {"name": "asc", "ticket": "desc"},
            ),  # all rows value for cov are not matching with verticapy. comparing sample records
            (
                "kurtosis",
                "kurt",
                (-2, 1),
                ["age"],
                None,
                {"name": "asc", "ticket": "desc"},
            ),  # few rows values are not matching with verticapy. comparing sample records
            (
                "jb",
                "jb",
                (-2, 1),
                ["age"],
                None,
                {"name": "asc", "ticket": "desc"},
            ),  # few rows values are not matching with verticapy. comparing sample records
            ("max", "max", (-1, 1), ["age"], None, {"name": "asc", "ticket": "desc"}),
            ("mean", "mean", (-1, 1), ["age"], None, {"name": "asc", "ticket": "desc"}),
            ("min", "min", (-1, 1), ["age"], None, {"name": "asc", "ticket": "desc"}),
            ("prod", "prod", (-1, 1), ["age"], None, {"name": "asc", "ticket": "desc"}),
            (
                "range",
                "range",
                (-1, 1),
                ["age"],
                None,
                {"name": "asc", "ticket": "desc"},
            ),
            ("sem", "sem", (-1, 1), ["age"], None, {"name": "asc", "ticket": "desc"}),
            (
                "skewness",
                "skew",
                (-1, 1),
                ["age"],
                None,
                {"name": "asc", "ticket": "desc"},
            ),  # few rows values are not matching with verticapy. comparing sample records
            ("sum", "sum", (-1, 1), ["age"], None, {"name": "asc", "ticket": "desc"}),
            ("std", "std", (-1, 1), ["age"], None, {"name": "asc", "ticket": "desc"}),
            ("var", "var", (-1, 1), ["age"], None, {"name": "asc", "ticket": "desc"}),
        ],
    )
    def test_rolling(
        self,
        titanic_vd_fun,
        vpy_func,
        py_func,
        window,
        columns,
        by,
        order_by,
    ):
        """
        test function - str_extract for vColumns
        """
        titanic_pdf = titanic_vd_fun.to_pandas()
        name = f"{vpy_func}_val"
        py_window = sum(-1 * w if w < 0 else w for w in window) + 1

        titanic_vd_fun.rolling(
            func=vpy_func,
            window=window,
            columns=columns,
            name=name,
            order_by=order_by,
        )
        # filling nan/inf/zero
        titanic_vd_fun[name] = st.case_when(
            st.isnan(titanic_vd_fun[name]),
            0,
            st.isinf(titanic_vd_fun[name]),
            0,
            st.zeroifnull(titanic_vd_fun[name]),
        )
        titanic_vd_fun = (
            titanic_vd_fun[:50]
            if vpy_func in ["corr", "cov", "beta", "skewness", "kurtosis", "jb"]
            else titanic_vd_fun
        )
        vpy_res = titanic_vd_fun[name].sum()

        pdf_new = titanic_pdf.sort_values(
            by=list(order_by.keys()),
            ascending=[i == "asc" for i in list(order_by.values())],
        ).reset_index(drop=True)

        # casting
        for idx, column in enumerate(columns):
            pdf_new[column] = pdf_new[column].astype(float)
            if idx == 0:
                titanic_pdf_roll_col0 = pdf_new[column].rolling(
                    py_window, min_periods=1, center=True
                )
            else:
                titanic_pdf_roll_col1 = pdf_new[column].rolling(
                    py_window, min_periods=1, center=True
                )

        if py_func == "aad":
            pdf_new["abs_mean"] = abs(
                pdf_new[columns[0]] - titanic_pdf_roll_col0.mean()
            )
            pdf_new[name] = (
                pdf_new["abs_mean"]
                .rolling(py_window, min_periods=1, center=True)
                .mean()
            )
        elif py_func in ["cov", "corr", "beta"]:
            pdf_new["cov"] = titanic_pdf_roll_col0.cov(pdf_new[columns[1]], ddof=0)
            if py_func == "cov":
                pdf_new[name] = pdf_new["cov"]
            elif py_func == "corr":
                pdf_new[f"std_{columns[0]}"] = titanic_pdf_roll_col0.std()
                pdf_new[f"std_{columns[1]}"] = titanic_pdf_roll_col1.std()
                pdf_new[name] = pdf_new["cov"] / (
                    pdf_new[f"std_{columns[0]}"].replace(0, np.nan)
                    * pdf_new[f"std_{columns[1]}"].replace(0, np.nan)
                )
            elif py_func == "beta":
                pdf_new[f"var_{columns[0]}"] = titanic_pdf_roll_col1.var().replace(
                    0, np.nan
                )
                pdf_new[name] = pdf_new["cov"] / pdf_new[f"var_{columns[0]}"]
            pdf_new = pdf_new[:50]
        elif py_func == "prod":
            pdf_new[name] = titanic_pdf_roll_col0.apply(np.prod)
        elif py_func == "sem":
            pdf_new[name] = getattr(titanic_pdf_roll_col0, py_func)(ddof=0)
        elif py_func in ["skew", "kurt", "jb"]:
            # skew
            pdf_new["skew1"] = pow(
                (pdf_new[columns[0]] - titanic_pdf_roll_col0.mean())
                / titanic_pdf_roll_col0.std(),
                3,
            )
            pdf_new["skew1_mean"] = (
                pdf_new["skew1"].rolling(py_window, min_periods=1, center=True).mean()
            )
            pdf_new["skew2"] = pow(titanic_pdf_roll_col0.count(), 2) / (
                (titanic_pdf_roll_col0.count() - 1)
                * (titanic_pdf_roll_col0.count() - 2)
            ).replace(0, np.nan)

            # kurt
            pdf_new["kurt1"] = (
                pow(titanic_pdf_roll_col0.count(), 1)
                * (titanic_pdf_roll_col0.count() + 1)
            ) / (
                (titanic_pdf_roll_col0.count() - 1)
                * (titanic_pdf_roll_col0.count() - 2)
                * (titanic_pdf_roll_col0.count() - 3)
            ).replace(
                0, np.nan
            )
            pdf_new["kurt2"] = pow(
                (pdf_new[columns[0]] - titanic_pdf_roll_col0.mean())
                / titanic_pdf_roll_col0.std(),
                4,
            )
            pdf_new["kurt2_mean"] = (
                pdf_new["kurt2"].rolling(4, min_periods=1, center=True).sum()
            )
            pdf_new["kurt3"] = (
                3
                * pow((titanic_pdf_roll_col0.count() - 1), 2)
                / (
                    (titanic_pdf_roll_col0.count() - 2)
                    * (titanic_pdf_roll_col0.count() - 3)
                ).replace(0, np.nan)
            )

            if py_func == "skew":
                pdf_new[name] = pdf_new["skew1_mean"] * pdf_new["skew2"]
                pdf_new[name] = pdf_new[name].fillna(0)
            elif py_func == "kurt":
                pdf_new[name] = (pdf_new["kurt1"] * pdf_new["kurt2_mean"]) - pdf_new[
                    "kurt3"
                ]
                pdf_new[name] = pdf_new[name].fillna(0)
            else:
                pdf_new["skew"] = pdf_new["skew1_mean"] * pdf_new["skew2"]
                pdf_new["skew"] = pdf_new["skew"].fillna(0)

                pdf_new["kurt"] = (pdf_new["kurt1"] * pdf_new["kurt2_mean"]) - pdf_new[
                    "kurt3"
                ]
                pdf_new["kurt"] = pdf_new["kurt"].fillna(0)

                pdf_new[name] = (titanic_pdf_roll_col0.count() / 6) * (
                    pow(pdf_new["skew"], 2) + pow((pdf_new["kurt"]), 2) / 4
                )
            pdf_new = pdf_new[:50]
        elif py_func == "range":
            pdf_new[name] = titanic_pdf_roll_col0.max() - titanic_pdf_roll_col0.min()
        else:
            pdf_new[name] = getattr(
                pdf_new[columns].rolling(py_window, center=True, min_periods=1), py_func
            )()

        py_res = pdf_new[name].sum()

        print(
            f"Rolling Function : {vpy_func} \nVerticaPy Result: {vpy_res} \nPython Result :{py_res}\n"
        )

        assert vpy_res == pytest.approx(py_res, rel=1e-02)

    @pytest.mark.parametrize(
        "func, columns, by, order_by, name",
        [
            ("cummax", "number", "state", "date", "cummax_num"),
            ("cummin", "number", "state", "date", "cummin_num"),
            ("cumprod", "number", "state", "date", "cumprod_num"),
            ("cumsum", "number", "state", "date", "cumsum_num"),
        ],
    )
    def test_cum_func(self, amazon_vd, func, columns, by, order_by, name):
        """
        test function - cumulative functions
        """
        amazon_pdf = amazon_vd.to_pandas()
        getattr(amazon_vd, func)(
            column=columns, by=[by], order_by=[order_by], name=name
        ).sort([by, order_by])
        vpy_res = amazon_vd[name].sum()

        py_res = getattr(amazon_pdf.groupby(by=[by])[columns], func)().sum()

        print(f"VerticaPy Result: {vpy_res} \nPython Result :{py_res}\n")

        assert vpy_res == pytest.approx(py_res)
