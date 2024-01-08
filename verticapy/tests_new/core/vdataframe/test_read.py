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
import pytest


class TestRead:
    """
    test class for read functions test
    """

    @pytest.mark.parametrize("exclude_columns", [["pclass", "age"], None])
    def test_get_columns(self, titanic_vd, exclude_columns):
        """
        test function - get_columns
        """
        titanic_pdf = titanic_vd.to_pandas()

        vpy_col_names = [
            col.replace('"', "")
            for col in titanic_vd.get_columns(exclude_columns=exclude_columns)
        ]

        exclude_columns = exclude_columns if exclude_columns else [None]
        py_col_names = [
            col for col in list(titanic_pdf.columns) if col not in exclude_columns
        ]

        assert vpy_col_names == py_col_names

    @pytest.mark.parametrize(
        "function_type, columns, limit",
        [
            ("vDataFrame", "age", None),
            ("vDataFrame", "age", 10),
            ("vcolumn", "ticket", 2),
            ("vcolumn", "ticket", None),
        ],
    )
    @pytest.mark.parametrize("func", ["head", "tail"])
    def test_head_tail(self, titanic_vd, func, function_type, columns, limit):
        """
        test function - head
        """
        titanic_pdf = titanic_vd.to_pandas()

        vdf = titanic_vd.sort({"age": "desc", "name": "asc"})
        pdf = titanic_pdf.sort_values(
            by=["age", "name"], ascending=[False, True]
        ).reset_index(drop=True)

        if function_type == "vDataFrame":
            if limit:
                vpy_res = getattr(vdf, func)(limit=limit)
                py_res = getattr(pdf, func)(n=limit)
            else:
                vpy_res = getattr(vdf, func)()
                py_res = getattr(pdf, func)()

            assert len(vpy_res.to_vdf()) == len(py_res)
        else:
            if limit:
                vpy_res = list(
                    chain(*getattr(vdf[columns], func)(limit=limit).to_list())
                )
                py_res = getattr(pdf[columns], func)(n=limit).values.tolist()
            else:
                vpy_res = list(chain(*getattr(vdf[columns], func)().to_list()))
                py_res = getattr(pdf[columns], func)().values.tolist()

            assert vpy_res == py_res

    @pytest.mark.parametrize(
        "function_type, limit, offset, columns, select_column",
        [
            ("vDataFrame", 2, 5, None, "name"),
            ("vDataFrame", 2, None, None, "name"),
            ("vDataFrame", None, 6, None, "name"),
            ("vDataFrame", None, None, None, "name"),
            ("vDataFrame", 4, 20, ["ticket", "home.dest"], None),
            ("vDataFrame", 4, None, ["ticket", "home.dest"], None),
            ("vDataFrame", None, 7, ["ticket"], None),
            ("vcolumn", 2, 5, "ticket", "name"),
            ("vcolumn", 2, None, "ticket", "name"),
            ("vcolumn", None, 5, "ticket", "name"),
            ("vcolumn", None, None, "ticket", "name"),
        ],
    )
    def test_iloc(
        self, titanic_vd, function_type, limit, offset, columns, select_column
    ):
        """
        test function - iloc
        """
        titanic_pdf = titanic_vd.to_pandas()

        vdf = titanic_vd.sort({"age": "desc", "name": "asc"})
        pdf = titanic_pdf.sort_values(
            by=["age", "name"], ascending=[False, True]
        ).reset_index(drop=True)
        pdf.index = pdf.index + 1

        if function_type == "vDataFrame":
            if limit and offset and columns is None:
                vpy_res = vdf.iloc(limit=limit, offset=offset)[select_column]
                py_res = list(
                    chain(
                        *pdf.iloc[offset : offset + limit][
                            [select_column]
                        ].values.tolist()
                    )
                )
            elif limit and offset is None and columns is None:
                vpy_res = vdf.iloc(limit=limit)[select_column]
                py_res = list(chain(*pdf.iloc[:limit][[select_column]].values.tolist()))
            elif limit is None and offset and columns is None:
                vpy_res = vdf.iloc(offset=offset)[select_column]
                py_res = list(
                    chain(
                        *pdf.iloc[offset : offset + 5][[select_column]].values.tolist()
                    )
                )
            elif columns:
                limit = limit if limit else 5
                offset = offset if offset else 0
                vpy_res = list(
                    chain(
                        *vdf.iloc(
                            limit=limit,
                            offset=offset,
                            columns=columns,
                        ).to_list()
                    )
                )
                py_res = list(
                    chain(*pdf.iloc[offset : offset + limit][columns].values.tolist())
                )
            else:
                vpy_res = vdf.iloc()[select_column]
                py_res = list(chain(*pdf.iloc[:5][[select_column]].values.tolist()))
        else:
            if limit and offset:
                vpy_res = list(
                    chain(*vdf[columns].iloc(limit=limit, offset=offset).to_list())
                )
                py_res = pdf[columns].iloc[offset : offset + limit].values.tolist()
            elif limit and offset is None:
                vpy_res = list(chain(*vdf[columns].iloc(limit=limit).to_list()))
                py_res = pdf[columns].iloc[:limit].values.tolist()
            elif limit is None and offset:
                vpy_res = list(chain(*vdf[columns].iloc(offset=offset).to_list()))
                py_res = pdf[columns].iloc[offset : offset + 5].values.tolist()
            else:
                vpy_res = list(chain(*vdf[columns].iloc().to_list()))
                py_res = pdf[columns].iloc[:5].values.tolist()

        assert vpy_res == py_res

    def test_shape(self, titanic_vd):
        """
        test function - shape
        """
        titanic_pdf = titanic_vd.to_pandas()

        assert titanic_vd.shape() == titanic_pdf.shape

    @pytest.mark.parametrize(
        "_columns",
        ["name", ["name", "ticket"]],
    )
    def test_select(self, titanic_vd, _columns):
        """
        test function - select
        """
        titanic_pdf = titanic_vd.to_pandas()
        py_res = titanic_pdf[_columns].values.tolist()
        if isinstance(_columns, list):
            py_res = list(chain(*py_res))

        vpy_res = list(chain(*titanic_vd.select(columns=_columns).to_list()))

        assert vpy_res == py_res

    @pytest.mark.parametrize(
        "n, column",
        [(4, "fare"), (None, "fare")],
    )
    @pytest.mark.parametrize("func", ["nlargest", "nsmallest"])
    def test_nlargest_nsmallest(self, titanic_vd, func, n, column):
        """
        test function - nlargest
        """
        titanic_pdf = titanic_vd.to_pandas()
        titanic_pdf[column] = titanic_pdf[column].astype(float)

        if n:
            vpy_res = getattr(titanic_vd[column], func)(n=n)[column]
            py_res = getattr(titanic_pdf[column], func)(n=n).values.tolist()
        else:
            vpy_res = getattr(titanic_vd[column], func)()[column]
            py_res = getattr(titanic_pdf[column], func)(n=10).values.tolist()

        assert vpy_res == py_res
