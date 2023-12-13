"""
Copyright  (c)  2018-2023 Open Text  or  one  of its
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
import os
import json
import pandas as pd
import pytest
import verticapy as vp
from verticapy.utilities import read_json, TableSample


class TestPivot:
    """
    test class for Pivot functions test
    """

    @pytest.mark.parametrize(
        "validation_type",
        ["columns", "data"],
    )
    def test_flat_vmap(self, validation_type):
        """
        test function - flat_vmap
        """
        path_to_json = os.path.dirname(vp.__file__) + "/datasets/data/laliga/2009.json"

        # verticapy
        vpy_laliga = read_json(path_to_json)
        vpy_data = vpy_laliga[["away_team.managers"]].flat_vmap(
            vmap_col="away_team.managers"
        )

        # python
        with open(path_to_json, encoding="utf-8") as f:
            json_data = json.loads(f.read())
        pds = pd.DataFrame(json_data)["away_team"]
        py_data = pd.json_normalize(pds[:2], "managers")

        if validation_type == "columns":
            vpy_res = [
                col.split(".")[-1].replace('"', "")
                for col in vpy_data.get_columns()[1:]
            ].sort()
            py_res = [col.split(".")[-1] for col in py_data.columns.to_list()].sort()
        else:
            vpy_res = vpy_data.isin({"away_team.managers.0.id": ["211"]})[
                ["away_team.managers.0.name"]
            ].to_list()[0]
            py_res = py_data.loc[py_data["id"].isin([211])]["name"].to_list()

        print(f"Verticapy Result: {vpy_res} \nPython Result: {py_res}")
        assert vpy_res == py_res

    def test_merge_similar_names(self):
        """
        test function - merge_similar_names
        """
        x = TableSample(
            {
                "age": [50, None, None, None],
                "information.age": [100, None, 30, None],
                "dict.age": [None, 80, None, None],
                "age.people": [None, None, None, 10],
                "num": [1, 2, 3, 4],
            }
        ).to_vdf()
        result = x.merge_similar_names(skip_word=["information.", "dict.", ".people"])
        assert result[["age"]][0] == [50]

    @pytest.mark.parametrize(
        "index, columns, values, aggr",
        [
            ("date", "state", "number", "sum"),
        ],
    )
    def test_narrow(self, amazon_vd, index, columns, values, aggr):
        """
        test function - narrow
        """
        amazon_pdf = amazon_vd.to_pandas()
        vpy_pv = amazon_vd.pivot(
            index=index,
            columns=columns,
            values=values,
            aggr=aggr,
        )
        py_pv = pd.pivot_table(
            amazon_pdf,
            index=[index],
            columns=[columns],
            values=values,
            aggfunc=aggr,
        )

        vpy_res = vpy_pv.narrow(index, col_name=columns, val_name=values)[
            columns
        ].count()

        py_res = pd.melt(
            py_pv, ignore_index=False, var_name=columns, value_name=values
        )[columns].count()

        assert vpy_res == py_res

    @pytest.mark.parametrize(
        "index, columns, values, aggr, parm, value",
        [
            ("date", "state", "number", "sum", "columns_val", None),
            ("date", "state", "number", "sum", "agg_col", "ACRE"),
            ("date", "state", "number", "sum", "prefix", "pv_"),
        ],
    )
    def test_pivot(self, amazon_vd, index, columns, values, aggr, parm, value):
        """
        test function - pivot
        """
        amazon_pdf = amazon_vd.to_pandas()
        _vpy_res = amazon_vd.pivot(
            index=index,
            columns=columns,
            values=values,
            aggr=aggr,
            prefix=value if parm == "prefix" else "",
        )
        _py_res = pd.pivot_table(
            amazon_pdf,
            index=[index],
            columns=[columns],
            values=values,
            aggfunc=aggr,
        )

        if parm == "columns_val":
            vpy_res = [v.replace('"', "") for v in _vpy_res.get_columns()][1:]
            py_res = _py_res.columns.to_list()
        elif parm == "agg_col":
            vpy_res = _vpy_res[value].sum()
            py_res = _py_res[value].sum()
        else:
            vpy_res = [v.replace('"', "") for v in _vpy_res.get_columns()][1]
            py_res = f"{value}{_py_res.columns.to_list()[0]}"

        assert vpy_res == py_res
