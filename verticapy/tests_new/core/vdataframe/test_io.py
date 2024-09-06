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
import os
import pickle
import shutil

import geopandas
import numpy as np
import pandas as pd
import pytest

import verticapy as vp
from verticapy.connection import current_cursor
from verticapy.sql.sys import current_session, username


def tear_down(path: str) -> None:
    """
    This functions removes files/directories
    """
    if os.path.isfile(path):
        print(f"Deleting file {path}")
        os.remove(path)
    if os.path.isdir(path) and path.startswith("/tmp/"):
        print(f"Deleting directory {path}")
        shutil.rmtree(path)


class TestIO:
    """
    test class for IO operations
    """

    def test_copy(self, titanic_vd_fun):
        """
        test function - copy
        """
        titanic_copy = titanic_vd_fun.copy()

        assert titanic_copy.get_columns() == titanic_vd_fun.get_columns()

    def test_load(self, titanic_vd_fun):
        """
        test function - load
        """
        titanic_vd_fun._vars["saving"] = []
        titanic_vd_fun.save()
        assert len(titanic_vd_fun._vars["saving"]) == 1

        titanic_vd_fun.filter("age < 40")
        titanic_vd_fun["embarked"].drop()
        assert titanic_vd_fun.shape() == (760, 13)

        result = titanic_vd_fun.load()
        assert len(result._vars["saving"]) == 0
        assert result.shape() == (1234, 14)

    def test_save(self, titanic_vd_fun):
        """
        test function - save
        """
        titanic_vd_fun._vars["saving"] = []
        titanic_vd_fun.save()
        assert len(titanic_vd_fun._vars["saving"]) == 1

    @pytest.mark.parametrize(
        "idx, sep, na_rep, quotechar, usecols, header, new_header, order_by, n_files",
        [
            (1, "|", None, None, None, None, None, None, 1),
            (2, None, "Unknown", None, None, None, None, None, 1),
            (3, None, None, "'", None, None, None, None, 1),
            (4, None, None, None, ["name"], None, None, None, 1),
            # failed Header=False
            # (5, None, None, None, None, False, None, None, 1),
            (6, None, None, None, None, None, None, ["name", "age", "fare"], 4),
            (7, None, None, None, None, None, ["name_new", "fare_new"], None, 1),
        ],
    )
    def test_to_csv(
        self,
        titanic_vd_fun,
        idx,
        sep,
        na_rep,
        quotechar,
        usecols,
        header,
        new_header,
        order_by,
        n_files,
    ):
        """
        test function - to_csv
        """
        path = f"/tmp/verticapy_test_to_csv{idx}"
        columns = ["name", "fare"]
        data = titanic_vd_fun

        sep = sep if sep else ","
        header = not header
        file_name = f"{path}.csv"
        quote = '"' if new_header else ""
        vdf_row = vp.vDataFrame({columns[0]: [None], columns[1]: [1.5]})

        tear_down(path)
        if n_files == 1:
            raw_vdf = data.select(columns).sort({columns[0]: "asc", columns[1]: "desc"})
            vdf = raw_vdf[:2].append(vdf_row)

            vdf.to_csv(
                file_name,
                sep=sep,
                na_rep=na_rep,
                usecols=usecols,
                header=header,
                new_header=new_header,
                order_by=order_by,
                n_files=n_files,
            )
            with open(file_name, "r") as file:
                result = file.read()
            # print(result)
            columns = new_header if new_header else columns

            if usecols:
                expected = f"""{columns[0]}\n"Abbing, Mr. Anthony"\n"Abbott, Master. Eugene Joseph"\n{na_rep if na_rep else ''}"""
            else:
                expected = f"""{quote}{columns[0]}{quote}{sep}{quote}{columns[1]}{quote}\n"Abbing, Mr. Anthony"{sep}7.55000\n"Abbott, Master. Eugene Joseph"{sep}20.25000\n{na_rep if na_rep else ''}{sep}1.50000"""
            # print(expected)

            assert result == expected
        else:
            tear_down(path)
            data.to_csv(path=path, n_files=n_files, order_by=order_by)
            titanic_test = vp.read_csv(f"{path}/*.csv")

            assert titanic_test.shape() == (1234, 14)

        tear_down(path)

    @pytest.mark.parametrize(
        "inplace,db_filter,nb_split,order_by,segmented_by",
        [
            (False, "age > 40", 3, None, None),
            (False, "age > 40", 3, {"age": "asc"}, None),
            (True, "age > 40", 3, None, ["age"]),
        ],
    )
    @pytest.mark.parametrize("relation_type", ["view", "table", "temporary", "local"])
    def test_to_db(
        self,
        schema_loader,
        titanic_vd_fun,
        relation_type,
        inplace,
        db_filter,
        nb_split,
        order_by,
        segmented_by,
    ):
        """
        test function - to_db
        """
        relation_name = f"verticapy_titanic_{relation_type}"
        data = titanic_vd_fun
        usecols = ["age", "fare", "survived"]
        catalog = "view_columns" if relation_type == "view" else "columns"
        schema = "v_temp_schema" if relation_type == "local" else schema_loader

        vp.drop(
            f"{schema}.{relation_name}",
            method="view" if relation_type == "view" else "table",
        )
        data.to_db(
            name=relation_name
            if relation_type == "local"
            else f"{schema}.{relation_name}",
            usecols=usecols,
            relation_type=relation_type,
            db_filter=db_filter,
            nb_split=nb_split,
            order_by=order_by,
            segmented_by=segmented_by,
        )
        titanic_tmp = vp.vDataFrame(f"{schema}.{relation_name}")
        assert titanic_tmp.shape() == (220, 4)
        assert titanic_tmp["_verticapy_split_"].min() == 0
        assert titanic_tmp["_verticapy_split_"].max() == 2
        current_cursor().execute(
            f"SELECT table_name FROM {catalog} WHERE table_schema='{schema}' and table_name = '{relation_name}'"
        )
        result = current_cursor().fetchone()
        assert result[0] == relation_name

    def test_to_geopandas(self, world_vd):
        """
        test function - to_geopandas
        """
        gdf = world_vd.to_geopandas(geometry="geometry")
        assert isinstance(gdf, geopandas.GeoDataFrame) and gdf.shape == (177, 4)

    @pytest.mark.parametrize(
        "idx, usecols, order_by, n_files",
        [
            (1, None, None, 1),
            (2, None, ["name", "age", "fare"], 4),
            (3, ["name", "fare"], ["name", "age", "fare"], 4),
        ],
    )
    def test_to_json(self, titanic_vd_fun, idx, usecols, order_by, n_files):
        """
        test function - to_json
        """
        path = f"/tmp/verticapy_test_to_json{idx}"
        columns = ["name", "fare"]
        data = titanic_vd_fun
        file_name = f"{path}.json"

        tear_down(path)
        if n_files == 1:
            raw_vdf = data.select(columns).sort({columns[0]: "asc", columns[1]: "desc"})
            vdf = raw_vdf[:2]

            vdf.to_json(
                path=file_name,
                n_files=n_files,
            )
            with open(file_name, "r") as file:
                result = file.read()
            # print(result)
            expected = '[\n{"name": "Abbing, Mr. Anthony", "fare": 7.55000},\n{"name": "Abbott, Master. Eugene Joseph", "fare": 20.25000}\n]'
            # print(expected)

            assert result == expected
        else:
            tear_down(path)
            data.to_json(path=path, usecols=usecols, n_files=n_files, order_by=order_by)
            titanic_test = vp.read_json(f"{path}/*.json")

            if usecols:
                assert titanic_test.shape() == (1234, 2)
            else:
                assert titanic_test.shape() == (1234, 14)

        tear_down(path)

    def test_to_list(self, titanic_vd_fun):
        """
        test function - to_list
        """
        result = titanic_vd_fun.select(["age", "survived"])[:20].to_list()
        assert isinstance(result, list) and len(result) == 20 and len(result[0]) == 2

    def test_to_numpy(self, titanic_vd_fun):
        """
        test function - to_numpy
        """
        result = titanic_vd_fun.select(["age", "survived"])[:20].to_numpy()
        assert isinstance(result, np.ndarray) and result.shape == (20, 2)

    def test_to_pandas(self, titanic_vd_fun):
        """
        test function - to_pandas
        """
        result = titanic_vd_fun.to_pandas()
        assert isinstance(result, pd.DataFrame) and result.shape == (1234, 14)

    @pytest.mark.parametrize(
        "by, order_by",
        [
            ("pclass", {"name": "asc"}),
            (None, None),
        ],
    )
    def test_to_parquet(self, titanic_vd_fun, by, order_by):
        """
        test function - to_parquet
        """
        session_id = f"{current_session()}_{username()}_{by}"
        name = "/tmp/parquet_test_{}".format(session_id)

        result = titanic_vd_fun.to_parquet(
            name, compression="gzip", by=by, order_by=order_by
        )
        assert result["Rows Exported"][0] == 1234

    def test_to_pickle(self, titanic_vd_fun):
        """
        test function - to_pickle
        """
        titanic_vd_fun.select(["age", "survived"])[:20].to_pickle("save.p")
        pickle.DEFAULT_PROTOCOL = 4
        result_tmp = pickle.load(open("save.p", "rb"))
        assert result_tmp.shape() == (20, 2)
        os.remove("save.p")

    @pytest.mark.parametrize(
        "usecols,overwrite,shape, expected",
        [
            (None, None, "Point", (202, 3)),
            (["country", "Geometry"], None, "Polygon", (1, 3)),
            (None, True, "MultiPolygon", (1, 5)),
        ],
    )
    def test_to_shp(self, cities_vd, world_vd, usecols, overwrite, shape, expected):
        """
        test function - to_shp
        """
        name = f"shp_test_{shape}"
        path = "/tmp/"

        tear_down(f"/tmp/{name}.shp")
        vp.drop(name=f"public.{name}")

        if shape == "Point":
            data = cities_vd
        else:
            world_vd_copy = world_vd.copy()
            data = world_vd_copy.filter(
                f"country = '{'Greenland' if shape == 'Polygon' else 'Antarctica'}'"
            )

        data.to_shp(
            name=name,
            path=path,
            usecols=usecols,
            overwrite=overwrite if overwrite or overwrite is False else True,
            shape=shape,
        )
        vdf = vp.read_shp(f"/tmp/{name}.shp")
        assert vdf.shape() == expected

        vp.drop(name=f"public.{name}")
