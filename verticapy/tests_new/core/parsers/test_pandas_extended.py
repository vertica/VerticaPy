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

# Pytest
import pytest

# Standard Python Modules
import time

from vertica_python.errors import CopyRejected

# Other Modules
import pandas

# VerticaPy
from verticapy import (
    drop,
)
from verticapy.connection import current_cursor
from verticapy.core.parsers.pandas import read_pandas
from verticapy.datasets import load_titanic


class TestPandasExtended:
    def test_read_pandas_abort_on_error(self, titanic_vd):
        """
        Tries to use read_pandas() to load a dataframe into a table
        that has the right column names, but the wrong column type
        for the data format. Asserts that abort_on_error behaves as
        expected.
        """
        pandas_df = titanic_vd.to_pandas()
        assert pandas_df.shape == (1234, 14)
        random_name = f"titanic_hack_{int(time.time())}"
        try:
            current_cursor().execute(
                f"create table public.{random_name} like"
                f" {titanic_vd} excluding projections"
            ).fetchall()
            current_cursor().execute(
                f'alter table public.{random_name} drop column "survived"'
            ).fetchall()
            current_cursor().execute(
                f'alter table public.{random_name} add column "survived" interval'
            ).fetchall()
            with pytest.raises(CopyRejected):
                read_pandas(
                    df=pandas_df,
                    name=random_name,
                    schema="public",
                    insert=True,
                    abort_on_error=True,
                )
        finally:
            current_cursor().execute(
                f"drop table if exists public.{random_name}"
            ).fetchall()
