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
import logging
from typing import Set

import pytest

import verticapy as vp
from verticapy._utils._sql._sys import _executeSQL
from verticapy.performance.vertica.collection.profile_import import ProfileImport, ProfileImportError
from verticapy.datasets import load_amazon
from verticapy.core.vdataframe import vDataFrame


class TestProfileImport:
    """
    Test Base Class.
    """

    def test_empty_schema(self, schema_loader):
        pi = ProfileImport(target_schema=schema_loader,
                           key='no_such_key',
                           filename='no_such_file.tar',
                           skip_create_table=True)
        with pytest.raises(ProfileImportError, 
                           match=f"Missing [0-9]+ tables in schema {schema_loader}"):
            pi.check_schema()

    def test_no_such_schema(self):
        pi = ProfileImport(target_schema='no_such_schema',
                           key='no_such_key',
                           filename='no_such_file.tar',
                           skip_create_table=True)
        
        with pytest.raises(ProfileImportError, 
                           match=f"Schema no_such_schema does not exist"):
            pi.check_schema()

    def test_create_tables(self, schema_loader):
        test_key = 'aaa'
        pi = ProfileImport(target_schema=schema_loader,
                           key=test_key,
                           filename='no_such_file.tar')
        # creates tables if they don't exist
        pi.check_schema()
        
        should_be_created = ["qprof_collection_events_aaa"]
        created_tables = self._get_set_of_tables_in_schema(schema_loader, test_key)

        missing_tables = []
        for tname in should_be_created:
            if tname not in created_tables:
                missing_tables.append(tname)

        assert len(missing_tables) == 0, f"Failed to create tables: [{','.join(missing_tables)}]"

    @staticmethod
    def _get_set_of_tables_in_schema(target_schema:str, key:str) -> Set[str]:
        print("Looking for tables in schema: {target_schema}")
        result = _executeSQL(f"""SELECT table_name FROM v_catalog.tables 
                    WHERE 
                        table_schema = '{target_schema}'
                        and table_name ilike '%_{key}';
                    """,
                    method='fetchall')
        existing_tables = set()
        for row in result:
            print(f"created table {row[0]}")
            existing_tables.add(row[0])
        return existing_tables 
    