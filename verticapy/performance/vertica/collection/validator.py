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

from typing import List, Mapping
from io import StringIO

from verticapy._utils._sql._sys import _executeSQL
from verticapy.performance.vertica.collection.collection_tables import CollectionTable, AllTableNames


class CollectionValidator:
    """Confirms that a schema contains a collection of tables that can be used by the `QueryProfiler`."""

    def __init__(self, target_schema:str, key:str) -> None:
        """
        `target_schema`: the schema where we will search for tables

        """
        self.target_schema = target_schema
        self.key = key

        self._do_validation()

    def _do_validation(self) -> None:
        current_tables = self._get_all_tables()
        
        self._check_tables_exist(current_tables)
        self._check_empty_tables(current_tables)

    def _check_tables_exist(self, tables: Mapping[str, CollectionTable]) -> None:
        obsv_tables = _executeSQL(f"""select table_name from v_catalog.tables 
                                  where table_schema = '{self.target_schema}'
                                  """,
                              method="fetchall")
        obsv_map = set(x[0] for x in obsv_tables)

        missing_tables = []
        for t in tables.values():
            import_name = t.get_import_name()
            if import_name not in obsv_map:
                missing_tables.append(import_name)

        if len(missing_tables) > 0:
            raise ValueError(f"Missing tables in schema {self.target_schema}: {missing_tables}")

        
            
    def _get_all_tables(self) -> Mapping[str, CollectionTable]:
        current_tables = {}
        for name in AllTableNames:
            name_str = name.value
            c = CollectionTable(name_str, self.target_schema, self.key)
            current_tables[name.name] = c

        return current_tables
    
    def _check_empty_tables(self, tables: Mapping[str, CollectionTable]) -> None:
        empty_tables = []
        count_union_query = StringIO()
        for i, table in enumerate(tables.values()):
            import_name = table.get_import_name_fq()
            if i == 0:
                count_union_query.write(f"""select 
                                            '{import_name}' as table_name, 
                                            count(*)
                                        from {import_name}
                                        """)
            else:
                count_union_query.write(f"""UNION ALL (
                                        select 
                                            '{import_name}' as table_name, 
                                            count(*)
                                        from {import_name} )
                                        """)

        row_count_table = _executeSQL(count_union_query.getvalue(),
                            method="fetchall")
        print(f"Row count for all tables is: {row_count_table}")
        for r in row_count_table:
            table_name = r[0]
            row_count = r[1]
            print(f"Table name : {r[0]} rows {r[1]}")
                
        if len(empty_tables) > 0:
            raise ValueError(f"Expected 0 empty tables, but found {len(empty_tables)} : {','.join(empty_tables)}")
            
    





