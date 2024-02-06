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
import logging
from typing import Set

from verticapy._utils._sql._sys import _executeSQL

from .collection_tables import getAllCollectionTables, AllTableTypes


class ProfileImportError(Exception):
    pass


class ProfileImport:
    """
    Loads data from a profile collection into a running database.
    Can create schemas and tables to load the profile into.
    """

    def __init__(
        self,
        target_schema: str,
        key: str,
        filename: str,
        skip_create_table: bool = False,
    ) -> None:
        """
        Load a query performance profile in `filename` into tables
        in schema `target_schema` with suffix `key`.

        `target_schema`: the schema to load the profile into
        `key`
        """
        # TODO: at this time, we don't run any checks automatically
        # during __init__. Instead, we run the chceks manually. Running
        # the checks manually facilitates early testing.

        self.target_schema = target_schema
        self.key = key
        self.filename = filename
        self.input_file_obj = None
        self.skip_create_table = skip_create_table
        self.logger = logging.getLogger("ProfileImport")

    def check_file(self) -> None:
        """
        Checks to see that the file exists and that we can open it
        """
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"File {self.filename} does not exist")

        # Recall that open will raise errors if
        # the file does not exist
        self.input_file_obj = open(self.filename, "r")

    def check_schema(self) -> None:
        """
        Checks to see that the schema and expected tables exist.
        Optionally creates the schema and expected tables.
        """
        if self.skip_create_table:
            self._schema_exists_or_raise()
            self._tables_exist_or_raise()
            return
        self._create_schema_if_not_exists()
        self._create_tables_if_not_exists()

    def _schema_exists_or_raise(self) -> None:
        result = _executeSQL(
            f"""SELECT COUNT(*) FROM v_catalog.schemata 
                    WHERE schema_name = '{self.target_schema}';
                    """,
            method="fetchall",
        )
        if result[0][0] < 1:
            raise ProfileImportError(f"Schema {self.target_schema} does not exist")

    def _tables_exist_or_raise(self) -> None:
        tables_in_schema = self._get_set_of_tables_in_schema()
        all_tables = getAllCollectionTables(
            target_schema=self.target_schema, key=self.key
        )
        missing_tables = []
        for ctable in all_tables.values():
            search_name = ctable.get_import_name()
            if search_name not in tables_in_schema:
                missing_tables.append(search_name)
        if len(missing_tables) == 0:
            return

        # At least one table is missing
        # Throw an execption
        raise ProfileImportError(
            f"Missing {len(missing_tables)} tables"
            f" in schema {self.target_schema}."
            f" Missing: [ {','.join(missing_tables)} ]"
        )

    def _get_set_of_tables_in_schema(self) -> Set[str]:
        result = _executeSQL(
            f"""SELECT table_name FROM v_catalog.tables 
                    WHERE 
                        table_schema = '{self.target_schema}'
                        and table_name ilike '%_{self.key}';
                    """,
            method="fetchall",
        )
        existing_tables = set()
        for row in result:
            existing_tables.add(row[0])
        return existing_tables

    def _create_schema_if_not_exists(self) -> None:
        _executeSQL(
            f"""CREATE SCHEMA IF NOT EXISTS {self.target_schema};
                    """
        )

    def _create_tables_if_not_exists(self) -> None:
        all_tables = getAllCollectionTables(
            target_schema=self.target_schema, key=self.key
        )
        for ctable in all_tables.values():
            self.logger.info(f"Running create statements for {ctable.name}")
            table_sql = ctable.get_create_table_sql()
            proj_sql = ctable.get_create_projection_sql()
            _executeSQL(table_sql, method="fetchall")
            _executeSQL(proj_sql, method="fetchall")
