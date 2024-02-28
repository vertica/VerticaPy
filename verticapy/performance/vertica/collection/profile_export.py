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
import logging
import os
from pathlib import Path
import tarfile
from typing import Set, List, Mapping

import pandas as pd

from verticapy._utils._sql._sys import _executeSQL
from verticapy.core.vdataframe import vDataFrame
from verticapy.core.parsers.pandas import read_pandas

from verticapy.performance.vertica.collection.collection_tables import (
    AllTableTypes,
    BundleVersion,
    CollectionTable,
    getAllCollectionTables,
)


class ProfileExportError(Exception):
    pass


class ProfileExport:
    def __init__(
        self,
        target_schema: str,
        key: str,
        filename: Path,
    ) -> None:
        """
        Store common parameters for profile export
        """
        if not isinstance(target_schema, str):
            raise TypeError(f"Expected target_schema to have type str but found type {type(target_schema)}")
        self.target_schema = target_schema
        self.key = key
        self.filename = filename
        self.logger = logging.getLogger("ProfileExport")
        self.bundle_version = BundleVersion.LATEST

        self.tmp_path = os.getcwd()

    @property
    def tmp_path(self) -> Path:
        """
        ``ProfileImport`` creates temporary directories in ``tmp_path``
        when unpacking bundles.
        """
        return self._tmp_path

    @tmp_path.setter
    def tmp_path(self, val: os.PathLike) -> None:
        if not isinstance(val, str) and not isinstance(val, os.PathLike):
            raise TypeError(
                f"Cannot set tmp_dir to value of type {type(val)}. Must be type string or PathLike."
            )
        self._tmp_path = val

    def export(self):
        """
        Export the tables in  ``self.schema`` with ``self.key`` into ``self.filename``
        """
        self._tables_exist_or_raise()

    def _tables_exist_or_raise(self) -> None:
        tables_in_schema = self._get_set_of_tables_in_schema()
       
        missing_tables = []
        
        all_tables = self._get_modified_collection_tables()
        for ctable in all_tables.values():
            search_name = ctable.get_import_name()
            if search_name not in tables_in_schema:
                missing_tables.append(search_name)
        if len(missing_tables) == 0:
            return

        # At least one table is missing
        # Throw an execption
        raise ProfileExportError(
            f"Missing {len(missing_tables)} tables"
            f" in schema {self.target_schema}."
            f" Missing: [ {','.join(missing_tables)} ]"
        )
    
    def _get_modified_collection_tables(self) -> Mapping[str, CollectionTable]:
        """
        Returns a mapping of names to CollectionTable instances.
        Filters out some tables that do not current exist because they
        are not created during collection
        """
        all_tables = getAllCollectionTables(
            target_schema=self.target_schema, 
            key=self.key, 
            version=self.bundle_version
        )
        result = {}
        for ctable in all_tables.values():
            if (ctable.table_type == AllTableTypes.COLLECTION_INFO
                or ctable.table_type == AllTableTypes.COLLECTION_EVENTS
                or ctable.table_type == AllTableTypes.EXPORT_EVENTS):
                continue
            result[ctable.table_type.name] = ctable
        return result

    
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
        