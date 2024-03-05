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
    ExportMetadata,
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
        self.filename = filename if isinstance(filename, Path) else Path(filename)
        self.logger = logging.getLogger("ProfileExport")

        # Usually the version of export will be "LATEST"
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
        export_metadata = self._save_tables()
        self._bundle_tables(export_metadata)

    def _tables_exist_or_raise(self) -> None:
        tables_in_schema = self._get_set_of_tables_in_schema()
       
        missing_tables = []
        
        all_tables = getAllCollectionTables(self.target_schema, 
                                            self.key, 
                                            self.bundle_version)
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
    
    def _save_tables(self) -> ExportMetadata:
        """
        """
        all_tables = getAllCollectionTables(self.target_schema,
                                            self.key,
                                            self.bundle_version)
        tmp_path = self.filename.parent
        all_table_metadata = []
        for ctable in all_tables.values():
            exported_meta = ctable.export_table(tmp_path)
            all_table_metadata.append(exported_meta)

        metadata_file = tmp_path / "profile_metadata.json"
        export_metadata = ExportMetadata(metadata_file,
                                         self.bundle_version,
                                         all_table_metadata)
        
        export_metadata.write_to_file()

        return export_metadata

    def _bundle_tables(self, export_metadata: ExportMetadata):
        """
        """
        
        self.tarfile_obj = tarfile.open(self.filename, 'w')
        for t in export_metadata.tables:
            self.tarfile_obj.add(t.file_name, 
                                 arcname=t.file_name.name)
        self.tarfile_obj.add(export_metadata.file_name,
                             arcname=export_metadata.file_name.name)
        self.tarfile_obj.close()
        for t in export_metadata.tables:
            os.remove(t.file_name)
        
        # Everything went ok, clean it up
        os.remove(export_metadata.file_name)


        