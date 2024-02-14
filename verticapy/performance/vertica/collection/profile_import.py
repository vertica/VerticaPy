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
import tarfile
from pathlib import Path
import random
from typing import Set, List

import pandas as pd

from verticapy._utils._sql._sys import _executeSQL
from verticapy.core.vdataframe import vDataFrame

from .collection_tables import getAllCollectionTables, AllTableTypes, BundleVersion


class ProfileImportError(Exception):
    pass


class ProfileImport:
    """
    ProfileImport loads data from a profile collection into a running database.
    Can create schemas and tables to load the profile into.
    """

    def __init__(
        self,
        target_schema: str,
        key: str,
        filename: Path,
    ) -> None:
        """
        Load a query performance profile in ``filename`` into tables
        in schema ``target_schema`` with suffix ``key``.

        Parameters
        ------------
        target_schema: str
            The schema to load the profile into. The schema will
            be created unless ``skip_create_table`` is ``True``.
        key: str
            The suffix for tables to store profiling data.
        filename: str
            The tarball of parquet files to load into the database.
        """

        self.target_schema = target_schema
        self.key = key
        self.filename = filename
        self.logger = logging.getLogger("ProfileImport")

        # initialize internal attributes
        self.tarfile_obj = None
        self.bundle_version = None
        self.raise_when_missing_files = False
        self.skip_create_table = False
        self.tmp_path = os.getcwd()

    @property
    def skip_create_table(self) -> bool:
        """
        ProfileImport will SKIP running ddl statements to create new
        tables when skip_creat_table is True. Otherwise it will run
        DDL statements. Default value is False.

        The DDL statements are:
           CREATE TABLE ... IF NOT EXISTS
           CREATE PROJECTION ... IF NOT EXISTS
        """
        return self._skip_create_table

    @skip_create_table.setter
    def skip_create_table(self, val: bool) -> None:
        if not isinstance(val, bool):
            raise TypeError(
                f"Cannot set skip_create_table to value of type {type(val)}. Must be type bool."
            )
        self._skip_create_table = val

    @property
    def raise_when_missing_files(self) -> bool:
        """
        ProfileImport will raise and exception if a bundle lacks
        any of the expected files and raise_when_missing_files is True.
        Otherwise, it will write a warning to the log. Default value is False.
        """
        return self._raise_when_missing_files

    @raise_when_missing_files.setter
    def raise_when_missing_files(self, val: bool) -> None:
        if not isinstance(val, bool):
            raise TypeError(
                f"Cannot set raise_when_missing_files to value of type {type(val)}. Must be type bool."
            )
        self._raise_when_missing_files = val

    @property
    def tmp_path(self) -> Path:
        """
        ProfileImport will raise and exception if a bundle lacks
        any of the expected files and raise_when_missing_files is True.
        Otherwise, it will write a warning to the log. Default value is False.
        """
        return self._tmp_path

    @tmp_path.setter
    def tmp_path(self, val: os.PathLike) -> None:
        if not isinstance(val, str) and not isinstance(val, os.PathLike):
            raise TypeError(
                f"Cannot set tmp_dir to value of type {type(val)}. Must be type string or PathLike."
            )
        self._tmp_path = val

    def check_file(self) -> None:
        """
        Checks to see that the file exists and that we can open it
        """
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"File {self.filename} does not exist")

        unpack_dir = self._unpack_bundle()
        self.bundle_version = self._calculate_bundle_version(unpack_dir)
        self._check_for_missing_files(unpack_dir, self.bundle_version)
        self._load_vdataframes(unpack_dir, self.bundle_version)

    def check_schema(self) -> None:
        """
        Checks to see that the schema and expected tables exist.
        Optionally creates the schema and expected tables.
        """
        if self.bundle_version is None:
            self.bundle_version = BundleVersion.LATEST
            self.logger.info(
                f"Set bunlde version to latest ({self.bundle_version}) because"
                f"it was not set previously"
            )
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
            target_schema=self.target_schema, key=self.key, version=self.bundle_version
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
            target_schema=self.target_schema, key=self.key, version=self.bundle_version
        )
        for ctable in all_tables.values():
            self.logger.info(f"Running create statements for {ctable.name}")
            table_sql = ctable.get_create_table_sql()
            proj_sql = ctable.get_create_projection_sql()
            _executeSQL(table_sql, method="fetchall")
            _executeSQL(proj_sql, method="fetchall")

    def _unpack_bundle(self) -> Path:
        """
        Unpacks the bundle into a temp directory

        Returns
        --------
        Returns a path
        """

        self.tarfile_obj = tarfile.open(self.filename, "r")
        names = "\n".join(self.tarfile_obj.getnames())
        print(f"Files in the archive: {names}")
        # There are other ways to generate a
        tmp_dir = self.tmp_path / Path(
            f"profile_import_run{random.randint(10000, 20000)}"
        )

        print(f"Creating temporary directory: {tmp_dir}")

        tmp_dir.mkdir()

        self.tarfile_obj.extractall(tmp_dir)

        print(f"Extracted files: {[x for x in tmp_dir.iterdir()]}")

        return tmp_dir

    def _calculate_bundle_version(self, unpack_dir: Path) -> BundleVersion:
        metadata_file = unpack_dir / "profile_metadata.json"
        if not metadata_file.exists():
            self.logger.info(f"Did not find metadata file {metadata_file}")
            return BundleVersion.V1

        return BundleVersion.LATEST

    def _check_for_missing_files(
        self, unpack_dir: Path, version: BundleVersion
    ) -> None:
        unpacked_files = set([x for x in unpack_dir.iterdir()])
        missing_files = []

        all_tables = getAllCollectionTables(
            target_schema=self.target_schema, key=self.key, version=version
        )
        for ctable in all_tables.values():
            expected_file_path = unpack_dir / Path(ctable.get_parquet_file_name())
            if expected_file_path not in unpacked_files:
                missing_files.append(expected_file_path)

        self._handle_missing_files(unpack_dir, missing_files)

    def _handle_missing_files(
        self, unpack_dir: Path, missing_files: List[Path]
    ) -> None:
        if len(missing_files) == 0:
            return
        message = (
            f"Bundle {self.filename} unpacked in directory {unpack_dir}"
            f" lacks {len(missing_files)} files: {','.join([str(f) for f in missing_files])}"
        )
        if not self.raise_when_missing_files:
            self.logger.warning(message)
            return
        raise ImportError(message)

    def _load_vdataframes(self, unpack_dir: Path, version: BundleVersion) -> None:
        unpacked_files = set([x for x in unpack_dir.iterdir()])

        all_tables = getAllCollectionTables(
            target_schema=self.target_schema, key=self.key, version=version
        )
        for ctable in all_tables.values():
            expected_file_path = unpack_dir / Path(ctable.get_parquet_file_name())
            if expected_file_path not in unpacked_files:
                self.logger.info(f"Skipping missing file {expected_file_path}")
                continue
            pd_dataframe = pd.read_parquet(expected_file_path)
            cols = list(pd_dataframe.columns)
            self.logger.info(f"File {expected_file_path} has columns {cols}")
            # Next PR: load the vdataframe into the database
