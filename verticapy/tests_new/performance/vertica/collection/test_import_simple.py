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
from pathlib import Path
import shutil
from typing import Set

import pytest
from verticapy._utils._sql._sys import _executeSQL
from verticapy.performance.vertica.collection.profile_import import (
    ProfileImport,
    ProfileImportError,
)


class TestProfileImport:
    """
    Collection of tests for ProfileImport class
    """

    @pytest.fixture(scope="class")
    def tmp_path_with_test_bundles(self, tmp_path_factory):
        test_package_dir = Path(__file__).parent
        class_tmp_path = tmp_path_factory.mktemp("test_profile_import")
        logging.info(f"tmp_path_with_test_bundles is {class_tmp_path}")
        for f in test_package_dir.iterdir():
            if f.match("*.tar"):
                shutil.copy(f, class_tmp_path)
        yield class_tmp_path
        # No cleanup to do: tmp_path_factory will do it for us

    def test_empty_schema(self, schema_loader):
        """Confirm that the profile import fails when schema is present but
        tables are missing, and the user specifies not to create the tables
        """
        pi = ProfileImport(
            target_schema=schema_loader,
            key="no_such_key",
            filename="no_such_file.tar",
        )
        pi.skip_create_table = True
        with pytest.raises(
            ProfileImportError, match=f"Missing [0-9]+ tables in schema {schema_loader}"
        ):
            pi.check_schema()

    def test_no_such_schema(self):
        """
        Confirm that profile import fails when schema is absent and
        the user specifies not to create the schema.
        """
        pi = ProfileImport(
            target_schema="no_such_schema",
            key="no_such_key",
            filename="no_such_file.tar",
        )
        pi.skip_create_table = True
        with pytest.raises(
            ProfileImportError, match=f"Schema no_such_schema does not exist"
        ):
            pi.check_schema()

    def test_missing_bundle(self):
        """
        Confirm ProfileImport raises when user specifies a file that does not exist
        """
        fname = "no_such_file.tar"
        pi = ProfileImport(
            target_schema="schema_not_used", key="no_such_key", filename=fname
        )
        pi.skip_create_table = True
        with pytest.raises(FileNotFoundError, match=f"File {fname} does not exist"):
            pi.check_file()

    def test_untar_file(self, tmp_path_with_test_bundles):
        """
        Untars a valid file, identifies the version, and checks for
        missing parquet files.
        """

        fname = tmp_path_with_test_bundles / "feb01_cqvs_ndv20.tar"
        pi = ProfileImport(
            target_schema="schema_not_used",
            key="no_such_key",
            filename=fname,
        )
        pi.skip_create_table = True
        pi.raise_when_missing_files = True
        pi.tmp_path = tmp_path_with_test_bundles
        # check_file shouldn't raise any errors because the input is valid
        pi.check_file()

    def test_untar_incomplete_file(self, tmp_path_with_test_bundles):
        """
        Confirm ProfileImport raises when user specifies a file that does not exist
        """

        fname = tmp_path_with_test_bundles / "feb01_cqvs_missing_parquet.tar"
        pi = ProfileImport(
            target_schema="schema_not_used",
            key="no_such_key",
            filename=fname,
        )
        pi.skip_create_table = True
        pi.raise_when_missing_files = False
        pi.tmp_path = tmp_path_with_test_bundles
        # check_file was configured to log warnings instead of printing errors
        pi.check_file()

        # Now test that we raise appropriately
        pi2 = ProfileImport(
            target_schema="schema_not_used",
            key="no_such_key",
            filename=fname,
        )
        pi2.skip_create_table = True
        pi2.tmp_path = tmp_path_with_test_bundles
        pi2.raise_when_missing_files = True
        with pytest.raises(ProfileImportError, match=f"Bundle .* lacks [0-9]+ files"):
            pi2.check_file()

    def test_create_tables_copy_data(self, schema_loader, tmp_path_with_test_bundles):
        fname = tmp_path_with_test_bundles / "feb20_demo_djr_v03.tar"

        pi = ProfileImport(
            # schema and target will be once this test copies
            # files into a schema
            target_schema=schema_loader,
            key="test123",
            filename=fname,
        )
        pi.skip_create_table = False
        pi.raise_when_missing_files = True
        pi.tmp_path = tmp_path_with_test_bundles
        pi.check_schema_and_load_file()

        tables_and_rows = [
            ("qprof_collection_events_test123", 20),
            ("qprof_collection_info_test123", 2),
            ("qprof_dc_explain_plans_test123", 191),
            ("qprof_dc_query_executions_test123", 138),
            ("qprof_dc_requests_issued_test123", 2),
            ("qprof_execution_engine_profiles_test123", 63373),
            ("qprof_export_events_test123", 12),
            ("qprof_host_resources_test123", 12),
            ("qprof_query_consumption_test123", 2),
            ("qprof_query_plan_profiles_test123", 179),
            ("qprof_query_profiles_test123", 2),
            ("qprof_resource_pool_status_test123", 216),
        ]

        for table, row in tables_and_rows:
            result = _executeSQL(
                f"""select count(*) from {schema_loader}.{table}""", method="fetchall"
            )
            assert len(result) == 1
            assert (
                result[0][0] == row
            ), f"table {table} expected {row} observed {result[0][0]}"

    def test_create_tables(self, schema_loader):
        """
        Confirm that all of the expected tables are created in an
        existing schema.
        """
        test_key = "aaa"
        pi = ProfileImport(
            target_schema=schema_loader, key=test_key, filename="no_such_file.tar"
        )
        # creates tables if they don't exist
        pi.check_schema()

        # The list below test checks the latest version of the tables
        # is created.
        should_be_created = [
            "qprof_dc_explain_plans_aaa",
            "qprof_dc_query_executions_aaa",
            "qprof_dc_requests_issued_aaa",
            "qprof_execution_engine_profiles_aaa",
            "qprof_host_resources_aaa",
            "qprof_query_consumption_aaa",
            "qprof_query_plan_profiles_aaa",
            "qprof_query_profiles_aaa",
            "qprof_resource_pool_status_aaa",
        ]
        created_tables = self._get_set_of_tables_in_schema(schema_loader, test_key)

        missing_tables = []
        for tname in should_be_created:
            if tname not in created_tables:
                missing_tables.append(tname)

        assert (
            len(missing_tables) == 0
        ), f"Failed to create tables: [{','.join(missing_tables)}]"

    def test_skip_create_table_property(self):
        pi = ProfileImport(
            target_schema="no_such_schema", key="any_key", filename="no_such_file.tar"
        )
        pi.raise_when_missing_files = True
        with pytest.raises(TypeError):
            pi.raise_when_missing_files = "not valid to set to string"

        pi.skip_create_table = True
        with pytest.raises(TypeError):
            pi.skip_create_table = "cannot be set to string"

        pi.tmp_path = "/tmp"
        with pytest.raises(TypeError):
            pi.tmp_path = None

    @staticmethod
    def _get_set_of_tables_in_schema(target_schema: str, key: str) -> Set[str]:
        """
        Test utility method: get all the qprof tables in a schema
        """
        logging.info(f"Looking for tables in schema: {target_schema}")
        result = _executeSQL(
            f"""SELECT table_name FROM v_catalog.tables 
                    WHERE 
                        table_schema = '{target_schema}'
                        and table_name ilike '%_{key}';
                    """,
            method="fetchall",
        )
        existing_tables = set()
        for row in result:
            logging.info(f"created table {row[0]}")
            existing_tables.add(row[0])
        return existing_tables
