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
import os
import tarfile

import pytest

from verticapy.performance.vertica import QueryProfiler
from verticapy.performance.vertica.collection.profile_import import (
    ProfileImport,
)
from verticapy._utils._sql._sys import _executeSQL


class TestQueryProfilerSimple:
    """
    Contains tests related to creating bundles of parquet data
    """

    TEST_QUERY_TEMPLATE = """ 
    SELECT 
        date, 
        MONTH(date) AS month, 
        AVG(number) AS avg_number
    """

    def test_profile_export(self, amazon_vd, schema_loader, tmp_path):
        """
        Produces a export bundle and asserts that the list of files in
        the bundle matches the expected list of files.
        """

        request = f"""
        {self.TEST_QUERY_TEMPLATE}
        FROM 
            {amazon_vd}
        GROUP BY 1;
        """

        qp = QueryProfiler(request, target_schema=schema_loader)

        outfile = tmp_path / "qprof_test_001.tar"
        logging.info(f"Writing to file: {outfile}")
        qp.export_profile(filename=outfile)
        logging.info(
            f"Files in tmpdir: {','.join([str(x) for x in tmp_path.iterdir()])}"
        )
        assert os.path.exists(outfile)

        tf = tarfile.open(outfile)

        tarfile_contents = set(tf.getnames())

        expected_files = set(
            [
                "dc_explain_plans.parquet",
                "dc_query_executions.parquet",
                "dc_requests_issued.parquet",
                "execution_engine_profiles.parquet",
                "host_resources.parquet",
                "query_consumption.parquet",
                "query_plan_profiles.parquet",
                "query_profiles.parquet",
                "resource_pool_status.parquet",
                "profile_metadata.json",
            ]
        )

        # Recall: symmetric difference is all elements that are in just
        # one set.
        set_diff = tarfile_contents.symmetric_difference(expected_files)

        assert len(set_diff) == 0

    def _get_set_of_tables_in_schema(self, target_schema, key):
        """
        Returns a set of table names in a schema that have the
        suffix key.
        """
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
            existing_tables.add(row[0])
        return existing_tables

    def test_high_level_import_export(self, amazon_vd, schema_loader, tmp_path):
        """
        Produces a export bundle and then imports the bundle. Checks to see that the
        tables are loaded.
        """
        request = f"""
        {self.TEST_QUERY_TEMPLATE}
        FROM 
            {amazon_vd}
        GROUP BY 1;
        """

        qp = QueryProfiler(request, target_schema=schema_loader)

        outfile = tmp_path / "qprof_test_002.tar"
        logging.info(f"Writing to file: {outfile}")
        qp.export_profile(filename=outfile)
        assert os.path.exists(outfile)
        key = "reload789"
        new_qp = QueryProfiler.import_profile(
            target_schema=schema_loader,
            key_id=key,
            filename=outfile,
            tmp_dir=tmp_path,
        )

        assert new_qp is not None
        # We don't validate the qplan tree in this test. We just want to ensure that it is
        # produced.
        new_qp.get_qplan_tree()

        # Also check that the tables were created ok
        loaded_tables = self._get_set_of_tables_in_schema(schema_loader, key)
        expected_tables = [
            f"qprof_dc_explain_plans_{key}",
            f"qprof_dc_query_executions_{key}",
            f"qprof_dc_requests_issued_{key}",
            f"qprof_execution_engine_profiles_{key}",
            f"qprof_host_resources_{key}",
            f"qprof_query_consumption_{key}",
            f"qprof_query_plan_profiles_{key}",
            f"qprof_query_profiles_{key}",
            f"qprof_resource_pool_status_{key}",
        ]
        for t in expected_tables:
            assert t in loaded_tables
            result = _executeSQL(
                f"select count(*) from {schema_loader}.{t}", method="fetchall"
            )
            assert result[0][0] > 0
