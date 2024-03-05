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


class TestQueryProfilerSimple:
    """
    Contains tests related to creating bundles of parquet data
    """

    def test_profile_export(self, amazon_vd, schema_loader, tmp_path):
        logging.info(f"Amazon vd is {amazon_vd}")
        request = f"""
        SELECT 
            date, 
            MONTH(date) AS month, 
            AVG(number) AS avg_number 
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

    def test_profile_full_lifecycle(self, amazon_vd, schema_loader, tmp_path):
        request = f"""
        SELECT 
            date, 
            MONTH(date) AS month, 
            AVG(number) AS avg_number 
        FROM 
            {amazon_vd}
        GROUP BY 1;
        """

        qp = QueryProfiler(request, target_schema=schema_loader)

        outfile = tmp_path / "qprof_test_002.tar"
        logging.info(f"Writing to file: {outfile}")
        qp.export_profile(filename=outfile)
        assert os.path.exists(outfile)

        pi = ProfileImport(
            target_schema=schema_loader,
            key="reload123",
            filename=outfile,
        )
        unpack_tmp = tmp_path / "unpack"
        unpack_tmp.mkdir()
        pi.tmp_path = unpack_tmp
        pi.check_schema_and_load_file()

    def test_high_level_import_export(self, amazon_vd, schema_loader, tmp_path):
        request = f"""
        SELECT 
            date, 
            MONTH(date) AS month, 
            AVG(number) AS avg_number 
        FROM 
            {amazon_vd}
        GROUP BY 1;
        """

        qp = QueryProfiler(request, target_schema=schema_loader)

        outfile = tmp_path / "qprof_test_002.tar"
        logging.info(f"Writing to file: {outfile}")
        qp.export_profile(filename=outfile)
        assert os.path.exists(outfile)

        new_qp = QueryProfiler.import_profile(
            target_schema=schema_loader,
            key_id="reload789",
            filename=outfile,
            tmp_dir=tmp_path,
        )

        assert new_qp is not None
        # We don't validate the qplan tree in this test. We just want to ensure that it is
        # produced.
        new_qp.get_qplan_tree()
