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

from verticapy.performance.vertica import QueryProfiler



class TestQueryProfilerSimple:
    """
    Test Base Class.
    """
    def test_profile_export(self, amazon_vd, schema_loader, tmp_path):
        request = f"""
        SELECT 
            date, 
            MONTH(date) AS month, 
            AVG(number) AS avg_number 
        FROM 
            {schema_loader}.amazon 
        GROUP BY 1;
        """

        qp = QueryProfiler(request,
                        target_schema=schema_loader)
        
        # Assert that the tables exist?

        outfile = tmp_path / "qprof_test_001.tar"
        logging.info(f"Writing to file: {outfile}")
        qp.export_profile(filename=outfile)
        logging.info(f"Files in tmpdir: {','.join([str(x) for x in tmp_path.iterdir()])}")
        assert os.path.exists(outfile)

        unpack_dir = tmp_path / "unpack_test_profile_export"
        tf = tarfile.open(outfile)

        expected_files = set(["dc_explain_plans.parquet",
            "dc_query_executions.parquet",
            "dc_requests_issued.parquet",
            "execution_engine_profiles.parquet",
            "host_resources.parquet",
            "query_consumption.parquet",
            "query_plan_profiles.parquet",
            "query_profiles.parquet",
            "resource_pool_status.parquet",
            "metadata.json"])

        for fname in tf.getnames():
            assert fname in expected_files
            