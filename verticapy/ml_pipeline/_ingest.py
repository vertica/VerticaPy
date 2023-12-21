#!/usr/bin/env python3
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
"""
This script runs the Vertica Machine Learning Pipeline Ingestion
"""

from ._helper import required_keywords, execute_and_add


def ingestion(ingest, pipeline_name, table):
    meta_sql = ""
    if required_keywords(ingest, ["from"]):
        data_loader_sql = ""
        retry_limit = (
            "DEFAULT" if "retry_limit" not in ingest else ingest["retry_limit"]
        )
        retention_interval = (
            "'14 days'"
            if "retention_interval" not in ingest
            else ingest["retention_interval"]
        )
        data_loader_sql += f"""CREATE DATA LOADER {pipeline_name + '_DATALOADER'}
        RETRY LIMIT {retry_limit} RETENTION INTERVAL {retention_interval} """

        data_loader_sql += f"AS COPY {table} FROM '{ingest['from']}' "
        if "delimiter" in ingest:
            data_loader_sql += f"DELIMITER '{ingest['delimiter']}' "
        data_loader_sql += "DIRECT;"
        meta_sql += execute_and_add(data_loader_sql)
        meta_sql += execute_and_add(
            f"EXECUTE DATA LOADER {pipeline_name + '_DATALOADER'};"
        )

        if "schedule" in ingest:
            meta_sql += execute_and_add(
                f"""CREATE SCHEDULE {pipeline_name + '_DL_SCHEDULE'}
                USING CRON '{ingest['schedule']}';"""
            )
            meta_sql += execute_and_add(
                f"""CREATE PROCEDURE {pipeline_name + '_DL_RUNNER'}() AS 
                $$ BEGIN EXECUTE 'EXECUTE DATA LOADER {pipeline_name + '_DATALOADER'}'; END; $$;"""
            )
            meta_sql += execute_and_add(
                f"""CREATE TRIGGER {pipeline_name + '_LOAD'} ON SCHEDULE 
                {pipeline_name + '_DL_SCHEDULE'} EXECUTE PROCEDURE {pipeline_name + '_DL_RUNNER'}() 
                AS DEFINER;"""
            )
    return meta_sql
