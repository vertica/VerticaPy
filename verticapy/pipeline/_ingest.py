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
from verticapy._utils._sql._sys import _executeSQL

from verticapy.pipeline._helper import (
    execute_and_return,
    is_valid_delimiter,
    required_keywords,
)


def ingestion(ingest: dict, pipeline_name: str, table: str) -> str:
    """
    Run the ingestion step
    of the pipeline.

    Parameters
    ----------
    ingestion: dict
        YAML object which outlines
        the steps of the operation.
    pipeline_name: str
        The prefix name of the intended
        pipeline to unify the creation of
        the objects.
    table: str
        The name of the table the pipeline
        is ingesting to.

    Returns
    -------
    str
        The SQL to replicate the
        steps of the yaml file.

    Examples
    --------
    This example demonstrates how to use the
    `ingestion` function to run the ingestion
    step of a pipeline.

    .. code-block:: python

        from your_module import ingestion

        # Define the ingestion steps in a YAML object
        ingest = {
            'from': '~/data/bucket/*',
            'delimiter': ',',
            'retry_limit': 'NONE',
            'retention_interval': "'15 days'",
        }

        # Define the pipeline name
        pipeline_name = "my_pipeline"

        # Specify the target table for ingestion
        table = "my_table"

        # Call the ingestion function
        sql_query = ingestion(ingest, pipeline_name, table)

        # Execute the SQL query
        _executeSQL(sql_query)
    """
    meta_sql = ""
    if required_keywords(ingest, ["from"]):
        data_loader_sql = ""
        retry_limit = (
            "DEFAULT" if "retry_limit" not in ingest else ingest["retry_limit"]
        )
        if retry_limit != "DEFAULT" and retry_limit != "NONE":
            # Type Check
            _executeSQL(f"SELECT INT {retry_limit};")

        retention_interval = (
            "'14 days'"
            if "retention_interval" not in ingest
            else ingest["retention_interval"]
        )
        # Type Check
        _executeSQL(f"SELECT INTERVAL {retention_interval}")
        data_loader_sql += f"""CREATE DATA LOADER {pipeline_name + '_DATALOADER'}
        RETRY LIMIT {retry_limit} RETENTION INTERVAL {retention_interval} """

        data_loader_sql += f"AS COPY {table} FROM '{ingest['from']}' "
        if "delimiter" in ingest:
            data_loader_sql += f"DELIMITER '{ingest['delimiter']}' "
            # Type Check
            if not is_valid_delimiter(ingest["delimiter"]):
                raise TypeError(
                    "Delimiter must have an ASCII value in the range E'\\000' to E'\\177' inclusive"
                )
        data_loader_sql += "DIRECT;"
        meta_sql += execute_and_return(data_loader_sql)
        meta_sql += execute_and_return(
            f"EXECUTE DATA LOADER {pipeline_name + '_DATALOADER'};"
        )

        if "schedule" in ingest:
            meta_sql += execute_and_return(
                f"""CREATE SCHEDULE {pipeline_name + '_DL_SCHEDULE'}
                USING CRON '{ingest['schedule']}';"""
            )
            meta_sql += execute_and_return(
                f"""CREATE PROCEDURE {pipeline_name + '_DL_RUNNER'}() AS 
                $$ BEGIN EXECUTE 'EXECUTE DATA LOADER {pipeline_name + '_DATALOADER'}'; END; $$;"""
            )
            meta_sql += execute_and_return(
                f"""CREATE TRIGGER {pipeline_name + '_LOAD'} ON SCHEDULE 
                {pipeline_name + '_DL_SCHEDULE'} EXECUTE PROCEDURE {pipeline_name + '_DL_RUNNER'}() 
                AS DEFINER;"""
            )
    return meta_sql
