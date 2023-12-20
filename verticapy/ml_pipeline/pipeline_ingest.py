#!/usr/bin/env python3

"""This script runs the Vertica Machine Learning Pipeline Ingestion"""

from pipeline_helper import required_keywords, execute_and_add


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
