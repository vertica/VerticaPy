#!/usr/bin/env python3
"""This script runs the Vertica Machine Learning Pipeline Parser"""

from ._helper import execute_and_add, to_sql


def schedule(schedule, model_sql, table_sql, pipeline_name):
    meta_sql = ""
    model_string = to_sql(model_sql)
    table_string = to_sql(table_sql)

    meta_sql += execute_and_add(
        f"CREATE SCHEDULE {pipeline_name + '_ML_SCHEDULE'} USING CRON '{schedule}';"
    )
    if table_sql != "":
        meta_sql += execute_and_add(
            f"""CREATE OR REPLACE PROCEDURE {pipeline_name + '_ML_RUNNER'}() AS
            $$ BEGIN EXECUTE 'DROP MODEL IF EXISTS {pipeline_name + '_MODEL'};
            {model_string} EXECUTE ' {table_string} END; $$;\n"""
        )
    else:
        meta_sql += execute_and_add(
            f"""CREATE OR REPLACE PROCEDURE {pipeline_name + '_ML_RUNNER'}() AS 
            $$ BEGIN EXECUTE 'DROP MODEL IF EXISTS {pipeline_name + '_MODEL'};
            {model_string} END; $$;\n"""
        )
    meta_sql += execute_and_add(
        f"""CREATE TRIGGER {pipeline_name + '_TRAIN'} ON SCHEDULE
        {pipeline_name + '_ML_SCHEDULE'} EXECUTE PROCEDURE {pipeline_name + '_ML_RUNNER'}()
        AS DEFINER;"""
    )

    return meta_sql
