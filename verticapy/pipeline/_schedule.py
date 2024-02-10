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
This script runs the Vertica Machine Learning Pipeline Parser.
"""
from verticapy.pipeline._helper import execute_and_return, to_sql


def scheduler(
    schedule: dict, model_sql: str, table_sql: str, pipeline_name: str
) -> str:
    """
    Run the schedule step
    of the pipeline.

    Parameters
    ----------
    schedule: dict
        YAML object which outlines
        the steps of the operation.
    model_sql: str
        The SQL required to replicate
        the model training.
    table_sql: str
        The SQL required to replicate
        the metric table.
    pipeline_name: str
        The prefix name of the intended
        pipeline to unify the creation
        of the objects.

    Returns
    -------
    str
        The SQL to replicate the steps of the yaml file.

    Examples
    --------
    This example demonstrates how to use the
    `schedule` function to run the schedule
    step of a pipeline.

    Here you can use an existing relation.

    .. code-block:: python

        from verticapy.datasets import load_titanic
        load_titanic() # Loading the titanic dataset in Vertica

        import verticapy as vp
        vp.vDataFrame("public.titanic")

    Now you can feed in the parameters.

    .. code-block:: python

        from verticapy.pipeline._schedule import scheduler

        # Define the cron format schedule
        schedule =  "* * * * *"
        model_sql = "SELECT RF_REGRESSOR('public.pipeline_MODEL', 'public.titanic', 'survived', '"family_size", "fares", "sexes", "ages");"
        table_sql = 'public.titanic'

        # Define the pipeline name
        pipeline_name = "pipeline"

        # Call the scheduler function
        sql_query = scheduler(schedule, model_sql, table_sql, pipeline_name)

        # Execute the SQL query
        _executeSQL(sql_query)
    """
    meta_sql = ""
    model_string = to_sql(model_sql)
    table_string = to_sql(table_sql)

    meta_sql += execute_and_return(
        f"CREATE SCHEDULE {pipeline_name + '_ML_SCHEDULE'} USING CRON '{schedule}';"
    )
    if table_sql != "":
        meta_sql += execute_and_return(
            f"""CREATE OR REPLACE PROCEDURE {pipeline_name + '_ML_RUNNER'}() AS
            $$ BEGIN EXECUTE 'DROP MODEL IF EXISTS {pipeline_name + '_MODEL'};
            {model_string} EXECUTE ' {table_string} END; $$;\n"""
        )
    else:
        meta_sql += execute_and_return(
            f"""CREATE OR REPLACE PROCEDURE {pipeline_name + '_ML_RUNNER'}() AS 
            $$ BEGIN EXECUTE 'DROP MODEL IF EXISTS {pipeline_name + '_MODEL'};
            {model_string} END; $$;\n"""
        )
    meta_sql += execute_and_return(
        f"""CREATE TRIGGER {pipeline_name + '_TRAIN'} ON SCHEDULE
        {pipeline_name + '_ML_SCHEDULE'} EXECUTE PROCEDURE {pipeline_name + '_ML_RUNNER'}()
        AS DEFINER;"""
    )

    return meta_sql
