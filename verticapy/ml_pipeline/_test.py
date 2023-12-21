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
This script runs the Vertica Machine Learning Pipeline Test.
"""

from verticapy import vDataFrame

from ._helper import execute_and_add, remove_comments
from . import _metrics


def testing(test, model, pipeline_name, cols):
    meta_sql = ""
    dummy = model.predict(
        pipeline_name + "_TEST_VIEW", X=cols, name="prediction", inplace=True
    )

    # CREATE A PREDICT VIEW
    meta_sql += execute_and_add(
        f"CREATE OR REPLACE VIEW {pipeline_name + '_PREDICT_VIEW'} AS SELECT * FROM "
        + dummy.current_relation()
        + ";"
    )

    
    # FOR multiple metrics
    table_sql = ""
    table_sql += f"DROP TABLE IF EXISTS {pipeline_name + '_METRIC_TABLE'};"
    table_sql += execute_and_add(
        f"CREATE TABLE {pipeline_name + '_METRIC_TABLE'}(metric_name Varchar(100), metric FLOAT);"
    )

    for nth_metric in test:
        metric = test[nth_metric]
        name = metric["name"]
        y_true = metric["y_true"]
        y_score = metric["y_score"]
        temp = eval(
            f"""_metrics.{name}('{y_true}', '{y_score}',
            '{pipeline_name + '_PREDICT_VIEW'}')""",
            globals(),
        )
        sub = remove_comments(temp[:-1])

        col_name = vDataFrame(input_relation=temp).get_columns()[0]
        if col_name is None:
            table_sql += execute_and_add(
                f"INSERT INTO {pipeline_name + '_METRIC_TABLE'} SELECT '{name}', "
                + temp.split("*/")[1]
            )
        else:
            table_sql += execute_and_add(
                f"""INSERT INTO {pipeline_name + '_METRIC_TABLE'} 
                SELECT '{name}', subquery.{col_name} FROM ("""
                + sub
                + ") as subquery;"
            )

    table_sql += execute_and_add(
        "COMMIT;"
    )  # This is one transaction: Otherwise, Vertica discards all changes.
    meta_sql += table_sql
    return meta_sql, table_sql
