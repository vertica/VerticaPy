#!/usr/bin/env python3

"""This script runs the Vertica Machine Learning Pipeline Parser"""

# base package imports
import yaml
import argparse
from tqdm import tqdm

# required packages
import verticapy as vp
from verticapy.datasets import *
from verticapy._utils._sql._sys import _executeSQL

# local
from .pipeline_helper import required_keywords, execute_and_add, setup
from . import pipeline_ingest
from . import pipeline_transform
from . import pipeline_train
from . import pipeline_test
from . import pipeline_schedule

parser = argparse.ArgumentParser(
    description="""Vertica Pipelines is an open source platform for
    managing data scientists machine learning pipelines.
    They are built on a human-readable data format: YAML."""
)
parser.add_argument("connection_file", help="Path to the connection yaml file")
parser.add_argument("input_file", help="Path to the input yaml file")
parser.add_argument(
    "-o",
    "--outfile",
    nargs="?",
    help="[Optional] Path to the output sql file. If unspecified input_file.sql will be used.",
    default=None,
    required=False,
)

args = parser.parse_args()
config_name = args.connection_file
file_name = args.input_file
output_name = args.outfile

with open(config_name, "r", encoding="utf-8") as file:
    connect = yaml.safe_load(file)
    if required_keywords(connect, ["host", "port", "database", "password", "user"]):
        vp.new_connection(
            {
                "host": connect["host"],
                "port": connect["port"],
                "database": connect["database"],
                "password": connect["password"],
                "user": connect["user"],
            },
            name="temp",
        )
        vp.connect("temp")


with open(file_name, "r", encoding="utf-8") as file:
    pipeline = yaml.safe_load(file)
    schema_name = pipeline["schema"]
    recipe = pipeline["recipe"]
    pipeline_name = schema_name + "." + recipe
    table = pipeline["table"]
    META_SQL = ""

    setup()

    # DROP OLD PIPELINE
    META_SQL += execute_and_add(f"CALL drop_pipeline('{schema_name}', '{recipe}');")

    # PUBLIC LOAD OF A TABLE
    table_split = table.split(".")
    if len(table_split) == 2 and table_split[0] == "public":
        try:
            eval(f'exec("load_{table_split[1]}()")', globals())
        except:
            pass

    # EXECUTE PIPELINE BLUEPRINT
    if "steps" in pipeline:
        steps = pipeline["steps"]
        step_count = len(steps.keys())
        pbar = tqdm(total=step_count)
        if "ingest" in steps:
            ingest = steps["ingest"]
            META_SQL += pipeline_ingest.ingestion(ingest, pipeline_name, table)
            pbar.update()

        VDF = None
        if "transform" in steps:
            transform = steps["transform"]
            VDF = pipeline_transform.transformation(transform, table)
            if "train" not in steps:
                META_SQL += execute_and_add(
                    f"CREATE OR REPLACE VIEW {pipeline_name + '_VIEW'} AS SELECT * FROM "
                    + VDF.current_relation()
                    + ";"
                )
            pbar.update()

        COLS = None
        MODEL = None
        MODEL_SQL = ""
        if "train" in steps:
            train = steps["train"]
            if VDF is None:
                VDF = vp.vDataFrame(table)
                COLS = VDF.get_columns()
            else:
                COLS = list(transform.keys())
            train_sql, MODEL, MODEL_SQL = pipeline_train.training(
                train, VDF, pipeline_name, COLS
            )
            META_SQL += train_sql
            pbar.update()

        TABLE_SQL = ""
        if "test" in steps:
            test = steps["test"]
            TEST_SQL, TABLE_SQL = pipeline_test.testing(test, MODEL, pipeline_name, COLS)
            META_SQL += TEST_SQL
            pbar.update()
            print(
                _executeSQL(
                    f"SELECT * FROM {pipeline_name + '_METRIC_TABLE'};"
                ).fetchall()
            )

        if "train" in steps and "schedule" in steps["train"]:
            schedule = steps["train"]["schedule"]
            META_SQL += pipeline_schedule.schedule(
                schedule, MODEL_SQL, TABLE_SQL, pipeline_name
            )

        if output_name is None:
            output_name = pipeline_name + ".sql"

        with open(output_name, "w", encoding="utf-8") as file:
            file.write(META_SQL)
