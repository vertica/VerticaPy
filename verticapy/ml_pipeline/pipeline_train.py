#!/usr/bin/env python3

"""This script runs the Vertica Machine Learning Pipeline Training"""

from verticapy.machine_learning.vertica import *
from verticapy.machine_learning.vertica.linear_model import *
from verticapy.machine_learning.vertica.svm import *
from verticapy.machine_learning.vertica.tree import *
from verticapy.machine_learning.vertica.neighbors import *
from verticapy.machine_learning.vertica.naive_bayes import *
from verticapy.machine_learning.vertica.tsa import *
from verticapy.machine_learning.vertica.decomposition import *
from verticapy.machine_learning.vertica.cluster import *

from .pipeline_helper import execute_and_add


def training(train, vdf, pipeline_name, cols):
    meta_sql = ""
    if "train_test_split" in train:
        info = train["train_test_split"]
        test_size = 0.33 if "test_size" not in info else info["test_size"]
        train_set, test_set = vdf.train_test_split(test_size)
        meta_sql += execute_and_add(
            f"CREATE OR REPLACE VIEW {pipeline_name + '_TRAIN_VIEW'} AS SELECT * FROM "
            + train_set.current_relation()
            + ";"
        )
        meta_sql += execute_and_add(
            f"CREATE OR REPLACE VIEW {pipeline_name + '_TEST_VIEW'} AS SELECT * FROM "
            + test_set.current_relation()
            + ";"
        )

    else:
        meta_sql += execute_and_add(
            f"CREATE OR REPLACE VIEW {pipeline_name + '_TRAIN_VIEW'} AS SELECT * FROM "
            + vdf.current_relation()
            + ";"
        )
        meta_sql += execute_and_add(
            f"CREATE OR REPLACE VIEW {pipeline_name + '_TEST_VIEW'} AS SELECT * FROM "
            + vdf.current_relation()
            + ";"
        )

    methods = list(train.keys())
    methods = [method for method in methods if "method" in method]
    for method in methods:
        tf = train[method]
        temp_str = ""
        name = tf["name"]
        target = tf["target"] if "target" in tf else ""
        params = tf["params"] if "params" in tf else []
        temp_str = ""
        for param in params:
            temp = params[param]
            if isinstance(temp, str):
                temp_str += f"{param} = '{params[param]}', "
            else:
                temp_str += f"{param} = {params[param]}, "
        temp_str = temp_str[:-2]
        eval(
            f"exec(\"model = {name}('{pipeline_name + '_MODEL'}', {temp_str})\")",
            globals(),
        )
        predictors = cols  # ['"col1"', '"col2"', '"col3"', '"col4"']
        if "include" in tf:
            predictors = tf["include"]
        if "exclude" in tf:
            predictors = list(set(predictors) - set(tf["exclude"]))
        if target == "":
            # UNSUPERVISED
            model.fit(pipeline_name + "_TRAIN_VIEW", predictors)
        else:
            # SUPERVISED
            model.fit(pipeline_name + "_TRAIN_VIEW", predictors, target)
        meta_sql += "\n"

        try:
            # SUPERVISED
            model_sql = model.get_vertica_attributes("call_string")["call_string"][0]
            if model_sql.split(" ")[0] != "SELECT":
                model_sql = "SELECT " + model_sql + ";"
        except vertica_python.errors.QueryError:
            # UNSUPERVISED
            model_sql = (
                "SELECT "
                + model.get_vertica_attributes("metrics")["metrics"][0]
                .split("Call:")[1]
                .replace("\n", " ")
                + ";"
            )
        meta_sql += model_sql + "\n"
    return meta_sql, model, model_sql
