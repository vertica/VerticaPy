#!/usr/bin/env python3

"""This script runs the Vertica Machine Learning Pipeline Transforming"""


from queue import Queue

import verticapy as vp


def reset_queue(column_queue):
    new_queue = Queue()
    while not column_queue.empty():
        col, _ = column_queue.get()
        new_queue.put((col, False))
    return new_queue


def transformation(transform, table):
    vdf = vp.vDataFrame(table)

    column_queue = Queue()
    for col in sorted(list(transform.keys())):
        column_queue.put((col, False))

    error_string = ""
    while not column_queue.empty():
        col, is_dep = column_queue.get()
        if is_dep:
            raise RuntimeError(
                "Error: All remaining columns have dependencies\n" + error_string
            )

        column_info = transform[col]
        is_created = False

        if "sql" in column_info:
            try:
                vdf.eval(name=col, expr=column_info["sql"])
                is_created = True
            except Exception as e:
                error_string += f"Error creating {col} in sql: {e}\n"
                column_queue.put((col, True))
                continue

        methods = list(column_info.keys())
        methods = sorted([method for method in methods if "transform_method" in method])
        for method in methods:
            tf = column_info[method]
            params = tf["params"] if "params" in tf else []
            temp_str = ""
            for param in params:
                temp = params[param]
                if isinstance(temp, str):
                    temp_str += f"{param} = '{params[param]}', "
                else:
                    temp_str += f"{param} = {params[param]}, "
            temp_str = temp_str[:-2]
            name = tf["name"]

            try:
                if not is_created:
                    eval(f"vdf.{name}(" + temp_str + f", name='{col}')")
                else:
                    # print(getattr(vdf,'regexp'))
                    eval(f"vdf['{col}'].{name}({temp_str})", locals())
            except Exception as e:
                error_string += f"Error creating {col} in methods: {e}\n"
                if is_created:
                    eval(f"vdf['{col}'].drop()", locals())
                column_queue.put((col, True))
                is_created = False
                break
        if is_created:
            error_string = ""
            column_queue = reset_queue(column_queue)
    return vdf
