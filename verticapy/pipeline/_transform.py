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
This script runs the Vertica Machine Learning Pipeline Transforming.
"""
from queue import Queue

from verticapy.core.vdataframe.base import vDataFrame


def reset_queue(column_queue: Queue) -> Queue:
    """
    Return a new Queue where all
    columns haven't been tried.
    """
    new_queue = Queue()
    while not column_queue.empty():
        col, _ = column_queue.get()
        new_queue.put((col, False))
    return new_queue


def transformation(transform: dict, table: str) -> vDataFrame:
    """
    This function takes a set of transforms,
    each containing a column name and instructions,
    and attempts to create the columns in the given
    table, according to the given instructions while
    handling exceptions and cyclic dependencies.

    Parameters
    ----------
    transform: dict
        YAML object which outlines
        the steps of the operation.
    table: str
        The name of the table the
        pipeline is ingesting to.

    Returns
    -------
    vDataFrame
        The transformed ``vDataFrame``.

    Description
    -----------
    - The function processes the columns in
      a queue-based approach, allowing for
      columns to be created in any order.
    - It tries to execute the instructions
      for each column to create the respective
      column.
    - If an exception occurs during column
      creation, it logs the error message in
      the 'error_string', moves the column to
      the back of the queue, and marks it as
      flagged.
    - When a column is successfully created,
      the error_string is reset, and all column
      flags are cleared in the queue.
    - If the next column in the queue has been
      flagged, it indicates that either errors
      persist or cyclic dependencies exist,
      in which case the error_string contains
      relevant error information.

    Example
    -------
    Here you can use an existing relation.

    .. code-block:: python

        from verticapy.datasets import load_titanic
        load_titanic() # Loading the titanic dataset in Vertica

        import verticapy as vp
        vp.vDataFrame("public.titanic")

    Now you can apply a transform.

    .. code-block:: python

        from verticapy.pipeline._transform import transformation

        # Define the transformation steps in a YAML object
        transform = {
                'family_size': {
                        'sql': 'parch+sibsp+1'
                },
                'fares': {
                        'sql': 'fare',
                        'transform_method': {
                                'name': 'fill_outliers',
                                'params': {
                                        'method': 'winsorize',
                                        'alpha': 0.03
                                }
                        }
                },
                'sexes': {
                        'sql': 'sex',
                        'transform_method': {
                                'name': 'label_encode'
                        }
                },
                'ages': {
                        'sql': 'age',
                        'transform_method': {
                                'name': 'fillna',
                                'params': {
                                        'method': 'mean',
                                        'by': ['pclass', 'sex']
                                }
                        }
                }
        }

        # Specify the target table for transformation
        table = "public.titanic"

        # Call the transformation function
        transformed_vdf = transformation(transform, table)

        # Display the transformed vDataFrame
        print(transformed_vdf)
    """
    vdf = vDataFrame(table)
    column_queue = Queue()
    for col in sorted(list(transform.keys())):
        column_queue.put((col, False))

    error_string = ""
    while not column_queue.empty():
        col, is_dep = column_queue.get()
        if is_dep:
            raise RuntimeError(
                "Error: All remaining columns either have an error or cyclic dependencies:\n"
                + error_string
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
                    temp_str += f'{param} = "{params[param]}", '
                else:
                    temp_str += f"{param} = {params[param]}, "
            temp_str = temp_str[:-2]
            name = tf["name"]

            try:
                if not is_created:
                    eval(f"vdf.{name}(" + temp_str + f", name='{col}')")
                else:
                    try:
                        eval(f"vdf['{col}'].{name}({temp_str})", locals())
                    except Exception as e:
                        vdf = eval(f"vdf.{name}({temp_str})", locals())
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
