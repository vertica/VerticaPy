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
from typing import Optional

import verticapy._config.config as conf
from verticapy._utils._sql._format import quote_ident

from verticapy.core.vdataframe.base import vDataFrame

from verticapy.sql.create import create_table
from verticapy.sql.drop import drop
from verticapy.sql.insert import insert_into

"""
Sample Datasets to do testing.
"""


def load_dataset_cl(
    schema: Optional[str] = None,
    table_name: str = "dataset_cl",
) -> vDataFrame:
    """
    Sample Dataset to do classification.
    """
    if not (schema):
        schema = conf.get_option("temp_schema")

    # Classification Dataset

    data = [
        [1, "Bus", "Male", 0, "Cheap", "Low"],
        [2, "Bus", "Male", 1, "Cheap", "Med"],
        [3, "Train", "Female", 1, "Cheap", "Med"],
        [4, "Bus", "Female", 0, "Cheap", "Low"],
        [5, "Bus", "Male", 1, "Cheap", "Med"],
        [6, "Train", "Male", 0, "Standard", "Med"],
        [7, "Train", "Female", 1, "Standard", "Med"],
        [8, "Car", "Female", 1, "Expensive", "Hig"],
        [9, "Car", "Male", 2, "Expensive", "Med"],
        [10, "Car", "Female", 2, "Expensive", "Hig"],
    ]
    input_relation = f"{quote_ident(schema)}.{quote_ident(table_name)}"

    drop(name=input_relation, method="table")
    create_table(
        table_name=table_name,
        schema=schema,
        dtype={
            "Id": "INT",
            "transportation": "VARCHAR",
            "gender": "VARCHAR",
            "owned cars": "INT",
            "cost": "VARCHAR",
            "income": "CHAR(4)",
        },
    )
    insert_into(table_name=table_name, schema=schema, data=data, copy=False)

    return vDataFrame(input_relation=input_relation)


def load_dataset_reg(
    schema: Optional[str] = None,
    table_name: str = "dataset_reg",
) -> vDataFrame:
    """
    Sample Dataset to do regression.
    """
    if not (schema):
        schema = conf.get_option("temp_schema")

    # Regression Dataset

    data = [
        [1, 0, "Male", 0, "Cheap", "Low"],
        [2, 0, "Male", 1, "Cheap", "Med"],
        [3, 1, "Female", 1, "Cheap", "Med"],
        [4, 0, "Female", 0, "Cheap", "Low"],
        [5, 0, "Male", 1, "Cheap", "Med"],
        [6, 1, "Male", 0, "Standard", "Med"],
        [7, 1, "Female", 1, "Standard", "Med"],
        [8, 2, "Female", 1, "Expensive", "Hig"],
        [9, 2, "Male", 2, "Expensive", "Med"],
        [10, 2, "Female", 2, "Expensive", "Hig"],
    ]
    input_relation = f"{quote_ident(schema)}.{quote_ident(table_name)}"

    drop(name=input_relation, method="table")
    create_table(
        table_name=table_name,
        schema=schema,
        dtype={
            "Id": "INT",
            "transportation": "INT",
            "gender": "VARCHAR",
            "owned cars": "INT",
            "cost": "VARCHAR",
            "income": "CHAR(4)",
        },
    )
    insert_into(table_name=table_name, schema=schema, data=data, copy=False)

    return vDataFrame(input_relation=input_relation)


def load_dataset_num(
    schema: Optional[str] = None,
    table_name: str = "dataset_num",
) -> vDataFrame:
    """
    Sample Dataset with numerical values.
    """
    if not (schema):
        schema = conf.get_option("temp_schema")

    # Numerical Dataset

    data = [
        [1, 7.2, 3.6, 6.1, 2.5],
        [2, 7.7, 2.8, 6.7, 2.0],
        [3, 7.7, 3.0, 6.1, 2.3],
        [4, 7.9, 3.8, 6.4, 2.0],
        [5, 4.4, 2.9, 1.4, 0.2],
        [6, 4.6, 3.6, 1.0, 0.2],
        [7, 4.7, 3.2, 1.6, 0.2],
        [8, 6.5, 2.8, 4.6, 1.5],
        [9, 6.8, 2.8, 4.8, 1.4],
        [10, 7.0, 3.2, 4.7, 1.4],
    ]
    input_relation = f"{quote_ident(schema)}.{quote_ident(table_name)}"

    drop(name=input_relation, method="table")
    create_table(
        table_name=table_name,
        schema=schema,
        dtype={
            "Id": "INT",
            "col1": "FLOAT",
            "col2": "FLOAT",
            "col3": "FLOAT",
            "col4": "FLOAT",
        },
    )
    insert_into(table_name=table_name, schema=schema, data=data, copy=False)

    return vDataFrame(input_relation=input_relation)
