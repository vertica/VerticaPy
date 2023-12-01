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
import datetime
from typing import Literal, Optional, Union

import numpy as np


def to_varchar(
    category: str,
    column: str = "{}",
) -> str:
    """
    Uses the correct SQL function to
    convert the input column to VARCHAR.

    Parameters
    ----------
    category: str
        Data Type category.
    column: str, optional
        Column to format.

    Returns
    -------
    str
        correctly formatted column.

    Examples
    --------
    The following code demonstrates
    the usage of the function.

    .. ipython:: python

        # Import the function.
        from verticapy._utils._sql._cast import to_varchar

        # binary
        to_varchar(
            category = 'binary',
            column = 'col',
        )

        # vmap
        to_varchar(
            category = 'vmap',
            column = 'col',
        )

        # spatial
        to_varchar(
            category = 'spatial',
            column = 'col',
        )

        # text
        to_varchar(
            category = 'text',
            column = 'col',
        )

    .. note::

        These functions serve as utilities to
        construct others, simplifying the overall
        code.
    """
    map_dict = {
        "vmap": f"MAPTOSTRING({column})",
        "binary": f"TO_HEX({column})",
        "spatial": f"ST_AsText({column})",
    }
    if category in map_dict:
        return map_dict[category]
    return column


def to_dtype_category(
    expr: type,
) -> Literal["float", "int", "text", "date", "complex", "undefined"]:
    """
    Returns the category associated
    with the Python input type.

    Parameters
    ----------
    expr: type
        Data Type.

    Returns
    -------
    str
        VerticaPy category.

    Examples
    --------
    The following code demonstrates
    the usage of the function.

    .. ipython:: python

        # Import the function.
        from verticapy._utils._sql._cast import to_dtype_category

        # float
        to_dtype_category(float)

        # list
        to_dtype_category(list)

        # dict
        to_dtype_category(dict)

        # str
        to_dtype_category(str)

    .. note::

        These functions serve as utilities to
        construct others, simplifying the overall
        code.
    """
    if hasattr(expr, "category"):
        category = expr.category()
    else:
        if isinstance(expr, float):
            category = "float"
        elif isinstance(expr, int):
            category = "int"
        elif isinstance(expr, str):
            category = "text"
        elif isinstance(expr, (datetime.date, datetime.datetime)):
            category = "date"
        elif isinstance(expr, (dict, list, np.ndarray)):
            category = "complex"
        else:
            category = "undefined"
    return category


def to_sql_dtype(dtype: Union[type, str]) -> Union[type, str]:
    """
    Returns the SQL type associated
    to the input Python type.

    Parameters
    ----------
    dtype: type | str
        Data Type to format.

    Returns
    -------
    type | str
        Correctly Formatted Data Type.

    Examples
    --------
    The following code demonstrates
    the usage of the function.

    .. ipython:: python

        # Import the function.
        from verticapy._utils._sql._cast import to_sql_dtype

        # float
        to_sql_dtype(float)

        # list
        to_sql_dtype(list)

        # dict
        to_sql_dtype(dict)

    .. note::

        These functions serve as utilities to
        construct others, simplifying the overall
        code.
    """
    if dtype in (str, "str", "string"):
        dtype = "varchar"
    elif dtype == float:
        dtype = "float"
    elif dtype == int:
        dtype = "integer"
    elif dtype == datetime.datetime:
        dtype = "datetime"
    elif dtype == datetime.date:
        dtype = "date"
    elif dtype == datetime.time:
        dtype = "time"
    elif dtype == datetime.timedelta:
        dtype = "interval"
    elif dtype == datetime.timezone:
        dtype = "timestamptz"
    elif dtype in (np.ndarray, np.array, list):
        dtype = "array"
    elif dtype == dict:
        dtype = "row"
    elif dtype == tuple:
        dtype = "set"
    elif isinstance(dtype, str):
        dtype = dtype.lower().strip()
    return dtype


def to_category(
    ctype: Optional[str] = None,
) -> Literal[
    "text",
    "int",
    "float",
    "date",
    "binary",
    "uuid",
    "vmap",
    "spatial",
    "complex",
    "undefined",
]:
    """
    Returns the category associated
    to the input SQL type.

    Parameters
    ----------
    ctype: str, optional
        Vertica Column Data Type.

    Returns
    -------
    str
        Column category.

    Examples
    --------
    The following code demonstrates
    the usage of the function.

    .. ipython:: python

        # Import the function.
        from verticapy._utils._sql._cast import to_category

        # array
        to_category('array')

        # bigint
        to_category('bigint')

        # geometry
        to_category('geometry')

        # varchar
        to_category('varchar')

    .. note::

        These functions serve as utilities to
        construct others, simplifying the overall
        code.
    """
    ctype = str(ctype).lower().strip()
    if ctype != "":
        if ctype.startswith(("array", "row", "set")):
            return "complex"
        elif ctype.startswith(("date", "interval", "smalldatetime", "time")):
            return "date"
        elif ctype.startswith(("bigint", "bool", "int", "smallint", "tinyint")):
            return "int"
        elif ctype.startswith(("decimal", "double", "float", "money", "num", "real")):
            return "float"
        elif ctype.startswith("geo"):
            return "spatial"
        elif ctype.startswith(("binary", "byte", "raw")):
            return "binary"
        elif ctype.startswith("uuid"):
            return "uuid"
        elif ctype.startswith("vmap"):
            return "vmap"
        else:
            return "text"
    else:
        return "undefined"
