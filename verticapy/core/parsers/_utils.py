"""
Copyright  (c)  2018-2025 Open Text  or  one  of its
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
import os
from typing import Literal


def extract_col_dt_from_query(query: str, field: str) -> tuple:
    """
    Extracts the column's data
    type from the INFER_DDL
    generated SQL statement.

    Parameters
    ----------
    query: str
        SQL query.
    field: str
        Field to extract.

    Returns
    -------
    tuple
        column's data type.

    Examples
    --------
    The following code demonstrates
    the usage of the function.

    .. ipython:: python

        # Import the function.
        from verticapy.core.parsers._utils import extract_col_dt_from_query

        # SQL INFER_DDL Query
        sql = 'create table "restaurants"('
        sql += '\n"chain" bool,'
        sql += '\n"cuisine" varchar,'
        sql += '\n"hours" Array[UNKNWON],'
        sql += '\n"location_city" Array[varchar],'
        sql += '\n"menu" Array[Row('
        sql += '\n"item" varchar,'
        sql += '\n"price" numeric'
        sql += '\n)],'
        sql += '\n"name" varchar'
        sql += '\n);"'

        # Example
        extract_col_dt_from_query(sql, field = 'location_city')

    .. note::

        These functions serve as utilities to
        construct others, simplifying the overall
        code.
    """
    n, m = len(query), len(field) + 2
    for i in range(n - m):
        current_word = query[i : i + m]
        if current_word.lower() == '"' + field.lower() + '"':
            i = i + m
            total_parenthesis = 0
            k = i + 1
            while ((query[i] != ",") or (total_parenthesis > 0)) and i < n - m:
                i += 1
                if query[i] in ("(", "[", "{"):
                    total_parenthesis += 1
                elif query[i] in (")", "]", "}"):
                    total_parenthesis -= 1
            return current_word, query[k:i]


def extract_compression(
    path: str,
) -> Literal["GZIP", "BZIP", "LZO", "ZSTD", "UNCOMPRESSED"]:
    """
    Extracts and returns the
    compression extension.

    Parameters
    ----------
    path: str
        File name.

    Returns
    -------
    str
        File extension.

    Examples
    --------
    The following code demonstrates
    the usage of the function.

    .. ipython:: python

        # Import the function.
        from verticapy.core.parsers._utils import extract_compression

        # GZIP
        extract_compression('my_file.gz')

        # BZIP
        extract_compression('my_file.bz')

        # UNCOMPRESSED
        extract_compression('my_file')

    .. note::

        These functions serve as utilities to
        construct others, simplifying the overall
        code.
    """
    file_extension = path.split(".")[-1].lower()
    lookup_table = {"gz": "GZIP", "bz": "BZIP", "lz": "LZO", "zs": "ZSTD"}
    if file_extension[0:2] in lookup_table:
        return lookup_table[file_extension[0:2]]
    else:
        return "UNCOMPRESSED"


def get_first_file(path: str, ext: str) -> str:
    """
    Returns the first file
    having the input extension.

    Parameters
    ----------
    path: str
        Folder path.
    ext: str
        File extension.

    Returns
    -------
    str
        File name.

    Examples
    --------
    The following code demonstrates
    the usage of the function.

    .. code-block:: python

        # Import the function.
        from verticapy.core.parsers._utils import get_first_file

        # Example
        get_first_file('my_path', 'csv')

        # -> It will return the first CSV file
        #### in the 'my_path' folder.

    .. note::

        These functions serve as utilities to
        construct others, simplifying the overall
        code.
    """
    directory_name = os.path.dirname(path)
    files = os.listdir(directory_name)
    for f in files:
        file_ext = f.split(".")[-1]
        if file_ext == ext:
            return directory_name + "/" + f
