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
from verticapy._utils._sql._format import clean_query, erase_comment


def is_longvar(ctype: str) -> bool:
    """
    Returns ``True`` if the input SQL
    ctype is a long varbinary or long
    varchar.

    Parameters
    ----------
    ctype: str
        Category type.

    Returns
    -------
    bool
        ``True`` if it is long varbinary
        or long varchar.

    Examples
    --------
    The following code demonstrates
    the usage of the function.

    .. ipython:: python

        # Import the function.
        from verticapy._utils._sql._check import is_longvar

        # Function examples.
        is_longvar('long varchar(600)')
        is_longvar('integer')

    .. note::

        These functions serve as utilities to
        construct others, simplifying the overall
        code.
    """
    return str(ctype).lower().startswith(("long varbina", "long varchar"))


def is_dql(query: str) -> bool:
    """
    Returns ``True`` if the input SQL
    query is a DQL statement (SELECT).

    Parameters
    ----------
    query: str
        SQL Query.

    Returns
    -------
    bool
        ``True`` if it is a DQL statement.

    Examples
    --------
    The following code demonstrates
    the usage of the function.

    .. ipython:: python

        # Import the function.
        from verticapy._utils._sql._check import is_dql

        # Generating a queries.
        query1 = "SELECT col1, SUM(col2) FROM my_table GROUP BY 1;"
        query2 = "INSERT INTO t SELECT * FROM my_table;"

        # Function examples.
        is_dql(query1)
        is_dql(query2)

    .. note::

        These functions serve as utilities to
        construct others, simplifying the overall
        code.
    """
    result = False
    query = clean_query(query)
    query = erase_comment(query)
    for idx, q in enumerate(query):
        if q not in (" ", "("):
            result = (
                query[idx:]
                .lower()
                .startswith(
                    (
                        "select ",
                        "with ",
                    )
                )
            )
            break
    return result


def is_procedure(query: str) -> bool:
    """
    Returns ``True`` if the input
    SQL query is a procedure.

    Parameters
    ----------
    query: str
        SQL Query.

    Returns
    -------
    bool
        ``True`` if it is a store procedure.

    Examples
    --------
    The following code demonstrates
    the usage of the function.

    .. ipython:: python

        # Import the function.
        from verticapy._utils._sql._check import is_procedure

        # Generating queries.
        query1 = "SELECT col1, SUM(col2) FROM my_table GROUP BY 1;"
        query2 = "CREATE OR REPLACE PROCEDURE t(...)"

        # Function examples.
        is_procedure(query1)
        is_procedure(query2)

    .. note::

        These functions serve as utilities to
        construct others, simplifying the overall
        code.
    """
    result = False
    query = clean_query(query)
    query = erase_comment(query)
    for idx, q in enumerate(query):
        if q not in (" ", "("):
            result = (
                query[idx:]
                .lower()
                .startswith(
                    (
                        "create procedure ",
                        "create or replace procedure ",
                    )
                )
            )
            break
    return result
