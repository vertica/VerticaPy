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
    Returns True if the input SQL ctype is
    a long varbinary or long varchar.
    """
    return ctype.lower().startswith(("long varbina", "long varchar"))


def is_dql(query: str) -> bool:
    """
    Returns True if the input SQL query
    is a DQL statement (SELECT).
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
    Returns True if the input SQL query
    is a procedure.
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
