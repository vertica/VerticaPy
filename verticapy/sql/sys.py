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
from verticapy._utils._sql._sys import _executeSQL


def current_session() -> int:
    """
    Returns the current DB session.

    Returns
    -------
    int
        DB session.

    Examples
    --------
    Displays the current DB session:

    .. ipython:: python

        from verticapy.sql import current_session

        current_session()

    .. note::

        The session is used as an identifier in
        the VerticaPy logs.

    .. seealso::
        | :py:meth:`~verticapy.username` : current DB username.
        | :py:meth:`~verticapy.has_privileges` : checks user privileges.
    """
    res = _executeSQL(
        query="SELECT /*+LABEL(current_session)*/ CURRENT_SESSION();",
        method="fetchfirstelem",
        print_time_sql=False,
    )
    return int(res.split(":")[1], base=16)


def username() -> str:
    """
    Returns the current DB username.

    Returns
    -------
    str
        Username.

    Examples
    --------
    Displays the current DB user name:

    .. ipython:: python

        from verticapy.sql import username

        username()

    .. note::

        The username can be important for determining
        privileges.

    .. seealso::
        | :py:meth:`~verticapy.current_session` : current DB session.
        | :py:meth:`~verticapy.has_privileges` : checks user privileges.
    """
    return _executeSQL(
        query="SELECT /*+LABEL(username)*/ USERNAME();",
        method="fetchfirstelem",
        print_time_sql=False,
    )


def does_table_exist(table_name: str, schema: str) -> bool:
    """
    Checks if the specified table exists.

    Parameters
    ----------
    table_name: str
        The table name.
    schema: str
        Schema name.

    Returns
    -------
    bool
        False if the table doesn't exist,
        or it exists but the user has no
        USAGE privilege on it.
        True otherwise.

    Examples
    --------
    Checks if a table exist:

    .. ipython:: python

        from verticapy.sql import does_table_exist

        does_table_exist(
            table_name = "fake_name",
            schema = "fake_schema",
        )

    .. note::

        Checks if the table exists, but it will not raise
        any errors; instead, it returns a boolean value,
        True or False.
    """
    query = f"SELECT COUNT(*) FROM tables WHERE table_name='{table_name:}' AND table_schema='{schema}';"
    result = _executeSQL(query, title="Does the table exist?", method="fetchfirstelem")
    if result == 0:
        return False
    return True


def does_view_exist(view_name: str, schema: str) -> bool:
    """
    Checks if the specified view exists.

    Parameters
    ----------
    view_name: str
        The view name.
    schema: str
        Schema name.

    Returns
    -------
    bool
        False if the view doesn't exist,
        or it exists but the user has no
        USAGE privilege on it.
        True otherwise.

    Examples
    --------
    Checks if a table exist:

    .. ipython:: python

        from verticapy.sql import does_view_exist

        does_view_exist(
            view_name = "fake_name",
            schema = "fake_schema",
        )

    .. note::

        Checks if the view exists, but it will not raise
        any errors; instead, it returns a boolean value,
        True or False.
    """
    query = f"SELECT COUNT(*) FROM v_catalog.views WHERE table_name='{view_name:}' AND table_schema='{schema}';"
    result = _executeSQL(query, title="Does the table exist?", method="fetchfirstelem")
    if result == 0:
        return False
    return True


def has_privileges(
    object_name: str, object_schema: str, privileges: list, raise_error: bool = False
) -> bool:
    """
    Checks if the user has all the privileges on
    the object_schema.object_name object.

    Parameters
    ----------
    object_name: str
        The object name.
    object_schema: str
        Schema name.
    privileges: list
        The list of privileges.
    raise_error: bool, optional
        It raises an error if not all privileges
        are granted.

    Returns
    -------
    bool
        True if the user has been granted the
        list of privileges on the object.
        False otherwise.

    Examples
    --------
    ...

    .. seealso::
        | :py:meth:`~verticapy.current_session` : current DB session.
        | :py:meth:`~verticapy.username` : current DB username.
    """
    query_superuser = f"""
        SELECT 
            is_super_user 
        FROM v_catalog.users 
        WHERE user_name = current_user();"""
    query_grant = f"""
        SELECT 
            privileges_description 
        FROM grants
        WHERE object_schema='{object_schema}' 
          AND object_name='{object_name}'
          AND grantee=current_user();"""

    try:
        is_superuser = _executeSQL(
            query_superuser, title="is superuser?", method="fetchfirstelem"
        )
        if is_superuser:
            return True

        result = _executeSQL(query_grant, title="Cheking privileges", method="fetchrow")
        if result is None:
            raise AttributeError(
                f"There is no privilege on {object_schema}.{object_name}."
            )

        result = result[0].lower()
        granted_privileges = result.split(", ")
        # there might be a '*' after a privilege name
        for index, item in enumerate(granted_privileges):
            if item[len(item) - 1] == "*":
                granted_privileges[index] = item[0 : len(item) - 1]

        for x in privileges:
            x.lower()
            if not (x in granted_privileges):
                raise AttributeError(
                    f"The privilege {x} on {object_schema}.{object_name} is required."
                )

        return True
    except Exception as exc:
        if raise_error:
            raise AttributeError("Cheking privileges failed") from exc
        return False
