import pytest
import logging
import os

# verticapy is notebook integration
import verticapy as vp

# vertica_python is client connection
import vertica_python


@pytest.fixture()
def vp_connect_get_name():
    logging.info("Trying to connect")
    info_map = get_connection_info()
    connection_name = "pytest connection"
    # TODO: when this fails, the error message has
    # verbose stack information, but not instructions
    # for how to fix things
    vp.new_connection(info_map, name="pytest connection")
    logging.info("Connection complete")
    return connection_name


@pytest.fixture
def raw_client_cursor():
    info_map = get_connection_info(verbose=False)
    conn = None
    try:
        conn = vertica_python.connect(**info_map)
        yield conn.cursor()
    finally:
        if conn is not None:
            conn.close()


def get_connection_info(verbose=True):
    # Fetch from the environment
    host = os.environ.get("VP_TEST_HOST", None)
    db = os.environ.get("VP_TEST_DATABASE", None)
    port = os.environ.get("VP_TEST_PORT", "5433")
    password = os.environ.get("VP_TEST_PASSWORD", None)

    _throw_if_none(host, "VP_TEST_HOST")
    _throw_if_none(db, "VP_TEST_DATABASE")

    # User is also required, but it could come from
    # two sources
    test_user = os.environ.get("VP_TEST_USER", None)
    user = test_user if test_user is not None else os.environ.get("USER", None)

    if user is None:
        raise ValueError(
            "Neither VP_TEST_USER nor USER are set, but the test framework expected them to be set"
        )
    msg = (
        "\nConnection info:\n"
        f"host = {host}\n"
        f"port = {port}\n"
        f"database = {db}\n"
        f"user = {user}\n"
        f"password = {'EMPTY' if password is None else '(REDACTED)'}\n"
    )
    if verbose:
        print(msg)
    logging.info(msg)

    return {
        "host": host,
        "port": port,
        "database": db,
        "password": password,
        "user": user,
        "connection_timeout": 5,
    }


def _throw_if_none(value, env_var):
    if value is None:
        raise ValueError(
            f"{env_var} was not set, but the test framework expected it to be set"
        )
