import pytest
import logging

# verticapy is notebook integration
import verticapy as vp

# vertica_python is client connection
import vertica_python

@pytest.fixture()
def vp_connect_get_name():
    logging.info("Trying to connect")
    info_map = get_connection_info()
    connection_name = "pytest connection"
    vp.new_connection(info_map,
                      name="pytest connection")
    logging.info("Connection complete")
    return connection_name

@pytest.fixture
def raw_client_cursor():
    info_map = get_connection_info()
    conn = None
    try:
        conn = vertica_python.connect(**info_map)
        yield conn.cursor()
    finally:
        if conn is not None:
            conn.close()

def get_connection_info():
    # TODO: hard-coded today
    # TODO: Should read from environment veriables
    return {"host": "engdev4.verticacorp.com",
            "port": "32788",
            "database": "jslaunwhite",
            "password": "$vertica$",
            "user": "jslaunwhite",
            "connection_timeout": 5}
