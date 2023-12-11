import pytest
import logging

import verticapy as vp
import vertica_python

from verticapy.performance.vertica import QueryProfiler

from io import StringIO

def test_profile_simple(vp_connect_get_name, raw_client_cursor):
    """Create a query profiler and run the steps on a simple query"""

    table_name = "test_profile_simple"
    setup_dummy_table(raw_client_cursor, table_name)
    request = f"SELECT count(*) from {table_name};"
    qp = QueryProfiler(request)
    check_version(qp)
    check_request(qp, "SELECT COUNT(*)")

def setup_dummy_table(raw_client_cursor, table_name):
    raw_client_cursor.execute(f"DROP table if exists {table_name};")
    raw_client_cursor.fetchall()
    raw_client_cursor.execute(f"CREATE TABLE {table_name} (x int);")
    raw_client_cursor.fetchall()

    raw_client_cursor.executemany(f"insert into {table_name} values (?);",
                                  [(1,),
                                   (2,),
                                   (3,),
                                   (4,)],
                                  use_prepared_statements=True)
    raw_client_cursor.fetchall()

def check_version(qp):
    version_tuple = qp.get_version()
    logging.info(f"Version is: {version_tuple}")
    # version tuple can be
    #  (24, 1, 0)    (dev build)
    #  (11, 0, 1, 2) (release build)
    assert (len(version_tuple) == 3 or len(version_tuple) == 4)
    assert (version_tuple[0] >= 23 or version_tuple[0] in [12, 11])

def check_request(qp, fragment):
    sql = qp.get_request(indent_sql=False)
    # sql won't match the input query exactly
    # input query = select count(*) from foo;
    # stored query = PROFILE select count(*) from foo;
    logging.info(f"Request retreived is: {sql}")
    assert fragment.lower() in sql.lower()

def test_connect_perf(vp_connect_get_name, raw_client_cursor):
    """ Connect to the vertica server
    Run a very simple verticapy connection via dataframe
    """
    df = vp.vDataFrame("""
    /*test_connect_perf*/
    SELECT 1 as x
    UNION ALL
    SELECT 2 as x
    UNION ALL
    SELECT 3 as x
    """)
    # Check that the dataframe has the expected values
    avg = df["x"].avg()
    logging.info(f"Average from the data frame is: {avg}")
    assert avg == 2

    # Confirm that the dataframe really did connect
    # and send the query
    # TODO: we could clear the dc table first?
    raw_client_cursor.execute("""
    SELECT count(*)
    FROM dc_requests_issued
    WHERE request ilike '%test_connect_perf%';
    """)

    rset = raw_client_cursor.fetchall()
    logging.info(f"Result set looking for test queries {rset}")
    assert len(rset) == 1
    assert rset[0][0] > 0
