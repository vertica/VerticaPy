import pytest
import logging

import verticapy as vp
import vertica_python


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
