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
import pytest
import logging

import verticapy as vp
import vertica_python

from verticapy.performance.vertica import QueryProfiler
from verticapy.datasets import load_amazon
from io import StringIO


def test_profile_simple():
    """Create a query profiler and run the steps on a simple query"""
    assert len(vp.available_connections()) > 0
    amzn = setup_dummy_table_run_query()
    request = f"""
    SELECT 
        date, 
        MONTH(date) AS month, 
        AVG(number) AS avg_number 
    FROM 
        public.amazon 
    GROUP BY 1;
    """
    qp = QueryProfiler(request)
    check_version(qp)
    check_request(qp, "avg(number) AS avg_number")


def setup_dummy_table_run_query():
    amzn = load_amazon()
    return amzn


def check_version(qp):
    version_tuple = qp.get_version()
    logging.info(f"Version is: {version_tuple}")
    # version tuple can be
    #  (24, 1, 0)    (dev build)
    #  (11, 0, 1, 2) (release build)
    assert len(version_tuple) == 3 or len(version_tuple) == 4
    assert version_tuple[0] >= 23 or version_tuple[0] in [12, 11]


def check_request(qp, fragment):
    sql = qp.get_request(indent_sql=False)
    # sql won't match the input query exactly
    # input query = select count(*) from foo;
    # stored query = PROFILE select count(*) from foo;
    logging.info(f"Request retreived is: {sql}")
    assert fragment.lower() in sql.lower()
