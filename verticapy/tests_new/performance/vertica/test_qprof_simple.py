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
import logging

import verticapy as vp

from verticapy.performance.vertica import QueryProfiler
from verticapy.datasets import load_amazon
from verticapy.core.vdataframe import vDataFrame


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


def setup_dummy_table_run_query() -> vDataFrame:
    amzn = load_amazon()
    return amzn


def check_version(qp: QueryProfiler) -> None:
    version_tuple = qp.get_version()
    logging.info(f"Version is: {version_tuple}")
    # version tuple can be
    #  (24, 1, 0)    (dev build)
    #  (11, 0, 1, 2) (release build)
    assert len(version_tuple) == 3 or len(version_tuple) == 4
    assert version_tuple[0] >= 23 or version_tuple[0] in [12, 11]


def check_request(qp: QueryProfiler, fragment: str) -> None:
    sql = qp.get_request(indent_sql=False)
    # sql won't match the input query exactly
    # input query = select count(*) from foo;
    # stored query = PROFILE select count(*) from foo;
    logging.info(f"Request retreived is: {sql}")
    assert fragment.lower() in sql.lower()


def check_duration(qp: QueryProfiler) -> None:
    duration = qp.get_qduration("s")
    logging.info(f"Query duration was {duration} seconds")
    # Duration is about 10 ms
    assert duration > 0.0001
    return


def check_query_events(qp: QueryProfiler) -> None:
    events = qp.get_query_events()
    assert not events.empty()
    cols = events.get_columns()
    assert len(cols) == 9
    assert "event_type" in cols

    col_indexes = {name: index for index, name in enumerate(cols)}
    AUTOPROJ_EVENT_TYPE = "AUTO_PROJECTION_USED"
    found_autoproj_event = False

    for x in range(len(events)):
        row = events[x]
        found_autoproj_event |= look_for_autoproj(
            row, col_indexes["event_type"], AUTOPROJ_EVENT_TYPE
        )

    assert found_autoproj_event


def look_for_autoproj(row: list, index: int, target: str) -> bool:
    return True if row[index] == target else False
