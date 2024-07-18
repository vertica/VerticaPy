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

from verticapy.performance.vertica.qprof import QueryProfiler


class QueryProfilerStats(QueryProfiler):
    """
    Base class to do some specific
    performance tests.
    """

    def test_client_data(self):
        """
        This test can be used to check
        if the time to send the data to
        client is too huge. It can reveal
        network or terminal problems.

        Returns
        -------
        tuple
            ('Send Data to Client' exec time,
             total exec time, ratio).
        """
        query = f"""
            SELECT
                exec_us,
                total,
                exec_us / total AS ratio
            FROM
            (
                SELECT
                    *,
                    SUM(exec_us) OVER () AS total
                FROM
                (
                    SELECT
                        activity, SUM(exec_us) AS exec_us
                    FROM {self.get_activity_time()}
                    GROUP BY 1
                ) AS q0
            ) AS q1
            WHERE activity = 'Send Data to Client'
        """
        res = _executeSQL(
            query,
            title="Getting the 'Send Data to Client' ratio",
            method="fetchrow",
        )
        return (res[0], res[1], float(res[2]))
