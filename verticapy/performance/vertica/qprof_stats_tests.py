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

        .. note::

            If the ratio is bigger than 20%
            and the total exec time grater
            than 3s. It probably means that
            there is a network or terminal
            problem.

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
            WHERE activity = 'Send Data to Client';
        """
        res = _executeSQL(
            query,
            title="Getting the 'Send Data to Client' ratio.",
            method="fetchrow",
        )
        return (res[0], res[1], float(res[2]))

    def test_exec_time(self):
        """
        Checks if the parser of the SQL is
        taking too much time.
        PreparePlan, CompilePlan should be
        under the second and ExecutePlan
        should take most of the time.

        .. note::

            If Vertica is taking too much
            time to parse the query, something
            wrong: one example could be that the
            logging parameter causing to spend a
            lot of time to parse the data.
            It can be due to a system parameter.

        Returns
        -------
        tuple
            (exec time, total time, ratio).
        """
        query = f"""
            SELECT 
                DECODE(step, 'ExecutePlan', 1, 0) AS step,
                SUM(elapsed) AS elapsed
            FROM
                {self.get_qsteps(show=False)}
            GROUP BY 1
            ORDER BY 1
        """
        res = _executeSQL(
            query,
            title="Getting the exec time vs the total.",
            method="fetchall",
        )
        total_time = res[0][1] + res[1][1]
        return res[1][1], total_time, res[1][1] / total_time

    def test_pool_queue_wait_time(self):
        """
        This test can be used to see if
        a pool takes too much time to be
        allocated.

        The optimizer creates the plan to
        execute the query. If the queue_wait_time
        is not near to 0, it means the resource
        pool is not giving the needed memory
        to the query. It means we do not have
        enough memory to execute the query.

        Returns
        -------
        list
            (node_name, pool_name,
            queue_wait_time_seconds).
        """
        query = f"""
            SELECT
                node_name, 
                pool_name, 
                queue_wait_time / '00:00:01'::INTERVAL AS queue_wait_time_seconds
            FROM
                (
                    SELECT 
                        node_name, 
                        pool_name,
                        SUM(queue_wait_time) AS queue_wait_time
                    FROM
                        {self.get_resource_acquisition()}
                    GROUP BY 1, 2
                ) AS q0
            WHERE queue_wait_time > '1 second'::INTERVAL
        """
        res = _executeSQL(
            query,
            title="Getting the queue wait time for each pool.",
            method="fetchall",
        )
        return res

    def test_query_events(self):
        """
        This test can be used to check
        all types of events. They are
        classified in three categories:
        informational, warning, critical

        .. note::

            The tables' columns are the
            following:
                node_name,
                event_category,
                event_type,
                event_description,
                suggested_action

        Returns
        -------
        tuple of list
            (informational, warning, critical).
        """
        query = f"""
            SELECT
                node_name,
                event_category,
                event_type,
                event_description,
                suggested_action
            FROM
                {self.get_query_events()}
            ORDER BY 2, 3;
        """
        res = _executeSQL(
            query,
            title="Getting all the query events.",
            method="fetchall",
        )
        informational, warning, critical = [], [], []
        for event in res:
            if event[2] in (
                "AUTO_PROJECTION_USED",
                "GROUP_BY_SPILLED",
                "INVALID COST",
                "PATTERN_MATCH_NMEE",
                "PREDICATE OUTSIDE HISTOGRAM",
                "RESEGMENTED_MANY_ROWS",
                "RLE_OVERRIDDEN",
            ):
                warning += [event]
            elif event[2] in (
                "DELETE WITH NON OPTIMIZED PROJECTION",
                "JOIN_SPILLED",
                "MEMORY LIMIT HIT",
                "NO HISTOGRAM",
            ):
                critical += [event]
            else:
                informational += [event]
        return (informational, warning, critical)
