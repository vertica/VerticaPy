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

    @staticmethod
    def _get_time_conv(time_ms: float) -> tuple[float, str]:
        """
        Util method to convert time
        to the right unit.
        """
        optime, unit = time_ms, "milliseconds"
        if optime > 1000:
            optime = optime / 1000
            unit = "seconds"
            if optime > 3600:
                optime = optime / 3600
                unit = "hours"
        optime = round(optime, 2)
        return optime, unit

    def main_tests(self):
        """
        This is the main test to run to
        get all the information needed
        to understand queries performances.

        Events are classified in three
        categories:
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

        # Getting all Vertica Events
        informational, warning, critical = self.query_events_test()

        # Client Data Test
        client_data_test = self.client_data_test()

        percent = round(client_data_test[-1] * 100, 2)
        optime, unit = self._get_time_conv(client_data_test[1])
        extime, exunit = self._get_time_conv(client_data_test[2])

        if percent > 40:
            description = (
                "The time to send data to the client is alarmingly higher than "
                f"expected, taking {optime} {unit} which represents {percent}% "
                f"of the total execution time of {extime} {exunit}."
            )
            recommended_action = (
                "Check your network connection and terminal "
                "configuration for critical issues."
            )
            critical += [
                [
                    client_data_test[0],
                    "NETWORK",
                    "TRANSMISSION_TIME_TO_CLIENT_CRITICAL",
                    description,
                    recommended_action,
                ]
            ]
        elif percent > 20:
            description = (
                "The time to send data to the client is a bit higher than "
                f"expected, taking {optime} {unit} which represents {percent}% "
                f"of the total execution time of {extime} {exunit}."
            )
            recommended_action = (
                "Check your network connection and terminal configuration."
            )
            warning += [
                [
                    client_data_test[0],
                    "NETWORK",
                    "TRANSMISSION_TIME_TO_CLIENT_HIGH",
                    description,
                    recommended_action,
                ]
            ]
        else:
            description = (
                "The time to send data to the client is reasonable, taking "
                f"only {optime} {unit} which is just {percent}% of the "
                f"total execution time of {extime} {exunit}."
            )
            recommended_action = ""
            informational += [
                [
                    client_data_test[0],
                    "NETWORK",
                    "TRANSMISSION_TIME_TO_CLIENT_REASONABLE",
                    description,
                    recommended_action,
                ]
            ]

        # Parser Test
        exec_time_test = self.exec_time_test()

        percent = round(exec_time_test[-1] * 100, 2)
        optime, unit = self._get_time_conv(exec_time_test[0])
        extime, exunit = self._get_time_conv(exec_time_test[1])
        extime_ms = exec_time_test[1]

        if percent > 50 and extime_ms > 5000:
            description = (
                "The time to parse the data and generate the plan "
                f"is alarmingly higher than expected, taking {optime} "
                f"{unit}, which represents {percent}% of the total time "
                f"({extime} {exunit}) for parsing and executing."
            )
            recommended_action = (
                "Please check your system parameters for critical issues."
            )
            critical += [
                [
                    "initiator",
                    "OPTIMIZATION",
                    "PARSING_TIME_CRITICAL",
                    description,
                    recommended_action,
                ]
            ]
        elif percent > 30 and extime_ms > 5000:
            description = (
                "The time to parse the data and generate the plan "
                f"is a bit higher than expected, taking {optime} {unit}, "
                f"which represents {percent}% of the total time "
                f"({extime} {exunit}) for parsing and executing."
            )
            recommended_action = (
                "Please check your system parameters for possible issues."
            )
            warning += [
                [
                    "initiator",
                    "OPTIMIZATION",
                    "PARSING_TIME_HIGH",
                    description,
                    recommended_action,
                ]
            ]
        else:
            if extime_ms > 5000:
                description = (
                    "The time to parse the data and generate the plan "
                    f"is reasonable, taking only {optime} {unit} which "
                    f"is just {percent}% of the total execution time "
                    f"of {extime} {exunit}."
                )
            else:
                description = (
                    "The time to parse the data and generate the plan "
                    f"is reasonable, taking only {optime} {unit}."
                )
            recommended_action = ""
            informational += [
                [
                    "initiator",
                    "OPTIMIZATION",
                    "PARSING_TIME_REASONABLE",
                    description,
                    recommended_action,
                ]
            ]

        # Resource Pool Test
        pooltime = self.pool_queue_wait_time_test()

        if len(pooltime) == 0:
            description = (
                "All resource pool queue wait times " "are within acceptable limits."
            )
            recommended_action = ""
            informational += [
                [
                    "initiator",
                    "EXECUTION",
                    "RP_QUEUE_WAIT_TIME_REASONABLE",
                    description,
                    recommended_action,
                ]
            ]
        else:
            for node_name, pool_name, qts in pooltime:
                description = (
                    "Some resource pools have queue wait "
                    "times higher than expected. Resource "
                    f"pool {pool_name} is taking {qts} seconds "
                    "to be allocated."
                )
                recommended_action = (
                    "Consider adjusting the MAXMEMORYSIZE and "
                    "PLANNEDCONCURRENCY resource pools so that "
                    "the optimizer has sufficient memory. On a "
                    "heavily used system, this event may occur "
                    "more frequently."
                )
                warning += [
                    [
                        node_name,
                        "EXECUTION",
                        "RP_QUEUE_WAIT_TIME_HIGH",
                        description,
                        recommended_action,
                    ]
                ]

        return informational, warning, critical

    def client_data_test(self):
        """
        This test can be used to check
        if the time to send the data to
        client is too huge. It can reveal
        network or terminal problems.

        .. note::

            If the ratio is bigger than 20%
            and the total exec time greater
            than 3s. It probably means that
            there is a network or terminal
            problem.

        Returns
        -------
        tuple
            (node name,
             'Send Data to Client' exec time,
             total exec time, ratio).
        """
        query = f"""
            SELECT
                node_name,
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
                        node_name,
                        activity, 
                        SUM(exec_us) AS exec_us
                    FROM {self.get_activity_time()}
                    GROUP BY 1, 2
                ) AS q0
            ) AS q1
            WHERE activity = 'Send Data to Client';
        """
        res = _executeSQL(
            query,
            title="Getting the 'Send Data to Client' ratio.",
            method="fetchrow",
        )
        return (res[0], res[1], res[2], float(res[3]))

    def exec_time_test(self):
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

    def pool_queue_wait_time_test(self):
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

    def query_events_test(self):
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
