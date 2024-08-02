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
import copy
from typing import Optional

from verticapy._typing import NoneType
from verticapy._utils._sql._sys import _executeSQL

from verticapy.performance.vertica.qprof import QueryProfiler


class QueryProfilerStats(QueryProfiler):
    """
    Base class to do some specific
    performance tests.
    """

    @staticmethod
    def _get_time_conv(
        time_us: Optional[float], out_unit: Optional[str] = None
    ) -> tuple[float, str]:
        """
        Utility method to convert time
        to the right unit.
        """
        if time_us is None:
            return 0.0, "milliseconds"
        if out_unit == "microseconds":
            return round(float(time_us), 2), out_unit
        optime, unit = float(time_us / 1000), "milliseconds"
        if out_unit == "milliseconds":
            return round(float(optime), 2), out_unit
        if optime > 1000000:
            optime = optime / 1000000
            unit = "seconds"
            if out_unit == "seconds":
                return round(optime, 2), out_unit
            if optime > 3600:
                optime = optime / 3600
                unit = "hours"
        optime = round(optime, 2)
        return optime, unit

    def _get_sql_action(
        self, informational: list[list], warning: list[list], critical: list[list]
    ) -> tuple[list[list]]:
        """
        Takes as inputs all the query
        events (informational, warning,
        critical) and returns the associated
        SQL action.
        """
        informational_final = copy.deepcopy(informational)
        warning_final = copy.deepcopy(warning)
        critical_final = copy.deepcopy(critical)

        for idx, qe in enumerate(informational_final):
            informational_final[idx] = qe + [""]
        for idx, qe in enumerate(warning_final):
            if qe[2] == "PREDICATE OUTSIDE HISTOGRAM" and qe[4].startswith(
                "analyze_statistics"
            ):
                warning_final[idx] = qe + ["SELECT " + qe[4]]
            else:
                warning_final[idx] = qe + [""]
        for idx, qe in enumerate(critical_final):
            if qe[2] == "NO HISTOGRAM" and qe[4].startswith("analyze_statistics"):
                critical_final[idx] = qe + ["SELECT " + qe[4]]
            else:
                critical_final[idx] = qe + [""]
        return informational_final, warning_final, critical_final

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
                suggested_action,
                action_sql

        Returns
        -------
        tuple of list
            (informational, warning, critical).
        """

        # Getting all Vertica Events
        informational, warning, critical = self.query_events_test()

        # Client Data Test
        client_data_test = self.client_data_test()

        if not (isinstance(client_data_test, NoneType)):
            percent = round(client_data_test[-1] * 100, 2)
            optime, unit = self._get_time_conv(client_data_test[1])
            extime, exunit = self._get_time_conv(client_data_test[2], unit)

            if percent > 40:
                description = (
                    "The time to send data to the client is alarmingly higher than "
                    f"expected, taking {optime:,} {unit} which represents {percent:,}% "
                    f"of the total execution time of {extime:,} {exunit}."
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
                    f"expected, taking {optime:,} {unit} which represents {percent:,}% "
                    f"of the total execution time of {extime:,} {exunit}."
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
                    f"only {optime:,} {unit} which is just {percent:,}% of the "
                    f"total execution time of {extime:,} {exunit}."
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

        if not (isinstance(exec_time_test, NoneType)):
            percent = round(exec_time_test[-1] * 100, 2)
            optime, unit = self._get_time_conv(exec_time_test[0])
            extime, exunit = self._get_time_conv(exec_time_test[1], unit)
            extime_us = exec_time_test[1]

            if percent > 50 and extime_us > 5000000:
                description = (
                    "The time to parse the data and generate the plan "
                    f"is alarmingly higher than expected, taking {optime:,} "
                    f"{unit}, which represents {percent:,}% of the total time "
                    f"({extime:,} {exunit}) for parsing and executing."
                )
                recommended_action = (
                    "Please check your system parameters for critical issues."
                )
                critical += [
                    [
                        "Query Initiator",
                        "OPTIMIZATION",
                        "PARSING_TIME_CRITICAL",
                        description,
                        recommended_action,
                    ]
                ]
            elif percent > 30 and extime_us > 5000000:
                description = (
                    "The time to parse the data and generate the plan "
                    f"is a bit higher than expected, taking {optime:,} {unit}, "
                    f"which represents {percent:,}% of the total time "
                    f"({extime:,} {exunit}) for parsing and executing."
                )
                recommended_action = (
                    "Please check your system parameters for possible issues."
                )
                warning += [
                    [
                        "Query Initiator",
                        "OPTIMIZATION",
                        "PARSING_TIME_HIGH",
                        description,
                        recommended_action,
                    ]
                ]
            else:
                if extime_us > 5000000:
                    description = (
                        "The time to parse the data and generate the plan "
                        f"is reasonable, taking only {optime:,} {unit} which "
                        f"is just {percent:,}% of the total execution time "
                        f"of {extime:,} {exunit}."
                    )
                else:
                    description = (
                        "The time to parse the data and generate the plan "
                        f"is reasonable, taking only {optime:,} {unit}."
                    )
                recommended_action = ""
                informational += [
                    [
                        "Query Initiator",
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
                "All resource pool queue wait times are within acceptable limits."
            )
            recommended_action = ""
            informational += [
                [
                    "Query Initiator",
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
                    f"pool {pool_name} is taking {qts:,} seconds "
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

        # Segmentation Test
        segmentation_test = self.segmentation_test()

        for (
            table_name,
            projection_name,
            node_name,
            row_count,
            avg_row_count,
            ratio,
        ) in segmentation_test:
            percent = round(ratio * 100, 2)
            row_count = int(row_count)
            avg_row_count = int(avg_row_count)

            if percent > 50:
                description = (
                    f"The table '{table_name}' is poorly segmented for projection "
                    f"'{projection_name}' on node '{node_name}'. It has {row_count:,} "
                    f"rows, compared to an average of {avg_row_count:,} rows on "
                    f"other nodes, representing a deviation of {percent:,}%. Please "
                    "resegment to rectify this issue."
                )
                recommended_action = (
                    "Please resegment to rectify this issue. You can "
                    "use DBD to create a more uniform segmentation."
                )
                warning += [
                    [
                        node_name,
                        "OPTIMIZATION",
                        "DATA_SEGMENTATION_BAD",
                        description,
                        recommended_action,
                    ]
                ]
            elif percent > 30:
                description = (
                    f"The table '{table_name}' is not ideally segmented for projection "
                    f"'{projection_name}' on node '{node_name}'. It has {row_count:,} "
                    f"rows, compared to an average of {avg_row_count:,} rows on "
                    f"other nodes, representing a deviation of {percent:,}%. The deviation "
                    "is high but within acceptable limits, so no immediate action is "
                    "required."
                )
                recommended_action = (
                    "Please resegment to rectify this issue. You can "
                    "use DBD to create a more uniform segmentation."
                )
                warning += [
                    [
                        node_name,
                        "OPTIMIZATION",
                        "DATA_SEGMENTATION_FAIR",
                        description,
                        recommended_action,
                    ]
                ]
            else:
                description = (
                    f"The table '{table_name}' is well-segmented for projection "
                    f"'{projection_name}' on node '{node_name}'. It has {row_count:,} "
                    f"rows, closely matching the average of {avg_row_count:,} rows on "
                    f"other nodes, with a deviation of {percent:,}% which is within "
                    "acceptable limits."
                )
                recommended_action = ""
                informational += [
                    [
                        node_name,
                        "OPTIMIZATION",
                        "DATA_SEGMENTATION_GOOD",
                        description,
                        recommended_action,
                    ]
                ]

        # Clock Time VS Exec Time
        clock_exec_time_test = self.clock_exec_time_test()

        for (
            node_name,
            operator_name,
            path_id,
            clock_time_us,
            exec_time_us,
            ratio,
        ) in clock_exec_time_test:
            if ratio is None:
                percent = 0
            else:
                percent = round(ratio * 100, 2)
            clock_time, ct_unit = self._get_time_conv(clock_time_us)
            exec_time, et_unit = self._get_time_conv(exec_time_us, ct_unit)

            if (exec_time_us is None or exec_time_us < 5000000) and (
                clock_time_us is None or clock_time_us < 5000000
            ):
                description = (
                    f"The clock time ({clock_time:,} {ct_unit}) for "
                    f"node '{node_name}' in PATH ID {path_id} with "
                    f"operator '{operator_name}' is comparable to "
                    f"execution time ({exec_time:,} {et_unit})."
                )
                recommended_action = ""
                informational += [
                    [
                        node_name,
                        "EXECUTION",
                        "CLOCK_EXEC_TIME_DIFFERENCE_REASONABLE",
                        description,
                        recommended_action,
                    ]
                ]
            elif percent > 20:
                description = (
                    f"The clock time ({clock_time:,} {ct_unit}) for node"
                    f" '{node_name}' in PATH ID {path_id} with operator"
                    f" '{operator_name}' differs significantly than the execution"
                    f" time ({exec_time:,} {et_unit}). The ratio stands at approximately"
                    f" {percent:,}%. Take immediate action."
                )
                recommended_action = (
                    "There could be multiple reasons for the difference between "
                    "execution time and clock time. Investigate to gain more "
                    "insights. Sometimes the discrepancy is expected, such as "
                    "in the case of a JOIN waiting for a SCAN. Other times, it "
                    "could be due to I/O operations, which can be analyzed further."
                )
                critical += [
                    [
                        node_name,
                        "EXECUTION",
                        "CLOCK_EXEC_TIME_DIFFERENCE_CRITICAL",
                        description,
                        recommended_action,
                    ]
                ]
            elif percent > 10:
                description = (
                    f"The clock time ({clock_time:,} {ct_unit}) for "
                    f"node '{node_name}' in PATH ID {path_id} with"
                    f" operator '{operator_name}' differs"
                    f" than the execution time ({exec_time:,} {et_unit})."
                    f" The ratio stands at approximately {percent:,}%."
                    " This can be concerning. You can investigate to improve "
                    "query performance."
                )
                recommended_action = (
                    "There could be multiple reasons for the difference between "
                    "execution time and clock time. Investigate to gain more "
                    "insights. Sometimes the discrepancy is expected, such as "
                    "in the case of a JOIN waiting for a SCAN. Other times, it "
                    "could be due to I/O operations, which can be analyzed further."
                )
                warning += [
                    [
                        node_name,
                        "EXECUTION",
                        "CLOCK_EXEC_TIME_DIFFERENCE_HIGH",
                        description,
                        recommended_action,
                    ]
                ]
            else:
                description = (
                    f"The clock time ({clock_time:,} {ct_unit}) for "
                    f"node '{node_name}' in PATH ID {path_id} with "
                    f"operator '{operator_name}' is comparable to "
                    f"execution time ({exec_time:,} {et_unit}). "
                    f"This increase is only {percent:,}."
                )
                recommended_action = ""
                informational += [
                    [
                        node_name,
                        "EXECUTION",
                        "CLOCK_EXEC_TIME_DIFFERENCE_REASONABLE",
                        description,
                        recommended_action,
                    ]
                ]

        informational, warning, critical = self._get_sql_action(
            informational, warning, critical
        )
        nodes = self._get_current_nodes()
        informational_final = []
        warning_final = []
        critical_final = []
        for alert in informational:
            if alert[0] in nodes:
                informational_final += [alert]
        for alert in warning:
            if alert[0] in nodes:
                warning_final += [alert]
        for alert in critical:
            if alert[0] in nodes:
                critical_final += [alert]
        return informational_final, warning_final, critical_final

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
        if isinstance(res, NoneType):
            return None
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
        if not res:
            return None
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

    def segmentation_test(self):
        """
        This test can be used to check
        if the data are correctly
        segmented.

        .. note::

            The tables' columns are the
            following:
                table_name,
                projection_name,
                node_name,
                row_count,
                avg_row_count,
                ratio

            A ratio equals to 0 means
            perfectly segmented data.
            And a ratio equals to 1
            means very poorly segmented
            data.

        Returns
        -------
        list of list
            (table_name, projection_name,
             node_name, row_count, avg_row_count,
             ratio).
        """
        query = f"""
            SELECT
                table_name,
                projection_name,
                node_name,
                row_count,
                avg_row_count,
                ABS(row_count - avg_row_count) / avg_row_count AS ratio
            FROM
                (
                    SELECT
                        table_name,
                        projection_name,
                        node_name,
                        row_count,
                        AVG(row_count) OVER (PARTITION BY projection_name) AS avg_row_count
                    FROM
                        {self.get_proj_data_distrib()}
                ) AS q0;
        """
        res = _executeSQL(
            query,
            title="Getting all the query events.",
            method="fetchall",
        )
        return res

    def clock_exec_time_test(self):
        """
        This test can be used to check
        if the execution time is near to
        the clock time.

        If the clock time is much greater
        than the execution time, it means
        that the thread has been in a wait
        state for some time, waiting for
        something.

        .. note::

            The tables' columns are the
            following:
                node_name,
                operator_name,
                path_id,
                clock_time_us,
                exec_time_us,
                ratio

            A ratio greater than 20% can
            be seen as critical.

        Returns
        -------
        list of list
            (node_name, operator_name,
             path_id, clock_time_us,
             exec_time_us, ratio).
        """
        query = f"""
            SELECT 
                node_name,
                operator_name,
                path_id,
                clock_time_us,
                exec_time_us,
                (CASE
                    WHEN exec_time_us = 0 OR clock_time_us = 0
                        THEN NULL
                    WHEN exec_time_us > clock_time_us
                        THEN (exec_time_us - clock_time_us) / exec_time_us
                    ELSE
                        (clock_time_us - exec_time_us) / clock_time_us
                END) AS ratio
            FROM {self.get_qexecution_report()}
        """
        res = _executeSQL(
            query,
            title="Getting all the query events.",
            method="fetchall",
        )
        return res
