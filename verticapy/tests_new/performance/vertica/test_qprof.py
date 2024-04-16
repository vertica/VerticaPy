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
from contextlib import redirect_stdout
import io
import json
import logging
import re
from typing import Any, Literal, Tuple

from graphviz import Digraph
import numpy as np
import pandas as pd
import pytest

import verticapy as vp
from verticapy.connection import current_cursor
from verticapy.core.vdataframe import vDataFrame
from verticapy.datasets import load_titanic
from verticapy.performance.vertica import QueryProfiler
from verticapy.performance.vertica.qprof_utility import QprofUtility
from verticapy.tests_new.performance.vertica import QPROF_SQL1, QPROF_SQL2

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
logger = logging.getLogger(__name__)


class QueryInfo:
    """
    test class for Query info
    """

    def __init__(self, query, label):
        self.query = query
        self.label = label


class QprofAttr:
    """
    test class for Query attributes
    """

    def __init__(self, **kwargs):
        self.transactions = kwargs["transactions"]
        self.requests = kwargs["requests"]
        self.request_labels = kwargs["request_labels"]
        self.qdurations = kwargs["qdurations"]
        self.key_id = None
        self.target_schema = {"v_internal": "qprof_test", "v_monitor": "qprof_test"}
        self.target_tables = {}
        self.v_tables_dtypes = []
        self.tables_dtypes = []
        self.overwrite = False


@pytest.fixture(name="qprof_data", scope="class")
def qprof_data(titanic_vd: pytest.fixture, schema_loader: pytest.fixture) -> QprofAttr:
    """
    fixture for qprof data setup

    This fixture generates collections of test values for the QueryProfiler constructor.
    The constructor uses different types of QueryProfiler initialization values. For example,
    the QueryProfiler constructor can use a query (string), or a transaction id and statement id, or a schema.
    """
    # titanic = titanic_vd.to_pandas()

    transactions, qdurations, request_labels = [], [], []

    qprof_sql3 = f"""SELECT /*+LABEL('QueryProfiler_sql3_requests_UT')*/ ticket, substr(ticket, 1, 5) AS ticket, AVG(age) AS avg_age FROM {schema_loader}.titanic GROUP BY 1"""

    queries = [QPROF_SQL1, QPROF_SQL2, qprof_sql3]

    q_infos = [
        QueryInfo(QPROF_SQL1, "QueryProfiler_sql1_requests_UT"),
        QueryInfo(QPROF_SQL2, "QueryProfiler_sql2_requests_UT"),
        QueryInfo(qprof_sql3, "QueryProfiler_sql3_requests_UT"),
    ]

    for q_info in q_infos:
        request_labels.append(q_info.label)
        current_cursor().execute(q_info.query)
        sql = f"select transaction_id, statement_id from v_monitor.query_requests where request_label = '{q_info.label}' and request not like 'PROFILE%' ORDER BY start_timestamp DESC LIMIT 1"
        res = current_cursor().execute(sql).fetchall()
        logger.info(res)
        transaction_id, statement_id = res[0] if isinstance(res[0], list) == 1 else res
        transactions.append((transaction_id, statement_id))

        qduration_sql = f"""SELECT query_duration_us FROM v_monitor.query_profiles WHERE transaction_id={transaction_id} AND statement_id={statement_id}"""
        qduration_res = current_cursor().execute(qduration_sql).fetchall()[0][0]
        # logger.info(qduration_res)
        qdurations.append(qduration_res)
        # logger.info(qdurations)

    return QprofAttr(
        **{
            "transactions": transactions,
            "requests": queries,
            "request_labels": request_labels,
            "qdurations": qdurations,
        }
    )


def _get_chart_data(
    qprof_attr_obj: Any, kind: Literal["bar", "barh", "pie"]
) -> Tuple[pd.Series, pd.Series]:
    """
    function to get chart data
    """
    if kind == "bar":
        x = qprof_attr_obj.data[0].x
        y = qprof_attr_obj.data[0].y
    elif kind == "barh":
        x = qprof_attr_obj.data[0].y
        y = qprof_attr_obj.data[0].x
    elif kind == "pie":
        x = qprof_attr_obj.data[0].labels
        y = qprof_attr_obj.data[0].values

    return x, y


def _get_time_div(unit: Literal["s", "m", "h"]) -> int:
    """
    function to get time div
    """
    if unit == "m":
        div = 1000000 * 60
    elif unit == "h":
        div = 1000000 * 3600
    else:
        div = 1000000

    return div


def _compare_pandas(l_df: pd.DataFrame, r_df: pd.DataFrame) -> pd.DataFrame:
    """
    function to compare two pandas dataframes
    """
    res = l_df.compare(
        r_df,
        result_names=(
            "left",
            "right",
        ),
    )
    return res


class TestQueryProfiler:
    """
    test class for QueryProfiler
    """

    _target_tables = [
        "dc_requests_issued",
        "dc_query_executions",
        "dc_explain_plans",
        "execution_engine_profiles",
        "query_plan_profiles",
        "query_profiles",
        "resource_pool_status",
        "host_resources",
    ]

    @pytest.mark.parametrize(
        "test_name, transactions, key_id, resource_pool, target_schema, overwrite, add_profile, check_tables",
        [
            ("transactions", "integer", None, None, None, False, False, True),
            ("transactions", "list_of_integers", None, None, None, False, False, True),
            ("transactions", "tuple", None, None, None, False, False, True),
            ("transactions", "list_of_tuples", None, None, None, False, False, True),
            ("transactions", "single_sql", None, None, None, False, False, True),
            ("transactions", "multiple_sql", None, None, None, False, False, True),
            ("target_schema", "tuple", None, None, "qprof_test", False, False, True),
            # ("key_id", "tuple", "qprof_key_id", None, None, True, False, True),
            # need to check on this
            # ("key_id", "tuple", "qprof_key_id", None, "qprof_test", True, False, True),
            # need to check on this
            # ("resource_pool", "tuple", None, "aaa", "qprof_test", False, False, True),
            (
                "target_schema",
                "tuple",
                None,
                None,
                {"v_internal": "qprof_test", "v_monitor": "qprof_test"},
                False,
                False,
                True,
            ),
            (
                "target_schema",
                "tuple",
                None,
                None,
                {"v_internal": "qprof_test1", "v_monitor": "qprof_test2"},
                False,
                False,
                True,
            ),
            # need to check on this
            # ("overwrite", "tuple", "ut_key_12", None, "qprof_test", False, False, True),
            # overwrite failing. should overwrite.
            # ("overwrite", "tuple", "ut_key_12", None, "qprof_test", True, False, True),
            ("add_profile", "single_sql", None, None, None, False, True, True),
            # need to check. how this would be tested
            ("check_tables", "integer", None, None, None, False, False, True),
            # need to check. how this would be tested
            ("check_tables", "integer", None, None, None, False, False, False),
        ],
    )
    def test_query_profiler(
        self,
        qprof_data,
        test_name: str,
        transactions,
        key_id,
        resource_pool,
        target_schema,
        overwrite,
        add_profile,
        check_tables,
    ):
        """
        test function for query_profiler

        This function test ``QueryProfiler`` class and its parameters

        - Step 1: Get ``QueryProfiler`` object by passing different combinations of parameters value
                to ``QueryProfiler`` class
        - Step 2: Get parameter value from ``QueryProfiler`` object
        - Step 3: Get corresponding parameter value from ``qprof_data`` fixture
        - Step 4: compare step 2 and step 3 results
        """

        # need to check as target_schema is not getting created by vpy
        current_cursor().execute("CREATE SCHEMA IF NOT EXISTS qprof_test")
        current_cursor().execute("CREATE SCHEMA IF NOT EXISTS qprof_test1")
        current_cursor().execute("CREATE SCHEMA IF NOT EXISTS qprof_test2")

        if test_name == "transactions" and transactions == "integer":
            qprof = QueryProfiler(transactions=qprof_data.transactions[1][0])
            expected_res = (qprof_data.transactions[1][0], 1)
            actual_res = qprof.transactions[0]
        elif test_name == "transactions" and transactions == "list_of_integers":
            qprof = QueryProfiler(
                transactions=[
                    qprof_data.transactions[0][0],
                    qprof_data.transactions[1][0],
                ]
            )
            expected_res = [
                (qprof_data.transactions[0][0], 1),
                (qprof_data.transactions[1][0], 1),
            ]
            actual_res = qprof.transactions
        elif test_name == "transactions" and transactions == "tuple" and not overwrite:
            qprof = QueryProfiler(transactions=qprof_data.transactions[1])
            expected_res = qprof_data.transactions[1]
            actual_res = qprof.transactions[0]
        elif test_name == "transactions" and transactions == "list_of_tuples":
            qprof = QueryProfiler(transactions=qprof_data.transactions)
            expected_res = qprof_data.transactions
            actual_res = qprof.transactions
        elif (
            test_name == "transactions"
            and transactions == "single_sql"
            and not add_profile
        ):
            qprof = QueryProfiler(
                transactions=qprof_data.requests[0], add_profile=add_profile
            )
            expected_res = qprof_data.requests[0]
            actual_res = qprof.request
        elif test_name == "transactions" and transactions == "multiple_sql":
            qprof = QueryProfiler(
                transactions=qprof_data.requests, add_profile=add_profile
            )
            expected_res = qprof_data.requests
            actual_res = qprof.requests
        elif test_name == "key_id":
            qprof = QueryProfiler(
                transactions=qprof_data.transactions[1],
                key_id=key_id,
                target_schema=target_schema,
            )
            expected_res = key_id
            actual_res = qprof.key_id
        elif test_name == "resource_pool":
            qprof = QueryProfiler(
                transactions=qprof_data.transactions[1],
                resource_pool=resource_pool,
            )
            logger.info(qprof.get_rp_status())
        elif test_name == "target_schema":
            qprof = QueryProfiler(
                transactions=qprof_data.transactions[1],
                key_id=key_id,
                target_schema=target_schema,
                overwrite=overwrite,
            )
            expected_res = target_schema
            if isinstance(target_schema, str):
                actual_res = list(qprof.target_schema.values())[-1]
            else:
                actual_res = qprof.target_schema
        elif test_name == "overwrite":
            qprof = QueryProfiler(
                transactions=qprof_data.transactions[1],
                key_id=key_id,
                target_schema=target_schema,
                overwrite=overwrite,
            )
            expected_res = [f"qprof_{table}_{key_id}" for table in self._target_tables]
            actual_res = list(qprof.target_tables.values())
        elif test_name == "add_profile":
            qprof = QueryProfiler(
                transactions=qprof_data.requests[0], add_profile=add_profile
            )
            expected_res = True
            actual_res = "PROFILE" in qprof.request
        elif test_name == "check_tables":
            qprof = QueryProfiler(
                transactions=qprof_data.requests[0], check_tables=add_profile
            )
            expected_res = True
            actual_res = "PROFILE" in qprof.request

        logger.info(
            f"Test Name: {test_name}, Expected result: {expected_res},  Actual result: {actual_res}"
        )
        assert expected_res == actual_res

    @pytest.mark.parametrize(
        "attribute, schema",
        [
            ("transactions", None),
            # ("requests", None),  # failing as schema name is random
            ("request_labels", None),
            ("qdurations", None),
            # ("key_id", None),  # failed. no key id in log
            # ("key_id", "qprof_test"),  # failed. no key id in log(for exp result)
            ("request", None),
            ("qduration", None),
            ("transaction_id", None),
            ("statement_id", None),
            ("target_schema", None),
            ("target_schema", "qprof_test"),
            ("target_tables", None),  # with no schema, no table is listed
            # ("target_tables", "qprof_test"),  # need to check. extra tables
            ("v_tables_dtypes", None),  # with no schema, no tables is listed
            ("v_tables_dtypes", "qprof_test"),
            ("tables_dtypes", None),  # with no schema, no datatype is generated
            ("tables_dtypes", "qprof_test"),
            ("overwrite", None),
            ("overwrite", "qprof_test"),
        ],
    )
    def test_qprof_attributes(self, qprof_data, attribute, schema):
        """
        test function for query_profiler attributes

        This function tests ``QueryProfiler`` attributes

        key_id and target_tables:
        To test key_id/target_tables, we need to know the key_id/target_tables generated.
        Generated key_id/target_tables is read from stdout and compared against key_id/
        from qprof.key_id/target_tables attribute.
        """
        # need to check as target_schema is not getting created by vpy
        current_cursor().execute("create schema if not exists qprof_test")
        if (
            attribute
            in [
                "key_id",
                "target_schema",
                "target_tables",
                "tables_dtypes",
                "overwrite",
            ]
            and schema
        ):
            # reading stdout for key_id and target_tables validation
            f = io.StringIO()
            with redirect_stdout(f):
                qprof = QueryProfiler(
                    transactions=qprof_data.transactions[0],
                    target_schema="qprof_test",
                    overwrite=attribute == "overwrite",
                )
            s = f.getvalue()
            logger.info(s)
        else:
            qprof = QueryProfiler(transactions=qprof_data.transactions)

        actual_res = getattr(qprof, attribute)
        logger.info("<<<<<<<<<<<<<<<<<<< Actual result >>>>>>>>>>>>>>>>>>>>>>")
        logger.info(actual_res)

        if attribute in ["key_id", "target_tables"]:
            expected_res = "" if attribute == "key_id" else {}
            if schema:
                # reading stdout data for key_id and target_tables
                for line in s.splitlines():
                    logger.info(line)
                    if attribute == "key_id" and line.startswith(
                        "The key used to build up the tables is"
                    ):
                        expected_res = line.split(
                            " ",
                        )[-1].strip()
                    elif attribute == "target_tables" and line.startswith("Copy of v_"):
                        k, v = re.findall(r"\.(\w+)", line)
                        expected_res[k] = v
        elif attribute in ["tables_dtypes", "v_tables_dtypes"]:
            expected_res = []
            _actual_res = []
            for _, export_table in qprof.target_tables.items():
                dtypes_map = (
                    vDataFrame(f"select * from qprof_test.{export_table} limit 0")
                    .dtypes()
                    .values
                )
                for col, dtype in zip(dtypes_map["index"], dtypes_map["dtype"]):
                    expected_res.append([col.replace('"', ""), dtype.lower()])

            # changing case of datatype to lower
            for i, _ in enumerate(actual_res):
                for actual in actual_res[i]:
                    _actual_res.append([actual[0], actual[1].lower()])
            actual_res = _actual_res

        elif attribute in ["request", "qduration"]:
            expected_res = getattr(qprof_data, f"{attribute}s")[0]
            # expected_res = f"qprof_data.{attribute}s"[0]
        elif attribute in ["transaction_id", "statement_id"]:
            expected_res = (
                qprof_data.transactions[0][0]
                if attribute == "transaction_id"
                else qprof_data.transactions[0][1]
            )
        elif attribute in ["target_schema"] and schema is None:
            expected_res = None
        elif attribute in ["overwrite"]:
            expected_res = schema is not None
        else:
            expected_res = getattr(qprof_data, attribute)

        logger.info("<<<<<<<<<<<<<<<<<<< Expected result >>>>>>>>>>>>>>>>>>>>>>")
        logger.info(expected_res)

        assert expected_res == actual_res

    @pytest.mark.parametrize("unit", ["s", "m", "h"])
    def test_get_qduration(self, qprof_data, unit):
        """
        test function for get_qduration
        """
        transaction_id, statement_id = qprof_data.transactions[0]

        logger.info("<<<<<<<<<<<<<<<<<<<< actual result >>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        actual_qprof_qdur = QueryProfiler(
            transactions=(transaction_id, statement_id)
        ).get_qduration(unit=unit)

        logger.info("<<<<<<<<<<<<<<<<<<<< expected result >>>>>>>>>>>>>>>>>>>>>>>>>>")
        div = _get_time_div(unit)
        expected_qprof_qdur = float(qprof_data.qdurations[0] / div)
        logger.info(
            f"actual_qprof_qdur: {actual_qprof_qdur}, expected_qprof_qdur: {expected_qprof_qdur}"
        )
        assert actual_qprof_qdur == pytest.approx(expected_qprof_qdur)

    @pytest.mark.parametrize(
        "indent_sql, print_sql, return_html",
        [
            # (True, False, True),  # fail. returns None
            (False, True, False),
            (True, True, True),
            # (False, False, False),  # fail returning sql
        ],
    )
    def test_get_request(self, qprof_data, indent_sql, print_sql, return_html):
        """
        test function for get_request
        """
        transaction_id, statement_id = qprof_data.transactions[0]

        actual_qprof_request = QueryProfiler(
            transactions=(transaction_id, statement_id)
        ).get_request(
            indent_sql=indent_sql, print_sql=print_sql, return_html=return_html
        )
        logger.info(actual_qprof_request)
        if indent_sql and not print_sql and return_html:
            assert actual_qprof_request is not None
        if not indent_sql and print_sql and not return_html:
            assert actual_qprof_request == qprof_data.requests[0]
        elif indent_sql and print_sql and return_html:
            assert qprof_data.request_labels[0] in actual_qprof_request
        else:
            assert actual_qprof_request is None

    def test_get_version(self, qprof_data):
        """
        test function for get_version
        """
        transaction_id, statement_id = qprof_data.transactions[0]
        _actual_qprof_version = QueryProfiler(
            transactions=(transaction_id, statement_id)
        ).get_version()
        actual_qprof_version = f"Vertica Analytic Database v{_actual_qprof_version[0]}.{_actual_qprof_version[1]}.{_actual_qprof_version[2]}-{_actual_qprof_version[3]}"

        expected_qprof_version = (
            current_cursor().execute("select version()").fetchall()[0][0]
        )
        logger.info(
            f"actual vertica version: {actual_qprof_version}, expected vertica version: {expected_qprof_version}"
        )

        assert actual_qprof_version == expected_qprof_version

    @pytest.mark.skip(reason="Need to check all required tables")
    @pytest.mark.parametrize("table_name", [None, "host_resources"])
    def test_get_table(self, qprof_data, table_name):
        """
        test function for get_table
        """

        transaction_id, statement_id = qprof_data.transactions[0]
        actual_qprof_tables = QueryProfiler(
            transactions=(transaction_id, statement_id)
        ).get_table(table_name=table_name)

        logger.info(
            f"actual get_table: {actual_qprof_tables}, expected get_table: {self._target_tables}"
        )
        if table_name:
            assert current_cursor().execute(
                f"select count(*) from {table_name}"
            ).fetchall()[0][0] == len(actual_qprof_tables)
        else:
            assert self._target_tables == actual_qprof_tables

    def test_get_queries(self, qprof_data):
        """
        test function for get_queries

        **Steps to get actual result**
        - Step 1: Get transactions from ``qprof_data`` fixture
        - Step 2: Get ``QueryProfiler`` object by passing ``transaction_id`` and ``statement_id``
                to ``QueryProfiler`` class
        - Step 3: Get all queries and its details using ``get_queries`` method with different
                combinations of parameters (if applicable). ``get_queries``
                method returns vDataFrame.

        **Steps to get expected result**
        - Step 4: Get all queries and its details from ``qprof_data`` fixture

        **Steps to compare actual and expected results**
        - Step 5: compare actual and expected pandas dataframe using pandas compare function.
        - Step 6: compare function returns integers. if return value is ZERO, then actual and expected
                results are same, else it's not same and test will fail.
        """
        # transaction_id, statement_id = qprof_data.transactions[0]
        logger.info("<<<<<<<<<<<<<<<<<<<< actual result >>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        actual_qprof_queries = (
            QueryProfiler(transactions=qprof_data.transactions)
            .get_queries()
            .to_pandas()
            .astype({"qduration": float})
        )
        logger.info(f"Actual Result: {actual_qprof_queries}")

        logger.info("<<<<<<<<<<<<<<<<<< expected result >>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        expected_qprof_queries = pd.DataFrame(
            {
                "index": list(range(len(qprof_data.transactions))),
                "is_current": [True]
                + [False for _ in range(len(qprof_data.transactions) - 1)],
                "transaction_id": [t for t, _ in qprof_data.transactions],
                "statement_id": [s for _, s in qprof_data.transactions],
                "request_label": qprof_data.request_labels,
                "request": [
                    query.strip().replace("\n", "") for query in qprof_data.requests
                ],
                "qduration": [
                    float(duration / 1000000) for duration in qprof_data.qdurations
                ],
            }
        )
        logger.info(f"Expected Result: {expected_qprof_queries}")

        logger.info("<<<<<<<<<<<<<<<<<<<<<< compare result >>>>>>>>>>>>>>>>>>>>>>>>>>>")
        res = _compare_pandas(expected_qprof_queries, actual_qprof_queries)
        logger.info(f"Compare Result: {res}")

        assert len(res) == 0

    @pytest.mark.parametrize(
        "show",
        [
            True,
            False,
        ],
    )
    @pytest.mark.parametrize(
        "category_order",
        [
            # "trace",  # need to implement
            "category ascending",
            "category descending",
            "total ascending",
            "total descending",
            # "min ascending",  # need to implement
            # "min descending",  # need to implement
            # "max ascending",  # need to implement
            # "max descending",  # need to implement
            "sum ascending",
            "sum descending",
            # "mean ascending",  # need to implement
            # "mean descending",  # need to implement
            # "median ascending",  # need to implement
            # "median descending",  # need to implement
        ],
    )
    @pytest.mark.parametrize(
        "kind",
        [
            "bar",
            "barh",
        ],
    )
    @pytest.mark.parametrize(
        "unit",
        [
            "s",
            "m",
            "h",
        ],
    )
    # code always sorts on sub_step desc for show=false
    def test_get_qsteps(self, unit, kind, category_order, show):
        """
        test function for get_qsteps

        - Step 1: Get ``QueryProfiler`` object by passing SQL to ``QueryProfiler`` class

        **Steps to get actual result**
        - Step 2: Get query execution steps chart using ``get_qsteps`` method with different combinations of parameters.
                .. Note::
                    it is drilldown chart (highcharts)
        - Step 3: For ``show=True``
                ``get_qsteps`` method returns highcharts object (drilldown), so it need to unroll.
                get series and drilldown data from highcharts objects, and then join these two datasets
                based on step name.
                get step name from series data, and sub_step name and elapsed_time details from drilldown data

                For ``show=False``
                ``get_qsteps`` method returns vDataFrame.
        - Step 4: convert it to pandas dataframe followed by sorting based on key column(s).

        **Steps to get expected result**
        - Step 5: Get ``completion_time``, ``time``, ``execution_step`` from ``v_internal.dc_query_executions`` table
                using ``transaction_id`` and ``statement_id``
        - Step 6: split ``execution_step`` into ``step`` and ``sub_step`` based on ":" delimiter
        - Step 7: Get ``elapsed_time`` by subtracting ``time`` from ``completion_time``, and Convert ``elapsed_time``
                into appropriate time interval (Seconds/Minutes/Hours)
        - Step 8: if ``category_order`` is category then sort data based on category else sort dataset
                based on below logic:
                # 1. Get all records where ``step`` and ``sub_step`` names are not same. This gives
                    level 2 data in drilldown chart
                # 2. Get all records where ``step`` and ``sub_step`` names are same. This gives
                    level 1 data in drilldown chart
                # 3. Merge ``uneq_sub_step`` (level 2) and e``q_sub_step`` (level 1) based on step name. sort
                    it on ``ascending_flag``

        **Steps to compare actual and expected results**
        - Step 9: compare actual and expected pandas dataframe using pandas compare function.
        - Step 10: compare function returns integers. if return value is ZERO, then actual and expected
                results are same, else it's not same and test will fail.
        """

        qprof = QueryProfiler(transactions=QPROF_SQL2)
        qprof_steps = qprof.get_qsteps(
            unit=unit, kind=kind, categoryorder=category_order, show=show
        )

        # ************************* expected result ************************
        # logic to get step, sub-step and elapsed_time(in s,m and h)
        transaction_id, statement_id = qprof.transactions[0]
        expected_qplans_pdf = vDataFrame(
            f"select completion_time, time, execution_step from v_internal.dc_query_executions where transaction_id={transaction_id} and statement_id={statement_id}"
        ).to_pandas()

        # step and sub_step
        expected_qplans_pdf[["step", "_sub_step"]] = (
            expected_qplans_pdf["execution_step"]
            .str.split(":", expand=True)
            .rename({0: "step", 1: "_sub_step"}, axis=1)
        )
        expected_qplans_pdf["sub_step"] = expected_qplans_pdf.apply(
            lambda x: x["step"] if pd.isnull(x["_sub_step"]) else x["_sub_step"], axis=1
        )

        # elapsed_time
        expected_qplans_pdf["elapsed_time"] = (
            expected_qplans_pdf["completion_time"] - expected_qplans_pdf["time"]
        ).dt.total_seconds()
        div = _get_time_div(unit) / 1000000
        expected_qplans_pdf["elapsed_time"] = round(
            expected_qplans_pdf["elapsed_time"] / div, 10
        )
        expected_qplans_pdf = expected_qplans_pdf[["step", "sub_step", "elapsed_time"]]
        # logger.info(expected_qplans_pdf)

        # ***********************************************************  #
        if show:
            series_data = qprof_steps.data_temp[0].data
            drilldown_data = qprof_steps.drilldown_data_temp
            qsteps_arrays = []
            for _series_data in series_data:
                for _drilldown_data in drilldown_data:
                    if _series_data["name"] == _drilldown_data.name:
                        qsteps_arrays.extend(
                            [
                                (
                                    _series_data["name"],
                                    data[0],
                                    round(data[1], 10),
                                )
                                for data in _drilldown_data.data
                            ]
                        )
            actual_qplans = pd.DataFrame(
                qsteps_arrays, columns=["step", "sub_step", "elapsed_time"]
            )

            # ************ sorting logic ***************************
            ascending_flag = "ascending" in category_order

            sort_key = "step" if "category" in category_order else "elapsed_time"

            # if category_order is category, then it needs be sorted based on step and sub_step name
            if "category" in category_order:
                expected_qplans = (
                    expected_qplans_pdf[["step", "sub_step", "elapsed_time"]]
                    .sort_values(by=["step", "sub_step"], ascending=ascending_flag)
                    .reset_index(drop=True)
                )
            # if category_order is not category, then it needs be sorted based on below logic
            else:
                # 1. Get all records where step and sub_step names are not same.
                # This gives level 2 data in drilldown chart
                uneq_sub_step = (
                    expected_qplans_pdf.loc[
                        (expected_qplans_pdf.sub_step != expected_qplans_pdf.step)
                    ]
                    .sort_values(
                        by=[
                            "step",
                            "sub_step"
                            if "category" in category_order
                            else "elapsed_time",
                        ],
                        ascending=[ascending_flag, ascending_flag],
                    )
                    .reset_index(drop=True)
                )
                # logger.info("Unequal steps .................")
                # logger.info(uneq_sub_step)

                # 2. Get all records where step and sub_step names are same.
                # This gives level 1 data in drilldown chart
                eq_sub_step = (
                    expected_qplans_pdf.loc[
                        (expected_qplans_pdf.sub_step == expected_qplans_pdf.step)
                    ]
                    .sort_values(by=sort_key, ascending=ascending_flag)
                    .reset_index(drop=True)
                )
                # logger.info("Equal steps .................")
                # logger.info(eq_sub_step)

                # 3. Merge uneq_sub_step and eq_sub_step based on step name and sort it based on ascending_flag
                expected_qplans = pd.DataFrame(data=None, columns=uneq_sub_step.columns)
                for step in eq_sub_step["step"].values.tolist():
                    if ascending_flag:
                        expected_qplans = pd.concat(
                            [
                                expected_qplans,
                                (uneq_sub_step[uneq_sub_step["step"] == step]),
                                (eq_sub_step[eq_sub_step["step"] == step]),
                            ]
                        )
                    else:
                        expected_qplans = pd.concat(
                            [
                                expected_qplans,
                                (eq_sub_step[eq_sub_step["step"] == step]),
                                (uneq_sub_step[uneq_sub_step["step"] == step]),
                            ]
                        )
                expected_qplans.reset_index(drop=True, inplace=True)
                # logger.info("Merged unequl and equal sub-plans ........................")
                # logger.info(expected_qplans)
        else:
            actual_qplans = (
                qprof_steps.to_pandas()
                .rename(columns={"substep": "sub_step", "elapsed": "elapsed_time"})
                .sort_values(by=["sub_step"])
                .round({"elapsed_time": 10})
                .reset_index(drop=True)
            )

            expected_qplans = expected_qplans_pdf.sort_values(
                by=["sub_step"]
            ).reset_index(drop=True)

        logger.info("<<<<<<<<<<<<<<<<<<<< actual result >>>>>>>>>>>>>>>>>>>>>>>>>>>")
        logger.info(f"Actual Result: {actual_qplans}")

        logger.info("<<<<<<<<<<<<<<<<< expected result >>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        logger.info(f"Expected Result: {expected_qplans}")

        logger.info("<<<<<<<<<<<<<<<<<<<<<< compare result >>>>>>>>>>>>>>>>>>>>>>>>>>>")
        res = _compare_pandas(expected_qplans, actual_qplans)
        logger.info(f"Compare Result: {res}")

        assert len(res) == 0

    @pytest.mark.parametrize(
        "return_report, print_plan",
        [
            (False, True),
            (True, False),
            (True, True),
            (False, False),
        ],
    )
    def test_get_qplan(self, qprof_data, return_report, print_plan):
        """
        test function for get_qplan

        - Step 1: Get ``QueryProfiler`` object by passing SQL to ``QueryProfiler`` class

        **Steps to get actual result**
        - Step 2: Get query plan chart using ``get_qplan`` method with different
                combinations of parameters (if applicable).
        - Step 3: convert return result to pandas dataframe followed by sorting based on key column(s).

        **Steps to get expected result**
        - Step 4: Get query plan from ``v_internal.dc_explain_plans``
                table using ``transaction_id`` and ``statement_id``, and repeat step 3

        **Steps to compare actual and expected results**
        - Step 5: compare actual and expected pandas dataframe using pandas compare function.
        - Step 6: compare function returns integers. if return value is ZERO, then actual and expected
                results are same, else it's not same and test will fail.
        """
        # need to check. get_qplan is not returning anything with transaction_id and statement_id.
        # Also, its printing report 2 times
        qprof = QueryProfiler(transactions=qprof_data.requests[1])
        _actual_qprof_qplan = qprof.get_qplan(
            return_report=return_report, print_plan=print_plan
        )

        transaction_id, statement_id = qprof.transactions[0]
        query = f"""SELECT statement_id AS stmtid, path_id, path_line_index, path_line FROM v_internal.dc_explain_plans WHERE transaction_id={transaction_id} AND statement_id={statement_id} ORDER BY 1, 2, 3;"""

        if return_report:
            actual_qprof_qplan = _actual_qprof_qplan.sort(
                ["stmtid", "path_id", "path_line_index"]
            ).to_pandas()

            expected_qprof_qplan = (
                vDataFrame(query)
                .to_pandas()
                .sort_values(by=["stmtid", "path_id", "path_line_index"])
                .reset_index(drop=True)
            )
            logger.info("<<<<<<<<<<<<<<<<<< Actual result >>>>>>>>>>>>>>>>>>>>>>>>>>")
            logger.info(f"Actual Result: {actual_qprof_qplan}")

            logger.info("<<<<<<<<<<<<<<<<<<< Expected result >>>>>>>>>>>>>>>>>>>>>>>>>")
            logger.info(f"Expected Result: {expected_qprof_qplan}")

            logger.info("<<<<<<<<<<<<<<<<<<<< compare output >>>>>>>>>>>>>>>>>>>>>>>>")
            res = _compare_pandas(expected_qprof_qplan, actual_qprof_qplan)
            logger.info(f"Compare result: {res}")

            assert len(res) == 0
        else:
            actual_qprof_qplan = _actual_qprof_qplan

            _expected_qprof_qplan = current_cursor().execute(query).fetchall()
            expected_qprof_qplan = "\n".join(
                [line[3] for line in _expected_qprof_qplan]
            )
            logger.info("<<<<<<<<<<<<<<<<<< Actual result >>>>>>>>>>>>>>>>>>>>>>>>>>")
            logger.info(f"Actual Result: {actual_qprof_qplan}")

            logger.info("<<<<<<<<<<<<<<<<<<< Expected result >>>>>>>>>>>>>>>>>>>>>>>>>")
            logger.info(f"Expected Result: {expected_qprof_qplan}")

            assert expected_qprof_qplan == actual_qprof_qplan

    @pytest.mark.parametrize(
        "path_id, path_id_info, show_ancestors, pic_path, return_graphviz",
        [
            # getting error for show_ancestors = False (local variable 'ancestors' referenced before assignment)
            # (3, None, False, None, False),
            (None, 1, True, None, False),
            (None, [0, 1], True, None, False),
            (None, None, True, "/tmp/qplan_tree_test_file", False),
            (None, None, True, None, True),
        ],
    )
    @pytest.mark.parametrize(
        "metric",
        [
            "cost",
            "rows",
            "exec_time_ms",
            "est_rows",
            "proc_rows",
            "prod_rows",
            "rle_prod_rows",
            "clock_time_us",
            # "cstall_us", # ZeroDivisionError
            # "pstall_us", # ZeroDivisionError
            "mem_res_mb",
            # "mem_all_mb",  # ZeroDivisionError
        ],
    )
    def test_get_qplan_tree(
        self,
        path_id,
        path_id_info,
        show_ancestors,
        metric,
        pic_path,
        return_graphviz,
    ):
        """
        test function for get_qplan_tree

        - Step 1: Get ``QueryProfiler`` object by passing SQL to ``QueryProfiler`` class

        **Steps to get actual result**
        - Step 2: Get query plan tree using ``get_qplan_tree`` method with different
                combinations of parameters (if applicable). ``get_qplan_tree`` returns graphviz object
        - Step 3: Convert graphviz object to json object and parse it

        **Steps to get expected result**
        - Step 4: Get query plan using ``get_qplan`` method and parse it

        **Steps to compare actual and expected results**
        - Step 5: compare actual and expected query plan strings
        """
        # need to check. get_qplan_tree is not returning anything with transaction_id, statement_id
        qprof = QueryProfiler(transactions=QPROF_SQL2)
        qprof_qplan_tree = qprof.get_qplan_tree(
            path_id=path_id,
            path_id_info=path_id_info,
            show_ancestors=show_ancestors,
            metric=metric,
            pic_path=pic_path,
            return_graphviz=return_graphviz,
        )
        # actual
        ss1 = ""
        if return_graphviz:
            graph = Digraph()
            source_lines = str(qprof_qplan_tree).splitlines()
            # Remove 'digraph tree {'
            source_lines.pop(0)
            # Remove the closing brackets '}'
            source_lines.pop(-1)
            # Append the nodes to body
            graph.body += source_lines
        else:
            graph = qprof_qplan_tree

        json_string = graph.pipe("json").decode()
        json_dict = json.loads(json_string)
        for obj in json_dict["objects"]:
            if "tooltip" in obj.keys():
                _ss1 = ""
                for s in obj["tooltip"].split("\n"):
                    print(
                        "THIS IS WHAT WE GET:", QprofUtility._get_metrics_name(metric)
                    )
                    if s != "" and not s.startswith(
                        QprofUtility._get_metrics_name(metric)
                    ):
                        _ss1 += s + ""
                    # for return_graphviz = True
                    if return_graphviz and QprofUtility._get_metrics_name(metric) in s:
                        # remove metric info from the string i.e. 'Query plan cost'
                        _ss1 = (
                            re.search(rf".+{QprofUtility._get_metrics_name(metric)}", s)
                            .group(0)
                            .replace(QprofUtility._get_metrics_name(metric), "")
                        )
                ss1 += _ss1.strip().replace('"', "'") + " "
        logger.info("<<<<<<<<<<<<<<<<<<< Actual result >>>>>>>>>>>>>>>>>>>>>>")
        logger.info(f"Actual Result: {ss1}")

        # ***********expected ******************************************
        ss2 = ""
        qplan = qprof.get_qplan().split("\n")
        n = len(qplan)
        rows, tmp_rows = [], []
        for i in range(n):
            if "PATH ID: " in qplan[i] and i > 0:
                rows += ["\n".join(tmp_rows)]
                tmp_rows = []
            tmp_rows += [qplan[i]]
        rows += ["\n".join(tmp_rows)]

        for r in rows:
            _ss2 = ""
            for s in r.split("\n"):
                # replace non alphabet char with ''
                _ss2 += re.sub(r"^\W+", "", s) + ""
            ss2 += _ss2.strip().replace('"', "'") + " "
        logger.info("<<<<<<<<<<<<<<<<<<< Expected result >>>>>>>>>>>>>>>>>>>>>>")
        logger.info(f"Expected Result: {ss2}")

        assert ss2 == ss1

    @pytest.mark.parametrize(
        "show",
        [
            # True,
            False,
        ],
    )
    @pytest.mark.parametrize(
        "kind",
        [
            "bar",
            "barh",
            "pie",
        ],
    )
    @pytest.mark.parametrize(
        "category_order",
        [
            # "trace",
            # "category ascending",
            # "category descending",
            # "total ascending",
            "total descending",
        ],
    )
    @pytest.mark.parametrize(
        "unit",
        [
            "s",
            "m",
            "h",
        ],
    )
    @pytest.mark.skip(
        reason="Getting duplicate rows by running test multiple times with different parameters"
    )
    def test_get_qplan_profile(self, qprof_data, unit, kind, category_order, show):
        """
        test function for get_qplan_profile

        - Step 1: Get ``QueryProfiler`` object by passing SQL to ``QueryProfiler`` class
        **Steps to get actual result**
        - Step 2: Get query plan chart using ``get_qplan_profile`` method with different combinations of parameters.
        - Step 3: For ``show=True``
                ``get_qplan_profile`` method returns plotly graph object. Get x (labels for pie)
                and y(values for pie) axes values from graph object.

                For ``show=False``
                ``get_qplan_profile`` method returns vDataFrame.
        - Step 4: Group result based required columns. convert it to pandas dataframe
                followed by sorting based on key column(s).

        **Steps to get expected result**
        - Step 5: Get query plan from ``v_monitor.query_plan_profiles`` table using ``transaction_id``
                and ``statement_id``, and repeat step 4

        **Steps to compare actual and expected results**
        - Step 6: compare actual and expected pandas dataframe using pandas compare function.
        - Step 7: compare function returns integers. if return value is ZERO, then actual and expected
                results are same, else it's not same and test will fail.
        """

        # need to check. getting error with transaction_id, statement_id
        vp.set_option("plotting_lib", "plotly")
        logger.info(vp.get_option("plotting_lib"))
        div = _get_time_div(unit)

        qprof = QueryProfiler(qprof_data.requests[1])
        qprof_qplan_profile = qprof.get_qplan_profile(
            unit=unit, kind=kind, categoryorder=category_order, show=show
        )
        logger.info("<<<<<<<<<<<<<<<<<<< Actual result >>>>>>>>>>>>>>>>>>>>>>")
        if show:
            logger.info(qprof_qplan_profile.data[0])
            path_line, total_run_time = _get_chart_data(qprof_qplan_profile, kind)
            qprof_qplan_profile_pdf = pd.DataFrame(
                {"path_line": path_line, "total_run_time": total_run_time}
            )
            logger.info(qprof_qplan_profile_pdf)
        else:
            qprof_qplan_profile = qprof_qplan_profile.filter(
                "running_time IS NOT NULL"
            )[["path_line", "running_time"]]
            logger.info(qprof_qplan_profile)
            qprof_qplan_profile_pdf = qprof_qplan_profile.groupby(
                columns=["path_line"],
                expr=[f"sum(running_time) as total_run_time"],
            ).to_pandas()

            logger.info(qprof_qplan_profile_pdf)

        actual_qprof_qplan_profile = (
            qprof_qplan_profile_pdf.astype({"total_run_time": np.float64})
            .sort_values(by=["path_line"])
            .reset_index(drop=True)
        )
        actual_qprof_qplan_profile["total_run_time"] = round(
            actual_qprof_qplan_profile["total_run_time"], 10
        )
        logger.info(f"Actual Result: {actual_qprof_qplan_profile}")

        logger.info("<<<<<<<<<<<<<<<<< Expected result >>>>>>>>>>>>>>>>>>>>>>")
        transaction_id, statement_id = qprof.transactions[0]
        query = f"select left(path_line, 80) AS path_line, sum(running_time) total_run_time FROM v_monitor.query_plan_profiles WHERE transaction_id={transaction_id} and statement_id={statement_id} AND running_time IS NOT NULL group by left(path_line, 80) order by path_line"
        expected_qprof_qplan_profile = vDataFrame(query).to_pandas()
        # logger.info(expected_qprof_qplan_profile['total_run_time'])
        for i in range(len(expected_qprof_qplan_profile["total_run_time"])):
            expected_qprof_qplan_profile["total_run_time"][i] = int(
                f"{expected_qprof_qplan_profile['total_run_time'][i].seconds}{expected_qprof_qplan_profile['total_run_time'][i].microseconds}"
            )
        expected_qprof_qplan_profile["total_run_time"] = (
            expected_qprof_qplan_profile["total_run_time"] / div
        )
        expected_qprof_qplan_profile = (
            expected_qprof_qplan_profile.astype({"total_run_time": np.float64})
            .sort_values(by=["path_line"])
            .reset_index(drop=True)
        )
        expected_qprof_qplan_profile["total_run_time"] = round(
            expected_qprof_qplan_profile["total_run_time"], 10
        )
        logger.info(f"Expected Result: {expected_qprof_qplan_profile}")

        logger.info("<<<<<<<<<<<<<<<<< comparison result >>>>>>>>>>>>>>>>>>>>>")
        res = _compare_pandas(expected_qprof_qplan_profile, actual_qprof_qplan_profile)
        logger.info(f"Compare Result: {res}")

        assert len(res) == 0

    def test_get_query_events(self, qprof_data):
        """
        test function for get_query_events

        **Steps to get actual result**
        - Step 1: Get ``transaction_id`` and ``statement_id`` from ``qprof_data`` fixture
        - Step 2: Get ``QueryProfiler`` object by passing ``transaction_id`` and ``statement_id``
                to ``QueryProfiler`` class
        - Step 3: Get query events details using ``get_query_events`` method with different
                combinations of parameters (if applicable). ``get_query_events``
                method returns vDataFrame.

        **Steps to get expected result**
        - Step 4: Get query events from ``v_monitor.query_events`` table using ``transaction_id`` and ``statement_id``

        **Steps to compare actual and expected results**
        - Step 5: compare actual and expected pandas dataframe using pandas compare function.
        - Step 6: compare function returns integers. if return value is ZERO, then actual and expected
                results are same, else it's not same and test will fail.
        """
        transaction_id, statement_id = qprof_data.transactions[2]

        logger.info("<<<<<<<<<<<<<<<<<<<<<<< Actual result >>>>>>>>>>>>>>>>>>>>>>>>")
        actual_qprof_events = (
            QueryProfiler(transactions=(transaction_id, statement_id))
            .get_query_events()
            .to_pandas()
        )
        logger.info(actual_qprof_events)

        logger.info("<<<<<<<<<<<<<<<<<<<<<<< Expected result >>>>>>>>>>>>>>>>>>>>>>>>")
        query = f"SELECT event_timestamp, node_name, event_category, event_type, event_description, operator_name, path_id, event_details, suggested_action FROM v_monitor.query_events WHERE transaction_id={transaction_id} AND statement_id={statement_id} ORDER BY 1"
        expected_qprof_events = vDataFrame(query).to_pandas()
        logger.info(expected_qprof_events)

        logger.info("<<<<<<<<<<<<<<<<<<<<<<< Compare result >>>>>>>>>>>>>>>>>>>>>>>>")
        res = _compare_pandas(expected_qprof_events, actual_qprof_events)
        logger.info(res)

        assert len(res) == 0

    @pytest.mark.parametrize(
        "reverse, show",
        [
            (False, False),
            # (False, True), Error - 'Axes' object has no attribute 'data'
            # (True, True),  # need to implement when reverse is True
            (True, False),
        ],
    )
    @pytest.mark.parametrize("kind", ["bar", "barh"])
    @pytest.mark.parametrize(
        "category_order",
        [
            # "trace",
            # "category ascending",
            # "category descending",
            # "total ascending",
            "total descending",
            # "min ascending",
            # "min descending",
            # "max ascending",
            # "max descending",
            # "sum ascending",
            # "sum descending",
            # "mean ascending",
            # "mean descending",
            # "median ascending",
            # "median descending",
        ],
    )
    def test_get_cpu_time(self, category_order, kind, reverse, show):
        """
        test function for get_cpu_time

        - Step 1: Get ``QueryProfiler`` object by passing SQL to ``QueryProfiler`` class
        **Steps to get actual result**
        - Step 2: Get cpu time chart using get_cpu_time method with different combinations of parameters.
        - Step 3: For ``show=True``
                get_cpu_time method returns plotly graph object. Get x (labels for pie)
                and y(values for pie) axes values from graph object.

                For ``show=False``
                ``get_cpu_time`` method returns vDataFrame.
        - Step 4: Group result based required columns. convert it to pandas dataframe
                followed by sorting based on key column(s).

        **Steps to get expected result**
        - Step 5: Get cpu time from ``v_monitor.execution_engine_profiles`` table using ``transaction_id``
                and ``statement_id``, and repeat step 4

        **Steps to compare actual and expected results**
        - Step 6: compare actual and expected pandas dataframe using pandas compare function.
        - Step 7: compare function returns integers. if return value is ZERO, then actual and expected
                results are same, else it's not same and test will fail.
        """
        # need to check. getting error with transaction_id and statement_id
        vp.set_option("plotting_lib", "plotly")
        logger.info(vp.get_option("plotting_lib"))

        qprof = QueryProfiler(QPROF_SQL2)
        actual_qprof_cpu_time_raw = qprof.get_cpu_time(
            kind=kind, reverse=reverse, categoryorder=category_order, show=show
        )

        transaction_id, statement_id = qprof.transactions[0]
        query = f"SELECT node_name, path_id, counter_value counter_value FROM v_monitor.execution_engine_profiles WHERE TRIM(counter_name) = 'execution time (us)' and transaction_id={transaction_id} AND statement_id={statement_id}"
        expected_qprof_cpu_time_raw = vDataFrame(query)

        logger.info(actual_qprof_cpu_time_raw)
        # logger.info(dir(actual_qprof_cpu_time_raw))
        if show:
            node_name = actual_qprof_cpu_time_raw.data[0].name
            path_id, counter_value = _get_chart_data(actual_qprof_cpu_time_raw, kind)
            actual_qprof_cpu_time_pdf = pd.DataFrame(
                {
                    "node_name": node_name,
                    "path_id": path_id,
                    "counter_value": counter_value,
                }
            )
            expected_qprof_cpu_time_pdf = expected_qprof_cpu_time_raw.groupby(
                columns=["node_name", "path_id"],
                expr=["sum(counter_value) as counter_value"],
            ).to_pandas()
        else:
            actual_qprof_cpu_time_pdf = actual_qprof_cpu_time_raw.to_pandas()
            expected_qprof_cpu_time_pdf = expected_qprof_cpu_time_raw.to_pandas()

        actual_qprof_cpu_time = (
            actual_qprof_cpu_time_pdf.astype({"path_id": np.int64})
            .sort_values(by=["node_name", "path_id", "counter_value"])
            .reset_index(drop=True)
        )
        expected_qprof_cpu_time = expected_qprof_cpu_time_pdf.sort_values(
            by=["node_name", "path_id", "counter_value"]
        ).reset_index(drop=True)

        logger.info("<<<<<<<<<<<<<<<<<<<<<<< Actual result >>>>>>>>>>>>>>>>>>>>>>>>")
        logger.info(f"Actual Result: {actual_qprof_cpu_time}")

        logger.info("<<<<<<<<<<<<<<<<<<<<< Expected result >>>>>>>>>>>>>>>>>>>>>>>>>")
        logger.info(f"Expected Result: {expected_qprof_cpu_time}")

        logger.info("<<<<<<<<<<<<<<<<<<<<< compare result >>>>>>>>>>>>>>>>>>>>>>>>>")
        res = _compare_pandas(expected_qprof_cpu_time, actual_qprof_cpu_time)
        logger.info(f"Compare Result: {res}")

        assert len(res) == 0

    def test_get_qexecution_report(self):
        """
        test function for get_qexecution_report

        - Step 1: Get ``QueryProfiler`` object by passing SQL to ``QueryProfiler`` class
        **Steps to get actual result**
        - Step 2: Get query execution report using ``get_qexecution_report`` method with different
                combinations of parameters (if applicable). ``get_qexecution_report`` method
                returns vDataFrame.
        - Step 3: convert return result to pandas dataframe followed by sorting based on key column(s).

        **Steps to get expected result**
        - Step 4: Get query execution report from ``v_monitor.execution_engine_profiles``
                table using ``transaction_id`` and ``statement_id``, and repeat step 3

        **Steps to compare actual and expected results**
        - Step 5: compare actual and expected pandas dataframe using pandas compare function.
        - Step 6: compare function returns integers. if return value is ZERO, then actual and expected
                results are same, else it's not same and test will fail.
        """

        # need to check. no results, with transaction_id, statement_id
        # current_cursor().execute("create schema if not exists qprof_test")
        # logger.info(qprof_data)
        qprof_common = QueryProfiler(transactions=QPROF_SQL2)

        logger.info("<<<<<<<<<<<<<<<<<<< Actual result >>>>>>>>>>>>>>>>>>>>>>")
        actual_qprof_qexecution_report = (
            qprof_common.get_qexecution_report()
            .to_pandas()[["node_name", "operator_name", "path_id"]]
            .sort_values(by=["node_name", "operator_name", "path_id"])
            .reset_index(drop=True)
        )
        logger.info(f"Actual Result: {actual_qprof_qexecution_report}")

        logger.info("<<<<<<<<<<<<<<<<< Expected result >>>>>>>>>>>>>>>>>>>>>>")
        transaction_id, statement_id = qprof_common.transactions[0]
        query = f"SELECT node_name, operator_name, path_id FROM v_monitor.execution_engine_profiles WHERE transaction_id={transaction_id} AND statement_id={statement_id} group by node_name, operator_name, path_id"
        expected_qprof_qexecution_report = (
            vDataFrame(query)
            .to_pandas()
            .sort_values(by=["node_name", "operator_name", "path_id"])
            .reset_index(drop=True)
        )
        logger.info(f"Expected Result: {expected_qprof_qexecution_report}")

        res = _compare_pandas(
            expected_qprof_qexecution_report, actual_qprof_qexecution_report
        )
        logger.info("<<<<<<<<<<<<<<< compare result >>>>>>>>>>>>>>>>>")
        logger.info(f"Compare result: {res}")

        assert len(res) == 0

    @pytest.mark.parametrize(
        "node_name, path_id, multi, rows, cols",
        [
            # (None, None, True, 3, 3),
            (None, None, False, 3, 3),
        ],
    )
    @pytest.mark.parametrize(
        "show",
        [
            False,
            # True, Error - 'Axes' object has no attribute 'data'
        ],
    )
    # need to check. order in not changing
    @pytest.mark.parametrize(
        "categoryorder",
        [
            # "trace",
            # "category ascending",
            # "category descending",
            "total ascending",
            # "total descending",
            # "min ascending",
            # "min descending",
            # "max ascending",
            # "max descending",
            # "sum ascending",
            # "sum descending",
            # "mean ascending",
            # "mean descending",
            # "median ascending",
            # "median descending",
        ],
    )
    @pytest.mark.parametrize(
        "metric",
        [
            # "all",
            # "exec_time_ms",
            "est_rows",
            # "proc_rows",
            # "prod_rows",
            # "rle_prod_rows",
            # "clock_time_us",
            # "cstall_us",
            # "pstall_us",
            # "mem_res_mb",
            # "mem_all_mb",
        ],
    )
    @pytest.mark.parametrize(
        "kind",
        [
            "bar",
            "barh",
            "pie",
        ],
    )
    def test_get_qexecution(
        self,
        node_name,
        metric,
        path_id,
        kind,
        multi,
        categoryorder,
        rows,
        cols,
        show,
    ):
        """
        test function for get_qexecution

        - Step 1: Get ``QueryProfiler`` object by passing SQL to ``QueryProfiler`` class
        **Steps to get actual result**
        - Step 2: Get query execution chart using ``get_qexecution`` method with different combinations of parameters.
        - Step 3: For ``show=True``
                ``get_qexecution`` method returns plotly graph object. Get x (labels for pie)
                and y(values for pie) axes values from graph object.

                For ``show=False``
                ``get_qexecution`` method returns vDataFrame.
        - Step 4: Group result based required columns. convert it to pandas dataframe
                followed by sorting based on key column(s).

        **Steps to get expected result**
        - Step 5: Get query execution report (vDataframe) using ``get_qexecution_report`` method, and repeat step 4

        **Steps to compare actual and expected results**
        - Step 6: compare actual and expected pandas dataframe using pandas compare function.
        - Step 7: compare function returns integers. if return value is ZERO, then actual and expected
                results are same, else it's not same and test will fail.
        """
        # need to check. getting error
        # need to check as target_schema is not getting created by vpy
        # logger.info(qprof_data.transactions[0])
        # transaction_id, statement_id = qprof_data.transactions[0]

        # setting default plotting to plotly
        vp.set_option("plotting_lib", "plotly")
        logger.info(vp.get_option("plotting_lib"))

        qprof = QueryProfiler(transactions=QPROF_SQL2)
        qprof_qexecution = qprof.get_qexecution(
            node_name=node_name,
            metric=metric,
            path_id=path_id,
            kind=kind,
            multi=multi,
            categoryorder=categoryorder,
            rows=rows,
            cols=cols,
            show=show,
        )
        # logger.info(qprof_qexecution)
        logger.info("<<<<<<<<<<<<<<<<<<<<<<< Actual result >>>>>>>>>>>>>>>>>>>>>>>>")
        if show:
            operator_name, execution_time = _get_chart_data(qprof_qexecution, kind)
            actual_qprof_qexecution_pdf = pd.DataFrame(
                {"operator_name": operator_name, metric: execution_time}
            )
            logger.info(actual_qprof_qexecution_pdf)
        else:
            actual_qprof_qexecution_pdf = qprof_qexecution.groupby(
                columns=["operator_name"],
                expr=[f"sum({metric}) as {metric}"],
            ).to_pandas()
            logger.info(actual_qprof_qexecution_pdf)

        actual_qprof_qexecution = (
            actual_qprof_qexecution_pdf.replace({"operator_name": {"Others": "Root"}})
            .fillna(0)
            .astype({f"{metric}": np.float64})
            .sort_values(by=["operator_name"])
            .reset_index(drop=True)
        )
        logger.info(f"Actual Result: {actual_qprof_qexecution}")

        logger.info("<<<<<<<<<<<<<<<<<<<<< Expected result >>>>>>>>>>>>>>>>>>>>>>>>>")
        expected_qprof_qexecution_raw = qprof.get_qexecution_report()[
            ["operator_name", metric]
        ]
        expected_qprof_qexecution_pdf = expected_qprof_qexecution_raw.groupby(
            columns=["operator_name"],
            expr=[f"sum({metric}) as {metric}"],
        ).to_pandas()

        expected_qprof_qexecution = (
            expected_qprof_qexecution_pdf.fillna(0)
            .astype({f"{metric}": np.float64})
            .sort_values(by=["operator_name"])
            .reset_index(drop=True)
        )
        logger.info(f"Expected Result: {expected_qprof_qexecution}")

        logger.info("<<<<<<<<<<<<<<<<<<<<< compare result >>>>>>>>>>>>>>>>>>>>>>>>")
        res = _compare_pandas(expected_qprof_qexecution, actual_qprof_qexecution)
        logger.info(f"Compare Result: {res}")

        assert len(res) == 0

    def test_get_rp_status(self, qprof_data):
        """
        test function for get_rp_status

        **Steps to get actual result**
        - Step 1: Get transaction_id, and statement_id from ``qprof_data`` fixture
        - Step 2: Pass ``transaction_id`` and ``statement_id`` to ``QueryProfiler`` object, and get
                RP status using ``get_rp_status`` method (``get_rp_status`` method returns vDataFrame)
        - Step 3: convert return result to pandas dataframe followed by sorting based on key column(s).

        **Steps to get expected result**
        - Step 4: Get RP status from ``v_monitor.resource_pool_status`` table, and repeat step 3

        **Steps to compare actual and expected results**
        - Step 5: compare actual and expected pandas dataframe using pandas compare function.
        - Step 6: compare function returns integers. if return value is ZERO, then actual and expected
                results are same, else it's not same and test will fail.
        """
        logger.info("<<<<<<<<<<<<<<<<<<<<<<< Actual result >>>>>>>>>>>>>>>>>>>>>>>>")
        transaction_id, statement_id = qprof_data.transactions[1]
        actual_qprof_rp = (
            QueryProfiler(transactions=(transaction_id, statement_id))
            .get_rp_status()
            .to_pandas()
            .sort_values(by="pool_oid")
            .reset_index(drop=True)
        )
        logger.info(f"Actual Result: {actual_qprof_rp}")

        logger.info("<<<<<<<<<<<<<<<<<<<<<<< Expected result >>>>>>>>>>>>>>>>>>>>>>>>")
        query = "SELECT * FROM v_monitor.resource_pool_status ORDER BY pool_oid"
        expected_qprof_rp = vDataFrame(query).to_pandas()
        logger.info(f"Expected Result: {expected_qprof_rp}")

        logger.info("<<<<<<<<<<<<<<<<<<<<< compare result >>>>>>>>>>>>>>>>>>>>>>>>")
        res = _compare_pandas(expected_qprof_rp, actual_qprof_rp)
        logger.info(f"Compare Result: {res}")

        assert len(actual_qprof_rp) == len(expected_qprof_rp)

    @pytest.mark.skip(reason="failing when test runs in parallel")
    def test_get_cluster_config(self, qprof_data):
        """
        test function for get_cluster_config

        **Steps to get actual result**
        - Step 1: Get ``transaction_id`` and ``statement_id`` from ``qprof_data`` fixture
        - Step 2: Get ``QueryProfiler`` object by passing ``transaction_id`` and ``statement_id``
                to ``QueryProfiler`` class
        - Step 3: Get cluster configuration details using ``get_cluster_config`` method with different
                combinations of parameters (if applicable). ``get_cluster_config``
                method returns vDataFrame.
        - Step 4: convert return result to pandas dataframe followed by sorting based on key column(s).

        **Steps to get expected result**
        - Step 5: Get cluster configuration from ``v_monitor.host_resources`` table, and repeat step 4

        **Steps to compare actual and expected results**
        - Step 6: compare actual and expected pandas dataframe using pandas compare function.
        - Step 7: compare function returns integers. if return value is ZERO, then actual and expected
                results are same, else it's not same and test will fail.
        """
        transaction_id, statement_id = qprof_data.transactions[2]
        logger.info("<<<<<<<<<<<<<<<<<<<<<<< Actual result >>>>>>>>>>>>>>>>>>>>>>>>")
        actual_qprof_cluster_config = (
            QueryProfiler(transactions=(transaction_id, statement_id))
            .get_cluster_config()
            .to_pandas()
            .sort_values(by="host_name")
            .reset_index(drop=True)
        )
        logger.info(f"Actual Result: {actual_qprof_cluster_config}")

        logger.info("<<<<<<<<<<<<<<<<<<<<<<< Expected result >>>>>>>>>>>>>>>>>>>>>>>>")
        query = "SELECT * FROM v_monitor.host_resources ORDER BY host_name"
        expected_qprof_cluster_config = vDataFrame(query).to_pandas()
        logger.info(f"Expected Result: {expected_qprof_cluster_config}")

        logger.info("<<<<<<<<<<<<<<<<<<<<< compare result >>>>>>>>>>>>>>>>>>>>>>>>")
        res = _compare_pandas(
            expected_qprof_cluster_config, actual_qprof_cluster_config
        )
        logger.info(f"Compare Result: {res}")

        assert len(res) == 0
