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
import io
import re
import logging
from contextlib import redirect_stdout
import pandas as pd
import numpy as np
import pytest
from verticapy.performance.vertica import QueryProfiler
from verticapy.connection import current_cursor
from verticapy.core.vdataframe import vDataFrame

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
logger = logging.getLogger(__name__)


class TestQueryProfiler:
    """
    test class for QueryProfiler
    """

    query2 = f"""SELECT /*+LABEL('QueryProfiler_query_requests_independent')*/ transaction_id, statement_id, request, request_duration from query_requests where start_timestamp > now() - interval'1 hour' order by request_duration desc limit 10;"""

    # def get_trans_statement_id(self):
    #     print('Running v_monitor.query_requests to get transaction_id, statement_id ............................')
    #     sql = f"select transaction_id, statement_id from v_monitor.query_requests where request_label = 'QueryProfiler_query_requests_independent' ORDER BY start_timestamp DESC LIMIT 1"
    #     res = current_cursor().execute(sql).fetchall()
    #     print(res)
    #     transaction_id, statement_id = res[0] if isinstance(res[0], list) == 1 else res
    #     print(f"transaction_id: {transaction_id}, statement_id: {statement_id}")
    #     return transaction_id, statement_id

    @pytest.fixture(name="qprof_data", scope="class")
    def data_setup(self, amazon_vd, titanic_vd, schema_loader):
        """
        test function for query_profiler
        """
        transactions = []
        qdurations = []
        query_label_map = {
            "query1": "QueryProfiler_titanic_UT",
            "query2": "QueryProfiler_query_requests_UT",
        }

        query1 = f"""SELECT /*+LABEL('QueryProfiler_titanic_UT')*/ ticket, substr(ticket, 1, 5) AS ticket, AVG(age) AS avg_age FROM {schema_loader}.titanic GROUP BY 1"""
        query2 = f"""SELECT /*+LABEL('QueryProfiler_query_requests_UT')*/ transaction_id, statement_id, request, request_duration from query_requests where start_timestamp > now() - interval'1 hour' order by request_duration desc limit 10"""

        queries = [query1, query2]

        for key, query in zip(query_label_map.keys(), queries):
            current_cursor().execute(query)
            sql = f"select transaction_id, statement_id from query_requests where request_label = '{query_label_map[key]}' and request not like 'PROFILE%' ORDER BY start_timestamp DESC LIMIT 1"
            res = current_cursor().execute(sql).fetchall()
            print(res)
            transaction_id, statement_id = (
                res[0] if isinstance(res[0], list) == 1 else res
            )
            transactions.append((transaction_id, statement_id))

            qduration_sql = f"""SELECT query_duration_us FROM v_monitor.query_profiles WHERE transaction_id={transaction_id} AND statement_id={statement_id}"""
            qduration_res = current_cursor().execute(qduration_sql).fetchall()[0][0]
            print(qduration_res)
            qdurations.append(qduration_res)
            print(qdurations)

        qprof_attr_map = {
            "transactions": transactions,
            "requests": queries,
            "request_labels": list(query_label_map.values()),
            "qdurations": qdurations,
            "key_id": None,
            # "request": queries[0],
            # "qduration": qdurations[0],
            # "transaction_id": transactions[0][0],
            # "statement_id": transactions[0][1],
            "target_schema": {"v_internal": "qprof_test", "v_monitor": "qprof_test"},
            "target_tables": {},
            "v_tables_dtypes": [],
            "tables_dtypes": [],
            "overwrite": False,
            # "query1": queries[0],
            # "query2": queries[1],
        }

        return qprof_attr_map

    @pytest.mark.parametrize(
        "transactions, key_id, resource_pool, target_schema, overwrite, add_profile, check_tables",
        [
            # ("integer", None, None, None, False, False, True),
            # ("list_of_integers", None, None, None, False, False, True),
            # ("tuple", None, None, None, False, False, True),
            # ("list_of_tuples", None, None, None, False, False, True),
            # ("single_sql", None, None, None, False, False, True),
            # ("multiple_sql", None, None, None, False, False, True),
            ("tuple", None, None, None, True, False, True),  # overwrite
            # ("single_sql", None, None, None, False, True, True),  # add_profile
            # ("integer", None, None, None, False, False, False),  # check_tables
        ],
    )
    def test_query_profiler(
        self,
        qprof_data,
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
        """

        # query = self.data_setup()
        if transactions == "integer" and not add_profile:
            qprof = QueryProfiler(transactions=qprof_data["transactions"][1][0])
            expected_res = (qprof_data["transactions"][1][0], 1)
            actual_res = qprof.transactions[0]
        elif transactions == "list_of_integers":
            qprof = QueryProfiler(
                transactions=[
                    qprof_data["transactions"][0][0],
                    qprof_data["transactions"][1][0],
                ]
            )
            expected_res = [
                (qprof_data["transactions"][0][0], 1),
                (qprof_data["transactions"][1][0], 1),
            ]
            actual_res = qprof.transactions
        elif transactions == "tuple" and not overwrite:
            qprof = QueryProfiler(transactions=qprof_data["transactions"][1])
            expected_res = qprof_data["transactions"][1]
            actual_res = qprof.transactions[0]
        elif transactions == "list_of_tuples":
            qprof = QueryProfiler(transactions=qprof_data["transactions"])
            expected_res = qprof_data["transactions"]
            actual_res = qprof.transactions
        elif transactions == "single_sql" and not add_profile:
            qprof = QueryProfiler(
                transactions=qprof_data["requests"][0], add_profile=add_profile
            )
            expected_res = qprof_data["requests"][0]
            actual_res = qprof.request
        elif transactions == "multiple_sql":
            qprof = QueryProfiler(
                transactions=qprof_data["requests"], add_profile=add_profile
            )
            expected_res = qprof_data["requests"]
            actual_res = qprof.requests
        elif transactions is None and key_id:
            qprof = QueryProfiler(
                transactions=(qprof_data[1][0], qprof_data[2][0]),
                target_schema="qprof_test",
                overwrite=False,
            )
        elif transactions == "tuple" and overwrite:
            logger.info("Running test with overwrite set to True ...................")
            qprof = QueryProfiler(transactions=qprof_data["transactions"][1], overwrite=overwrite)
            expected_res = qprof_data["transactions"][1]
            actual_res = qprof.transactions[0]
        elif transactions == "single_sql" and add_profile:
            logger.info("Running test with add_profile set to True ...................")
            qprof = QueryProfiler(
                transactions=qprof_data["requests"][0], add_profile=add_profile
            )
            expected_res = True
            actual_res = True if "PROFILE" in qprof.request else False
        # else:
        #     qprof = QueryProfiler(transactions=(qprof_data["transactions"][1]))

        logger.info(f"Expected result: {expected_res},  Actual result: {actual_res}")
        # assert expected_res == actual_res

    # @pytest.mark.parametrize(
    #     "attribute, schema",
    #     [
    #         ("transactions", None),
    #         ("requests", None),
    #         ("request_labels", None),
    #         ("qdurations", None),
    #         # ("key_id", None),  # fail. no key id in log
    #         ("key_id", "qprof_test"),
    #         ("request", None),
    #         ("qduration", None),
    #         ("transaction_id", None),
    #         ("statement_id", None),
    #         ("target_schema", None),
    #         ("target_schema", "qprof_test"),
    #         ("target_tables", None),  # with no schema, no table is listed
    #         ("target_tables", "qprof_test"),
    #         ("v_tables_dtypes", None),  # with no schema, no tables is listed
    #         ("v_tables_dtypes", "qprof_test"),
    #         ("tables_dtypes", None),  # with no schema, no datatype is generated
    #         ("tables_dtypes", "qprof_test"),
    #         ("overwrite", None),
    #         ("overwrite", "qprof_test"),
    #     ],
    # )
    # def test_qprof_attributes(self, qprof_data, attribute, schema):
    #     print(qprof_data)
    #     # need to check as target_schema is not getting created by vpy
    #     current_cursor().execute(f"create schema if not exists qprof_test")
    #     if (
    #         attribute
    #         in [
    #             "key_id",
    #             "target_schema",
    #             "target_tables",
    #             "tables_dtypes",
    #             "overwrite",
    #         ]
    #         and schema
    #     ):
    #         # reading stdout for key_id
    #         f = io.StringIO()
    #         with redirect_stdout(f):
    #             qprof = QueryProfiler(
    #                 transactions=qprof_data["transactions"][0],
    #                 target_schema="qprof_test",
    #                 overwrite=True if attribute == "overwrite" else False,
    #             )
    #         s = f.getvalue()
    #         # print(s)
    #     else:
    #         qprof = QueryProfiler(transactions=qprof_data["transactions"])
    #
    #     actual_res = getattr(qprof, attribute)
    #     print("<<<<<<<<<<<<<<<<<<< Actual result >>>>>>>>>>>>>>>>>>>>>>")
    #     print(actual_res)
    #     if attribute in ["key_id", "target_tables"]:
    #         expected_res = "" if attribute == "key_id" else {}
    #         if schema:
    #             for line in s.splitlines():
    #                 # print(line)
    #                 if attribute == "key_id" and line.startswith(
    #                     "The key used to build up the tables is"
    #                 ):
    #                     expected_res = line.split(
    #                         " ",
    #                     )[-1].strip()
    #                 elif attribute == "target_tables" and line.startswith("Copy of v_"):
    #                     k, v = re.findall(r"\.(\w+)", line)
    #                     expected_res[k] = v
    #     elif attribute in ["tables_dtypes", "v_tables_dtypes"]:
    #         expected_res = []
    #         _actual_res = []
    #         for table, export_table in qprof.target_tables.items():
    #             dtypes_map = (
    #                 vDataFrame(f"select * from qprof_test.{export_table} limit 0")
    #                 .dtypes()
    #                 .values
    #             )
    #             for col, dtype in zip(dtypes_map["index"], dtypes_map["dtype"]):
    #                 expected_res.append([col.replace('"', ""), dtype.lower()])
    #
    #         # changing case of datatype to lower
    #         for i in range(len(actual_res)):
    #             for j in range(len(actual_res[i])):
    #                 _actual_res.append(
    #                     [actual_res[i][j][0], actual_res[i][j][1].lower()]
    #                 )
    #         actual_res = _actual_res
    #
    #     elif attribute in ["request", "qduration"]:
    #         expected_res = qprof_data[f"{attribute}s"][0]
    #     elif attribute in ["transaction_id", "statement_id"]:
    #         expected_res = (
    #             qprof_data["transactions"][0][0]
    #             if attribute == "transaction_id"
    #             else qprof_data["transactions"][0][1]
    #         )
    #     elif attribute in ["target_schema"] and schema is None:
    #         expected_res = None
    #     elif attribute in ["overwrite"]:
    #         expected_res = True if schema else False
    #     else:
    #         expected_res = qprof_data[attribute]
    #
    #     print("<<<<<<<<<<<<<<<<<<< Expected result >>>>>>>>>>>>>>>>>>>>>>")
    #     print(expected_res)
    #
    #     assert expected_res == actual_res
    #
    # @pytest.mark.parametrize("unit", ["s", "m", "h"])
    # def test_get_qduration(self, qprof_data, unit):
    #     current_cursor().execute("create schema if not exists qprof_test")
    #
    #     transaction_id, statement_id = qprof_data["transactions"][0]
    #     actual_qprof_qdur = QueryProfiler(
    #         transactions=(transaction_id, statement_id)
    #     ).get_qduration(unit=unit)
    #
    #     expected_qprof_qdur = float(qprof_data["qduration"] / 1000000)
    #     if unit == "m":
    #         expected_qprof_qdur = expected_qprof_qdur / 60
    #     elif unit == "h":
    #         expected_qprof_qdur = expected_qprof_qdur / 3600
    #     else:
    #         expected_qprof_qdur = expected_qprof_qdur
    #
    #     print(
    #         f"actual_qprof_qdur: {actual_qprof_qdur}, expected_qprof_qdur: {expected_qprof_qdur}"
    #     )
    #     assert actual_qprof_qdur == pytest.approx(expected_qprof_qdur)
    #
    # @pytest.mark.parametrize(
    #     "indent_sql, print_sql, return_html",
    #     [
    #         (True, False, True),
    #         (False, True, False),
    #         (True, True, True),
    #         (False, False, False),
    #     ],
    # )
    # def test_get_request(self, qprof_data, indent_sql, print_sql, return_html):
    #     current_cursor().execute("create schema if not exists qprof_test")
    #
    #     transaction_id, statement_id = qprof_data["transactions"][0]
    #     actual_qprof_request = QueryProfiler(
    #         transactions=(transaction_id, statement_id)
    #     ).get_request(
    #         indent_sql=indent_sql, print_sql=print_sql, return_html=return_html
    #     )
    #
    #     # print(
    #     #     f"actual_qprof_qdur: {actual_qprof_request}, expected_qprof_qdur: {expected_qprof_qdur}"
    #     # )
    #     # assert actual_qprof_qdur == pytest.approx(expected_qprof_qdur)
    #
    # def test_get_verion(self, qprof_data):
    #     current_cursor().execute("create schema if not exists qprof_test")
    #
    #     transaction_id, statement_id = qprof_data["transactions"][0]
    #     _actual_qprof_version = QueryProfiler(
    #         transactions=(transaction_id, statement_id)
    #     ).get_version()
    #     actual_qprof_version = f"Vertica Analytic Database v{_actual_qprof_version[0]}.{_actual_qprof_version[1]}.{_actual_qprof_version[2]}-{_actual_qprof_version[3]}"
    #
    #     expected_qprof_version = (
    #         current_cursor().execute("select version()").fetchall()[0][0]
    #     )
    #     print(
    #         f"actual vertica version: {actual_qprof_version}, expected vertica version: {expected_qprof_version}"
    #     )
    #
    #     assert actual_qprof_version == expected_qprof_version
    #
    # def test_get_table(self, qprof_data):  # need to check on table names
    #     expected_qprof_tables = [
    #         "dc_requests_issued",
    #         "dc_query_executions",
    #         "dc_explain_plans",
    #         "execution_engine_profiles",
    #         "query_plan_profiles",
    #         "query_profiles",
    #         "resource_pool_status",
    #         "host_resources",
    #     ]
    #     transaction_id, statement_id = qprof_data["transactions"][0]
    #     actual_qprof_tables = QueryProfiler(
    #         transactions=(transaction_id, statement_id)
    #     ).get_table()
    #
    #     assert expected_qprof_tables == actual_qprof_tables
    #
    # def test_get_queries(self, qprof_data):  # need to check on table names
    #     current_cursor().execute("create schema if not exists qprof_test")
    #
    #     transaction_id, statement_id = qprof_data["transactions"][0]
    #     actual_qprof_queries = (
    #         QueryProfiler(transactions=qprof_data["transactions"])
    #         .get_queries()
    #         .to_pandas()
    #         .astype({"qduration": float})
    #     )
    #     print(actual_qprof_queries)
    #
    #     expected_qprof_queries = pd.DataFrame(
    #         {
    #             "index": [i for i in range(len(qprof_data["transactions"]))],
    #             "is_current": [True]
    #             + [False for _ in range(len(qprof_data["transactions"]) - 1)],
    #             "transaction_id": [t for t, _ in qprof_data["transactions"]],
    #             "statement_id": [s for _, s in qprof_data["transactions"]],
    #             "request_label": qprof_data["request_labels"],
    #             "request": [
    #                 query.strip().replace("\n", "") for query in qprof_data["requests"]
    #             ],
    #             "qduration": [
    #                 float(duration / 1000000) for duration in qprof_data["qdurations"]
    #             ],
    #         }
    #     )
    #     print(expected_qprof_queries)
    #
    #     res = expected_qprof_queries.compare(
    #         actual_qprof_queries,
    #         result_names=(
    #             "left",
    #             "right",
    #         ),
    #     )
    #     print(res)
    #
    #     assert False if len(res) > 0 else True
    #
    # @pytest.mark.parametrize(
    #     "unit, kind, category_order, show",
    #     [
    #         ("s", "bar", "sum descending", True),
    #         # ("m", "bar", "sum descending", True),
    #         # ("h", "bar", "sum descending", True),
    #         ("s", "barh", "sum descending", True),
    #         ("s", "bar", "sum ascending", True),
    #         ("s", "bar", "category ascending", True),
    #         # ("m", "barh", "category descending", True),
    #         # ("s", "bar", "total ascending", True),  # need to check, how this works
    #         # ("m", "barh", "total descending", True),  # need to check, how this works
    #         # ("s", "bar", "min ascending", True),  # need to check, how this works
    #         # ("m", "barh", "min descending", True),  # need to check, how this works
    #         # ("s", "bar", "max ascending", True),  # need to check, how this works
    #         # ("m", "barh", "max descending", True),  # need to check, how this works
    #         # ("s", "bar", "mean ascending", True),  # need to check, how this works
    #         # ("m", "barh", "mean descending", True),  # need to check, how this works
    #         # ("s", "bar", "median ascending", True),  # need to check, how this works
    #         # ("m", "barh", "median descending", True),  # need to check, how this works
    #         # (
    #         #     "s",
    #         #     "bar",
    #         #     "sum ascending",
    #         #     False,
    #         # ),  # need to check. failed as code it always shorts on sub_step desc
    #         # ("m", "bar", "sum descending", False),  # need to check. failed as code it always shorts on sub_step desc
    #     ],
    # )
    # def test_get_qsteps(self, qprof_data, unit, kind, category_order, show):
    #     # print(qprof_data)
    #     current_cursor().execute(
    #         "create schema if not exists qprof_test"
    #     )  # need to check as target_schema is not getting created by vpy
    #     # print(qprof_data["transactions"][0])
    #     transaction_id, statement_id = qprof_data["transactions"][0]
    #     qprof_steps = QueryProfiler(
    #         transactions=(transaction_id, statement_id)
    #     ).get_qsteps(unit=unit, kind=kind, categoryorder=category_order, show=show)
    #
    #     if show:
    #         series_data = qprof_steps.data_temp[0].data
    #         drilldown_data = qprof_steps.drilldown_data_temp
    #         qsteps_arrays = []
    #         for series_idx in range(len(series_data)):
    #             for drilldown_idx in range(len(drilldown_data)):
    #                 if (
    #                     series_data[series_idx]["name"]
    #                     == drilldown_data[drilldown_idx].name
    #                 ):
    #                     qsteps_arrays.extend(
    #                         [
    #                             (
    #                                 series_data[series_idx]["name"],
    #                                 data[0],
    #                                 round(data[1], 6),
    #                             )
    #                             for data in drilldown_data[drilldown_idx].data
    #                         ]
    #                     )
    #         actual_qplans_pdf = pd.DataFrame(
    #             qsteps_arrays, columns=["step", "sub_step", "elapsed_time"]
    #         )
    #     else:
    #         actual_qplans_pdf = qprof_steps.to_pandas().rename(
    #             columns={"substep": "sub_step", "elapsed": "elapsed_time"}
    #         )
    #     print(
    #         "<<<<<<<<<<<<<<<<<<<<<<<< actual result >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    #     )
    #     print(actual_qplans_pdf)
    #
    #     # qsteps_map["xAxis_title"] = (
    #     #     qprof_steps.options["xAxis"].title.text
    #     #     if qprof_steps.options["xAxis"].title
    #     #     else None
    #     # )
    #     # qsteps_map["yAxis_title"] = qprof_steps.options["yAxis"].title.text
    #
    #     if "ascending" in category_order:
    #         ascending_flag = True
    #     else:
    #         ascending_flag = False
    #     sort_key = "step" if "category" in category_order else "elapsed_time"
    #
    #     _expected_qplans_pdf = vDataFrame(
    #         f"select completion_time, time, execution_step from v_internal.dc_query_executions where transaction_id={transaction_id} and statement_id={statement_id}"
    #     ).to_pandas()
    #     _expected_qplans_pdf[["step", "_sub_step"]] = (
    #         _expected_qplans_pdf["execution_step"]
    #         .str.split(":", expand=True)
    #         .rename({0: "step", 1: "_sub_step"}, axis=1)
    #     )
    #     _expected_qplans_pdf["sub_step"] = _expected_qplans_pdf.apply(
    #         lambda x: x["step"] if pd.isnull(x["_sub_step"]) else x["_sub_step"], axis=1
    #     )
    #     _expected_qplans_pdf["elapsed_time"] = (
    #         _expected_qplans_pdf["completion_time"] - _expected_qplans_pdf["time"]
    #     ).dt.total_seconds()
    #     if unit == "m":
    #         _expected_qplans_pdf["elapsed_time"] = round(
    #             _expected_qplans_pdf["elapsed_time"] / 60, 6
    #         )
    #     elif unit == "h":
    #         _expected_qplans_pdf["elapsed_time"] = round(
    #             _expected_qplans_pdf["elapsed_time"] / 3600, 6
    #         )
    #     else:
    #         _expected_qplans_pdf["elapsed_time"] = round(
    #             _expected_qplans_pdf["elapsed_time"], 6
    #         )
    #
    #     if "category" in category_order:
    #         expected_qplans_pdf = (
    #             _expected_qplans_pdf[["step", "sub_step", "elapsed_time"]]
    #             .sort_values(by=["step", "sub_step"], ascending=ascending_flag)
    #             .reset_index(drop=True)
    #         )
    #     else:
    #         uneq = (
    #             _expected_qplans_pdf[["step", "sub_step", "elapsed_time"]]
    #             .loc[(_expected_qplans_pdf.sub_step != _expected_qplans_pdf.step)]
    #             .sort_values(
    #                 by=[
    #                     "step",
    #                     "sub_step" if "category" in category_order else "elapsed_time",
    #                 ],
    #                 ascending=[ascending_flag, ascending_flag],
    #             )
    #             .reset_index(drop=True)
    #         )
    #         eq = (
    #             _expected_qplans_pdf[["step", "sub_step", "elapsed_time"]]
    #             .loc[(_expected_qplans_pdf.sub_step == _expected_qplans_pdf.step)]
    #             .sort_values(by=sort_key, ascending=ascending_flag)
    #             .reset_index(drop=True)
    #         )
    #         expected_qplans_pdf = pd.DataFrame(data=None, columns=uneq.columns)
    #         for s in eq["step"].values.tolist():
    #             if ascending_flag:
    #                 expected_qplans_pdf = pd.concat(
    #                     [
    #                         expected_qplans_pdf,
    #                         (uneq[uneq["step"] == s]),
    #                         (eq[eq["step"] == s]),
    #                     ]
    #                 )
    #             else:
    #                 expected_qplans_pdf = pd.concat(
    #                     [
    #                         expected_qplans_pdf,
    #                         (eq[eq["step"] == s]),
    #                         (uneq[uneq["step"] == s]),
    #                     ]
    #                 )
    #         expected_qplans_pdf = expected_qplans_pdf.reset_index(drop=True)
    #
    #     print(
    #         "<<<<<<<<<<<<<<<<<<<<<<<< expected result >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    #     )
    #     print(expected_qplans_pdf)
    #
    #     print(
    #         "<<<<<<<<<<<<<<<<<<<<<<<< compare output >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    #     )
    #     res = expected_qplans_pdf.compare(actual_qplans_pdf)
    #     print(res)
    #
    #     assert False if len(res) > 0 else True
    #
    # @pytest.mark.parametrize(
    #     "return_report, print_plan",
    #     [
    #         # (False, True),
    #         (True, False),
    #         # (True, True),
    #         # (False, False)
    #     ],
    # )
    # def test_get_qplan(
    #     self, qprof_data, return_report, print_plan
    # ):  # need to check. get_qplan is not returning anything with transaction_id and statement_id.
    #     # Also, its printing report 2 times
    #     # current_cursor().execute(
    #     #     "create schema if not exists qprof_test"
    #     # )  # need to check as target_schema is not getting created by vpy
    #     # print(qprof_data["transactions"][0])
    #     # transaction_id, statement_id = qprof_data["transactions"][0]
    #     qprof_common = QueryProfiler(transactions=qprof_data["query2"])
    #     actual_qprof_qplan = (
    #         qprof_common.get_qplan(return_report=return_report, print_plan=print_plan)
    #         .sort(["stmtid", "path_id", "path_line_index"])
    #         .to_pandas()
    #     )
    #     print(actual_qprof_qplan)
    #
    #     transaction_id, statement_id = qprof_common.transactions[0]
    #     query = f"""SELECT statement_id AS stmtid, path_id, path_line_index, path_line FROM v_internal.dc_explain_plans WHERE transaction_id={transaction_id} AND statement_id={statement_id} ORDER BY 1, 2, 3;"""
    #     expected_qprof_qplan = (
    #         vDataFrame(query)
    #         .to_pandas()
    #         .sort_values(by=["stmtid", "path_id", "path_line_index"])
    #         .reset_index(drop=True)
    #     )
    #     print(expected_qprof_qplan)
    #
    #     print(
    #         "<<<<<<<<<<<<<<<<<<<<<<<< compare output >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    #     )
    #     res = expected_qprof_qplan.compare(actual_qprof_qplan)
    #     print(res)
    #
    #     assert False if len(res) > 0 else True
    #
    # @pytest.mark.parametrize(
    #     "path_id, path_id_info, show_ancestors, metric, pic_path, return_graphviz",
    #     [(None, None, True, "rows", None, False)],
    # )
    # def test_get_qplan_tree(
    #     self,
    #     path_id,
    #     path_id_info,
    #     show_ancestors,
    #     metric,
    #     pic_path,
    #     return_graphviz,
    # ):  # need to check. get_qplan_tree is not returning anything with transaction_id, statement_id
    #     current_cursor().execute(
    #         "create schema if not exists qprof_test"
    #     )  # need to check as target_schema is not getting created by vpy
    #     # print(qprof_data["transactions"][0])
    #     qprof_qplan_tree = QueryProfiler(transactions=self.query2).get_qplan_tree(
    #         path_id=path_id,
    #         path_id_info=path_id_info,
    #         show_ancestors=show_ancestors,
    #         metric=metric,
    #         pic_path=pic_path,
    #         return_graphviz=return_graphviz,
    #     )
    #     print(qprof_qplan_tree)
    #
    # @pytest.mark.parametrize(
    #     "unit, kind, category_order, show",
    #     [
    #         ("s", "bar", "total descending", True),
    #         # ("m", "bar", "sum descending", True),
    #     ],
    # )
    # def test_get_qplan_profile(
    #     self, qprof_data, unit, kind, category_order, show
    # ):  # need to check. getting error with transaction_id, statement_id
    #     current_cursor().execute(
    #         "create schema if not exists qprof_test"
    #     )  # need to check as target_schema is not getting created by vpy
    #     # print(qprof_data["transactions"][0])
    #     # transaction_id, statement_id = qprof_data["transactions"][1]
    #     qprof_common = QueryProfiler(self.query2)
    #     qprof_qplan_profile = qprof_common.get_qplan_profile(
    #         unit=unit, kind=kind, categoryorder=category_order, show=show
    #     )
    #     actual_qprof_qplan_profile = (
    #         pd.DataFrame(
    #             {
    #                 "path_line": qprof_qplan_profile.data[0].x,
    #                 "total_run_time": qprof_qplan_profile.data[0].y,
    #             }
    #         )
    #         .astype({"total_run_time": np.float64})
    #         .sort_values(by=["total_run_time"])
    #         .reset_index(drop=True)
    #     )
    #     print("<<<<<<<<<<<<<<<<<<< Actual execution report >>>>>>>>>>>>>>>>>>>>>>")
    #     print(actual_qprof_qplan_profile)
    #
    #     transaction_id, statement_id = qprof_common.transactions[0]
    #     query = f"select left(path_line, 80) AS path_line, sum(running_time) total_run_time FROM v_monitor.query_plan_profiles WHERE transaction_id={transaction_id} and statement_id={statement_id} AND running_time IS NOT NULL group by left(path_line, 80) order by total_run_time"
    #     expected_qprof_qplan_profile = vDataFrame(query).to_pandas()
    #     for i in range(len(expected_qprof_qplan_profile["total_run_time"])):
    #         expected_qprof_qplan_profile["total_run_time"][i] = (
    #             int(
    #                 f"{expected_qprof_qplan_profile['total_run_time'][i].seconds}{expected_qprof_qplan_profile['total_run_time'][i].microseconds}"
    #             )
    #             / 1000000
    #         )
    #     expected_qprof_qplan_profile = expected_qprof_qplan_profile.sort_values(
    #         by=["total_run_time"]
    #     ).reset_index(drop=True)
    #     print("<<<<<<<<<<<<<<<<< Expected execution report >>>>>>>>>>>>>>>>>>>>>>")
    #     print(expected_qprof_qplan_profile)
    #
    #     res = expected_qprof_qplan_profile.compare(
    #         actual_qprof_qplan_profile,
    #         result_names=(
    #             "left",
    #             "right",
    #         ),
    #     )
    #     print("<<<<<<<<<<<<<<< compare result actual vs expected >>>>>>>>>>>>>>>>>")
    #     print(res)
    #
    #     assert False if len(res) > 0 else True
    #
    # def test_get_query_events(self, qprof_data):
    #     current_cursor().execute("create schema if not exists qprof_test")
    #
    #     transaction_id, statement_id = qprof_data["transactions"][0]
    #     actual_qprof_events = (
    #         QueryProfiler(transactions=(transaction_id, statement_id))
    #         .get_query_events()
    #         .to_pandas()
    #     )
    #
    #     query = f"SELECT event_timestamp, node_name, event_category, event_type, event_description, operator_name, path_id, event_details, suggested_action FROM v_monitor.query_events WHERE transaction_id={transaction_id} AND statement_id={statement_id} ORDER BY 1"
    #     expected_qprof_events = vDataFrame(query).to_pandas()
    #
    #     res = expected_qprof_events.compare(actual_qprof_events)
    #     print(res)
    #
    #     assert False if len(res) > 0 else True
    #
    # @pytest.mark.parametrize(
    #     "kind, reverse, category_order, show",
    #     [
    #         ("bar", False, "sum max", True),
    #     ],
    # )
    # def test_cup_time(
    #     self, qprof_data, kind, reverse, category_order, show
    # ):  # need to check. getting with transaction_id and statement_id
    #     # current_cursor().execute(
    #     #     "create schema if not exists qprof_test"
    #     # )  # need to check as target_schema is not getting created by vpy
    #     # print(qprof_data["transactions"][0])
    #     # transaction_id, statement_id = qprof_data["transactions"][1]
    #     qprof_common = QueryProfiler(self.query2)
    #     qprof_cpu_time = qprof_common.get_cpu_time(
    #         kind=kind, reverse=reverse, categoryorder=category_order, show=show
    #     )
    #     print("<<<<<<<<<<<<<<<<<<< Actual execution report >>>>>>>>>>>>>>>>>>>>>>")
    #     actual_qprof_cpu_time = (
    #         pd.DataFrame(
    #             {
    #                 "node_name": qprof_cpu_time.data[0].name,
    #                 "path_id": qprof_cpu_time.data[0].x,
    #                 "counter_sum": qprof_cpu_time.data[0].y,
    #             }
    #         )
    #         .astype({"path_id": np.int64})
    #         .sort_values(by=["node_name", "path_id", "counter_sum"])
    #         .reset_index(drop=True)
    #     )
    #     print(actual_qprof_cpu_time)
    #
    #     transaction_id, statement_id = qprof_common.transactions[0]
    #     query = f"SELECT node_name, path_id, sum(counter_value) counter_sum FROM v_monitor.execution_engine_profiles WHERE TRIM(counter_name) = 'execution time (us)' and transaction_id={transaction_id} AND statement_id={statement_id} group by node_name, path_id"
    #     expected_qprof_cpu_time = (
    #         vDataFrame(query)
    #         .to_pandas()
    #         .sort_values(by=["node_name", "path_id", "counter_sum"])
    #         .reset_index(drop=True)
    #     )
    #     print("<<<<<<<<<<<<<<<<< Expected execution report >>>>>>>>>>>>>>>>>>>>>>")
    #     print(expected_qprof_cpu_time)
    #
    #     res = expected_qprof_cpu_time.compare(
    #         actual_qprof_cpu_time,
    #         result_names=(
    #             "left",
    #             "right",
    #         ),
    #     )
    #     print("<<<<<<<<<<<<<<< compare result actual vs expected >>>>>>>>>>>>>>>>>")
    #     print(res)
    #
    #     assert False if len(res) > 0 else True
    #
    # def test_get_qexecution_report(self):
    #     # need to check. no results, with transaction_id, statement_id
    #     # current_cursor().execute("create schema if not exists qprof_test")
    #     # print(qprof_data)
    #     qprof_common = QueryProfiler(transactions=self.query2)
    #
    #     actual_qprof_qexecution_report = (
    #         qprof_common.get_qexecution_report()
    #         .to_pandas()[["node_name", "operator_name", "path_id"]]
    #         .sort_values(by=["node_name", "operator_name", "path_id"])
    #         .reset_index(drop=True)
    #     )
    #
    #     print("<<<<<<<<<<<<<<<<<<< Actual execution report >>>>>>>>>>>>>>>>>>>>>>")
    #     print(actual_qprof_qexecution_report)
    #     transaction_id, statement_id = qprof_common.transactions[0]
    #
    #     query = f"SELECT node_name, operator_name, path_id FROM v_monitor.execution_engine_profiles WHERE transaction_id={transaction_id} AND statement_id={statement_id} group by node_name, operator_name, path_id"
    #     expected_qprof_qexecution_report = (
    #         vDataFrame(query)
    #         .to_pandas()
    #         .sort_values(by=["node_name", "operator_name", "path_id"])
    #         .reset_index(drop=True)
    #     )
    #     print("<<<<<<<<<<<<<<<<< Expected execution report >>>>>>>>>>>>>>>>>>>>>>")
    #     print(expected_qprof_qexecution_report)
    #
    #     res = expected_qprof_qexecution_report.compare(
    #         actual_qprof_qexecution_report,
    #         result_names=(
    #             "left",
    #             "right",
    #         ),
    #     )
    #     print("<<<<<<<<<<<<<<< compare result actual vs expected >>>>>>>>>>>>>>>>>")
    #     print(res)
    #
    #     assert False if len(res) > 0 else True
    #
    # @pytest.mark.parametrize(
    #     "node_name, metric, path_id, kind, multi, categoryorder, rows, cols, show",
    #     [
    #         (None, None, None, None, None, None, None, None, None),
    #     ],
    # )
    # def test_get_qexecution(
    #     self,
    #     qprof_data,
    #     node_name,
    #     metric,
    #     path_id,
    #     kind,
    #     multi,
    #     categoryorder,
    #     rows,
    #     cols,
    #     show,
    # ):  # need to check. getting error
    #     current_cursor().execute(
    #         "create schema if not exists qprof_test"
    #     )  # need to check as target_schema is not getting created by vpy
    #     # print(qprof_data["transactions"][0])
    #     transaction_id, statement_id = qprof_data["transactions"][0]
    #     qprof_steps = QueryProfiler(transactions=qprof_data["query2"]).get_qexecutione(
    #         node_name=node_name,
    #         metric=metric,
    #         path_id=path_id,
    #         kind=kind,
    #         multi=multi,
    #         categoryorder=categoryorder,
    #         rows=rows,
    #         cols=cols,
    #         show=show,
    #     )
    #
    # def test_get_rp_status(self, qprof_data):
    #     current_cursor().execute("create schema if not exists qprof_test")
    #
    #     transaction_id, statement_id = qprof_data["transactions"][1]
    #     actual_qprof_rp = (
    #         QueryProfiler(transactions=(transaction_id, statement_id))
    #         .get_rp_status()
    #         .to_pandas()
    #         .sort_values(by="pool_oid")
    #         .reset_index(drop=True)
    #     )
    #     print(actual_qprof_rp)
    #
    #     query = f"SELECT * FROM v_monitor.resource_pool_status ORDER BY pool_oid"
    #     expected_qprof_rp = vDataFrame(query).to_pandas()
    #     print(expected_qprof_rp)
    #
    #     res = expected_qprof_rp.compare(actual_qprof_rp)
    #     print(res)
    #
    #     assert len(actual_qprof_rp) == len(expected_qprof_rp)
    #
    # def test_get_cluster_config(self, qprof_data):
    #     current_cursor().execute("create schema if not exists qprof_test")
    #
    #     transaction_id, statement_id = qprof_data["transactions"][1]
    #     actual_qprof_cluster_config = (
    #         QueryProfiler(transactions=(transaction_id, statement_id))
    #         .get_cluster_config()
    #         .to_pandas()
    #         .sort_values(by="host_name")
    #         .reset_index(drop=True)
    #     )
    #     print(actual_qprof_cluster_config)
    #
    #     query = f"SELECT * FROM v_monitor.host_resources ORDER BY host_name"
    #     expected_qprof_cluster_config = vDataFrame(query).to_pandas()
    #     print(expected_qprof_cluster_config)
    #
    #     res = expected_qprof_cluster_config.compare(
    #         actual_qprof_cluster_config,
    #         result_names=(
    #             "left",
    #             "right",
    #         ),
    #     )
    #     print(res)
    #
    #     assert False if len(res) > 0 else True
