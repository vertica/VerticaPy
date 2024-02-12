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
QPROF_SQL1 = """SELECT /*+LABEL('QueryProfiler_sql1_requests_UT')*/ transaction_id, statement_id, request, request_duration FROM v_monitor.query_requests WHERE start_timestamp > NOW() - INTERVAL'1 hour' ORDER BY request_duration DESC LIMIT 10"""
QPROF_SQL2 = """SELECT /*+LABEL('QueryProfiler_sql2_requests_UT')*/ transaction_id, statement_id, request, request_duration FROM v_monitor.query_requests WHERE start_timestamp > NOW() - INTERVAL'1 hour' ORDER BY request_duration DESC LIMIT 10"""
