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
import pandas as pd
import pytest
from verticapy.performance.vertica.tree import PerformanceTree
from verticapy.performance.vertica import QueryProfiler
from verticapy.connection import current_cursor
from verticapy.core.vdataframe import vDataFrame


class TestTree:
    """
    test class for tree
    """

    def test_to_graphviz(self):
        """
        test function for plot_tree
        """
        qprof = QueryProfiler(
            "select transaction_id, statement_id, request, request_duration from query_requests where start_timestamp > now() - interval'1 hour' order by request_duration desc limit 10;"
        )
        tree = PerformanceTree(qprof.get_qplan())
        res = tree.to_graphviz()

        assert "digraph Tree {\n\tgraph" in res and "0 -> 1" in res

    def test_plot_tree(self):
        """
        test function for plot_tree
        """
        qprof = QueryProfiler(
            "select transaction_id, statement_id, request, request_duration from query_requests where start_timestamp > now() - interval'1 hour' order by request_duration desc limit 10;"
        )
        tree = PerformanceTree(qprof.get_qplan())
        res = tree.plot_tree()

        assert tree.to_graphviz() == res.source.strip()
