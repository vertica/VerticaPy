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
import graphviz

from verticapy.performance.vertica import QueryProfiler
from verticapy.performance.vertica.tree import PerformanceTree

from verticapy.tests_new.performance.vertica import QPROF_SQL2


class TestTree:
    """
    test class for tree
    """

    def test_to_graphviz(self):
        """
        test function for plot_tree
        """
        qprof = QueryProfiler(QPROF_SQL2)
        tree = PerformanceTree(qprof.get_qplan())
        res = tree.to_graphviz()

        assert "digraph Tree {\n\tgraph" in res and "0 -> 1" in res

    def test_plot_tree(self):
        """
        test function for plot_tree
        """
        qprof = QueryProfiler(QPROF_SQL2)
        tree = PerformanceTree(qprof.get_qplan())
        res = tree.plot_tree()

        assert (
            isinstance(res, graphviz.sources.Source)
            and tree.to_graphviz() == res.source.strip()
        )
