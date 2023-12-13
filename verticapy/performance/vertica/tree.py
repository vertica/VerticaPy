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
from typing import Optional

import verticapy._config.config as conf

if conf.get_import_success("graphviz"):
    import graphviz
    from graphviz import Source


class PerformanceTree:
    """
    Performance Tree object.
    It is used to simplify the
    export of the Query Plan
    to graphviz.

    Attributes
    ----------
    rows: str
        ``str`` representing
        the Query Plan.

    Examples
    --------
    The following code demonstrates
    the usage of the class.

    .. code-block:: python

        # Import the function.
        from verticapy.performance.vertica.tree import PerformanceTree

        # Creating the object.
        tree = PerformanceTree(qplan) # qplan being a query plan

        # Exporting to Graphviz.
        tree.to_graphviz()

        # Plot the Graphviz tree.
        tree.plot_tree()

    .. note::

        This class serve as utilities to
        construct others, simplifying the
        overall code.
    """

    # Init Functions

    def __init__(
        self,
        rows: str,
    ) -> None:
        self.rows = rows.split("\n")

    # Special Methods

    @staticmethod
    def _get_label(row: str) -> str:
        """
        Gets the label from
        Query Plan chart.

        Parameters
        ----------
        row: str
            Tree row.

        Returns
        -------
        str
            label.

        Examples
        --------
        See :py:meth:`verticapy.performance.vertica.tree`
        for more information.
        """
        res = row
        while len(res) > 0 and res[0] in ("+", "-", " ", "|", ">"):
            res = res[1:]
        i, n = 0, len(res)
        while i < n and (res[i].isalpha() or res[i] in (" ",)):
            i += 1
        return res[:i]

    @staticmethod
    def _get_level(row: str) -> tuple[int, bool]:
        """
        Gets the level of the
        specific row.

        Parameters
        ----------
        row: str
            Tree row.

        Returns
        -------
        tuple
            level, is_initiator

        Examples
        --------
        See :py:meth:`verticapy.performance.vertica.tree`
        for more information.
        """
        res = ""
        i, n = 0, len(row)
        while i < n and row[i] in ("+", "-", " ", "|", ">"):
            res += row[i]
            i += 1
        return (res.count("|") + res.count("+-") - 1, "+-" in res)

    def _get_all_level_initiator(self, level: int) -> list[int]:
        """
        Gets the all the possible
        level initiator of the
        specific level.

        Parameters
        ----------
        level: int
            Tree level.

        Returns
        -------
        list
            all possible initiators.

        Examples
        --------
        See :py:meth:`verticapy.performance.vertica.tree`
        for more information.
        """
        res = []
        for row in self.rows:
            level_initiator = self._get_level(row)
            if level_initiator[0] == level and level_initiator[1]:
                res += [level]
        return res

    @staticmethod
    def _get_last_initiator(level_initiators: list[int], level: int) -> int:
        """
        Gets the last level initiator
        of a specific level.

        Parameters
        ----------
        level_initiators: list
            ``list`` of initiators.
        level: int
            Tree level.

        Returns
        -------
        int
            last initiator.

        Examples
        --------
        See :py:meth:`verticapy.performance.vertica.tree`
        for more information.
        """
        for i, l in enumerate(level_initiators):
            if level > l:
                return level_initiators[i - 1]
        return level_initiators[-1]

    def _gen_labels(self) -> str:
        """
        Generates the Graphviz
        labels.

        Returns
        -------
        str
            Graphviz labels.

        Examples
        --------
        See :py:meth:`verticapy.performance.vertica.tree`
        for more information.
        """
        res = ""
        n = len(self.rows)
        for i in range(n):
            label = self._get_label(self.rows[i])
            res += f'\t{i} [label="{label}"];\n'
        return res

    def _gen_links(self) -> str:
        """
        Generates the Graphviz
        links.

        Returns
        -------
        str
            Graphviz links.

        Examples
        --------
        See :py:meth:`verticapy.performance.vertica.tree`
        for more information.
        """
        res = ""
        n = len(self.rows)
        for i in range(n):
            id_initiator = self._get_level(self.rows[i])[0]
            level_initiators = self._get_all_level_initiator(id_initiator)
            id_initiator = self._get_last_initiator(level_initiators, id_initiator)
            if i != id_initiator:
                res += f"\t{id_initiator} -> {i};\n"
        return res

    def to_graphviz(self) -> str:
        """
        Exports the object
        the Graphviz.

        Returns
        -------
        str
            Graphviz ``str``.

        Examples
        --------
        See :py:meth:`verticapy.performance.vertica.tree`
        for more information.
        """
        res = "digraph Tree {\n"
        res += "\tnode [shape=circle, style=filled, fillcolor=lightblue, fontcolor=black, width=0.6, height=0.6];\n"
        res += "\tedge [color=black, style=solid];\n\n"
        res += self._gen_labels() + "\n"
        res += self._gen_links() + "\n"
        res += "}"
        return res

    def plot_tree(
        self,
        pic_path: Optional[str] = None,
    ) -> "Source":
        """
        Draws the tree.
        Requires the graphviz
        module.

        Parameters
        ----------
        pic_path: str, optional
            Absolute path to save
            the image of the tree.

        Returns
        -------
        graphviz.Source
            graphviz object.

        Examples
        --------
        See :py:meth:`verticapy.performance.vertica.tree`
        for more information.
        """
        if not conf.get_import_success("graphviz"):
            raise ImportError(
                "The graphviz module doesn't seem to be "
                "installed in your environment.\nTo be "
                "able to use this method, you'll have to "
                "install it.\n[Tips] Run: 'pip3 install "
                "graphviz' in your terminal to install "
                "the module."
            )
        res = graphviz.Source(self.to_graphviz())
        if pic_path:
            res.render(filename=pic_path)
        return res
