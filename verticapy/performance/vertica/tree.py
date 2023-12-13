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
import html
import re
import math
from typing import Literal, Optional, Union
import numpy as np

import verticapy._config.config as conf
from verticapy._typing import NoneType

if conf.get_import_success("graphviz"):
    import graphviz
    from graphviz import Source


class PerformanceTree:
    """
    Performance Tree object.
    It is used to simplify the
    export of the Query Plan
    to graphviz.

    Parameters
    ----------
    rows: str
        ``str`` representing
        the Query Plan.
    root: int, optional
        A path ID used to filter
        the tree elements by
        starting from it.
    metric: str, optional
        The metric used to color
        the tree nodes. One of
        the following:

        - cost
        - rows
        - None (no specific color)
    show_ancestors: bool, optional
        If set to ``True`` the
        ancestors are also
        displayed.
    show_nodes_info: list, optional
        List of nodes for which
        a tooltip node will be
        created.

    Attributes
    ----------
    The attributes are the same
    as the parameters.

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
        root: int = 1,
        metric: Literal[None, "cost", "rows"] = "rows",
        show_ancestors: bool = True,
        show_nodes_info: Optional[list] = None,
        style: dict = {},
    ) -> None:
        qplan = rows.split("\n")
        n = len(qplan)
        self.rows, tmp_rows = [], []
        for i in range(n):
            if "PATH ID: " in qplan[i] and i > 0:
                self.rows += ["\n".join(tmp_rows)]
                tmp_rows = []
            tmp_rows += [qplan[i]]
        self.rows += ["\n".join(tmp_rows)]
        if isinstance(root, int) and root > 0:
            self.root = root - 1
        else:
            raise ValueError(
                "Wrong value for parameter 'root': "
                "It has to be a strictly positive int.\n"
                f"Found {root}."
            )
        if metric in [None, "cost", "rows"]:
            self.metric = metric
        else:
            raise ValueError(
                "Wrong value for parameter 'metric': "
                "It has to be one of the following: None, 'cost', 'rows'.\n"
                f"Found {metric}."
            )
        self.show_ancestors = show_ancestors
        d = copy.deepcopy(style)
        for color in ("color_low", "color_high"):
            if color not in d:
                if color == "color_low":
                    d[color] = (0, 255, 0)
                else:
                    d[color] = (255, 0, 0)
            elif isinstance(d[color], str):
                d[color] = self._color_string_to_tuple(d[color])
        if "fillcolor" not in d:
            d["fillcolor"] = "#ADD8E6"
        if "shape" not in d:
            d["shape"] = "circle"
        if "fontcolor" not in d:
            d["fontcolor"] = "#000000"
        if "width" not in d:
            d["width"] = 0.6
        if "height" not in d:
            d["height"] = 0.6
        if "edge_color" not in d:
            d["edge_color"] = "#000000"
        if "edge_style" not in d:
            d["edge_style"] = "solid"
        if "info_color" not in d:
            d["info_color"] = "#DFDFDF"
        if "info_fontcolor" not in d:
            d["info_fontcolor"] = "#000000"
        self.style = d
        if isinstance(show_nodes_info, list):
            self.show_nodes_info = [i - 1 for i in show_nodes_info]
        else:
            self.show_nodes_info = []

    # Utils
    @staticmethod
    def _color_string_to_tuple(color_string: str) -> tuple[int, int, int]:
        """
        Converts a color ``str``
        to a ``tuple``.

        Parameters
        ----------
        color_string: str
            color.

        Returns
        -------
        tuple
            color ``r,g,b``.

        Examples
        --------
        See :py:meth:`verticapy.performance.vertica.tree`
        for more information.
        """

        # Check if the string starts with '#', remove it if present
        color_string = color_string.lstrip("#")

        # Convert the hexadecimal string to RGB values
        r = int(color_string[0:2], 16)
        g = int(color_string[2:4], 16)
        b = int(color_string[4:6], 16)

        # Return the RGB tuple
        return r, g, b

    def _generate_gradient_color(self, intensity: float) -> str:
        """
        Generates a gradient
        color based on an
        ``intensity``.

        Parameters
        ----------
        intensity: float
            Intensity.

        Returns
        -------
        str
            color.

        Examples
        --------
        See :py:meth:`verticapy.performance.vertica.tree`
        for more information.
        """
        # Ensure intensity is between 0 and 1
        intensity = max(0, min(1, intensity))

        # Computing the color
        colors = np.array(self.style["color_high"]) * intensity + np.array(
            self.style["color_low"]
        ) * (1 - intensity)

        # Calculate RGB values based on intensity
        red = int(colors[0])
        green = int(colors[1])
        blue = int(colors[2])

        # Format the color string
        color = f"#{red:02X}{green:02X}{blue:02X}"

        return color

    @staticmethod
    def _map_unit(unit: str):
        """
        Maps the input unit.

        Parameters
        ----------
        unit: str
            Unit.

        Returns
        -------
        str
            range: ``K``, ``M``,
            ``B`` or ``None``.

        Examples
        --------
        See :py:meth:`verticapy.performance.vertica.tree`
        for more information.
        """
        if unit == "K":
            return 1000
        elif unit == "M":
            return 1000000
        elif unit == "B":
            return 1000000000
        elif unit.isalpha():
            return 1000000000000
        else:
            return None

    @staticmethod
    def _format_row(row: str) -> str:
        """
        Format the input row.

        Parameters
        ----------
        row: str
            Tree row.

        Returns
        -------
        str
            formatted row.

        Examples
        --------
        See :py:meth:`verticapy.performance.vertica.tree`
        for more information.
        """
        rows = row.split("\n")
        n = len(rows)
        for i in range(n):
            x = rows[i]
            while len(x) > 0 and x[0] in ("+", "-", " ", "|", ">"):
                x = x[1:]
            rows[i] = x
        return ("\n".join(rows)).replace("\n", "\n\n")

    @staticmethod
    def _format_number(nb: int) -> str:
        """
        Format the input number.

        Parameters
        ----------
        nb: int
            Number.

        Returns
        -------
        str
            formatted number.

        Examples
        --------
        See :py:meth:`verticapy.performance.vertica.tree`
        for more information.
        """
        if nb < 1e3:
            return str(int(nb))
        if nb < 1e6:
            return f"{round(nb / 1e3)}K"
        elif nb < 1e9:
            return f"{round(nb / 1e6)}M"
        elif nb < 1e12:
            return f"{round(nb / 1e9)}B"
        else:
            return f"{round(nb / 1e12)}T"

    # Special Methods

    @staticmethod
    def _get_label(row: str, return_path_id: bool = True) -> Union[str, int]:
        """
        Gets the label from
        Query Plan chart.

        Parameters
        ----------
        row: str
            Tree row.
        return_path_id: bool, optional
            If set to ``True`` returns
            the path ID instead.

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
        if return_path_id:
            res = int(res.split("PATH ID: ")[1].split(")")[0])
        return res

    @staticmethod
    def _get_level(row: str) -> int:
        """
        Gets the level of the
        specific row.

        Parameters
        ----------
        row: str
            Tree row.

        Returns
        -------
        int
            level.

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
        return res.count("|")

    def _get_metric(self, row: str) -> int:
        """
        Gets the metric of the
        specific row.

        Parameters
        ----------
        row: str
            Tree row.

        Returns
        -------
        int
            metric

        Examples
        --------
        See :py:meth:`verticapy.performance.vertica.tree`
        for more information.
        """
        if self.metric == "rows" and "Rows: " in row:
            res = row.split("Rows: ")[1].split(" ")[0]
        elif self.metric == "cost" and "Cost: " in row:
            res = row.split("Cost: ")[1].split(",")[0]
        else:
            return None
        if res[-1] in ("]",):
            res = res[:-1]
        unit = self._map_unit(res[-1])
        res = int(re.sub(r"[^0-9]", "", res))
        if isinstance(unit, NoneType):
            return res
        return res * unit

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
        res = [0]
        for idx, row in enumerate(self.rows):
            row_level = self._get_level(row)
            if row_level + 1 == level:
                res += [idx]
        return res

    @staticmethod
    def _get_last_initiator(level_initiators: list[int], tree_id: int) -> int:
        """
        Gets the last level initiator
        of a specific level.

        Parameters
        ----------
        level_initiators: list
            ``list`` of initiators.
        tree_id: int
            Tree ID.

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
            if int(tree_id) < l:
                return level_initiators[i - 1]
        return level_initiators[-1]

    @staticmethod
    def _find_descendants(node: int, relationships: list[tuple[int, int]]) -> list:
        """
        Method used to find all descendants
        (children, grandchildren, and so on)
        of a specific node in a tree-like
        structure represented by parent-child
        relationships.

        Parameters
        ----------
        node: int
            Node ID.
        relationships: list
            ``list`` of tuple
            ``(parent, child)``

        Returns
        -------
        list
            list of descendants.

        Examples
        --------
        See :py:meth:`verticapy.performance.vertica.tree`
        for more information.
        """
        if node == 0:
            return [x[0] for x in relationships] + [x[1] for x in relationships]

        descendants = []

        # Recursive function to find descendants
        def find_recursive(current_node):
            nonlocal descendants
            children = [
                child for parent, child in relationships if parent == current_node
            ]
            descendants.extend(children)
            for child in children:
                find_recursive(child)

        # Start the recursive search from the specified node
        find_recursive(node)

        return descendants

    @staticmethod
    def _find_ancestors(node: int, relationships: list[tuple[int, int]]) -> list:
        """
        Method used to find all ancestors
        (parents, grandparents, and so on)
        of a specific node in a tree-like
        structure represented by parent-child
        relationships.

        Parameters
        ----------
        node: int
            Node ID.
        relationships: list
            ``list`` of tuple
            ``(parent, child)``

        Returns
        -------
        list
            list of ancestors.

        Examples
        --------
        See :py:meth:`verticapy.performance.vertica.tree`
        for more information.
        """
        if node == 0:
            return []

        ancestors = []

        # Recursive function to find ancestors
        def find_recursive(current_node):
            if current_node == 0:
                return
            nonlocal ancestors
            parents = [
                parent for parent, child in relationships if child == current_node
            ]
            ancestors.extend(parents)
            for parent in parents:
                find_recursive(parent)

        # Start the recursive search from the specified node
        find_recursive(node)

        return ancestors

    def _gen_relationships(self) -> list[tuple[int, int]]:
        """
        Generates the relationships
        ``list``.

        Returns
        -------
        list
            list of the different
            relationships.

        Examples
        --------
        See :py:meth:`verticapy.performance.vertica.tree`
        for more information.
        """
        n = len(self.rows)
        relationships = []
        for i in range(n):
            level = self._get_level(self.rows[i])
            level_initiators = self._get_all_level_initiator(level)
            id_initiator = self._get_last_initiator(level_initiators, i)
            relationships += [(id_initiator, i)]
        return relationships

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
        n, res = len(self.rows), ""
        if not (isinstance(self.metric, NoneType)):
            all_metrics = [
                math.log(1 + self._get_metric(self.rows[i])) for i in range(n)
            ]
            m_min, m_max = min(all_metrics), max(all_metrics)
        relationships = self._gen_relationships()
        links = self._find_descendants(self.root, relationships) + [self.root]
        if self.show_ancestors:
            links += self._find_ancestors(self.root, relationships)
        for i in range(n):
            if not (isinstance(self.metric, NoneType)):
                alpha = (all_metrics[i] - m_min) / (m_max - m_min)
                color = self._generate_gradient_color(alpha)
            else:
                color = self.style["fillcolor"]
            label = self._get_label(self.rows[i])
            if i in links:
                row = self._format_row(self.rows[i].replace('"', "'"))
                res += f'\t{i} [label="{label}", style="filled", fillcolor="{color}", tooltip="{row}", fixedsize=true];\n'
                if i in self.show_nodes_info:
                    info_color = self.style["info_color"]
                    info_fontcolor = self.style["info_fontcolor"]
                    html_content = html.escape(row).replace(
                        "\n", '</td></tr><tr><td align="left">'
                    )
                    label = (
                        '<<table border="0" cellborder="0" cellspacing="0" width="40">'
                    )
                    label += f'\t<tr><td align="left">{html_content}</td></tr>\n'
                    label += "</table>>"
                    res += f'\t{10000 - i} [shape=plaintext, fontcolor="{info_fontcolor}", style="filled", fillcolor="{info_color}", width=0.6, height=0.6, fontsize=8, label={label}];\n'
            if i == self.root and i > 0 and self.show_ancestors:
                row = self._format_row(self.rows[self.root].replace('"', "'"))
                res += f'\t{n} [label="{self.root + 1}", style="filled", fillcolor="{color}", tooltip="{row}"];\n'
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
        res, n = "", len(self.rows)
        relationships = self._gen_relationships()
        links = self._find_descendants(self.root, relationships)
        info_color = self.style["info_color"]
        if self.show_ancestors:
            links += self._find_ancestors(self.root, relationships)
        for i in range(n):
            parent, child = relationships[i]
            if parent != child and child in links:
                res += f"\t{parent} -> {child} [dir=back];\n"
            if child == self.root and i > 0 and self.show_ancestors:
                res += f"\t{parent} -> {n} [dir=back];\n"
            if i in self.show_nodes_info:
                res += f'\t{10000 - i} -> {i} [dir=none, color="{info_color}"];\n'
        return res

    def _gen_legend(self) -> str:
        """
        Generates the Graphviz
        legend.

        Returns
        -------
        str
            Graphviz legend.

        Examples
        --------
        See :py:meth:`verticapy.performance.vertica.tree`
        for more information.
        """
        n = len(self.rows)
        all_metrics = [math.log(1 + self._get_metric(self.rows[i])) for i in range(n)]
        m_min, m_max = min(all_metrics), max(all_metrics)
        cats = [0.0, 0.25, 0.5, 0.75, 1.0]
        cats = [
            self._format_number(int(math.exp(x * (m_max - m_min) + m_min) - 1))
            for x in cats
        ]
        alpha0 = self._generate_gradient_color(0.0)
        alpha025 = self._generate_gradient_color(0.25)
        alpha050 = self._generate_gradient_color(0.5)
        alpha075 = self._generate_gradient_color(0.75)
        alpha1 = self._generate_gradient_color(1.0)
        res = '\tlegend [shape=plaintext, fillcolor=white, label=<<table border="0" cellborder="1" cellspacing="0">'
        res += f'<tr><td bgcolor="#DFDFDF">Legend</td></tr>'
        res += f'<tr><td bgcolor="{alpha0}">{cats[0]}</td></tr>'
        res += f'<tr><td bgcolor="{alpha025}">{cats[1]}</td></tr>'
        res += f'<tr><td bgcolor="{alpha050}">{cats[2]}</td></tr>'
        res += f'<tr><td bgcolor="{alpha075}">{cats[3]}</td></tr>'
        res += f'<tr><td bgcolor="{alpha1}">{cats[4]}</td></tr>'
        res += "</table>>]\n\n"
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
        fillcolor = self.style["fillcolor"]
        shape = self.style["shape"]
        fontcolor = self.style["fontcolor"]
        width = self.style["width"]
        height = self.style["height"]
        edge_color = self.style["edge_color"]
        edge_style = self.style["edge_style"]
        res = "digraph Tree {\n"
        res += f'\tnode [shape={shape}, style=filled, fillcolor="{fillcolor}", fontcolor="{fontcolor}", width={width}, height={height}];\n'
        res += f'\tedge [color="{edge_color}", style={edge_style}];\n'
        if not (isinstance(self.metric, NoneType)):
            res += self._gen_legend()
        else:
            res += "\n"
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
