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

from typing import Any, Literal

import verticapy._config.config as conf

if conf.get_import_success("IPython"):
    import ipywidgets as widgets
    from IPython.display import display


class Visualizer:
    """
    A class that displays any widget it gets
    in two boxes, one for the settings and the
    other for the item to display.
    The item to display can be any thing from a
    chart to a table.

    Parameters
    ----------
    settings_wids: list, optional
        the widgets to display in the settings box.
    graph_wids:  list, optional
        the widgets to display in the graph box.
    settings_layout_override: dict[str, str], optional
        values that will override the default layout of the
        settings box. Use it to customize the settings box.
    graph_layout_override: list, optional
        values that will override the default layout of the
        graph box. Use it to customize the graph box.
    orientation: str, optional
        how both boxes are displayed. One of the following:

        - h: the boxes are displayed horizontally
        - v: the boxes are dsiplayed vertically
    """

    def __init__(
        self,
        settings_wids: list = [],
        graph_wids: list = [],
        settings_layout_override: dict[str, str] = {},
        graph_layout_override: dict[str, str] = {},
        orientation: Literal[
            "h",
            "v",
        ] = "h",
    ) -> None:
        settings_layout = self.get_settings_box_h_default_layout()
        graph_layout = self.get_graph_box_h_default_layout()
        if orientation == "v":
            settings_layout = self.get_settings_box_v_default_layout()
            graph_layout = self.get_graph_box_v_default_layout()
        for k, v in settings_layout_override.items():
            settings_layout[k] = v
        for k, v in graph_layout_override.items():
            graph_layout[k] = v
        self.settings_wids = settings_wids
        self.graph_wids = graph_wids
        self.settings_box = widgets.Output(layout=settings_layout)
        self.graph_box = widgets.Output(layout=graph_layout)
        self.orientation = orientation

    def display(self):
        """
        Display the settings and graph boxes
        """
        settings = widgets.VBox(self.settings_wids)
        graph = widgets.VBox(self.graph_wids)
        if self.orientation == "v":
            settings = widgets.HBox(self.settings_wids)
        with self.settings_box:
            display(settings)
        with self.graph_box:
            self.graph_box.clear_output(wait=True)
            display(graph)
        if self.orientation == "v":
            display(widgets.VBox([self.settings_box, self.graph_box]))
            return
        display(widgets.HBox([self.settings_box, self.graph_box]))

    @staticmethod
    def _accordion(children: list, titles: list) -> widgets.Accordion:
        """
        Build accordions from a list of widgets.

        Args:
            children (list): list of widgets that
                will be displayed by the accordions
            titles (list): title of each accordion

        Returns:
            widgets.Accordion: Accordion widget
        """
        titles = tuple(titles)
        return widgets.Accordion(children=children, titles=titles)

    @staticmethod
    def get_settings_box_h_default_layout() -> dict[str, str]:
        """
        ...
        """
        return {
            "width": "25%",
            "border": "1px solid gray",
            "height": "700px",
        }

    @staticmethod
    def get_graph_box_h_default_layout() -> dict[str, str]:
        """
        ...
        """
        return {
            "width": "75%",
            "border": "1px solid gray",
            "height": "700px",
            "overflow": "auto",
            "justify_content": "center",
            "align_items": "center",
        }

    @staticmethod
    def get_graph_box_v_default_layout() -> dict[str, str]:
        """
        ...
        """
        return {
            "border": "1px solid gray",
            "height": "700px",
            "overflow": "auto",
        }

    @staticmethod
    def get_settings_box_v_default_layout() -> dict[str, str]:
        """
        ...
        """
        return {
            "border": "1px solid gray",
        }


class Item:
    """
    A wrapper class that gets simple
    widget like IntText, Dropdown, ...,
    and prepend the widget name in a nice
    button.
    """

    def __init__(
        self,
        name: str,
        widget: Any,
        description: str = "",
    ):
        self.name = name
        self.child = widget
        self.description = description

    def get_child(self):
        """
        ...
        """
        return self.child

    def get_item(self):
        """
        ...
        """
        label = widgets.Button(disabled=True, description=self.name)
        if self.description:
            label.tooltip = self.description
        label.style.button_color = "lightgreen"
        label.layout = {"width": "40%"}
        self.child.layout = {"width": "60%"}
        return widgets.HBox([label, self.child])

    def get_child_attr(self, name: str):
        """
        ...
        """
        if hasattr(self.child, name):
            return self.child.__getattribute__(name)
        return None

    def set_child_attr(self, name: str, value: Any):
        """
        ...
        """
        if hasattr(self.child, name):
            self.child.__setattr__(name, value)


def makeItems(items: list[tuple]) -> dict[str, Item]:
    """
    Creates a dict of Item.

    Args:
        items (list[tuple]): a list of
        attributes for Item. Each element
        is a tuple that contains the name,
        widget and description for an Item.

    Returns:
        dict[str, Item]: a dict of Item. the
        key is the name attribute of the Item.
    """
    item_dict = {}
    for item in items:
        item_dict[item[0]] = Item(
            name=item[0],
            widget=item[1],
            description=item[2],
        )
    return item_dict
