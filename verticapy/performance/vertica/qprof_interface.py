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

import uuid
from typing import Optional, Union

import verticapy._config.config as conf
from verticapy.jupyter._javascript import read_package_file, replace_value
from verticapy.jupyter._widgets import Visualizer, Item, makeItems
from verticapy.performance.vertica.qprof import QueryProfiler
from verticapy.performance.vertica.qprof_utility import QprofUtility

if conf.get_import_success("IPython"):
    from IPython.display import display, HTML
    import ipywidgets as widgets


class QueryProfilerInterface(QueryProfiler):
    """
    A class that inherits
    ``QueryProfiler`` and
    adds visualization
    features.
    """

    def __init__(
        self,
        transactions: Union[None, str, list[int], list[tuple[int, int]]] = None,
        key_id: Optional[str] = None,
        resource_pool: Optional[str] = None,
        target_schema: Union[None, str, dict] = None,
        overwrite: bool = False,
        add_profile: bool = True,
        check_tables: bool = True,
        iterchecks: bool = False,
    ) -> None:
        super().__init__(
            transactions,
            key_id,
            resource_pool,
            target_schema,
            overwrite,
            add_profile,
            check_tables,
            iterchecks,
        )

        self.apply_tree = widgets.Checkbox(
            value=False, description="Apply tree", disabled=False, indent=False
        )
        self.tree_style = {}

        self.qsteps_switch = widgets.Checkbox(
            value=False, description="Qsteps switch", disabled=False, indent=False
        )

        # buttons to navigate through transactions
        next_button = widgets.Button(description="Next Query")
        prev_button = widgets.Button(description="Previous Query")
        next_button.on_click(self.next_button_clicked)
        prev_button.on_click(self.prev_button_clicked)
        self.transaction_buttons = widgets.HBox([prev_button, next_button])

        # Query Inofrmation - Query Text & Time
        self.query_display_info = widgets.HTML(
            value=f"""
        <b>Execution Time:</b> {self.get_qduration()} <br>
        <b>Target Schema:</b> {self.target_schema["v_internal"] if self.target_schema else ''} <br>
        <b>Transaction ID:</b> {self.transaction_id} <br>
        <b>Statement ID:</b> {self.statement_id} <br>
        <b>Key ID:</b> {self.key_id}
        """
        )
        self.query_display = widgets.VBox(
            [
                widgets.HTML(
                    layout={
                        "max_height": "320px",
                        "overflow_y": "auto",
                        "padding-left": "10px",
                    }
                )
            ]
        )
        self.update_query_display()

        self.index_widget = widgets.IntText(
            description="Index:", value=self.transactions_idx
        )
        # build the path_order array and set up a dropdown menu
        # to select a path id
        rows = QprofUtility._get_rows(self.get_qplan(print_plan=False))
        options = [None] + QprofUtility._get_path_order(rows)
        self.pathid_dropdown = Item(
            name="Path ID",
            widget=widgets.Dropdown(options=options, value=options[0]),
            description="select the path id for prunning the tree",
        )
        # button to apply the path id settings on the tree
        self.refresh_pathids = widgets.Button(description="Refresh")
        self.refresh_pathids.on_click(self.refresh_clicked)

        self.step_idx = widgets.IntText(description="Index:", value=0)

        # graph headers
        self.cpu_header = widgets.HTML(
            value=f"<h1><b>CPU Time by node and path_id - [query_idx: {self.index_widget.value}]</b></h1>"
        )
        self.qpt_header = widgets.HTML(
            value=f"<h1><b>Query Plan Tree - [query_idx: {self.index_widget.value}]</b></h1>"
        )
        self.qsteps_header = widgets.HTML(
            value=f"<h1><b>Query Execution Steps - [query_idx: {self.index_widget.value}]</b></h1>"
        )

    def get_qplan_tree(self, use_javascript=True):
        """
        Draws an interactive Query plan tree.

        Args:
            use_javascript (bool, optional): use javascript on tree.
            Defaults to ``True``.
        """
        self.use_javascript = use_javascript
        # widget for choosing the metrics
        options_dropwdown = [
            QprofUtility._get_metrics_name(i) for i in QprofUtility._get_metrics()
        ]
        dropdown1 = widgets.Dropdown(
            options=options_dropwdown,
            description="Metric # 1:",
            value="Execution time in ms",
            layout={"width": "260px"},
        )
        dropdown2 = widgets.Dropdown(
            options=options_dropwdown,
            description="Metric # 2:",
            value="Produced row count",
            layout={"width": "260px"},
        )
        tags = widgets.VBox([dropdown1, dropdown2])
        temp_rel_widget = widgets.ToggleButtons(
            options=["Temporary Relations", "Combined"],
            disabled=False,
            tooltips=[
                "Show separate temporary relations",
                "Show a condensed tree without any separate temporary relations",
            ],
        )
        self.colors = makeItems(
            [
                (
                    "color low",
                    widgets.ColorPicker(concise=False, value="#00ff00", disabled=False),
                    "",
                ),
                (
                    "color high",
                    widgets.ColorPicker(concise=False, value="#ff0000", disabled=False),
                    "",
                ),
            ]
        )

        tree_button = widgets.Button(description="Apply")
        tree_button.on_click(self.apply_tree_settings)
        tree_button_box = widgets.HBox(
            [tree_button], layout={"justify_content": "center"}
        )
        tree_settings = [
            temp_rel_widget,
            self.colors["color low"].get_item(),
            self.colors["color high"].get_item(),
            tree_button_box,
        ]
        self.tree_style = {
            "color_low": self.colors["color low"].get_child_attr("value"),
            "color_high": self.colors["color high"].get_child_attr("value"),
        }

        refresh_pathids_box = widgets.HBox(
            [self.refresh_pathids], layout={"justify_content": "center"}
        )
        accordion_items = {
            "Metrics": tags,
            "Path ID": widgets.VBox(
                [self.pathid_dropdown.get_item(), refresh_pathids_box]
            ),
            "Tree style": widgets.VBox(tree_settings),
            "Query Text": self.query_display,
        }
        query_text_index = list(accordion_items.keys()).index("Query Text")
        accordions = Visualizer._accordion(
            list(accordion_items.values()), accordion_items.keys()
        )
        accordions.selected_index = query_text_index
        header_box = widgets.HBox(
            [self.qpt_header], layout={"justify_content": "center"}
        )
        controls = {
            "metric1": tags.children[0],
            "metric2": tags.children[1],
            "index": self.index_widget,
            "path_id": self.pathid_dropdown.get_child(),
            "apply_tree_clicked": self.apply_tree,
            "temp_display": temp_rel_widget,
        }
        interactive_output = widgets.interactive_output(
            self.update_qplan_tree, controls
        )
        settings = [accordions, self.transaction_buttons, self.query_display_info]
        viz = Visualizer(
            settings_wids=settings, graph_wids=[header_box, interactive_output]
        )
        viz.display()

    def update_qplan_tree(
        self,
        metric1,
        metric2,
        index,
        path_id,
        apply_tree_clicked,
        temp_display,
    ):
        """
        Callback function that displays the Query Plan Tree.
        """
        metric = [
            QprofUtility._get_metrics_name(i, inv=True) for i in [metric1, metric2]
        ]
        if len(metric) == 0:
            metric = ["rows"]
        graph_id = "g" + str(uuid.uuid4())
        if self.pathid_dropdown.get_child_attr("disabled"):
            path_id = None
        if self.use_javascript == False:
            graph = super().get_qplan_tree(
                metric=metric,
                path_id=path_id,
                color_low=self.tree_style["color_low"],
                color_high=self.tree_style["color_high"],
                use_temp_relation=False if temp_display == "Combined" else True,
            )  # type: ignore
            html_widget = widgets.HTML(value=graph.pipe(format="svg").decode("utf-8"))
            box = widgets.HBox([html_widget])
            box.layout.justify_content = "center"
            display(box)
        else:
            raw = super().get_qplan_tree(
                metric=metric,
                path_id=path_id,
                return_graphviz=True,
                color_low=self.tree_style["color_low"],
                color_high=self.tree_style["color_high"],
                use_temp_relation=False if temp_display == "Combined" else True,
            )
            output = read_package_file("html/index.html")
            output = replace_value(output, "var dotSrc = [];", f"var dotSrc = `{raw}`;")
            output = replace_value(output, 'id="graph"', f'id="{graph_id}"')
            output = replace_value(output, "#graph", f"#{graph_id}")
            display(HTML(output))
        self.qpt_header.value = (
            f"<h1><b>Query Plan Tree - [query_idx: {index}]</b></h1>"
        )

    # Event handlers for the buttons
    def next_button_clicked(self, button):
        """
        Callback function triggered when
        the user click on the button to
        go to the next transaction/query.

        Args:
            button (Any): represents the button that was clicked
        """
        button.disabled = True
        self.next()
        self.pathid_dropdown.set_child_attr("disabled", True)
        self.refresh_pathids.disabled = False
        self.index_widget.value = (self.index_widget.value + 1) % len(self.transactions)
        self.update_query_display()
        button.disabled = False

    def prev_button_clicked(self, button):
        """
        Callback function triggered
        when the user click on the
        button to go to the previous
        transaction/query.

        Args:
            button (Any): represents the button that was clicked
        """
        button.disabled = True
        self.previous()
        self.pathid_dropdown.set_child_attr("disabled", True)
        self.refresh_pathids.disabled = False
        self.index_widget.value = (self.index_widget.value - 1) % len(self.transactions)
        self.update_query_display()
        button.disabled = False

    def refresh_clicked(self, button):
        """
        ...
        """
        button.disabled = True
        rows = QprofUtility._get_rows(self.get_qplan(print_plan=False))
        options = [None] + QprofUtility._get_path_order(rows)
        self.pathid_dropdown.set_child_attr("options", options)
        self.pathid_dropdown.set_child_attr("disabled", False)

    def apply_tree_settings(self, _):
        """
        ...
        """
        self.tree_style = {
            "color_low": self.colors["color low"].get_child_attr("value"),
            "color_high": self.colors["color high"].get_child_attr("value"),
        }
        self.apply_tree.value = not self.apply_tree.value

    def update_query_display(self):
        """
        Updates the query display text widget with the current query.
        """
        current_query = self.get_request(print_sql=False, return_html=True)
        self.query_display.children[0].value = current_query
        self.query_display_info.value = f"""
        <b>Execution Time:</b> {self.get_qduration()} <br>
        <b>Target Schema:</b> {self.target_schema["v_internal"] if self.target_schema else ''} <br>
        <b>Transaction ID:</b> {self.transaction_id} <br>
        <b>Statement ID:</b> {self.statement_id} <br>
        <b>Key ID:</b> {self.key_id}
        """

    ##########################################################################

    def get_qsteps(self):
        """
        Returns an interactive Query Execution Steps chart.
        """
        self.qsteps_items = makeItems(
            [
                (
                    "unit",
                    widgets.Dropdown(options=["s", "m", "h"], value="s"),
                    "Unit used to draw the chart",
                ),
                (
                    "kind",
                    widgets.Dropdown(options=["bar", "barh"], value="bar"),
                    "Chart type",
                ),
                (
                    "categoryorder",
                    widgets.Dropdown(
                        options=QprofUtility._get_categoryorder(),
                        value="sum descending",
                    ),
                    "How to sort the bars",
                ),
            ]
        )
        self.qsteps_controls = {
            "unit": self.qsteps_items["unit"].get_child_attr("value"),
            "kind": self.qsteps_items["kind"].get_child_attr("value"),
            "categoryorder": self.qsteps_items["categoryorder"].get_child_attr("value"),
        }
        qsteps_button = widgets.Button(description="Apply")
        qsteps_button.on_click(self.qsteps_clicked)
        qsteps_button_box = widgets.HBox(
            [qsteps_button], layout={"justify_content": "center"}
        )

        accordion_items = {
            "qsteps settings": widgets.VBox(
                [
                    self.qsteps_items["unit"].get_item(),
                    self.qsteps_items["kind"].get_item(),
                    self.qsteps_items["categoryorder"].get_item(),
                    qsteps_button_box,
                ]
            )
        }
        accordions = Visualizer._accordion(
            list(accordion_items.values()), accordion_items.keys()
        )
        qsteps_settings = [
            accordions,
            self.transaction_buttons,
        ]

        controls = {
            "query_idx": self.index_widget,
            "clicked": self.qsteps_switch,
        }

        interactive_output = widgets.interactive_output(self.update_qsteps, controls)
        header_box = widgets.HBox(
            [self.qsteps_header], layout={"justify_content": "center"}
        )
        settings_layout = {}
        graph_layout = {}
        settings_layout["width"] = "25%"
        settings_layout["height"] = "500px"
        graph_layout["width"] = "75%"
        graph_layout["height"] = "500px"
        viz = Visualizer(
            settings_wids=qsteps_settings,
            graph_wids=[header_box, interactive_output],
            settings_layout_override=settings_layout,
            graph_layout_override=graph_layout,
        )
        viz.display()

    def update_qsteps(
        self,
        query_idx,
        clicked,
    ):
        """
        Callback function that displays the Query Execution Steps.
        """
        display(
            super().get_qsteps(
                unit=self.qsteps_controls["unit"],
                kind=self.qsteps_controls["kind"],
                categoryorder=self.qsteps_controls["categoryorder"],
            )
        )
        self.qsteps_header.value = (
            f"<h1><b>Query Execution Steps - [query_idx: {query_idx}]</b></h1>"
        )

    def qsteps_clicked(self, _):
        self.qsteps_controls = {
            "unit": self.qsteps_items["unit"].get_child_attr("value"),
            "kind": self.qsteps_items["kind"].get_child_attr("value"),
            "categoryorder": self.qsteps_items["categoryorder"].get_child_attr("value"),
        }
        self.qsteps_switch.value = not self.qsteps_switch.value

    #############################################################

    def get_cpu_time(self):
        """
        Returns an interactive CPU Time
        by node and ``path_id``.
        """
        cpu_items = makeItems(
            [
                (
                    "kind",
                    widgets.Dropdown(options=["bar", "barh"], value="bar"),
                    "Chart type",
                ),
                (
                    "categoryorder",
                    widgets.Dropdown(
                        options=QprofUtility._get_categoryorder(),
                        value="max descending",
                    ),
                    "How to sort the bars",
                ),
            ]
        )

        controls = {
            "query_idx": self.index_widget,
            "kind": cpu_items["kind"].get_child(),
            "categoryorder": cpu_items["categoryorder"].get_child(),
        }
        interactive_output = widgets.interactive_output(self.update_cpu_time, controls)
        header_box = widgets.HBox(
            [self.cpu_header], layout={"justify_content": "center"}
        )
        cpu_settings = [
            cpu_items["kind"].get_item(),
            cpu_items["categoryorder"].get_item(),
            self.transaction_buttons,
        ]
        viz = Visualizer(
            settings_wids=[widgets.VBox(cpu_settings)],
            graph_wids=[header_box, interactive_output],
            orientation="v",
        )
        viz.display()

    def update_cpu_time(
        self,
        query_idx,
        kind,
        categoryorder,
    ):
        """
        Callback function that displays the
        CPU Time by node and ``path_id``.
        """
        display(
            super().get_cpu_time(
                kind=kind,
                categoryorder=categoryorder,
            )
        )
        self.cpu_header.value = (
            f"<h1><b>CPU Time by node and path_id - [query_idx: {query_idx}]</b></h1>"
        )

    ##########################################################

    def step(self):
        """
        Function to return the
        ``QueryProfiler`` Step.
        """
        steps_id = self.get_step_funcs()
        self.number_of_steps = len(steps_id)
        next_step = widgets.Button(description="Next step")
        previous_step = widgets.Button(description="Previous step")
        next_step.on_click(self.next_step_clicked)
        previous_step.on_click(self.prev_step_clicked)
        self.step_text = widgets.HTML(value=f"Step {self.step_idx.value}")
        step_navigation = widgets.HBox([previous_step, self.step_text, next_step])
        interactive_output = widgets.interactive_output(
            self.update_step, {"step_idx": self.step_idx}
        )
        out = widgets.Output()
        with out:
            out.clear_output(wait=True)
            display(interactive_output)
        display(widgets.VBox([step_navigation, out]))

    def update_step(self, step_idx):
        """
        Callback function that
        calls a step function
        based on the step idx.
        """
        steps_id = self.get_step_funcs()
        if steps_id[step_idx] == NotImplemented:
            display("NotImplemented")
        else:
            display(steps_id[step_idx]())
        self.step_text.value = f"Step {step_idx}"

    def next_step_clicked(self, button):
        """
        ...
        """
        button.disabled = True
        self.step_idx.value = (self.step_idx.value + 1) % self.number_of_steps
        button.disabled = False

    def prev_step_clicked(self, button):
        """
        ...
        """
        button.disabled = True
        self.step_idx.value = (self.step_idx.value - 1) % self.number_of_steps
        button.disabled = False

    def get_step_funcs(self):
        """
        ...
        """
        return {
            0: self.get_version,
            1: self.get_request,
            2: self.get_qduration,
            3: self.get_qsteps,
            4: NotImplemented,
            5: self.get_qplan_tree,
            6: self.get_qplan_profile,
            7: NotImplemented,
            8: NotImplemented,
            9: NotImplemented,
            10: self.get_query_events,
            11: NotImplemented,
            12: self.get_cpu_time,
            13: NotImplemented,
            14: self.get_qexecution,
            15: NotImplemented,
            16: NotImplemented,
            17: NotImplemented,
            18: NotImplemented,
            19: NotImplemented,
            20: self.get_rp_status,
            21: self.get_cluster_config,
        }
