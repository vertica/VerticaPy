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
from verticapy.performance.vertica.qprof_stats_tests import QueryProfilerStats
from verticapy.performance.vertica.qprof_utility import QprofUtility

if conf.get_import_success("IPython"):
    from IPython.display import display, HTML
    import ipywidgets as widgets


class QueryProfilerInterface(QueryProfilerStats):
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
        session_control: Union[None, dict, list[dict], str, list[str]] = None,
        overwrite: bool = False,
        add_profile: bool = True,
        check_tables: bool = True,
        iterchecks: bool = False,
        print_info: bool = True,
    ) -> None:
        super().__init__(
            transactions=transactions,
            key_id=key_id,
            resource_pool=resource_pool,
            target_schema=target_schema,
            session_control=session_control,
            overwrite=overwrite,
            add_profile=add_profile,
            check_tables=check_tables,
            iterchecks=iterchecks,
            print_info=print_info,
        )

        self.apply_tree = widgets.Checkbox(
            value=False, description="Apply tree", disabled=False, indent=False
        )
        self.tree_style = {}

        self.qsteps_switch = widgets.Checkbox(
            value=False, description="Qsteps switch", disabled=False, indent=False
        )

        self.accordions = None

        # buttons to navigate through transactions
        next_button = widgets.Button(description="Next Query")
        prev_button = widgets.Button(description="Previous Query")
        # self.test_output = widgets.Output()
        next_button.on_click(self.next_button_clicked)
        prev_button.on_click(self.prev_button_clicked)
        self.transaction_buttons = widgets.VBox(
            [widgets.HBox([prev_button, next_button])]
        )
        # Jumpt to query dropdown
        self.query_select_dropdown = widgets.Dropdown(
            description="Jump to query",
            options=[i for i in range(len(self.get_queries()))],
        )
        self.query_select_dropdown.style.description_width = "100px"
        # Success and Failure HTML
        self.success_html = """
        <svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' width='24' height='24'>
            <circle cx='12' cy='12' r='12' fill='#4CAF50'/>
            <path d='M9 19c-.256 0-.512-.098-.707-.293l-5-5c-.39-.39-.39-1.024 0-1.414s1.024-.39 1.414 0L9 16.586l10.293-10.293c.39-.39 1.024-.39 1.414 0s.39 1.024 0 1.414l-11 11c-.195.195-.451.293-.707.293z' fill='white'/>
        </svg>
        """
        self.failure_html = """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
            <circle cx="12" cy="12" r="12" fill="#F44336"/>
            <path d="M15.5355 8.46447a1 1 0 0 0-1.4142 0L12 10.5858 9.87868 8.46447a1 1 0 0 0-1.4142 1.4142L10.5858 12 8.46447 14.1213a1 1 0 0 0 1.4142 1.4142L12 13.4142l2.1213 2.1213a1 1 0 0 0 1.4142-1.4142L13.4142 12l2.1213-2.1213a1 1 0 0 0 0-1.4142z" fill="white"/>
        </svg>
        """
        # Query Inofrmation - Query Text & Time
        self.query_display_info = widgets.HTML(
            value=f"""
        <b>Query Execution Success:</b> {self.success_html if self.query_success else self.failure_html} <br>
        <b>Execution Time:</b> {self.get_qduration()} (seconds)<br>
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
        self.session_param_display = []
        self.update_session_param_display()
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
        # Tooltip search
        self.tooltip_search_widget_text = widgets.Text(
            placeholder="Enter part of tooltip to search"
        )
        self.tooltip_search_widget_button = widgets.Button(
            description="Search by Tooltip",
            layout=widgets.Layout(width="200px"),
        )
        self.tooltip_search_dummy = widgets.Text()
        self.tooltip_search_widget_button.on_click(
            self.tooltip_search_widget_button_action
        )
        horizontal_line = widgets.HTML(
            value="<hr style='border: 1px solid black; width: 100px;'>"
        )
        self.tooltip_search_widget = widgets.VBox(
            [
                horizontal_line,
                self.tooltip_search_widget_text,
                self.tooltip_search_widget_button,
            ],
            layout=widgets.Layout(
                justify_content="center", align_items="center", width="100%"
            ),
        )

        # Opeartor Search
        self.search_operator_dummy = widgets.Text()
        self.search_operator_options = self._get_all_op()
        self.search_operator_dropdown1 = widgets.Dropdown(
            options=[None] + self.search_operator_options,
            description="Critera # 1:",
            value=None,
            layout={"width": "260px"},
        )
        self.search_operator_dropdown2 = widgets.Dropdown(
            options=[None] + self.search_operator_options,
            description="Critera # 2:",
            value=None,
            layout={"width": "260px"},
        )
        self.search_operator_button = widgets.Button(
            description="Search by operators",
            layout=widgets.Layout(width="200px"),
        )
        self.search_operator_button.on_click(self.search_operator_button_button_action)
        self.search_operator_widget = widgets.VBox(
            [
                horizontal_line,
                self.search_operator_dropdown1,
                self.search_operator_dropdown2,
                self.search_operator_button,
            ],
            layout=widgets.Layout(
                justify_content="center", align_items="center", width="100%"
            ),
        )
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

    def get_qplan_tree(self, use_javascript=True, hide_settings=False, **style_kwargs):
        """
        Draws an interactive Query plan tree.

        Args:
            use_javascript (bool, optional): use javascript on tree.
            Defaults to ``True``.
        """
        self.style_kwargs = style_kwargs
        self.use_javascript = use_javascript
        # widget for choosing the metrics
        options_dropwdown = [
            QprofUtility._get_metrics_name(i) for i in QprofUtility._get_metrics()
        ]
        dropdown1 = widgets.Dropdown(
            options=options_dropwdown,
            description="Metric # 1:",
            value="Execution time in \u00b5s",
            layout={"width": "260px"},
        )
        dropdown2 = widgets.Dropdown(
            options=options_dropwdown,
            description="Metric # 2:",
            value="Produced row count",
            layout={"width": "260px"},
        )
        tooltip_aggregated_widget = widgets.Checkbox(
            value=True,
            # layout={"width": "260px"},
            description="Aggregate",
            disabled=False,
        )
        tooltip_operator_widget = widgets.Checkbox(
            value=False,
            # layout={"width": "260px"},
            description="Operator",
            disabled=False,
        )
        tooltip_descriptor_widget = widgets.Checkbox(
            value=False,
            # layout={"width": "260px"},
            description="Descriptor",
            disabled=False,
        )
        tooltip_header_widget = widgets.HTML("<b>Select tooltip info:</b>")
        tooltip_complete_widget = widgets.VBox(
            [
                tooltip_aggregated_widget,
                tooltip_operator_widget,
                tooltip_descriptor_widget,
            ],
            #    layout={"width": "260px"},
        )
        tags = widgets.VBox(
            [dropdown1, dropdown2, tooltip_header_widget, tooltip_complete_widget]
        )
        temp_rel_widget = widgets.ToggleButtons(
            options=["Temporary Relations", "Combined"],
            disabled=False,
            tooltips=[
                "Show separate temporary relations",
                "Show a condensed tree without any separate temporary relations",
            ],
        )
        projections_dml_widget = widgets.ToggleButtons(
            options=["DML projections", "No DML projections"],
            disabled=False,
            tooltips=[
                "If the operation is a DML, all target projections are displayed",
                "Default view",
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
            projections_dml_widget,
            self.colors["color low"].get_item(),
            self.colors["color high"].get_item(),
            tree_button_box,
        ]
        self.tree_style = {
            "color_low": self.colors["color low"].get_child_attr("value"),
            "color_high": self.colors["color high"].get_child_attr("value"),
            **self.style_kwargs,
        }

        refresh_pathids_box = widgets.HBox(
            [self.refresh_pathids], layout={"justify_content": "center"}
        )
        accordion_items = {
            "Metrics": tags,
            "Path ID": widgets.VBox(
                [
                    self.pathid_dropdown.get_item(),
                    # refresh_pathids_box,
                    self.tooltip_search_widget,
                    self.search_operator_widget,
                ]
            ),
            "Tree style": widgets.VBox(tree_settings),
            "Query text": self.query_display,
            "Session Parameters": self.session_param_display,
        }
        query_text_index = list(accordion_items.keys()).index("Query text")
        self.accordions = Visualizer._accordion(
            list(accordion_items.values()), accordion_items.keys()
        )
        self.accordions.selected_index = query_text_index
        header_box = widgets.HBox(
            [self.qpt_header], layout={"justify_content": "center"}
        )
        controls = {
            "index": self.query_select_dropdown,
            "metric1": tags.children[0],
            "metric2": tags.children[1],
            "display_tooltip_agg_metrics": tags.children[3].children[0],
            "display_tooltip_op_metrics": tags.children[3].children[1],
            "display_tooltip_descriptors": tags.children[3].children[2],
            "path_id": self.pathid_dropdown.get_child(),
            "apply_tree_clicked": self.apply_tree,
            "temp_display": temp_rel_widget,
            "projection_display": projections_dml_widget,
            "tooltip_filter": self.tooltip_search_dummy,
            "op_filter": self.search_operator_dummy,
        }
        interactive_output = widgets.interactive_output(
            self.update_qplan_tree, controls
        )
        if hide_settings:
            self.accordions.layout.display = "none"
            self.transaction_buttons.layout.display = "none"
            self.query_select_dropdown.layout.display = "none"
            self.query_display_info.layout.display = "none"
        settings = [
            self.accordions,
            self.transaction_buttons,
            self.query_select_dropdown,
            self.query_display_info,
        ]
        viz = Visualizer(
            settings_wids=settings,
            graph_wids=[header_box, interactive_output],
            orientation="v" if hide_settings else "h",
        )
        viz.display()

    def update_qplan_tree(
        self,
        metric1,
        metric2,
        display_tooltip_agg_metrics,
        display_tooltip_op_metrics,
        display_tooltip_descriptors,
        index,
        path_id,
        apply_tree_clicked,
        temp_display,
        projection_display,
        tooltip_filter,
        op_filter,
    ):
        """
        Callback function that displays the Query Plan Tree.
        """
        # Create an output widget to hold the hourglass and the tree
        output = widgets.Output()
        display(output)

        # Show hourglass in the output before starting long-running task
        with output:
            output.clear_output(wait=True)  # Clear any previous content
            # Create the hourglass icon
            hourglass_icon = widgets.HTML(
                value='<i class="fa fa-hourglass-half" style="font-size:48px;color:gray;"></i>',
                layout=widgets.Layout(display="flex", justify_content="center"),
            )
            vbox = widgets.VBox(
                [hourglass_icon],
                layout=widgets.Layout(justify_content="center", align_items="center"),
            )
            display(vbox)

        # Processing the inputs and generate the tree (long running task)
        metric = [
            QprofUtility._get_metrics_name(i, inv=True) for i in [metric1, metric2]
        ]
        if len(metric) == 0:
            metric = ["rows"]

        graph_id = "g" + str(uuid.uuid4())
        self.query_select_button_selected(index)
        if self.pathid_dropdown.get_child_attr("disabled"):
            path_id = None

        # Ensure the hourglass stays displayed during the processing of get_qplan_tree
        if self.use_javascript == False:
            graph = super().get_qplan_tree(
                metric=metric,
                path_id=path_id,
                color_low=self.tree_style["color_low"],
                color_high=self.tree_style["color_high"],
                use_temp_relation=False if temp_display == "Combined" else True,
                display_projections_dml=False
                if projection_display == "Default"
                else True,
                return_html=False,
                display_tooltip_agg_metrics=display_tooltip_agg_metrics,
                display_tooltip_op_metrics=display_tooltip_op_metrics,
                display_tooltip_descriptors=display_tooltip_descriptors,
                tooltip_filter=tooltip_filter,
                op_filter=eval(op_filter) if op_filter != "" else None,
                **self.style_kwargs,
            )  # type: ignore

            # After long-running task is done, update output with the result
            with output:
                output.clear_output(
                    wait=True
                )  # Clear the hourglass before displaying the tree
                html_widget = widgets.HTML(
                    value=graph.pipe(format="svg").decode("utf-8")
                )
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
                display_projections_dml=False
                if projection_display == "Default"
                else True,
                return_html=False,
                display_tooltip_agg_metrics=display_tooltip_agg_metrics,
                display_tooltip_op_metrics=display_tooltip_op_metrics,
                display_tooltip_descriptors=display_tooltip_descriptors,
                tooltip_filter=tooltip_filter,
                op_filter=eval(op_filter) if op_filter != "" else None,
                **self.style_kwargs,
            )

            output_html = read_package_file("html/index.html")
            output_html = replace_value(
                output_html, "var dotSrc = [];", f"var dotSrc = `{raw}`;"
            )
            output_html = replace_value(output_html, 'id="graph"', f'id="{graph_id}"')
            output_html = replace_value(output_html, "#graph", f"#{graph_id}")

            with output:
                output.clear_output(wait=True)  # Clear the hourglass
                display(HTML(output_html))

        # Update the header after the tree is displayed
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
        total = len(self.get_queries())
        current = self.query_select_dropdown.value
        if current == (total - 1):
            next = 0
        elif current != 0:
            next = current % (total - 1) + 1
        else:
            next = 1
        self.query_select_dropdown.value = next
        self.refresh_dropwdown_inside_path_id()
        # self.next()
        # self.pathid_dropdown.set_child_attr("disabled", True)
        # self.refresh_pathids.disabled = False
        # self.index_widget.value = (self.index_widget.value + 1) % len(self.transactions)
        # self.step_idx.value = self.index_widget.value
        # self.update_query_display()
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
        total = len(self.get_queries())
        current = self.query_select_dropdown.value
        if current == 0:
            previous = total - 1
        else:
            previous = (current - 1) % (total - 1)
        self.query_select_dropdown.value = previous
        self.previous()
        self.refresh_dropwdown_inside_path_id()
        # self.pathid_dropdown.set_child_attr("disabled", True)
        # self.refresh_pathids.disabled = False
        # self.index_widget.value = (self.index_widget.value - 1) % len(self.transactions)
        # self.step_idx.value = self.index_widget.value
        # self.update_query_display()
        button.disabled = False

    def tooltip_search_widget_button_action(self, button):
        button.disabled = True
        self.tooltip_search_dummy.value = self.tooltip_search_widget_text.value
        button.disabled = False

    def search_operator_button_button_action(self, button):
        button.disabled = True
        values = None
        value1 = self.search_operator_dropdown1.value
        value2 = self.search_operator_dropdown2.value
        if value1 != None:
            if value2 != None:
                values = [value1, value2]
            else:
                values = [value1]
        else:
            if value2 != None:
                values = [value2]
        self.search_operator_dummy.value = str(values) if not None else ""
        button.disabled = False

    def query_select_button_selected(self, selection):
        """
        Callback function triggered
        when the user selects the index of
        a particular query.

        Args:
            Dropdown selection (Any): represents the selection
        """
        # self.pathid_dropdown.set_child_attr("disabled", True)
        self.refresh_pathids.disabled = False
        self.index_widget.value = selection
        self.step_idx.value = selection
        self.set_position(selection)
        self.update_query_display()
        self.update_session_param_display()

    def refresh_clicked(self, button):
        """
        ...
        """
        button.disabled = True
        rows = QprofUtility._get_rows(self.get_qplan(print_plan=False))
        options = [None] + QprofUtility._get_path_order(rows)
        self.pathid_dropdown.set_child_attr("options", options)
        self.pathid_dropdown.set_child_attr("disabled", False)

    def refresh_dropwdown_inside_path_id(self):
        """
        ...
        """
        # Update path id dropdown options
        rows = QprofUtility._get_rows(self.get_qplan(print_plan=False))
        options = [None] + QprofUtility._get_path_order(rows)
        self.pathid_dropdown.set_child_attr("options", options)
        self.pathid_dropdown.set_child_attr("disabled", False)
        # Update Search dropdown options
        self.search_operator_options = self._get_all_op()
        self.search_operator_dropdown1.options = [None] + self.search_operator_options
        self.search_operator_dropdown2.options = [None] + self.search_operator_options

    def apply_tree_settings(self, _):
        """
        ...
        """
        self.tree_style = {
            "color_low": self.colors["color low"].get_child_attr("value"),
            "color_high": self.colors["color high"].get_child_attr("value"),
            **self.style_kwargs,
        }
        self.apply_tree.value = not self.apply_tree.value

    def update_query_display(self):
        """
        Updates the query display text widget with the current query.
        """
        current_query = self.get_request(print_sql=False, return_html=True)
        self.query_display.children[0].value = current_query
        self.query_display_info.value = f"""
        <b>Query Execution Success:</b> {self.success_html if self.query_success else self.failure_html} <br>
        <b>Execution Time:</b> {self.get_qduration()} (seconds)<br>
        <b>Target Schema:</b> {self.target_schema["v_internal"] if self.target_schema else ''} <br>
        <b>Transaction ID:</b> {self.transaction_id} <br>
        <b>Statement ID:</b> {self.statement_id} <br>
        <b>Key ID:</b> {self.key_id}
        """

    def update_session_param_display(self):
        """
        Updates the Session parameter display text widget with the current query.
        """
        rows = []
        dict_list = self.session_params_non_default_current
        if isinstance(dict_list, dict):
            dict_list = [dict_list]
        for dictionary in dict_list:
            for key, value in dictionary.items():
                # Create a key-value pair layout
                key_label = widgets.HTML(
                    value=f"<b>{key}:</b>",
                    # layout=widgets.Layout(width='200px', text_align='right')
                )
                value_label = widgets.HTML(
                    value=f"{value}",
                    # layout=widgets.Layout(width='200px', text_align='left')
                )
                # Arrange key and value side by side in a horizontal box
                row = widgets.HBox(
                    [key_label, value_label], layout=widgets.Layout(padding="5px")
                )
                rows.append(row)

        # Add a centered title to the widget display
        title = widgets.HTML(
            value="<h4 style='background-color: #f0f0f0; padding: 5px; border-radius: 5px; margin: 0; text-align: center;'>Non-default Parameters</h4>"
        )

        # Create a VBox for the entire display (title + key-value pairs)
        self.session_param_display = widgets.VBox([title] + rows)

    ##########################################################################

    def get_qsteps_(self):
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
            self.accordions,
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
            super().get_qsteps_(
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
            3: self.get_qsteps_,
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


class QueryProfilerComparison:
    """
    Initializes a QueryProfilerComparison object with two QueryProfilerInterface instances for side-by-side comparison.

    Parameters:
    qprof1 (QueryProfilerInterface): The first QueryProfilerInterface instance for comparison.
    qprof2 (QueryProfilerInterface): The second QueryProfilerInterface instance for comparison.
    """

    def __init__(self, qprof1, qprof2):
        self.qprof1 = qprof1
        self.qprof2 = qprof2

        self.query_info = self._create_query_info()

        self.dual_effect = True

        # Initial update of the trees
        nooutput = widgets.Output()
        with nooutput:
            self.qprof1.get_qplan_tree()
            self.qprof2.get_qplan_tree()

        if self.dual_effect:
            # Replace the children tuple of qprof2 with a new one that copies qprof1's first accordion child
            self.qprof2.accordions.children = (
                self.qprof1.accordions.children[0],
            ) + self.qprof2.accordions.children[1:]

            # Sync the accordion selection between qprof1 and qprof2
            self._sync_accordion_selection()

        self.controls = self._create_controls()
        self.side_by_side_ui = widgets.VBox([self.query_info, self.controls])

    def _create_controls(self):
        def create_interactive_controls(qprof):
            controls = {
                "index": qprof.query_select_dropdown,
                "metric1": qprof.accordions.children[0].children[0],
                "metric2": qprof.accordions.children[0].children[1],
                "display_tooltip_agg_metrics": qprof.accordions.children[0]
                .children[3]
                .children[0],
                "display_tooltip_op_metrics": qprof.accordions.children[0]
                .children[3]
                .children[1],
                "display_tooltip_descriptors": qprof.accordions.children[0]
                .children[3]
                .children[2],
                "path_id": qprof.pathid_dropdown.get_child(),
                "apply_tree_clicked": qprof.apply_tree,
                "temp_display": qprof.accordions.children[2].children[0],
                "projection_display": qprof.accordions.children[2].children[1],
            }
            return widgets.interactive_output(qprof.update_qplan_tree, controls)

        q1_control = self.qprof1.accordions
        q1_control.selected_index = None
        q1_control.layout.width = "50%"
        q1_interactive = create_interactive_controls(self.qprof1)
        q1_interactive = widgets.HBox(
            [q1_interactive],
            layout=widgets.Layout(width="50%", border="1px solid black"),
        )

        q2_control = self.qprof2.accordions
        q2_control.selected_index = None
        q2_control.layout.width = "50%"
        q2_interactive = create_interactive_controls(self.qprof2)
        q2_interactive = widgets.HBox(
            [q2_interactive],
            layout=widgets.Layout(width="50%", border="1px solid black"),
        )

        return widgets.VBox(
            [
                widgets.HBox([q1_control, q2_control]),
                widgets.HBox([q1_interactive, q2_interactive]),
            ]
        )

    def _sync_accordion_selection(self):
        """
        Synchronizes the accordion selection of qprof1 and qprof2.
        When an accordion is selected in qprof1, it automatically updates the selection in qprof2.
        """

        def on_accordion_change(change):
            """
            Callback function to update qprof2's accordion selection when qprof1's accordion selection changes.
            """
            if change["name"] == "selected_index" and change["new"] is not None:
                self.qprof2.accordions.selected_index = change["new"]

        # Observe changes in the selected_index of qprof1's accordion
        self.qprof1.accordions.observe(on_accordion_change, names="selected_index")

    def _create_query_info(self):
        # Get and set the layout for the query display info for both qprof1 and qprof2
        q1_info = self.qprof1.query_display_info
        q1_info.layout.display = "block"
        q1_info.layout.width = "50%"
        q1_info.layout.border = "1px solid black"

        q2_info = self.qprof2.query_display_info
        q2_info.layout.display = "block"
        q2_info.layout.width = "50%"
        q2_info.layout.border = "1px solid black"

        # Return an HBox containing the query display information side by side
        return widgets.HBox([q1_info, q2_info])

    def display(self):
        # Display the final side-by-side UI
        display(self.side_by_side_ui)
