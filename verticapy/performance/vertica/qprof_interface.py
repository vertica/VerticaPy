"""
Copyright  (c)  2018-2025 Open Text  or  one  of its
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
import types
from typing import Optional, Union

import verticapy._config.config as conf
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._print import print_message
from verticapy._utils._sql._vertica_version import check_minimum_version

from verticapy.jupyter._javascript import read_package_file, replace_value
from verticapy.jupyter._widgets import Visualizer, Item, makeItems
from verticapy.performance.vertica.qprof_stats_tests import QueryProfilerStats
from verticapy.performance.vertica.qprof_utility import QprofUtility

if conf.get_import_success("IPython"):
    import ipywidgets as widgets


class QueryProfilerInterface(QueryProfilerStats):
    """
    A class that inherits
    ``QueryProfiler`` and
    adds visualization
    features.
    """

    @check_minimum_version
    @save_verticapy_logs
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
        self.output = widgets.Output()
        self.apply_tree = widgets.Checkbox(
            value=False, description="Apply tree", disabled=False, indent=False
        )
        self.tree_style = {}

        self.qsteps_switch = widgets.Checkbox(
            value=False, description="Qsteps switch", disabled=False, indent=False
        )

        self._accordions = None

        # buttons to navigate through transactions
        next_button = widgets.Button(description="Next Query")
        prev_button = widgets.Button(description="Previous Query")
        # self.test_output = widgets.Output()
        next_button.on_click(self._next_button_clicked)
        prev_button.on_click(self._prev_button_clicked)
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
        self._success_html = """
        <svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' width='24' height='24'>
            <circle cx='12' cy='12' r='12' fill='#4CAF50'/>
            <path d='M9 19c-.256 0-.512-.098-.707-.293l-5-5c-.39-.39-.39-1.024 0-1.414s1.024-.39 1.414 0L9 16.586l10.293-10.293c.39-.39 1.024-.39 1.414 0s.39 1.024 0 1.414l-11 11c-.195.195-.451.293-.707.293z' fill='white'/>
        </svg>
        """
        self._failure_html = """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
            <circle cx="12" cy="12" r="12" fill="#F44336"/>
            <path d="M15.5355 8.46447a1 1 0 0 0-1.4142 0L12 10.5858 9.87868 8.46447a1 1 0 0 0-1.4142 1.4142L10.5858 12 8.46447 14.1213a1 1 0 0 0 1.4142 1.4142L12 13.4142l2.1213 2.1213a1 1 0 0 0 1.4142-1.4142L13.4142 12l2.1213-2.1213a1 1 0 0 0 0-1.4142z" fill="white"/>
        </svg>
        """
        # Query Inofrmation - Query Text & Time
        self._query_display_info = widgets.HTML(
            value=f"""
        <b>Query Execution Success:</b> {self._success_html if self.query_success else self._failure_html} <br>
        <b>Execution Time:</b> {self.get_qduration()} (seconds)<br>
        <b>Target Schema:</b> {self.target_schema["v_internal"] if self.target_schema else ''} <br>
        <b>Transaction ID:</b> {self.transaction_id} <br>
        <b>Statement ID:</b> {self.statement_id} <br>
        <b>Key ID:</b> {self.key_id}
        """
        )
        self._query_display = widgets.VBox(
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
        self._update_query_display()
        self.session_param_display = []
        self._update_session_param_display()
        self.tooltip_display_output = widgets.Output()
        self.tooltip_display_output.layout.height = "300px"
        self.tooltip_display_dropdown = widgets.Dropdown(options=[None], value=None)
        self._update_tooltip_display_dropdown()
        self.tooltip_display_dropdown.observe(
            self._display_tooltip_detail, names="value"
        )
        self.tooltip_display = widgets.VBox(
            [self.tooltip_display_dropdown, self.tooltip_display_output]
        )
        self._index_widget = widgets.IntText(
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
        self.refresh_pathids.on_click(self._refresh_clicked)
        # Tooltip search
        self._tooltip_search_widget_text = widgets.Text(
            placeholder="Enter part of tooltip to search"
        )
        self._tooltip_search_widget_button = widgets.Button(
            description="Search by Tooltip",
            icon="search",
            layout=widgets.Layout(width="200px"),
        )
        self._tooltip_search_dummy = widgets.Text()
        self._tooltip_search_widget_button.on_click(
            self._tooltip_search_widget_button_action
        )
        horizontal_line = widgets.HTML(
            value="<hr style='border: 1px solid black; width: 100px;'>"
        )
        self._tooltip_search_widget = widgets.VBox(
            [
                horizontal_line,
                self._tooltip_search_widget_text,
                self._tooltip_search_widget_button,
            ],
            layout=widgets.Layout(
                justify_content="center", align_items="center", width="100%"
            ),
        )

        # Opeartor Search
        self._search_operator_dummy = widgets.Text()
        self._search_operator_options = self._get_all_op()
        self._search_operator_dropdown1 = widgets.Dropdown(
            options=[None] + self._search_operator_options,
            description="Critera # 1:",
            value=None,
            layout={"width": "260px"},
        )
        self._search_operator_dropdown2 = widgets.Dropdown(
            options=[None] + self._search_operator_options,
            description="Critera # 2:",
            value=None,
            layout={"width": "260px"},
        )
        self._search_operator_button = widgets.Button(
            description="Search by operators",
            icon="search",
            layout=widgets.Layout(width="200px"),
        )
        self._search_operator_button.on_click(
            self._search_operator_button_button_action
        )
        self._search_operator_widget = widgets.VBox(
            [
                horizontal_line,
                self._search_operator_dropdown1,
                self._search_operator_dropdown2,
                self._search_operator_button,
            ],
            layout=widgets.Layout(
                justify_content="center", align_items="center", width="100%"
            ),
        )

        # Reset all search metrics
        self._reset_search_button = widgets.Button(
            description="Reset",
            layout=widgets.Layout(width="100px"),
            button_style="warning",
            icon="refresh",
        )
        self._reset_search_button.on_click(self._reset_search_button_action)
        self._reset_search_button_widget = widgets.VBox(
            [
                horizontal_line,
                self._reset_search_button,
            ],
            layout=widgets.Layout(
                justify_content="center", align_items="center", width="100%"
            ),
        )

        self._step_idx = widgets.IntText(description="Index:", value=0)

        # graph headers
        self._cpu_header = widgets.HTML(
            value=f"<h1><b>CPU Time by node and path_id - [query_idx: {self._index_widget.value}]</b></h1>"
        )
        self._qpt_header = widgets.HTML(
            value=f"<h1><b>Query Plan Tree - [query_idx: {self._index_widget.value}]</b></h1>"
        )
        self._qsteps_header = widgets.HTML(
            value=f"<h1><b>Query Execution Steps - [query_idx: {self._index_widget.value}]</b></h1>"
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
        tree_button.on_click(self._apply_tree_settings)
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
                    self._tooltip_search_widget,
                    self._search_operator_widget,
                    self._reset_search_button_widget,
                ]
            ),
            "Tree style": widgets.VBox(tree_settings),
            "Query text": self._query_display,
            "Session Parameters": self.session_param_display,
            "Detailed Tooltip": self.tooltip_display,
            "Summary": self._query_display_info,
        }
        # query_text_index = list(accordion_items.keys()).index("Query text")
        summary_index = list(accordion_items.keys()).index("Summary")
        self._accordions = Visualizer._accordion(
            list(accordion_items.values()), accordion_items.keys()
        )
        self._accordions.selected_index = summary_index
        header_box = widgets.HBox(
            [self._qpt_header], layout={"justify_content": "center"}
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
            "tooltip_filter": self._tooltip_search_dummy,
            "op_filter": self._search_operator_dummy,
        }
        self._interactive_output = widgets.interactive_output(
            self._update_qplan_tree, controls
        )
        if hide_settings:
            self._accordions.layout.display = "none"
            self.transaction_buttons.layout.display = "none"
            self.query_select_dropdown.layout.display = "none"
        settings = [
            self._accordions,
            self.transaction_buttons,
            self.query_select_dropdown,
        ]
        viz = Visualizer(
            settings_wids=settings,
            graph_wids=[header_box, self._interactive_output],
            orientation="v" if hide_settings else "h",
        )
        viz.display()

    def _update_qplan_tree(
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
        tooltip_filter=None,
        op_filter=None,
    ):
        """
        Callback function that displays the Query Plan Tree.
        """
        # Create an output widget to hold the hourglass and the tree
        print_message(self.output, "display")

        # Show hourglass in the output before starting long-running task
        with self.output:
            self.output.clear_output(wait=True)  # Clear any previous content
            # Create the hourglass icon
            hourglass_icon = widgets.HTML(
                value='<i class="fa fa-hourglass-half" style="font-size:48px;color:gray;"></i>',
                layout=widgets.Layout(display="flex", justify_content="center"),
            )
            vbox = widgets.VBox(
                [hourglass_icon],
                layout=widgets.Layout(justify_content="center", align_items="center"),
            )
            print_message(vbox, "display")

        # Processing the inputs and generate the tree (long running task)
        metric = [
            QprofUtility._get_metrics_name(i, inv=True) for i in [metric1, metric2]
        ]
        if len(metric) == 0:
            metric = ["rows"]

        graph_id = "g" + str(uuid.uuid4())
        self._query_select_button_selected(index)
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
                op_filter=eval(op_filter)
                if op_filter != "" and op_filter != None
                else None,
                **self.style_kwargs,
            )  # type: ignore

            # After long-running task is done, update output with the result
            with self.output:
                self.output.clear_output(
                    wait=True
                )  # Clear the hourglass before displaying the tree
                html_widget = widgets.HTML(
                    value=graph.pipe(format="svg").decode("utf-8")
                )
                box = widgets.HBox([html_widget])
                box.layout.justify_content = "center"
                print_message(box, "display")
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
                op_filter=eval(op_filter)
                if op_filter != "" and op_filter != None
                else None,
                **self.style_kwargs,
            )

            output_html = read_package_file("html/index.html")
            output_html = replace_value(
                output_html, "var dotSrc = [];", f"var dotSrc = `{raw}`;"
            )
            output_html = replace_value(output_html, 'id="graph"', f'id="{graph_id}"')
            output_html = replace_value(output_html, "#graph", f"#{graph_id}")

            with self.output:
                self.output.clear_output(wait=True)  # Clear the hourglass
                print_message(output_html, "display")

        # Update the header after the tree is displayed
        self._qpt_header.value = (
            f"<h1><b>Query Plan Tree - [query_idx: {index}]</b></h1>"
        )

    # Event handlers for the buttons
    def _next_button_clicked(self, button):
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
        self._refresh_dropwdown_inside_path_id()
        # self.next()
        # self.pathid_dropdown.set_child_attr("disabled", True)
        # self.refresh_pathids.disabled = False
        # self._index_widget.value = (self._index_widget.value + 1) % len(self.transactions)
        # self._step_idx.value = self._index_widget.value
        # self._update_query_display()
        button.disabled = False

    def _prev_button_clicked(self, button):
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
        self._refresh_dropwdown_inside_path_id()
        # self.pathid_dropdown.set_child_attr("disabled", True)
        # self.refresh_pathids.disabled = False
        # self._index_widget.value = (self._index_widget.value - 1) % len(self.transactions)
        # self._step_idx.value = self._index_widget.value
        # self._update_query_display()
        button.disabled = False

    def _tooltip_search_widget_button_action(self, button):
        button.disabled = True
        self._tooltip_search_dummy.value = self._tooltip_search_widget_text.value
        button.disabled = False

    def _search_operator_button_button_action(self, button):
        button.disabled = True
        values = None
        value1 = self._search_operator_dropdown1.value
        value2 = self._search_operator_dropdown2.value
        if value1 != None:
            if value2 != None:
                values = [value1, value2]
            else:
                values = [value1]
        else:
            if value2 != None:
                values = [value2]
        self._search_operator_dummy.value = str(values) if not None else ""
        button.disabled = False

    def _reset_search_button_action(self, button):
        button.disabled = True
        self._search_operator_dropdown1.value = None
        self._search_operator_dropdown2.value = None
        self._tooltip_search_widget_text.value = ""
        self._search_operator_dummy.value = ""
        self._tooltip_search_dummy.value = ""
        self.pathid_dropdown.set_child_attr("value", None)
        button.disabled = False

    def _query_select_button_selected(self, selection):
        """
        Callback function triggered
        when the user selects the index of
        a particular query.

        Args:
            Dropdown selection (Any): represents the selection
        """
        # self.pathid_dropdown.set_child_attr("disabled", True)
        self.refresh_pathids.disabled = False
        self._index_widget.value = selection
        self._step_idx.value = selection
        self.set_position(selection)
        self._update_query_display()
        self._update_session_param_display()
        self._update_tooltip_display_dropdown()

    def _refresh_clicked(self, button):
        """
        ...
        """
        button.disabled = True
        rows = QprofUtility._get_rows(self.get_qplan(print_plan=False))
        options = [None] + QprofUtility._get_path_order(rows)
        self.pathid_dropdown.set_child_attr("options", options)
        self.pathid_dropdown.set_child_attr("disabled", False)

    def _refresh_dropwdown_inside_path_id(self):
        """
        ...
        """
        # Update path id dropdown options
        rows = QprofUtility._get_rows(self.get_qplan(print_plan=False))
        options = [None] + QprofUtility._get_path_order(rows)
        self.pathid_dropdown.set_child_attr("options", options)
        self.pathid_dropdown.set_child_attr("disabled", False)
        # Update Search dropdown options
        self._search_operator_options = self._get_all_op()
        self._search_operator_dropdown1.options = [None] + self._search_operator_options
        self._search_operator_dropdown2.options = [None] + self._search_operator_options

    def _apply_tree_settings(self, _):
        """
        ...
        """
        self.tree_style = {
            "color_low": self.colors["color low"].get_child_attr("value"),
            "color_high": self.colors["color high"].get_child_attr("value"),
            **self.style_kwargs,
        }
        self.apply_tree.value = not self.apply_tree.value

    def _update_query_display(self):
        """
        Updates the query display text widget with the current query.
        """
        current_query = self.get_request(print_sql=False, return_html=True)
        self._query_display.children[0].value = current_query
        self._query_display_info.value = f"""
        <b>Query Execution Success:</b> {self._success_html if self.query_success else self._failure_html} <br>
        <b>Execution Time:</b> {self.get_qduration()} (seconds)<br>
        <b>Target Schema:</b> {self.target_schema["v_internal"] if self.target_schema else ''} <br>
        <b>Transaction ID:</b> {self.transaction_id} <br>
        <b>Statement ID:</b> {self.statement_id} <br>
        <b>Key ID:</b> {self.key_id}
        """

    def _update_session_param_display(self):
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

    def _update_tooltip_display_dropdown(self):
        """
        Update the options for dropdown display
        """
        options = list(self._get_tooltips().keys())
        self.tooltip_display_dropdown.options = [None] + options
        self._tooltip_compelte = self._get_tooltips()

    def _display_tooltip_detail(self, change):
        new_value = change["new"] if "new" in change else None
        with self.tooltip_display_output:
            self.tooltip_display_output.clear_output(wait=True)
            print(
                self._tooltip_compelte[new_value]
            ) if new_value is not None else print()

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
            self._accordions,
            self.transaction_buttons,
        ]

        controls = {
            "query_idx": self._index_widget,
            "clicked": self.qsteps_switch,
        }

        interactive_output = widgets.interactive_output(self.update_qsteps, controls)
        header_box = widgets.HBox(
            [self._qsteps_header], layout={"justify_content": "center"}
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
        print_message(
            super().get_qsteps_(
                unit=self.qsteps_controls["unit"],
                kind=self.qsteps_controls["kind"],
                categoryorder=self.qsteps_controls["categoryorder"],
            ),
            "display",
        )
        self._qsteps_header.value = (
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
            "query_idx": self._index_widget,
            "kind": cpu_items["kind"].get_child(),
            "categoryorder": cpu_items["categoryorder"].get_child(),
        }
        interactive_output = widgets.interactive_output(self.update_cpu_time, controls)
        header_box = widgets.HBox(
            [self._cpu_header], layout={"justify_content": "center"}
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
        print_message(
            super().get_cpu_time(
                kind=kind,
                categoryorder=categoryorder,
            ),
            "display",
        )
        self._cpu_header.value = (
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
        next_step.on_click(self._next_step_clicked)
        previous_step.on_click(self._prev_step_clicked)
        self.step_text = widgets.HTML(value=f"Step {self._step_idx.value}")
        step_navigation = widgets.HBox([previous_step, self.step_text, next_step])
        interactive_output = widgets.interactive_output(
            self.update_step, {"step_idx": self._step_idx}
        )
        out = widgets.Output()
        with out:
            out.clear_output(wait=True)
            print_message(interactive_output, "display")
        print_message(widgets.VBox([step_navigation, out]), "display")

    def update_step(self, step_idx):
        """
        Callback function that
        calls a step function
        based on the step idx.
        """
        steps_id = self.get_step_funcs()
        if steps_id[step_idx] == NotImplemented:
            print_message("NotImplemented", "display")
        else:
            print_message(steps_id[step_idx](), "display")
        self.step_text.value = f"Step {step_idx}"

    def _next_step_clicked(self, button):
        """
        ...
        """
        button.disabled = True
        self._step_idx.value = (self._step_idx.value + 1) % self.number_of_steps
        button.disabled = False

    def _prev_step_clicked(self, button):
        """
        ...
        """
        button.disabled = True
        self._step_idx.value = (self._step_idx.value - 1) % self.number_of_steps
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
    Initializes a ``QueryProfilerComparison`` object with
    two ``QueryProfilerInterface`` instances for side-by-side
    comparison.

    Parameters:
     - qprof1 (``QueryProfilerInterface``): The first
        ``QueryProfilerInterface`` instance for comparison.
     - qprof2 (``QueryProfilerInterface``): The second
        ``QueryProfilerInterface`` instance for comparison.
    """

    @check_minimum_version
    @save_verticapy_logs
    def __init__(self, qprof1, qprof2):
        self.qprof1 = qprof1
        self.qprof2 = qprof2

        self.dual_effect = True
        # Initial update of the trees
        nooutput = widgets.Output()
        with nooutput:
            self.qprof1.get_qplan_tree()
            self.qprof2.get_qplan_tree()

        metric_toggle_button = widgets.ToggleButton(
            value=True,
            disabled=False,
            button_style="success",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Turn on to sync with left QueryProfiler object.",
            layout=widgets.Layout(width="40px", height="40px"),
            icon="sync",
        )

        # by default making Sync on
        self.sync_all_checkboxes()
        self._sync_metric_values(True)

        def on_metric_toggle_change(change):
            if change["new"]:
                metric_toggle_button.button_style = "success"
                self.sync_all_checkboxes()
                self._sync_metric_values(True)

            if not change["new"]:
                metric_toggle_button.button_style = ""
                self._sync_metric_values(False)
                self.unsync_all_checkboxes()

        metric_toggle_button.observe(on_metric_toggle_change, names="value")
        accordion_children = list(qprof2._accordions.children[0].children)
        accordion_children.insert(len(accordion_children), metric_toggle_button)
        qprof2._accordions.children[0].children = tuple(accordion_children)

        if self.dual_effect:
            # Sync the accordion selection between qprof1 and qprof2
            self._sync_accordion_selection()

        # Separate control creation for qprof1 and qprof2
        self.controls = self._create_controls()
        self.side_by_side_ui = self.controls

    def _create_qprof1_controls(self):
        """
        Creates interactive controls for ``qprof1``.
        """
        interactive_output = self.qprof1._interactive_output
        return widgets.HBox(
            [interactive_output],
            layout=widgets.Layout(width="50%", border="1px solid black"),
        )

    def _create_qprof2_controls(self):
        """
        Creates interactive controls for ``qprof2``.
        """
        interactive_output = self.qprof2._interactive_output
        return widgets.HBox(
            [interactive_output],
            layout=widgets.Layout(width="50%", border="1px solid black"),
        )

    def _create_controls(self):
        """
        Creates side-by-side controls for
        both ``qprof1`` and ``qprof2``.
        """
        q1_control = self.qprof1._accordions
        q1_control.layout.width = "50%"

        q2_control = self.qprof2._accordions
        q2_control.layout.width = "50%"

        # Use separate functions to create the interactive controls
        q1_interactive = self._create_qprof1_controls()
        q2_interactive = self._create_qprof2_controls()

        return widgets.VBox(
            [
                widgets.HBox([q1_control, q2_control]),
                widgets.HBox([q1_interactive, q2_interactive]),
            ]
        )

    def _sync_accordion_selection(self):
        """
        Synchronizes the accordion selection of ``qprof1``
        and ``qprof2``.  When an accordion is selected in
        ``qprof1``, it automatically updates the selection
        in ``qprof2``.
        """

        def on_accordion_change(change):
            """
            Callback function to update ``qprof2`` accordion
            selection when ``qprof1`` accordion selection changes.
            """
            if change["name"] == "selected_index":
                self.qprof2._accordions.selected_index = change["new"]

        # Observe changes in the selected_index of qprof1's accordion
        self.qprof1._accordions.observe(on_accordion_change, names="selected_index")

    def _sync_metric_values(self, switch):
        """
        Sync the metric tab values for both ``qprof`` objects
        """

        def on_metric_dropdown1_change(change):
            """
            Callback function to update ``qprof2`` dropdown
            selection when ``qprof1`` dropdown selection changes.
            """
            val = change["new"]
            self.qprof2._accordions.children[0].children[0].value = val

        # Observe changes in the selected_index of qprof1's accordion
        if switch:
            if (
                not on_metric_dropdown1_change
                in self.qprof1._accordions.children[0]
                .children[0]
                ._trait_notifiers["value"]["change"]
            ):
                self.qprof1._accordions.children[0].children[0].observe(
                    on_metric_dropdown1_change, names="value"
                )
        else:
            # Removing the observer function to sync
            observers = (
                self.qprof1._accordions.children[0]
                .children[0]
                ._trait_notifiers["value"]["change"]
            )
            for observer in observers:
                if (
                    isinstance(observer, types.FunctionType)
                    and observer.__name__ == "on_metric_dropdown1_change"
                ):
                    observers.remove(observer)
                    break

        def on_metric_dropdown2_change(change):
            """
            Callback function to update ``qprof2``
            dropdown selection when ``qprof1``
            dropdown selection changes.
            """
            val = change["new"]
            self.qprof2._accordions.children[0].children[1].value = val

        # Observe changes in the selected_index of qprof1's accordion
        if switch:
            if (
                not on_metric_dropdown2_change
                in self.qprof1._accordions.children[0]
                .children[1]
                ._trait_notifiers["value"]["change"]
            ):
                self.qprof1._accordions.children[0].children[1].observe(
                    on_metric_dropdown2_change, names="value"
                )
        else:
            # Removing the observer function to sync
            observers = (
                self.qprof1._accordions.children[0]
                .children[1]
                ._trait_notifiers["value"]["change"]
            )
            for observer in observers:
                if (
                    isinstance(observer, types.FunctionType)
                    and observer.__name__ == "on_metric_dropdown2_change"
                ):
                    observers.remove(observer)
                    break

    def sync_all_checkboxes(self):
        """
        Syncs all checkboxes between qprof1 and qprof2.
        """

        def sync_checkboxes(checkbox1, checkbox2):
            """
            Syncs the values of two checkboxes.

            Parameters:
             - ``checkbox1 ``(Checkbox): The first checkbox widget.
             - ``checkbox2`` (Checkbox): The second checkbox widget.
            """

            def on_checkbox_change(change):
                """
                Callback function to sync the checkbox values.
                """
                checkbox2.value = change["new"]

            checkbox1.observe(on_checkbox_change, names="value")

        # Assuming the checkboxes are at children[0].children[3].children for both qprof1 and qprof2
        checkbox_agg_q1 = self.qprof1._accordions.children[0].children[3].children[0]
        checkbox_op_q1 = self.qprof1._accordions.children[0].children[3].children[1]
        checkbox_desc_q1 = self.qprof1._accordions.children[0].children[3].children[2]

        checkbox_agg_q2 = self.qprof2._accordions.children[0].children[3].children[0]
        checkbox_op_q2 = self.qprof2._accordions.children[0].children[3].children[1]
        checkbox_desc_q2 = self.qprof2._accordions.children[0].children[3].children[2]

        # Sync the checkboxes
        sync_checkboxes(checkbox_agg_q1, checkbox_agg_q2)
        sync_checkboxes(checkbox_op_q1, checkbox_op_q2)
        sync_checkboxes(checkbox_desc_q1, checkbox_desc_q2)

    def unsync_all_checkboxes(self):
        """
        Unbind synchronization of checkboxes
        between ``qprof1`` and ``qprof2``.
        """

        def unsync_checkboxes(checkbox1):
            """
            Unbind the checkbox observers.
            """
            observers = checkbox1._trait_notifiers["value"]["change"]
            for observer in observers:
                if (
                    isinstance(observer, types.FunctionType)
                    and observer.__name__ == "on_checkbox_change"
                ):
                    observers.remove(observer)
                    break

        # Assuming the checkboxes are at children[0].children[3].children for both qprof1 and qprof2
        checkbox_agg_q1 = self.qprof1._accordions.children[0].children[3].children[0]
        checkbox_op_q1 = self.qprof1._accordions.children[0].children[3].children[1]
        checkbox_desc_q1 = self.qprof1._accordions.children[0].children[3].children[2]

        # Unsync the checkboxes
        unsync_checkboxes(checkbox_agg_q1)
        unsync_checkboxes(checkbox_op_q1)
        unsync_checkboxes(checkbox_desc_q1)

    def get_qplan_tree(self):
        """
        Displays the final side-by-side UI.
        """
        print_message(self.side_by_side_ui, "display")

    display = get_qplan_tree
