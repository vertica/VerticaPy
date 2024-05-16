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
import html
import shutil
from typing import Optional

import verticapy._config.config as conf
from verticapy._typing import NoneType
from verticapy._utils._sql._cast import to_category
from verticapy._utils._sql._format import format_type
from verticapy._utils._logo import verticapy_logo_html


def print_table(
    data_columns,
    is_finished: bool = True,
    offset: int = 0,
    repeat_first_column: bool = False,
    first_element: Optional[str] = None,
    return_html: bool = False,
    dtype: Optional[dict] = None,
    percent: Optional[dict] = None,
) -> str:
    """
    Returns the HTML code or string used to display the final
    relation.
    """

    # Colors Definition & options
    theme = conf.get_option("theme")

    if theme == "light":
        bgcolor = "#FFFFFF"
        bgcolor_null = "#EEEEEE"
        fontcolor_null = "#999999"
        bgcolor_other = "#FAFAFA"
        border_color_1 = "#AAAAAA"
        border_color_2 = "#DDDDDD"
        fontcolor = "#000000"
        fontcolor_abc = "#000000"
        fontcolor_index_j = "#000000"
        fontcolor_index_i = "#000000"
    elif theme == "dark":
        bgcolor = "#000000"
        bgcolor_null = "#222222"
        fontcolor_null = "#999999"
        bgcolor_other = "#18181A"
        border_color_1 = "#555555"
        border_color_2 = "#333333"
        fontcolor = "#FFFFCC"
        fontcolor_abc = "#959DAD"
        fontcolor_index_j = "#1A6AFF"
        fontcolor_index_i = "#FFFFFF"
    elif theme == "sphinx":
        bgcolor = "var(--color-announcement-background)"
        bgcolor_null = "var(--color-background-hover)"
        fontcolor_null = "#999999"
        bgcolor_other = "var(--color-admonition-background)"
        border_color_1 = "#888888"
        border_color_2 = "#555555"
        fontcolor = "var(--color-content-foreground)"
        fontcolor_abc = "#959DAD"
        fontcolor_index_j = "#1A6AFF"
        fontcolor_index_i = "var(--color-announcement-text)"
    else:
        raise ValueError("Unrecognized 'theme'.")

    maxwidth = conf.get_option("max_cellwidth")
    maxwidth = max(280, maxwidth)
    maxheight = conf.get_option("max_tableheight")
    maxheight = max(300, maxheight)

    # Main Function
    dtype, percent = format_type(dtype, percent, dtype=dict)
    if not return_html:
        data_columns_rep = [] + data_columns
        if repeat_first_column:
            del data_columns_rep[0]
            columns_ljust_val = min(
                len(max([str(item) for item in data_columns[0]], key=len)) + 4, 40
            )
        else:
            columns_ljust_val = len(str(len(data_columns[0]))) + 2
        screen_columns = shutil.get_terminal_size().columns
        formatted_text = ""
        rjust_val = []
        for idx in range(0, len(data_columns_rep)):
            rjust_val += [
                min(
                    len(max([str(item) for item in data_columns_rep[idx]], key=len))
                    + 2,
                    40,
                )
            ]
        total_column_len = len(data_columns_rep[0])
        while rjust_val != []:
            columns_to_print = [data_columns_rep[0]]
            columns_rjust_val = [rjust_val[0]]
            max_screen_size = int(screen_columns) - 14 - int(rjust_val[0])
            del data_columns_rep[0]
            del rjust_val[0]
            while (max_screen_size > 0) and (rjust_val != []):
                columns_to_print += [data_columns_rep[0]]
                columns_rjust_val += [rjust_val[0]]
                max_screen_size = max_screen_size - 7 - int(rjust_val[0])
                del data_columns_rep[0]
                del rjust_val[0]
            if repeat_first_column:
                columns_to_print = [data_columns[0]] + columns_to_print
            else:
                columns_to_print = [
                    [i + offset for i in range(0, total_column_len)]
                ] + columns_to_print
            columns_to_print[0][0] = first_element
            columns_rjust_val = [columns_ljust_val] + columns_rjust_val
            column_count = len(columns_to_print)
            for i in range(0, total_column_len):
                for k in range(0, column_count):
                    val = columns_to_print[k][i]
                    if len(str(val)) > 40:
                        val = str(val)[0:37] + "..."
                    if k == 0:
                        formatted_text += str(val).ljust(columns_rjust_val[k])
                    else:
                        formatted_text += str(val).rjust(columns_rjust_val[k]) + "  "
                if rjust_val != []:
                    formatted_text += " \\\\"
                formatted_text += "\n"
            if not is_finished and (i == total_column_len - 1):
                for k in range(0, column_count):
                    if k == 0:
                        formatted_text += "...".ljust(columns_rjust_val[k])
                    else:
                        formatted_text += "...".rjust(columns_rjust_val[k]) + "  "
                if rjust_val != []:
                    formatted_text += " \\\\"
                formatted_text += "\n"
        return formatted_text
    else:
        if not repeat_first_column:
            data_columns = [
                [""] + list(range(1 + offset, len(data_columns[0]) + offset))
            ] + data_columns
        m, n = len(data_columns), len(data_columns[0])
        cell_width = []
        for row in data_columns:
            cell_width += [
                min(5 * max([len(str(item)) for item in row]) + 80, maxwidth)
            ]
        html_table = '<div class="verticapy_table"><table>'
        for i in range(n):
            if i == 0:
                html_table += '<thead style="display: table; ">'
            if i == 1 and n > 0:
                html_table += (
                    '<tbody style="display: block; max-height: '
                    f'{maxheight}px; overflow-y: scroll;">'
                )
            html_table += "<tr>"
            for j in range(m):
                val = data_columns[j][i]
                if isinstance(val, str):
                    val = html.escape(val)
                if isinstance(val, NoneType):
                    val = "[null]"
                    color = fontcolor_null
                else:
                    if isinstance(val, bool) and (
                        conf.get_option("mode") in ("full", None)
                    ):
                        val = (
                            "<center>&#9989;</center>"
                            if (val)
                            else "<center>&#10060;</center>"
                        )
                    if j == 0:
                        color = fontcolor_index_j
                    elif i == 0:
                        color = fontcolor_index_i
                    else:
                        color = fontcolor
                html_table += '<td style="background-color: '
                if (
                    (j == 0)
                    or (i == 0)
                    or (conf.get_option("mode") not in ("full", None))
                ):
                    html_table += f" {bgcolor}; "
                elif val == "[null]":
                    html_table += f" {bgcolor_null}; "
                else:
                    html_table += f" {bgcolor_other}; "
                html_table += f"color: {color}; white-space:nowrap; "
                if conf.get_option("mode") in ("full", None):
                    if (j == 0) or (i == 0):
                        html_table += f"border: 1px solid {border_color_1}; "
                    else:
                        html_table += f"border-top: 1px solid {border_color_2}; "
                        if ((j == m - 1) and (i == n - 1)) or (j == m - 1):
                            html_table += f"border-right: 1px solid {border_color_1}; "
                        else:
                            html_table += f"border-right: 1px solid {border_color_2}; "
                        if ((j == m - 1) and (i == n - 1)) or (i == n - 1):
                            html_table += f"border-bottom: 1px solid {border_color_1}; "
                        else:
                            html_table += f"border-bottom: 1px solid {border_color_2}; "
                if i == 0:
                    html_table += "height: 30px; "
                if (j == 0) or (cell_width[j] < 120):
                    html_table += "text-align: center; "
                else:
                    html_table += "text-align: center; "
                html_table += (
                    f"min-width: {cell_width[j]}px; " f'max-width: {cell_width[j]}px;"'
                )
                if conf.get_option("insert_comma_numbers"):
                    try:
                        float(val)
                        val = "{:,}".format(val)
                    except:
                        pass
                if (j == 0) or (i == 0):
                    if j != 0:
                        type_val, category, missing_values = "", "", ""
                        if data_columns[j][0] in dtype and (
                            conf.get_option("mode") in ("full", None)
                        ):
                            if dtype[data_columns[j][0]] != "undefined":
                                type_val = dtype[data_columns[j][0]].capitalize()
                                category = to_category(type_val)
                                if (category == "spatial") or (
                                    (
                                        "lat" in val.lower().split(" ")
                                        or "latitude" in val.lower().split(" ")
                                        or "lon" in val.lower().split(" ")
                                        or "longitude" in val.lower().split(" ")
                                    )
                                    and category == "float"
                                ):
                                    category = '<div style="margin-bottom: 6px;">&#x1f30e;</div>'
                                elif type_val.lower() == "boolean":
                                    category = '<div style="margin-bottom: 6px; color: #0073E7;">010</div>'
                                elif category in ("int", "float", "binary", "uuid"):
                                    category = '<div style="margin-bottom: 6px; color: #19A26B;">123</div>'
                                elif category == "text":
                                    category = f'<div style="margin-bottom: 6px; color: {fontcolor_abc}">Abc</div>'
                                elif category in ("complex", "vmap"):
                                    category = '<div style="margin-bottom: 6px;">&#128736;</div>'
                                elif category == "date":
                                    category = '<div style="margin-bottom: 6px;">&#128197;</div>'
                            else:
                                category = '<div style="margin-bottom: 6px;"></div>'
                        if type_val != "":
                            ctype = (
                                '<div style="overflow-y: scroll; color: #FE5016; '
                                f'margin-top: 6px; font-size: 0.95em;">{type_val}</div>'
                            )
                        else:
                            ctype = '<div style="color: #FE5016; margin-top: 6px; font-size: 0.95em;"></div>'
                        if data_columns[j][0] in percent:
                            per = int(float(percent[data_columns[j][0]]))
                            if per == 100:
                                diff = 36
                            elif per > 10:
                                diff = 30
                            else:
                                diff = 24
                            missing_values = (
                                f'<div style="float: right; margin-top: 6px;">{per}%</div><div '
                                f'style="width: calc(100% - {diff}px); height: 8px; margin-top: '
                                f'10px; border: 1px solid black;"><div style="width: {per}%; '
                                'height: 6px; background-color: orange;"></div></div>'
                            )
                    else:
                        ctype, missing_values, category = "", "", ""
                    if (i == 0) and (j == 0):
                        if dtype and (conf.get_option("mode") in ("full", None)):
                            val = verticapy_logo_html(size="45px")
                        else:
                            val = ""
                    elif cell_width[j] > 240:
                        val = (
                            f'<input style="background-color: {bgcolor}; font-weight: bold;'
                            f"color: {fontcolor}; border: none; text-align: center; width: "
                            f'{cell_width[j] - 10}px;" type="text" value="{val}" readonly>'
                        )
                    html_table += f">{category}<b>{val}</b>{ctype}{missing_values}</td>"
                elif cell_width[j] > 240:
                    background = bgcolor_null if val == "[null]" else bgcolor_other
                    if conf.get_option("mode") not in ("full", None):
                        background = f"{bgcolor}"
                    html_table += (
                        f'><input style="background-color: {bgcolor}; border: none; '
                        f'color: {fontcolor}; text-align: center; width: {cell_width[j] - 10}px;" '
                        f'type="text" value="{val}" readonly></td>'
                    )
                else:
                    html_table += f">{val}</td>"
            html_table += "</tr>"
            if i == 0:
                html_table += "</thead>"
            if i == n - 1 and n > 0:
                html_table += "</tbody>"
        html_table += "</table></div>"
        return html_table
