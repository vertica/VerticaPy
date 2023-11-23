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
import datetime
import json
import os
import uuid
from typing import Any, Optional, TextIO

import numpy as np

from verticapy._typing import ArrayLike, NoneType
from verticapy._utils._logo import verticapy_logo_html
from verticapy._utils._sql._cast import to_category
from verticapy._utils._sql._format import format_type

"""
Utils Functions.
"""


def find_package_file(*path) -> str:
    """
    Returns the full path to a file from the itables
    package.
    """
    current_path = os.path.dirname(__file__)
    return os.path.join(current_path, *path)


def read_package_file(*path) -> TextIO:
    """
    Returns the content of a file from the itables
    package.
    """
    with open(find_package_file(*path), encoding="utf-8") as fp:
        return fp.read()


def replace_value(template: str, pattern: str, value: Any, count: int = 1) -> str:
    """
    Sets the given pattern to the desired value in the template,
    after making sure that the pattern is found exactly once.
    """
    assert isinstance(template, str)
    assert template.count(pattern) == count
    return template.replace(pattern, value)


"""
Utils Classes.
"""


class DateTimeEncoder(json.JSONEncoder):
    """
    Subclass JSONEncoder
    """

    # Override the default method
    def default(self, obj: Any) -> Any:
        """
        Overrides the default method.
        """
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()


"""
Main Functions.
"""


def beautiful_header(
    header: str,
    dtype: Optional[dict] = None,
    percent: Optional[dict] = None,
) -> str:
    """
    Transforms the header columns according to the type.
    """
    dtype, percent = format_type(dtype, percent, dtype=dict)
    n = len(header)
    for i in range(1, n):
        val = header[i]
        type_val, category, missing_values = "", "", ""
        if val in dtype:
            if dtype[val] != "undefined":
                type_val = dtype[val].capitalize()
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
                    category = (
                        '<div style="margin-bottom: 6px; color: #0073E7;">010</div>'
                    )
                elif category in ("int", "float", "binary", "uuid"):
                    category = (
                        '<div style="margin-bottom: 6px; color: #19A26B;">123</div>'
                    )
                elif category == "text":
                    category = '<div style="margin-bottom: 6px;">Abc</div>'
                elif category == "date":
                    category = '<div style="margin-bottom: 6px;">&#128197;</div>'
            else:
                category = '<div style="margin-bottom: 6px;"></div>'
        if type_val != "":
            ctype = (
                '<div style="color: #FE5016; margin-top: 6px; '
                f'font-size: 0.95em;">{dtype[val].capitalize()}</div>'
            )
        else:
            ctype = '<div style="color: #FE5016; margin-top: 6px; font-size: 0.95em;"></div>'
        if val in percent:
            per = int(float(percent[val]))
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
        val = f"{category}<b>{val}</b>{ctype}{missing_values}"
        header[i] = f'<div style="padding-left: 15px;">{val}</div>'
    return header


def _table_header(
    head: list,
    table_id: str,
    style: str,
    classes: str,
    dtype: Optional[dict] = None,
) -> str:
    """
    Returns the HTML table header. Rows are not included.
    """
    dtype = format_type(dtype, dtype=dict)
    head = beautiful_header(
        head,
        dtype=dtype,
    )
    thead = "<thead>"
    thead += "<tr>"
    thead_style = "border: 1px solid #AAAAAA;"
    thead_style_first = "border: 1px solid #AAAAAA; min-width: 95px; max-width: 95px;"
    for i in range(0, len(head)):
        if i == 0:
            logo = f'<div style="padding-left: 15px;">{verticapy_logo_html(size="45px")}</div>'
            thead += f'<td style="{thead_style_first}">{logo}</td>'
        else:
            thead += f'<td style="{thead_style}">{head[i]}</td>'
        # thead += f'<th style="border: 1px solid #AAAAAA;">{elmt}</th>'
    thead += "</tr>"
    thead += "<thead>"
    loading = "<td>Loading...</td>"
    tbody = f"<tr>{loading}</tr>"
    if style:
        style = f'style="{style}"'
    else:
        style = ""
    return f"""
        <div class="container">
            <table id="{table_id}" class="{classes}" {style}>{thead}
                <tbody>{tbody}</tbody>
            </table>
        </div>"""


def clean_data(data: ArrayLike) -> ArrayLike:
    """
    Cleans the data to improve the html display.
    """
    for i in range(0, len(data)):
        for j in range(0, len(data[0])):
            if j == 0:
                data[i][j] = f"<b>{data[i][j]}</b>"
                continue
            else:
                val = data[i][j]
                if isinstance(val, bool) is False and not isinstance(val, NoneType):
                    data[i][
                        j
                    ] = f"""
                        <div style="background-color: transparent;
                             border: none; text-align: center; 
                             width: 100%; scrollbar-width: none; 
                             overflow-x: scroll; white-space: nowrap;">
                            {val}
                        </div>"""
                    continue

                if isinstance(val, bool):
                    val = (
                        "<center>&#9989;</center>"
                        if (val)
                        else "<center>&#10060;</center>"
                    )
                    data[i][j] = val
                    continue
                if isinstance(val, NoneType):
                    data[i][j] = "[null]"
    return data


def datatables_repr(
    data_columns: ArrayLike,
    repeat_first_column: bool = False,
    offset: int = 0,
    dtype: Optional[dict] = None,
) -> str:
    """
    Returns the HTML/javascript representation of the table.
    """
    dtype = format_type(dtype, dtype=dict)
    if not repeat_first_column:
        index_column = list(range(1 + offset, len(data_columns[0]) + offset))
        data_columns = [[""] + [i for i in index_column]] + data_columns
    columns = []
    data = []
    for dc in data_columns:
        columns.append(dc[0])
        data.append(dc[1:])
    data = np.array(data).T.tolist()

    data = clean_data(data)
    output = read_package_file("html/html_template_connected.html")
    style = "width:100%"
    classes = ["hover", "row-border"]
    if isinstance(classes, list):
        classes = " ".join(classes)
    tableId = str(uuid.uuid4())
    table_header = _table_header(
        columns,
        tableId,
        style,
        classes,
        dtype=dtype,
    )
    output = replace_value(
        output,
        '<table id="table_id"><thead><tr><th>A</th></tr></thead></table>',
        table_header,
    )
    output = replace_value(output, "#table_id", f"#{tableId}", 2)
    output = replace_value(
        output,
        "<style></style>",
        f"""<style>
        {read_package_file("html/style.css")}
        </style>""",
    )
    dt_data = json.dumps(data, cls=DateTimeEncoder)
    output = replace_value(output, "const data = [];", f"const data = {dt_data};")

    return output
