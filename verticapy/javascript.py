# (c) Copyright [2018-2022] Micro Focus or one of its affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# |_     |~) _  _| _  /~\    _ |.
# |_)\/  |_)(_|(_||   \_/|_|(_|||
#    /
#              ____________       ______
#             / __        `\     /     /
#            |  \/         /    /     /
#            |______      /    /     /
#                   |____/    /     /
#          _____________     /     /
#          \           /    /     /
#           \         /    /     /
#            \_______/    /     /
#             ______     /     /
#             \    /    /     /
#              \  /    /     /
#               \/    /     /
#                    /     /
#                   /     /
#                   \    /
#                    \  /
#                     \/
#                    _
# \  / _  __|_. _ _ |_)
#  \/ (/_|  | |(_(_|| \/
#                     /
# VerticaPy is a Python library with scikit-like functionality for conducting
# data science projects on data stored in Vertica, taking advantage Vertica’s
# speed and built-in analytics and machine learning features. It supports the
# entire data science life cycle, uses a ‘pipeline’ mechanism to sequentialize
# data transformation operations, and offers beautiful graphical options.
#
# VerticaPy aims to do all of the above. The idea is simple: instead of moving
# data around for processing, VerticaPy brings the logic to the data.
#
#
# Modules
#
# Standard Python Modules
import numpy as np
import os, uuid, json
from json import JSONEncoder
import datetime

# VerticaPy Modules
import verticapy
from verticapy.toolbox import get_category_from_vertica_type

#
#
# Functions to use to create an nteractive table.
#
# ---#
def _table_header(
    head: list, table_id, style, classes, dtype: dict = {},
):
    """This function returns the HTML table header. Rows are not included."""

    head = beautiful_header(head, dtype=dtype,)
    thead = "<thead>"
    thead += "<tr>"
    thead_style = "border: 1px solid #AAAAAA;"
    thead_style_first = "border: 1px solid #AAAAAA; min-width: 95px; max-width: 95px;"
    for i in range(0, len(head)):
        if i == 0:
            logo = f'<div style="padding-left: 15px;">{verticapy.gen_verticapy_logo_html(size="45px")}</div>'
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
    return f"""<div class="container"><table id="{table_id}" class="{classes}" {style}>{thead}<tbody>{tbody}</tbody></table></div>"""


# ---#
def replace_value(template, pattern, value, count=1):
    """Set the given pattern to the desired value in the template,
    after making sure that the pattern is found exactly once."""
    assert isinstance(template, str)
    assert template.count(pattern) == count
    return template.replace(pattern, value)


# ---#
def clean_data(data):
    """Clean the data to improve the html display"""

    for i in range(0, len(data)):
        for j in range(0, len(data[0])):
            if j == 0:
                data[i][j] = f"<b>{data[i][j]}</b>"
                continue
            else:
                val = data[i][j]
                if isinstance(val, bool) is False and val != None:
                    data[i][j] = (
                        '<div style="background-color: transparent; '
                        "border: none; text-align: center; width: 100%;"
                        'scrollbar-width: none; overflow-x: scroll; white-space: nowrap;">'
                        "{0}</div>"
                    ).format(val)
                    continue

                if isinstance(val, bool):
                    val = (
                        "<center>&#9989;</center>"
                        if (val)
                        else "<center>&#10060;</center>"
                    )
                    data[i][j] = val
                    continue
                if val == None:
                    data[i][j] = "[null]"

    return data


# ---#
def datatables_repr(
    data_columns, repeat_first_column: bool = False, offset: int = 0, dtype: dict = {},
):
    """Return the HTML/javascript representation of the table"""

    if not (repeat_first_column):
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
    table_header = _table_header(columns, tableId, style, classes, dtype=dtype,)
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


# ---#
def beautiful_header(
    header, dtype: dict = {}, percent: dict = {},
):
    """Transform the header columns according to the type"""

    n = len(header)
    for i in range(1, n):
        val = header[i]
        type_val, category, missing_values = "", "", ""
        if val in dtype:
            if dtype[val] != "undefined":
                type_val = dtype[val].capitalize()
                category = get_category_from_vertica_type(type_val)
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
                'font-size: 0.95em;">{0}</div>'
            ).format(dtype[val].capitalize())
        else:
            ctype = '<div style="color: #FE5016; margin-top: 6px; font-size: 0.95em;"></div>'
        if val in percent:
            per = int(float(percent[val]))
            try:
                if per == 100:
                    diff = 36
                elif per > 10:
                    diff = 30
                else:
                    diff = 24
            except:
                pass
            missing_values = (
                '<div style="float: right; margin-top: 6px;">{0}%</div><div '
                'style="width: calc(100% - {1}px); height: 8px; margin-top: '
                '10px; border: 1px solid black;"><div style="width: {0}%; '
                'height: 6px; background-color: orange;"></div></div>'
            ).format(per, diff)
        val = "{}<b>{}</b>{}{}".format(category, val, ctype, missing_values)
        header[i] = f'<div style="padding-left: 15px;">{val}</div>'
    return header


# ---#
def read_package_file(*path):
    """Return the content of a file from the itables package"""
    with open(find_package_file(*path), encoding="utf-8") as fp:
        return fp.read()


# ---#
def find_package_file(*path):
    """Return the full path to a file from the itables package"""
    current_path = os.path.dirname(__file__)
    return os.path.join(current_path, *path)


# subclass JSONEncoder
class DateTimeEncoder(JSONEncoder):
    # Override the default method
    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
