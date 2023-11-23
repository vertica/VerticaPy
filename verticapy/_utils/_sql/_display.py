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
import shutil
from typing import Optional

import verticapy._config.config as conf
from verticapy._utils._sql._format import clean_query, indent_vpy_sql

if conf.get_import_success("IPython"):
    from IPython.display import HTML, display


def print_query(query: str, title: Optional[str] = None) -> None:
    """
    Displays the input query.
    """
    screen_columns = shutil.get_terminal_size().columns
    query_print = clean_query(query)
    query_print = indent_vpy_sql(query)
    if conf.get_import_success("IPython"):
        display(HTML(f"<h4>{title}</h4>"))
        query_print = query_print.replace("\n", " <br>").replace("  ", " &emsp; ")
        display(HTML(query_print))
    else:
        print(f"$ {title} $\n")
        print(query_print)
        print("-" * int(screen_columns) + "\n")


def print_time(elapsed_time: float) -> None:
    """
    Displays the input time.
    """
    screen_columns = shutil.get_terminal_size().columns
    if conf.get_import_success("IPython"):
        display(HTML(f"<div><b>Execution: </b> {round(elapsed_time, 3)}s</div>"))
    else:
        print(f"Execution: {round(elapsed_time, 3)}s")
        print("-" * int(screen_columns) + "\n")
