"""
(c)  Copyright  [2018-2023]  OpenText  or one of its
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
import importlib
from typing import Any, Callable, Literal, Optional

from verticapy._typing import NoneType

from verticapy._config.validators import (
    bool_validator,
    in_validator,
    optional_positive_int_validator,
    str_validator,
    st_positive_int_validator,
)
from verticapy.errors import OptionError


def get_import_success(module: str) -> bool:
    """
    Confirms whether a module was successfully
    imported.
    """
    return not isinstance(importlib.util.find_spec(module), NoneType)


class Option:
    key: str
    val: Any
    defval: Any
    doc: Optional[str] = None
    validator: Callable[[Any], Literal[True]]
    map_: dict[str, list]

    def __init__(
        self,
        key: str,
        defval: Any,
        doc: str,
        validator: Callable[[Any], Literal[True]],
        map_: Optional[dict[str, list]] = None,
    ) -> None:
        self.key = key
        self.val = defval
        self.defval = defval
        self.doc = doc
        self.validator = validator
        self.map_ = copy.deepcopy(map_)


_all_options: dict[str, Option] = {}


def get_option(key: str) -> Any:
    """
    Returns the value of a specified option.
    """
    return _all_options[key].val


def register_option(op: Option) -> None:
    _all_options[op.key] = op


def set_option(key: str, value: Any = None) -> None:
    """
    Sets VerticaPy options.

    Parameters
    ----------
    key: str
        Option to set.
    cache: bool
        If set to True, vDataFrames save the
        computed aggregations in-memory.
    colors: list
        List of colors used to draw the graphics.
    color_style: str
        Style used to color the graphics, one of the
        following:
        "rgb", "sunset", "retro", "shimbg", "swamp",
        "med", "orchid", "magenta", "orange",
        "vintage", "vivid", "berries", "refreshing",
        "summer", "tropical", "india", "default".
    count_on: bool
        If set to True, the total number of rows in
        vDataFrames and TableSamples is computed and
        displayed in the footer (if footer_on is True).
    footer_on: bool
        If set to True, vDataFrames and TableSamples
        show a footer that includes information about
        the displayed rows and columns.
    interactive: bool
        If set to True, VerticaPy outputs are displayed
        in interactive tables.
    max_columns: int
        Maximum number of columns to display. If the
        specified value is invalid, max_columns is
        not changed.
    max_rows: int
        Maximum number of rows to display. If the
        specified value is invalid, max_row is not
        changed.
    mode: str
        Display mode for VerticaPy outputs, either:
            full  : VerticaPy regular display mode.
            light : Minimalist display mode.
    percent_bar: bool
        If set to True, the percent of non-missing
        values is displayed.
    print_info: bool
        If set to True, information is printed each
        time the vDataFrame is modified.
    random_state: int
        Integer used to seed random number generation
        in VerticaPy.
    save_query_profile: bool
        If set to True, all function calls are stored in
        the query profile table. This makes it possible
        to differentiate the VerticaPy logs from the
        Vertica logs. If set to False, this functionality
        is deactivated.
    sql_on: bool
        If set to True, displays all SQL queries.
    temp_schema: str
        Specifies the temporary schema that certain
        methods/functions use to create intermediate
        objects, if needed.
    time_on: bool
        If set to True, displays the elasped time for
        all SQL queries.
    tqdm: bool
        If set to True, a loading bar is displayed when
        using iterative functions.
    value: object, optional
        New value of the option.
    """
    if key in _all_options:
        op = _all_options[key]
        op.validator(value)
        if isinstance(op.map_, dict) and isinstance(value, str):
            op.val = op.map_[value]
        elif isinstance(value, NoneType):
            op.val = op.defval
        else:
            op.val = value

    else:
        raise OptionError(f"Option '{key}' does not exist.")


register_option(Option("cache", True, "", bool_validator))
register_option(Option("interactive", False, "", bool_validator))
register_option(Option("count_on", False, "", bool_validator))
register_option(Option("footer_on", True, "", bool_validator))
register_option(Option("max_columns", 50, "", st_positive_int_validator))
register_option(Option("max_rows", 100, "", st_positive_int_validator))
register_option(Option("mode", "full", "", in_validator(["full", "light"])))
register_option(Option("percent_bar", False, "", bool_validator))
register_option(Option("print_info", True, "", bool_validator))
register_option(Option("save_query_profile", True, "", bool_validator))
register_option(Option("sql_on", False, "", bool_validator))
register_option(Option("random_state", None, "", optional_positive_int_validator))
register_option(Option("temp_schema", "public", "", str_validator))
register_option(Option("time_on", False, "", bool_validator))
register_option(Option("tqdm", True, "", bool_validator))
register_option(
    Option(
        "plotting_lib",
        "matplotlib",
        "",
        in_validator(["highcharts", "matplotlib", "plotly"]),
    )
)
