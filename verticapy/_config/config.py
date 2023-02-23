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
import warnings
from typing import Any, Literal, Union, Optional, overload

GEOPANDAS_ON: bool
try:
    from geopandas import GeoDataFrame
    from shapely import wkt

    GEOPANDAS_ON = True
except:
    GEOPANDAS_ON = False

GRAPHVIZ_ON: bool
try:
    import graphviz

    GRAPHVIZ_ON = True
except:
    GRAPHVIZ_ON = False

ISNOTEBOOK: bool = False
try:
    import IPython

    shell = get_ipython().__class__.__name__
    if shell == "ZMQInteractiveShell":
        ISNOTEBOOK = True  # Jupyter notebook or qtconsole
except:
    pass

PARSER_IMPORT: bool
try:
    from dateutil.parser import parse

    PARSER_IMPORT = True
except:
    PARSER_IMPORT = False

from verticapy.errors import ParameterError

COLORS_OPTIONS: dict[str, list] = {
    "rgb": ["red", "green", "blue", "orange", "yellow", "gray"],
    "sunset": ["#36688D", "#F3CD05", "#F49F05", "#F18904", "#BDA589"],
    "retro": ["#A7414A", "#282726", "#6A8A82", "#A37C27", "#563838"],
    "shimbg": ["#0444BF", "#0584F2", "#0AAFF1", "#EDF259", "#A79674"],
    "swamp": ["#6465A5", "#6975A6", "#F3E96B", "#F28A30", "#F05837"],
    "med": ["#ABA6BF", "#595775", "#583E2E", "#F1E0D6", "#BF9887"],
    "orchid": ["#192E5B", "#1D65A6", "#72A2C0", "#00743F", "#F2A104"],
    "magenta": ["#DAA2DA", "#DBB4DA", "#DE8CF0", "#BED905", "#93A806"],
    "orange": ["#A3586D", "#5C4A72", "#F3B05A", "#F4874B", "#F46A4E"],
    "vintage": ["#80ADD7", "#0ABDA0", "#EBF2EA", "#D4DCA9", "#BF9D7A"],
    "vivid": ["#C0334D", "#D6618F", "#F3D4A0", "#F1931B", "#8F715B"],
    "berries": ["#BB1924", "#EE6C81", "#F092A5", "#777CA8", "#AFBADC"],
    "refreshing": ["#003D73", "#0878A4", "#1ECFD6", "#EDD170", "#C05640"],
    "summer": ["#728CA3", "#73C0F4", "#E6EFF3", "#F3E4C6", "#8F4F06"],
    "tropical": ["#7B8937", "#6B7436", "#F4D9C1", "#D72F01", "#F09E8C"],
    "india": ["#F1445B", "#65734B", "#94A453", "#D9C3B1", "#F03625"],
    "default": ["#FE5016", "#263133", "#0073E7", "#FDE159", "#33C180", "#FF454F"],
}

_options: dict = {
    "cache": True,
    "colors": None,
    "interactive": False,
    "count_on": False,
    "footer_on": True,
    "max_columns": 50,
    "max_rows": 100,
    "mode": None,
    "overwrite_model": True,
    "percent_bar": None,
    "print_info": True,
    "save_query_profile": True,
    "sql_on": False,
    "random_state": None,
    "temp_schema": "public",
    "time_on": False,
    "tqdm": True,
}


def init_interactive_mode(all_interactive: bool = False) -> None:
    """Activate the datatables representation for all the vDataFrames."""
    set_option("interactive", all_interactive)
    return None


def get_option(
    option: Literal[
        "cache",
        "colors",
        "interactive",
        "count_on",
        "footer_on",
        "max_columns",
        "max_rows",
        "mode",
        "overwrite_model",
        "percent_bar",
        "print_info",
        "save_query_profile",
        "sql_on",
        "temp_schema",
        "time_on",
        "tqdm",
    ],
) -> Any:
    return _options[option]


@overload
def set_option(
    option: Literal["color_style"], value: Literal[tuple(COLORS_OPTIONS)]
) -> None:
    ...


@overload
def set_option(option: Literal["mode"], value: Literal["light", "full"]) -> None:
    ...


def set_option(
    option: Literal[
        "cache",
        "colors",
        "color_style",
        "interactive",
        "count_on",
        "footer_on",
        "max_columns",
        "max_rows",
        "mode",
        "overwrite_model",
        "percent_bar",
        "print_info",
        "random_state",
        "save_query_profile",
        "sql_on",
        "temp_schema",
        "time_on",
        "tqdm",
    ],
    value: Any = None,
) -> None:
    """
    Sets VerticaPy options.

    Parameters
    ----------
    option: str
        Option to use.
        cache              : bool
            If set to True, the vDataFrame will save in memory the computed
            aggregations.
        colors             : list
            List of the colors used to draw the graphics.
        color_style        : str
            Style used to color the graphics, one of the following:
            "rgb", "sunset", "retro", "shimbg", "swamp", "med", "orchid", 
            "magenta", "orange", "vintage", "vivid", "berries", "refreshing", 
            "summer", "tropical", "india", "default".
        count_on           : bool
            If set to True, the total number of rows in vDataFrames and TableSamples is  
            computed and displayed in the footer (if footer_on is True).
        footer_on          : bool
            If set to True, vDataFrames and TableSamples show a footer that includes information 
            about the displayed rows and columns.
        interactive        : bool
            If set to True, verticaPy outputs will be displayed on interactive tables. 
        max_columns        : int
            Maximum number of columns to display. If the parameter is incorrect, 
            nothing is changed.
        max_rows           : int
            Maximum number of rows to display. If the parameter is incorrect, 
            nothing is changed.
        mode               : str
            How to display VerticaPy outputs.
                full  : VerticaPy regular display mode.
                light : Minimalist display mode.
        overwrite_model: bool
            If set to True and you try to train a model with an existing name. 
            It will be automatically overwritten.
        percent_bar        : bool
            If set to True, it displays the percent of non-missing values.
        print_info         : bool
            If set to True, information will be printed each time the vDataFrame 
            is modified.
        random_state       : int
            Integer used to seed the random number generation in VerticaPy.
        save_query_profile : str / list / bool
            If set to "all" or True, all function calls are stored in the query 
            profile table. This makes it possible to differentiate the VerticaPy 
            logs from the Vertica logs.
            You can also provide a list of specific methods to store. For example: 
            if you specify ["corr", "describe"], only the logs associated with 
            those two methods are stored. 
            If set to False, this functionality is deactivated.
        sql_on             : bool
            If set to True, displays all the SQL queries.
        temp_schema        : str
            Specifies the temporary schema that certain methods/functions use to 
            create intermediate objects, if needed. 
        time_on            : bool
            If set to True, displays all the SQL queries elapsed time.
        tqdm               : bool
            If set to True, a loading bar is displayed when using iterative 
            functions.
    value: object, optional
        New value of option.
    """
    wrong_value = False
    if option == "colors":
        if isinstance(value, list):
            _options["colors"] = [str(elem) for elem in value]
        else:
            wrong_value = True
    elif option == "color_style":
        if value == None:
            value = "default"
        if value in COLORS_OPTIONS:
            _options["colors"] = COLORS_OPTIONS[value]
        else:
            wrong_value = True
    elif option == "max_columns":
        if isinstance(value, int) and value > 0:
            _options["max_columns"] = int(value)
        else:
            wrong_value = True
    elif option == "max_rows":
        if isinstance(value, int) and value >= 0:
            _options["max_rows"] = int(value)
        else:
            wrong_value = True
    elif option == "mode":
        if value in ["light", "full"]:
            _options["mode"] = value
        else:
            wrong_value = True
    elif option == "random_state":
        if isinstance(value, int) and (value < 0):
            raise ParameterError("Random State Value must be positive.")
        if isinstance(value, int):
            _options["random_state"] = value
        elif value == None:
            _options["random_state"] = None
        else:
            wrong_value = True
    elif option in (
        "print_info",
        "sql_on",
        "time_on",
        "count_on",
        "cache",
        "footer_on",
        "tqdm",
        "overwrite_model",
        "percent_bar",
        "interactive",
    ):
        if value in (True, False, None):
            _options[option] = value
        else:
            wrong_value = True
    elif option == "save_query_profile":
        if value == "all":
            value = True
        elif not (isinstance(value, (bool, list))):
            wrong_value = True
        if not (wrong_value):
            _options[option] = value
    elif option == "temp_schema":
        if isinstance(value, str):
            _options["temp_schema"] = value
        else:
            wrong_value = True
    else:
        raise ParameterError(f"Option '{option}' does not exist.")
    if wrong_value:
        warning_message = "The parameter value is incorrect. Nothing was changed."
        warnings.warn(warning_message, Warning)
    return None
