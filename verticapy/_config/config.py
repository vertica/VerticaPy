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
from typing import Any, Callable, Literal

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

from verticapy._config.validators import (
    bool_validator,
    color_validator,
    in_validator,
    optional_bool_validator,
    optional_positive_int_validator,
    str_validator,
    st_positive_int_validator,
)
from verticapy.errors import OptionError

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


class Option:
    key: str
    val: Any
    defval: Any
    doc: str = ""
    validator: Callable[[Any], Literal[True]]

    def __init__(
        self, key: str, defval: Any, doc: str, validator: Callable[[Any], Literal[True]]
    ) -> None:
        self.key = key
        self.val = defval
        self.defval = defval
        self.doc = doc
        self.validator = validator
        return None


_all_options: dict[str, Option] = {}


def get_option(key: str) -> Any:
    return _all_options[key].val


def register_option(op: Option) -> None:
    _all_options[op.key] = op
    return None


def set_option(key: str, value: Any = None) -> None:
    """
    Sets VerticaPy options.

    Parameters
    ----------
    key: str
        Option to use.
        cache: bool
            If set to True, the vDataFrame will save in memory the computed
            aggregations.
        colors: list
            List of the colors used to draw the graphics.
        color_style: str
            Style used to color the graphics, one of the following:
            "rgb", "sunset", "retro", "shimbg", "swamp", "med", "orchid", 
            "magenta", "orange", "vintage", "vivid", "berries", "refreshing", 
            "summer", "tropical", "india", "default".
        count_on: bool
            If set to True, the total number of rows in vDataFrames and TableSamples is  
            computed and displayed in the footer (if footer_on is True).
        footer_on: bool
            If set to True, vDataFrames and TableSamples show a footer that includes information 
            about the displayed rows and columns.
        interactive: bool
            If set to True, verticaPy outputs will be displayed on interactive tables. 
        max_columns: int
            Maximum number of columns to display. If the parameter is incorrect, 
            nothing is changed.
        max_rows: int
            Maximum number of rows to display. If the parameter is incorrect, 
            nothing is changed.
        mode: str
            How to display VerticaPy outputs.
                full: VerticaPy regular display mode.
                light: Minimalist display mode.
        overwrite_model: bool
            If set to True and you try to train a model with an existing name. 
            It will be automatically overwritten.
        percent_bar: bool
            If set to True, it displays the percent of non-missing values.
        print_info: bool
            If set to True, information will be printed each time the vDataFrame 
            is modified.
        random_state: int
            Integer used to seed the random number generation in VerticaPy.
        save_query_profile: str / list / bool
            If set to "all" or True, all function calls are stored in the query 
            profile table. This makes it possible to differentiate the VerticaPy 
            logs from the Vertica logs.
            You can also provide a list of specific methods to store. For example: 
            if you specify ["corr", "describe"], only the logs associated with 
            those two methods are stored. 
            If set to False, this functionality is deactivated.
        sql_on: bool
            If set to True, displays all the SQL queries.
        temp_schema: str
            Specifies the temporary schema that certain methods/functions use to 
            create intermediate objects, if needed. 
        time_on: bool
            If set to True, displays all the SQL queries elapsed time.
        tqdm: bool
            If set to True, a loading bar is displayed when using iterative 
            functions.
    value: object, optional
        New value of option.
    """
    if key in _all_options:
        op = _all_options[key]
        op.validator(value)
        if value == None:
            op.val = op.defval
        else:
            op.val = value
    else:
        raise OptionError(f"Option '{key}' does not exist.")


register_option(Option("cache", True, "", bool_validator))
register_option(Option("colors", None, "", color_validator))
register_option(Option("interactive", False, "", bool_validator))
register_option(Option("count_on", False, "", bool_validator))
register_option(Option("footer_on", True, "", bool_validator))
register_option(Option("max_columns", 50, "", st_positive_int_validator))
register_option(Option("max_rows", 100, "", st_positive_int_validator))
register_option(Option("mode", "full", "", in_validator(["full", "light"])))
register_option(Option("overwrite_model", True, "", bool_validator))
register_option(Option("percent_bar", False, "", bool_validator))
register_option(Option("print_info", True, "", bool_validator))
register_option(Option("save_query_profile", True, "", bool_validator))
register_option(Option("sql_on", False, "", bool_validator))
register_option(Option("random_state", None, "", optional_positive_int_validator))
register_option(Option("temp_schema", "public", "", str_validator))
register_option(Option("time_on", False, "", bool_validator))
register_option(Option("tqdm", True, "", bool_validator))
