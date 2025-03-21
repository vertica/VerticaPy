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
import copy
import importlib
from typing import Any, Callable, Literal, Optional

from verticapy._typing import NoneType

from verticapy._config.validators import (
    bool_validator,
    in_validator,
    optional_positive_int_validator,
    optional_str_validator,
    str_validator,
    st_positive_int_validator,
)
from verticapy.errors import OptionError


def get_import_success(module: str) -> bool:
    """
    Confirms whether a module was
    successfully imported.

    Parameters
    ----------
    module: str
        Module name.

    Returns
    -------
    bool
        True if the module is installed
        correctly.

    Examples
    --------
    Let's perform two checks: one for an existing
    module and one for a non-existent one.

    .. ipython:: python

        # Importing the function
        from verticapy._config.config import get_import_success

        # Checking an existing module
        get_import_success('verticapy')

        # Checking a fake module
        get_import_success('fake_module_test')

    .. note::

        This function is used to test if optional
        modules are installed to enhance VerticaPy
        capabilities.
    """
    return not isinstance(importlib.util.find_spec(module), NoneType)


class Option:
    """
    New Option.

    Parameters
    ----------
    key: str
        Option name.
    defval: Any
        The default value.
    doc: str
        Option documentation.
    validator: Callable
        Function to check the input in
        case of option change.
    map_: dict, optional
        Dictionary to Map to the Right Value.

    Attributes
    ----------
    The attributes are identical to the input
    parameters.

    Examples
    --------
    The following option is a string that can
    take one of three values: 'A', 'B', or 'C'.

    .. ipython:: python

        # Importing the validator.
        from verticapy._config.validators import in_validator

        # Importing the Option class
        from verticapy._config.config import Option

        # Building the Option
        Option(
            "abc_test", # Option Name
            "A", # Default Value
            "ABC Checker.", # Doc
            in_validator(["A", "B", "C"]), # Validator
        )

    .. note::

        This class is utilized to create consistent
        options that can be used throughout the entire
        API.

    .. seealso::

        | :py:meth:`~verticapy.register_option` :
            Registers the input option.
        | :py:meth:`~verticapy.set_option` :
            Sets VerticaPy options.
    """

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

    Parameters
    ----------
    key: str
        Option name.

    Returns
    -------
    Any
        The selected option.

    Examples
    --------
    Let's get the value of an existing option.

    .. ipython:: python

        # Importing the function
        from verticapy._config.config import get_option

        # Checking the 'plotting_lib' option
        get_option('plotting_lib')

    .. note::

        This function is utilized to get the options
        that can be used throughout the entire API.

    .. seealso::

        | :py:meth:`~verticapy.set_option` :
            Sets VerticaPy options.
    """
    return _all_options[key].val


def register_option(op: Option) -> None:
    """
    Registers the input option.

    Parameters
    ----------
    op: Option
        Option to register.

    Examples
    --------
    The following option is a string that can
    take one of three values: 'A', 'B', or 'C'.

    .. ipython:: python

        # Importing the validator.
        from verticapy._config.validators import in_validator

        # Importing the Option class
        from verticapy._config.config import Option

        # Building the Option
        opt = Option(
            "abc_test", # Option Name
            "A", # Default Value
            "ABC Checker.", # Doc
            in_validator(["A", "B", "C"]), # Validator
        )
        display(opt)

    Now, let's register the option.

    .. ipython:: python

        # Importing the function
        from verticapy._config.config import register_option

        # Registering the option
        register_option(opt)

    .. note::

        This function is utilized to create consistent
        options that can be used throughout the entire
        API.

    .. seealso::

        | :py:meth:`~verticapy.set_option` :
            Sets VerticaPy options.
    """
    _all_options[op.key] = op


def set_option(key: str, value: Any = None) -> None:
    """
    Sets VerticaPy options.

    .. important::

        Some options may not be available in VerticaPy
        1.0.0. To access all available options, please
        use VerticaPy 1.0.1 or a higher version.

    Parameters
    ----------
    key: str
        Option to set, one of the following:

        - cache:
            [bool]
            If set to True, vDataFrames save the
            computed aggregations in-memory.
        - colors:
            [list]
            List of colors used to draw the graphics.
        - color_style:
            [str]
            Style used to color the graphics, one of the
            following:
            "rgb", "sunset", "retro", "shimbg", "swamp",
            "med", "orchid", "magenta", "orange",
            "vintage", "vivid", "berries", "refreshing",
            "summer", "tropical", "india", "default".
        - count_on:
            [bool]
            If set to ``True``, the total number of rows in
            :py:class:`~vDataFrame` and :py:class:`~verticapy.core.tablesample.base.TableSample`
            is computed and displayed in the footer (if
            ``footer_on is True``).
        - footer_on:
            [bool]
            If set to ``True``, :py:class:`~vDataFrame` and
            :py:class:`~verticapy.core.tablesample.base.TableSample` show a footer that includes
            information about the displayed rows and columns.
        - interactive:
            [bool]
            If set to ``True``, VerticaPy outputs are
            displayed in interactive tables.
        - label_separator:
            [str]
            Separator used to separate the query label from
            the ``label_suffix``. The default value is ``__``.
        - label_suffix:
            [str]
            Label suffix to add to VerticaPy's query labels.
            It can be useful to track some specific activities.
            For example: Looking which user runs some specific
            VerticaPy functions. The default value is ``None``.
        - max_columns:
            [int]
            Maximum number of columns to display. If the
            specified value is invalid, ``max_columns`` is
            not changed.
        - max_rows:
            [int]
            Maximum number of rows to display. If the
            specified value is invalid, ``max_row`` is
            not changed.
        - max_cellwidth:
            Maximum width of any VerticaPy table's cell.
            Can not be lesser than 280.
            Default: 280
        - max_tableheight:
            Maximum height of VerticaPy tables.
            Can not be lesser than 300.
            Default: 300
        - mode:
            [str]
            Display mode for VerticaPy outputs, either:

            **full**:
                VerticaPy regular display mode.

            **light**:
                Minimalist display mode.
        - percent_bar:
            [bool]
            If set to ``True``, the percent of non-missing
            values is displayed.
        - plotting_lib:
            [str]
            Plotting library used to draw
            the different graphics.
            One of the following:
            plotly | highcharts | matplotlib
        - print_info:
            [bool]
            If set to ``True``, information is printed each
            time the :py:class:`~vDataFrame` is modified.
        - random_state:
            [int]
            Integer used to seed random number generation
            in VerticaPy.
        - save_query_profile:
            [bool]
            If set to ``True``, all function calls are stored
            in the query profile table. This makes it possible
            to differentiate the VerticaPy logs from the
            Vertica logs. If set to ``False``, this functionality
            is deactivated.
        - sql_on:
            [bool]
            If set to ``True``, displays all SQL queries.
        - temp_schema:
            [str]
            Specifies the temporary schema that certain
            methods / functions use to create intermediate
            objects, if needed.
        - theme:
            [str]
            Theme used to display the
            VerticaPy objects.
            One of the following:
            light | dark | sphinx (only for doc rendering)
        - time_on:
            [bool]
            If set to ``True``, displays the elasped time
            for all SQL queries.
        - tqdm:
            [bool]
            If set to ``True``, a loading bar is displayed
            when using iterative functions.
        - verbosity
            [int]
            API Verbosity, must be a value between 0 and 3.
            0 (silent) | 1 (warning) | 2 (info) | 3 (debug)

    value: object, optional
        New value of the option.

    Examples
    --------
    Import and load the titanic dataset:

    .. hint::

        VerticaPy provides multiple datasets, all of
        which have loaders in the datasets module.

    .. code-block:: python

        from verticapy.datasets import load_titanic

        titanic = load_titanic()
        display(titanic)

    .. ipython:: python
        :suppress:

        from verticapy.datasets import load_titanic
        from verticapy import set_option

        titanic = load_titanic()
        html_file = open("SPHINX_DIRECTORY/figures/_config_config_set_option_1.html", "w")
        html_file.write(titanic._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/_config_config_set_option_1.html

    Import the set_option function:

    .. code-block:: python

        from verticapy import set_option

    Customize vDataFrame Display Settings
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Turn on the ``count_on`` option, which
    displays the total number of elements
    in the dataset:

    .. code-block:: python

        set_option("count_on", True)
        display(titanic)

    .. warning::

        Exercise caution when enabling this option,
        as it may result in decreased performance.
        VerticaPy will perform calculations to determine
        the number of elements in a displayed :py:class:`~vDataFrame`,
        which can have an impact on overall system performance.

    .. ipython:: python
        :suppress:

        set_option("count_on", True)
        html_file = open("SPHINX_DIRECTORY/figures/_config_config_set_option_2.html", "w")
        html_file.write(titanic._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/_config_config_set_option_2.html

    Turn off the display footer:

    .. code-block:: python

        set_option("footer_on", False)
        display(titanic)

    .. ipython:: python
        :suppress:

        set_option("footer_on", False)
        html_file = open("SPHINX_DIRECTORY/figures/_config_config_set_option_3.html", "w")
        html_file.write(titanic._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/_config_config_set_option_3.html

    Sets the maximum number of columns displayed:

    .. note::

        By setting this parameter, we retrieve
        fewer elements from the database, resulting
        in faster visualization.

    .. code-block:: python

        set_option("max_columns", 3)
        display(titanic)

    .. ipython:: python
        :suppress:

        set_option("max_columns", 3)
        html_file = open("SPHINX_DIRECTORY/figures/_config_config_set_option_4.html", "w")
        html_file.write(titanic._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/_config_config_set_option_4.html

    Sets the maximum number of rows displayed:

    .. code-block:: python

        set_option("max_rows", 5)
        display(titanic)

    .. warning::

        Exercise caution when using high
        values for ``max_rows`` and ``max_columns``
        options, as it may lead to an excessive
        amount of data being loaded into memory.
        This can potentially slow down your
        notebook's performance.

    .. ipython:: python
        :suppress:

        set_option("max_rows", 5)
        html_file = open("SPHINX_DIRECTORY/figures/_config_config_set_option_5.html", "w")
        html_file.write(titanic._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/_config_config_set_option_5.html

    Sets the display to light mode:

    .. code-block:: python

        set_option("mode", "light")
        display(titanic)

    .. hint::

        The light mode option streamlines the
        display of :py:class:`~vDataFrame`, creating
        a more minimalistic appearance that can
        enhance the fluidity of your notebook.

    .. ipython:: python
        :suppress:

        set_option("mode", "light")
        html_file = open("SPHINX_DIRECTORY/figures/_config_config_set_option_6.html", "w")
        html_file.write(titanic._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/_config_config_set_option_6.html

    Sets the display to full mode:

    .. code-block:: python

        set_option("mode", "full")
        display(titanic)

    .. ipython:: python
        :suppress:

        set_option("mode", "full")
        html_file = open("SPHINX_DIRECTORY/figures/_config_config_set_option_7.html", "w")
        html_file.write(titanic._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/_config_config_set_option_7.html

    Turn on the missing values percent bar:

    .. code-block:: python

        set_option("percent_bar", True)
        display(titanic)

    .. ipython:: python
        :suppress:

        set_option("percent_bar", True)
        html_file = open("SPHINX_DIRECTORY/figures/_config_config_set_option_8.html", "w")
        html_file.write(titanic._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/_config_config_set_option_8.html

    SQL Generation and Execution Times
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Displays the queries and their execution times:

    .. note::

        Vertica sometimes caches the SQL query,
        resulting in no displayed SQL.

    .. code-block:: python

        set_option("sql_on", True)
        set_option("time_on", True)
        titanic["age"].max()

    **Computing the different aggregations**.

    SELECT /*+LABEL('vDataFrame.aggregate')*/ MAX("age") FROM "public"."titanic" LIMIT 1

    **Execution**: 0.072s

    ``80.0``

    Hides the queries and execution times:

    .. ipython:: python

        set_option("sql_on", False)
        set_option("time_on", False)

    Seed Randomness
    ^^^^^^^^^^^^^^^^

    Sets the seed for the random number
    generator and seeds the random state:

    .. ipython:: python

        set_option("random_state", 2)
        titanic.sample(0.1).shape()

    Change general API colors
    ^^^^^^^^^^^^^^^^^^^^^^^^^^

    Change the graphic colors:

    .. important::

        The API will exclusively use these colors
        for drawing graphics.

    .. ipython:: python

        set_option("colors", ["blue", "red"])
        titanic.hist(["pclass", "survived"])

    .. ipython:: python
        :suppress:

        set_option("plotting_lib", "plotly")
        fig = titanic.hist(["pclass", "survived"])
        fig.write_html("SPHINX_DIRECTORY/figures/_config_config_set_option_hist.html")

    .. raw:: html
      :file: SPHINX_DIRECTORY/figures/_config_config_set_option_hist.html

    .. warning::

        This can be unstable if not enough colors are
        provided. It is advised to use the plotting
        library color options to switch colors.

    .. ipython:: python
        :suppress:

        set_option("colors", None)

    Utilities
    ^^^^^^^^^^

    Change the temporary schema:

    .. important::

        The temporary schema is utilized to create
        elements that should be dropped at the end
        of function execution. In the case of error,
        the element might still exist and will need
        to be manually dropped.

    .. ipython:: python

        set_option("temp_schema", "public")

    .. hint::

        The ``cache`` option enables you to cache
        the aggregations, speeding up the process.
        However, it should only be used on static
        tables; otherwise, the statistics might
        become biased.

    For a full list of the available options, see
    the list for the ``key`` parameter at the top
    of the page.

    .. seealso::

        | :py:meth:`~verticapy.get_option` :
            Returns the value of a specified option.
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


# Performance
register_option(Option("cache", True, "", bool_validator))
register_option(Option("random_state", None, "", optional_positive_int_validator))

# SQL
# Temp Schema
register_option(Option("temp_schema", "public", "", str_validator))
# Query & Time
register_option(Option("sql_on", False, "", bool_validator))
register_option(Option("time_on", False, "", bool_validator))

# Display & Plotting
register_option(Option("interactive", False, "", bool_validator))
register_option(Option("mode", "full", "", in_validator(["full", "light"])))
register_option(
    Option(
        "plotting_lib",
        "plotly",
        "",
        in_validator(["highcharts", "matplotlib", "plotly"]),
    )
)
register_option(
    Option(
        "theme",
        "light",
        "",
        in_validator(["light", "dark", "sphinx"]),
    )
)
register_option(Option("tqdm", True, "", bool_validator))

# Formatting
register_option(Option("count_on", False, "", bool_validator))
register_option(Option("footer_on", True, "", bool_validator))
register_option(Option("insert_comma_numbers", False, "", bool_validator))
register_option(Option("max_cellwidth", 280, "", st_positive_int_validator))  # min 280
register_option(Option("max_columns", 50, "", st_positive_int_validator))
register_option(Option("max_rows", 100, "", st_positive_int_validator))
register_option(
    Option("max_tableheight", 300, "", st_positive_int_validator)
)  # min 300
register_option(Option("percent_bar", False, "", bool_validator))


# Query Labels & Profile
register_option(Option("label_separator", None, "", optional_str_validator))
register_option(Option("label_suffix", None, "", optional_str_validator))
register_option(Option("save_query_profile", True, "", bool_validator))

# Verbosity
register_option(Option("print_info", True, "", bool_validator))
register_option(
    Option(
        "verbosity",
        2,
        "",
        in_validator([0, 1, 2, 3]),
    )
)
