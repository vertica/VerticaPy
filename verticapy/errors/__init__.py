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


class ConversionError(Exception):
    """
    An exception raised when an error occurs during
    the process of converting data from one type or
    format to another.

    This error typically indicates issues such as
    data type discrepancies, incompatible formats,
    or challenges in the conversion process.

    Examples
    --------
    The following function verifies whether the input, treated
    as a varchar, is of float datatype. If not, it raises an
    error.

    .. code-block:: python

        def is_convertible(dtype: str):

            if dtype != "float":

                raise ConversionError(
                    "The function exclusively accepts float values."
                )
    .. note::

        Errors can be employed when implementing new functions
        to provide clear explanations for why a particular
        option or feature is not functioning as intended.
        It is essential to review all available errors to
        choose the most appropriate one.
    """

    ...


class EmptyParameter(Exception):
    """
    An exception raised when a function or operation requires
    a parameter, and the provided parameter is empty or lacks
    the required information. This error typically indicates
    that essential information is missing, hindering the
    successful execution of the function or operation.

    Examples
    --------
    The following function checks if the input
    list is not empty. It raises an error if
    that is the case.

    .. code-block:: python

        from verticapy.errors import EmptyParameter

        def is_empty_list(L: list):

            if L in ([], None):

                raise EmptyParameter(
                    "The input list is empty."
                )

    .. note::

        Errors can be employed when implementing new functions
        to provide clear explanations for why a particular
        option or feature is not functioning as intended.
        It is essential to review all available errors to
        choose the most appropriate one.
    """

    ...


class ExtensionError(Exception):
    """
    An exception raised when an error occurs related to the use
    or integration of extensions. This error typically indicates
    issues such as unsupported or incompatible extensions,
    configuration problems, or challenges in the extension's
    functionality, hindering its successful use within a system
    or application.

    Examples
    --------
    The following function checks if the input
    file is a CSV. It raises an error if
    that is not the case.

    .. code-block:: python

        from verticapy.errors import ExtensionError

        def is_csv(file_name: str):

            if file_name.lower().split(".")[1] != 'csv':

                raise ExtensionError(
                    "The input file is not a CSV."
                )

    .. note::

        Errors can be employed when implementing new functions
        to provide clear explanations for why a particular
        option or feature is not functioning as intended.
        It is essential to review all available errors to
        choose the most appropriate one.
    """

    ...


class FunctionError(Exception):
    """
    An exception raised when an error occurs during the
    execution of a function. This error typically indicates
    issues such as incorrect usage, invalid parameters,
    or challenges within the function's implementation,
    preventing the successful completion of the intended
    operation.

    Examples
    --------
    The following function checks if the input
    aggregation is supported. It raises an error
    if that is not the case.

    .. code-block:: python

        from verticapy.errors import ModelError

        def is_supported_agg(agg_type: str):

            if agg_type.lower() not in ('avg', 'sum', 'var'):

                raise FunctionError(
                    f"The aggregation '{agg_type}' is not yet supported."
                )

    .. note::

        ``FunctionError`` is a subclass of ``ValueError``.

    .. note::

        Errors can be employed when implementing new functions
        to provide clear explanations for why a particular
        option or feature is not functioning as intended.
        It is essential to review all available errors to
        choose the most appropriate one.
    """

    ...


class MissingColumn(Exception):
    """
    An exception raised when attempting to access or
    manipulate a specific column within a dataset that
    is missing or not present. This error typically
    indicates that the requested column is not available
    in the dataset, hindering the successful execution
    of the operation.

    Examples
    --------
    The following function checks for the existence
    of the input column and raises an error if it
    does not exist.

    .. code-block:: python

        from verticapy._utils._sql._sys import _executeSQL
        from verticapy.errors import MissingColumn

        def does_column_exists(table_name: str, column_name: str):

            query = f"SELECT * FROM columns WHERE table_name = '{table_name}' AND column_name = '{column_name}';"
            result = _executeSQL(query, method="fetchall")
            assert result, MissingColumn(
                f"The column '{column_name}' does not exist."
            )

    .. note::

        Errors can be employed when implementing new functions
        to provide clear explanations for why a particular
        option or feature is not functioning as intended.
        It is essential to review all available errors to
        choose the most appropriate one.
    """

    ...


class MissingRelation(Exception):
    """
    An exception raised when attempting to access or
    perform an operation on a relational data structure
    that is missing or not available. This error
    typically signifies the absence of the required
    dataset or data structure needed for the intended
    operation.

    Examples
    --------
    The following function checks for the existence
    of the input relation and raises an error if it
    does not exist.

    .. code-block:: python

        from verticapy._utils._sql._sys import _executeSQL
        from verticapy.errors import MissingRelation

        def does_table_exists(table_name: str):

            query = f"SELECT * FROM columns WHERE table_name = '{table_name}';"
            result = _executeSQL(query, method="fetchall")
            assert result, MissingRelation(
                f"The relation '{table_name}' does not exist."
            )

    .. note::

        Errors can be employed when implementing new functions
        to provide clear explanations for why a particular
        option or feature is not functioning as intended.
        It is essential to review all available errors to
        choose the most appropriate one.
    """

    ...


class MissingSchema(Exception):
    """
    An exception raised when attempting to perform an
    operation that requires a specified schema or
    structure that is missing or not provided. This
    error typically indicates the absence of essential
    information needed for the operation to proceed
    successfully.

    Examples
    --------
    The following function checks for the existence
    of the input schema and raises an error if it
    does not exist.

    .. code-block:: python

        from verticapy._utils._sql._sys import _executeSQL
        from verticapy.errors import MissingSchema

        def does_schema_exists(schema: str):

            query = f"SELECT * FROM columns WHERE table_schema = '{schema}';"
            result = _executeSQL(query, method="fetchall")
            assert result, MissingSchema(
                f"The schema '{schema}' does not exist."
            )

    .. note::

        Errors can be employed when implementing new functions
        to provide clear explanations for why a particular
        option or feature is not functioning as intended.
        It is essential to review all available errors to
        choose the most appropriate one.
    """

    ...


class ModelError(Exception):
    """
    An exception raised when an error occurs during the
    modeling process, indicating issues such as invalid
    model configurations, incompatible data, or challenges
    in the training or prediction phase. This error is
    typically encountered in machine learning contexts
    when there are issues with the model's structure,
    data, or parameters.

    Examples
    --------
    The following function checks if the input
    model is supported.

    .. code-block:: python

        from verticapy.errors import ModelError

        def is_supported_model(model_type: str):

            if model_type.lower() not in ('linear_reg', 'logistic_reg'):

                raise OptionError(
                    f"The model type '{model_type}' is not yet supported."
                )

    .. note::

        Errors can be employed when implementing new functions
        to provide clear explanations for why a particular
        option or feature is not functioning as intended.
        It is essential to review all available errors to
        choose the most appropriate one.
    """

    ...


class OptionError(Exception):
    """
    An exception raised when an error occurs due to invalid
    or unsupported options provided in a configuration or
    settings context. This error typically signifies issues
    such as unrecognized or incompatible options, hindering
    the correct configuration of a system or module.

    Examples
    --------
    The following function checks if the input
    chart is supported.

    .. code-block:: python

        from verticapy.errors import OptionError

        def is_supported_chart(kind: str):

            if kind not in ('bar', 'pie'):

                raise OptionError(
                    f"The option '{kind}' is not supported."
                )

    .. note::

        ``OptionError`` is a subclass of ``ValueError``.

    .. note::

        Errors can be employed when implementing new functions
        to provide clear explanations for why a particular
        option or feature is not functioning as intended.
        It is essential to review all available errors to
        choose the most appropriate one.
    """

    ...


class ParsingError(Exception):
    """
    An exception raised when an error occurs during
    the parsing of data or code. This error indicates
    challenges such as syntax errors, incorrect
    formatting, or issues with interpreting the
    structure of the input, hindering the successful
    parsing process.

    Examples
    --------
    The following function checks if the input
    string does not start with flower brackets.

    .. code-block:: python

        from verticapy.errors import ParsingError

        def does_not_start_fb(x: str):

            if x.startswith(('{', '}')):

                raise ParsingError(
                    "The file can not starts with flower brackets."
                )

    .. note::

        Errors can be employed when implementing new functions
        to provide clear explanations for why a particular
        option or feature is not functioning as intended.
        It is essential to review all available errors to
        choose the most appropriate one.
    """

    ...


class QueryError(Exception):
    """
    An exception raised when an error occurs during
    the execution of a query. This error typically
    indicates issues such as syntax errors, invalid
    queries, or problems with the underlying data
    structure, preventing the successful execution
    of the query.

    Examples
    --------
    The following function checks if the query
    includes a 'SELECT' statement.

    .. code-block:: python

        from verticapy.errors import QueryError

        def include_select(query: str):

            if 'SELECT ' not in query.upper():

                raise QueryError(
                    "The query must include a 'SELECT' statement."
                )

    .. note::

        Errors can be employed when implementing new functions
        to provide clear explanations for why a particular
        option or feature is not functioning as intended.
        It is essential to review all available errors to
        choose the most appropriate one.
    """

    ...


class VersionError(Exception):
    """
    An exception raised when an error occurs due to
    incompatible or unexpected software version
    dependencies.

    This error indicates that the current version of
    a component or module is not compatible with the
    required or expected version, leading to operational
    issues.

    Examples
    --------
    The following function checks if the version is greater
    than or equal to 9; if not, it raises an error.

    .. code-block:: python

        from verticapy.errors import VersionError

        def version_ge_9(version: tuple):

            if version[0] < 9:

                raise VersionError(
                    "The feature requires a version greater than or equal to 9."
                )

    .. note::

        Errors can be employed when implementing new functions
        to provide clear explanations for why a particular
        option or feature is not functioning as intended.
        It is essential to review all available errors to
        choose the most appropriate one.
    """

    ...
