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
from typing import Literal, Optional


def verticapy_agg_name(key: str, method: Optional[Literal["vertica"]] = "") -> str:
    """
    Returns the VerticaPy name of the input key.
    """
    key = key.lower()
    if key in ("median", "med"):
        key = "50%"
    elif key in ("approx_median", "approximate_median"):
        key = "approx_50%"
    elif key == "100%":
        key = "max"
    elif key == "0%":
        key = "min"
    elif key == "approximate_count_distinct":
        key = "approx_unique"
    elif key == "approximate_count_distinct":
        key = "approx_unique"
    elif key == "ema":
        key = "exponential_moving_average"
    elif key == "mean":
        key = "avg"
    elif key in ("stddev", "stdev"):
        key = "std"
    elif key == "product":
        key = "prod"
    elif key == "variance":
        key = "var"
    elif key == "kurt":
        key = "kurtosis"
    elif key == "skew":
        key = "skewness"
    elif key in ("top1", "mode"):
        key = "top"
    elif key == "top1_percent":
        key = "top_percent"
    elif "%" == key[-1]:
        start = 7 if len(key) >= 7 and key.startswith("approx_") else 0
        if float(key[start:-1]) == int(float(key[start:-1])):
            key = f"{int(float(key[start:-1]))}%"
            if start == 7:
                key = "approx_" + key
    elif key == "row":
        key = "row_number"
    elif key == "first":
        key = "first_value"
    elif key == "last":
        key = "last_value"
    elif key == "next":
        key = "lead"
    elif key in ("prev", "previous"):
        key = "lag"
    if method == "vertica":
        if key == "var":
            key = "variance"
        elif key == "std":
            key = "stddev"
    return key


def param_docstring(*args):
    """
    Constructs and inserts a parameter docstring into the decorated function's existing docstring.
    The decorator accepts a dictionary of parameter descriptions and then the keys to the parameters
    of the decorated function. For example:

    @param_docstring(PARAMETER_DESCRIPTIONS, 'y_true', 'y_score', 'input_relation', 'pos_label')

    The decorator inserts the supplied parameter descriptions inbetween the function's description
    and the Returns section. For instance, in the following docstring, the above decorator would
    insert the parameter descriptions inbetween 'Computes the Confusion Matrix' and 'Returns':

    Computes the Confusion Matrix.

    Returns
    -------
    Array
        confusion matrix.

    When several functions share the same parameters, this decorator can be used to improve code
    readability and doc consistency.

    Note: To preserve correct spacing, add four spaces before the parameter name in the
    dictionary value and for each following line. For example:

    ...
    'y_true': '''    y_true: str
        Response column.''',
    'y_score': '''    y_score: str
        Prediction.''',
    'input_relation': '''    input_relation: SQLRelation
        Relation used for scoring. This relation can
        be a view, table, or a customized relation (if
        an alias is used at the end of the relation).
        For example: (SELECT ... FROM ...) x''',
    ...
    """
    param_defs = args[0]
    parameter_docstring = """Parameters
    ----------\n"""
    for param in args[1:]:
        parameter_docstring += param_defs[param] + "\n"
    parameter_docstring += """
    Returns
    -------"""

    def docstring_decorator(func):
        existing_docstring = func.__doc__
        existing_docstring = existing_docstring.split(
            """Returns
    -------"""
        )
        func.__doc__ = (
            existing_docstring[0] + parameter_docstring + existing_docstring[1]
        )

        return func

    return docstring_decorator
