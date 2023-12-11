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
import inspect
import importlib
from typing import Optional
import numpy as np

from verticapy._typing import NoneType
from verticapy._utils._object import create_new_vdf


def count_functions_classes_methods(
    module_name: str, class_: Optional[str] = None
) -> np.ndarray:
    """
    Counts the number of functions,
    classes and methods in a specific
    module.

    Parameters
    ----------
    module_name: str
        Name of the module
        to inspect.
    class_: str, optional
        Class to inspect.

    Returns
    -------
    np.ndarray
        ``functions,classes,methods``

    Examples
    --------
    The following code demonstrates the
    usage of the function.

    .. ipython:: python

        # Import the function.
        from verticapy._utils._inspect_statistics import count_functions_classes_methods

        # Example.
        count_functions_classes_methods("verticapy.machine_learning.vertica")

    .. note::

        These functions serve as utilities to
        generate some elements in the Sphinx
        documentation.
    """
    # Import the module dynamically
    module = importlib.import_module(module_name)

    # Get all members of the module
    members = inspect.getmembers(module)

    # Count functions, classes, and methods
    function_count = sum(1 for name, member in members if inspect.isfunction(member))
    class_count = sum(1 for name, member in members if inspect.isclass(member))

    attribute_count = 0
    for name, member in members:
        if inspect.isclass(member) and (
            isinstance(class_, NoneType) or class_ in str(member)
        ):
            class_members = inspect.getmembers(member)
            attribute_count += sum(
                [
                    int(not (str(l[0]).startswith("_")))
                    for l in inspect.getmembers(member)
                ]
            )

    return np.array([function_count, class_count, attribute_count])


def count_verticapy_functions():
    """
    Counts the number of functions,
    classes and methods in many
    verticapy modules.

    Returns
    -------
    dict
        ``dictionary`` with the
        section name and the
        number of elements:
        ``functions,classes,methods``

    Examples
    --------
    The following code demonstrates
    the usage of the function.

    .. ipython:: python

        # Import the function.
        from verticapy._utils._inspect_statistics import count_verticapy_functions

        # Example.
        count_verticapy_functions()

    .. note::

        These functions serve as utilities to
        generate some elements in the Sphinx
        documentation.
    """
    all_funs = [
        ("vDataFrame", "verticapy.core.vdataframe", "vDataFrame"),
        ("vDataColumn", "verticapy.core.vdataframe", "vDataColumn"),
        ("TableSample", "verticapy.core.tablesample", "TableSample"),
        ("VerticaModels", "verticapy.machine_learning.vertica", None),
        ("inMemoryModels", "verticapy.machine_learning.memmodel", None),
        ("Metrics", "verticapy.machine_learning.metrics", None),
        ("Model Selection", "verticapy.machine_learning.model_selection", None),
        (
            "Statistical Tests",
            "verticapy.machine_learning.model_selection.statistical_tests",
            None,
        ),
        ("Plotting Matplotlib", "verticapy.plotting._matplotlib", None),
        ("Plotting Highcharts", "verticapy.plotting._highcharts", None),
        ("Plotting Plotly", "verticapy.plotting._plotly", None),
        ("SQL Functions", "verticapy.sql.functions", None),
        ("SQL Statements", "verticapy.sql", None),
        ("SQL Geo Extensions", "verticapy.sql.geo", None),
        ("Datasets", "verticapy.datasets", None),
    ]
    res = {}
    for fun, mod, class_ in all_funs:
        res[fun] = count_functions_classes_methods(mod, class_)
    return res


def summarise_verticapy_functions():
    """
    Returns a summary of the
    entire VerticaPy module.

    Returns
    -------
    list
        ``list`` with the
        section name and the
        number of elements:
        ``title,nb_functions``

    Examples
    --------
    The following code demonstrates
    the usage of the function.

    .. ipython:: python

        # Import the function.
        from verticapy._utils._inspect_statistics import summarise_verticapy_functions

        # Example.
        summarise_verticapy_functions()

    .. note::

        These functions serve as utilities to
        generate some elements in the Sphinx
        documentation.
    """
    f = count_verticapy_functions()
    res = []
    res += [("Loaders & Generators", "Datasets", f["Datasets"][0])]
    res += [("Data Visualization Functions", "Matplotlib", f["Plotting Matplotlib"][1])]
    res += [("Data Visualization Functions", "Highcharts", f["Plotting Highcharts"][1])]
    res += [("Data Visualization Functions", "Plotly", f["Plotting Plotly"][1])]
    res += [
        ("Data Preparation/Exploration Functions", "vDataFrame", f["vDataFrame"][2])
    ]
    res += [
        ("Data Preparation/Exploration Functions", "vDataColumn", f["vDataColumn"][2])
    ]
    res += [
        ("Data Preparation/Exploration Functions", "TableSample", f["TableSample"][2])
    ]
    res += [("SQL Functions & Extensions", "SQL Functions", f["SQL Functions"][0])]
    res += [("SQL Functions & Extensions", "SQL Statements", f["SQL Statements"][0])]
    res += [
        ("SQL Functions & Extensions", "SQL Geo Extensions", f["SQL Geo Extensions"][0])
    ]
    res += [("Machine Learning", "Statistical Tests", f["Statistical Tests"][0])]
    res += [("Machine Learning", "Algorithms/Functions", sum(f["VerticaModels"][0:2]))]
    res += [("Machine Learning", "Extensions", f["inMemoryModels"][1])]
    res += [("Machine Learning", "Metrics", f["Metrics"][0])]
    res += [("Machine Learning", "Evaluation Functions", f["Model Selection"][0])]
    res += [("Total", sum([x[-1] for x in res]))]
    return res


def verticapy_stats_vdf():
    stats = summarise_verticapy_functions()[:-1]
    res = {
        "category": [x[0] for x in stats],
        "subcategory": [x[1] for x in stats],
        "number": [x[2] for x in stats],
    }
    return create_new_vdf(res)


def summarise_verticapy_pie():
    vdf = verticapy_stats_vdf()
    return vdf.pie(
        ["category", "subcategory"], method="sum", of="number", max_cardinality=1000
    )
