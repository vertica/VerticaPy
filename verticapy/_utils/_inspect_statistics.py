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
import numpy as np


def count_functions_classes_methods(module_name: str) -> np.ndarray:
    """
    Counts the number of functions,
    classes and methods in a specific
    module.

    Parameters
    ----------
    module_name: str
        Name of the module
        to inspect.

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
        if inspect.isclass(member):
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
        ("vDataFrame", "verticapy.core.vdataframe"),
        ("TableSample", "verticapy.core.tablesample"),
        ("VerticaModels", "verticapy.machine_learning.vertica"),
        ("inMemoryModels", "verticapy.machine_learning.memmodel"),
        ("Metrics", "verticapy.machine_learning.metrics"),
        ("Model Selection", "verticapy.machine_learning.model_selection"),
        (
            "Statistical Tests",
            "verticapy.machine_learning.model_selection.statistical_tests",
        ),
        ("Plotting Matplotlib", "verticapy.plotting._matplotlib"),
        ("Plotting Highcharts", "verticapy.plotting._highcharts"),
        ("Plotting Plotly", "verticapy.plotting._plotly"),
        ("SQL Functions", "verticapy.sql.functions"),
        ("SQL Statements", "verticapy.sql"),
        ("SQL Geo Extensions", "verticapy.sql.geo"),
        ("Datasets", "verticapy.datasets"),
    ]
    res = {}
    for fun, mod in all_funs:
        res[fun] = count_functions_classes_methods(mod)
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
    res += [("Number of Datasets", f["Datasets"][0])]
    cnt = f["Plotting Matplotlib"] + f["Plotting Highcharts"] + f["Plotting Plotly"]
    res += [("Number of Data Exploration Functions", cnt[1])]
    res += [("Number of Data Preparation Functions", f["vDataFrame"][2])]
    cnt = f["SQL Functions"] + f["SQL Statements"] + f["SQL Geo Extensions"]
    res += [("Number of SQL Functions", cnt[0])]
    res += [("Number of Statistical Tests", f["Statistical Tests"][0])]
    res += [("Number of ML Algorithms", f["VerticaModels"][1])]
    res += [("Number of ML Extensions", f["inMemoryModels"][1])]
    res += [("Number of ML Metrics", f["Metrics"][0])]
    res += [("Number of ML Evaluation", f["Model Selection"][0])]
    return res
