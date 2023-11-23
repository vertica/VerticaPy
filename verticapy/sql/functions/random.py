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
from verticapy.core.string_sql.base import StringSQL


def random() -> StringSQL:
    """
    Returns a Random Number.

    Returns
    -------
    StringSQL
        SQL string.

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [1, 2, 3, 4]})
        # use the random function to create a "split" column
        df["split"] = vpf.random()
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [1, 2, 3, 4]})
        df["split"] = vpf.random()
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_random_random.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_random_random.html
    """
    return StringSQL("RANDOM()", "float")


def randomint(n: int) -> StringSQL:
    """
    Returns a Random Number from 0 through n - 1.

    Parameters
    ----------
    n: int
        Integer Value.

    Returns
    -------
    StringSQL
        SQL string.

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [1, 2, 3, 4]})
        # use the randomint function to create a "split" column
        df["split"] = vpf.randomint(10)
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [1, 2, 3, 4]})
        df["split"] = vpf.randomint(10)
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_random_randomint.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_random_randomint.html
    """
    return StringSQL(f"RANDOMINT({n})", "int")


def seeded_random(random_state: int) -> StringSQL:
    """
    Returns a Seeded Random Number using the input
    random state.

    Parameters
    ----------
    random_state: int
        Integer used to seed the randomness.

    Returns
    -------
    StringSQL
        SQL string.

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [1, 2, 3, 4]})
        # use the seeded_random function to create a "split" column
        df["split"] = vpf.seeded_random(10)
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [1, 2, 3, 4]})
        df["split"] = vpf.seeded_random(10)
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_random_seeded_random.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_random_seeded_random.html
    """
    return StringSQL(f"SEEDED_RANDOM({random_state})", "float")
