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
import collections
from typing import Literal, Optional, Union
import warnings

import numpy as np

import pandas as pd

from verticapy.connection.global_connection import get_global_connection
from verticapy._typing import SQLColumns
from verticapy._utils._object import read_pd
from verticapy._utils._sql._cast import to_category
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._check import is_longvar, is_dql
from verticapy._utils._sql._format import (
    clean_query,
    extract_precision_scale,
    extract_subquery,
    format_schema_table,
    format_type,
    quote_ident,
    schema_relation,
)
from verticapy.errors import MissingRelation

from verticapy.core.vdataframe._plotting_animated import vDFAnimatedPlot
from verticapy.core.vdataframe._plotting import vDCPlot

from verticapy.core.string_sql.base import StringSQL
from verticapy.core.tablesample.base import TableSample

from verticapy.sql.dtypes import get_data_types
from verticapy.sql.flex import (
    compute_flextable_keys,
    isvmap,
    isflextable,
)

###                                          _____
#   _______    ______ ____________    ____  \    \
#   \      |  |      |\           \   \   \ /____/|
#    |     /  /     /| \           \   |  |/_____|/
#    |\    \  \    |/   |    /\     |  |  |    ___
#    \ \    \ |    |    |   |  |    |  |   \__/   \
#     \|     \|    |    |    \/     | /      /\___/|
#      |\         /|   /           /|/      /| | | |
#      | \_______/ |  /___________/ ||_____| /\|_|/
#       \ |     | /  |           | / |     |/
#        \|_____|/   |___________|/  |_____|
#
###


class vDataFrame(vDFAnimatedPlot):
    """
    An  object that records all user
    modifications, allowing users to
    manipulate  the  relation  without
    mutating the underlying data in
    Vertica. When changes are made,
    the :py:class:`~vDataFrame` queries
    the Vertica database, which aggregates
    and returns the final result. The
    :py:class:`~vDataFrame` creates, for
    each column of the relation, a Virtual
    Column (:py:class:`~vDataColumn`) that
    stores the column alias an all
    user transformations.

    Parameters
    ----------

    input_relation: str | TableSample | pandas.DataFrame | list | numpy.ndarray | dict, optional
        If the input_relation is of type
        ``str``, it must represent the
        relation (view, table, or
        temporary table) used to create
        the object.
        To get a specific ``schema`` relation,
        your string must include both the
        relation and schema: ``'schema.relation'``
        or ``'"schema"."relation"'``.
        Alternatively, you can use  the
        'schema' parameter, in which case
        the ``input_relation`` must exclude
        the ``schema`` name. It can also be
        the SQL query used to create the
        :py:class:`~vDataFrame`.
        If it is a ``pandas.DataFrame``, a
        temporary local table is created.
        Otherwise, the vDataFrame is created
        using the generated SQL code of multiple
        UNIONs.

    usecols: SQLColumns, optional
        When ``input_relation`` is not an
        array-like type:
        List of columns used to create the
        object. As Vertica is a columnar DB,
        including  less columns  makes  the
        process faster. Do not hesitate to
        exclude useless columns.
        Otherwise: List of column names.
    schema: str, optional
        The  schema of the relation.
        Specifying a schema  allows
        you to specify a table within
        a particular schema, or to
        specify  a ``schema`` and
        ``relation`` name that contain
        period '.' characters. If
        specified, the ``input_relation``
        cannot include a ``schema``.
    external: bool, optional
        A  boolean  to indicate whether
        it is an external table.
        If set to ``True``, a Connection
        Identifier Database must be
        defined.
    symbol: str, optional
        Symbol used to identify the
        external connection.
        One of the following:
        ``"$", "€", "£", "%", "@", "&", "§", "?", "!"``
    sql_push_ext: bool, optional
        If  set to ``True``, the  external
        :py:class:`~vDataFrame` attempts to
        push the entire query to the external
        table (only DQL statements
        - SELECT;  for other statements,
        use SQL Magic  directly).
        This can increase performance but
        might increase the error rate.
        For instance, some DBs might not
        support the same SQL as Vertica.

    Attributes
    ----------
    vDataColumns : :py:class:`~vDataColumn`
        Each :py:class:`~vDataColumn` of the
        :py:class:`~vDataFrame` is accessible
        by specifying its name between
        brackets. For example, to access
        the :py:class:`~vDataColumn` "myVC":
        ``vDataFrame["myVC"]``.

    Examples
    --------
    In this example, we will look
    at some of the ways how we can
    create a :py:class:`~vDataFrame`.

    - From ``dictionary``
    - From ``numpy.array``
    - From ``pandas.DataFrame``
    - From SQL Query
    - From a table

    After that we will also look at
    the mathematical operators that
    are available:

    - Pandas-Like
    - SQL-Like

    Lastly, we will look at some
    examples of applications of
    functions that be applied
    directly on the
    :py:class:`~vDataFrame`.

    ----

    Let's begin by importing `VerticaPy`.

    .. ipython:: python

        import verticapy as vp

    .. hint::

        By assigning an alias to :py:mod:`verticapy`,
        we mitigate the risk of code collisions with
        other libraries. This precaution is necessary
        because verticapy uses commonly known function
        names like "average" and "median", which can
        potentially lead to naming conflicts. The use
        of an alias ensures that the functions from
        :py:mod:`verticapy` are used as intended without
        interfering with functions from other libraries.

    Dictionary
    ^^^^^^^^^^^

    This is the most direct way to
    create a :py:class:`~vDataFrame`:

    .. ipython:: python

        vdf = vp.vDataFrame(
            {
                "cats": ["A", "B", "C"],
                "reps": [2, 4, 8],
            },
        )

    .. ipython:: python
        :suppress:

        result = vdf
        html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_base_1.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/core_vDataFrame_base_1.html

    Numpy Array
    ^^^^^^^^^^^^

    We can also use a ``numpy.array``:

    .. ipython:: python

        import numpy as np

        vdf = vp.vDataFrame(
            np.array(
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                ],
            ),
            usecols = [
                "col_A",
                "col_B",
                "col_C",
            ],
        )

    .. ipython:: python
        :suppress:

        result = vdf
        html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_base_2.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/core_vDataFrame_base_2.html

    Pandas DataFrame
    ^^^^^^^^^^^^^^^^^

    We can also use a ``pandas.DataFrame`` object:

    .. ipython:: python

        # Import Pandas library
        import pandas as pd

        # Create the data dictionary
        data = {
            'Name': ['John', 'Ali', 'Pheona'],
            'Age': [25, 30, 22],
            'City': ['New York', 'Gaza', 'Los Angeles'],
        }

        # Create the Pandas DataFrame object
        df = pd.DataFrame(data)

        # Create a vDataFrame
        vdf = vp.vDataFrame(df)

    .. ipython:: python
        :suppress:

        result = vdf
        html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_base_3.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/core_vDataFrame_base_3.html

    SQL Query
    ^^^^^^^^^^

    We can also use a SQL Query:

    .. ipython:: python

        # Write a SQL Query to fetch three rows from the Titanic table
        sql_query = "SELECT age, sex FROM public.titanic LIMIT 3;"

        # Create a vDataFrame
        vdf = vp.vDataFrame(sql_query)

    .. ipython:: python
        :suppress:

        result = vdf
        html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_base_4.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/core_vDataFrame_base_4.html

    Table
    ^^^^^^

    A table can also be directly ingested:

    .. ipython:: python

        # Create a vDataFrame from the titanic table in public schema
        vdf = vp.vDataFrame("public.titanic")

    .. ipython:: python
        :suppress:

        result = vdf
        html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_base_4.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/core_vDataFrame_base_4.html

    Mathematical Operators
    ~~~~~~~~~~~~~~~~~~~~~~~

    We can use all the common mathematical
    operators on the :py:class:`~vDataFrame`.

    Pandas-Like
    ^^^^^^^^^^^^

    First let us re-create a simple
    :py:class:`~vDataFrame`:

    .. ipython:: python

        vdf = vp.vDataFrame(
            {
                "cats": ["A", "B", "C"],
                "reps": [2, 4, 8],
            },
        )

    In order to search for a specific
    string value of a specific column:

    .. ipython:: python

        result = vdf[vdf["cats"] == "A"]

    .. ipython:: python
        :suppress:

        html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_base_5.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/core_vDataFrame_base_5.html

    Similarly we can perform a mathematical
    operations as well for numerical columns:

    .. ipython:: python

        result = vdf[vdf["reps"] > 2]

    .. ipython:: python
        :suppress:

        html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_base_5.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/core_vDataFrame_base_5.html

    Both operators could also be combined:

    .. ipython:: python

        result = vdf[vdf["reps"] > 2][vdf["cats"] == "C"]

    .. ipython:: python
        :suppress:

        html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_base_5_2.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/core_vDataFrame_base_5_2.html

    We can also perform mathematical calculations
    on the elements inside the :py:class:`~vDataFrame`
    quite conveniently:

    .. ipython:: python

        vdf["new"] = abs(vdf["reps"] * 4 - 100)

    .. ipython:: python
        :suppress:

        result = vdf
        html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_base_6.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/core_vDataFrame_base_6.html

    SQL-Like
    ^^^^^^^^^

    SQL queries can be directly applied
    on the :py:class:`~vDataFrame` using
    :py:class:`~StringSQL`. This adds a new
    level of flexibility to the :py:class:`~vDataFrame`.
    :py:class:`~StringSQL` allows the user
    to generate formatted SQL queries in
    a string form. Since any SQL query in
    string format can be passed to the
    :py:class:`~vDataFrame`, you can seamlessly
    pass the output of :py:class:`~StringSQL`
    directly to the :py:class:`~vDataFrame`.

    .. ipython:: python

        # Create the SQL Query using StringSQL
        sql_query = vp.StringSQL("reps > 2")

        # Get the output as a vDataFrame
        result = vdf[sql_query]

    .. ipython:: python
        :suppress:

        html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_base_7.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/core_vDataFrame_base_7.html

    .. note::

        Have a look at :py:class:`~StringSQL`
        for more details.

    Another example of a slightly
    advanced SQL Query could be:

    .. ipython:: python

        # Create the SQL Query using StringSQL
        sql_query = vp.StringSQL("reps BETWEEN 3 AND 8 AND cats = 'B'")

        # Get the output as a vDataFrame
        result = vdf[sql_query]

    .. ipython:: python
        :suppress:

        html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_base_8.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/core_vDataFrame_base_8.html

    ----

    Direct Functions
    ~~~~~~~~~~~~~~~~~

    There are many methods that can be directly
    used by :py:class:`~vDataFrame`. Let us look
    at how conveiently we can call them. Here is
    an example of the ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.describe`
    method:

    .. ipython:: python

        # Import the dataset
        from verticapy.datasets import load_titanic

        # Create vDataFrame
        vdf = load_titanic()

        # Summarize the vDataFrame
        vdf.describe()

    .. ipython:: python
        :suppress:

        result = vdf.describe()
        html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_base_7.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/core_vDataFrame_base_7.html

    .. note::

        Explore :py:class:`~vDataFrame`
        and :py:class:`~vDataColumn`
        different methods to see more
        examples.

    .. seealso::

        | :py:class:`~vDataColumn` :
            Columns of :py:class:`~vDataFrame` object.
    """

    @property
    def object_type(self) -> Literal["vDataFrame"]:
        return "vDataFrame"

    @save_verticapy_logs
    def __init__(
        self,
        input_relation: Union[str, list, dict, pd.DataFrame, np.ndarray, TableSample],
        usecols: Optional[SQLColumns] = None,
        schema: Optional[str] = None,
        external: bool = False,
        symbol: str = "$",
        sql_push_ext: bool = True,
        _empty: bool = False,
        _is_sql_magic: int = 0,
        _clean_query: bool = True,
    ) -> None:
        self._vars = {
            "allcols_ind": -1,
            "count": -1,
            "clean_query": _clean_query,
            "exclude_columns": [],
            "history": [],
            "isflex": False,
            "max_columns": -1,
            "max_rows": -1,
            "order_by": {},
            "saving": [],
            "sql_push_ext": external and sql_push_ext,
            "sql_magic_result": _is_sql_magic,
            "symbol": symbol,
            "where": [],
            "has_dpnames": False,
        }
        schema = quote_ident(schema)
        usecols = format_type(usecols, dtype=list)

        if external:
            if isinstance(input_relation, str) and input_relation:
                if schema:
                    input_relation = (
                        f"{quote_ident(schema)}.{quote_ident(input_relation)}"
                    )
                else:
                    input_relation = quote_ident(input_relation)
                cols = ", ".join(usecols) if usecols else "*"
                query = f"SELECT {cols} FROM {input_relation}"

            else:
                raise ValueError(
                    "Parameter 'input_relation' must be a nonempty str "
                    "when using external tables."
                )

            gb_conn = get_global_connection()

            if symbol in gb_conn.get_external_connections:
                query = symbol * 3 + query + symbol * 3

            else:
                raise ConnectionError(
                    "No corresponding Connection Identifier Database is "
                    f"defined (Using the symbol '{symbol}'). Use the "
                    "function connect.set_external_connection to set "
                    "one with the correct symbol."
                )

        if isinstance(input_relation, (TableSample, list, np.ndarray, dict)):
            self._from_object(input_relation, usecols)
            return

        elif isinstance(input_relation, pd.DataFrame):
            self._from_pandas(input_relation, usecols)
            return

        elif not _empty:
            if isinstance(input_relation, str) and is_dql(input_relation):
                # Cleaning the Query
                sql = input_relation
                if _clean_query:
                    sql = clean_query(input_relation)
                    sql = extract_subquery(sql)

                # Filtering some columns
                if usecols:
                    usecols_tmp = ", ".join(quote_ident(usecols))
                    sql = f"SELECT {usecols_tmp} FROM ({sql}) VERTICAPY_SUBTABLE"

                # Getting the main relation information
                main_relation = f"({sql}) VERTICAPY_SUBTABLE"
                dtypes = get_data_types(sql)
                isflex = False

            else:
                if not schema:
                    schema, input_relation = schema_relation(input_relation)
                schema = quote_ident(schema)
                input_relation = quote_ident(input_relation)
                main_relation = format_schema_table(schema, input_relation)
                isflex = isflextable(
                    table_name=input_relation[1:-1], schema=schema[1:-1]
                )
                if isflex:
                    dtypes = compute_flextable_keys(
                        flex_name=f"{schema}.{input_relation[1:-1]}", usecols=usecols
                    )
                    if not dtypes:
                        raise ValueError(
                            f"The flextable {schema}.{input_relation[1:-1]} is empty."
                        )
                else:
                    dtypes = get_data_types(
                        table_name=input_relation[1:-1],
                        schema=schema[1:-1],
                        usecols=usecols,
                    )

            columns = [quote_ident(dt[0]) for dt in dtypes]
            if len(columns) == 0:
                raise MissingRelation(f"No table or views {input_relation} found.")
            if not usecols:
                allcols_ind = len(columns)
            else:
                allcols_ind = -1
            self._vars = {
                **self._vars,
                "allcols_ind": allcols_ind,
                "clean_query": _clean_query,
                "columns": columns,
                "isflex": isflex,
                "main_relation": main_relation,
            }

            # Checking for duplicated names
            cols = [dt[0] for dt in dtypes]
            dnames = [
                item for item, count in collections.Counter(cols).items() if count > 1
            ]
            if len(dnames) > 0:
                warning_message = (
                    "your vDataFrame includes duplicated names, "
                    "it may be due to not assigning aliases to "
                    "some columns generated by scalar operations. "
                    "This can result in an unstable object. To "
                    "resolve this issue, provide aliases to your "
                    "queries."
                )
                warnings.warn(warning_message, Warning)
                self._vars["has_dpnames"] = True

            # Creating the vDataColumns
            for column, ctype in dtypes:
                column_ident = quote_ident(column)
                category = to_category(ctype)
                if is_longvar(ctype):
                    if isflex or isvmap(
                        expr=self._vars["main_relation"],
                        column=column,
                    ):
                        category = "vmap"
                        precision = extract_precision_scale(ctype)[0]
                        if precision:
                            ctype = f"VMAP({precision})"
                        else:
                            ctype = "VMAP"
                new_vDataColumn = vDataColumn(
                    column_ident,
                    parent=self,
                    transformations=[
                        (
                            quote_ident(column),
                            ctype,
                            category,
                        )
                    ],
                )
                setattr(self, column_ident, new_vDataColumn)
                setattr(self, column_ident[1:-1], new_vDataColumn)
                new_vDataColumn._init = False

    def _from_object(
        self,
        object_: Union[np.ndarray, list, TableSample, dict],
        columns: Optional[SQLColumns] = None,
    ) -> None:
        """
        Creates a vDataFrame from an input object.
        """
        columns = format_type(columns, dtype=list)

        if isinstance(object_, (list, np.ndarray)):
            if isinstance(object_, list):
                object_ = np.array(object_)

            if len(object_.shape) != 2:
                raise ValueError(
                    "vDataFrames can only be created with two-dimensional objects."
                )

            d = {}
            nb_cols = len(object_[0])
            n = len(columns)
            for idx in range(nb_cols):
                col_name = columns[idx] if idx < n else f"col{idx}"
                d[col_name] = [l[idx] for l in object_]
            tb = TableSample(d)

        elif isinstance(object_, dict):
            tb = TableSample(object_)

        else:
            tb = object_

        if len(columns) > 0:
            tb_final = {}
            for col in columns:
                tb_final[col] = tb[col]
            tb = TableSample(tb_final)

        self.__init__(input_relation=tb.to_sql(), _clean_query=False)

    def _from_pandas(
        self,
        object_: pd.DataFrame,
        usecols: Optional[SQLColumns] = None,
    ) -> None:
        """
        Creates a vDataFrame from a pandas.DataFrame.
        """
        usecols = format_type(usecols, dtype=list)
        args = object_[usecols] if len(usecols) > 0 else object_
        vdf = read_pd(args)
        self.__init__(input_relation=vdf._vars["main_relation"], _clean_query=False)


##
#   __   ___  ______     ______     __         __  __     __    __     __   __
#  /\ \ /  / /\  ___\   /\  __ \   /\ \       /\ \/\ \   /\ "-./  \   /\ "-.\ \
#  \ \ \' /  \ \ \____  \ \ \/\ \  \ \ \____  \ \ \_\ \  \ \ \-./\ \  \ \ \-.  \
#   \ \__/    \ \_____\  \ \_____\  \ \_____\  \ \_____\  \ \_\ \ \_\  \ \_\\"\_\
#    \/_/      \/_____/   \/_____/   \/_____/   \/_____/   \/_/  \/_/   \/_/ \/_/
##


class vDataColumn(vDCPlot, StringSQL):
    """
    Python object that stores all user
    transformations. If the :py:class:`~vDataFrame`
    represents the entire relation, a
    :py:class:`~vDataColumn` can be seen
    as one column of that relation.
    Through its abstractions, :py:class:`~vDataColumn`
    simplify several processes.

    Parameters
    ----------
    alias: str
        :py:class:`~vDataColumn` alias.
    transformations: list, optional
        List of the different  transformations.
        Each transformation must be similar to
        the following: ``(function, type, category)``
    parent: vDataFrame, optional
        Parent of the :py:class:`~vDataColumn`.
        One :py:class:`~vDataFrame` can have
        multiple children :py:class:`~vDataColumn`,
        whereas one :py:class:`~vDataColumn` can
        only have one parent.
    catalog: dict, optional
        Catalog where each key corresponds to an
        aggregation. :py:class:`~vDataColumn` will
        memorize the already computed aggregations
        to increase performance. The catalog is
        updated when the parent :py:class:`~vDataFrame`
        is modified.

    Attributes
    ----------
    alias, str:
        :py:class:`~vDataColumn` alias.
    catalog, dict:
        Catalog of pre-computed aggregations.
    parent, vDataFrame:
        Parent of the :py:class:`~vDataColumn`.
    transformations, str:
        List of the different transformations.

    Examples
    --------
    Let's begin by importing `VerticaPy`.

    .. ipython:: python

        import verticapy as vp

    .. hint::

        By assigning an alias to :py:mod:`verticapy`,
        we mitigate the risk of code collisions with
        other libraries. This precaution is necessary
        because verticapy uses commonly known function
        names like "average" and "median", which can
        potentially lead to naming conflicts. The use
        of an alias ensures that the functions from
        :py:mod:`verticapy` are used as intended without
        interfering with functions from other libraries.

    Let's create a :py:class:`~vDataFrame`
    with two :py:class:`~vDataColumn`:

    .. ipython:: python

        vdf = vp.vDataFrame(
            {
                "cats": ["A", "B", "C"],
                "reps": [2, 4, 8],
            },
        )

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/core_vDataFrame_base_1.html

    "cats" and "reps" are :py:class:`~vDataColumn`
    objects. They can be accessed the same way as
    a ``dictionary`` or a ``pandas.DataFrame``.
    They represent the columns of the entire
    relation.

    For example, the following code will access
    the :py:class:`~vDataColumn` "cats":

    .. code-block:: python

        vdf["cats"]

    .. note::

        :py:class:`~vDataColumn` are columns inside a
        :py:class:`~vDataFrame`; they have their own
        methods but cannot exist without a parent
        :py:class:`~vDataFrame`. Please refer to
        :py:class:`~vDataFrame` to see an entire
        example.

    .. seealso::

        | :py:class:`~vDataFrame` :
            Main VerticaPy dataset object.
    """

    @property
    def object_type(self) -> Literal["vDataColumn"]:
        return "vDataColumn"

    def __init__(
        self,
        alias: str,
        transformations: Optional[list] = None,
        parent: Optional[vDataFrame] = None,
        catalog: Optional[dict] = None,
    ) -> None:
        self._parent = parent
        self._alias = alias
        self._transf = format_type(transformations, dtype=list)
        catalog = format_type(catalog, dtype=dict)
        self._catalog = {
            "cov": {},
            "pearson": {},
            "spearman": {},
            "spearmand": {},
            "kendall": {},
            "cramer": {},
            "biserial": {},
            "regr_avgx": {},
            "regr_avgy": {},
            "regr_count": {},
            "regr_intercept": {},
            "regr_r2": {},
            "regr_slope": {},
            "regr_sxx": {},
            "regr_sxy": {},
            "regr_syy": {},
        }
        for key in catalog:
            self._catalog[key] = catalog[key]
        self._init_transf = self._transf[0][0]
        if self._init_transf == "___VERTICAPY_UNDEFINED___":
            self._init_transf = self._alias
        self._init = True
