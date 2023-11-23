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
from typing import Literal, Optional, Union

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
    An  object that  records  all  user modifications, allowing
    users to  manipulate  the  relation  without  mutating  the
    underlying  data in  Vertica.  When  changes are  made, the
    vDataFrame queries the Vertica database,  which  aggregates
    and returns  the final result. The vDataFrame  creates, for
    each column of the relation, a Virtual Column (vDataColumn)
    that stores the column alias an all user transformations.

    Parameters
    ----------
    input_relation: str / TableSample / pandas.DataFrame
                       / list / numpy.ndarray / dict, optional
        If the input_relation is of type str, it must represent
        the relation  (view, table, or temporary table) used to
        create the object.
        To  get a  specific  schema relation,  your string must
        include both the relation and schema: 'schema.relation'
        or '"schema"."relation"'.
        Alternatively, you can use  the  'schema' parameter, in
        which  case the input_relation must exclude the  schema
        name.  It can also be the SQL query used to create  the
        vDataFrame.
        If it is a pandas.DataFrame, a temporary local table is
        created. Otherwise, the vDataFrame is created using the
        generated SQL code of multiple UNIONs.
    usecols: SQLColumns, optional
        When input_relation is not an array-like type:
        List of columns used to create the object. As Vertica
        is  a  columnar DB, including  less columns  makes  the
        process faster.  Do not hesitate to exclude useless
        columns.
        Otherwise:
        List of column names.
    schema: str, optional
        The  schema of the relation. Specifying a schema  allows
        you to specify a table within a particular schema, or to
        specify  a schema and relation name that contain  period
        '.' characters. If specified, the input_relation  cannot
        include a schema.
    external: bool, optional
        A  boolean  to indicate whether it is an external table.
        If set to True, a Connection Identifier Database must be
        defined.
    symbol: str, optional
        Symbol used to identify the external connection.
        One of the following:
        "$", "€", "£", "%", "@", "&", "§", "?", "!"
    sql_push_ext: bool, optional
        If  set to True, the  external vDataFrame attempts to  push
        the entire query to the external table (only DQL statements
        - SELECT;  for other statements,  use SQL Magic  directly).
        This can increase performance  but might increase the error
        rate. For instance, some DBs might not support the same SQL
        as Vertica.

    Attributes
    ----------
    vDataColumns : vDataColumn
        Each   vDataColumn  of  the  vDataFrame  is  accessible   by
        specifying its name between brackets. For example, to access
        the vDataColumn "myVC": vDataFrame["myVC"].
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
        _is_sql_magic: bool = False,
    ) -> None:
        self._vars = {
            "allcols_ind": -1,
            "count": -1,
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
                "columns": columns,
                "isflex": isflex,
                "main_relation": main_relation,
            }

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

        self.__init__(input_relation=tb.to_sql())

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
        self.__init__(input_relation=vdf._vars["main_relation"])


##
#   __   ___  ______     ______     __         __  __     __    __     __   __
#  /\ \ /  / /\  ___\   /\  __ \   /\ \       /\ \/\ \   /\ "-./  \   /\ "-.\ \
#  \ \ \' /  \ \ \____  \ \ \/\ \  \ \ \____  \ \ \_\ \  \ \ \-./\ \  \ \ \-.  \
#   \ \__/    \ \_____\  \ \_____\  \ \_____\  \ \_____\  \ \_\ \ \_\  \ \_\\"\_\
#    \/_/      \/_____/   \/_____/   \/_____/   \/_____/   \/_/  \/_/   \/_/ \/_/
##


class vDataColumn(vDCPlot, StringSQL):
    """
    Python object that stores all user transformations. If   the
    vDataFrame  represents the entire relation, a vDataColumn can
    be seen  as  one  column of  that  relation. Through its
    abstractions, vDataColumns simplify several processes.

    Parameters
    ----------
    alias: str
        vDataColumn alias.
    transformations: list, optional
        List of the different  transformations. Each transformation
        must be similar to the following: (function, type, category)
    parent: vDataFrame, optional
        Parent of the vDataColumn. One vDataFrame can have multiple
        children vDataColumns, whereas one vDataColumn can only have
        one parent.
    catalog: dict, optional
        Catalog where each key corresponds to an aggregation.
        vDataColumns will memorize the already computed aggregations
        to increase performance. The catalog is updated when the
        parent vDataFrame is modified.

    Attributes
    ----------
    alias, str           : vDataColumn alias.
    catalog, dict        : Catalog of pre-computed aggregations.
    parent, vDataFrame   : Parent of the vDataColumn.
    transformations, str : List of the different transformations.
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
