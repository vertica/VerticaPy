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
import copy, warnings
from typing import Literal, Union
import numpy as np

import pandas as pd

from verticapy._config.config import _options
from verticapy._utils._cast import to_category
from verticapy._utils._collect import save_verticapy_logs
from verticapy._utils._sql._check import is_longvar, is_sql_select
from verticapy._utils._sql._execute import _executeSQL
from verticapy._utils._sql._format import (
    clean_query,
    extract_precision_scale,
    extract_subquery,
    format_schema_table,
    quote_ident,
    schema_relation,
)
from verticapy.connection._global import (
    _external_connections,
    SPECIAL_SYMBOLS,
)
from verticapy.errors import (
    ConnectionError,
    MissingRelation,
    ParameterError,
    QueryError,
)

from verticapy.core.vdataframe.aggregate import vDFAGG, vDCAGG
from verticapy.core.vdataframe.corr import vDFCORR, vDCCORR
from verticapy.core.vdataframe.encoding import vDFENCODE, vDCENCODE
from verticapy.core.vdataframe.eval import vDFEVAL, vDCEVAL
from verticapy.core.vdataframe.fill import vDFFILL, vDCFILL
from verticapy.core.vdataframe.filter import vDFFILTER, vDCFILTER
from verticapy.core.vdataframe.io import vDFIO
from verticapy.core.vdataframe.join_union_sort import vDFJUS
from verticapy.core.vdataframe.machine_learning import vDFML
from verticapy.core.vdataframe.math import vDFMATH, vDCMATH
from verticapy.core.vdataframe.normalize import vDFNORM, vDCNORM
from verticapy.core.vdataframe.pivot import vDFPIVOT
from verticapy.core.vdataframe.plotting import vDFPLOT, vDCPLOT
from verticapy.core.vdataframe.read import vDFREAD, vDCREAD
from verticapy.core.vdataframe.rolling import vDFROLL
from verticapy.core.vdataframe.sys import vDFSYS, vDCSYS
from verticapy.core.vdataframe.text import vDFTEXT, vDCTEXT
from verticapy.core.vdataframe.typing import vDFTYPING, vDCTYPING
from verticapy.core.vdataframe.utils import vDFUTILS

from verticapy.core.str_sql.base import str_sql
from verticapy.core.tablesample.base import TableSample

from verticapy.sql.dtypes import get_data_types
from verticapy.sql.flex import (
    compute_flextable_keys,
    isvmap,
    isflextable,
)
from verticapy.sql.parsers.pandas import read_pandas

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


class vDataFrame(
    vDFAGG,
    vDFCORR,
    vDFENCODE,
    vDFEVAL,
    vDFFILL,
    vDFFILTER,
    vDFIO,
    vDFJUS,
    vDFMATH,
    vDFML,
    vDFNORM,
    vDFPIVOT,
    vDFPLOT,
    vDFREAD,
    vDFROLL,
    vDFSYS,
    vDFTEXT,
    vDFTYPING,
    vDFUTILS,
):
    """
An object that records all user modifications, allowing users to 
manipulate the relation without mutating the underlying data in Vertica. 
When changes are made, the vDataFrame queries the Vertica database, which 
aggregates and returns the final result. The vDataFrame creates, for each ]
column of the relation, a Virtual Column (vDataColumn) that stores the column 
alias an all user transformations. 

Parameters
----------
input_relation: str / TableSample / pandas.DataFrame 
                   / list / numpy.ndarray / dict, optional
    If the input_relation is of type str, it must represent the relation 
    (view, table, or temporary table) used to create the object. 
    To get a specific schema relation, your string must include both the 
    relation and schema: 'schema.relation' or '"schema"."relation"'. 
    Alternatively, you can use the 'schema' parameter, in which case 
    the input_relation must exclude the schema name.
    It can also be the SQL query used to create the vDataFrame.
    If it is a pandas.DataFrame, a temporary local table is created.
    Otherwise, the vDataFrame is created using the generated SQL code 
    of multiple UNIONs. 
columns: str / list, optional
    List of column names. Only used when input_relation is an array-like type.
usecols: str / list, optional
    List of columns to use to create the object. As Vertica is a columnar 
    DB including less columns makes the process faster. Do not hesitate 
    to not include useless columns.
schema: str, optional
    The schema of the relation. Specifying a schema allows you to specify a 
    table within a particular schema, or to specify a schema and relation name 
    that contain period '.' characters. If specified, the input_relation cannot 
    include a schema.
external: bool, optional
    A boolean to indicate whether it is an external table. If set to True, a
    Connection Identifier Database must be defined.
    See the connect.set_external_connection function for more information.
symbol: str, optional
    One of the following:
    "$", "€", "£", "%", "@", "&", "§", "?", "!"
    Symbol used to identify the external connection.
    See the connect.set_external_connection function for more information.
sql_push_ext: bool, optional
    If set to True, the external vDataFrame attempts to push the entire query 
    to the external table (only DQL statements - SELECT; for other statements,
    use SQL Magic directly). This can increase performance but might increase 
    the error rate. For instance, some DBs might not support the same SQL as 
    Vertica.
_empty: bool, optional
    If set to True, the vDataFrame will be empty. You can use this to create 
    a custom vDataFrame and bypass the initialization check.

Attributes
----------
_vars: dict
    Dictionary containing all vDataFrame attributes.
        allcols_ind, int      : Integer, used to optimize the SQL 
                                code generation.
        columns, list         : List of the vDataColumn names.
        count, int            : Number of elements of the vDataFrame 
                                (catalog).
        exclude_columns, list : vDataColumns to exclude from the final 
                                relation.
        history, list         : vDataFrame history (user modifications).
        isflex, bool          : True if it is a Flex vDataFrame.
        main_relation, str    : Relation to use to build the vDataFrame 
                                (first floor).
        order_by, dict        : Dictionary of all rules to sort the 
                                vDataFrame.
        saving, list          : List used to reconstruct the 
                                vDataFrame.
        where, list           : List of all rules to filter the 
                                vDataFrame.
        max_colums, int       : Maximum number of columns to display.
        max_rows, int         : Maximum number of rows to display.
vDataColumns : vDataColumn
    Each vDataColumn of the vDataFrame is accessible by by specifying its name 
    between brackets. For example, to access the vDataColumn "myVC": 
    vDataFrame["myVC"].
    """

    @save_verticapy_logs
    def __init__(
        self,
        input_relation: Union[str, pd.DataFrame, np.ndarray, list, TableSample, dict],
        columns: Union[str, list] = [],
        usecols: Union[str, list] = [],
        schema: str = "",
        external: bool = False,
        symbol: Literal[tuple(SPECIAL_SYMBOLS)] = "$",
        sql_push_ext: bool = True,
        _empty: bool = False,
    ) -> None:
        # Main Attributes
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
            "sql_magic_result": False,
            "where": [],
        }
        isflex = False
        # Initialization
        if isinstance(input_relation, str) and is_sql_select(input_relation):
            sql = input_relation
        else:
            sql = ""
        schema = quote_ident(schema)
        if isinstance(usecols, str):
            usecols = [usecols]
        if isinstance(columns, str):
            columns = [columns]

        if external:

            if input_relation:

                if not (isinstance(input_relation, str)):
                    raise ParameterError(
                        "Parameter 'input_relation' must be a string "
                        "when using external tables."
                    )
                input_relation = f"{quote_ident(schema)}.{quote_ident(input_relation)}"
                cols = ", ".join(usecols) if usecols else "*"
                query = f"SELECT {cols} FROM {input_relation}"

            else:
                query = sql

            if symbol in _external_connections:
                sql = symbol * 3 + query + symbol * 3

            else:
                raise ConnectionError(
                    "No corresponding Connection Identifier Database is "
                    f"defined (Using the symbol '{symbol}'). Use the "
                    "function connect.set_external_connection to set "
                    "one with the correct symbol."
                )

        self._vars = {
            **self._vars,
            "sql_push_ext": external and sql_push_ext,
            "symbol": symbol,
        }

        if isinstance(input_relation, (TableSample, list, np.ndarray, dict)):

            if isinstance(input_relation, (list, np.ndarray)):

                if isinstance(input_relation, list):
                    input_relation = np.array(input_relation)

                if len(input_relation.shape) != 2:
                    raise ParameterError(
                        "vDataFrames can only be created with two-dimensional objects."
                    )

                d = {}
                nb_cols = len(input_relation[0])
                for idx in range(nb_cols):
                    col_name = columns[idx] if idx < len(columns) else f"col{idx}"
                    d[col_name] = [l[idx] for l in input_relation]
                tb = TableSample(d)

            elif isinstance(input_relation, dict):

                tb = TableSample(input_relation)

            else:

                tb = input_relation

            if usecols:

                tb_final = {}
                for col in usecols:
                    tb_final[col] = tb[col]
                tb = TableSample(tb_final)

            self.__init__(
                tb.to_sql(),
                external=external,
                symbol=symbol,
                sql_push_ext=sql_push_ext,
            )

        elif isinstance(input_relation, pd.DataFrame):

            argv = input_relation[usecols] if usecols else input_relation
            vdf = read_pandas(argv)
            self.__init__(input_relation=vdf._vars["main_relation"])

        elif not (_empty):

            if sql:

                # Cleaning the Query
                sql = clean_query(sql)
                sql = extract_subquery(sql)

                # Filtering some columns
                if usecols:
                    usecols_tmp = ", ".join([quote_ident(col) for col in usecols])
                    sql = f"SELECT {usecols_tmp} FROM ({sql}) VERTICAPY_SUBTABLE"

                # Getting the main relation information
                main_relation = f"({sql}) VERTICAPY_SUBTABLE"
                dtypes = get_data_types(sql)

            else:

                if not (schema):
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
                    if not (dtypes):
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
            if not (usecols):
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
                        expr=self._vars["main_relation"], column=column,
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
                    transformations=[(quote_ident(column), ctype, category,)],
                )
                setattr(self, column_ident, new_vDataColumn)
                setattr(self, column_ident[1:-1], new_vDataColumn)
                new_vDataColumn._init = False

    def _new_vdatacolumn(self, *argv, **kwds):
        return vDataColumn(*argv, **kwds)

    def _new_vdataframe(self, *argv, **kwds):
        return vDataFrame(*argv, **kwds)


##
#   __   ___  ______     ______     __         __  __     __    __     __   __
#  /\ \ /  / /\  ___\   /\  __ \   /\ \       /\ \/\ \   /\ "-./  \   /\ "-.\ \
#  \ \ \' /  \ \ \____  \ \ \/\ \  \ \ \____  \ \ \_\ \  \ \ \-./\ \  \ \ \-.  \
#   \ \__/    \ \_____\  \ \_____\  \ \_____\  \ \_____\  \ \_\ \ \_\  \ \_\\"\_\
#    \/_/      \/_____/   \/_____/   \/_____/   \/_____/   \/_/  \/_/   \/_/ \/_/
##


class vDataColumn(
    vDCAGG,
    vDCCORR,
    vDCENCODE,
    vDCEVAL,
    vDCFILL,
    vDCFILTER,
    vDCMATH,
    vDCNORM,
    vDCPLOT,
    vDCREAD,
    vDCSYS,
    vDCTEXT,
    vDCTYPING,
    str_sql,
):
    """
Python object which that stores all user transformations. If the vDataFrame
represents the entire relation, a vDataColumn can be seen as one column of that
relation. vDataColumns simplify several processes with its abstractions.

Parameters
----------
alias: str
    vDataColumn alias.
transformations: list, optional
    List of the different transformations. Each transformation must be similar
    to the following: (function, type, category)  
parent: vDataFrame, optional
    Parent of the vDataColumn. One vDataFrame can have multiple children vDataColumns 
    whereas one vDataColumn can only have one parent.
catalog: dict, optional
    Catalog where each key corresponds to an aggregation. vDataColumns will memorize
    the already computed aggregations to gain in performance. The catalog will
    be updated when the parent vDataFrame is modified.

Attributes
----------
    alias, str           : vDataColumn alias.
    catalog, dict        : Catalog of pre-computed aggregations.
    parent, vDataFrame   : Parent of the vDataColumn.
    transformations, str : List of the different transformations.
    """

    def __init__(
        self, alias: str, transformations: list = [], parent=None, catalog: dict = {},
    ) -> None:
        self._parent = parent
        self._alias = alias
        self._transf = copy.deepcopy(transformations)
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
