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
#
#
# Modules
#
# Standard Python Modules
import warnings
from typing import Union, Literal

# Other modules
import pandas as pd
import numpy as np

# VerticaPy Modules
from verticapy.sql.parsers import read_pandas
from verticapy.core.tablesample import tablesample
from verticapy._utils._collect import save_verticapy_logs
from verticapy.errors import (
    ConnectionError,
    MissingRelation,
    ParameterError,
    QueryError,
)
from verticapy.sql.dtypes import get_data_types
from verticapy.sql.flex import (
    isvmap,
    isflextable,
    compute_flextable_keys,
)
from verticapy._utils._cast import to_category
from verticapy.sql.read import vDataFrameSQL
from verticapy._utils._sql import _executeSQL
from verticapy.sql._utils import (
    quote_ident,
    schema_relation,
    format_schema_table,
    clean_query,
)
from verticapy.connect import (
    EXTERNAL_CONNECTION,
    SPECIAL_SYMBOLS,
)
from verticapy._config.config import OPTIONS

from verticapy.core.str_sql import str_sql
from verticapy.core.vdataframe.aggregate import vDFAGG, vDCAGG
from verticapy.core.vdataframe.corr import vDFCORR, vDCCORR
from verticapy.core.vdataframe.io import vDFIO
from verticapy.core.vdataframe.rolling import vDFROLL
from verticapy.core.vdataframe.plotting import vDFPLOT, vDCPLOT
from verticapy.core.vdataframe.filter import vDFFILTER, vDCFILTER
from verticapy.core.vdataframe.join_union_sort import vDFJUS
from verticapy.core.vdataframe.machine_learning import vDFML
from verticapy.core.vdataframe.math import vDFMATH, vDCMATH
from verticapy.core.vdataframe.sys import vDFSYS, vDCSYS
from verticapy.core.vdataframe.typing import vDFTYPING, vDCTYPING
from verticapy.core.vdataframe.read import vDFREAD, vDCREAD
from verticapy.core.vdataframe.text import vDFTEXT, vDCTEXT
from verticapy.core.vdataframe.utils import vDFUTILS
from verticapy.core.vdataframe.encoding import vDFENCODE, vDCENCODE
from verticapy.core.vdataframe.normalize import vDFNORM, vDCNORM
from verticapy.core.vdataframe.eval import vDFEVAL, vDCEVAL
from verticapy.core.vdataframe.fill import vDFFILL, vDCFILL
from verticapy.core.vdataframe.pivot import vDFPIVOT


###
#                                           _____
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
    vDFIO,
    vDFROLL,
    vDFPLOT,
    vDFFILTER,
    vDFJUS,
    vDFML,
    vDFMATH,
    vDFSYS,
    vDFTYPING,
    vDFREAD,
    vDFTEXT,
    vDFUTILS,
    vDFENCODE,
    vDFNORM,
    vDFEVAL,
    vDFFILL,
    vDFPIVOT,
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
input_relation: str / tablesample / pandas.DataFrame 
                   / list / numpy.ndarray / dict, optional
    If the input_relation is of type str, it must represent the relation 
    (view, table, or temporary table) used to create the object. 
    To get a specific schema relation, your string must include both the 
    relation and schema: 'schema.relation' or '"schema"."relation"'. 
    Alternatively, you can use the 'schema' parameter, in which case 
    the input_relation must exclude the schema name.
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
sql: str, optional
    A SQL query used to create the vDataFrame. If specified, the parameter 
    'input_relation' must be empty.
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
empty: bool, optional
    If set to True, the vDataFrame will be empty. You can use this to create 
    a custom vDataFrame and bypass the initialization check.

Attributes
----------
_VERTICAPY_VARIABLES_: dict
    Dictionary containing all vDataFrame attributes.
        allcols_ind, int      : Integer, used to optimize the SQL 
                                code generation.
        columns, list         : List of the vDataColumn names.
        count, int            : Number of elements of the vDataFrame 
                                (catalog).
        exclude_columns, list : vDataColumns to exclude from the final 
                                relation.
        external, bool        : True if it is an External vDataFrame.
        history, list         : vDataFrame history (user modifications).
        input_relation, str   : Name of the vDataFrame.
        isflex, bool          : True if it is a Flex vDataFrame.
        main_relation, str    : Relation to use to build the vDataFrame 
                                (first floor).
        order_by, dict        : Dictionary of all rules to sort the 
                                vDataFrame.
        saving, list          : List used to reconstruct the 
                                vDataFrame.
        schema, str           : Schema of the input relation.
        where, list           : List of all rules to filter the 
                                vDataFrame.
        max_colums, int       : Maximum number of columns to display.
        max_rows, int         : Maximum number of rows to display.
vDataColumns : vDataColumn
    Each vDataColumn of the vDataFrame is accessible by by specifying its name 
    between brackets. For example, to access the vDataColumn "myVC": 
    vDataFrame["myVC"].
    """

    #
    # Special Methods
    #

    @save_verticapy_logs
    def __init__(
        self,
        input_relation: Union[
            str, pd.DataFrame, np.ndarray, list, tablesample, dict
        ] = "",
        columns: Union[str, list] = [],
        usecols: Union[str, list] = [],
        schema: str = "",
        sql: str = "",
        external: bool = False,
        symbol: Literal[tuple(SPECIAL_SYMBOLS)] = "$",
        sql_push_ext: bool = True,
        empty: bool = False,
    ):
        # Initialization
        if not (isinstance(input_relation, (pd.DataFrame, np.ndarray))):
            assert input_relation or sql or empty, ParameterError(
                "The parameters 'input_relation' and 'sql' cannot both be empty."
            )
            assert not (input_relation) or not (sql) or empty, ParameterError(
                "Either 'sql' or 'input_relation' must be empty."
            )
        else:
            assert not (sql) or empty, ParameterError(
                "Either 'sql' or 'input_relation' must be empty."
            )
        assert isinstance(input_relation, str) or not (schema), ParameterError(
            "schema must be empty when the 'input_relation' is not of type str."
        )
        assert not (sql) or not (schema), ParameterError(
            "schema must be empty when the parameter 'sql' is not empty."
        )
        if isinstance(usecols, str):
            usecols = [usecols]
        if isinstance(columns, str):
            columns = [columns]

        if external:
            if input_relation:
                assert isinstance(input_relation, str), ParameterError(
                    "Parameter 'input_relation' must be a string when using "
                    "external tables."
                )
                if schema:
                    relation = f"{schema}.{input_relation}"
                else:
                    relation = str(input_relation)
                cols = ", ".join(usecols) if usecols else "*"
                query = f"SELECT {cols} FROM {input_relation}"

            else:
                query = sql

            if symbol in EXTERNAL_CONNECTION:
                sql = symbol * 3 + query + symbol * 3

            else:
                raise ConnectionError(
                    "No corresponding Connection Identifier Database is "
                    f"defined (Using the symbol '{symbol}'). Use the "
                    "function connect.set_external_connection to set "
                    "one with the correct symbol."
                )

        self._VERTICAPY_VARIABLES_ = {
            "count": -1,
            "allcols_ind": -1,
            "max_rows": -1,
            "max_columns": -1,
            "sql_magic_result": False,
            "isflex": False,
            "external": external,
            "symbol": symbol,
            "sql_push_ext": external and sql_push_ext,
        }

        if isinstance(input_relation, (tablesample, list, np.ndarray, dict)):

            tb = input_relation

            if isinstance(input_relation, (list, np.ndarray)):

                if isinstance(input_relation, list):
                    input_relation = np.array(input_relation)

                assert len(input_relation.shape) == 2, ParameterError(
                    "vDataFrames can only be created with two-dimensional objects."
                )

                tb = {}
                nb_cols = len(input_relation[0])
                for idx in range(nb_cols):
                    col_name = columns[idx] if idx < len(columns) else f"col{idx}"
                    tb[col_name] = [l[idx] for l in input_relation]
                tb = tablesample(tb)

            elif isinstance(input_relation, dict):

                tb = tablesample(tb)

            if usecols:
                tb_final = {}
                for col in usecols:
                    tb_final[col] = tb[col]
                tb = tablesample(tb_final)

            relation = f"({tb.to_sql()}) sql_relation"
            vDataFrameSQL(relation, name="", schema="", vdf=self)

        elif isinstance(input_relation, pd.DataFrame):

            if usecols:
                df = pandas_to_vertica(input_relation[usecols])
            else:
                df = pandas_to_vertica(input_relation)
            schema = df._VERTICAPY_VARIABLES_["schema"]
            input_relation = df._VERTICAPY_VARIABLES_["input_relation"]
            self.__init__(input_relation=input_relation, schema=schema)

        elif sql:

            # Cleaning the Query
            sql_tmp = clean_query(sql)
            sql_tmp = f"({sql_tmp}) VERTICAPY_SUBTABLE"

            # Filtering some columns
            if usecols:
                usecols_tmp = ", ".join([quote_ident(col) for col in usecols])
                sql_tmp = f"(SELECT {usecols_tmp} FROM {sql_tmp}) VERTICAPY_SUBTABLE"

            # vDataFrame of the Query
            vDataFrameSQL(sql_tmp, name="", schema="", vdf=self)

        elif not (empty):

            if not (schema):
                schema, input_relation = schema_relation(input_relation)
            self._VERTICAPY_VARIABLES_["schema"] = schema.replace('"', "")
            self._VERTICAPY_VARIABLES_["input_relation"] = input_relation.replace(
                '"', ""
            )
            table_name = self._VERTICAPY_VARIABLES_["input_relation"].replace("'", "''")
            schema = self._VERTICAPY_VARIABLES_["schema"].replace("'", "''")
            isflex = isflextable(table_name=table_name, schema=schema)
            self._VERTICAPY_VARIABLES_["isflex"] = isflex
            if isflex:
                columns_dtype = compute_flextable_keys(
                    flex_name=f'"{schema}".{table_name}', usecols=usecols
                )
            else:
                columns_dtype = get_data_types(
                    table_name=table_name, schema=schema, usecols=usecols
                )
            columns_dtype = [(str(dt[0]), str(dt[1])) for dt in columns_dtype]
            columns = ['"' + dt[0].replace('"', "_") + '"' for dt in columns_dtype]
            if not (usecols):
                self._VERTICAPY_VARIABLES_["allcols_ind"] = len(columns)
            assert columns != [], MissingRelation(
                f"No table or views '{self._VERTICAPY_VARIABLES_['input_relation']}' found."
            )
            self._VERTICAPY_VARIABLES_["columns"] = [col for col in columns]
            for col_dtype in columns_dtype:
                column, dtype = col_dtype[0], col_dtype[1]
                if '"' in column:
                    column_str = column.replace('"', "_")
                    warning_message = (
                        f'A double quote " was found in the column {column}, '
                        f"its alias was changed using underscores '_' to {column_str}."
                    )
                    warnings.warn(warning_message, Warning)
                category = to_category(dtype)
                if (dtype.lower()[0:12] in ("long varbina", "long varchar")) and (
                    isflex
                    or isvmap(
                        expr=format_schema_table(schema, table_name), column=column,
                    )
                ):
                    category = "vmap"
                    dtype = (
                        "VMAP(" + "(".join(dtype.split("(")[1:])
                        if "(" in dtype
                        else "VMAP"
                    )
                column_name = '"' + column.replace('"', "_") + '"'
                new_vDataColumn = vDataColumn(
                    column_name,
                    parent=self,
                    transformations=[(quote_ident(column), dtype, category,)],
                )
                setattr(self, column_name, new_vDataColumn)
                setattr(self, column_name[1:-1], new_vDataColumn)
                new_vDataColumn.init = False
            other_parameters = {
                "exclude_columns": [],
                "where": [],
                "order_by": {},
                "history": [],
                "saving": [],
                "main_relation": format_schema_table(
                    self._VERTICAPY_VARIABLES_["schema"],
                    self._VERTICAPY_VARIABLES_["input_relation"],
                ),
            }
            self._VERTICAPY_VARIABLES_ = {
                **self._VERTICAPY_VARIABLES_,
                **other_parameters,
            }


##
#
#   __   ___  ______     ______     __         __  __     __    __     __   __
#  /\ \ /  / /\  ___\   /\  __ \   /\ \       /\ \/\ \   /\ "-./  \   /\ "-.\ \
#  \ \ \' /  \ \ \____  \ \ \/\ \  \ \ \____  \ \ \_\ \  \ \ \-./\ \  \ \ \-.  \
#   \ \__/    \ \_____\  \ \_____\  \ \_____\  \ \_____\  \ \_\ \ \_\  \ \_\\"\_\
#    \/_/      \/_____/   \/_____/   \/_____/   \/_____/   \/_/  \/_/   \/_/ \/_/
#
##


class vDataColumn(
    vDCAGG,
    vDCPLOT,
    vDCMATH,
    vDCTYPING,
    vDCFILTER,
    vDCREAD,
    vDCSYS,
    vDCTEXT,
    vDCCORR,
    vDCENCODE,
    vDCNORM,
    vDCEVAL,
    vDCFILL,
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

    #
    # Special Methods
    #

    def __init__(
        self, alias: str, transformations: list = [], parent=None, catalog: dict = {},
    ):
        self.parent, self.alias, self.transformations = (
            parent,
            alias,
            [elem for elem in transformations],
        )
        self.catalog = {
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
        for elem in catalog:
            self.catalog[elem] = catalog[elem]
        self.init_transf = self.transformations[0][0]
        if self.init_transf == "___VERTICAPY_UNDEFINED___":
            self.init_transf = self.alias
        self.init = True
