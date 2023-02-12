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
import random, time, re, decimal, warnings, pickle, datetime, math, os, copy, sys
from collections.abc import Iterable
from itertools import combinations_with_replacement
from typing import Union, Literal

pickle.DEFAULT_PROTOCOL = 4

# Other modules
import multiprocessing
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import scipy.stats as scipy_st
import scipy.special as scipy_special
from matplotlib.lines import Line2D

# Jupyter - Optional
try:
    from IPython.display import HTML, display
except:
    pass

# VerticaPy Modules
from verticapy.sql.parsers.pandas import pandas_to_vertica
from verticapy.core.tablesample import tablesample
from verticapy.core.vcolumn import vColumn
from verticapy.learn.memmodel import memModel
from verticapy.plotting._highcharts import hchart_from_vdf
from verticapy._utils._collect import save_verticapy_logs
from verticapy.errors import (
    ConnectionError,
    EmptyParameter,
    FunctionError,
    MissingColumn,
    MissingRelation,
    ParameterError,
    QueryError,
)
from verticapy.sql.drop import drop
from verticapy.sql.dtypes import get_data_types
from verticapy.sql.flex import (
    isvmap,
    isflextable,
    compute_flextable_keys,
    compute_vmap_keys,
)
from verticapy._version import vertica_version
from verticapy._config.config import current_random
from verticapy._utils._cast import to_category, to_varchar
from verticapy._utils._gen import gen_name, gen_tmp_name
from verticapy.sql.read import to_tablesample, vDataFrameSQL, readSQL
from verticapy._utils._sql import _executeSQL
from verticapy.plotting._matplotlib.core import updated_dict
from verticapy.plotting._colors import gen_colors, gen_cmap
from verticapy.core.str_sql import str_sql
from verticapy.sql._utils._format import (
    format_magic,
    quote_ident,
    schema_relation,
    format_schema_table,
    clean_query,
)
from verticapy.core._utils._merge import gen_coalesce, group_similar_names
from verticapy.core._utils._map import verticapy_agg_name
from verticapy.connect.connect import (
    EXTERNAL_CONNECTION,
    SPECIAL_SYMBOLS,
)
from verticapy._config.config import OPTIONS
from verticapy.core.vdataframe.aggregate import vDFAGG
from verticapy.core.vdataframe.corr import vDFCORR
from verticapy.core.vdataframe.io import vDFIO
from verticapy.core.vdataframe.rolling import vDFROLL
from verticapy.core.vdataframe.plotting import vDFPLOT
from verticapy.core.vdataframe.filter import vDFFILTER

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


class vDataFrame(vDFAGG, vDFCORR, vDFIO, vDFROLL, vDFPLOT, vDFFILTER):
    """
An object that records all user modifications, allowing users to 
manipulate the relation without mutating the underlying data in Vertica. 
When changes are made, the vDataFrame queries the Vertica database, which 
aggregates and returns the final result. The vDataFrame creates, for each ]
column of the relation, a Virtual Column (vColumn) that stores the column 
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
        columns, list         : List of the vColumn names.
        count, int            : Number of elements of the vDataFrame 
                                (catalog).
        exclude_columns, list : vColumns to exclude from the final 
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
vColumns : vColumn
    Each vColumn of the vDataFrame is accessible by by specifying its name 
    between brackets. For example, to access the vColumn "myVC": 
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
                new_vColumn = vColumn(
                    column_name,
                    parent=self,
                    transformations=[(quote_ident(column), dtype, category,)],
                )
                setattr(self, column_name, new_vColumn)
                setattr(self, column_name[1:-1], new_vColumn)
                new_vColumn.init = False
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

    def __abs__(self):
        return self.copy().abs()

    def __ceil__(self, n):
        vdf = self.copy()
        columns = vdf.numcol()
        for elem in columns:
            if vdf[elem].category() == "float":
                vdf[elem].apply_fun(func="ceil", x=n)
        return vdf

    def __floor__(self, n):
        vdf = self.copy()
        columns = vdf.numcol()
        for elem in columns:
            if vdf[elem].category() == "float":
                vdf[elem].apply_fun(func="floor", x=n)
        return vdf

    def __getitem__(self, index):

        if isinstance(index, slice):
            assert index.step in (1, None), ValueError(
                "vDataFrame doesn't allow slicing having steps different than 1."
            )
            index_stop = index.stop
            index_start = index.start
            if not (isinstance(index_start, int)):
                index_start = 0
            if index_start < 0:
                index_start += self.shape()[0]
            if isinstance(index_stop, int):
                if index_stop < 0:
                    index_stop += self.shape()[0]
                limit = index_stop - index_start
                if limit <= 0:
                    limit = 0
                limit = f" LIMIT {limit}"
            else:
                limit = ""
            query = f"""
                (SELECT * 
                FROM {self.__genSQL__()}
                {self.__get_last_order_by__()} 
                OFFSET {index_start}{limit}) VERTICAPY_SUBTABLE"""
            return vDataFrameSQL(query)

        elif isinstance(index, int):
            columns = self.get_columns()
            for idx, elem in enumerate(columns):
                if self[elem].category() == "float":
                    columns[idx] = f"{elem}::float"
            if index < 0:
                index += self.shape()[0]
            return _executeSQL(
                query=f"""
                    SELECT /*+LABEL('vDataframe.__getitem__')*/ 
                        {', '.join(columns)} 
                    FROM {self.__genSQL__()}
                    {self.__get_last_order_by__()} 
                    OFFSET {index} LIMIT 1""",
                title="Getting the vDataFrame element.",
                method="fetchrow",
                sql_push_ext=self._VERTICAPY_VARIABLES_["sql_push_ext"],
                symbol=self._VERTICAPY_VARIABLES_["symbol"],
            )

        elif isinstance(index, (str, str_sql)):
            is_sql = False
            if isinstance(index, vColumn):
                index = index.alias
            elif isinstance(index, str_sql):
                index = str(index)
                is_sql = True
            try:
                new_index = self.format_colnames(index)
                return getattr(self, new_index)
            except:
                if is_sql:
                    return self.search(conditions=index)
                else:
                    return getattr(self, index)

        elif isinstance(index, Iterable):
            try:
                return self.select(columns=[str(col) for col in index])
            except:
                return self.search(conditions=[str(col) for col in index])

        else:
            return getattr(self, index)

    def __iter__(self):
        columns = self.get_columns()
        return (col for col in columns)

    def __len__(self):
        return int(self.shape()[0])

    def __nonzero__(self):
        return self.shape()[0] > 0 and not (self.empty())

    def __repr__(self):
        if self._VERTICAPY_VARIABLES_["sql_magic_result"] and (
            self._VERTICAPY_VARIABLES_["main_relation"][-10:] == "VSQL_MAGIC"
        ):
            return readSQL(
                self._VERTICAPY_VARIABLES_["main_relation"][1:-12],
                OPTIONS["time_on"],
                OPTIONS["max_rows"],
            ).__repr__()
        max_rows = self._VERTICAPY_VARIABLES_["max_rows"]
        if max_rows <= 0:
            max_rows = OPTIONS["max_rows"]
        return self.head(limit=max_rows).__repr__()

    def _repr_html_(self, interactive=False):
        if self._VERTICAPY_VARIABLES_["sql_magic_result"] and (
            self._VERTICAPY_VARIABLES_["main_relation"][-10:] == "VSQL_MAGIC"
        ):
            self._VERTICAPY_VARIABLES_["sql_magic_result"] = False
            return readSQL(
                self._VERTICAPY_VARIABLES_["main_relation"][1:-12],
                OPTIONS["time_on"],
                OPTIONS["max_rows"],
            )._repr_html_(interactive)
        max_rows = self._VERTICAPY_VARIABLES_["max_rows"]
        if max_rows <= 0:
            max_rows = OPTIONS["max_rows"]
        return self.head(limit=max_rows)._repr_html_(interactive)

    def __round__(self, n):
        vdf = self.copy()
        columns = vdf.numcol()
        for elem in columns:
            if vdf[elem].category() == "float":
                vdf[elem].apply_fun(func="round", x=n)
        return vdf

    def __setattr__(self, attr, val):
        if isinstance(val, (str, str_sql, int, float)) and not isinstance(val, vColumn):
            val = str(val)
            if self.is_colname_in(attr):
                self[attr].apply(func=val)
            else:
                self.eval(name=attr, expr=val)
        elif isinstance(val, vColumn) and not (val.init):
            final_trans, n = val.init_transf, len(val.transformations)
            for i in range(1, n):
                final_trans = val.transformations[i][0].replace("{}", final_trans)
            self.eval(name=attr, expr=final_trans)
        else:
            self.__dict__[attr] = val

    def __setitem__(self, index, val):
        setattr(self, index, val)

    #
    # Semi Special Methods
    #

    def __add_to_history__(self, message: str):
        """
    VERTICAPY stores the user modification and help the user to look at 
    what he/she did. This method is to use to add a customized message in the 
    vDataFrame history attribute.
        """
        self._VERTICAPY_VARIABLES_["history"] += [
            "{" + time.strftime("%c") + "}" + " " + message
        ]
        return self

    def __genSQL__(
        self, split: bool = False, transformations: dict = {}, force_columns: list = [],
    ):
        """
    Method to use to generate the SQL final relation. It will look at all 
    transformations to build a nested query where each transformation will 
    be associated to a specific floor.

    Parameters
    ----------
    split: bool, optional
        Adds a split column __verticapy_split__ in the relation 
        which can be to use to downsample the data.
    transformations: dict, optional
        Dictionary of columns and their respective transformation. It 
        will be to use to test if an expression is correct and can be 
        added it in the final relation.
    force_columns: list, optional
        Columns to use to generate the final relation.

    Returns
    -------
    str
        The SQL final relation.
        """
        # The First step is to find the Max Floor
        all_imputations_grammar = []
        force_columns_copy = [col for col in force_columns]
        if not (force_columns):
            force_columns = [col for col in self._VERTICAPY_VARIABLES_["columns"]]
        for column in force_columns:
            all_imputations_grammar += [
                [transformation[0] for transformation in self[column].transformations]
            ]
        for column in transformations:
            all_imputations_grammar += [transformations[column]]
        max_transformation_floor = len(max(all_imputations_grammar, key=len))
        # We complete all virtual columns transformations which do not have enough floors
        # with the identity transformation x :-> x in order to generate the correct SQL query
        for imputations in all_imputations_grammar:
            diff = max_transformation_floor - len(imputations)
            if diff > 0:
                imputations += ["{}"] * diff
        # We find the position of all filters in order to write them at the correct floor
        where_positions = [item[1] for item in self._VERTICAPY_VARIABLES_["where"]]
        max_where_pos = max(where_positions + [0])
        all_where = [[] for item in range(max_where_pos + 1)]
        for i in range(0, len(self._VERTICAPY_VARIABLES_["where"])):
            all_where[where_positions[i]] += [self._VERTICAPY_VARIABLES_["where"][i][0]]
        all_where = [
            " AND ".join([f"({elem})" for elem in condition]) for condition in all_where
        ]
        for i in range(len(all_where)):
            if all_where[i] != "":
                all_where[i] = f" WHERE {all_where[i]}"
        # We compute the first floor
        columns = force_columns + [column for column in transformations]
        first_values = [item[0] for item in all_imputations_grammar]
        transformations_first_floor = False
        for i in range(0, len(first_values)):
            if (first_values[i] != "___VERTICAPY_UNDEFINED___") and (
                first_values[i] != columns[i]
            ):
                first_values[i] = f"{first_values[i]} AS {columns[i]}"
                transformations_first_floor = True
        if (transformations_first_floor) or (
            self._VERTICAPY_VARIABLES_["allcols_ind"] != len(first_values)
        ):
            table = f"""
                SELECT 
                    {', '.join(first_values)} 
                FROM {self._VERTICAPY_VARIABLES_['main_relation']}"""
        else:
            table = f"""SELECT * FROM {self._VERTICAPY_VARIABLES_["main_relation"]}"""
        # We compute the other floors
        for i in range(1, max_transformation_floor):
            values = [item[i] for item in all_imputations_grammar]
            for j in range(0, len(values)):
                if values[j] == "{}":
                    values[j] = columns[j]
                elif values[j] != "___VERTICAPY_UNDEFINED___":
                    values_str = values[j].replace("{}", columns[j])
                    values[j] = f"{values_str} AS {columns[j]}"
            table = f"SELECT {', '.join(values)} FROM ({table}) VERTICAPY_SUBTABLE"
            if len(all_where) > i - 1:
                table += all_where[i - 1]
            if (i - 1) in self._VERTICAPY_VARIABLES_["order_by"]:
                table += self._VERTICAPY_VARIABLES_["order_by"][i - 1]
        where_final = (
            all_where[max_transformation_floor - 1]
            if (len(all_where) > max_transformation_floor - 1)
            else ""
        )
        # Only the last order_by matters as the order_by will never change
        # the final relation
        try:
            order_final = self._VERTICAPY_VARIABLES_["order_by"][
                max_transformation_floor - 1
            ]
        except:
            order_final = ""
        for vml_undefined in [
            ", ___VERTICAPY_UNDEFINED___",
            "___VERTICAPY_UNDEFINED___, ",
            "___VERTICAPY_UNDEFINED___",
        ]:
            table = table.replace(vml_undefined, "")
        random_func = current_random()
        split = f", {random_func} AS __verticapy_split__" if (split) else ""
        if (where_final == "") and (order_final == ""):
            if split:
                table = f"SELECT *{split} FROM ({table}) VERTICAPY_SUBTABLE"
            table = f"({table}) VERTICAPY_SUBTABLE"
        else:
            table = f"({table}) VERTICAPY_SUBTABLE{where_final}{order_final}"
            table = f"(SELECT *{split} FROM {table}) VERTICAPY_SUBTABLE"
        if (self._VERTICAPY_VARIABLES_["exclude_columns"]) and not (split):
            if not (force_columns_copy):
                force_columns_copy = self.get_columns()
            force_columns_copy = ", ".join(force_columns_copy)
            table = f"""
                (SELECT 
                    {force_columns_copy}{split} 
                FROM {table}) VERTICAPY_SUBTABLE"""
        main_relation = self._VERTICAPY_VARIABLES_["main_relation"]
        all_main_relation = f"(SELECT * FROM {main_relation}) VERTICAPY_SUBTABLE"
        table = table.replace(all_main_relation, main_relation)
        return table

    def __get_catalog_value__(
        self, column: str = "", key: str = "", method: str = "", columns: list = []
    ):
        """
    VERTICAPY stores the already computed aggregations to avoid useless 
    computations. This method returns the stored aggregation if it was already 
    computed.
        """
        if not (OPTIONS["cache"]):
            return "VERTICAPY_NOT_PRECOMPUTED"
        if column == "VERTICAPY_COUNT":
            if self._VERTICAPY_VARIABLES_["count"] < 0:
                return "VERTICAPY_NOT_PRECOMPUTED"
            total = self._VERTICAPY_VARIABLES_["count"]
            if not (isinstance(total, (int, float))):
                return "VERTICAPY_NOT_PRECOMPUTED"
            return total
        elif method:
            method = verticapy_agg_name(method.lower())
            if columns[1] in self[columns[0]].catalog[method]:
                return self[columns[0]].catalog[method][columns[1]]
            else:
                return "VERTICAPY_NOT_PRECOMPUTED"
        key = verticapy_agg_name(key.lower())
        column = self.format_colnames(column)
        try:
            if (key == "approx_unique") and ("unique" in self[column].catalog):
                key = "unique"
            result = (
                "VERTICAPY_NOT_PRECOMPUTED"
                if key not in self[column].catalog
                else self[column].catalog[key]
            )
        except:
            result = "VERTICAPY_NOT_PRECOMPUTED"
        if result != result:
            result = None
        if ("top" not in key) and (result == None):
            return "VERTICAPY_NOT_PRECOMPUTED"
        return result

    def __get_last_order_by__(self):
        """
    Returns the last column used to sort the data.
        """
        max_pos, order_by = 0, ""
        columns_tmp = [elem for elem in self.get_columns()]
        for column in columns_tmp:
            max_pos = max(max_pos, len(self[column].transformations) - 1)
        if max_pos in self._VERTICAPY_VARIABLES_["order_by"]:
            order_by = self._VERTICAPY_VARIABLES_["order_by"][max_pos]
        return order_by

    def __get_sort_syntax__(self, columns: list):
        """
    Returns the SQL syntax to use to sort the input columns.
        """
        if not (columns):
            return ""
        if isinstance(columns, dict):
            order_by = []
            for col in columns:
                column_name = self.format_colnames(col)
                if columns[col].lower() not in ("asc", "desc"):
                    warning_message = (
                        f"Method of {column_name} must be in (asc, desc), "
                        f"found '{columns[col].lower()}'\nThis column was ignored."
                    )
                    warnings.warn(warning_message, Warning)
                else:
                    order_by += [f"{column_name} {columns[col].upper()}"]
        else:
            order_by = [quote_ident(col) for col in columns]
        return f" ORDER BY {', '.join(order_by)}"

    def __isexternal__(self):
        """
    Returns true if it is an external vDataFrame.
        """
        return self._VERTICAPY_VARIABLES_["external"]

    def __update_catalog__(
        self,
        values: dict = {},
        erase: bool = False,
        columns: list = [],
        matrix: str = "",
        column: str = "",
    ):
        """
    VERTICAPY stores the already computed aggregations to avoid useless 
    computations. This method stores the input aggregation in the vColumn catalog.
        """
        columns = self.format_colnames(columns)
        agg_dict = {
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
        if erase:
            if not (columns):
                columns = self.get_columns()
            for column in columns:
                self[column].catalog = copy.deepcopy(agg_dict)
            self._VERTICAPY_VARIABLES_["count"] = -1
        elif matrix:
            matrix = verticapy_agg_name(matrix.lower())
            if matrix in agg_dict:
                for elem in values:
                    val = values[elem]
                    try:
                        val = float(val)
                    except:
                        pass
                    self[column].catalog[matrix][elem] = val
        else:
            columns = [elem for elem in values]
            columns.remove("index")
            for column in columns:
                for i in range(len(values["index"])):
                    key, val = values["index"][i].lower(), values[column][i]
                    if key not in ["listagg"]:
                        key = verticapy_agg_name(key)
                        try:
                            val = float(val)
                            if val - int(val) == 0:
                                val = int(val)
                        except:
                            pass
                        if val != val:
                            val = None
                        self[column].catalog[key] = val

    def __vDataFrameSQL__(self, table: str, func: str, history: str):
        """
    This method is to use to build a vDataFrame based on a relation
        """
        schema = self._VERTICAPY_VARIABLES_["schema"]
        history = self._VERTICAPY_VARIABLES_["history"] + [history]
        saving = self._VERTICAPY_VARIABLES_["saving"]
        return vDataFrameSQL(table, func, schema, history, saving)

    #
    # Methods used to check & format the inputs
    #

    def format_colnames(
        self,
        *argv,
        columns: Union[str, list, dict] = [],
        expected_nb_of_cols: Union[int, list] = [],
        raise_error: bool = True,
    ):
        """
    Method used to format the input columns by using the vDataFrame columns' names.

    Parameters
    ----------
    *argv: str / list / dict, optional
        List of columns' names to format. It allows to use as input multiple
        objects and to get all of them formatted.
        Example: self.format_colnames(x0, x1, x2) will return x0_f, x1_f, 
        x2_f where xi_f represents xi correctly formatted.
    columns: str / list / dict, optional
        List of columns' names to format.
    expected_nb_of_cols: int / list
        [Only used for the function first argument]
        List of the expected number of columns.
        Example: If expected_nb_of_cols is set to [2, 3], the parameters
        'columns' or the first argument of argv should have exactly 2 or
        3 elements. Otherwise, the function will raise an error.
    raise_error: bool, optional
        If set to True and if there is an error, it will be raised.

    Returns
    -------
    str / list
        Formatted columns' names.
        """
        from verticapy.stats._utils import levenshtein

        if argv:
            result = []
            for arg in argv:
                result += [self.format_colnames(columns=arg, raise_error=raise_error)]
            if len(argv) == 1:
                result = result[0]
        else:
            if not (columns) or isinstance(columns, (int, float)):
                return copy.deepcopy(columns)
            if raise_error:
                if isinstance(columns, str):
                    cols_to_check = [columns]
                else:
                    cols_to_check = copy.deepcopy(columns)
                all_columns = self.get_columns()
                for column in cols_to_check:
                    result = []
                    if column not in all_columns:
                        min_distance, min_distance_op = 1000, ""
                        is_error = True
                        for col in all_columns:
                            if quote_ident(column).lower() == quote_ident(col).lower():
                                is_error = False
                                break
                            else:
                                ldistance = levenshtein(column, col)
                                if ldistance < min_distance:
                                    min_distance, min_distance_op = ldistance, col
                        if is_error:
                            error_message = (
                                f"The Virtual Column '{column}' doesn't exist."
                            )
                            if min_distance < 10:
                                error_message += f"\nDid you mean '{min_distance_op}' ?"
                            raise MissingColumn(error_message)

            if isinstance(columns, str):
                result = columns
                vdf_columns = self.get_columns()
                for col in vdf_columns:
                    if quote_ident(columns).lower() == quote_ident(col).lower():
                        result = col
                        break
            elif isinstance(columns, dict):
                result = {}
                for col in columns:
                    key = self.format_colnames(col, raise_error=raise_error)
                    result[key] = columns[col]
            else:
                result = []
                for col in columns:
                    result += [self.format_colnames(col, raise_error=raise_error)]
        if raise_error:
            if isinstance(expected_nb_of_cols, int):
                expected_nb_of_cols = [expected_nb_of_cols]
            if len(expected_nb_of_cols) > 0:
                if len(argv) > 0:
                    columns = argv[0]
                n = len(columns)
                if n not in expected_nb_of_cols:
                    x = "|".join([str(nb) for nb in expected_nb_of_cols])
                    raise ParameterError(
                        f"The number of Virtual Columns expected is [{x}], found {n}."
                    )
        return result

    def is_colname_in(self, column: str):
        """
    Method used to check if the input column name is used by the vDataFrame.
    If not, the function raises an error.

    Parameters
    ----------
    column: str
        Input column.

    Returns
    -------
    bool
        True if the column is used by the vDataFrame
        False otherwise.
        """
        columns = self.get_columns()
        column = quote_ident(column).lower()
        for col in columns:
            if column == quote_ident(col).lower():
                return True
        return False

    def get_nearest_column(self, column: str):
        """
    Method used to find the nearest column's name to the input one.

    Parameters
    ----------
    column: str
        Input column.

    Returns
    -------
    tuple
        (nearest column, levenstein distance)
        """
        from verticapy.stats._utils import levenshtein

        columns = self.get_columns()
        col = column.replace('"', "").lower()
        result = (columns[0], levenshtein(col, columns[0].replace('"', "").lower()))
        if len(columns) == 1:
            return result
        for col in columns:
            if col != result[0]:
                current_col = col.replace('"', "").lower()
                d = levenshtein(current_col, col)
                if result[1] > d:
                    result = (col, d)
        return result

    #
    # Interactive display
    #

    def idisplay(self):
        """This method displays the interactive table. It is used when 
        you don't want to activate interactive table for all vDataFrames."""
        return display(HTML(self.copy()._repr_html_(interactive=True)))

    #
    # Methods
    #

    @save_verticapy_logs
    def abs(self, columns: Union[str, list] = []):
        """
    Applies the absolute value function to all input vColumns. 

    Parameters
    ----------
    columns: str / list, optional
        List of the vColumns names. If empty, all numerical vColumns will 
        be used.

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame.apply    : Applies functions to the input vColumns.
    vDataFrame.applymap : Applies a function to all vColumns.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns = self.numcol() if not (columns) else self.format_colnames(columns)
        func = {}
        for column in columns:
            if not (self[column].isbool()):
                func[column] = "ABS({})"
        return self.apply(func)

    @save_verticapy_logs
    def add_duplicates(self, weight: Union[int, str], use_gcd: bool = True):
        """
    Duplicates the vDataFrame using the input weight.

    Parameters
    ----------
    weight: str / integer
        vColumn or integer representing the weight.
    use_gcd: bool
        If set to True, uses the GCD (Greatest Common Divisor) to reduce all 
        common weights to avoid unnecessary duplicates.

    Returns
    -------
    vDataFrame
        the output vDataFrame
        """
        if isinstance(weight, str):
            weight = self.format_colnames(weight)
            assert self[weight].category() == "int", TypeError(
                "The weight vColumn category must be "
                f"'integer', found {self[weight].category()}."
            )
            L = sorted(self[weight].distinct())
            gcd, max_value, n = L[0], L[-1], len(L)
            assert gcd >= 0, ValueError(
                "The weight vColumn must only include positive integers."
            )
            if use_gcd:
                if gcd != 1:
                    for i in range(1, n):
                        if gcd != 1:
                            gcd = math.gcd(gcd, L[i])
                        else:
                            break
            else:
                gcd = 1
            columns = self.get_columns(exclude_columns=[weight])
            vdf = self.search(self[weight] != 0, usecols=columns)
            for i in range(2, int(max_value / gcd) + 1):
                vdf = vdf.append(
                    self.search((self[weight] / gcd) >= i, usecols=columns)
                )
        else:
            assert weight >= 2 and isinstance(weight, int), ValueError(
                "The weight must be an integer greater or equal to 2."
            )
            vdf = self.copy()
            for i in range(2, weight + 1):
                vdf = vdf.append(self)
        return vdf

    @save_verticapy_logs
    def analytic(
        self,
        func: str,
        columns: Union[str, list] = [],
        by: Union[str, list] = [],
        order_by: Union[dict, list] = [],
        name: str = "",
        offset: int = 1,
        x_smoothing: float = 0.5,
        add_count: bool = True,
    ):
        """
    Adds a new vColumn to the vDataFrame by using an advanced analytical 
    function on one or two specific vColumns.

    \u26A0 Warning : Some analytical functions can make the vDataFrame 
                     structure more resource intensive. It is best to check 
                     the structure of the vDataFrame using the 'current_relation' 
                     method and to save it using the 'to_db' method with 
                     the parameters 'inplace = True' and 
                     'relation_type = table'

    Parameters
    ----------
    func: str
        Function to apply.
            aad          : average absolute deviation
            beta         : Beta Coefficient between 2 vColumns
            count        : number of non-missing elements
            corr         : Pearson's correlation between 2 vColumns
            cov          : covariance between 2 vColumns
            dense_rank   : dense rank
            ema          : exponential moving average
            first_value  : first non null lead
            iqr          : interquartile range
            kurtosis     : kurtosis
            jb           : Jarque-Bera index 
            lead         : next element
            lag          : previous element
            last_value   : first non null lag
            mad          : median absolute deviation
            max          : maximum
            mean         : average
            median       : median
            min          : minimum
            mode         : most occurent element
            q%           : q quantile (ex: 50% for the median)
            pct_change   : ratio between the current value and the previous one
            percent_rank : percent rank
            prod         : product
            range        : difference between the max and the min
            rank         : rank
            row_number   : row number
            sem          : standard error of the mean
            skewness     : skewness
            sum          : sum
            std          : standard deviation
            unique       : cardinality (count distinct)
            var          : variance
                Other analytical functions could work if it is part of 
                the DB version you are using.
    columns: str / list, optional
        Input vColumns. It can be a list of one or two elements.
    by: str / list, optional
        vColumns used in the partition.
    order_by: dict / list, optional
        List of the vColumns to use to sort the data using asc order or
        dictionary of all sorting methods. For example, to sort by "column1"
        ASC and "column2" DESC, write {"column1": "asc", "column2": "desc"}
    name: str, optional
        Name of the new vColumn. If empty a default name based on the other
        parameters will be generated.
    offset: int, optional
        Lead/Lag offset if parameter 'func' is the function 'lead'/'lag'.
    x_smoothing: float, optional
        The smoothing parameter of the 'ema' if the function is 'ema'. It must be in [0;1]
    add_count: bool, optional
        If the function is the 'mode' and this parameter is True then another column will 
        be added to the vDataFrame with the mode number of occurences.

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame.eval    : Evaluates a customized expression.
    vDataFrame.rolling : Computes a customized moving window.
        """
        if isinstance(by, str):
            by = [by]
        if isinstance(order_by, str):
            order_by = [order_by]
        if isinstance(columns, str):
            if columns:
                columns = [columns]
            else:
                columns = []
        columns, by = self.format_colnames(columns, by)
        by_name = ["by"] + by if (by) else []
        by_order = ["order_by"] + [elem for elem in order_by] if (order_by) else []
        if not (name):
            name = gen_name([func] + columns + by_name + by_order)
        func = func.lower()
        by = ", ".join(by)
        by = f"PARTITION BY {by}" if (by) else ""
        order_by = self.__get_sort_syntax__(order_by)
        func = verticapy_agg_name(func.lower(), method="vertica")
        if func in (
            "max",
            "min",
            "avg",
            "sum",
            "count",
            "stddev",
            "median",
            "variance",
            "unique",
            "top",
            "kurtosis",
            "skewness",
            "mad",
            "aad",
            "range",
            "prod",
            "jb",
            "iqr",
            "sem",
        ) or ("%" in func):
            if order_by and not (OPTIONS["print_info"]):
                print(
                    f"\u26A0 '{func}' analytic method doesn't need an "
                    "order by clause, it was ignored"
                )
            elif not (columns):
                raise MissingColumn(
                    "The parameter 'column' must be a vDataFrame Column "
                    f"when using analytic method '{func}'"
                )
            if func in ("skewness", "kurtosis", "aad", "mad", "jb"):
                random_nb = random.randint(0, 10000000)
                column_str = columns[0].replace('"', "")
                mean_name = f"{column_str}_mean_{random_nb}"
                median_name = f"{column_str}_median_{random_nb}"
                std_name = f"{column_str}_std_{random_nb}"
                count_name = f"{column_str}_count_{random_nb}"
                all_cols = [elem for elem in self._VERTICAPY_VARIABLES_["columns"]]
                if func == "mad":
                    self.eval(median_name, f"MEDIAN({columns[0]}) OVER ({by})")
                else:
                    self.eval(mean_name, f"AVG({columns[0]}) OVER ({by})")
                if func not in ("aad", "mad"):
                    self.eval(std_name, f"STDDEV({columns[0]}) OVER ({by})")
                    self.eval(count_name, f"COUNT({columns[0]}) OVER ({by})")
                if func == "kurtosis":
                    self.eval(
                        name,
                        f"""AVG(POWER(({columns[0]} - {mean_name}) 
                          / NULLIFZERO({std_name}), 4)) OVER ({by}) 
                          * POWER({count_name}, 2) 
                          * ({count_name} + 1) 
                          / NULLIFZERO(({count_name} - 1) 
                          * ({count_name} - 2) 
                          * ({count_name} - 3)) 
                          - 3 * POWER({count_name} - 1, 2) 
                          / NULLIFZERO(({count_name} - 2) 
                          * ({count_name} - 3))""",
                    )
                elif func == "skewness":
                    self.eval(
                        name,
                        f"""AVG(POWER(({columns[0]} - {mean_name}) 
                         / NULLIFZERO({std_name}), 3)) OVER ({by}) 
                         * POWER({count_name}, 2) 
                         / NULLIFZERO(({count_name} - 1) 
                         * ({count_name} - 2))""",
                    )
                elif func == "jb":
                    self.eval(
                        name,
                        f"""{count_name} / 6 * (POWER(AVG(POWER(({columns[0]} 
                          - {mean_name}) / NULLIFZERO({std_name}), 3)) OVER ({by}) 
                          * POWER({count_name}, 2) / NULLIFZERO(({count_name} - 1) 
                          * ({count_name} - 2)), 2) + POWER(AVG(POWER(({columns[0]} 
                          - {mean_name}) / NULLIFZERO({std_name}), 4)) OVER ({by}) 
                          * POWER({count_name}, 2) * ({count_name} + 1) 
                          / NULLIFZERO(({count_name} - 1) * ({count_name} - 2) 
                          * ({count_name} - 3)) - 3 * POWER({count_name} - 1, 2) 
                          / NULLIFZERO(({count_name} - 2) * ({count_name} - 3)), 2) / 4)""",
                    )
                elif func == "aad":
                    self.eval(
                        name, f"AVG(ABS({columns[0]} - {mean_name})) OVER ({by})",
                    )
                elif func == "mad":
                    self.eval(
                        name, f"AVG(ABS({columns[0]} - {median_name})) OVER ({by})",
                    )
            elif func == "top":
                if not (by):
                    by_str = f"PARTITION BY {columns[0]}"
                else:
                    by_str = f"{by}, {columns[0]}"
                self.eval(name, f"ROW_NUMBER() OVER ({by_str})")
                if add_count:
                    name_str = name.replace('"', "")
                    self.eval(
                        f"{name_str}_count",
                        f"NTH_VALUE({name}, 1) OVER ({by} ORDER BY {name} DESC)",
                    )
                self[name].apply(
                    f"NTH_VALUE({columns[0]}, 1) OVER ({by} ORDER BY {{}} DESC)"
                )
            elif func == "unique":
                self.eval(
                    name,
                    f"""DENSE_RANK() OVER ({by} ORDER BY {columns[0]} ASC) 
                      + DENSE_RANK() OVER ({by} ORDER BY {columns[0]} DESC) - 1""",
                )
            elif "%" == func[-1]:
                try:
                    x = float(func[0:-1]) / 100
                except:
                    raise FunctionError(
                        f"The aggregate function '{fun}' doesn't exist. "
                        "If you want to compute the percentile x of the "
                        "element please write 'x%' with x > 0. Example: "
                        "50% for the median."
                    )
                self.eval(
                    name,
                    f"PERCENTILE_CONT({x}) WITHIN GROUP(ORDER BY {columns[0]}) OVER ({by})",
                )
            elif func == "range":
                self.eval(
                    name,
                    f"MAX({columns[0]}) OVER ({by}) - MIN({columns[0]}) OVER ({by})",
                )
            elif func == "iqr":
                self.eval(
                    name,
                    f"""PERCENTILE_CONT(0.75) WITHIN GROUP(ORDER BY {columns[0]}) OVER ({by}) 
                      - PERCENTILE_CONT(0.25) WITHIN GROUP(ORDER BY {columns[0]}) OVER ({by})""",
                )
            elif func == "sem":
                self.eval(
                    name,
                    f"STDDEV({columns[0]}) OVER ({by}) / SQRT(COUNT({columns[0]}) OVER ({by}))",
                )
            elif func == "prod":
                self.eval(
                    name,
                    f"""DECODE(ABS(MOD(SUM(CASE 
                                            WHEN {columns[0]} < 0 
                                            THEN 1 ELSE 0 END) 
                                       OVER ({by}), 2)), 0, 1, -1) 
                     * POWER(10, SUM(LOG(ABS({columns[0]}))) 
                                 OVER ({by}))""",
                )
            else:
                self.eval(name, f"{func.upper()}({columns[0]}) OVER ({by})")
        elif func in (
            "lead",
            "lag",
            "row_number",
            "percent_rank",
            "dense_rank",
            "rank",
            "first_value",
            "last_value",
            "exponential_moving_average",
            "pct_change",
        ):
            if not (columns) and func in (
                "lead",
                "lag",
                "first_value",
                "last_value",
                "pct_change",
            ):
                raise ParameterError(
                    "The parameter 'columns' must be a vDataFrame column when "
                    f"using analytic method '{func}'"
                )
            elif (columns) and func not in (
                "lead",
                "lag",
                "first_value",
                "last_value",
                "pct_change",
                "exponential_moving_average",
            ):
                raise ParameterError(
                    "The parameter 'columns' must be empty when using analytic"
                    f" method '{func}'"
                )
            if (by) and (order_by):
                order_by = f" {order_by}"
            if func in ("lead", "lag"):
                info_param = f", {offset}"
            elif func in ("last_value", "first_value"):
                info_param = " IGNORE NULLS"
            elif func == "exponential_moving_average":
                info_param = f", {x_smoothing}"
            else:
                info_param = ""
            if func == "pct_change":
                self.eval(
                    name, f"{columns[0]} / (LAG({columns[0]}) OVER ({by}{order_by}))",
                )
            else:
                columns0 = columns[0] if (columns) else ""
                self.eval(
                    name,
                    f"{func.upper()}({columns0}{info_param}) OVER ({by}{order_by})",
                )
        elif func in ("corr", "cov", "beta"):
            if order_by:
                print(
                    f"\u26A0 '{func}' analytic method doesn't need an "
                    "order by clause, it was ignored"
                )
            assert len(columns) == 2, MissingColumn(
                "The parameter 'columns' includes 2 vColumns when using "
                f"analytic method '{func}'"
            )
            if columns[0] == columns[1]:
                if func == "cov":
                    expr = f"VARIANCE({columns[0]}) OVER ({by})"
                else:
                    expr = 1
            else:
                if func == "corr":
                    den = f" / (STDDEV({columns[0]}) OVER ({by}) * STDDEV({columns[1]}) OVER ({by}))"
                elif func == "beta":
                    den = f" / (VARIANCE({columns[1]}) OVER ({by}))"
                else:
                    den = ""
                expr = f"""
                    (AVG({columns[0]} * {columns[1]}) OVER ({by}) 
                   - AVG({columns[0]}) OVER ({by}) 
                   * AVG({columns[1]}) OVER ({by})){den}"""
            self.eval(name, expr)
        else:
            try:
                self.eval(
                    name,
                    f"{func.upper()}({columns[0]}{info_param}) OVER ({by}{order_by})",
                )
            except:
                raise FunctionError(
                    f"The aggregate function '{func}' doesn't exist or is not "
                    "managed by the 'analytic' method. If you want more "
                    "flexibility use the 'eval' method."
                )
        if func in ("kurtosis", "skewness", "jb"):
            self._VERTICAPY_VARIABLES_["exclude_columns"] += [
                quote_ident(mean_name),
                quote_ident(std_name),
                quote_ident(count_name),
            ]
        elif func == "aad":
            self._VERTICAPY_VARIABLES_["exclude_columns"] += [quote_ident(mean_name)]
        elif func == "mad":
            self._VERTICAPY_VARIABLES_["exclude_columns"] += [quote_ident(median_name)]
        return self

    @save_verticapy_logs
    def append(
        self,
        input_relation: Union[str, str_sql],
        expr1: Union[str, list] = [],
        expr2: Union[str, list] = [],
        union_all: bool = True,
    ):
        """
    Merges the vDataFrame with another one or an input relation and returns 
    a new vDataFrame.

    Parameters
    ----------
    input_relation: str / vDataFrame
        Relation to use to do the merging.
    expr1: str / list, optional
        List of pure-SQL expressions from the current vDataFrame to use during merging.
        For example, 'CASE WHEN "column" > 3 THEN 2 ELSE NULL END' and 'POWER("column", 2)' 
        will work. If empty, all vDataFrame vColumns will be used. Aliases are 
        recommended to avoid auto-naming.
    expr2: str / list, optional
        List of pure-SQL expressions from the input relation to use during the merging.
        For example, 'CASE WHEN "column" > 3 THEN 2 ELSE NULL END' and 'POWER("column", 2)' 
        will work. If empty, all input relation columns will be used. Aliases are 
        recommended to avoid auto-naming.
    union_all: bool, optional
        If set to True, the vDataFrame will be merged with the input relation using an
        'UNION ALL' instead of an 'UNION'.

    Returns
    -------
    vDataFrame
       vDataFrame of the Union

    See Also
    --------
    vDataFrame.groupby : Aggregates the vDataFrame.
    vDataFrame.join    : Joins the vDataFrame with another relation.
    vDataFrame.sort    : Sorts the vDataFrame.
        """
        if isinstance(expr1, str):
            expr1 = [expr1]
        if isinstance(expr2, str):
            expr2 = [expr2]
        first_relation = self.__genSQL__()
        if isinstance(input_relation, str):
            second_relation = input_relation
        elif isinstance(input_relation, vDataFrame):
            second_relation = input_relation.__genSQL__()
        columns = ", ".join(self.get_columns()) if not (expr1) else ", ".join(expr1)
        columns2 = columns if not (expr2) else ", ".join(expr2)
        union = "UNION" if not (union_all) else "UNION ALL"
        table = f"""
            (SELECT 
                {columns} 
             FROM {first_relation}) 
             {union} 
            (SELECT 
                {columns2} 
             FROM {second_relation})"""
        return self.__vDataFrameSQL__(
            f"({table}) append_table",
            self._VERTICAPY_VARIABLES_["input_relation"],
            "[Append]: Union of two relations",
        )

    @save_verticapy_logs
    def apply(self, func: dict):
        """
    Applies each function of the dictionary to the input vColumns.

    Parameters
     ----------
     func: dict
        Dictionary of functions.
        The dictionary must be like the following: 
        {column1: func1, ..., columnk: funck}. Each function variable must
        be composed of two flower brackets {}. For example to apply the 
        function: x -> x^2 + 2 use "POWER({}, 2) + 2".

     Returns
     -------
     vDataFrame
        self

    See Also
    --------
    vDataFrame.applymap : Applies a function to all vColumns.
    vDataFrame.eval     : Evaluates a customized expression.
        """
        func = self.format_colnames(func)
        for column in func:
            self[column].apply(func[column])
        return self

    @save_verticapy_logs
    def applymap(self, func: str, numeric_only: bool = True):
        """
    Applies a function to all vColumns. 

    Parameters
    ----------
    func: str
        The function.
        The function variable must be composed of two flower brackets {}. 
        For example to apply the function: x -> x^2 + 2 use "POWER({}, 2) + 2".
    numeric_only: bool, optional
        If set to True, only the numerical columns will be used.

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame.apply : Applies functions to the input vColumns.
        """
        function = {}
        columns = self.numcol() if numeric_only else self.get_columns()
        for column in columns:
            function[column] = (
                func if not (self[column].isbool()) else func.replace("{}", "{}::int")
            )
        return self.apply(function)

    @save_verticapy_logs
    def interpolate(
        self,
        ts: str,
        rule: Union[str, datetime.timedelta],
        method: dict = {},
        by: Union[str, list] = [],
    ):
        """
    Computes a regular time interval vDataFrame by interpolating the missing 
    values using different techniques.

    Parameters
    ----------
    ts: str
        TS (Time Series) vColumn to use to order the data. The vColumn type 
        must be date like (date, datetime, timestamp...)
    rule: str / time
        Interval used to create the time slices. The final interpolation is 
        divided by these intervals. For example, specifying '5 minutes' 
        creates records separated by time intervals of '5 minutes' 
    method: dict, optional
        Dictionary, with the following format, of interpolation methods:
        {"column1": "interpolation1" ..., "columnk": "interpolationk"}
        Interpolation methods must be one of the following:
            bfill  : Interpolates with the final value of the time slice.
            ffill  : Interpolates with the first value of the time slice.
            linear : Linear interpolation.
    by: str / list, optional
        vColumns used in the partition.

    Returns
    -------
    vDataFrame
        object result of the interpolation.

    See Also
    --------
    vDataFrame[].fillna  : Fills the vColumn missing values.
    vDataFrame[].slice   : Slices the vColumn.
        """
        if isinstance(by, str):
            by = [by]
        method, ts, by = self.format_colnames(method, ts, by)
        all_elements = []
        for column in method:
            assert method[column] in (
                "bfill",
                "backfill",
                "pad",
                "ffill",
                "linear",
            ), ParameterError(
                "Each element of the 'method' dictionary must be "
                "in bfill|backfill|pad|ffill|linear"
            )
            if method[column] in ("bfill", "backfill"):
                func, interp = "TS_LAST_VALUE", "const"
            elif method[column] in ("pad", "ffill"):
                func, interp = "TS_FIRST_VALUE", "const"
            else:
                func, interp = "TS_FIRST_VALUE", "linear"
            all_elements += [f"{func}({column}, '{interp}') AS {column}"]
        table = f"SELECT {{}} FROM {self.__genSQL__()}"
        tmp_query = [f"slice_time AS {quote_ident(ts)}"]
        tmp_query += [quote_ident(column) for column in by]
        tmp_query += all_elements
        table = table.format(", ".join(tmp_query))
        partition = ""
        if by:
            partition = ", ".join([quote_ident(column) for column in by])
            partition = f"PARTITION BY {partition} "
        table += f""" 
            TIMESERIES slice_time AS '{rule}' 
            OVER ({partition}ORDER BY {quote_ident(ts)}::timestamp)"""
        return self.__vDataFrameSQL__(
            f"({table}) interpolate",
            "interpolate",
            "[interpolate]: The data was resampled",
        )

    asfreq = interpolate

    @save_verticapy_logs
    def astype(self, dtype: dict):
        """
    Converts the vColumns to the input types.

    Parameters
    ----------
    dtype: dict
        Dictionary of the different types. Each key of the dictionary must 
        represent a vColumn. The dictionary must be similar to the 
        following: {"column1": "type1", ... "columnk": "typek"}

    Returns
    -------
    vDataFrame
        self
        """
        for column in dtype:
            self[self.format_colnames(column)].astype(dtype=dtype[column])
        return self

    @save_verticapy_logs
    def bool_to_int(self):
        """
    Converts all booleans vColumns to integers.

    Returns
    -------
    vDataFrame
        self
    
    See Also
    --------
    vDataFrame.astype : Converts the vColumns to the input types.
        """
        columns = self.get_columns()
        for column in columns:
            if self[column].isbool():
                self[column].astype("int")
        return self

    def catcol(self, max_cardinality: int = 12):
        """
    Returns the vDataFrame categorical vColumns.
    
    Parameters
    ----------
    max_cardinality: int, optional
        Maximum number of unique values to consider integer vColumns as categorical.

    Returns
    -------
    List
        List of the categorical vColumns names.
    
    See Also
    --------
    vDataFrame.get_columns : Returns a list of names of the vColumns in the vDataFrame.
    vDataFrame.numcol      : Returns a list of names of the numerical vColumns in the 
                             vDataFrame.
        """
        # -#
        columns = []
        for column in self.get_columns():
            if (self[column].category() == "int") and not (self[column].isbool()):
                is_cat = _executeSQL(
                    query=f"""
                        SELECT 
                            /*+LABEL('vDataframe.catcol')*/ 
                            (APPROXIMATE_COUNT_DISTINCT({column}) < {max_cardinality}) 
                        FROM {self.__genSQL__()}""",
                    title="Looking at columns with low cardinality.",
                    method="fetchfirstelem",
                    sql_push_ext=self._VERTICAPY_VARIABLES_["sql_push_ext"],
                    symbol=self._VERTICAPY_VARIABLES_["symbol"],
                )
            elif self[column].category() == "float":
                is_cat = False
            else:
                is_cat = True
            if is_cat:
                columns += [column]
        return columns

    @save_verticapy_logs
    def cdt(
        self,
        columns: Union[str, list] = [],
        max_cardinality: int = 20,
        nbins: int = 10,
        tcdt: bool = True,
        drop_transf_cols: bool = True,
    ):
        """
    Returns the complete disjunctive table of the vDataFrame.
    Numerical features are transformed to categorical using
    the 'discretize' method. Applying PCA on TCDT leads to MCA 
    (Multiple correspondence analysis).

    \u26A0 Warning : This method can become computationally expensive when
                     used with categorical variables with many categories.

    Parameters
    ----------
    columns: str / list, optional
        List of the vColumns names.
    max_cardinality: int, optional
        For any categorical variable, keeps the most frequent categories and 
        merges the less frequent categories into a new unique category.
    nbins: int, optional
        Number of bins used for the discretization (must be > 1).
    tcdt: bool, optional
        If set to True, returns the transformed complete disjunctive table 
        (TCDT). 
    drop_transf_cols: bool, optional
        If set to True, drops the columns used during the transformation.

    Returns
    -------
    vDataFrame
        the CDT relation.
        """
        if isinstance(columns, str):
            columns = [columns]
        if columns:
            columns = self.format_colnames(columns)
        else:
            columns = self.get_columns()
        vdf = self.copy()
        columns_to_drop = []
        for elem in columns:
            if vdf[elem].isbool():
                vdf[elem].astype("int")
            elif vdf[elem].isnum():
                vdf[elem].discretize(nbins=nbins)
                columns_to_drop += [elem]
            elif vdf[elem].isdate():
                vdf[elem].drop()
            else:
                vdf[elem].discretize(method="topk", k=max_cardinality)
                columns_to_drop += [elem]
        new_columns = vdf.get_columns()
        vdf.one_hot_encode(
            columns=columns,
            max_cardinality=max(max_cardinality, nbins) + 2,
            drop_first=False,
        )
        new_columns = vdf.get_columns(exclude_columns=new_columns)
        if drop_transf_cols:
            vdf.drop(columns=columns_to_drop)
        if tcdt:
            for elem in new_columns:
                sum_cat = vdf[elem].sum()
                vdf[elem].apply(f"{{}} / {sum_cat} - 1")
        return vdf

    @save_verticapy_logs
    def chaid(
        self,
        response: str,
        columns: Union[str, list],
        nbins: int = 4,
        method: Literal["smart", "same_width"] = "same_width",
        RFmodel_params: dict = {},
        **kwds,
    ):
        """
    Returns a CHAID (Chi-square Automatic Interaction Detector) tree.
    CHAID is a decision tree technique based on adjusted significance testing 
    (Bonferroni test).

    Parameters
    ----------
    response: str
        Categorical response vColumn.
    columns: str / list
        List of the vColumn names. The maximum number of categories for each
        categorical column is 16; categorical columns with a higher cardinality
        are discarded.
    nbins: int, optional
        Integer in the range [2,16], the number of bins used 
        to discretize the numerical features.
    method: str, optional
        The method with which to discretize the numerical vColumns, 
        one of the following:
            same_width : Computes bins of regular width.
            smart      : Uses a random forest model on a response column to find the best
                interval for discretization.
    RFmodel_params: dict, optional
        Dictionary of the parameters of the random forest model used to compute the best splits 
        when 'method' is 'smart'. If the response column is numerical (but not of type int or bool), 
        this function trains and uses a random forest regressor. Otherwise, this function 
        trains a random forest classifier.
        For example, to train a random forest with 20 trees and a maximum depth of 10, use:
            {"n_estimators": 20, "max_depth": 10}

    Returns
    -------
    memModel
        An independent model containing the result. For more information, see
        learn.memmodel.
        """
        from verticapy.machine_learning._utils import get_match_index

        if "process" not in kwds or kwds["process"]:
            if isinstance(columns, str):
                columns = [columns]
            assert 2 <= nbins <= 16, ParameterError(
                "Parameter 'nbins' must be between 2 and 16, inclusive."
            )
            columns = self.chaid_columns(columns)
            if not (columns):
                raise ValueError("No column to process.")
        idx = 0 if ("node_id" not in kwds) else kwds["node_id"]
        p = self.pivot_table_chi2(response, columns, nbins, method, RFmodel_params)
        categories, split_predictor, is_numerical, chi2 = (
            p["categories"][0],
            p["index"][0],
            p["is_numerical"][0],
            p["chi2"][0],
        )
        split_predictor_idx = get_match_index(
            split_predictor,
            columns
            if "process" not in kwds or kwds["process"]
            else kwds["columns_init"],
        )
        tree = {
            "split_predictor": split_predictor,
            "split_predictor_idx": split_predictor_idx,
            "split_is_numerical": is_numerical,
            "chi2": chi2,
            "is_leaf": False,
            "node_id": idx,
        }
        if is_numerical:
            if categories:
                if ";" in categories[0]:
                    categories = sorted(
                        [float(c.split(";")[1][0:-1]) for c in categories]
                    )
                    ctype = "float"
                else:
                    categories = sorted([int(c) for c in categories])
                    ctype = "int"
            else:
                categories, ctype = [], "int"
        if "process" not in kwds or kwds["process"]:
            classes = self[response].distinct()
        else:
            classes = kwds["classes"]
        if len(columns) == 1:
            if categories:
                if is_numerical:
                    column = "(CASE "
                    for c in categories:
                        column += f"WHEN {split_predictor} <= {c} THEN {c} "
                    column += f"ELSE NULL END)::{ctype} AS {split_predictor}"
                else:
                    column = split_predictor
                result = _executeSQL(
                    query=f"""
                        SELECT 
                            /*+LABEL('vDataframe.chaid')*/ 
                            {split_predictor}, 
                            {response}, 
                            (cnt / SUM(cnt) 
                                OVER (PARTITION BY {split_predictor}))::float 
                                AS proba 
                        FROM 
                            (SELECT 
                                {column}, 
                                {response}, 
                                COUNT(*) AS cnt 
                             FROM {self.__genSQL__()} 
                             WHERE {split_predictor} IS NOT NULL 
                               AND {response} IS NOT NULL 
                             GROUP BY 1, 2) x 
                        ORDER BY 1;""",
                    title="Computing the CHAID tree probability.",
                    method="fetchall",
                    sql_push_ext=self._VERTICAPY_VARIABLES_["sql_push_ext"],
                    symbol=self._VERTICAPY_VARIABLES_["symbol"],
                )
            else:
                result = []
            children = {}
            for c in categories:
                children[c] = {}
                for cl in classes:
                    children[c][cl] = 0.0
            for elem in result:
                children[elem[0]][elem[1]] = elem[2]
            for elem in children:
                idx += 1
                children[elem] = {
                    "prediction": [children[elem][c] for c in children[elem]],
                    "is_leaf": True,
                    "node_id": idx,
                }
            tree["children"] = children
            if "process" not in kwds or kwds["process"]:
                return memModel("CHAID", attributes={"tree": tree, "classes": classes})
            return tree, idx
        else:
            tree["children"] = {}
            columns_tmp = columns.copy()
            columns_tmp.remove(split_predictor)
            for c in categories:
                if is_numerical:
                    vdf = self.search(
                        f"""{split_predictor} <= {c}
                        AND {split_predictor} IS NOT NULL
                        AND {response} IS NOT NULL""",
                        usecols=columns_tmp + [response],
                    )
                else:
                    vdf = self.search(
                        f"""{split_predictor} = '{c}'
                        AND {split_predictor} IS NOT NULL
                        AND {response} IS NOT NULL""",
                        usecols=columns_tmp + [response],
                    )
                tree["children"][c], idx = vdf.chaid(
                    response,
                    columns_tmp,
                    nbins,
                    method,
                    RFmodel_params,
                    process=False,
                    columns_init=columns,
                    classes=classes,
                    node_id=idx + 1,
                )
            if "process" not in kwds or kwds["process"]:
                return memModel("CHAID", attributes={"tree": tree, "classes": classes})
            return tree, idx

    @save_verticapy_logs
    def chaid_columns(self, columns: list = [], max_cardinality: int = 16):
        """
    Function used to simplify the code. It returns the columns picked by
    the CHAID algorithm.

    Parameters
    ----------
    columns: list
        List of the vColumn names.
    max_cardinality: int, optional
        The maximum number of categories for each categorical column. Categorical 
        columns with a higher cardinality are discarded.

    Returns
    -------
    list
        columns picked by the CHAID algorithm
        """
        columns_tmp = columns.copy()
        if not (columns_tmp):
            columns_tmp = self.get_columns()
            remove_cols = []
            for col in columns_tmp:
                if self[col].category() not in ("float", "int", "text") or (
                    self[col].category() == "text"
                    and self[col].nunique() > max_cardinality
                ):
                    remove_cols += [col]
        else:
            remove_cols = []
            columns_tmp = self.format_colnames(columns_tmp)
            for col in columns_tmp:
                if self[col].category() not in ("float", "int", "text") or (
                    self[col].category() == "text"
                    and self[col].nunique() > max_cardinality
                ):
                    remove_cols += [col]
                    if self[col].category() not in ("float", "int", "text"):
                        warning_message = (
                            f"vColumn '{col}' is of category '{self[col].category()}'. "
                            "This method only accepts categorical & numerical inputs. "
                            "This vColumn was ignored."
                        )
                    else:
                        warning_message = (
                            f"vColumn '{col}' has a too high cardinality "
                            f"(> {max_cardinality}). This vColumn was ignored."
                        )
                    warnings.warn(warning_message, Warning)
        for col in remove_cols:
            columns_tmp.remove(col)
        return columns_tmp

    def copy(self):
        """
    Returns a deep copy of the vDataFrame.

    Returns
    -------
    vDataFrame
        The copy of the vDataFrame.
        """
        return copy.deepcopy(self)

    @save_verticapy_logs
    def case_when(self, name: str, *argv):
        """
    Creates a new feature by evaluating some conditions.
    
    Parameters
    ----------
    name: str
        Name of the new feature.
    argv: object
        Infinite Number of Expressions.
        The expression generated will look like:
        even: CASE ... WHEN argv[2 * i] THEN argv[2 * i + 1] ... END
        odd : CASE ... WHEN argv[2 * i] THEN argv[2 * i + 1] ... ELSE argv[n] END

    Returns
    -------
    vDataFrame
        self
    
    See Also
    --------
    vDataFrame[].decode : Encodes the vColumn using a User Defined Encoding.
    vDataFrame.eval : Evaluates a customized expression.
        """
        from verticapy.stats.math.math import case_when

        return self.eval(name=name, expr=case_when(*argv))

    def current_relation(self, reindent: bool = True):
        """
    Returns the current vDataFrame relation.

    Parameters
    ----------
    reindent: bool, optional
        Reindent the text to be more readable. 

    Returns
    -------
    str
        The formatted current vDataFrame relation.
        """
        from verticapy.sql._utils._format import indentSQL

        if reindent:
            return indentSQL(self.__genSQL__())
        else:
            return self.__genSQL__()

    def datecol(self):
        """
    Returns a list of the vColumns of type date in the vDataFrame.

    Returns
    -------
    List
        List of all vColumns of type date.

    See Also
    --------
    vDataFrame.catcol : Returns a list of the categorical vColumns in the vDataFrame.
    vDataFrame.numcol : Returns a list of names of the numerical vColumns in the 
                        vDataFrame.
        """
        columns = []
        cols = self.get_columns()
        for column in cols:
            if self[column].isdate():
                columns += [column]
        return columns

    def del_catalog(self):
        """
    Deletes the current vDataFrame catalog.

    Returns
    -------
    vDataFrame
        self
        """
        self.__update_catalog__(erase=True)
        return self

    @save_verticapy_logs
    def dtypes(self):
        """
    Returns the different vColumns types.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.
        """
        values = {"index": [], "dtype": []}
        for column in self.get_columns():
            values["index"] += [column]
            values["dtype"] += [self[column].ctype()]
        return tablesample(values)

    @save_verticapy_logs
    def duplicated(
        self, columns: Union[str, list] = [], count: bool = False, limit: int = 30
    ):
        """
    Returns the duplicated values.

    Parameters
    ----------
    columns: str / list, optional
        List of the vColumns names. If empty, all vColumns will be selected.
    count: bool, optional
        If set to True, the method will also return the count of each duplicates.
    limit: int, optional
        The limited number of elements to be displayed.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.drop_duplicates : Filters the duplicated values.
        """
        if not (columns):
            columns = self.get_columns()
        elif isinstance(columns, str):
            columns = [columns]
        columns = self.format_colnames(columns)
        columns = ", ".join(columns)
        main_table = f"""
            (SELECT 
                *, 
                ROW_NUMBER() OVER (PARTITION BY {columns}) AS duplicated_index 
             FROM {self.__genSQL__()}) duplicated_index_table 
             WHERE duplicated_index > 1"""
        if count:
            total = _executeSQL(
                query=f"""
                    SELECT 
                        /*+LABEL('vDataframe.duplicated')*/ COUNT(*) 
                    FROM {main_table}""",
                title="Computing the number of duplicates.",
                method="fetchfirstelem",
                sql_push_ext=self._VERTICAPY_VARIABLES_["sql_push_ext"],
                symbol=self._VERTICAPY_VARIABLES_["symbol"],
            )
            return total
        result = to_tablesample(
            query=f"""
                SELECT 
                    {columns},
                    MAX(duplicated_index) AS occurrence 
                FROM {main_table} 
                GROUP BY {columns} 
                ORDER BY occurrence DESC LIMIT {limit}""",
            sql_push_ext=self._VERTICAPY_VARIABLES_["sql_push_ext"],
            symbol=self._VERTICAPY_VARIABLES_["symbol"],
        )
        result.count = _executeSQL(
            query=f"""
                SELECT 
                    /*+LABEL('vDataframe.duplicated')*/ COUNT(*) 
                FROM 
                    (SELECT 
                        {columns}, 
                        MAX(duplicated_index) AS occurrence 
                     FROM {main_table} 
                     GROUP BY {columns}) t""",
            title="Computing the number of distinct duplicates.",
            method="fetchfirstelem",
            sql_push_ext=self._VERTICAPY_VARIABLES_["sql_push_ext"],
            symbol=self._VERTICAPY_VARIABLES_["symbol"],
        )
        return result

    def empty(self):
        """
    Returns True if the vDataFrame is empty.

    Returns
    -------
    bool
        True if the vDataFrame has no vColumns.
        """
        return not (self.get_columns())

    @save_verticapy_logs
    def eval(self, name: str, expr: Union[str, str_sql]):
        """
    Evaluates a customized expression.

    Parameters
    ----------
    name: str
        Name of the new vColumn.
    expr: str
        Expression in pure SQL to use to compute the new feature.
        For example, 'CASE WHEN "column" > 3 THEN 2 ELSE NULL END' and
        'POWER("column", 2)' will work.

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame.analytic : Adds a new vColumn to the vDataFrame by using an advanced 
        analytical function on a specific vColumn.
        """
        if isinstance(expr, str_sql):
            expr = str(expr)
        name = quote_ident(name.replace('"', "_"))
        if self.is_colname_in(name):
            raise NameError(
                f"A vColumn has already the alias {name}.\n"
                "By changing the parameter 'name', you'll "
                "be able to solve this issue."
            )
        try:
            query = f"SELECT {expr} AS {name} FROM {self.__genSQL__()} LIMIT 0"
            ctype = get_data_types(query, name[1:-1].replace("'", "''"),)
        except:
            raise QueryError(
                f"The expression '{expr}' seems to be incorrect.\nBy "
                "turning on the SQL with the 'set_option' function, "
                "you'll print the SQL code generation and probably "
                "see why the evaluation didn't work."
            )
        if not (ctype):
            ctype = "undefined"
        elif (ctype.lower()[0:12] in ("long varbina", "long varchar")) and (
            self._VERTICAPY_VARIABLES_["isflex"]
            or isvmap(expr=f"({query}) VERTICAPY_SUBTABLE", column=name,)
        ):
            category = "vmap"
            ctype = "VMAP(" + "(".join(ctype.split("(")[1:]) if "(" in ctype else "VMAP"
        else:
            category = to_category(ctype=ctype)
        all_cols, max_floor = self.get_columns(), 0
        for column in all_cols:
            column_str = column.replace('"', "")
            if (quote_ident(column) in expr) or (
                re.search(re.compile(f"\\b{column_str}\\b"), expr)
            ):
                max_floor = max(len(self[column].transformations), max_floor)
        transformations = [
            (
                "___VERTICAPY_UNDEFINED___",
                "___VERTICAPY_UNDEFINED___",
                "___VERTICAPY_UNDEFINED___",
            )
            for i in range(max_floor)
        ] + [(expr, ctype, category)]
        new_vColumn = vColumn(name, parent=self, transformations=transformations)
        setattr(self, name, new_vColumn)
        setattr(self, name.replace('"', ""), new_vColumn)
        new_vColumn.init = False
        new_vColumn.init_transf = name
        self._VERTICAPY_VARIABLES_["columns"] += [name]
        self.__add_to_history__(
            f"[Eval]: A new vColumn {name} was added to the vDataFrame."
        )
        return self

    @save_verticapy_logs
    def expected_store_usage(self, unit: str = "b"):
        """
    Returns the vDataFrame expected store usage. 

    Parameters
    ----------
    unit: str, optional
        unit used for the computation
        b : byte
        kb: kilo byte
        gb: giga byte
        tb: tera byte

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.memory_usage : Returns the vDataFrame memory usage.
        """
        if unit.lower() == "kb":
            div_unit = 1024
        elif unit.lower() == "mb":
            div_unit = 1024 * 1024
        elif unit.lower() == "gb":
            div_unit = 1024 * 1024 * 1024
        elif unit.lower() == "tb":
            div_unit = 1024 * 1024 * 1024 * 1024
        else:
            unit, div_unit = "b", 1
        total, total_expected = 0, 0
        columns = self.get_columns()
        values = self.aggregate(func=["count"], columns=columns).transpose().values
        values["index"] = [
            f"expected_size ({unit})",
            f"max_size ({unit})",
            "type",
        ]
        for column in columns:
            ctype = self[column].ctype()
            if (
                (ctype[0:4] == "date")
                or (ctype[0:4] == "time")
                or (ctype[0:8] == "interval")
                or (ctype == "smalldatetime")
            ):
                maxsize, expsize = 8, 8
            elif "int" in ctype:
                maxsize, expsize = 8, self[column].store_usage()
            elif ctype[0:4] == "bool":
                maxsize, expsize = 1, 1
            elif (
                (ctype[0:5] == "float")
                or (ctype[0:6] == "double")
                or (ctype[0:4] == "real")
            ):
                maxsize, expsize = 8, 8
            elif (
                (ctype[0:7] in ("numeric", "decimal"))
                or (ctype[0:6] == "number")
                or (ctype[0:5] == "money")
            ):
                try:
                    size = sum(
                        [
                            int(item)
                            for item in ctype.split("(")[1].split(")")[0].split(",")
                        ]
                    )
                except:
                    size = 38
                maxsize, expsize = size, size
            elif ctype[0:7] == "varchar":
                try:
                    size = int(ctype.split("(")[1].split(")")[0])
                except:
                    size = 80
                maxsize, expsize = size, self[column].store_usage()
            elif (ctype[0:4] == "char") or (ctype[0:3] == "geo") or ("binary" in ctype):
                try:
                    size = int(ctype.split("(")[1].split(")")[0])
                    maxsize, expsize = size, size
                except:
                    if ctype[0:3] == "geo":
                        maxsize, expsize = 10000000, 10000
                    elif "long" in ctype:
                        maxsize, expsize = 32000000, 10000
                    else:
                        maxsize, expsize = 65000, 1000
            elif ctype[0:4] == "uuid":
                maxsize, expsize = 16, 16
            else:
                maxsize, expsize = 80, self[column].store_usage()
            maxsize /= div_unit
            expsize /= div_unit
            values[column] = [expsize, values[column][0] * maxsize, ctype]
            total_expected += values[column][0]
            total += values[column][1]
        values["separator"] = [
            len(columns) * self.shape()[0] / div_unit,
            len(columns) * self.shape()[0] / div_unit,
            "",
        ]
        total += values["separator"][0]
        total_expected += values["separator"][0]
        values["header"] = [
            (sum([len(item) for item in columns]) + len(columns)) / div_unit,
            (sum([len(item) for item in columns]) + len(columns)) / div_unit,
            "",
        ]
        total += values["header"][0]
        total_expected += values["header"][0]
        values["rawsize"] = [total_expected, total, ""]
        return tablesample(values=values).transpose()

    @save_verticapy_logs
    def explain(self, digraph: bool = False):
        """
    Provides information on how Vertica is computing the current vDataFrame
    relation.

    Parameters
    ----------
    digraph: bool, optional
        If set to True, returns only the digraph of the explain plan.

    Returns
    -------
    str
        explain plan
        """
        result = _executeSQL(
            query=f"""
                EXPLAIN 
                SELECT 
                    /*+LABEL('vDataframe.explain')*/ * 
                FROM {self.__genSQL__()}""",
            title="Explaining the Current Relation",
            method="fetchall",
            sql_push_ext=self._VERTICAPY_VARIABLES_["sql_push_ext"],
            symbol=self._VERTICAPY_VARIABLES_["symbol"],
        )
        result = [elem[0] for elem in result]
        result = "\n".join(result)
        if not (digraph):
            result = result.replace("------------------------------\n", "")
            result = result.replace("\\n", "\n\t")
            result = result.replace(", ", ",").replace(",", ", ").replace("\n}", "}")
        else:
            result = "digraph G {" + result.split("digraph G {")[1]
        return result

    @save_verticapy_logs
    def fillna(self, val: dict = {}, method: dict = {}, numeric_only: bool = False):
        """
    Fills the vColumns missing elements using specific rules.

    Parameters
    ----------
    val: dict, optional
        Dictionary of values. The dictionary must be similar to the following:
        {"column1": val1 ..., "columnk": valk}. Each key of the dictionary must
        be a vColumn. The missing values of the input vColumns will be replaced
        by the input value.
    method: dict, optional
        Method to use to impute the missing values.
            auto    : Mean for the numerical and Mode for the categorical vColumns.
            mean    : Average.
            median  : Median.
            mode    : Mode (most occurent element).
            0ifnull : 0 when the vColumn is null, 1 otherwise.
                More Methods are available on the vDataFrame[].fillna method.
    numeric_only: bool, optional
        If parameters 'val' and 'method' are empty and 'numeric_only' is set
        to True then all numerical vColumns will be imputed by their average.
        If set to False, all categorical vColumns will be also imputed by their
        mode.

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame[].fillna : Fills the vColumn missing values. This method is more 
        complete than the vDataFrame.fillna method by allowing more parameters.
        """
        print_info = OPTIONS["print_info"]
        OPTIONS["print_info"] = False
        try:
            if not (val) and not (method):
                cols = self.get_columns()
                for column in cols:
                    if numeric_only:
                        if self[column].isnum():
                            self[column].fillna(method="auto")
                    else:
                        self[column].fillna(method="auto")
            else:
                for column in val:
                    self[self.format_colnames(column)].fillna(val=val[column])
                for column in method:
                    self[self.format_colnames(column)].fillna(method=method[column],)
            return self
        finally:
            OPTIONS["print_info"] = print_info

    @save_verticapy_logs
    def flat_vmap(
        self,
        vmap_col: Union[str, list] = [],
        limit: int = 100,
        exclude_columns: list = [],
    ):
        """
    Flatten the selected VMap. A new vDataFrame is returned.
    
    \u26A0 Warning : This function might have a long runtime and can make your
                     vDataFrame less performant. It makes many calls to the
                     MAPLOOKUP function, which can be slow if your VMap is
                     large.

    Parameters
    ----------
    vmap_col: str / list, optional
        List of VMap columns to flatten.
    limit: int, optional
        Maximum number of keys to consider for each VMap. Only the most occurent 
        keys are used.
    exclude_columns: list, optional
        List of VMap columns to exclude.

    Returns
    -------
    vDataFrame
        object with the flattened VMaps.
        """
        if not (vmap_col):
            vmap_col = []
            all_cols = self.get_columns()
            for col in all_cols:
                if self[col].isvmap():
                    vmap_col += [col]
        if isinstance(vmap_col, str):
            vmap_col = [vmap_col]
        exclude_columns_final, vmap_col_final = (
            [quote_ident(col).lower() for col in exclude_columns],
            [],
        )
        for col in vmap_col:
            if quote_ident(col).lower() not in exclude_columns_final:
                vmap_col_final += [col]
        if not (vmap_col):
            raise EmptyParameter("No VMAP was detected.")
        maplookup = []
        for vmap in vmap_col_final:
            keys = compute_vmap_keys(expr=self, vmap_col=vmap, limit=limit)
            keys = [k[0] for k in keys]
            for k in keys:
                column = quote_ident(vmap)
                alias = quote_ident(vmap.replace('"', "") + "." + k.replace('"', ""))
                maplookup += [f"MAPLOOKUP({column}, '{k}') AS {alias}"]
        return self.select(self.get_columns() + maplookup)

    def get_columns(self, exclude_columns: Union[str, list] = []):
        """
    Returns the vDataFrame vColumns.

    Parameters
    ----------
    exclude_columns: str / list, optional
        List of the vColumns names to exclude from the final list. 

    Returns
    -------
    List
        List of all vDataFrame columns.

    See Also
    --------
    vDataFrame.catcol  : Returns all categorical vDataFrame vColumns.
    vDataFrame.datecol : Returns all vDataFrame vColumns of type date.
    vDataFrame.numcol  : Returns all numerical vDataFrame vColumns.
        """
        # -#
        if isinstance(exclude_columns, str):
            exclude_columns = [columns]
        columns = [elem for elem in self._VERTICAPY_VARIABLES_["columns"]]
        result = []
        exclude_columns = [elem for elem in exclude_columns]
        exclude_columns += [
            elem for elem in self._VERTICAPY_VARIABLES_["exclude_columns"]
        ]
        exclude_columns = [elem.replace('"', "").lower() for elem in exclude_columns]
        for column in columns:
            if column.replace('"', "").lower() not in exclude_columns:
                result += [column]
        return result

    @save_verticapy_logs
    def one_hot_encode(
        self,
        columns: Union[str, list] = [],
        max_cardinality: int = 12,
        prefix_sep: str = "_",
        drop_first: bool = True,
        use_numbers_as_suffix: bool = False,
    ):
        """
    Encodes the vColumns using the One Hot Encoding algorithm.

    Parameters
    ----------
    columns: str / list, optional
        List of the vColumns to use to train the One Hot Encoding model. If empty, 
        only the vColumns having a cardinality lesser than 'max_cardinality' will 
        be used.
    max_cardinality: int, optional
        Cardinality threshold to use to determine if the vColumn will be taken into
        account during the encoding. This parameter is used only if the parameter 
        'columns' is empty.
    prefix_sep: str, optional
        Prefix delimitor of the dummies names.
    drop_first: bool, optional
        Drops the first dummy to avoid the creation of correlated features.
    use_numbers_as_suffix: bool, optional
        Uses numbers as suffix instead of the vColumns categories.

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame[].decode       : Encodes the vColumn using a user defined Encoding.
    vDataFrame[].discretize   : Discretizes the vColumn.
    vDataFrame[].get_dummies  : Computes the vColumns result of One Hot Encoding.
    vDataFrame[].label_encode : Encodes the vColumn using the Label Encoding.
    vDataFrame[].mean_encode  : Encodes the vColumn using the Mean Encoding of a response.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns = self.format_colnames(columns)
        if not (columns):
            columns = self.get_columns()
        cols_hand = True if (columns) else False
        for column in columns:
            if self[column].nunique(True) < max_cardinality:
                self[column].get_dummies(
                    "", prefix_sep, drop_first, use_numbers_as_suffix
                )
            elif cols_hand and OPTIONS["print_info"]:
                warning_message = (
                    f"The vColumn '{column}' was ignored because of "
                    "its high cardinality.\nIncrease the parameter "
                    "'max_cardinality' to solve this issue or use "
                    "directly the vColumn get_dummies method."
                )
                warnings.warn(warning_message, Warning)
        return self

    get_dummies = one_hot_encode

    def head(self, limit: int = 5):
        """
    Returns the vDataFrame head.

    Parameters
    ----------
    limit: int, optional
        Number of elements to display.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.tail : Returns the vDataFrame tail.
        """
        return self.iloc(limit=limit, offset=0)

    def iloc(self, limit: int = 5, offset: int = 0, columns: Union[str, list] = []):
        """
    Returns a part of the vDataFrame (delimited by an offset and a limit).

    Parameters
    ----------
    limit: int, optional
        Number of elements to display.
    offset: int, optional
        Number of elements to skip.
    columns: str / list, optional
        A list containing the names of the vColumns to include in the result. 
        If empty, all vColumns will be selected.


    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.head : Returns the vDataFrame head.
    vDataFrame.tail : Returns the vDataFrame tail.
        """
        # -#
        if isinstance(columns, str):
            columns = [columns]
        if offset < 0:
            offset = max(0, self.shape()[0] - limit)
        columns = self.format_colnames(columns)
        if not (columns):
            columns = self.get_columns()
        all_columns = []
        for column in columns:
            cast = to_varchar(self[column].category(), column)
            all_columns += [f"{cast} AS {column}"]
        title = (
            "Reads the final relation using a limit "
            f"of {limit} and an offset of {offset}."
        )
        result = to_tablesample(
            query=f"""
                SELECT 
                    {', '.join(all_columns)} 
                FROM {self.__genSQL__()}
                {self.__get_last_order_by__()} 
                LIMIT {limit} OFFSET {offset}""",
            title=title,
            max_columns=self._VERTICAPY_VARIABLES_["max_columns"],
            sql_push_ext=self._VERTICAPY_VARIABLES_["sql_push_ext"],
            symbol=self._VERTICAPY_VARIABLES_["symbol"],
        )
        pre_comp = self.__get_catalog_value__("VERTICAPY_COUNT")
        if pre_comp != "VERTICAPY_NOT_PRECOMPUTED":
            result.count = pre_comp
        elif OPTIONS["count_on"]:
            result.count = self.shape()[0]
        result.offset = offset
        result.name = self._VERTICAPY_VARIABLES_["input_relation"]
        columns = self.get_columns()
        all_percent = True
        for column in columns:
            if not ("percent" in self[column].catalog):
                all_percent = False
        all_percent = (all_percent or (OPTIONS["percent_bar"] == True)) and (
            OPTIONS["percent_bar"] != False
        )
        if all_percent:
            percent = self.aggregate(["percent"], columns).transpose().values
        for column in result.values:
            result.dtype[column] = self[column].ctype()
            if all_percent:
                result.percent[column] = percent[self.format_colnames(column)][0]
        return result

    def info(self):
        """
    Displays information about the different vDataFrame transformations.

    Returns
    -------
    str
        information on the vDataFrame modifications
        """
        if len(self._VERTICAPY_VARIABLES_["history"]) == 0:
            result = "The vDataFrame was never modified."
        elif len(self._VERTICAPY_VARIABLES_["history"]) == 1:
            result = "The vDataFrame was modified with only one action: "
            result += "\n * " + self._VERTICAPY_VARIABLES_["history"][0]
        else:
            result = "The vDataFrame was modified many times: "
            for modif in self._VERTICAPY_VARIABLES_["history"]:
                result += "\n * " + modif
        return result

    @save_verticapy_logs
    def isin(self, val: dict):
        """
    Looks if some specific records are in the vDataFrame and it returns the new 
    vDataFrame of the search.

    Parameters
    ----------
    val: dict
        Dictionary of the different records. Each key of the dictionary must 
        represent a vColumn. For example, to check if Badr Ouali and 
        Fouad Teban are in the vDataFrame. You can write the following dict:
        {"name": ["Teban", "Ouali"], "surname": ["Fouad", "Badr"]}

    Returns
    -------
    vDataFrame
        The vDataFrame of the search.
        """
        val = self.format_colnames(val)
        n = len(val[list(val.keys())[0]])
        result = []
        for i in range(n):
            tmp_query = []
            for column in val:
                if val[column][i] == None:
                    tmp_query += [f"{quote_ident(column)} IS NULL"]
                else:
                    val_str = str(val[column][i]).replace("'", "''")
                    tmp_query += [f"{quote_ident(column)} = '{val_str}'"]
            result += [" AND ".join(tmp_query)]
        return self.search(" OR ".join(result))

    @save_verticapy_logs
    def join(
        self,
        input_relation,
        on: Union[tuple, dict, list] = {},
        on_interpolate: dict = {},
        how: Literal[
            "left", "right", "cross", "full", "natural", "self", "inner", ""
        ] = "natural",
        expr1: Union[str, list] = ["*"],
        expr2: Union[str, list] = ["*"],
    ):
        """
    Joins the vDataFrame with another one or an input relation.

    \u26A0 Warning : Joins can make the vDataFrame structure heavier. It is 
                     recommended to always check the current structure 
                     using the 'current_relation' method and to save it using the 
                     'to_db' method with the parameters 'inplace = True' and 
                     'relation_type = table'

    Parameters
    ----------
    input_relation: str/vDataFrame
        Relation to use to do the merging.
    on: tuple / dict / list, optional
        If it is a list then:
        List of 3-tuples. Each tuple must include (key1, key2, operator)—where
        key1 is the key of the vDataFrame, key2 is the key of the input relation,
        and operator can be one of the following:
                     '=' : exact match
                     '<' : key1  < key2
                     '>' : key1  > key2
                    '<=' : key1 <= key2
                    '>=' : key1 >= key2
                 'llike' : key1 LIKE '%' || key2 || '%'
                 'rlike' : key2 LIKE '%' || key1 || '%'
           'linterpolate': key1 INTERPOLATE key2
           'rinterpolate': key2 INTERPOLATE key1
        Some operators need 5-tuples: (key1, key2, operator, operator2, x)—where
        operator2 is a simple operator (=, >, <, <=, >=), x is a float or an integer, 
        and operator is one of the following:
                 'jaro' : JARO(key1, key2) operator2 x
                'jarow' : JARO_WINCKLER(key1, key2) operator2 x
                  'lev' : LEVENSHTEIN(key1, key2) operator2 x
        
        If it is a dictionary then:
        This parameter must include all the different keys. It must be similar 
        to the following:
        {"relationA_key1": "relationB_key1" ..., "relationA_keyk": "relationB_keyk"}
        where relationA is the current vDataFrame and relationB is the input relation
        or the input vDataFrame.
    on_interpolate: dict, optional
        Dictionary of all different keys. Used to join two event series together 
        using some ordered attribute, event series joins let you compare values from 
        two series directly, rather than having to normalize the series to the same 
        measurement interval. The dict must be similar to the following:
        {"relationA_key1": "relationB_key1" ..., "relationA_keyk": "relationB_keyk"}
        where relationA is the current vDataFrame and relationB is the input relation
        or the input vDataFrame.
    how: str, optional
        Join Type.
            left    : Left Join.
            right   : Right Join.
            cross   : Cross Join.
            full    : Full Outer Join.
            natural : Natural Join.
            inner   : Inner Join.
    expr1: str / list, optional
        List of the different columns in pure SQL to select from the current 
        vDataFrame, optionally as aliases. Aliases are recommended to avoid 
        ambiguous names. For example: 'column' or 'column AS my_new_alias'. 
    expr2: str / list, optional
        List of the different columns in pure SQL to select from the input 
        relation optionally as aliases. Aliases are recommended to avoid 
        ambiguous names. For example: 'column' or 'column AS my_new_alias'.

    Returns
    -------
    vDataFrame
        object result of the join.

    See Also
    --------
    vDataFrame.append  : Merges the vDataFrame with another relation.
    vDataFrame.groupby : Aggregates the vDataFrame.
    vDataFrame.sort    : Sorts the vDataFrame.
        """
        if isinstance(expr1, str):
            expr1 = [expr1]
        if isinstance(expr2, str):
            expr2 = [expr2]
        if isinstance(on, tuple):
            on = [on]
        # Giving the right alias to the right relation
        def create_final_relation(relation: str, alias: str):
            if (
                ("SELECT" in relation.upper())
                and ("FROM" in relation.upper())
                and ("(" in relation)
                and (")" in relation)
            ):
                return f"(SELECT * FROM {relation}) AS {alias}"
            else:
                return f"{relation} AS {alias}"

        # List with the operators
        if str(how).lower() == "natural" and (on or on_interpolate):
            raise ParameterError(
                "Natural Joins cannot be computed if any of "
                "the parameters 'on' or 'on_interpolate' are "
                "defined."
            )
        on_list = []
        if isinstance(on, dict):
            on_list += [(key, on[key], "=") for key in on]
        else:
            on_list += [elem for elem in on]
        on_list += [(key, on[key], "linterpolate") for key in on_interpolate]
        # Checks
        self.format_colnames([elem[0] for elem in on_list])
        if isinstance(input_relation, vDataFrame):
            input_relation.format_colnames([elem[1] for elem in on_list])
            relation = input_relation.__genSQL__()
        else:
            relation = input_relation
        # Relations
        first_relation = create_final_relation(self.__genSQL__(), alias="x")
        second_relation = create_final_relation(relation, alias="y")
        # ON
        on_join = []
        all_operators = [
            "=",
            ">",
            ">=",
            "<",
            "<=",
            "llike",
            "rlike",
            "linterpolate",
            "rinterpolate",
            "jaro",
            "jarow",
            "lev",
        ]
        simple_operators = all_operators[0:5]
        for elem in on_list:
            key1, key2, op = quote_ident(elem[0]), quote_ident(elem[1]), elem[2]
            if op not in all_operators:
                raise ValueError(
                    f"Incorrect operator: '{op}'.\nCorrect values: {', '.join(simple_operators)}."
                )
            if op in ("=", ">", ">=", "<", "<="):
                on_join += [f"x.{key1} {op} y.{key2}"]
            elif op == "llike":
                on_join += [f"x.{key1} LIKE '%' || y.{key2} || '%'"]
            elif op == "rlike":
                on_join += [f"y.{key2} LIKE '%' || x.{key1} || '%'"]
            elif op == "linterpolate":
                on_join += [f"x.{key1} INTERPOLATE PREVIOUS VALUE y.{key2}"]
            elif op == "rinterpolate":
                on_join += [f"y.{key2} INTERPOLATE PREVIOUS VALUE x.{key1}"]
            elif op in ("jaro", "jarow", "lev"):
                if op in ("jaro", "jarow"):
                    vertica_version(condition=[12, 0, 2])
                else:
                    vertica_version(condition=[10, 1, 0])
                op2, x = elem[3], elem[4]
                if op2 not in simple_operators:
                    raise ValueError(
                        f"Incorrect operator: '{op2}'.\nCorrect values: {', '.join(simple_operators)}."
                    )
                map_to_fun = {
                    "jaro": "JARO_DISTANCE",
                    "jarow": "JARO_WINKLER_DISTANCE",
                    "lev": "EDIT_DISTANCE",
                }
                fun = map_to_fun[op]
                on_join += [f"{fun}(x.{key1}, y.{key2}) {op2} {x}"]
        # Final
        on_join = " ON " + " AND ".join(on_join) if on_join else ""
        expr = [f"x.{key}" for key in expr1] + [f"y.{key}" for key in expr2]
        expr = "*" if not (expr) else ", ".join(expr)
        if how:
            how = " " + how.upper() + " "
        table = (
            f"SELECT {expr} FROM {first_relation}{how}JOIN {second_relation} {on_join}"
        )
        return self.__vDataFrameSQL__(
            f"({table}) VERTICAPY_SUBTABLE",
            "join",
            "[Join]: Two relations were joined together",
        )

    @save_verticapy_logs
    def load(self, offset: int = -1):
        """
    Loads a previous structure of the vDataFrame. 

    Parameters
    ----------
    offset: int, optional
        offset of the saving. Example: -1 to load the last saving.

    Returns
    -------
    vDataFrame
        vDataFrame of the loading.

    See Also
    --------
    vDataFrame.save : Saves the current vDataFrame structure.
        """
        save = self._VERTICAPY_VARIABLES_["saving"][offset]
        vdf = pickle.loads(save)
        return vdf

    @save_verticapy_logs
    def memory_usage(self):
        """
    Returns the vDataFrame memory usage. 

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.expected_store_usage : Returns the expected store usage.
        """
        total = sum(
            [sys.getsizeof(elem) for elem in self._VERTICAPY_VARIABLES_]
        ) + sys.getsizeof(self)
        values = {"index": ["object"], "value": [total]}
        columns = [elem for elem in self._VERTICAPY_VARIABLES_["columns"]]
        for column in columns:
            values["index"] += [column]
            values["value"] += [self[column].memory_usage()]
            total += self[column].memory_usage()
        values["index"] += ["total"]
        values["value"] += [total]
        return tablesample(values=values)

    @save_verticapy_logs
    def merge_similar_names(self, skip_word: Union[str, list]):
        """
    Merges columns with similar names. The function generates a COALESCE 
    statement that merges the columns into a single column that excludes 
    the input words. Note that the order of the variables in the COALESCE 
    statement is based on the order of the 'get_columns' method.
    
    Parameters
    ---------- 
    skip_word: str / list, optional
        List of words to exclude from the provided column names. 
        For example, if two columns are named 'age.information.phone' 
        and 'age.phone' AND skip_word is set to ['.information'], then 
        the two columns will be merged together with the following 
        COALESCE statement:
        COALESCE("age.phone", "age.information.phone") AS "age.phone"

    Returns
    -------
    vDataFrame
        An object containing the merged element.
        """
        if isinstance(skip_word, str):
            skip_word = [skip_word]
        columns = self.get_columns()
        group_dict = group_similar_names(columns, skip_word=skip_word)
        sql = f"""
            (SELECT 
                {gen_coalesce(group_dict)} 
            FROM {self.__genSQL__()}) VERTICAPY_SUBTABLE"""
        return self.__vDataFrameSQL__(
            sql,
            "merge_similar_names",
            "[merge_similar_names]: The columns were merged.",
        )

    @save_verticapy_logs
    def narrow(
        self,
        index: Union[str, list],
        columns: Union[str, list] = [],
        col_name: str = "column",
        val_name: str = "value",
    ):
        """
    Returns the Narrow Table of the vDataFrame using the input vColumns.

    Parameters
    ----------
    index: str / list
        Index(es) used to identify the Row.
    columns: str / list, optional
        List of the vColumns names. If empty, all vColumns except the index(es)
        will be used.
    col_name: str, optional
        Alias of the vColumn representing the different input vColumns names as 
        categories.
    val_name: str, optional
        Alias of the vColumn representing the different input vColumns values.

    Returns
    -------
    vDataFrame
        the narrow table object.

    See Also
    --------
    vDataFrame.pivot : Returns the pivot table of the vDataFrame.
        """
        index, columns = self.format_colnames(index, columns)
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(index, str):
            index = [index]
        if not (columns):
            columns = self.numcol()
        for idx in index:
            if idx in columns:
                columns.remove(idx)
        query = []
        all_are_num, all_are_date = True, True
        for column in columns:
            if not (self[column].isnum()):
                all_are_num = False
            if not (self[column].isdate()):
                all_are_date = False
        for column in columns:
            conv = ""
            if not (all_are_num) and not (all_are_num):
                conv = "::varchar"
            elif self[column].category() == "int":
                conv = "::int"
            column_str = column.replace("'", "''")[1:-1]
            query += [
                f"""
                (SELECT 
                    {', '.join(index)}, 
                    '{column_str}' AS {col_name}, 
                    {column}{conv} AS {val_name} 
                FROM {self.__genSQL__()})"""
            ]
        query = " UNION ALL ".join(query)
        query = f"({query}) VERTICAPY_SUBTABLE"
        return self.__vDataFrameSQL__(
            query, "narrow", f"[Narrow]: Narrow table using index = {index}",
        )

    melt = narrow

    @save_verticapy_logs
    def normalize(
        self,
        columns: Union[str, list] = [],
        method: Literal["zscore", "robust_zscore", "minmax"] = "zscore",
    ):
        """
    Normalizes the input vColumns using the input method.

    Parameters
    ----------
    columns: str / list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.
    method: str, optional
        Method to use to normalize.
            zscore        : Normalization using the Z-Score (avg and std).
                (x - avg) / std
            robust_zscore : Normalization using the Robust Z-Score (median and mad).
                (x - median) / (1.4826 * mad)
            minmax        : Normalization using the MinMax (min and max).
                (x - min) / (max - min)

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame.outliers    : Computes the vDataFrame Global Outliers.
    vDataFrame[].normalize : Normalizes the vColumn. This method is more complete 
        than the vDataFrame.normalize method by allowing more parameters.
        """
        if isinstance(columns, str):
            columns = [columns]
        no_cols = True if not (columns) else False
        columns = self.numcol() if not (columns) else self.format_colnames(columns)
        for column in columns:
            if self[column].isnum() and not (self[column].isbool()):
                self[column].normalize(method=method)
            elif (no_cols) and (self[column].isbool()):
                pass
            elif OPTIONS["print_info"]:
                warning_message = (
                    f"The vColumn {column} was skipped.\n"
                    "Normalize only accept numerical data types."
                )
                warnings.warn(warning_message, Warning)
        return self

    def numcol(self, exclude_columns: list = []):
        """
    Returns a list of names of the numerical vColumns in the vDataFrame.

    Parameters
    ----------
    exclude_columns: list, optional
        List of the vColumns names to exclude from the final list. 

    Returns
    -------
    List
        List of numerical vColumns names. 
    
    See Also
    --------
    vDataFrame.catcol      : Returns the categorical type vColumns in the vDataFrame.
    vDataFrame.get_columns : Returns the vColumns of the vDataFrame.
        """
        columns, cols = [], self.get_columns(exclude_columns=exclude_columns)
        for column in cols:
            if self[column].isnum():
                columns += [column]
        return columns

    @save_verticapy_logs
    def outliers(
        self,
        columns: Union[str, list] = [],
        name: str = "distribution_outliers",
        threshold: float = 3.0,
        robust: bool = False,
    ):
        """
    Adds a new vColumn labeled with 0 and 1. 1 means that the record is a global 
    outlier.

    Parameters
    ----------
    columns: str / list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.
    name: str, optional
        Name of the new vColumn.
    threshold: float, optional
        Threshold equals to the critical score.
    robust: bool
        If set to True, the score used will be the Robust Z-Score instead of 
        the Z-Score.

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame.normalize : Normalizes the input vColumns.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns = self.format_colnames(columns) if (columns) else self.numcol()
        if not (robust):
            result = self.aggregate(func=["std", "avg"], columns=columns).values
        else:
            result = self.aggregate(
                func=["mad", "approx_median"], columns=columns
            ).values
        conditions = []
        for idx, col in enumerate(result["index"]):
            if not (robust):
                conditions += [
                    f"""
                    ABS({col} - {result['avg'][idx]}) 
                    / NULLIFZERO({result['std'][idx]}) 
                    > {threshold}"""
                ]
            else:
                conditions += [
                    f"""
                    ABS({col} - {result['approx_median'][idx]}) 
                    / NULLIFZERO({result['mad'][idx]} * 1.4826) 
                    > {threshold}"""
                ]
        self.eval(name, f"(CASE WHEN {' OR '.join(conditions)} THEN 1 ELSE 0 END)")
        return self

    @save_verticapy_logs
    def pivot(
        self,
        index: str,
        columns: str,
        values: str,
        aggr: str = "sum",
        prefix: str = "",
    ):
        """
    Returns the Pivot of the vDataFrame using the input aggregation.

    Parameters
    ----------
    index: str
        vColumn to use to group the elements.
    columns: str
        The vColumn used to compute the different categories, which then act 
        as the columns in the pivot table.
    values: str
        The vColumn whose values populate the new vDataFrame.
    aggr: str, optional
        Aggregation to use on 'values'. To use complex aggregations, 
        you must use braces: {}. For example, to aggregate using the 
        aggregation: x -> MAX(x) - MIN(x), write "MAX({}) - MIN({})".
    prefix: str, optional
        The prefix for the pivot table's column names.

    Returns
    -------
    vDataFrame
        the pivot table object.

    See Also
    --------
    vDataFrame.narrow      : Returns the Narrow table of the vDataFrame.
    vDataFrame.pivot_table : Draws the pivot table of one or two columns based on an 
        aggregation.
        """
        index, columns, values = self.format_colnames(index, columns, values)
        aggr = aggr.upper()
        if "{}" not in aggr:
            aggr += "({})"
        new_cols = self[columns].distinct()
        new_cols_trans = []
        for elem in new_cols:
            if elem == None:
                new_cols_trans += [
                    aggr.replace(
                        "{}",
                        f"(CASE WHEN {columns} IS NULL THEN {values} ELSE NULL END)",
                    )
                    + f"AS '{prefix}NULL'"
                ]
            else:
                new_cols_trans += [
                    aggr.replace(
                        "{}",
                        f"(CASE WHEN {columns} = '{elem}' THEN {values} ELSE NULL END)",
                    )
                    + f"AS '{prefix}{elem}'"
                ]
        return self.__vDataFrameSQL__(
            f"""
            (SELECT 
                {index},
                {", ".join(new_cols_trans)}
             FROM {self.__genSQL__()}
             GROUP BY 1) VERTICAPY_SUBTABLE""",
            "pivot",
            (
                f"[Pivot]: Pivot table using index = {index} & "
                f"columns = {columns} & values = {values}"
            ),
        )

    @save_verticapy_logs
    def pivot_table_chi2(
        self,
        response: str,
        columns: Union[str, list] = [],
        nbins: int = 16,
        method: Literal["smart", "same_width"] = "same_width",
        RFmodel_params: dict = {},
    ):
        """
    Returns the chi-square term using the pivot table of the response vColumn 
    against the input vColumns.

    Parameters
    ----------
    response: str
        Categorical response vColumn.
    columns: str / list, optional
        List of the vColumn names. The maximum number of categories for each
        categorical columns is 16. Categorical columns with a higher cardinality
        are discarded.
    nbins: int, optional
        Integer in the range [2,16], the number of bins used to discretize 
        the numerical features.
    method: str, optional
        The method to use to discretize the numerical vColumns.
            same_width : Computes bins of regular width.
            smart      : Uses a random forest model on a response column to find the best
                interval for discretization.
    RFmodel_params: dict, optional
        Dictionary of the parameters of the random forest model used to compute the best splits 
        when 'method' is 'smart'. If the response column is numerical (but not of type int or bool), 
        this function trains and uses a random forest regressor.  Otherwise, this function 
        trains a random forest classifier.
        For example, to train a random forest with 20 trees and a maximum depth of 10, use:
            {"n_estimators": 20, "max_depth": 10}

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns, response = self.format_colnames(columns, response)
        assert 2 <= nbins <= 16, ParameterError(
            "Parameter 'nbins' must be between 2 and 16, inclusive."
        )
        columns = self.chaid_columns(columns)
        for col in columns:
            if quote_ident(response) == quote_ident(col):
                columns.remove(col)
                break
        if not (columns):
            raise ValueError("No column to process.")
        if self.shape()[0] == 0:
            return {
                "index": columns,
                "chi2": [0.0 for col in columns],
                "categories": [[] for col in columns],
                "is_numerical": [self[col].isnum() for col in columns],
            }
        vdf = self.copy()
        for col in columns:
            if vdf[col].isnum():
                vdf[col].discretize(
                    method=method,
                    nbins=nbins,
                    response=response,
                    RFmodel_params=RFmodel_params,
                )
        response = vdf.format_colnames(response)
        if response in columns:
            columns.remove(response)
        chi2_list = []
        for col in columns:
            tmp_res = vdf.pivot_table(
                columns=[col, response], max_cardinality=(10000, 100), show=False
            ).to_numpy()[:, 1:]
            tmp_res = np.where(tmp_res == "", "0", tmp_res)
            tmp_res = tmp_res.astype(float)
            i = 0
            all_chi2 = []
            for row in tmp_res:
                j = 0
                for col_in_row in row:
                    all_chi2 += [
                        col_in_row ** 2 / (sum(tmp_res[i]) * sum(tmp_res[:, j]))
                    ]
                    j += 1
                i += 1
            val = sum(sum(tmp_res)) * (sum(all_chi2) - 1)
            k, r = tmp_res.shape
            dof = (k - 1) * (r - 1)
            pval = scipy_st.chi2.sf(val, dof)
            chi2_list += [(col, val, pval, dof, vdf[col].distinct(), self[col].isnum())]
        chi2_list = sorted(chi2_list, key=lambda tup: tup[1], reverse=True)
        result = {
            "index": [chi2[0] for chi2 in chi2_list],
            "chi2": [chi2[1] for chi2 in chi2_list],
            "p_value": [chi2[2] for chi2 in chi2_list],
            "dof": [chi2[3] for chi2 in chi2_list],
            "categories": [chi2[4] for chi2 in chi2_list],
            "is_numerical": [chi2[5] for chi2 in chi2_list],
        }
        return tablesample(result)

    @save_verticapy_logs
    def polynomial_comb(self, columns: Union[str, list] = [], r: int = 2):
        """
    Returns a vDataFrame containing different product combination of the 
    input vColumns. This function is ideal for bivariate analysis.

    Parameters
    ----------
    columns: str / list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.
    r: int, optional
        Degree of the polynomial.

    Returns
    -------
    vDataFrame
        the Polynomial object.
        """
        if isinstance(columns, str):
            columns = [columns]
        if not (columns):
            numcol = self.numcol()
        else:
            numcol = self.format_colnames(columns)
        vdf = self.copy()
        all_comb = combinations_with_replacement(numcol, r=r)
        for elem in all_comb:
            name = "_".join(elem)
            vdf.eval(name.replace('"', ""), expr=" * ".join(elem))
        return vdf

    @save_verticapy_logs
    def recommend(
        self,
        unique_id: str,
        item_id: str,
        method: Literal["count", "avg", "median"] = "count",
        rating: Union[str, tuple] = "",
        ts: str = "",
        start_date: Union[str, int, float, datetime.datetime, datetime.date] = "",
        end_date: Union[str, int, float, datetime.datetime, datetime.date] = "",
    ):
        """
    Recommend items based on the Collaborative Filtering (CF) technique.
    The implementation is the same as APRIORI algorithm, but is limited to pairs 
    of items.

    Parameters
    ----------
    unique_id: str
        Input vColumn corresponding to a unique ID. It is a primary key.
    item_id: str
        Input vColumn corresponding to an item ID. It is a secondary key used to 
        compute the different pairs.
    method: str, optional
        Method used to recommend.
            count  : Each item will be recommended based on frequencies of the
                     different pairs of items.
            avg    : Each item will be recommended based on the average rating
                     of the different item pairs with a differing second element.
            median : Each item will be recommended based on the median rating
                     of the different item pairs with a differing second element.
    rating: str / tuple, optional
        Input vColumn including the items rating.
        If the 'rating' type is 'tuple', it must composed of 3 elements: 
        (r_vdf, r_item_id, r_name) where:
            r_vdf is an input vDataFrame.
            r_item_id is an input vColumn which must includes the same id as 'item_id'.
            r_name is an input vColumn including the items rating. 
    ts: str, optional
        TS (Time Series) vColumn to use to order the data. The vColumn type must be
        date like (date, datetime, timestamp...) or numerical.
    start_date: str / int / float / date, optional
        Input Start Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is lesser than November 1993 the 3rd.
    end_date: str / int / float / date, optional
        Input End Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is greater than November 1993 the 3rd.

    Returns
    -------
    vDataFrame
        The vDataFrame of the recommendation.
        """
        unique_id, item_id, ts = self.format_colnames(unique_id, item_id, ts)
        vdf = self.copy()
        assert (
            method == "count" or rating
        ), f"Method '{method}' can not be used if parameter 'rating' is empty."
        if rating:
            assert isinstance(rating, str) or len(rating) == 3, ParameterError(
                "Parameter 'rating' must be of type str or composed of "
                "exactly 3 elements: (r_vdf, r_item_id, r_name)."
            )
            assert (
                method != "count"
            ), "Method 'count' can not be used if parameter 'rating' is defined."
            rating = self.format_colnames(rating)
        if ts:
            if start_date and end_date:
                vdf = self.search(f"{ts} BETWEEN '{start_date}' AND '{end_date}'")
            elif start_date:
                vdf = self.search(f"{ts} >= '{start_date}'")
            elif end_date:
                vdf = self.search(f"{ts} <= '{end_date}'")
        vdf = (
            vdf.join(
                vdf,
                how="left",
                on={unique_id: unique_id},
                expr1=[f"{item_id} AS item1"],
                expr2=[f"{item_id} AS item2"],
            )
            .groupby(["item1", "item2"], ["COUNT(*) AS cnt"])
            .search("item1 != item2 AND cnt > 1")
        )
        order_columns = "cnt DESC"
        if method in ("avg", "median"):
            fun = "AVG" if method == "avg" else "APPROXIMATE_MEDIAN"
            if isinstance(rating, str):
                r_vdf = self.groupby([item_id], [f"{fun}({rating}) AS score"])
                r_item_id = item_id
                r_name = "score"
            else:
                r_vdf, r_item_id, r_name = rating
                r_vdf = r_vdf.groupby([r_item_id], [f"{fun}({r_name}) AS {r_name}"])
            vdf = vdf.join(
                r_vdf,
                how="left",
                on={"item1": r_item_id},
                expr2=[f"{r_name} AS score1"],
            ).join(
                r_vdf,
                how="left",
                on={"item2": r_item_id},
                expr2=[f"{r_name} AS score2"],
            )
            order_columns = "score2 DESC, score1 DESC, cnt DESC"
        vdf["rank"] = f"ROW_NUMBER() OVER (PARTITION BY item1 ORDER BY {order_columns})"
        return vdf

    @save_verticapy_logs
    def regexp(
        self,
        column: str,
        pattern: str,
        method: Literal[
            "count",
            "ilike",
            "instr",
            "like",
            "not_ilike",
            "not_like",
            "replace",
            "substr",
        ] = "substr",
        position: int = 1,
        occurrence: int = 1,
        replacement: str = "",
        return_position: int = 0,
        name: str = "",
    ):
        """
    Computes a new vColumn based on regular expressions. 

    Parameters
    ----------
    column: str
        Input vColumn to use to compute the regular expression.
    pattern: str
        The regular expression.
    method: str, optional
        Method to use to compute the regular expressions.
            count     : Returns the number times a regular expression matches 
                each element of the input vColumn. 
            ilike     : Returns True if the vColumn element contains a match 
                for the regular expression.
            instr     : Returns the starting or ending position in a vColumn 
                element where a regular expression matches. 
            like      : Returns True if the vColumn element matches the regular 
                expression.
            not_ilike : Returns True if the vColumn element does not match the 
                case-insensitive regular expression.
            not_like  : Returns True if the vColumn element does not contain a 
                match for the regular expression.
            replace   : Replaces all occurrences of a substring that match a 
                regular expression with another substring.
            substr    : Returns the substring that matches a regular expression 
                within a vColumn.
    position: int, optional
        The number of characters from the start of the string where the function 
        should start searching for matches.
    occurrence: int, optional
        Controls which occurrence of a pattern match in the string to return.
    replacement: str, optional
        The string to replace matched substrings.
    return_position: int, optional
        Sets the position within the string to return.
    name: str, optional
        New feature name. If empty, a name will be generated.

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame.eval : Evaluates a customized expression.
        """
        column = self.format_colnames(column)
        pattern_str = pattern.replace("'", "''")
        expr = f"REGEXP_{method.upper()}({column}, '{pattern_str}'"
        if method == "replace":
            replacement_str = replacement.replace("'", "''")
            expr += f", '{replacement_str}'"
        if method in ("count", "instr", "replace", "substr"):
            expr += f", {position}"
        if method in ("instr", "replace", "substr"):
            expr += f", {occurrence}"
        if method == "instr":
            expr += f", {return_position}"
        expr += ")"
        gen_name([method, column])
        return self.eval(name=name, expr=expr)

    @save_verticapy_logs
    def save(self):
        """
    Saves the current structure of the vDataFrame. 
    This function is useful for loading previous transformations.

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame.load : Loads a saving.
        """
        vdf = self.copy()
        self._VERTICAPY_VARIABLES_["saving"] += [pickle.dumps(vdf)]
        return self

    @save_verticapy_logs
    def sessionize(
        self,
        ts: str,
        by: Union[str, list] = [],
        session_threshold: str = "30 minutes",
        name: str = "session_id",
    ):
        """
    Adds a new vColumn to the vDataFrame which will correspond to sessions 
    (user activity during a specific time). A session ends when ts - lag(ts) 
    is greater than a specific threshold.

    Parameters
    ----------
    ts: str
        vColumn used as timeline. It will be to use to order the data. It can be
        a numerical or type date like (date, datetime, timestamp...) vColumn.
    by: str / list, optional
        vColumns used in the partition.
    session_threshold: str, optional
        This parameter is the threshold which will determine the end of the 
        session. For example, if it is set to '10 minutes' the session ends
        after 10 minutes of inactivity.
    name: str, optional
        The session name.

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame.analytic : Adds a new vColumn to the vDataFrame by using an advanced 
        analytical function on a specific vColumn.
        """
        if isinstance(by, str):
            by = [by]
        by, ts = self.format_colnames(by, ts)
        partition = ""
        if by:
            partition = f"PARTITION BY {', '.join(by)}"
        expr = f"""CONDITIONAL_TRUE_EVENT(
                    {ts}::timestamp - LAG({ts}::timestamp) 
                  > '{session_threshold}') 
                  OVER ({partition} ORDER BY {ts})"""
        return self.eval(name=name, expr=expr)

    @save_verticapy_logs
    def score(
        self,
        y_true: str,
        y_score: str,
        method: str,  # TODO Literal[tuple(FUNCTIONS_DICTIONNARY)]
        nbins: int = 30,
    ):
        """
    Computes the score using the input columns and the input method.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    method: str
        The method to use to compute the score.
            --- For Classification ---
            accuracy    : Accuracy
            auc         : Area Under the Curve (ROC)
            best_cutoff : Cutoff which optimised the ROC Curve prediction.
            bm          : Informedness = tpr + tnr - 1
            csi         : Critical Success Index = tp / (tp + fn + fp)
            f1          : F1 Score 
            logloss     : Log Loss
            mcc         : Matthews Correlation Coefficient 
            mk          : Markedness = ppv + npv - 1
            npv         : Negative Predictive Value = tn / (tn + fn)
            prc_auc     : Area Under the Curve (PRC)
            precision   : Precision = tp / (tp + fp)
            recall      : Recall = tp / (tp + fn)
            specificity : Specificity = tn / (tn + fp)
            --- For Regression ---
            max    : Max Error
            mae    : Mean Absolute Error
            median : Median Absolute Error
            mse    : Mean Squared Error
            msle   : Mean Squared Log Error
            r2     : R squared coefficient
            var    : Explained Variance  
            --- Plots ---
            roc  : ROC Curve
            prc  : PRC Curve
            lift : Lift Chart
    nbins: int, optional
        Number of bins used to compute some of the metrics (AUC, PRC AUC...)

    Returns
    -------
    float / tablesample
        score / tablesample of the curve

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        from verticapy.machine_learning.metrics import FUNCTIONS_DICTIONNARY

        y_true, y_score = self.format_colnames(y_true, y_score)
        fun = FUNCTIONS_DICTIONNARY[method]
        argv = [y_true, y_score, self.__genSQL__()]
        kwds = {}
        if method in ("accuracy", "acc"):
            kwds["pos_label"] = None
        elif method in ("best_cutoff", "best_threshold"):
            kwds["nbins"] = nbins
            kwds["best_threshold"] = True
        elif method in ("roc_curve", "roc", "prc_curve", "prc", "lift_chart", "lift"):
            kwds["nbins"] = nbins
        return FUNCTIONS_DICTIONNARY[method](*argv, **kwds)

    def shape(self):
        """
    Returns the number of rows and columns of the vDataFrame.

    Returns
    -------
    tuple
        (number of lines, number of columns)
        """
        m = len(self.get_columns())
        pre_comp = self.__get_catalog_value__("VERTICAPY_COUNT")
        if pre_comp != "VERTICAPY_NOT_PRECOMPUTED":
            return (pre_comp, m)
        self._VERTICAPY_VARIABLES_["count"] = _executeSQL(
            query=f"""
                SELECT 
                    /*+LABEL('vDataframe.shape')*/ COUNT(*) 
                FROM {self.__genSQL__()} LIMIT 1
            """,
            title="Computing the total number of elements (COUNT(*))",
            method="fetchfirstelem",
            sql_push_ext=self._VERTICAPY_VARIABLES_["sql_push_ext"],
            symbol=self._VERTICAPY_VARIABLES_["symbol"],
        )
        return (self._VERTICAPY_VARIABLES_["count"], m)

    @save_verticapy_logs
    def sort(self, columns: Union[str, dict, list]):
        """
    Sorts the vDataFrame using the input vColumns.

    Parameters
    ----------
    columns: str / dict / list
        List of the vColumns to use to sort the data using asc order or
        dictionary of all sorting methods. For example, to sort by "column1"
        ASC and "column2" DESC, write {"column1": "asc", "column2": "desc"}

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame.append  : Merges the vDataFrame with another relation.
    vDataFrame.groupby : Aggregates the vDataFrame.
    vDataFrame.join    : Joins the vDataFrame with another relation.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns = self.format_colnames(columns)
        max_pos = 0
        columns_tmp = [elem for elem in self._VERTICAPY_VARIABLES_["columns"]]
        for column in columns_tmp:
            max_pos = max(max_pos, len(self[column].transformations) - 1)
        self._VERTICAPY_VARIABLES_["order_by"][max_pos] = self.__get_sort_syntax__(
            columns
        )
        return self

    @save_verticapy_logs
    def swap(self, column1: Union[int, str], column2: Union[int, str]):
        """
    Swap the two input vColumns.

    Parameters
    ----------
    column1: str / int
        The first vColumn or its index to swap.
    column2: str / int
        The second vColumn or its index to swap.

    Returns
    -------
    vDataFrame
        self
        """
        if isinstance(column1, int):
            assert column1 < self.shape()[1], ParameterError(
                "The parameter 'column1' is incorrect, it is greater or equal "
                f"to the vDataFrame number of columns: {column1}>={self.shape()[1]}"
                "\nWhen this parameter type is 'integer', it must represent the index "
                "of the column to swap."
            )
            column1 = self.get_columns()[column1]
        if isinstance(column2, int):
            assert column2 < self.shape()[1], ParameterError(
                "The parameter 'column2' is incorrect, it is greater or equal "
                f"to the vDataFrame number of columns: {column2}>={self.shape()[1]}"
                "\nWhen this parameter type is 'integer', it must represent the "
                "index of the column to swap."
            )
            column2 = self.get_columns()[column2]
        column1, column2 = self.format_colnames(column1, column2)
        columns = self._VERTICAPY_VARIABLES_["columns"]
        all_cols = {}
        for idx, elem in enumerate(columns):
            all_cols[elem] = idx
        columns[all_cols[column1]], columns[all_cols[column2]] = (
            columns[all_cols[column2]],
            columns[all_cols[column1]],
        )
        return self

    def tail(self, limit: int = 5):
        """
    Returns the tail of the vDataFrame.

    Parameters
    ----------
    limit: int, optional
        Number of elements to display.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.head : Returns the vDataFrame head.
        """
        return self.iloc(limit=limit, offset=-1)

    @save_verticapy_logs
    def train_test_split(
        self,
        test_size: float = 0.33,
        order_by: Union[str, list, dict] = {},
        random_state: int = None,
    ):
        """
    Creates 2 vDataFrame (train/test) which can be to use to evaluate a model.
    The intersection between the train and the test is empty only if a unique
    order is specified.

    Parameters
    ----------
    test_size: float, optional
        Proportion of the test set comparint to the training set.
    order_by: str / dict / list, optional
        List of the vColumns to use to sort the data using asc order or
        dictionary of all sorting methods. For example, to sort by "column1"
        ASC and "column2" DESC, write {"column1": "asc", "column2": "desc"}
        Without this parameter, the seeded random number used to split the data
        into train and test can not garanty that no collision occurs. Use this
        parameter to avoid collisions.
    random_state: int, optional
        Integer used to seed the randomness.

    Returns
    -------
    tuple
        (train vDataFrame, test vDataFrame)
        """
        if isinstance(order_by, str):
            order_by = [order_by]
        order_by = self.__get_sort_syntax__(order_by)
        if not random_state:
            random_state = OPTIONS["random_state"]
        random_seed = (
            random_state
            if isinstance(random_state, int)
            else random.randint(-10e6, 10e6)
        )
        random_func = f"SEEDED_RANDOM({random_seed})"
        q = _executeSQL(
            query=f"""
                SELECT 
                    /*+LABEL('vDataframe.train_test_split')*/ 
                    APPROXIMATE_PERCENTILE({random_func} 
                        USING PARAMETERS percentile = {test_size}) 
                FROM {self.__genSQL__()}""",
            title="Computing the seeded numbers quantile.",
            method="fetchfirstelem",
        )
        test_table = f"""
            (SELECT * 
             FROM {self.__genSQL__()} 
             WHERE {random_func} < {q}{order_by}) x"""
        train_table = f"""
            (SELECT * 
             FROM {self.__genSQL__()} 
             WHERE {random_func} > {q}{order_by}) x"""
        return (
            vDataFrameSQL(relation=train_table),
            vDataFrameSQL(relation=test_table),
        )
