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
import warnings, copy
from collections.abc import Iterable
from typing import Union, Literal

# Other modules
import pandas as pd
import numpy as np

# Jupyter - Optional
try:
    from IPython.display import HTML, display
except:
    pass

# VerticaPy Modules
from verticapy.sql.parsers.pandas import pandas_to_vertica
from verticapy.core.tablesample import tablesample
from verticapy.core.vcolumn import vColumn
from verticapy._utils._collect import save_verticapy_logs
from verticapy.errors import (
    ConnectionError,
    MissingColumn,
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
from verticapy._utils._gen import gen_name
from verticapy.sql.read import vDataFrameSQL, readSQL
from verticapy._utils._sql import _executeSQL
from verticapy.core.str_sql import str_sql
from verticapy.sql._utils._format import (
    quote_ident,
    schema_relation,
    format_schema_table,
    clean_query,
)
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
from verticapy.core.vdataframe.transform_join import vDFTRANSFJOIN
from verticapy.core.vdataframe.machine_learning import vDFML
from verticapy.core.vdataframe.math import vDFMATH
from verticapy.core.vdataframe.sys import vDFSYS
from verticapy.core.vdataframe.typing import vDFTYPING
from verticapy.core.vdataframe.read import vDFREAD


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


class vDataFrame(
    vDFAGG,
    vDFCORR,
    vDFIO,
    vDFROLL,
    vDFPLOT,
    vDFFILTER,
    vDFTRANSFJOIN,
    vDFML,
    vDFMATH,
    vDFSYS,
    vDFTYPING,
    vDFREAD,
):
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
