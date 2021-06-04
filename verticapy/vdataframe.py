# (c) Copyright [2018-2021] Micro Focus or one of its affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# |_     |~) _  _| _  /~\    _ |.
# |_)\/  |_)(_|(_||   \_/|_|(_|||
#    /
#              ____________       ______
#             / __        `\     /     /
#            |  \/         /    /     /
#            |______      /    /     /
#                   |____/    /     /
#          _____________     /     /
#          \           /    /     /
#           \         /    /     /
#            \_______/    /     /
#             ______     /     /
#             \    /    /     /
#              \  /    /     /
#               \/    /     /
#                    /     /
#                   /     /
#                   \    /
#                    \  /
#                     \/
#
#                    _
# \  / _  __|_. _ _ |_)
#  \/ (/_|  | |(_(_|| \/
#                     /
# VerticaPy allows user to create vDataFrames (Virtual Dataframes).
# vDataFrames simplify data exploration, data cleaning and MACHINE LEARNING
# in VERTICA. It is an object which keeps in it all actions that the user
# wants to achieve and execute them when they are needed.
#
# The purpose is to bring the logic to the data and not the opposite !
#
#
# Modules
#
# Standard Python Modules
import random, time, shutil, re, decimal, warnings, pickle, datetime, math
from collections.abc import Iterable
from itertools import combinations_with_replacement

pickle.DEFAULT_PROTOCOL = 4

# VerticaPy Modules
import verticapy
from verticapy.vcolumn import vColumn
from verticapy.utilities import *
from verticapy.connect import read_auto_connect
from verticapy.toolbox import *
from verticapy.errors import *

##
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
# ---#
class vDataFrame:
    """
---------------------------------------------------------------------------
Python object which will keep in mind all user modifications 
in order to use the correct SQL code generation. It will send SQL 
queries to the Vertica DB which will aggregate and return the final 
result. vDataFrame will create for each column of the relation a 
Virtual Column (vColumn) which will store the column alias and all 
the user transformations. Thus, vDataFrame allows to do easy data 
preparation and exploration without modifying the data.

Parameters
----------
input_relation: str
    Relation (View, Table or Temporary Table) to use to create the object.
    To get a specific schema relation, your string must include both
    relation and schema: 'schema.relation' or '"schema"."relation"'.
    You can also use the 'schema' parameter to be less ambiguous.
    In this case input_relation must only be the relation name (it must 
    not include a schema).
cursor: DBcursor, optional
    Vertica database cursor. 
    For a cursor designed by Vertica, search for vertica_python
    For ODBC, search for pyodbc.
    For JDBC, search for jaydebeapi.
    For help, see utilities.vHelp.
dsn: str, optional
    Database DSN. OS File including the DB credentials.
    VERTICAPY will try to create a vertica_python cursor first.
    If it didn't find the library, it will try to create a pyodbc cursor.
    For help, see utilities.vHelp.
usecols: list, optional
    List of columns to use to create the object. As Vertica is a columnar DB
    including less columns makes the process faster. Do not hesitate to not include 
    useless columns.
schema: str, optional
    Relation schema. It can be to use to be less ambiguous and allow to create schema 
    and relation name with dots '.' inside.
empty: bool, optional
    If set to True, the created object will be empty. It can be to use to create customized 
    vDataFrame without going through the initialization check.

Attributes
----------
_VERTICAPY_VARIABLES_: dict
    Dictionary containing all vDataFrame attributes.
        allcols_ind, int      : Int to use to optimize the SQL code generation.
        columns, list         : List of the vColumns names.
        count, int            : Number of elements of the vDataFrame (catalog).
        cursor, DBcursor      : Vertica database cursor.
        dsn, str              : Vertica database DSN.
        exclude_columns, list : vColumns to exclude from the final relation.
        history, list         : vDataFrame history (user modifications).
        input_relation, str   : Name of the vDataFrame.
        main_relation, str    : Relation to use to build the vDataFrame (first floor).
        order_by, dict        : Dictionary of all rules to sort the vDataFrame.
        saving, list          : List to use to reconstruct the vDataFrame.
        schema, str           : Schema of the input relation.
        schema_writing, str   : Schema to use to create temporary tables when needed.
        where, list           : List of all rules to filter the vDataFrame.
vColumns : vColumn
    Each vColumn of the vDataFrame is accessible by entering its name between brackets
    For example to access to "myVC", you can write vDataFrame["myVC"].
    """

    #
    # Special Methods
    #
    # ---#
    def __init__(
        self,
        input_relation: str,
        cursor=None,
        dsn: str = "",
        usecols: list = [],
        schema: str = "",
        empty: bool = False,
    ):
        if isinstance(usecols, str):
            usecols = [usecols]
        check_types(
            [
                ("input_relation", input_relation, [str],),
                ("dsn", dsn, [str],),
                ("usecols", usecols, [list],),
                ("schema", schema, [str],),
                ("empty", empty, [bool],),
            ]
        )
        self._VERTICAPY_VARIABLES_ = {}
        self._VERTICAPY_VARIABLES_["count"] = -1
        self._VERTICAPY_VARIABLES_["allcols_ind"] = -1
        if not (empty):
            if not (cursor) and not (dsn):
                cursor = read_auto_connect().cursor()
            elif not (cursor):
                from verticapy import vertica_conn

                cursor = vertica_conn(dsn).cursor()
            else:
                check_cursor(cursor)
            self._VERTICAPY_VARIABLES_["dsn"] = dsn
            if not (schema):
                schema, input_relation = schema_relation(input_relation)
            self._VERTICAPY_VARIABLES_["schema"] = schema.replace('"', "")
            self._VERTICAPY_VARIABLES_["input_relation"] = input_relation.replace(
                '"', ""
            )
            self._VERTICAPY_VARIABLES_["cursor"] = cursor
            where = (
                " AND LOWER(column_name) IN ({})".format(
                    ", ".join(
                        [
                            "'{}'".format(elem.lower().replace("'", "''"))
                            for elem in usecols
                        ]
                    )
                )
                if (usecols)
                else ""
            )
            query = "SELECT column_name, data_type FROM ((SELECT column_name, data_type, ordinal_position FROM columns WHERE table_name = '{}' AND table_schema = '{}'{})".format(
                self._VERTICAPY_VARIABLES_["input_relation"].replace("'", "''"),
                self._VERTICAPY_VARIABLES_["schema"].replace("'", "''"),
                where,
            )
            query += " UNION (SELECT column_name, data_type, ordinal_position FROM view_columns WHERE table_name = '{}' AND table_schema = '{}'{})) x ORDER BY ordinal_position".format(
                self._VERTICAPY_VARIABLES_["input_relation"].replace("'", "''"),
                self._VERTICAPY_VARIABLES_["schema"].replace("'", "''"),
                where,
            )
            cursor.execute(query)
            columns_dtype = cursor.fetchall()
            columns_dtype = [(str(item[0]), str(item[1])) for item in columns_dtype]
            columns = [
                '"{}"'.format(elem[0].replace('"', "_")) for elem in columns_dtype
            ]
            if not (usecols):
                self._VERTICAPY_VARIABLES_["allcols_ind"] = len(columns)
            assert columns != [], MissingRelation("No table or views '{}' found.".format(self._VERTICAPY_VARIABLES_["input_relation"]))
            self._VERTICAPY_VARIABLES_["columns"] = [elem for elem in columns]
            for col_dtype in columns_dtype:
                column, dtype = col_dtype[0], col_dtype[1]
                if '"' in column:
                    warning_message = "A double quote \" was found in the column {}, its alias was changed using underscores '_' to {}.".format(
                        column, column.replace('"', "_")
                    )
                    warnings.warn(warning_message, Warning)
                new_vColumn = vColumn(
                    '"{}"'.format(column.replace('"', "_")),
                    parent=self,
                    transformations=[
                        (
                            '"{}"'.format(column.replace('"', '""')),
                            dtype,
                            category_from_type(dtype),
                        )
                    ],
                )
                setattr(self, '"{}"'.format(column.replace('"', "_")), new_vColumn)
                setattr(self, column.replace('"', "_"), new_vColumn)
            self._VERTICAPY_VARIABLES_["exclude_columns"] = []
            self._VERTICAPY_VARIABLES_["where"] = []
            self._VERTICAPY_VARIABLES_["order_by"] = {}
            self._VERTICAPY_VARIABLES_["history"] = []
            self._VERTICAPY_VARIABLES_["saving"] = []
            self._VERTICAPY_VARIABLES_["main_relation"] = '"{}"."{}"'.format(
                self._VERTICAPY_VARIABLES_["schema"],
                self._VERTICAPY_VARIABLES_["input_relation"],
            )
            self._VERTICAPY_VARIABLES_["schema_writing"] = ""

    # ---#
    def __abs__(self):
        return self.copy().abs()

    # ---#
    def __ceil__(self, n):
        vdf = self.copy()
        columns = vdf.numcol()
        for elem in columns:
            if vdf[elem].category() == "float":
                vdf[elem].apply_fun(func="ceil", x=n)
        return vdf

    # ---#
    def __floor__(self, n):
        vdf = self.copy()
        columns = vdf.numcol()
        for elem in columns:
            if vdf[elem].category() == "float":
                vdf[elem].apply_fun(func="floor", x=n)
        return vdf

    # ---#
    def __getitem__(self, index):
        if isinstance(index, slice):
            assert index.step in (1, None), ValueError("vDataFrame doesn't allow slicing having steps different than 1.")
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
                limit = " LIMIT {}".format(limit)
            else:
                limit = ""
            query = "(SELECT * FROM {}{} OFFSET {}{}) VERTICAPY_SUBTABLE".format(
                self.__genSQL__(), last_order_by(self), index_start, limit
            )
            return vdf_from_relation(
                query, cursor=self._VERTICAPY_VARIABLES_["cursor"]
            )
        elif isinstance(index, int):
            columns = self.get_columns()
            for idx, elem in enumerate(columns):
                if self[elem].category() == "float":
                    columns[idx] = "{}::float".format(elem)
            if index < 0:
                index += self.shape()[0]
            query = "SELECT {} FROM {}{} OFFSET {} LIMIT 1".format(
                ", ".join(columns), self.__genSQL__(), last_order_by(self), index
            )
            self.__executeSQL__(query=query, title="Gets the vDataFrame element.")
            return self._VERTICAPY_VARIABLES_["cursor"].fetchone()
        elif isinstance(index, (str, str_sql)):
            is_sql = False
            if isinstance(index, vColumn):
                index = index.alias
            elif isinstance(index, str_sql):
                index = str(index)
                is_sql = True
            new_index = vdf_columns_names([index], self)
            try:
                return getattr(self, new_index[0])
            except:
                if is_sql:
                    return self.search(conditions=index)
                else:
                    return getattr(self, index)
        elif isinstance(index, Iterable):
            try:
                return self.select(columns=[str(elem) for elem in index])
            except:
                return self.search(conditions=[str(elem) for elem in index])
        else:
            return getattr(self, index)

    # ---#
    def __iter__(self):
        columns = self.get_columns()
        return (elem for elem in columns)

    # ---#
    def __len__(self):
        return int(self.shape()[0])

    # ---#
    def __nonzero__(self):
        return self.shape()[0] > 0 and not (self.empty())

    # ---#
    def __repr__(self):
        return self.head(limit=verticapy.options["max_rows"]).__repr__()

    # ---#
    def _repr_html_(self):
        return self.head(limit=verticapy.options["max_rows"])._repr_html_()

    # ---#
    def __round__(self, n):
        vdf = self.copy()
        columns = vdf.numcol()
        for elem in columns:
            if vdf[elem].category() == "float":
                vdf[elem].apply_fun(func="round", x=n)
        return vdf

    # ---#
    def __setattr__(self, attr, val):
        if isinstance(val, (str, str_sql)) and not isinstance(val, vColumn):
            val = str(val)
            if column_check_ambiguous(attr, self.get_columns()):
                self[attr].apply(func=val)
            else:
                self.eval(name=attr, expr=val)
        else:
            self.__dict__[attr] = val

    # ---#
    def __setitem__(self, index, val):
        setattr(self, index, val)

    # ---#
    def __add_to_history__(self, message: str):
        """
    ---------------------------------------------------------------------------
    VERTICAPY stores the user modification and help the user to look at 
    what he/she did. This method is to use to add a customized message in the 
    vDataFrame history attribute.
        """
        check_types([("message", message, [str],)])
        self._VERTICAPY_VARIABLES_["history"] += [
            "{}{}{} {}".format("{", time.strftime("%c"), "}", message)
        ]
        return self

    # ---#
    def __aggregate_matrix__(
        self,
        method: str = "pearson",
        columns: list = [],
        round_nb: int = 3,
        show: bool = True,
        ax=None,
        **style_kwds,
    ):
        """
    ---------------------------------------------------------------------------
    Global method to use to compute the Correlation/Cov/Regr Matrix.

    See Also
    --------
    vDataFrame.corr : Computes the Correlation Matrix of the vDataFrame.
    vDataFrame.cov  : Computes the covariance matrix of the vDataFrame.
    vDataFrame.regr : Computes the regression matrix of the vDataFrame.
        """
        method_name = "Correlation"
        method_type = " using the method = '{}'".format(method)
        if method == "cov":
            method_name = "Covariance"
            method_type = ""
        columns = vdf_columns_names(columns, self)
        if method != "cramer":
            for column in columns:
                assert self[column].isnum(), TypeError(f"vColumn {column} must be numerical to compute the {method_name} Matrix{method_type}.")
        if len(columns) == 1:
            if method in ("pearson", "spearman", "kendall", "biserial", "cramer"):
                return 1.0
            elif method == "cov":
                return self[columns[0]].var()
        elif len(columns) == 2:
            pre_comp_val = self.__get_catalog_value__(method=method, columns=columns)
            if pre_comp_val != "VERTICAPY_NOT_PRECOMPUTED":
                return pre_comp_val
            cast_0 = "::int" if (self[columns[0]].isbool()) else ""
            cast_1 = "::int" if (self[columns[1]].isbool()) else ""
            if method in ("pearson", "spearman"):
                if columns[1] == columns[0]:
                    return 1
                table = (
                    self.__genSQL__()
                    if (method == "pearson")
                    else "(SELECT RANK() OVER (ORDER BY {}) AS {}, RANK() OVER (ORDER BY {}) AS {} FROM {}) rank_spearman_table".format(
                        columns[0],
                        columns[0],
                        columns[1],
                        columns[1],
                        self.__genSQL__(),
                    )
                )
                query = "SELECT CORR({}{}, {}{}) FROM {}".format(
                    columns[0], cast_0, columns[1], cast_1, table
                )
                title = "Computes the {} correlation between {} and {}.".format(
                    method, columns[0], columns[1]
                )
            elif method == "biserial":
                if columns[1] == columns[0]:
                    return 1
                elif (self[columns[1]].category() != "int") and (
                    self[columns[0]].category() != "int"
                ):
                    return float("nan")
                elif self[columns[1]].category() == "int":
                    if not (self[columns[1]].isbool()):
                        agg = (
                            self[columns[1]]
                            .aggregate(["approx_unique", "min", "max"])
                            .values[columns[1]]
                        )
                        if (agg[0] != 2) or (agg[1] != 0) or (agg[2] != 1):
                            return float("nan")
                    column_b, column_n = columns[1], columns[0]
                    cast_b, cast_n = cast_1, cast_0
                elif self[columns[0]].category() == "int":
                    if not (self[columns[0]].isbool()):
                        agg = (
                            self[columns[0]]
                            .aggregate(["approx_unique", "min", "max"])
                            .values[columns[0]]
                        )
                        if (agg[0] != 2) or (agg[1] != 0) or (agg[2] != 1):
                            return float("nan")
                    column_b, column_n = columns[0], columns[1]
                    cast_b, cast_n = cast_0, cast_1
                else:
                    return float("nan")
                query = "SELECT (AVG(DECODE({}{}, 1, {}{}, NULL)) - AVG(DECODE({}{}, 0, {}{}, NULL))) / STDDEV({}{}) * SQRT(SUM({}{}) * SUM(1 - {}{}) / COUNT(*) / COUNT(*)) FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL;".format(
                    column_b,
                    cast_b,
                    column_n,
                    cast_n,
                    column_b,
                    cast_b,
                    column_n,
                    cast_n,
                    column_n,
                    cast_n,
                    column_b,
                    cast_b,
                    column_b,
                    cast_b,
                    self.__genSQL__(),
                    column_n,
                    column_b,
                )
                title = "Computes the biserial correlation between {} and {}.".format(
                    column_b, column_n
                )
            elif method == "cramer":
                if columns[1] == columns[0]:
                    return 1
                table_0_1 = "SELECT {}, {}, COUNT(*) AS nij FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL GROUP BY 1, 2".format(
                    columns[0], columns[1], self.__genSQL__(), columns[0], columns[1]
                )
                table_0 = "SELECT {}, COUNT(*) AS ni FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL GROUP BY 1".format(
                    columns[0], self.__genSQL__(), columns[0], columns[1]
                )
                table_1 = "SELECT {}, COUNT(*) AS nj FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL GROUP BY 1".format(
                    columns[1], self.__genSQL__(), columns[0], columns[1]
                )
                sql = "SELECT COUNT(*) AS n, APPROXIMATE_COUNT_DISTINCT({}) AS k, APPROXIMATE_COUNT_DISTINCT({}) AS r FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL".format(
                    columns[0], columns[1], self.__genSQL__(), columns[0], columns[1]
                )
                self._VERTICAPY_VARIABLES_["cursor"].execute(sql)
                n, k, r = self._VERTICAPY_VARIABLES_["cursor"].fetchone()
                chi2 = "SELECT SUM((nij - ni * nj / {}) * (nij - ni * nj / {}) / ((ni * nj) / {})) AS chi2 FROM (SELECT * FROM ({}) table_0_1 LEFT JOIN ({}) table_0 ON table_0_1.{} = table_0.{}) x LEFT JOIN ({}) table_1 ON x.{} = table_1.{}".format(
                    n,
                    n,
                    n,
                    table_0_1,
                    table_0,
                    columns[0],
                    columns[0],
                    table_1,
                    columns[1],
                    columns[1],
                )
                self.__executeSQL__(
                    chi2,
                    title="Computes the CramerV correlation between {} and {} (Chi2 Statistic).".format(
                        columns[0], columns[1]
                    ),
                )
                result = self._VERTICAPY_VARIABLES_["cursor"].fetchone()[0]
                if min(k - 1, r - 1) == 0:
                    result = float("nan")
                else:
                    result = float(math.sqrt(result / n / min(k - 1, r - 1)))
                    if result > 1 or result < 0:
                        result = float("nan")
                return result
            elif method == "kendall":
                if columns[1] == columns[0]:
                    return 1
                n_ = "SQRT(COUNT(*))"
                n_c = "(SUM(((x.{}{} < y.{}{} AND x.{}{} < y.{}{}) OR (x.{}{} > y.{}{} AND x.{}{} > y.{}{}))::int))/2".format(
                    columns[0],
                    cast_0,
                    columns[0],
                    cast_0,
                    columns[1],
                    cast_1,
                    columns[1],
                    cast_1,
                    columns[0],
                    cast_0,
                    columns[0],
                    cast_0,
                    columns[1],
                    cast_1,
                    columns[1],
                    cast_1,
                )
                n_d = "(SUM(((x.{}{} > y.{}{} AND x.{}{} < y.{}{}) OR (x.{}{} < y.{}{} AND x.{}{} > y.{}{}))::int))/2".format(
                    columns[0],
                    cast_0,
                    columns[0],
                    cast_0,
                    columns[1],
                    cast_1,
                    columns[1],
                    cast_1,
                    columns[0],
                    cast_0,
                    columns[0],
                    cast_0,
                    columns[1],
                    cast_1,
                    columns[1],
                    cast_1,
                )
                n_1 = "(SUM((x.{}{} = y.{}{})::int)-{})/2".format(
                    columns[0], cast_0, columns[0], cast_0, n_
                )
                n_2 = "(SUM((x.{}{} = y.{}{})::int)-{})/2".format(
                    columns[1], cast_1, columns[1], cast_1, n_
                )
                n_0 = f"{n_} * ({n_} - 1)/2"
                tau_b = f"({n_c} - {n_d}) / sqrt(({n_0} - {n_1}) * ({n_0} - {n_2}))"
                query = "SELECT {} FROM (SELECT {}, {} FROM {}) x CROSS JOIN (SELECT {}, {} FROM {}) y"
                query = query.format(
                    tau_b,
                    columns[0],
                    columns[1],
                    self.__genSQL__(),
                    columns[0],
                    columns[1],
                    self.__genSQL__(),
                )
                title = "Computes the kendall correlation between {} and {}.".format(
                    columns[0], columns[1]
                )
            elif method == "cov":
                query = "SELECT COVAR_POP({}{}, {}{}) FROM {}".format(
                    columns[0], cast_0, columns[1], cast_1, self.__genSQL__()
                )
                title = "Computes the covariance between {} and {}.".format(
                    columns[0], columns[1]
                )
            try:
                self.__executeSQL__(query=query, title=title)
                result = self._VERTICAPY_VARIABLES_["cursor"].fetchone()[0]
            except:
                result = float("nan")
            self.__update_catalog__(
                values={columns[1]: result}, matrix=method, column=columns[0]
            )
            self.__update_catalog__(
                values={columns[0]: result}, matrix=method, column=columns[1]
            )
            if isinstance(result, decimal.Decimal):
                result = float(result)
            return result
        elif len(columns) > 2:
            try:
                nb_precomputed, n = 0, len(columns)
                for column1 in columns:
                    for column2 in columns:
                        pre_comp_val = self.__get_catalog_value__(
                            method=method, columns=[column1, column2]
                        )
                        if pre_comp_val != "VERTICAPY_NOT_PRECOMPUTED":
                            nb_precomputed += 1
                assert (nb_precomputed <= n * n / 3) and (method in ("pearson", "spearman"))
                table = (
                    self.__genSQL__()
                    if (method == "pearson")
                    else "(SELECT {} FROM {}) spearman_table".format(
                        ", ".join(
                            [
                                "RANK() OVER (ORDER BY {}) AS {}".format(
                                    column, column
                                )
                                for column in columns
                            ]
                        ),
                        self.__genSQL__(),
                    )
                )
                version(
                    cursor=self._VERTICAPY_VARIABLES_["cursor"], condition=[9, 2, 1]
                )
                self.__executeSQL__(
                    query="SELECT CORR_MATRIX({}) OVER () FROM {}".format(
                        ", ".join(columns), table
                    ),
                    title="Computes the {} Corr Matrix.".format(method),
                )
                result = self._VERTICAPY_VARIABLES_["cursor"].fetchall()
                corr_dict = {}
                for idx, column in enumerate(columns):
                    corr_dict[column] = idx
                n = len(columns)
                matrix = [[1 for i in range(0, n + 1)] for i in range(0, n + 1)]
                for elem in result:
                    i, j = (
                        corr_dict[str_column(elem[0])],
                        corr_dict[str_column(elem[1])],
                    )
                    matrix[i + 1][j + 1] = elem[2]
                matrix[0] = [""] + columns
                for idx, column in enumerate(columns):
                    matrix[idx + 1][0] = column
                title = "Correlation Matrix ({})".format(method)
            except:
                if method in ("pearson", "spearman", "kendall", "biserial", "cramer"):
                    title_query = "Compute all Correlations in a single query"
                    title = "Correlation Matrix ({})".format(method)
                    if method == "biserial":
                        i0, step = 0, 1
                    else:
                        i0, step = 1, 0
                elif method == "cov":
                    title_query = "Compute all covariances in a single query"
                    title = "Covariance Matrix"
                    i0, step = 0, 1
                n = len(columns)
                if verticapy.options["tqdm"]:
                    from tqdm.auto import tqdm

                    loop = tqdm(range(i0, n))
                else:
                    loop = range(i0, n)
                try:
                    all_list = []
                    nb_precomputed = 0
                    nb_loop = 0
                    for i in loop:
                        for j in range(0, i + step):
                            nb_loop += 1
                            cast_i = "::int" if (self[columns[i]].isbool()) else ""
                            cast_j = "::int" if (self[columns[j]].isbool()) else ""
                            pre_comp_val = self.__get_catalog_value__(
                                method=method, columns=[columns[i], columns[j]]
                            )
                            if pre_comp_val == None or pre_comp_val != pre_comp_val:
                                pre_comp_val = "NULL"
                            if pre_comp_val != "VERTICAPY_NOT_PRECOMPUTED":
                                all_list += [str(pre_comp_val)]
                                nb_precomputed += 1
                            elif method in ("pearson", "spearman"):
                                all_list += [
                                    "ROUND(CORR({}{}, {}{}), {})".format(
                                        columns[i], cast_i, columns[j], cast_j, round_nb
                                    )
                                ]
                            elif method == "kendall":
                                n_ = "SQRT(COUNT(*))"
                                n_c = "(SUM(((x.{}{} < y.{}{} AND x.{}{} < y.{}{}) OR (x.{}{} > y.{}{} AND x.{}{} > y.{}{}))::int))/2".format(
                                    columns[i],
                                    cast_i,
                                    columns[i],
                                    cast_i,
                                    columns[j],
                                    cast_j,
                                    columns[j],
                                    cast_j,
                                    columns[i],
                                    cast_i,
                                    columns[i],
                                    cast_i,
                                    columns[j],
                                    cast_j,
                                    columns[j],
                                    cast_j,
                                )
                                n_d = "(SUM(((x.{}{} > y.{}{} AND x.{}{} < y.{}{}) OR (x.{}{} < y.{}{} AND x.{}{} > y.{}{}))::int))/2".format(
                                    columns[i],
                                    cast_i,
                                    columns[i],
                                    cast_i,
                                    columns[j],
                                    cast_j,
                                    columns[j],
                                    cast_j,
                                    columns[i],
                                    cast_i,
                                    columns[i],
                                    cast_i,
                                    columns[j],
                                    cast_j,
                                    columns[j],
                                    cast_j,
                                )
                                n_1 = "(SUM((x.{}{} = y.{}{})::int)-{})/2".format(
                                    columns[i], cast_i, columns[i], cast_i, n_
                                )
                                n_2 = "(SUM((x.{}{} = y.{}{})::int)-{})/2".format(
                                    columns[j], cast_j, columns[j], cast_j, n_
                                )
                                n_0 = f"{n_} * ({n_} - 1)/2"
                                tau_b = f"({n_c} - {n_d}) / sqrt(({n_0} - {n_1}) * ({n_0} - {n_2}))"
                                all_list += [tau_b]
                            elif method == "cov":
                                all_list += [
                                    "COVAR_POP({}{}, {}{})".format(
                                        columns[i], cast_i, columns[j], cast_j
                                    )
                                ]
                            else:
                                raise
                    if method == "spearman":
                        rank = [
                            "RANK() OVER (ORDER BY {}) AS {}".format(column, column)
                            for column in columns
                        ]
                        table = "(SELECT {} FROM {}) rank_spearman_table".format(
                            ", ".join(rank), self.__genSQL__()
                        )
                    elif method == "kendall":
                        table = "(SELECT {} FROM {}) x CROSS JOIN (SELECT {} FROM {}) y".format(
                            ", ".join(columns),
                            self.__genSQL__(),
                            ", ".join(columns),
                            self.__genSQL__(),
                        )
                    else:
                        table = self.__genSQL__()
                    if nb_precomputed == nb_loop:
                        self._VERTICAPY_VARIABLES_["cursor"].execute(
                            "SELECT {}".format(", ".join(all_list))
                        )
                    else:
                        self.__executeSQL__(
                            query="SELECT {} FROM {}".format(
                                ", ".join(all_list), table
                            ),
                            title=title_query,
                        )
                    result = self._VERTICAPY_VARIABLES_["cursor"].fetchone()
                except:
                    n = len(columns)
                    result = []
                    for i in loop:
                        for j in range(0, i + step):
                            result += [
                                self.__aggregate_matrix__(
                                    method, [columns[i], columns[j]]
                                )
                            ]
                matrix = [[1 for i in range(0, n + 1)] for i in range(0, n + 1)]
                matrix[0] = [""] + columns
                for i in range(0, n + 1):
                    matrix[i][0] = columns[i - 1]
                k = 0
                for i in range(i0, n):
                    for j in range(0, i + step):
                        current = result[k]
                        k += 1
                        if current == None:
                            current = float("nan")
                        matrix[i + 1][j + 1] = current
                        matrix[j + 1][i + 1] = current
            if show:
                from verticapy.plot import cmatrix

                vmin = 0 if (method == "cramer") else -1
                if method in ("cov"):
                    vmin = None
                vmax = (
                    1
                    if (
                        method
                        in ("pearson", "spearman", "kendall", "biserial", "cramer")
                    )
                    else None
                )
                if "cmap" not in style_kwds:
                    from verticapy.plot import gen_cmap

                    cm1, cm2 = gen_cmap()
                    cmap = cm1 if (method == "cramer") else cm2
                    style_kwds["cmap"] = cmap
                cmatrix(
                    matrix,
                    columns,
                    columns,
                    n,
                    n,
                    vmax=vmax,
                    vmin=vmin,
                    title=title,
                    mround=round_nb,
                    ax=ax,
                    **style_kwds,
                )
            values = {"index": matrix[0][1 : len(matrix[0])]}
            del matrix[0]
            for column in matrix:
                values[column[0]] = column[1 : len(column)]
            for column1 in values:
                if column1 != "index":
                    val = {}
                    for idx, column2 in enumerate(values["index"]):
                        val[column2] = values[column1][idx]
                    self.__update_catalog__(values=val, matrix=method, column=column1)
            for elem in values:
                if elem != "index":
                    for idx in range(len(values[elem])):
                        if isinstance(values[elem][idx], decimal.Decimal):
                            values[elem][idx] = float(values[elem][idx])
            return tablesample(values=values)
        else:
            if method == "cramer":
                cols = self.catcol()
                assert len(cols) != 0, EmptyParameter("No categorical column found in the vDataFrame.")
            else:
                cols = self.numcol()
                assert len(cols) != 0, EmptyParameter("No numerical column found in the vDataFrame.")
            return self.__aggregate_matrix__(
                method=method, columns=cols, round_nb=round_nb, show=show, **style_kwds,
            )

    # ---#
    def __aggregate_vector__(
        self,
        focus: str,
        method: str = "pearson",
        columns: list = [],
        round_nb: int = 3,
        show: bool = True,
        ax=None,
        **style_kwds,
    ):
        """
    ---------------------------------------------------------------------------
    Global method to use to compute the Correlation/Cov/Beta Vector.

    See Also
    --------
    vDataFrame.corr : Computes the Correlation Matrix of the vDataFrame.
    vDataFrame.cov  : Computes the covariance matrix of the vDataFrame.
    vDataFrame.regr : Computes the regression matrix of the vDataFrame.
        """
        if not (columns):
            if method == "cramer":
                cols = self.catcol()
                assert cols, EmptyParameter("No categorical column found in the vDataFrame.")
            else:
                cols = self.numcol()
                assert cols, EmptyParameter("No numerical column found in the vDataFrame.")
        else:
            cols = vdf_columns_names(columns, self)
        if method != "cramer":
            method_name = "Correlation"
            method_type = " using the method = '{}'".format(method)
            if method == "cov":
                method_name = "Covariance"
                method_type = ""
            for column in cols:
                assert self[column].isnum(), TypeError(f"vColumn {column} must be numerical to compute the {method_name} Vector{method_type}.")
        if method in ("spearman", "pearson", "kendall", "cov") and (len(cols) >= 1):
            try:
                fail = 0
                cast_i = "::int" if (self[focus].isbool()) else ""
                all_list, all_cols = [], [focus]
                nb_precomputed = 0
                for column in cols:
                    if (
                        column.replace('"', "").lower()
                        != focus.replace('"', "").lower()
                    ):
                        all_cols += [column]
                    cast_j = "::int" if (self[column].isbool()) else ""
                    pre_comp_val = self.__get_catalog_value__(
                        method=method, columns=[focus, column]
                    )
                    if pre_comp_val == None or pre_comp_val != pre_comp_val:
                        pre_comp_val = "NULL"
                    if pre_comp_val != "VERTICAPY_NOT_PRECOMPUTED":
                        all_list += [str(pre_comp_val)]
                        nb_precomputed += 1
                    elif method in ("pearson", "spearman"):
                        all_list += [
                            "ROUND(CORR({}{}, {}{}), {})".format(
                                focus, cast_i, column, cast_j, round_nb
                            )
                        ]
                    elif method == "kendall":
                        n = "SQRT(COUNT(*))"
                        n_c = "(SUM(((x.{}{} < y.{}{} AND x.{}{} < y.{}{}) OR (x.{}{} > y.{}{} AND x.{}{} > y.{}{}))::int))/2".format(
                            focus,
                            cast_i,
                            focus,
                            cast_i,
                            column,
                            cast_j,
                            column,
                            cast_j,
                            focus,
                            cast_i,
                            focus,
                            cast_i,
                            column,
                            cast_j,
                            column,
                            cast_j,
                        )
                        n_d = "(SUM(((x.{}{} > y.{}{} AND x.{}{} < y.{}{}) OR (x.{}{} < y.{}{} AND x.{}{} > y.{}{}))::int))/2".format(
                            focus,
                            cast_i,
                            focus,
                            cast_i,
                            column,
                            cast_j,
                            column,
                            cast_j,
                            focus,
                            cast_i,
                            focus,
                            cast_i,
                            column,
                            cast_j,
                            column,
                            cast_j,
                        )
                        n_1 = "(SUM((x.{}{} = y.{}{})::int)-{})/2".format(
                            focus, cast_i, focus, cast_i, n
                        )
                        n_2 = "(SUM((x.{}{} = y.{}{})::int)-{})/2".format(
                            column, cast_j, column, cast_j, n
                        )
                        n_0 = f"{n} * ({n} - 1)/2"
                        tau_b = (
                            f"({n_c} - {n_d}) / sqrt(({n_0} - {n_1}) * ({n_0} - {n_2}))"
                        )
                        all_list += [tau_b]
                    elif method == "cov":
                        all_list += [
                            "COVAR_POP({}{}, {}{})".format(
                                focus, cast_i, column, cast_j
                            )
                        ]
                if method == "spearman":
                    rank = [
                        "RANK() OVER (ORDER BY {}) AS {}".format(column, column)
                        for column in all_cols
                    ]
                    table = "(SELECT {} FROM {}) rank_spearman_table".format(
                        ", ".join(rank), self.__genSQL__()
                    )
                elif method == "kendall":
                    table = "(SELECT {} FROM {}) x CROSS JOIN (SELECT {} FROM {}) y".format(
                        ", ".join(all_cols),
                        self.__genSQL__(),
                        ", ".join(all_cols),
                        self.__genSQL__(),
                    )
                else:
                    table = self.__genSQL__()
                if nb_precomputed == len(cols):
                    self._VERTICAPY_VARIABLES_["cursor"].execute(
                        "SELECT {}".format(", ".join(all_list))
                    )
                else:
                    self.__executeSQL__(
                        query="SELECT {} FROM {} LIMIT 1".format(
                            ", ".join(all_list), table
                        ),
                        title="Computes the Correlation Vector ({})".format(method),
                    )
                result = self._VERTICAPY_VARIABLES_["cursor"].fetchone()
                vector = [elem for elem in result]
            except:
                fail = 1
        if not (
            method in ("spearman", "pearson", "kendall", "cov") and (len(cols) >= 1)
        ) or (fail):
            vector = []
            for column in cols:
                if column.replace('"', "").lower() == focus.replace('"', "").lower():
                    vector += [1]
                else:
                    vector += [
                        self.__aggregate_matrix__(
                            method=method, columns=[column, focus]
                        )
                    ]
        vector = [0 if (elem == None) else elem for elem in vector]
        data = [(cols[i], vector[i]) for i in range(len(vector))]
        data.sort(key=lambda tup: abs(tup[1]), reverse=True)
        cols, vector = [elem[0] for elem in data], [elem[1] for elem in data]
        if show:
            from verticapy.plot import cmatrix

            vmin = 0 if (method == "cramer") else -1
            if method in ("cov"):
                vmin = None
            vmax = (
                1
                if (method in ("pearson", "spearman", "kendall", "biserial", "cramer"))
                else None
            )
            if "cmap" not in style_kwds:
                from verticapy.plot import gen_cmap

                cm1, cm2 = gen_cmap()
                cmap = cm1 if (method == "cramer") else cm2
                style_kwds["cmap"] = cmap
            title = "Correlation Vector of {} ({})".format(focus, method)
            cmatrix(
                [cols, [focus] + vector],
                cols,
                [focus],
                len(cols),
                1,
                vmax=vmax,
                vmin=vmin,
                title=title,
                mround=round_nb,
                is_vector=True,
                ax=ax,
                **style_kwds,
            )
        for idx, column in enumerate(cols):
            self.__update_catalog__(
                values={focus: vector[idx]}, matrix=method, column=column
            )
            self.__update_catalog__(
                values={column: vector[idx]}, matrix=method, column=focus
            )
        for idx in range(len(vector)):
            if isinstance(vector[idx], decimal.Decimal):
                vector[idx] = float(vector[idx])
        return tablesample(values={"index": cols, focus: vector})

    # ---#
    def __executeSQL__(self, query: str, title: str = ""):
        """
    ---------------------------------------------------------------------------
    Executes the input SQL Query.

    Parameters
    ----------
    query: str
        Input query.
    title: str, optional
        Query title. It is the tip to use to indicate the query meaning when
        turning on the SQL using the 'set_option' function. 

    Returns
    -------
    DBcursor
        The database cursor.
        """
        return executeSQL(self._VERTICAPY_VARIABLES_["cursor"], query, title,)

    # ---#
    def __genSQL__(
        self, split: bool = False, transformations: dict = {}, force_columns: list = [],
    ):
        """
    ---------------------------------------------------------------------------
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
        if not (force_columns):
            force_columns = [elem for elem in self._VERTICAPY_VARIABLES_["columns"]]
        for column in force_columns:
            all_imputations_grammar += [
                [item[0] for item in self[column].transformations]
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
            " AND ".join(["({})".format(elem) for elem in item]) for item in all_where
        ]
        for i in range(len(all_where)):
            if all_where[i] != "":
                all_where[i] = " WHERE {}".format(all_where[i])
        # We compute the first floor
        columns = force_columns + [column for column in transformations]
        first_values = [item[0] for item in all_imputations_grammar]
        transformations_first_floor = False
        for i in range(0, len(first_values)):
            if (first_values[i] != "___VERTICAPY_UNDEFINED___") and (
                first_values[i] != columns[i]
            ):
                first_values[i] = "{} AS {}".format(first_values[i], columns[i])
                transformations_first_floor = True
        if (transformations_first_floor) or (
            self._VERTICAPY_VARIABLES_["allcols_ind"] != len(first_values)
        ):
            table = "SELECT {} FROM {}".format(
                ", ".join(first_values), self._VERTICAPY_VARIABLES_["main_relation"]
            )
        else:
            table = "SELECT * FROM {}".format(
                self._VERTICAPY_VARIABLES_["main_relation"]
            )
        # We compute the other floors
        for i in range(1, max_transformation_floor):
            values = [item[i] for item in all_imputations_grammar]
            for j in range(0, len(values)):
                if values[j] == "{}":
                    values[j] = columns[j]
                elif values[j] != "___VERTICAPY_UNDEFINED___":
                    values[j] = "{} AS {}".format(
                        values[j].replace("{}", columns[j]), columns[j]
                    )
            table = "SELECT {} FROM ({}) VERTICAPY_SUBTABLE".format(
                ", ".join(values), table
            )
            if len(all_where) > i - 1:
                table += all_where[i - 1]
            if (i - 1) in self._VERTICAPY_VARIABLES_["order_by"]:
                table += self._VERTICAPY_VARIABLES_["order_by"][i - 1]
        where_final = (
            all_where[max_transformation_floor - 1]
            if (len(all_where) > max_transformation_floor - 1)
            else ""
        )
        # Only the last order_by matters as the order_by will never change the final relation
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
        random_func = random_function()
        split = ", {} AS __verticapy_split__".format(random_func) if (split) else ""
        if (where_final == "") and (order_final == ""):
            if split:
                table = "SELECT *{} FROM ({}) VERTICAPY_SUBTABLE".format(split, table)
            table = "({}) VERTICAPY_SUBTABLE".format(table)
        else:
            table = "({}) VERTICAPY_SUBTABLE{}{}".format(
                table, where_final, order_final
            )
            table = "(SELECT *{} FROM {}) VERTICAPY_SUBTABLE".format(split, table)
        if (self._VERTICAPY_VARIABLES_["exclude_columns"]) and not (split):
            table = "(SELECT {}{} FROM {}) VERTICAPY_SUBTABLE".format(
                ", ".join(self.get_columns()), split, table
            )
        main_relation = self._VERTICAPY_VARIABLES_["main_relation"]
        all_main_relation = "(SELECT * FROM {}) VERTICAPY_SUBTABLE".format(
            main_relation
        )
        table = table.replace(all_main_relation, main_relation)
        return table

    # ---#
    def __get_catalog_value__(
        self, column: str = "", key: str = "", method: str = "", columns: list = []
    ):
        """
    ---------------------------------------------------------------------------
    VERTICAPY stores the already computed aggregations to avoid useless 
    computations. This method returns the stored aggregation if it was already 
    computed.
        """
        if not (verticapy.options["cache"]):
            return "VERTICAPY_NOT_PRECOMPUTED"
        if column == "VERTICAPY_COUNT":
            if self._VERTICAPY_VARIABLES_["count"] < 0:
                return "VERTICAPY_NOT_PRECOMPUTED"
            total = self._VERTICAPY_VARIABLES_["count"]
            if not (isinstance(total, (int, float))):
                return "VERTICAPY_NOT_PRECOMPUTED"
            return total
        elif method:
            method = str_function(method.lower())
            if columns[1] in self[columns[0]].catalog[method]:
                return self[columns[0]].catalog[method][columns[1]]
            else:
                return "VERTICAPY_NOT_PRECOMPUTED"
        key = str_function(key.lower())
        column = vdf_columns_names([column], self)[0]
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

    # ---#
    def __update_catalog__(
        self,
        values: dict = {},
        erase: bool = False,
        columns: list = [],
        matrix: str = "",
        column: str = "",
    ):
        """
    ---------------------------------------------------------------------------
    VERTICAPY stores the already computed aggregations to avoid useless 
    computations. This method stores the input aggregation in the vColumn catalog.
        """
        columns = vdf_columns_names(columns, self)
        if erase:
            if not (columns):
                columns = self.get_columns()
            for column in columns:
                self[column].catalog = {
                    "cov": {},
                    "pearson": {},
                    "spearman": {},
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
            self._VERTICAPY_VARIABLES_["count"] = -1
        elif matrix:
            matrix = str_function(matrix.lower())
            if matrix in [
                "cov",
                "pearson",
                "spearman",
                "kendall",
                "cramer",
                "biserial",
                "regr_avgx",
                "regr_avgy",
                "regr_count",
                "regr_intercept",
                "regr_r2",
                "regr_slope",
                "regr_sxx",
                "regr_sxy",
                "regr_syy",
            ]:
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
                        key = str_function(key)
                        try:
                            val = float(val)
                            if val - int(val) == 0:
                                val = int(val)
                        except:
                            pass
                        if val != val:
                            val = None
                        self[column].catalog[key] = val

    # ---#
    def __vdf_from_relation__(self, table: str, func: str, history: str):
        """
    ---------------------------------------------------------------------------
    This method is to use to build a vDataFrame based on a relation
        """
        check_types(
            [
                ("table", table, [str],),
                ("func", func, [str],),
                ("history", history, [str],),
            ]
        )
        cursor = self._VERTICAPY_VARIABLES_["cursor"]
        dsn = self._VERTICAPY_VARIABLES_["dsn"]
        schema = self._VERTICAPY_VARIABLES_["schema"]
        schema_writing = self._VERTICAPY_VARIABLES_["schema_writing"]
        history = self._VERTICAPY_VARIABLES_["history"] + [history]
        saving = self._VERTICAPY_VARIABLES_["saving"]
        return vdf_from_relation(
            table, func, cursor, dsn, schema, schema_writing, history, saving,
        )

    #
    # Methods
    #
    # ---#
    def aad(self, columns: list = []):
        """
    ---------------------------------------------------------------------------
    Aggregates the vDataFrame using 'aad' (Average Absolute Deviation).

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        return self.aggregate(func=["aad"], columns=columns)

    # ---#
    def abs(self, columns: list = []):
        """
    ---------------------------------------------------------------------------
    Applies the absolute value function to all input vColumns. 

    Parameters
    ----------
    columns: list, optional
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
        check_types([("columns", columns, [list],)])
        columns_check(columns, self)
        columns = self.numcol() if not (columns) else vdf_columns_names(columns, self)
        func = {}
        for column in columns:
            if not (self[column].isbool()):
                func[column] = "ABS({})"
        return self.apply(func)

    # ---#
    def add_duplicates(
        self, weight: (int, str), use_gcd: bool = True,
    ):
        """
    ---------------------------------------------------------------------------
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
        check_types(
            [("weight", weight, [str, int,],),
             ("use_gcd", use_gcd, [bool,])]
        )
        if isinstance(weight, str):
            columns_check([weight], self)
            weight = vdf_columns_names([weight], self)[0]
            assert self[weight].category() == "int", TypeError("The weight vColumn category must be 'integer', found {}.".format(self[weight].category()))
            L = sorted(self[weight].distinct())
            gcd, max_value, n = L[0], L[-1], len(L)
            assert gcd >= 0, ValueError("The weight vColumn must only include positive integers.")
            if use_gcd:
                if gcd != 1:
                    for i in range(1, n):
                        if gcd != 1:
                            gcd = math.gcd(gcd, L[i])
                        else:
                            break
            else:
                gcd = 1
            columns = self.get_columns(exclude_columns = [weight])
            vdf = self.search(self[weight] != 0, usecols = columns)
            for i in range(2, int(max_value / gcd) + 1):
                vdf = vdf.append(self.search((self[weight] / gcd) >= i, usecols=columns))
        else:
            assert weight >= 2 and isinstance(weight, int), ValueError("The weight must be an integer greater or equal to 2.")
            vdf = self.copy()
            for i in range(2, weight + 1):
                vdf = vdf.append(self)
        return vdf

    # ---#
    def acf(
        self,
        column: str,
        ts: str,
        by: list = [],
        p: (int, list) = 12,
        unit: str = "rows",
        method: str = "pearson",
        acf_type: str = "bar",
        confidence: bool = True,
        alpha: float = 0.95,
        round_nb: int = 3,
        show: bool = True,
        ax=None,
        **style_kwds,
    ):
        """
    ---------------------------------------------------------------------------
    Computes the correlations of the input vColumn and its lags. 

    Parameters
    ----------
    ts: str
        TS (Time Series) vColumn to use to order the data. It can be of type date
        or a numerical vColumn.
    column: str
        Input vColumn to use to compute the Auto Correlation Plot.
    by: list, optional
        vColumns used in the partition.
    p: int/list, optional
        Int equals to the maximum number of lag to consider during the computation
        or List of the different lags to include during the computation.
        p must be positive or a list of positive integers.
    unit: str, optional
        Unit to use to compute the lags.
            rows: Natural lags
            else : Any time unit, for example you can write 'hour' to compute the hours
                lags or 'day' to compute the days lags.
    method: str, optional
        Method to use to compute the correlation.
            pearson   : Pearson's correlation coefficient (linear).
            spearman  : Spearman's correlation coefficient (monotonic - rank based).
            kendall   : Kendall's correlation coefficient (similar trends). The method
                        will compute the Tau-B coefficient.
                       \u26A0 Warning : This method uses a CROSS JOIN during computation 
                                        and is therefore computationally expensive at 
                                        O(n * n), where n is the total count of the 
                                        vDataFrame.
            cramer    : Cramer's V (correlation between categories).
            biserial  : Biserial Point (correlation between binaries and a numericals).
    acf_type: str, optional
        ACF Type.
            bar     : Classical Autocorrelation Plot using bars.
            heatmap : Draws the ACF heatmap.
            line    : Draws the ACF using a Line Plot.
    confidence: bool, optional
        If set to True, the confidence band width is drawn.
    alpha: float, optional
        Significance Level. Probability to accept H0. Only used to compute the confidence
        band width.
    round_nb: int, optional
        Round the coefficient using the input number of digits. It is used only if 
        acf_type is 'heatmap'.
    show: bool, optional
        If set to True, the Auto Correlation Plot will be drawn using Matplotlib.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.asfreq : Interpolates and computes a regular time interval vDataFrame.
    vDataFrame.corr   : Computes the Correlation Matrix of a vDataFrame.
    vDataFrame.cov    : Computes the covariance matrix of the vDataFrame.
    vDataFrame.pacf   : Computes the partial autocorrelations of the input vColumn.
        """
        if isinstance(method, str):
            method = method.lower()
        if isinstance(by, str):
            by = [by]
        check_types(
            [
                ("by", by, [list],),
                ("ts", ts, [str],),
                ("column", column, [str],),
                ("p", p, [int, float, list],),
                ("unit", unit, [str],),
                ("acf_type", acf_type, ["line", "heatmap", "bar"],),
                (
                    "method",
                    method,
                    ["pearson", "kendall", "spearman", "biserial", "cramer"],
                ),
                ("round_nb", round_nb, [int, float],),
                ("confidence", confidence, [bool],),
                ("alpha", alpha, [int, float],),
                ("show", show, [bool],),
            ]
        )
        columns_check([column, ts] + by, self)
        by = vdf_columns_names(by, self)
        column = vdf_columns_names([column], self)[0]
        ts = vdf_columns_names([ts], self)[0]
        if unit == "rows":
            table = self.__genSQL__()
        else:
            table = self.asfreq(
                ts=ts, rule="1 {}".format(unit), method={column: "linear"}, by=by
            ).__genSQL__()
        if isinstance(p, (int, float)):
            p = range(1, p + 1)
        by = "PARTITION BY {} ".format(", ".join(by)) if (by) else ""
        columns = [
            "LAG({}, {}) OVER ({}ORDER BY {}) AS lag_{}_{}".format(
                column, i, by, ts, i, gen_name([column])
            )
            for i in p
        ]
        relation = "(SELECT {} FROM {}) acf".format(
            ", ".join([column] + columns), table
        )
        if len(p) == 1:
            return self.__vdf_from_relation__(relation, "acf", "").corr(
                [], method=method
            )
        elif acf_type == "heatmap":
            return self.__vdf_from_relation__(relation, "acf", "").corr(
                [],
                method=method,
                round_nb=round_nb,
                focus=column,
                show=show,
                **style_kwds,
            )
        else:
            result = self.__vdf_from_relation__(relation, "acf", "").corr(
                [], method=method, focus=column, show=False
            )
            columns = [elem for elem in result.values["index"]]
            acf = [elem for elem in result.values[column]]
            acf_band = []
            if confidence:
                from scipy.special import erfinv

                for k in range(1, len(acf) + 1):
                    acf_band += [
                        math.sqrt(2)
                        * erfinv(alpha)
                        / math.sqrt(self[column].count() - k + 1)
                        * math.sqrt((1 + 2 * sum([acf[i] ** 2 for i in range(1, k)])))
                    ]
            if columns[0] == column:
                columns[0] = 0
            for i in range(1, len(columns)):
                columns[i] = int(columns[i].split("_")[1])
            data = [(columns[i], acf[i]) for i in range(len(columns))]
            data.sort(key=lambda tup: tup[0])
            del result.values[column]
            result.values["index"] = [elem[0] for elem in data]
            result.values["value"] = [elem[1] for elem in data]
            if acf_band:
                result.values["confidence"] = acf_band
            if show:
                from verticapy.plot import acf_plot

                acf_plot(
                    result.values["index"],
                    result.values["value"],
                    title="Autocorrelation",
                    confidence=acf_band,
                    type_bar=True if acf_type == "bar" else False,
                    ax=ax,
                    **style_kwds,
                )
            return result

    # ---#
    def aggregate(self, func: list, columns: list = [], ncols_block: int = 20,):
        """
    ---------------------------------------------------------------------------
    Aggregates the vDataFrame using the input functions.

    Parameters
    ----------
    func: list
        List of the different aggregations.
            aad            : average absolute deviation
            approx_unique  : approximative cardinality
            count          : number of non-missing elements
            cvar           : conditional value at risk
            dtype          : virtual column type
            iqr            : interquartile range
            kurtosis       : kurtosis
            jb             : Jarque-Bera index 
            mad            : median absolute deviation
            max            : maximum
            mean           : average
            median         : median
            min            : minimum
            mode           : most occurent element
            percent        : percent of non-missing elements
            q%             : q quantile (ex: 50% for the median)
            prod           : product
            range          : difference between the max and the min
            sem            : standard error of the mean
            skewness       : skewness
            sum            : sum
            std            : standard deviation
            topk           : kth most occurent element (ex: top1 for the mode)
            topk_percent   : kth most occurent element density
            unique         : cardinality (count distinct)
            var            : variance
                Other aggregations could work if it is part of 
                the DB version you are using.
    columns: list, optional
        List of the vColumns names. If empty, all vColumns 
        or only numerical vColumns will be used depending on the
        aggregations.
    ncols_block: int, optional
        Number of columns used per query. For query performance, it is better to
        pick up a balanced number. A small number will lead to the generation of
        multiple queries. A higher number will lead to the generation of one big
        SQL query.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.analytic : Adds a new vColumn to the vDataFrame by using an advanced 
        analytical function on a specific vColumn.
        """

        def agg_format(item):
            if isinstance(item, (float, int)):
                return "'{}'".format(item)
            elif isinstance(item, type(None)):
                return "NULL"
            else:
                return str(item)

        if isinstance(columns, str):
            columns = [columns]
        if isinstance(func, str):
            func = [func]
        check_types([("func", func, [list],), ("columns", columns, [list],), ("ncols_block", ncols_block, [int],),])
        columns_check(columns, self)
        if not (columns):
            columns = self.get_columns()
            for fun in func:
                cat_agg = [
                    "count",
                    "unique",
                    "approx_unique",
                    "approximate_count_distinct",
                    "dtype",
                    "percent",
                ]
                if ("top" not in fun) and (fun not in cat_agg):
                    columns = self.numcol()
                    break
        else:
            columns = vdf_columns_names(columns, self)
        if ncols_block < len(columns):
            i = ncols_block
            result = self.aggregate(func=func, columns=columns[0:ncols_block], ncols_block=ncols_block).transpose()
            if verticapy.options["tqdm"]:
                from tqdm.auto import tqdm

                loop = tqdm(range(ncols_block, len(columns), ncols_block))
            else:
                loop = range(ncols_block, len(columns), ncols_block)
            for i in loop:
                columns_tmp = columns[i:i+ncols_block]
                if columns_tmp:
                    result_tmp = self.aggregate(func=func, columns=columns_tmp, ncols_block=ncols_block).transpose()
                    for elem in result_tmp.values:
                        if elem != "index":
                            result.values[elem] = result_tmp[elem]
            return result.transpose()
        agg = [[] for i in range(len(columns))]
        nb_precomputed = 0
        for idx, column in enumerate(columns):
            cast = "::int" if (self[column].isbool()) else ""
            for fun in func:
                pre_comp = self.__get_catalog_value__(column, fun)
                if pre_comp != "VERTICAPY_NOT_PRECOMPUTED":
                    nb_precomputed += 1
                    if pre_comp == None or pre_comp != pre_comp:
                        expr = "NULL"
                    elif isinstance(pre_comp, (int, float)):
                        expr = pre_comp
                    else:
                        expr = "'{}'".format(str(pre_comp).replace("'", "''"))
                elif ("_percent" in fun.lower()) and (fun.lower()[0:3] == "top"):
                    n = fun.lower().replace("top", "").replace("_percent", "")
                    if n == "":
                        n = 1
                    try:
                        n = int(n)
                        assert n >= 1
                    except:
                        raise FunctionError(
                            "The aggregation '{}' doesn't exist. If you want to compute the frequence of the nth most occurent element please write 'topn_percent' with n > 0. Example: top2_percent for the frequency of the second most occurent element.".format(
                                fun
                            )
                        )
                    try:
                        expr = str(
                            self[column]
                            .topk(k=n, dropna=False)
                            .values["percent"][n - 1]
                        )
                    except:
                        expr = "0.0"
                elif (len(fun.lower()) > 2) and (fun.lower()[0:3] == "top"):
                    n = fun.lower()[3:] if (len(fun.lower()) > 3) else 1
                    try:
                        n = int(n)
                        assert n >= 1
                    except:
                        raise FunctionError(
                            "The aggregation '{}' doesn't exist. If you want to compute the nth most occurent element please write 'topn' with n > 0. Example: top2 for the second most occurent element.".format(
                                fun
                            )
                        )
                    expr = format_magic(self[column].mode(n=n))
                elif fun.lower() in ("mode"):
                    expr = format_magic(self[column].mode(n=1))
                elif fun.lower() in ("kurtosis", "kurt"):
                    count, avg, std = (
                        self.aggregate(func=["count", "avg", "stddev"], columns=columns)
                        .transpose()
                        .values[column]
                    )
                    if (
                        count == 0
                        or (std != std)
                        or (avg != avg)
                        or (std == None)
                        or (avg == None)
                    ):
                        expr = "NULL"
                    elif (count == 1) or (std == 0):
                        expr = "-3"
                    else:
                        expr = "AVG(POWER(({}{} - {}) / {}, 4))".format(
                            column, cast, avg, std
                        )
                        if count > 3:
                            expr += "* {} - 3 * {}".format(
                                count
                                * count
                                * (count + 1)
                                / (count - 1)
                                / (count - 2)
                                / (count - 3),
                                (count - 1) * (count - 1) / (count - 2) / (count - 3),
                            )
                        else:
                            expr += "* - 3"
                            expr += (
                                "* {}".format(count * count / (count - 1) / (count - 2))
                                if (count == 3)
                                else ""
                            )
                elif fun.lower() in ("skewness", "skew"):
                    count, avg, std = (
                        self.aggregate(func=["count", "avg", "stddev"], columns=columns)
                        .transpose()
                        .values[column]
                    )
                    if (
                        count == 0
                        or (std != std)
                        or (avg != avg)
                        or (std == None)
                        or (avg == None)
                    ):
                        expr = "NULL"
                    elif (count == 1) or (std == 0):
                        expr = "0"
                    else:
                        expr = "AVG(POWER(({}{} - {}) / {}, 3))".format(
                            column, cast, avg, std
                        )
                        if count >= 3:
                            expr += "* {}".format(
                                count * count / (count - 1) / (count - 2)
                            )
                elif fun.lower() in ("jb"):
                    count, avg, std = (
                        self.aggregate(func=["count", "avg", "stddev"], columns=columns)
                        .transpose()
                        .values[column]
                    )
                    if (count < 4) or (std == 0):
                        expr = "NULL"
                    else:
                        expr = "{} / 6 * (POWER(AVG(POWER(({}{} - {}) / {}, 3)) * {}, 2) + POWER(AVG(POWER(({}{} - {}) / {}, 4)) - 3 * {}, 2) / 4)".format(
                            count,
                            column,
                            cast,
                            avg,
                            std,
                            count * count / (count - 1) / (count - 2),
                            column,
                            cast,
                            avg,
                            std,
                            count * count / (count - 1) / (count - 2),
                        )
                elif fun.lower() == "dtype":
                    expr = "'{}'".format(self[column].ctype())
                elif fun.lower() == "range":
                    expr = "MAX({}{}) - MIN({}{})".format(column, cast, column, cast)
                elif fun.lower() == "unique":
                    expr = "COUNT(DISTINCT {})".format(column)
                elif fun.lower() in ("approx_unique", "approximate_count_distinct"):
                    expr = "APPROXIMATE_COUNT_DISTINCT({})".format(column)
                elif fun.lower() == "count":
                    expr = "COUNT({})".format(column)
                elif fun.lower() == "median":
                    expr = "APPROXIMATE_MEDIAN({}{})".format(column, cast)
                elif fun.lower() in ("std", "stddev", "stdev"):
                    expr = "STDDEV({}{})".format(column, cast)
                elif fun.lower() in ("var", "variance"):
                    expr = "VARIANCE({}{})".format(column, cast)
                elif fun.lower() in ("mean", "avg"):
                    expr = "AVG({}{})".format(column, cast)
                elif fun.lower() == "iqr":
                    expr = "APPROXIMATE_PERCENTILE({}{} USING PARAMETERS percentile = 0.75) - APPROXIMATE_PERCENTILE({}{} USING PARAMETERS percentile = 0.25)".format(
                        column, cast, column, cast
                    )
                elif "%" == fun[-1]:
                    try:
                        expr = "APPROXIMATE_PERCENTILE({}{} USING PARAMETERS percentile = {})".format(
                            column, cast, float(fun[0:-1]) / 100
                        )
                    except:
                        raise FunctionError(
                            "The aggregation '{}' doesn't exist. If you want to compute the percentile x of the element please write 'x%' with x > 0. Example: 50% for the median.".format(
                                fun
                            )
                        )
                elif fun.lower() == "cvar":
                    q95 = self[column].quantile(0.95)
                    expr = "AVG(CASE WHEN {}{} > {} THEN {}{} ELSE NULL END)".format(
                        column, cast, q95, column, cast
                    )
                elif fun.lower() == "sem":
                    expr = "STDDEV({}{}) / SQRT(COUNT({}))".format(column, cast, column)
                elif fun.lower() == "aad":
                    mean = self[column].avg()
                    expr = "SUM(ABS({}{} - {})) / COUNT({})".format(
                        column, cast, mean, column
                    )
                elif fun.lower() == "mad":
                    median = self[column].median()
                    expr = "APPROXIMATE_MEDIAN(ABS({}{} - {}))".format(
                        column, cast, median
                    )
                elif fun.lower() in ("prod", "product"):
                    expr = "DECODE(ABS(MOD(SUM(CASE WHEN {}{} < 0 THEN 1 ELSE 0 END), 2)), 0, 1, -1) * POWER(10, SUM(LOG(ABS({}{}))))".format(
                        column, cast, column, cast
                    )
                elif fun.lower() in ("percent", "count_percent"):
                    expr = "ROUND(COUNT({}) / {} * 100, 3)::float".format(
                        column, self.shape()[0]
                    )
                elif "{}" not in fun:
                    expr = "{}({}{})".format(fun.upper(), column, cast)
                else:
                    expr = fun.replace("{}", column)
                agg[idx] += [expr]
        for idx, elem in enumerate(func):
            if "AS " in str(elem).upper():
                try:
                    func[idx] = (
                        str(elem)
                        .lower()
                        .split("as ")[1]
                        .replace("'", "")
                        .replace('"', "")
                    )
                except:
                    pass
        values = {"index": func}
        try:
            if nb_precomputed == len(func) * len(columns):
                self._VERTICAPY_VARIABLES_["cursor"].execute(
                    "SELECT {}".format(
                        ", ".join([str(item) for sublist in agg for item in sublist])
                    )
                )
            else:
                self.__executeSQL__(
                    "SELECT {} FROM {} LIMIT 1".format(
                        ", ".join([str(item) for sublist in agg for item in sublist]),
                        self.__genSQL__(),
                    ),
                    title="Computes the different aggregations.",
                )
            result = [item for item in self._VERTICAPY_VARIABLES_["cursor"].fetchone()]
            try:
                result = [float(item) for item in result]
            except:
                pass
            values = {"index": func}
            i = 0
            for column in columns:
                values[column] = result[i : i + len(func)]
                i += len(func)
        except:
            try:
                query = [
                    "SELECT {} FROM vdf_table LIMIT 1".format(
                        ", ".join([agg_format(item) for item in elem])
                    )
                    for elem in agg
                ]
                query = (
                    " UNION ALL ".join(["({})".format(elem) for elem in query])
                    if (len(query) != 1)
                    else query[0]
                )
                query = "WITH vdf_table AS (SELECT * FROM {}) {}".format(
                    self.__genSQL__(), query
                )
                if nb_precomputed == len(func) * len(columns):
                    self._VERTICAPY_VARIABLES_["cursor"].execute(query)
                else:
                    self.__executeSQL__(
                        query,
                        title="Computes the different aggregations using UNION ALL.",
                    )
                result = self._VERTICAPY_VARIABLES_["cursor"].fetchall()
                for idx, elem in enumerate(result):
                    values[columns[idx]] = [item for item in elem]
            except:
                try:
                    for i, elem in enumerate(agg):
                        pre_comp_val = []
                        for fun in func:
                            pre_comp = self.__get_catalog_value__(columns[i], fun)
                            if pre_comp == "VERTICAPY_NOT_PRECOMPUTED":
                                query = "SELECT {} FROM {}".format(
                                    ", ".join([agg_format(item) for item in elem]),
                                    self.__genSQL__(),
                                )
                                self.__executeSQL__(
                                    query,
                                    title="Computes the different aggregations one vColumn at a time.",
                                )
                                pre_comp_val = []
                                break
                            pre_comp_val += [pre_comp]
                        if pre_comp_val:
                            values[columns[i]] = pre_comp_val
                        else:
                            values[columns[i]] = [
                                elem
                                for elem in self._VERTICAPY_VARIABLES_[
                                    "cursor"
                                ].fetchone()
                            ]
                except:
                    for i, elem in enumerate(agg):
                        values[columns[i]] = []
                        for j, agg_fun in enumerate(elem):
                            pre_comp = self.__get_catalog_value__(columns[i], func[j])
                            if pre_comp == "VERTICAPY_NOT_PRECOMPUTED":
                                query = "SELECT {} FROM {}".format(
                                    agg_fun, self.__genSQL__()
                                )
                                self.__executeSQL__(
                                    query,
                                    title="Computes the different aggregations one vColumn & one agg at a time.",
                                )
                                result = self._VERTICAPY_VARIABLES_[
                                    "cursor"
                                ].fetchone()[0]
                            else:
                                result = pre_comp
                            values[columns[i]] += [result]
        for elem in values:
            for idx in range(len(values[elem])):
                if isinstance(values[elem][idx], decimal.Decimal):
                    values[elem][idx] = float(values[elem][idx])
                elif isinstance(values[elem][idx], str) and "top" not in elem:
                    try:
                        values[elem][idx] = float(values[elem][idx])
                    except:
                        pass
        self.__update_catalog__(values)
        return tablesample(values=values).transpose()

    agg = aggregate
    # ---#
    def all(self, columns: list):
        """
    ---------------------------------------------------------------------------
    Aggregates the vDataFrame using 'bool_and'.

    Parameters
    ----------
    columns: list
        List of the vColumns names.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        return self.aggregate(func=["bool_and"], columns=columns)

    # ---#
    def analytic(
        self,
        func: str,
        columns: (str, list) = [],
        by: list = [],
        order_by: (dict, list) = [],
        name: str = "",
        offset: int = 1,
        x_smoothing: float = 0.5,
        add_count: bool = True,
    ):
        """
    ---------------------------------------------------------------------------
    Adds a new vColumn to the vDataFrame by using an advanced analytical function 
    on one or two specific vColumns.

    \u26A0 Warning : Some analytical functions can make the vDataFrame structure 
                     heavier. It is recommended to always check the current structure 
                     using the 'current_relation' method and to save it using the 
                     'to_db' method with the parameters 'inplace = True' and 
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
    columns: str, optional
        Input vColumns. It can be a list of one or two elements.
    by: list, optional
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
        check_types(
            [
                ("func", func, [str],),
                ("by", by, [list],),
                ("name", name, [str],),
                ("order_by", order_by, [list, dict],),
                ("columns", columns, [str, list],),
                ("add_count", add_count, [bool],),
                ("offset", offset, [int, float],),
                ("x_smoothing", x_smoothing, [int, float],),
            ]
        )
        columns_check([elem for elem in order_by] + by, self)
        if isinstance(columns, str):
            if columns:
                columns = [columns]
            else:
                columns = []
        columns_check(columns, self)
        columns = vdf_columns_names(columns, self)
        by_name = ["by"] + by if (by) else []
        by_order = ["order_by"] + [elem for elem in order_by] if (order_by) else []
        if not (name):
            name = gen_name([func] + columns + by_name + by_order)
        by = vdf_columns_names(by, self)
        func = func.lower()
        by = ", ".join(by)
        by = "PARTITION BY {}".format(by) if (by) else ""
        order_by = sort_str(order_by, self)
        func = str_function(func.lower(), method="vertica")
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
            if order_by and not(verticapy.options["print_info"]):
                print(
                    "\u26A0 '{}' analytic method doesn't need an order by clause, it was ignored".format(
                        func
                    )
                )
            elif not (columns):
                raise MissingColumn(
                    "The parameter 'column' must be a vDataFrame Column when using analytic method '{}'".format(
                        func
                    )
                )
            if func in ("skewness", "kurtosis", "aad", "mad", "jb"):
                mean_name = "{}_mean_{}".format(
                    columns[0].replace('"', ""), random.randint(0, 10000000)
                )
                median_name = "{}_median_{}".format(
                    columns[0].replace('"', ""), random.randint(0, 10000000)
                )
                std_name = "{}_std_{}".format(
                    columns[0].replace('"', ""), random.randint(0, 10000000)
                )
                count_name = "{}_count_{}".format(
                    columns[0].replace('"', ""), random.randint(0, 10000000)
                )
                all_cols = [elem for elem in self._VERTICAPY_VARIABLES_["columns"]]
                if func == "mad":
                    self.eval(
                        median_name, "MEDIAN({}) OVER ({})".format(columns[0], by)
                    )
                else:
                    self.eval(mean_name, "AVG({}) OVER ({})".format(columns[0], by))
                if func not in ("aad", "mad"):
                    self.eval(std_name, "STDDEV({}) OVER ({})".format(columns[0], by))
                    self.eval(count_name, "COUNT({}) OVER ({})".format(columns[0], by))
                if func == "kurtosis":
                    self.eval(
                        name,
                        "AVG(POWER(({} - {}) / NULLIFZERO({}), 4)) OVER ({}) * POWER({}, 2) * ({} + 1) / NULLIFZERO(({} - 1) * ({} - 2) * ({} - 3)) - 3 * POWER({} - 1, 2) / NULLIFZERO(({} - 2) * ({} - 3))".format(
                            columns[0],
                            mean_name,
                            std_name,
                            by,
                            count_name,
                            count_name,
                            count_name,
                            count_name,
                            count_name,
                            count_name,
                            count_name,
                            count_name,
                        ),
                    )
                elif func == "skewness":
                    self.eval(
                        name,
                        "AVG(POWER(({} - {}) / NULLIFZERO({}), 3)) OVER ({}) * POWER({}, 2) / NULLIFZERO(({} - 1) * ({} - 2))".format(
                            columns[0],
                            mean_name,
                            std_name,
                            by,
                            count_name,
                            count_name,
                            count_name,
                        ),
                    )
                elif func == "jb":
                    self.eval(
                        name,
                        "{} / 6 * (POWER(AVG(POWER(({} - {}) / NULLIFZERO({}), 3)) OVER ({}) * POWER({}, 2) / NULLIFZERO(({} - 1) * ({} - 2)), 2) + POWER(AVG(POWER(({} - {}) / NULLIFZERO({}), 4)) OVER ({}) * POWER({}, 2) * ({} + 1) / NULLIFZERO(({} - 1) * ({} - 2) * ({} - 3)) - 3 * POWER({} - 1, 2) / NULLIFZERO(({} - 2) * ({} - 3)), 2) / 4)".format(
                            count_name,
                            columns[0],
                            mean_name,
                            std_name,
                            by,
                            count_name,
                            count_name,
                            count_name,
                            columns[0],
                            mean_name,
                            std_name,
                            by,
                            count_name,
                            count_name,
                            count_name,
                            count_name,
                            count_name,
                            count_name,
                            count_name,
                            count_name,
                        ),
                    )
                elif func == "aad":
                    self.eval(
                        name,
                        "AVG(ABS({} - {})) OVER ({})".format(columns[0], mean_name, by),
                    )
                elif func == "mad":
                    self.eval(
                        name,
                        "AVG(ABS({} - {})) OVER ({})".format(
                            columns[0], median_name, by
                        ),
                    )
            elif func == "top":
                self.eval(
                    name,
                    "ROW_NUMBER() OVER ({})".format(
                        "PARTITION BY {}".format(columns[0])
                        if not (by)
                        else "{}, {}".format(by, columns[0])
                    ),
                )
                if add_count:
                    self.eval(
                        "{}_count".format(name.replace('"', "")),
                        "NTH_VALUE({}, 1) OVER ({} ORDER BY {} DESC)".format(
                            name, by, name
                        ),
                    )
                self[name].apply(
                    "NTH_VALUE({}, 1) OVER ({} ORDER BY {} DESC)".format(
                        columns[0], by, "{}"
                    )
                )
            elif func == "unique":
                self.eval(
                    name,
                    "DENSE_RANK() OVER ({} ORDER BY {} ASC) + DENSE_RANK() OVER ({} ORDER BY {} DESC) - 1".format(
                        by, columns[0], by, columns[0]
                    ),
                )
            elif "%" == func[-1]:
                try:
                    x = float(func[0:-1]) / 100
                except:
                    raise FunctionError(
                        "The aggregate function '{}' doesn't exist. If you want to compute the percentile x of the element please write 'x%' with x > 0. Example: 50% for the median.".format(
                            func
                        )
                    )
                self.eval(
                    name,
                    "PERCENTILE_CONT({}) WITHIN GROUP(ORDER BY {}) OVER ({})".format(
                        x, columns[0], by
                    ),
                )
            elif func == "range":
                self.eval(
                    name,
                    "MAX({}) OVER ({}) - MIN({}) OVER ({})".format(
                        columns[0], by, columns[0], by
                    ),
                )
            elif func == "iqr":
                self.eval(
                    name,
                    "PERCENTILE_CONT(0.75) WITHIN GROUP(ORDER BY {}) OVER ({}) - PERCENTILE_CONT(0.25) WITHIN GROUP(ORDER BY {}) OVER ({})".format(
                        columns[0], by, columns[0], by
                    ),
                )
            elif func == "sem":
                self.eval(
                    name,
                    "STDDEV({}) OVER ({}) / SQRT(COUNT({}) OVER ({}))".format(
                        columns[0], by, columns[0], by
                    ),
                )
            elif func == "prod":
                self.eval(
                    name,
                    "DECODE(ABS(MOD(SUM(CASE WHEN {} < 0 THEN 1 ELSE 0 END) OVER ({}), 2)), 0, 1, -1) * POWER(10, SUM(LOG(ABS({}))) OVER ({}))".format(
                        columns[0], by, columns[0], by
                    ),
                )
            else:
                self.eval(name, "{}({}) OVER ({})".format(func.upper(), columns[0], by))
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
                    "The parameter 'columns' must be a vDataFrame column when using analytic method '{}'".format(
                        func
                    )
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
                    "The parameter 'columns' must be empty when using analytic method '{}'".format(
                        func
                    )
                )
            if (by) and (order_by):
                order_by = " {}".format(order_by)
            if func in ("lead", "lag"):
                info_param = ", {}".format(offset)
            elif func in ("last_value", "first_value"):
                info_param = " IGNORE NULLS"
            elif func == "exponential_moving_average":
                info_param = ", {}".format(x_smoothing)
            else:
                info_param = ""
            if func == "pct_change":
                self.eval(
                    name,
                    "{} / (LAG({}) OVER ({}{}))".format(
                        columns[0], columns[0], by, order_by
                    ),
                )
            else:
                self.eval(
                    name,
                    "{}({}{}) OVER ({}{})".format(
                        func.upper(),
                        columns[0] if (columns) else "",
                        info_param,
                        by,
                        order_by,
                    ),
                )
        elif func in ("corr", "cov", "beta"):
            if order_by:
                print(
                    "\u26A0 '{}' analytic method doesn't need an order by clause, it was ignored".format(
                        func
                    )
                )
            assert len(columns) == 2, MissingColumn(f"The parameter 'columns' includes 2 vColumns when using analytic method '{func}'")
            if columns[0] == columns[1]:
                if func == "cov":
                    expr = "VARIANCE({}) OVER ({})".format(columns[0], by)
                else:
                    expr = 1
            else:
                if func == "corr":
                    den = " / (STDDEV({}) OVER ({}) * STDDEV({}) OVER ({}))".format(
                        columns[0], by, columns[1], by
                    )
                elif func == "beta":
                    den = " / (VARIANCE({}) OVER ({}))".format(columns[1], by)
                else:
                    den = ""
                expr = "(AVG({} * {}) OVER ({}) - AVG({}) OVER ({}) * AVG({}) OVER ({})){}".format(
                    columns[0], columns[1], by, columns[0], by, columns[1], by, den
                )
            self.eval(
                name, expr,
            )
        else:
            try:
                self.eval(
                    name,
                    "{}({}) OVER ({}{})".format(
                        func.upper(), columns[0], info_param, by, order_by
                    ),
                )
            except:
                raise FunctionError(
                    "The aggregate function '{}' doesn't exist or is not managed by the 'analytic' method. If you want more flexibility use the 'eval' method".format(
                        func
                    )
                )
        if func in ("kurtosis", "skewness", "jb"):
            self._VERTICAPY_VARIABLES_["exclude_columns"] += [
                str_column(mean_name),
                str_column(std_name),
                str_column(count_name),
            ]
        elif func in ("aad"):
            self._VERTICAPY_VARIABLES_["exclude_columns"] += [str_column(mean_name)]
        elif func in ("mad"):
            self._VERTICAPY_VARIABLES_["exclude_columns"] += [str_column(median_name)]
        return self

    # ---#
    def animated(
        self,
        ts: str,
        columns: list = [],
        by: str = "",
        start_date: (str, datetime.datetime, datetime.date,) = "",
        end_date: (str, datetime.datetime, datetime.date,) = "",
        kind: str = "auto",
        limit_over: int = 6,
        limit: int = 1000000,
        limit_labels: int = 6,
        ts_steps: dict = {"window": 100, "step": 5},
        bubble_img: dict = {"bbox": [], "img": ""},
        fixed_xy_lim: bool = False,
        date_in_title: bool = False,
        date_f = None,
        date_style_dict: dict = {},
        interval: int = 300,
        repeat: bool = True,
        return_html: bool = True,
        ax=None,
        **style_kwds,
    ):
        """
    ---------------------------------------------------------------------------
    Draws the animated chart.

    Parameters
    ----------
    ts: str
        TS (Time Series) vColumn to use to order the data. The vColumn type must be
        date like (date, datetime, timestamp...) or numerical.
    columns: list, optional
        List of the vColumns names.
    by: str, optional
        Categorical vColumn used in the partition.
    start_date: str / date, optional
        Input Start Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is lesser than November 1993 the 3rd.
    end_date: str / date, optional
        Input End Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is greater than November 1993 the 3rd.
    kind: str, optional
        Animation Type.
            auto   : Pick up automatically the type.
            bar    : Animated Bar Race.
            bubble : Animated Bubble Plot.
            pie    : Animated Pie Chart.
            ts     : Animated Time Series.
    limit_over: int, optional
        Limited number of elements to consider for each category.
    limit: int, optional
        Maximum number of data points to use.
    limit_labels: int, optional
        [Only used when kind = 'bubble']
        Maximum number of text labels to draw.
    ts_steps: dict, optional
        [Only used when kind = 'ts']
        dictionary including 2 keys.
            step   : number of elements used to update the time series.
            window : size of the window used to draw the time series.
    bubble_img: dict, optional
        [Only used when kind = 'bubble']
        dictionary including 2 keys.
            img  : Path to the image to display as background.
            bbox : List of 4 elements to delimit the boundaries of the final Plot.
                   It must be similar the following list: [xmin, xmax, ymin, ymax]
    fixed_xy_lim: bool, optional
        If set to True, the xlim and ylim will be fixed.
    date_in_title: bool, optional
        If set to True, the ts vColumn will be displayed in the title section.
    date_f: function, optional
        Function used to display the ts vColumn.
    date_style_dict: dict, optional
        Style Dictionary used to display the ts vColumn when date_in_title = False.
    interval: int, optional
        Number of ms between each update.
    repeat: bool, optional
        If set to True, the animation will be repeated.
    return_html: bool, optional
        If set to True and if using a Jupyter notebook, the HTML of the animation will be 
        generated.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    animation
        Matplotlib animation object
        """
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(kind, str):
            kind = kind.lower()
        check_types(
            [
                ("columns", columns, [list],),
                ("ts", ts, [str],),
                ("by", by, [str],),
                ("kind", kind, ["auto", "bar", "bubble", "ts", "pie",],),
                ("start_date", start_date, [str, datetime.datetime, datetime.date, int, float,],),
                ("end_date", end_date, [str, datetime.datetime, datetime.date, int, float,],),
                ("limit_over", limit_over, [int,],),
                ("limit_labels", limit_labels, [int,],),
                ("limit", limit, [int,],),
                ("fixed_xy_lim", fixed_xy_lim, [bool,],),
                ("date_in_title", date_in_title, [bool,],),
                ("date_style_dict", date_style_dict, [dict,],),
                ("interval", interval, [int,],),
                ("repeat", repeat, [bool,],),
                ("return_html", return_html, [bool,],),
                ("ts_steps", ts_steps, [dict,],),
                ("img", bubble_img["img"], [str,],),
                ("bbox", bubble_img["bbox"], [list,],),
            ]
        )
        if kind == "auto":
            if len(columns) > 3 or len(columns) <= 1:
                kind = "ts"
            elif len(columns) == 2:
                kind = "bar"
            else:
                kind = "bubble"
        assert kind == "ts" or columns, ParameterError("Parameter 'columns' can not be empty when using kind = '{}'.".format(kind))
        assert (2 <= len(columns) <= 4 and self[columns[0]].isnum() and self[columns[1]].isnum()) or kind != "bubble", ParameterError("Parameter 'columns' must include at least 2 numerical vColumns and maximum 4 vColumns when using kind = '{}'.".format(kind))
        columns_check(columns + [ts], self)
        columns = vdf_columns_names(columns, self)
        ts = vdf_columns_names([ts], self)[0]
        if by:
            columns_check([by], self)
            by = vdf_columns_names([by], self)[0]
        if kind == "bubble":
            from verticapy.plot import animated_bubble_plot
            if len(columns) == 3 and not(self[columns[2]].isnum()):
                label_name = columns[2]
                columns = columns[0:2]
            elif len(columns) >= 4:
                if not(self[columns[3]].isnum()):
                    label_name = columns[3]
                    columns = columns[0:3]
                else:
                    label_name = columns[2]
                    columns = columns[0:2] + [columns[3]]
            else:
                label_name = ""
            if "img" not in bubble_img:
                bubble_img["img"] = ""
            if "bbox" not in bubble_img:
                bubble_img["bbox"] = []
            return animated_bubble_plot(self,
                                        order_by=ts,
                                        columns=columns,
                                        label_name=label_name,
                                        by=by,
                                        order_by_start=start_date,
                                        order_by_end=end_date,
                                        limit_over=limit_over,
                                        limit=limit,
                                        lim_labels=limit_labels,
                                        fixed_xy_lim=fixed_xy_lim,
                                        date_in_title=date_in_title,
                                        date_f=date_f,
                                        date_style_dict=date_style_dict,
                                        interval=interval,
                                        repeat=repeat,
                                        return_html=return_html,
                                        img=bubble_img["img"],
                                        bbox=bubble_img["bbox"],
                                        ax=ax,
                                        **style_kwds,)
        elif kind in ("bar", "pie",):
            from verticapy.plot import animated_bar
            return animated_bar(self,
                                order_by=ts,
                                columns=columns,
                                by=by,
                                order_by_start=start_date,
                                order_by_end=end_date,
                                limit_over=limit_over,
                                limit=limit,
                                fixed_xy_lim=fixed_xy_lim,
                                date_in_title=date_in_title,
                                date_f=date_f,
                                date_style_dict=date_style_dict,
                                interval=interval,
                                repeat=repeat,
                                return_html=return_html,
                                pie=(kind == "pie"),
                                ax=ax,
                                **style_kwds,)
        else:
            from verticapy.plot import animated_ts_plot
            if (by):
                assert len(columns) == 1, ParameterError("Parameter 'columns' can not be empty when using kind = 'ts' and when parameter 'by' is not empty.")
                vdf = self.pivot(index=ts,
                                 columns=by,
                                 values=columns[0],)
            else:
                vdf = self
            columns = vdf.numcol()[0:limit_over]
            if "step" not in ts_steps:
                ts_steps["step"] = 5
            if "window" not in ts_steps:
                ts_steps["window"] = 100
            return animated_ts_plot(vdf,
                                    order_by=ts,
                                    columns=columns,
                                    order_by_start=start_date,
                                    order_by_end=end_date,
                                    limit=limit,
                                    fixed_xy_lim=fixed_xy_lim,
                                    window_size=ts_steps["window"],
                                    step=ts_steps["step"],
                                    interval=interval,
                                    repeat=repeat,
                                    return_html=return_html,
                                    ax=ax,
                                    **style_kwds,)

    # ---#
    def any(self, columns: list):
        """
    ---------------------------------------------------------------------------
    Aggregates the vDataFrame using 'bool_or'.

    Parameters
    ----------
    columns: list
        List of the vColumns names.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        return self.aggregate(func=["bool_or"], columns=columns)

    # ---#
    def append(
        self, input_relation, expr1: list = [], expr2: list = [], union_all: bool = True
    ):
        """
    ---------------------------------------------------------------------------
    Merges the vDataFrame with another one or an input relation and returns 
    a new vDataFrame.

    Parameters
    ----------
    input_relation: str / vDataFrame
        Relation to use to do the merging.
    expr1: list, optional
        List of pure-SQL expressions from the current vDataFrame to use during merging.
        For example, 'CASE WHEN "column" > 3 THEN 2 ELSE NULL END' and 'POWER("column", 2)' 
        will work. If empty, all vDataFrame vColumns will be used. Aliases are 
        recommended to avoid auto-naming.
    expr2: list, optional
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
        check_types(
            [
                ("expr1", expr1, [list],),
                ("expr2", expr2, [list],),
                ("union_all", union_all, [bool],),
                ("input_relation", input_relation, [vDataFrame, str,])
            ]
        )
        first_relation = self.__genSQL__()
        if isinstance(input_relation, str):
            second_relation = input_relation
        elif isinstance(input_relation, vDataFrame):
            second_relation = input_relation.__genSQL__()
        columns = ", ".join(self.get_columns()) if not (expr1) else ", ".join(expr1)
        columns2 = columns if not (expr2) else ", ".join(expr2)
        union = "UNION" if not (union_all) else "UNION ALL"
        table = "(SELECT {} FROM {}) {} (SELECT {} FROM {})".format(
            columns, first_relation, union, columns2, second_relation
        )
        query = "SELECT * FROM ({}) append_table".format(table)
        self.__executeSQL__(query=query, title="Merges the two relations.")
        return self.__vdf_from_relation__(
            "({}) append_table".format(table),
            self._VERTICAPY_VARIABLES_["input_relation"],
            "[Append]: Union of two relations",
        )

    # ---#
    def apply(self, func: dict):
        """
    ---------------------------------------------------------------------------
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
        check_types([("func", func, [dict],)])
        columns_check([elem for elem in func], self)
        for column in func:
            self[vdf_columns_names([column], self)[0]].apply(func[column])
        return self

    # ---#
    def applymap(self, func: str, numeric_only: bool = True):
        """
    ---------------------------------------------------------------------------
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
        check_types(
            [("func", func, [str],), ("numeric_only", numeric_only, [bool],),]
        )
        function = {}
        columns = self.numcol() if numeric_only else self.get_columns()
        for column in columns:
            function[column] = (
                func if not (self[column].isbool()) else func.replace("{}", "{}::int")
            )
        return self.apply(function)

    # ---#
    def asfreq(
        self, ts: str, rule: (str, datetime.timedelta), method: dict = {}, by: list = []
    ):
        """
    ---------------------------------------------------------------------------
    Computes a regular time interval vDataFrame by interpolating the missing 
    values using different techniques.

    Parameters
    ----------
    ts: str
        TS (Time Series) vColumn to use to order the data. The vColumn type must be
        date like (date, datetime, timestamp...)
    rule: str / time
        Interval to use to slice the time. For example, '5 minutes' will create records
        separated by '5 minutes' time interval.
    method: dict, optional
        Dictionary of all different methods of interpolation. The dict must be 
        similar to the following:
        {"column1": "interpolation1" ..., "columnk": "interpolationk"}
        3 types of interpolations are possible:
            bfill  : Constant propagation of the next value (Back Propagation).
            ffill  : Constant propagation of the first value (First Propagation).
            linear : Linear Interpolation.
    by: list, optional
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
        check_types(
            [
                ("ts", ts, [str],),
                ("rule", rule, [str, datetime.timedelta],),
                ("method", method, [dict],),
                ("by", by, [list],),
            ]
        )
        columns_check(by + [elem for elem in method], self)
        ts, by = vdf_columns_names([ts], self)[0], vdf_columns_names(by, self)
        all_elements = []
        for column in method:
            assert method[column] in ("bfill", "backfill", "pad", "ffill", "linear"), ParameterError("Each element of the 'method' dictionary must be in bfill|backfill|pad|ffill|linear")
            if method[column] in ("bfill", "backfill"):
                func, interp = "TS_LAST_VALUE", "const"
            elif method[column] in ("pad", "ffill"):
                func, interp = "TS_FIRST_VALUE", "const"
            else:
                func, interp = "TS_FIRST_VALUE", "linear"
            all_elements += [
                "{}({}, '{}') AS {}".format(
                    func,
                    vdf_columns_names([column], self)[0],
                    interp,
                    vdf_columns_names([column], self)[0],
                )
            ]
        table = "SELECT {} FROM {}".format("{}", self.__genSQL__())
        tmp_query = ["slice_time AS {}".format(str_column(ts))]
        tmp_query += [str_column(column) for column in by]
        tmp_query += all_elements
        table = table.format(", ".join(tmp_query))
        partition = (
            "PARTITION BY {} ".format(", ".join([str_column(column) for column in by]))
            if (by)
            else ""
        )
        table += " TIMESERIES slice_time AS '{}' OVER ({}ORDER BY {}::timestamp)".format(
            rule, partition, str_column(ts)
        )
        return self.__vdf_from_relation__(
            "({}) asfreq".format(table), "asfreq", "[Asfreq]: The data was resampled"
        )

    # ---#
    def astype(self, dtype: dict):
        """
    ---------------------------------------------------------------------------
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
        check_types([("dtype", dtype, [dict],)])
        columns_check([elem for elem in dtype], self)
        for column in dtype:
            self[vdf_columns_names([column], self)[0]].astype(dtype=dtype[column])
        return self

    # ---#
    def at_time(self, ts: str, time: (str, datetime.timedelta)):
        """
    ---------------------------------------------------------------------------
    Filters the vDataFrame by only keeping the records at the input time.

    Parameters
    ----------
    ts: str
        TS (Time Series) vColumn to use to filter the data. The vColumn type must be
        date like (date, datetime, timestamp...)
    time: str / time
        Input Time. For example, time = '12:00' will filter the data when time('ts') 
        is equal to 12:00.

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame.between_time : Filters the data between two time ranges.
    vDataFrame.first        : Filters the data by only keeping the first records.
    vDataFrame.filter       : Filters the data using the input expression.
    vDataFrame.last         : Filters the data by only keeping the last records.
        """
        check_types(
            [("ts", ts, [str],), ("time", time, [str, datetime.timedelta,],),]
        )
        columns_check([ts], self)
        self.filter("{}::time = '{}'".format(str_column(ts), time),)
        return self

    # ---#
    def avg(self, columns: list = []):
        """
    ---------------------------------------------------------------------------
    Aggregates the vDataFrame using 'avg' (Average).

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        return self.aggregate(func=["avg"], columns=columns)

    mean = avg
    # ---#
    def bar(
        self,
        columns: list,
        method: str = "density",
        of: str = "",
        max_cardinality: tuple = (6, 6),
        h: tuple = (None, None),
        hist_type: str = "auto",
        ax=None,
        **style_kwds,
    ):
        """
    ---------------------------------------------------------------------------
    Draws the bar chart of the input vColumns based on an aggregation.

    Parameters
    ----------
    columns: list
        List of the vColumns names. The list must have one or two elements.
    method: str, optional
        The method to use to aggregate the data.
            count   : Number of elements.
            density : Percentage of the distribution.
            mean    : Average of the vColumn 'of'.
            min     : Minimum of the vColumn 'of'.
            max     : Maximum of the vColumn 'of'.
            sum     : Sum of the vColumn 'of'.
            q%      : q Quantile of the vColumn 'of' (ex: 50% to get the median).
        It can also be a cutomized aggregation (ex: AVG(column1) + 5).
    of: str, optional
         The vColumn to use to compute the aggregation.
    h: tuple, optional
        Interval width of the vColumns 1 and 2 bars. It is only valid if the 
        vColumns are numerical. Optimized h will be computed if the parameter 
        is empty or invalid.
    max_cardinality: tuple, optional
        Maximum number of distinct elements for vColumns 1 and 2 to be used as 
        categorical (No h will be picked or computed)
    hist_type: str, optional
        The Histogram Type.
            auto          : Regular Bar Chart based on 1 or 2 vColumns.
            pyramid       : Pyramid Density Bar Chart. Only works if one of
                            the two vColumns is binary and the 'method' is 
                            set to 'density'.
            stacked       : Stacked Bar Chart based on 2 vColumns.
            fully_stacked : Fully Stacked Bar Chart based on 2 vColumns.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object

     See Also
     --------
     vDataFrame.boxplot     : Draws the Box Plot of the input vColumns.
     vDataFrame.hist        : Draws the histogram of the input vColumns based on an aggregation.
     vDataFrame.pivot_table : Draws the pivot table of vColumns based on an aggregation.
        """
        if isinstance(method, str):
            method = method.lower()
        if isinstance(hist_type, str):
            hist_type = hist_type.lower()
        if isinstance(columns, str):
            columns = [columns]
        check_types(
            [
                ("columns", columns, [list],),
                ("method", method, [str],),
                ("of", of, [str],),
                ("max_cardinality", max_cardinality, [list],),
                ("h", h, [list],),
                (
                    "hist_type",
                    hist_type,
                    [
                        "auto",
                        "fully_stacked",
                        "stacked",
                        "fully",
                        "fully stacked",
                        "pyramid",
                        "density",
                    ],
                ),
            ]
        )
        columns_check(columns, self, [1, 2])
        columns = vdf_columns_names(columns, self)
        if of:
            columns_check([of], self)
            of = vdf_columns_names([of], self)[0]
        if len(columns) == 1:
            return self[columns[0]].bar(method, of, 6, 0, 0, ax=ax, **style_kwds,)
        else:
            stacked, fully_stacked, density = False, False, False
            if hist_type.lower() in ("fully", "fully stacked", "fully_stacked"):
                fully_stacked = True
            elif hist_type.lower() == "stacked":
                stacked = True
            elif hist_type.lower() in ("pyramid", "density"):
                density = True
            from verticapy.plot import bar2D

            return bar2D(
                self,
                columns,
                method,
                of,
                max_cardinality,
                h,
                stacked,
                fully_stacked,
                density,
                ax=ax,
                **style_kwds,
            )

    # ---#
    def balance(
        self, column: str, method: str = "hybrid", x: float = 0.5, order_by: list = [],
    ):
        """
    ---------------------------------------------------------------------------
    Balances the dataset using the input method.

    \u26A0 Warning : If the data is not sorted, the generated SQL code may
                     differ between attempts.

    Parameters
    ----------
    column: str
        Column used to compute the different categories.
    method: str, optional
        The method with which to sample the data
            hybrid : hybrid sampling
            over   : oversampling
            under  : undersampling
    x: float, optional
        The desired ratio between the majority class and minority classes.
        Only used when method is 'over' or 'under'.
    order_by: list, optional
        vColumns used to sort the data.

    Returns
    -------
    vDataFrame
        balanced vDataFrame
        """
        if isinstance(method, str): 
            method = method.lower()
        check_types([("method", method, ["hybrid", "over", "under",],),
                     ("x", x, [float,],),
                     ("column", column, [str],),
                     ("order_by", order_by, [list],)],)
        assert 0 < x < 1, ParameterError("Parameter 'x' must be between 0 and 1")
        columns_check([column] + order_by, self,)
        column = vdf_columns_names([column], self)[0]
        order_by = vdf_columns_names(order_by, self)
        topk = self[column].topk()
        last_count, last_elem, n = topk["count"][-1], topk["index"][-1], len(topk["index"])
        if method == "over":
            last_count = last_count * x
        elif method == "under":
            last_count = last_count / x
        vdf = self.search("{} = '{}'".format(column, last_elem))
        for i in range(n - 1):
            vdf = vdf.append(self.search("{} = '{}'".format(column, topk["index"][i])).sample(n = last_count))
        vdf.sort(order_by)
        return vdf

    # ---#
    def between_time(
        self,
        ts: str,
        start_time: (str, datetime.timedelta),
        end_time: (str, datetime.timedelta),
    ):
        """
    ---------------------------------------------------------------------------
    Filters the vDataFrame by only keeping the records between two input times.

    Parameters
    ----------
    ts: str
        TS (Time Series) vColumn to use to filter the data. The vColumn type must be
        date like (date, datetime, timestamp...)
    start_time: str / time
        Input Start Time. For example, time = '12:00' will filter the data when 
        time('ts') is lesser than 12:00.
    end_time: str / time
        Input End Time. For example, time = '14:00' will filter the data when 
        time('ts') is greater than 14:00.

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame.at_time : Filters the data at the input time.
    vDataFrame.first   : Filters the data by only keeping the first records.
    vDataFrame.filter  : Filters the data using the input expression.
    vDataFrame.last    : Filters the data by only keeping the last records.
        """
        check_types(
            [
                ("ts", ts, [str],),
                ("start_time", start_time, [str, datetime.timedelta,],),
                ("end_time", end_time, [str, datetime.timedelta,],),
            ]
        )
        columns_check([ts], self)
        self.filter(
            "{}::time BETWEEN '{}' AND '{}'".format(
                str_column(ts), start_time, end_time
            ),
        )
        return self

    # ---#
    def bool_to_int(self):
        """
    ---------------------------------------------------------------------------
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

    # ---#
    def boxplot(
        self, columns: list = [], ax=None, **style_kwds,
    ):
        """
    ---------------------------------------------------------------------------
    Draws the Box Plot of the input vColumns. 

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will 
        be used.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object

    See Also
    --------
    vDataFrame.bar         : Draws the bar chart of the input vColumns based on an aggregation.
    vDataFrame.boxplot     : Draws the vColumn box plot.
    vDataFrame.hist        : Draws the histogram of the input vColumns based on an aggregation.
    vDataFrame.pivot_table : Draws the pivot table of vColumns based on an aggregation.
        """
        if isinstance(columns, str):
            columns = [columns]
        check_types([("columns", columns, [list],)])
        columns_check(columns, self)
        columns = vdf_columns_names(columns, self) if (columns) else self.numcol()
        from verticapy.plot import boxplot2D

        return boxplot2D(self, columns, ax=ax, **style_kwds,)

    # ---#
    def bubble(
        self,
        columns: list,
        size_bubble_col: str = "",
        catcol: str = "",
        cmap_col: str = "",
        max_nb_points: int = 20000,
        bbox: list = [],
        img: str = "",
        ax=None,
        **style_kwds,
    ):
        """
    ---------------------------------------------------------------------------
    Draws the bubble plot of the input vColumns.

    Parameters
    ----------
    columns: list
        List of the vColumns names. The list must have two elements.
    size_bubble_col: str
        Numerical vColumn to use to represent the Bubble size.
    catcol: str, optional
        Categorical column used as color.
    cmap_col: str, optional
        Numerical column used with a color map as color.
    max_nb_points: int, optional
        Maximum number of points to display.
    bbox: list, optional
        List of 4 elements to delimit the boundaries of the final Plot. 
        It must be similar the following list: [xmin, xmax, ymin, ymax]
    img: str, optional
        Path to the image to display as background.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
       Matplotlib axes object

    See Also
    --------
    vDataFrame.scatter : Draws the scatter plot of the input vColumns.
        """
        if isinstance(columns, str):
            columns = [columns]
        check_types(
            [
                ("columns", columns, [list],),
                ("size_bubble_col", size_bubble_col, [str],),
                ("cmap_col", cmap_col, [str],),
                ("max_nb_points", max_nb_points, [int, float],),
                ("bbox", bbox, [list],),
                ("img", img, [str],),
            ]
        )
        columns_check(columns, self, [2])
        columns = vdf_columns_names(columns, self)
        if catcol:
            columns_check([catcol], self)
            catcol = vdf_columns_names([catcol], self)[0]
        if size_bubble_col:
            columns_check([size_bubble_col], self)
            size_bubble_col = vdf_columns_names([size_bubble_col], self)[0]
        if cmap_col:
            columns_check([cmap_col], self)
            cmap_col = vdf_columns_names([cmap_col], self)[0]
        from verticapy.plot import bubble

        return bubble(
            self,
            columns + [size_bubble_col] if size_bubble_col else columns,
            catcol,
            cmap_col,
            max_nb_points,
            bbox,
            img,
            ax=ax,
            **style_kwds,
        )

    # ---#
    def catcol(self, max_cardinality: int = 12):
        """
    ---------------------------------------------------------------------------
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
        check_types([("max_cardinality", max_cardinality, [int, float],)])
        columns = []
        for column in self.get_columns():
            if (self[column].category() == "int") and not (self[column].isbool()):
                self._VERTICAPY_VARIABLES_["cursor"].execute(
                    "SELECT (APPROXIMATE_COUNT_DISTINCT({}) < {}) FROM {}".format(
                        column, max_cardinality, self.__genSQL__()
                    )
                )
                is_cat = self._VERTICAPY_VARIABLES_["cursor"].fetchone()[0]
            elif self[column].category() == "float":
                is_cat = False
            else:
                is_cat = True
            if is_cat:
                columns += [column]
        return columns

    # ---#
    def cdt(self, 
            columns: list = [],
            max_cardinality: int = 20,
            nbins: int = 10,
            tcdt: bool = True):
        """
    ---------------------------------------------------------------------------
    Returns the complete disjunctive table of the vDataFrame.
    Numerical features are transformed to categorical using
    the 'discretize' method. Applying PCA on TCDT leads to MCA 
    (Multiple correspondence analysis).

    \u26A0 Warning : This method can become computationally expensive when
                     used with categorical variables with many categories.

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names.
    max_cardinality: int, optional
        For any categorical variable, keeps the most frequent categories and 
        merges the less frequent categories into a new unique category.
    nbins: int, optional
        Number of bins used for the discretization (must be > 1).
    tcdt: bool, optional
        If set to True, returns the transformed complete disjunctive table 
        (TCDT). 

    Returns
    -------
    vDataFrame
        the CDT relation.
        """
        if isinstance(columns, str):
            columns = [columns]
        check_types(
            [
                ("columns", columns, [list],),
                ("tcdt", tcdt, [bool],),
                ("nbins", nbins, [int],),
                ("max_cardinality", max_cardinality, [int],),
            ]
        )
        if columns:
            columns_check(columns, self,)
            columns = vdf_columns_names(columns, self)
        else:
            columns = self.get_columns()
        vdf = self.copy()
        columns_to_drop = []
        for elem in columns:
            if vdf[elem].isbool():
                vdf[elem].astype("int")
            elif vdf[elem].isnum():
                vdf[elem].discretize(bins=nbins)
                columns_to_drop += [elem]
            elif vdf[elem].isdate():
                vdf[elem].drop()
            else:
                vdf[elem].discretize(method="topk", k=max_cardinality)
                columns_to_drop += [elem]
        vdf.one_hot_encode(columns=columns, max_cardinality=max(max_cardinality, nbins) + 2, drop_first=False)
        vdf.drop(columns=columns_to_drop)
        if tcdt:
            columns = vdf.get_columns()
            for elem in columns:
                sum_cat = vdf[elem].sum()
                vdf[elem].apply("{} / {} - 1".format("{}", sum_cat))
        return vdf

    # ---#
    def copy(self):
        """
    ---------------------------------------------------------------------------
    Returns a copy of the vDataFrame.

    Returns
    -------
    vDataFrame
        The copy of the vDataFrame.
        """
        copy_vDataFrame = vDataFrame("", empty=True)
        copy_vDataFrame._VERTICAPY_VARIABLES_["dsn"] = self._VERTICAPY_VARIABLES_["dsn"]
        copy_vDataFrame._VERTICAPY_VARIABLES_[
            "input_relation"
        ] = self._VERTICAPY_VARIABLES_["input_relation"]
        copy_vDataFrame._VERTICAPY_VARIABLES_[
            "main_relation"
        ] = self._VERTICAPY_VARIABLES_["main_relation"]
        copy_vDataFrame._VERTICAPY_VARIABLES_["schema"] = self._VERTICAPY_VARIABLES_[
            "schema"
        ]
        copy_vDataFrame._VERTICAPY_VARIABLES_["cursor"] = self._VERTICAPY_VARIABLES_[
            "cursor"
        ]
        copy_vDataFrame._VERTICAPY_VARIABLES_["columns"] = [
            item for item in self._VERTICAPY_VARIABLES_["columns"]
        ]
        copy_vDataFrame._VERTICAPY_VARIABLES_["where"] = [
            item for item in self._VERTICAPY_VARIABLES_["where"]
        ]
        copy_vDataFrame._VERTICAPY_VARIABLES_["order_by"] = {}
        for item in self._VERTICAPY_VARIABLES_["order_by"]:
            copy_vDataFrame._VERTICAPY_VARIABLES_["order_by"][
                item
            ] = self._VERTICAPY_VARIABLES_["order_by"][item]
        copy_vDataFrame._VERTICAPY_VARIABLES_["exclude_columns"] = [
            item for item in self._VERTICAPY_VARIABLES_["exclude_columns"]
        ]
        copy_vDataFrame._VERTICAPY_VARIABLES_["history"] = [
            item for item in self._VERTICAPY_VARIABLES_["history"]
        ]
        copy_vDataFrame._VERTICAPY_VARIABLES_["saving"] = [
            item for item in self._VERTICAPY_VARIABLES_["saving"]
        ]
        copy_vDataFrame._VERTICAPY_VARIABLES_[
            "schema_writing"
        ] = self._VERTICAPY_VARIABLES_["schema_writing"]
        for column in self._VERTICAPY_VARIABLES_["columns"]:
            new_vColumn = vColumn(
                column,
                parent=copy_vDataFrame,
                transformations=[elem for elem in self[column].transformations],
                catalog={},
            )
            setattr(copy_vDataFrame, column, new_vColumn)
            setattr(copy_vDataFrame, column[1:-1], new_vColumn)
        return copy_vDataFrame

    # ---#
    def case_when(
        self, name: str, *argv,
    ):
        """
    ---------------------------------------------------------------------------
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

        check_types([("name", name, [str],)])
        import verticapy.stats as st

        return self.eval(name=name, expr=st.case_when(*argv))

    # ---#
    def contour(
        self,
        columns: list,
        func,
        nbins: int = 100,
        ax=None,
        **style_kwds,
    ):
        """
    ---------------------------------------------------------------------------
    Draws the contour plot of the input function two input vColumns.

    Parameters
    ----------
    columns: list
        List of the vColumns names. The list must have two elements.
    func: function / str, optional
        Function used to compute the contour score. It can also be a SQL
        expression.
    nbins: int, optional
        Number of bins used to discretize the two input numerical vColumns.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object

     See Also
     --------
     vDataFrame.boxplot     : Draws the Box Plot of the input vColumns.
     vDataFrame.hist        : Draws the histogram of the input vColumns based on an aggregation.
     vDataFrame.pivot_table : Draws the pivot table of vColumns based on an aggregation.
        """
        check_types(
            [
                ("columns", columns, [list],),
                ("nbins", nbins, [int],),
            ]
        )
        columns_check(columns, self, [2])
        columns = vdf_columns_names(columns, self)
        from verticapy.plot import contour_plot

        return contour_plot(
            self,
            columns,
            func,
            nbins,
            ax=ax,
            **style_kwds,
        )

    # ---#
    def corr(
        self,
        columns: list = [],
        method: str = "pearson",
        round_nb: int = 3,
        focus: str = "",
        show: bool = True,
        ax=None,
        **style_kwds,
    ):
        """
    ---------------------------------------------------------------------------
    Computes the Correlation Matrix of the vDataFrame. 

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.
    method: str, optional
        Method to use to compute the correlation.
            pearson   : Pearson's correlation coefficient (linear).
            spearman  : Spearman's correlation coefficient (monotonic - rank based).
            kendall   : Kendall's correlation coefficient (similar trends). The method
                        will compute the Tau-B coefficient.
                        \u26A0 Warning : This method uses a CROSS JOIN during computation 
                                         and is therefore computationally expensive at 
                                         O(n * n), where n is the total count of the 
                                         vDataFrame.
            cramer    : Cramer's V (correlation between categories).
            biserial  : Biserial Point (correlation between binaries and a numericals).
    round_nb: int, optional
        Rounds the coefficient using the input number of digits.
    focus: str, optional
        Focus the computation on only one vColumn.
    show: bool, optional
        If set to True, the Correlation Matrix will be drawn using Matplotlib.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.acf  : Computes the correlations between a vColumn and its lags.
    vDataFrame.cov  : Computes the covariance matrix of the vDataFrame.
    vDataFrame.pacf : Computes the partial autocorrelations of the input vColumn.
    vDataFrame.regr : Computes the regression matrix of the vDataFrame. 
        """
        if isinstance(method, str):
            method = method.lower()
        if isinstance(columns, str):
            columns = [columns]
        check_types(
            [
                ("columns", columns, [list],),
                (
                    "method",
                    method,
                    ["pearson", "kendall", "spearman", "biserial", "cramer"],
                ),
                ("round_nb", round_nb, [int, float],),
                ("focus", focus, [str],),
                ("show", show, [bool],),
            ]
        )
        columns_check(columns, self)
        columns = vdf_columns_names(columns, self)
        if focus == "":
            return self.__aggregate_matrix__(
                method=method,
                columns=columns,
                round_nb=round_nb,
                show=show,
                ax=ax,
                **style_kwds,
            )
        else:
            columns_check([focus], self)
            focus = vdf_columns_names([focus], self)[0]
            return self.__aggregate_vector__(
                focus,
                method=method,
                columns=columns,
                round_nb=round_nb,
                show=show,
                ax=ax,
                **style_kwds,
            )

    # ---#
    def corr_pvalue(
        self, column1: str, column2: str, method: str = "pearson",
    ):
        """
    ---------------------------------------------------------------------------
    Computes the Correlation Coefficient of the two input vColumns and its pvalue. 

    Parameters
    ----------
    column1: str
        Input vColumn.
    column2: str
        Input vColumn.
    method: str, optional
        Method to use to compute the correlation.
            pearson   : Pearson's correlation coefficient (linear).
            spearman  : Spearman's correlation coefficient (monotonic - rank based).
            kendall   : Kendall's correlation coefficient (similar trends). 
                        Use kendallA to compute Tau-A, kendallB or kendall to compute 
                        Tau-B and kendallC to compute Tau-C.
                        \u26A0 Warning : This method uses a CROSS JOIN during computation 
                                         and is therefore computationally expensive at 
                                         O(n * n), where n is the total count of the 
                                         vDataFrame.
            cramer    : Cramer's V (correlation between categories).
            biserial  : Biserial Point (correlation between binaries and a numericals).

    Returns
    -------
    tuple
        (Correlation Coefficient, pvalue)

    See Also
    --------
    vDataFrame.corr : Computes the Correlation Matrix of the vDataFrame.
        """
        if isinstance(method, str):
            method = method.lower()
        check_types(
            [
                ("column1", column1, [str],),
                ("column2", column2, [str],),
                (
                    "method",
                    method,
                    [
                        "pearson",
                        "kendall",
                        "kendalla",
                        "kendallb",
                        "kendallc",
                        "spearman",
                        "biserial",
                        "cramer",
                    ],
                ),
            ]
        )

        from scipy.stats import t, norm, chi2
        from numpy import log

        columns_check([column1, column2], self)
        column1, column2 = vdf_columns_names([column1, column2], self)
        if method[0:7] == "kendall":
            if method == "kendall":
                kendall_type = "b"
            else:
                kendall_type = method[-1]
            method = "kendall"
        else:
            kendall_type = None
        if (method == "kendall" and kendall_type == "b") or (method != "kendall"):
            val = self.corr(columns=[column1, column2], method=method)
        sql = "SELECT COUNT(*) FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL;".format(
            self.__genSQL__(), column1, column2
        )
        self._VERTICAPY_VARIABLES_["cursor"].execute(sql)
        n = self._VERTICAPY_VARIABLES_["cursor"].fetchone()[0]
        if method in ("pearson", "biserial"):
            x = val * math.sqrt((n - 2) / (1 - val * val))
            pvalue = 2 * t.sf(abs(x), n - 2)
        elif method == "spearman":
            z = math.sqrt((n - 3) / 1.06) * 0.5 * log((1 + val) / (1 - val))
            pvalue = 2 * norm.sf(abs(z))
        elif method == "kendall":
            cast_i = "::int" if (self[column1].isbool()) else ""
            cast_j = "::int" if (self[column2].isbool()) else ""
            n_c = "(SUM(((x.{}{} < y.{}{} AND x.{}{} < y.{}{}) OR (x.{}{} > y.{}{} AND x.{}{} > y.{}{}))::int))/2".format(
                column1,
                cast_i,
                column1,
                cast_i,
                column2,
                cast_j,
                column2,
                cast_j,
                column1,
                cast_i,
                column1,
                cast_i,
                column2,
                cast_j,
                column2,
                cast_j,
            )
            n_d = "(SUM(((x.{}{} > y.{}{} AND x.{}{} < y.{}{}) OR (x.{}{} < y.{}{} AND x.{}{} > y.{}{}))::int))/2".format(
                column1,
                cast_i,
                column1,
                cast_i,
                column2,
                cast_j,
                column2,
                cast_j,
                column1,
                cast_i,
                column1,
                cast_i,
                column2,
                cast_j,
                column2,
                cast_j,
            )
            table = "(SELECT {} FROM {}) x CROSS JOIN (SELECT {} FROM {}) y".format(
                ", ".join([column1, column2]),
                self.__genSQL__(),
                ", ".join([column1, column2]),
                self.__genSQL__(),
            )
            self.__executeSQL__(
                "SELECT {}::float, {}::float FROM {};".format(n_c, n_d, table),
                title="Computing nc and nd.",
            )
            nc, nd = self._VERTICAPY_VARIABLES_["cursor"].fetchone()
            if kendall_type == "a":
                val = (nc - nd) / (n * (n - 1) / 2)
                Z = 3 * (nc - nd) / math.sqrt(n * (n - 1) * (2 * n + 5) / 2)
            elif kendall_type in ("b", "c"):
                self.__executeSQL__(
                    "SELECT SUM(verticapy_cnt * (verticapy_cnt - 1) * (2 * verticapy_cnt + 5)), SUM(verticapy_cnt * (verticapy_cnt - 1)), SUM(verticapy_cnt * (verticapy_cnt - 1) * (verticapy_cnt - 2)) FROM (SELECT {}, COUNT(*) AS verticapy_cnt FROM {} GROUP BY 1) VERTICAPY_SUBTABLE".format(
                        column1, self.__genSQL__()
                    ),
                    title="Computing vti.",
                )
                vt, v1_0, v2_0 = self._VERTICAPY_VARIABLES_["cursor"].fetchone()
                self.__executeSQL__(
                    "SELECT SUM(verticapy_cnt * (verticapy_cnt - 1) * (2 * verticapy_cnt + 5)), SUM(verticapy_cnt * (verticapy_cnt - 1)), SUM(verticapy_cnt * (verticapy_cnt - 1) * (verticapy_cnt - 2)) FROM (SELECT {}, COUNT(*) AS verticapy_cnt FROM {} GROUP BY 1) VERTICAPY_SUBTABLE".format(
                        column2, self.__genSQL__()
                    ),
                    title="Computing vui.",
                )
                vu, v1_1, v2_1 = self._VERTICAPY_VARIABLES_["cursor"].fetchone()
                v0 = n * (n - 1) * (2 * n + 5)
                v1 = v1_0 * v1_1 / (2 * n * (n - 1))
                v2 = v2_0 * v2_1 / (9 * n * (n - 1) * (n - 2))
                Z = (nc - nd) / math.sqrt((v0 - vt - vu) / 18 + v1 + v2)
                if kendall_type == "c":
                    sql = "SELECT APPROXIMATE_COUNT_DISTINCT({}) AS k, APPROXIMATE_COUNT_DISTINCT({}) AS r FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL".format(
                        column1, column2, self.__genSQL__(), column1, column2
                    )
                    self._VERTICAPY_VARIABLES_["cursor"].execute(sql)
                    k, r = self._VERTICAPY_VARIABLES_["cursor"].fetchone()
                    m = min(k, r)
                    val = 2 * (nc - nd) / (n * n * (m - 1) / m)
            pvalue = 2 * norm.sf(abs(Z))
        elif method == "cramer":
            sql = "SELECT APPROXIMATE_COUNT_DISTINCT({}) AS k, APPROXIMATE_COUNT_DISTINCT({}) AS r FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL".format(
                column1, column2, self.__genSQL__(), column1, column2
            )
            self._VERTICAPY_VARIABLES_["cursor"].execute(sql)
            k, r = self._VERTICAPY_VARIABLES_["cursor"].fetchone()
            x = val * val * n * min(k, r)
            pvalue = chi2.sf(x, (k - 1) * (r - 1))
        return (val, pvalue)

    # ---#
    def count(
        self,
        columns: list = [],
        percent: bool = True,
        sort_result: bool = True,
        desc: bool = True,
    ):
        """
    ---------------------------------------------------------------------------
    Aggregates the vDataFrame using a list of 'count' (Number of non-missing 
    values).

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all vColumns will be used.
    percent: bool, optional
        If set to True, the percentage of non-Missing value will be also computed.
    sort_result: bool, optional
        If set to True, the result will be sorted.
    desc: bool, optional
        If set to True and 'sort_result' is set to True, the result will be sorted desc.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        if isinstance(columns, str):
            columns = [columns]
        check_types(
            [
                ("columns", columns, [list],),
                ("percent", percent, [bool],),
                ("desc", desc, [bool],),
                ("sort_result", sort_result, [bool],),
            ]
        )
        columns_check(columns, self)
        columns = vdf_columns_names(columns, self)
        if not (columns):
            columns = self.get_columns()
        func = ["count", "percent"] if (percent) else ["count"]
        result = self.aggregate(func=func, columns=columns)
        if sort_result:
            sort = []
            for i in range(len(result.values["index"])):
                if percent:
                    sort += [
                        (
                            result.values["index"][i],
                            result.values["count"][i],
                            result.values["percent"][i],
                        )
                    ]
                else:
                    sort += [(result.values["index"][i], result.values["count"][i])]
            sort.sort(key=lambda tup: tup[1], reverse=desc)
            result.values["index"] = [elem[0] for elem in sort]
            result.values["count"] = [elem[1] for elem in sort]
            if percent:
                result.values["percent"] = [elem[2] for elem in sort]
        return result

    # ---#
    def cov(
        self,
        columns: list = [],
        focus: str = "",
        show: bool = True,
        ax=None,
        **style_kwds,
    ):
        """
    ---------------------------------------------------------------------------
    Computes the covariance matrix of the vDataFrame. 

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.
    focus: str, optional
        Focus the computation on only one vColumn.
    show: bool, optional
        If set to True, the Covariance Matrix will be drawn using Matplotlib.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.acf  : Computes the correlations between a vColumn and its lags.
    vDataFrame.corr : Computes the Correlation Matrix of the vDataFrame.
    vDataFrame.pacf : Computes the partial autocorrelations of the input vColumn.
    vDataFrame.regr : Computes the regression matrix of the vDataFrame.
        """
        if isinstance(columns, str):
            columns = [columns]
        check_types(
            [
                ("columns", columns, [list],),
                ("focus", focus, [str],),
                ("show", show, [bool],),
            ]
        )
        columns_check(columns, self)
        columns = vdf_columns_names(columns, self)
        if focus == "":
            return self.__aggregate_matrix__(
                method="cov", columns=columns, show=show, ax=ax, **style_kwds,
            )
        else:
            columns_check([focus], self)
            focus = vdf_columns_names([focus], self)[0]
            return self.__aggregate_vector__(
                focus, method="cov", columns=columns, show=show, ax=ax, **style_kwds,
            )

    # ---#
    def cummax(
        self, column: str, by: list = [], order_by: (dict, list) = [], name: str = ""
    ):
        """
    ---------------------------------------------------------------------------
    Adds a new vColumn to the vDataFrame by computing the cumulative maximum of
    the input vColumn.

    Parameters
    ----------
    column: str
        Input vColumn.
    by: list, optional
        vColumns used in the partition.
    order_by: dict / list, optional
        List of the vColumns to use to sort the data using asc order or
        dictionary of all sorting methods. For example, to sort by "column1"
        ASC and "column2" DESC, write {"column1": "asc", "column2": "desc"}
    name: str, optional
        Name of the new vColumn. If empty, a default name will be generated.

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame.rolling : Computes a customized moving window.
        """
        return self.rolling(
            func="max",
            columns=column,
            window=("UNBOUNDED", 0),
            by=by,
            order_by=order_by,
            name=name,
        )

    # ---#
    def cummin(
        self, column: str, by: list = [], order_by: (dict, list) = [], name: str = ""
    ):
        """
    ---------------------------------------------------------------------------
    Adds a new vColumn to the vDataFrame by computing the cumulative minimum of
    the input vColumn.

    Parameters
    ----------
    column: str
        Input vColumn.
    by: list, optional
        vColumns used in the partition.
    order_by: dict / list, optional
        List of the vColumns to use to sort the data using asc order or
        dictionary of all sorting methods. For example, to sort by "column1"
        ASC and "column2" DESC, write {"column1": "asc", "column2": "desc"}
    name: str, optional
        Name of the new vColumn. If empty, a default name will be generated.

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame.rolling : Computes a customized moving window.
        """
        return self.rolling(
            func="min",
            columns=column,
            window=("UNBOUNDED", 0),
            by=by,
            order_by=order_by,
            name=name,
        )

    # ---#
    def cumprod(
        self, column: str, by: list = [], order_by: (dict, list) = [], name: str = ""
    ):
        """
    ---------------------------------------------------------------------------
    Adds a new vColumn to the vDataFrame by computing the cumulative product of 
    the input vColumn.

    Parameters
    ----------
    column: str
        Input vColumn.
    by: list, optional
        vColumns used in the partition.
    order_by: dict / list, optional
        List of the vColumns to use to sort the data using asc order or
        dictionary of all sorting methods. For example, to sort by "column1"
        ASC and "column2" DESC, write {"column1": "asc", "column2": "desc"}
    name: str, optional
        Name of the new vColumn. If empty, a default name will be generated.

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame.rolling : Computes a customized moving window.
        """
        return self.rolling(
            func="prod",
            columns=column,
            window=("UNBOUNDED", 0),
            by=by,
            order_by=order_by,
            name=name,
        )

    # ---#
    def cumsum(
        self, column: str, by: list = [], order_by: (dict, list) = [], name: str = ""
    ):
        """
    ---------------------------------------------------------------------------
    Adds a new vColumn to the vDataFrame by computing the cumulative sum of the 
    input vColumn.

    Parameters
    ----------
    column: str
        Input vColumn.
    by: list, optional
        vColumns used in the partition.
    order_by: dict / list, optional
        List of the vColumns to use to sort the data using asc order or
        dictionary of all sorting methods. For example, to sort by "column1"
        ASC and "column2" DESC, write {"column1": "asc", "column2": "desc"}
    name: str, optional
        Name of the new vColumn. If empty, a default name will be generated.

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame.rolling : Computes a customized moving window.
        """
        return self.rolling(
            func="sum",
            columns=column,
            window=("UNBOUNDED", 0),
            by=by,
            order_by=order_by,
            name=name,
        )

    # ---#
    def current_relation(self, reindent: bool = True):
        """
    ---------------------------------------------------------------------------
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
        if reindent:
            return indentSQL(self.__genSQL__())
        else:
            return self.__genSQL__()

    # ---#
    def datecol(self):
        """
    ---------------------------------------------------------------------------
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

    # ---#
    def del_catalog(self):
        """
    ---------------------------------------------------------------------------
    Deletes the current vDataFrame catalog.

    Returns
    -------
    vDataFrame
        self
        """
        self.__update_catalog__(erase=True)
        return self

    # ---#
    def density(
        self,
        columns: list = [],
        bandwidth: float = 1.0,
        kernel: str = "gaussian",
        nbins: int = 50,
        xlim: tuple = None,
        ax=None,
        **style_kwds,
    ):
        """
    ---------------------------------------------------------------------------
    Draws the vColumns Density Plot.

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will 
        be selected.
    bandwidth: float, optional
        The bandwidth of the kernel.
    kernel: str, optional
        The method used for the plot.
            gaussian  : Gaussian Kernel.
            logistic  : Logistic Kernel.
            sigmoid   : Sigmoid Kernel.
            silverman : Silverman Kernel.
    nbins: int, optional
        Maximum number of points to use to evaluate the approximate density function.
        Increasing this parameter will increase the precision but will also increase 
        the time of the learning and the scoring phases.
    xlim: tuple, optional
        Set the x limits of the current axes.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object

    See Also
    --------
    vDataFrame[].hist : Draws the histogram of the vColumn based on an aggregation.
        """
        if isinstance(columns, str):
            columns = [columns]
        check_types(
            [
                ("columns", columns, [list],),
                ("kernel", kernel, ["gaussian", "logistic", "sigmoid", "silverman"],),
                ("bandwidth", bandwidth, [int, float],),
                ("nbins", nbins, [float, int],),
            ]
        )
        columns_check(columns, self)
        columns = vdf_columns_names(columns, self)
        if not (columns):
            columns = self.numcol()
        else:
            for column in columns:
                assert self[column].isnum(), TypeError(f"vColumn {column} is not numerical to draw KDE")
        assert columns, EmptyParameter("No Numerical Columns found to draw KDE.")
        from verticapy.plot import gen_colors
        from matplotlib.lines import Line2D

        colors = gen_colors()
        min_max = self.agg(func=["min", "max"], columns=columns)
        if not xlim:
            xmin = min(min_max["min"])
            xmax = max(min_max["max"])
        else:
            xmin, xmax = xlim
        custom_lines = []
        for idx, column in enumerate(columns):
            param = {"color": colors[idx % len(colors)]}
            ax = self[column].density(
                bandwidth=bandwidth,
                kernel=kernel,
                nbins=nbins,
                xlim=(xmin, xmax),
                ax=ax,
                **updated_dict(param, style_kwds, idx),
            )
            custom_lines += [
                Line2D([0], [0], color=colors[idx % len(colors)], lw=4),
            ]
        ax.legend(custom_lines, columns, loc="center left", bbox_to_anchor=[1, 0.5])
        ax.set_ylim(bottom=0)
        return ax

    # ---#
    def describe(self, method: str = "auto", columns: list = [], unique: bool = True, ncols_block: int = 20,):
        """
    ---------------------------------------------------------------------------
    Aggregates the vDataFrame using multiple statistical aggregations: min, 
    max, median, unique... depending on the types of the vColumns.

    Parameters
    ----------
    method: str, optional
        The describe method.
            all         : Aggregates all selected vDataFrame vColumns different 
                methods depending on the vColumn type (numerical dtype: numerical; 
                timestamp dtype: range; categorical dtype: length)
            auto        : Sets the method to 'numerical' if at least one vColumn 
                of the vDataFrame is numerical, 'categorical' otherwise.
            categorical : Uses only categorical aggregations.
            length      : Aggregates the vDataFrame using numerical aggregation 
                on the length of all selected vColumns.
             numerical   : Uses only numerical descriptive statistics which are 
                 computed in a faster way than the 'aggregate' method.
            range       : Aggregates the vDataFrame using multiple statistical
                aggregations - min, max, range...
            statistics  : Aggregates the vDataFrame using multiple statistical 
                aggregations - kurtosis, skewness, min, max...
    columns: list, optional
        List of the vColumns names. If empty, the vColumns will be selected
        depending on the parameter 'method'.
    unique: bool, optional
        If set to True, the cardinality of each element will be computed.
    ncols_block: int, optional
        Number of columns used per query. For query performance, it is better to
        pick up a balanced number. A small number will lead to the generation of
        multiple queries. A higher number will lead to the generation of one big
        SQL query.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        if isinstance(method, str):
            method = method.lower()
        if isinstance(columns, str):
            columns = [columns]
        check_types(
            [
                (
                    "method",
                    method,
                    [
                        "numerical",
                        "categorical",
                        "statistics",
                        "length",
                        "range",
                        "all",
                        "auto",
                    ],
                ),
                ("columns", columns, [list],),
                ("unique", unique, [bool],),
                ("ncols_block", ncols_block, [int],),
            ]
        )
        if method == "auto":
            method = "numerical" if (self.numcol()) else "categorical"
        columns_check(columns, self)
        columns = vdf_columns_names(columns, self)
        for i in range(len(columns)):
            columns[i] = str_column(columns[i])
        if method == "numerical":
            if not (columns):
                columns = self.numcol()
            else:
                for column in columns:
                    assert self[column].isnum(), TypeError(f"vColumn {column} must be numerical to run describe using parameter method = 'numerical'")
            assert columns, EmptyParameter("No Numerical Columns found to run describe using parameter method = 'numerical'.")
            if ncols_block < len(columns):
                i = ncols_block
                result = self.describe(method=method, columns=columns[0:ncols_block], unique=unique, ncols_block=ncols_block,).transpose()
                while i < len(columns):
                    columns_tmp = columns[i:i+ncols_block]
                    if columns_tmp:
                        result_tmp = self.describe(method=method, columns=columns_tmp, unique=unique, ncols_block=ncols_block,).transpose()
                        for elem in result_tmp.values:
                            if elem != "index":
                                result.values[elem] = result_tmp[elem]
                    i += ncols_block
                return result.transpose()
            try:
                version(
                    cursor=self._VERTICAPY_VARIABLES_["cursor"], condition=[9, 0, 0]
                )
                idx = [
                    "index",
                    "count",
                    "mean",
                    "std",
                    "min",
                    "25%",
                    "50%",
                    "75%",
                    "max",
                ]
                values = {}
                for key in idx:
                    values[key] = []
                col_to_compute = []
                for column in columns:
                    if self[column].isnum():
                        for fun in idx[1:]:
                            pre_comp = self.__get_catalog_value__(column, fun)
                            if pre_comp == "VERTICAPY_NOT_PRECOMPUTED":
                                col_to_compute += [column]
                                break
                    elif verticapy.options["print_info"]:
                        warning_message = "The vColumn {} is not numerical, it was ignored.\nTo get statistical information about all different variables, please use the parameter method = 'categorical'.".format(
                            column
                        )
                        warnings.warn(warning_message, Warning)
                for column in columns:
                    if column not in col_to_compute:
                        values["index"] += [column.replace('"', "")]
                        for fun in idx[1:]:
                            values[fun] += [self.__get_catalog_value__(column, fun)]
                if col_to_compute:
                    query = "SELECT SUMMARIZE_NUMCOL({}) OVER () FROM {}".format(
                        ", ".join(
                            [
                                elem
                                if not (self[elem].isbool())
                                else "{}::int".format(elem)
                                for elem in col_to_compute
                            ]
                        ),
                        self.__genSQL__(),
                    )
                    self.__executeSQL__(
                        query,
                        title="Computes the descriptive statistics of all numerical columns using SUMMARIZE_NUMCOL.",
                    )
                    query_result = self._VERTICAPY_VARIABLES_["cursor"].fetchall()
                    for i, key in enumerate(idx):
                        values[key] += [elem[i] for elem in query_result]
                    columns = [elem for elem in values["index"]]
            except:
                values = self.aggregate(
                    ["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
                    columns=columns,
                    ncols_block=ncols_block,
                ).values
            if unique:
                values["unique"] = self.aggregate(["unique"], columns=columns, ncols_block=ncols_block,).values[
                    "unique"
                ]
        elif method == "categorical":
            func = ["dtype", "unique", "count", "top", "top_percent"]
            if not (unique):
                del func[1]
            values = self.aggregate(func, columns=columns, ncols_block=ncols_block,).values
        elif method == "statistics":
            func = [
                "dtype",
                "percent",
                "count",
                "unique",
                "avg",
                "stddev",
                "min",
                "1%",
                "10%",
                "25%",
                "median",
                "75%",
                "90%",
                "99%",
                "max",
                "skewness",
                "kurtosis",
            ]
            if not (unique):
                del func[3]
            values = self.aggregate(func=func, columns=columns, ncols_block=ncols_block,).values
        elif method == "length":
            if not (columns):
                columns = self.get_columns()
            func = [
                "dtype",
                "percent",
                "count",
                "unique",
                "SUM(CASE WHEN LENGTH({}::varchar) = 0 THEN 1 ELSE 0 END) AS empty",
                "AVG(LENGTH({}::varchar)) AS avg_length",
                "STDDEV(LENGTH({}::varchar)) AS stddev_length",
                "MIN(LENGTH({}::varchar))::int AS min_length",
                "APPROXIMATE_PERCENTILE(LENGTH({}::varchar) USING PARAMETERS percentile = 0.25)::int AS '25%_length'",
                "APPROXIMATE_PERCENTILE(LENGTH({}::varchar) USING PARAMETERS percentile = 0.5)::int AS '50%_length'",
                "APPROXIMATE_PERCENTILE(LENGTH({}::varchar) USING PARAMETERS percentile = 0.75)::int AS '75%_length'",
                "MAX(LENGTH({}::varchar))::int AS max_length",
            ]
            if not (unique):
                del func[3]
            values = self.aggregate(func=func, columns=columns, ncols_block=ncols_block,).values
        elif method == "range":
            if not (columns):
                columns = []
                all_cols = self.get_columns()
                for idx, column in enumerate(all_cols):
                    if self[column].isnum() or self[column].isdate():
                        columns += [column]
            func = ["dtype", "percent", "count", "unique", "min", "max", "range"]
            if not (unique):
                del func[3]
            values = self.aggregate(func=func, columns=columns, ncols_block=ncols_block,).values
        elif method == "all":
            datecols, numcol, catcol = [], [], []
            if not(columns):
                columns = self.get_columns()
            for elem in columns:
                if self[elem].isnum():
                    numcol += [elem]
                elif self[elem].isdate():
                    datecols += [elem]
                else:
                    catcol += [elem]
            values = self.aggregate(
                func=[
                    "dtype",
                    "percent",
                    "count",
                    "unique",
                    "top",
                    "top_percent",
                    "avg",
                    "stddev",
                    "min",
                    "25%",
                    "50%",
                    "75%",
                    "max",
                    "range",
                ],
                columns=numcol,
                ncols_block=ncols_block,
            ).values
            values["empty"] = [None] * len(numcol)
            if datecols:
                tmp = self.aggregate(
                    func=[
                        "dtype",
                        "percent",
                        "count",
                        "unique",
                        "top",
                        "top_percent",
                        "min",
                        "max",
                        "range",
                    ],
                    columns=datecols,
                ).values
                for elem in [
                    "index",
                    "dtype",
                    "percent",
                    "count",
                    "unique",
                    "top",
                    "top_percent",
                    "min",
                    "max",
                    "range",
                ]:
                    values[elem] += tmp[elem]
                for elem in ["avg", "stddev", "25%", "50%", "75%", "empty"]:
                    values[elem] += [None] * len(datecols)
            if catcol:
                tmp = self.aggregate(
                    func=[
                        "dtype",
                        "percent",
                        "count",
                        "unique",
                        "top",
                        "top_percent",
                        "AVG(LENGTH({}::varchar)) AS avg",
                        "STDDEV(LENGTH({}::varchar)) AS stddev",
                        "MIN(LENGTH({}::varchar))::int AS min",
                        "APPROXIMATE_PERCENTILE(LENGTH({}::varchar) USING PARAMETERS percentile = 0.25)::int AS '25%'",
                        "APPROXIMATE_PERCENTILE(LENGTH({}::varchar) USING PARAMETERS percentile = 0.5)::int AS '50%'",
                        "APPROXIMATE_PERCENTILE(LENGTH({}::varchar) USING PARAMETERS percentile = 0.75)::int AS '75%'",
                        "MAX(LENGTH({}::varchar))::int AS max",
                        "MAX(LENGTH({}::varchar))::int - MIN(LENGTH({}::varchar))::int AS range",
                        "SUM(CASE WHEN LENGTH({}::varchar) = 0 THEN 1 ELSE 0 END) AS empty",
                    ],
                    columns=catcol,
                    ncols_block=ncols_block,
                ).values
                for elem in [
                    "index",
                    "dtype",
                    "percent",
                    "count",
                    "unique",
                    "top",
                    "top_percent",
                    "avg",
                    "stddev",
                    "min",
                    "25%",
                    "50%",
                    "75%",
                    "max",
                    "range",
                    "empty",
                ]:
                    values[elem] += tmp[elem]
            dtype, percent = {}, {}
            if isnotebook():
                for i in range(len(values["index"])):
                    dtype[values["index"][i]] = values["dtype"][i]
                    percent[values["index"][i]] = values["percent"][i]
                del values["dtype"]
                del values["percent"]
            return tablesample(values, percent=percent, dtype=dtype).transpose()
        self.__update_catalog__(tablesample(values).transpose().values)
        values["index"] = [str_column(elem) for elem in values["index"]]
        for elem in values:
            for i in range(len(values[elem])):
                if isinstance(values[elem][i], decimal.Decimal):
                    values[elem][i] = float(values[elem][i])
        return tablesample(values)

    # ---#
    def drop(self, columns: list = []):
        """
    ---------------------------------------------------------------------------
    Drops the input vColumns from the vDataFrame. Dropping vColumns means 
    not selecting them in the final SQL code generation.
    Be Careful when using this method. It can make the vDataFrame structure 
    heavier if some other vColumns are computed using the dropped vColumns.

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names.

    Returns
    -------
    vDataFrame
        self
        """
        if isinstance(columns, str):
            columns = [columns]
        check_types([("columns", columns, [list],)])
        columns_check(columns, self)
        columns = vdf_columns_names(columns, self)
        for column in columns:
            self[column].drop()
        return self

    # ---#
    def drop_duplicates(
        self, columns: list = [],
    ):
        """
    ---------------------------------------------------------------------------
    Filters the duplicated using a partition by the input vColumns.

    \u26A0 Warning : Dropping duplicates will make the vDataFrame structure 
                     heavier. It is recommended to always check the current structure 
                     using the 'current_relation' method and to save it using the 
                     'to_db' method with the parameters 'inplace = True' and 
                     'relation_type = table'

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all vColumns will be selected.

    Returns
    -------
    vDataFrame
        self
        """
        if isinstance(columns, str):
            columns = [columns]
        check_types(
            [("columns", columns, [list],),]
        )
        columns_check(columns, self)
        count = self.duplicated(columns=columns, count=True)
        if count:
            columns = (
                self.get_columns()
                if not (columns)
                else vdf_columns_names(columns, self)
            )
            name = (
                "__verticapy_duplicated_index__"
                + str(random.randint(0, 10000000))
                + "_"
            )
            self.eval(
                name=name,
                expr="ROW_NUMBER() OVER (PARTITION BY {})".format(", ".join(columns)),
            )
            self.filter('"{}" = 1'.format(name),)
            self._VERTICAPY_VARIABLES_["exclude_columns"] += ['"{}"'.format(name)]
        elif verticapy.options["print_info"]:
            print("No duplicates detected.")
        return self

    # ---#
    def dropna(
        self, columns: list = [],
    ):
        """
    ---------------------------------------------------------------------------
    Filters the vDataFrame where the input vColumns are missing.

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all vColumns will be selected.

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame.filter: Filters the data using the input expression.
        """
        if isinstance(columns, str):
            columns = [columns]
        check_types(
            [("columns", columns, [list],),]
        )
        columns_check(columns, self)
        columns = (
            self.get_columns() if not (columns) else vdf_columns_names(columns, self)
        )
        total = self.shape()[0]
        print_info = verticapy.options["print_info"]
        for column in columns:
            verticapy.options["print_info"] = False
            self[column].dropna()
            verticapy.options["print_info"] = print_info
        if verticapy.options["print_info"]:
            total -= self.shape()[0]
            if total == 0:
                print("Nothing was filtered.")
            else:
                conj = "s were " if total > 1 else " was "
                print("{} element{}filtered.".format(total, conj))
        return self

    # ---#
    def dtypes(self):
        """
    ---------------------------------------------------------------------------
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

    # ---#
    def duplicated(self, columns: list = [], count: bool = False, limit: int = 30):
        """
    ---------------------------------------------------------------------------
    Returns the duplicated values.

    Parameters
    ----------
    columns: list, optional
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
        if isinstance(columns, str):
            columns = [columns]
        check_types(
            [
                ("columns", columns, [list],),
                ("count", count, [bool],),
                ("limit", limit, [int, float],),
            ]
        )
        columns_check(columns, self)
        columns = (
            self.get_columns() if not (columns) else vdf_columns_names(columns, self)
        )
        query = "(SELECT *, ROW_NUMBER() OVER (PARTITION BY {}) AS duplicated_index FROM {}) duplicated_index_table WHERE duplicated_index > 1".format(
            ", ".join(columns), self.__genSQL__()
        )
        self.__executeSQL__(
            query="SELECT COUNT(*) FROM {}".format(query),
            title="Computes the number of duplicates.",
        )
        total = self._VERTICAPY_VARIABLES_["cursor"].fetchone()[0]
        if count:
            return total
        result = to_tablesample(
            "SELECT {}, MAX(duplicated_index) AS occurrence FROM {} GROUP BY {} ORDER BY occurrence DESC LIMIT {}".format(
                ", ".join(columns), query, ", ".join(columns), limit
            ),
            self._VERTICAPY_VARIABLES_["cursor"],
        )
        self.__executeSQL__(
            query="SELECT COUNT(*) FROM (SELECT {}, MAX(duplicated_index) AS occurrence FROM {} GROUP BY {}) t".format(
                ", ".join(columns), query, ", ".join(columns)
            ),
            title="Computes the number of distinct duplicates.",
        )
        result.count = self._VERTICAPY_VARIABLES_["cursor"].fetchone()[0]
        return result

    # ---#
    def empty(self):
        """
    ---------------------------------------------------------------------------
    Returns True if the vDataFrame is empty.

    Returns
    -------
    bool
        True if the vDataFrame has no vColumns.
        """
        return not (self.get_columns())

    # ---#
    def eval(self, name: str, expr: str):
        """
    ---------------------------------------------------------------------------
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
        check_types([("name", name, [str],), ("expr", expr, [str],)])
        name = str_column(name.replace('"', "_"))
        assert not(column_check_ambiguous(name, self.get_columns())), f"A vColumn has already the alias {name}.\nBy changing the parameter 'name', you'll be able to solve this issue."
        try:
            ctype = get_data_types(
                "SELECT {} AS {} FROM {} LIMIT 0".format(expr, name, self.__genSQL__()),
                self._VERTICAPY_VARIABLES_["cursor"],
                name.replace('"', "").replace("'", "''"),
            )
        except:
            try:
                ctype = get_data_types(
                    "SELECT {} AS {} FROM {} LIMIT 0".format(
                        expr, name, self.__genSQL__()
                    ),
                    self._VERTICAPY_VARIABLES_["cursor"],
                    name.replace('"', "").replace("'", "''"),
                    self._VERTICAPY_VARIABLES_["schema_writing"],
                )
            except:
                raise QueryError(f"The expression '{expr}' seems to be incorrect.\nBy turning on the SQL with the 'set_option' function, you'll print the SQL code generation and probably see why the evaluation didn't work.")
        ctype = ctype if (ctype) else "undefined"
        category = category_from_type(ctype=ctype)
        all_cols, max_floor = self.get_columns(), 0
        for column in all_cols:
            if (str_column(column) in expr) or (
                re.search(re.compile("\\b{}\\b".format(column.replace('"', ""))), expr)
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
        self._VERTICAPY_VARIABLES_["columns"] += [name]
        self.__add_to_history__(
            "[Eval]: A new vColumn {} was added to the vDataFrame.".format(name)
        )
        return self

    # ---#
    def expected_store_usage(self, unit: str = "b"):
        """
    ---------------------------------------------------------------------------
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
        check_types([("unit", unit, [str],)])
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
            "expected_size ({})".format(unit),
            "max_size ({})".format(unit),
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

    # ---#
    def explain(self, digraph: bool = False):
        """
    ---------------------------------------------------------------------------
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
        query = "EXPLAIN SELECT * FROM {}".format(self.__genSQL__())
        self.__executeSQL__(query=query, title="Explaining the Current Relation")
        result = self._VERTICAPY_VARIABLES_["cursor"].fetchall()
        result = [elem[0] for elem in result]
        result = "\n".join(result)
        if not (digraph):
            result = result.replace("------------------------------\n", "")
            result = result.replace("\\n", "\n\t")
            result = result.replace(", ", ",").replace(",", ", ").replace("\n}", "}")
        else:
            result = "digraph G {" + result.split("digraph G {")[1]
        return result

    # ---#
    def fillna(
        self, val: dict = {}, method: dict = {}, numeric_only: bool = False,
    ):
        """
    ---------------------------------------------------------------------------
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
        check_types(
            [
                ("val", val, [dict],),
                ("method", method, [dict],),
                ("numeric_only", numeric_only, [bool],),
            ]
        )
        columns_check([elem for elem in val] + [elem for elem in method], self)
        print_info = verticapy.options["print_info"]
        verticapy.options["print_info"] = False
        try:
            if not (val) and not (method):
                cols = self.get_columns()
                for column in cols:
                    if numeric_only:
                        if self[column].isnum():
                            self[column].fillna(method="auto",)
                    else:
                        self[column].fillna(method="auto",)
            else:
                for column in val:
                    self[vdf_columns_names([column], self)[0]].fillna(val=val[column],)
                for column in method:
                    self[vdf_columns_names([column], self)[0]].fillna(
                        method=method[column],
                    )
            verticapy.options["print_info"] = print_info
            return self
        except:
            verticapy.options["print_info"] = print_info
            raise

    # ---#
    def filter(
        self, conditions: (list, str) = [], *args, **kwds,
    ):
        """
    ---------------------------------------------------------------------------
    Filters the vDataFrame using the input expressions.

    Parameters
    ---------- 
    conditions: list / str, optional
        List of expressions. For example to keep only the records where the 
        vColumn 'column' is greater than 5 and lesser than 10 you can write 
        ['"column" > 5', '"column" < 10'].

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame.at_time      : Filters the data at the input time.
    vDataFrame.between_time : Filters the data between two time ranges.
    vDataFrame.first        : Filters the data by only keeping the first records.
    vDataFrame.last         : Filters the data by only keeping the last records.
    vDataFrame.search       : Searches the elements which matches with the input 
        conditions.
        """
        check_types(
            [("conditions", conditions, [list, str,],),]
        )
        count = self.shape()[0]
        conj = "s were " if count > 1 else " was "
        if not (isinstance(conditions, str)) or (args):
            if isinstance(conditions, str) or not (isinstance(conditions, Iterable)):
                conditions = [conditions]
            else:
                conditions = list(conditions)
            conditions += list(args)
            for condition in conditions:
                self.filter(str(condition), print_info=False,)
            count -= self.shape()[0]
            if count > 0:
                if verticapy.options["print_info"]:
                    print("{} element{}filtered".format(count, conj))
                self.__add_to_history__(
                    "[Filter]: {} element{}filtered using the filter '{}'".format(
                        count, conj, conditions
                    )
                )
            elif verticapy.options["print_info"]:
                print("Nothing was filtered.")
        else:
            max_pos = 0
            columns_tmp = [elem for elem in self._VERTICAPY_VARIABLES_["columns"]]
            for column in columns_tmp:
                max_pos = max(max_pos, len(self[column].transformations) - 1)
            new_count = self.shape()[0]
            self._VERTICAPY_VARIABLES_["where"] += [(conditions, max_pos)]
            try:
                self._VERTICAPY_VARIABLES_["cursor"].execute(
                    "SELECT COUNT(*) FROM {}".format(self.__genSQL__())
                )
                new_count = self._VERTICAPY_VARIABLES_["cursor"].fetchone()[0]
                count -= new_count
            except:
                del self._VERTICAPY_VARIABLES_["where"][-1]
                if verticapy.options["print_info"]:
                    warning_message = "The expression '{}' is incorrect.\nNothing was filtered.".format(
                        conditions
                    )
                    warnings.warn(warning_message, Warning)
                return self
            if count > 0:
                self.__update_catalog__(erase=True)
                self._VERTICAPY_VARIABLES_["count"] = new_count
                conj = "s were " if count > 1 else " was "
                if verticapy.options["print_info"] and "print_info" not in kwds:
                    print("{} element{}filtered.".format(count, conj))
                self.__add_to_history__(
                    "[Filter]: {} element{}filtered using the filter '{}'".format(
                        count, conj, conditions
                    )
                )
            else:
                del self._VERTICAPY_VARIABLES_["where"][-1]
                if verticapy.options["print_info"] and "print_info" not in kwds:
                    print("Nothing was filtered.")
        return self

    # ---#
    def first(
        self, ts: str, offset: str,
    ):
        """
    ---------------------------------------------------------------------------
    Filters the vDataFrame by only keeping the first records.

    Parameters
    ----------
    ts: str
        TS (Time Series) vColumn to use to filter the data. The vColumn type must be
        date like (date, datetime, timestamp...)
    offset: str
        Interval offset. For example, to filter and keep only the first 6 months of
        records, offset should be set to '6 months'.

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame.at_time      : Filters the data at the input time.
    vDataFrame.between_time : Filters the data between two time ranges.
    vDataFrame.filter       : Filters the data using the input expression.
    vDataFrame.last         : Filters the data by only keeping the last records.
        """
        check_types(
            [("ts", ts, [str],), ("offset", offset, [str],),]
        )
        ts = vdf_columns_names([ts], self)[0]
        query = "SELECT (MIN({}) + '{}'::interval)::varchar FROM {}".format(
            ts, offset, self.__genSQL__()
        )
        self.__executeSQL__(query, title="Gets the vDataFrame first values.")
        first_date = self._VERTICAPY_VARIABLES_["cursor"].fetchone()[0]
        self.filter("{} <= '{}'".format(ts, first_date),)
        return self

    # ---#
    def get_columns(self, exclude_columns: list = []):
        """
    ---------------------------------------------------------------------------
    Returns the vDataFrame vColumns.

    Parameters
    ----------
    exclude_columns: list, optional
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
        if isinstance(exclude_columns, str):
            exclude_columns = [columns]
        check_types([("exclude_columns", exclude_columns, [list],)])
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

    # ---#
    def get_dummies(
        self,
        columns: list = [],
        max_cardinality: int = 12,
        prefix_sep: str = "_",
        drop_first: bool = True,
        use_numbers_as_suffix: bool = False,
    ):
        """
    ---------------------------------------------------------------------------
    Encodes the vColumns using the One Hot Encoding algorithm.

    Parameters
    ----------
    columns: list, optional
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
        check_types(
            [
                ("columns", columns, [list],),
                ("max_cardinality", max_cardinality, [int, float],),
                ("prefix_sep", prefix_sep, [str],),
                ("drop_first", drop_first, [bool],),
                ("use_numbers_as_suffix", use_numbers_as_suffix, [bool],),
            ]
        )
        columns_check(columns, self)
        cols_hand = True if (columns) else False
        columns = (
            self.get_columns() if not (columns) else vdf_columns_names(columns, self)
        )
        for column in columns:
            if self[column].nunique(True) < max_cardinality:
                self[column].get_dummies(
                    "", prefix_sep, drop_first, use_numbers_as_suffix
                )
            elif cols_hand and verticapy.options["print_info"]:
                warning_message = "The vColumn {} was ignored because of its high cardinality.\nIncrease the parameter 'max_cardinality' to solve this issue or use directly the vColumn get_dummies method.".format(
                    column
                )
                warnings.warn(warning_message, Warning)
        return self

    one_hot_encode = get_dummies
    # ---#
    def groupby(
        self, columns: list, expr: list = [],
    ):
        """
    ---------------------------------------------------------------------------
    Aggregates the vDataFrame by grouping the elements.

    Parameters
    ----------
    columns: list
        List of the vColumns used for the grouping. It can also be customized 
        expressions.
    expr: list, optional
        List of the different aggregations in pure SQL. Aliases can also be given. 
        'SUM(column)' or 'AVG(column) AS my_new_alias' are correct whereas 'AVG' 
        is incorrect. Aliases are recommended to keep the track of the different 
        features and not have ambiguous names. The function MODE does not exist in 
        SQL for example but can be obtained using the 'analytic' method first and 
        then by grouping the result.

    Returns
    -------
    vDataFrame
        object result of the grouping.

    See Also
    --------
    vDataFrame.append   : Merges the vDataFrame with another relation.
    vDataFrame.analytic : Adds a new vColumn to the vDataFrame by using an advanced 
        analytical function on a specific vColumn.
    vDataFrame.join     : Joins the vDataFrame with another relation.
    vDataFrame.sort     : Sorts the vDataFrame.
        """
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(expr, str):
            expr = [expr]
        check_types([("columns", columns, [list],), ("expr", expr, [list],)])
        for i in range(len(columns)):
            column = vdf_columns_names([columns[i]], self)
            if column:
                columns[i] = column[0]
        relation = "(SELECT {} FROM {} GROUP BY {}) VERTICAPY_SUBTABLE".format(
            ", ".join([str(elem) for elem in columns] + [str(elem) for elem in expr]),
            self.__genSQL__(),
            ", ".join(
                [str(i + 1) for i in range(len([str(elem) for elem in columns]))]
            ),
        )
        return self.__vdf_from_relation__(
            relation,
            "groupby",
            "[Groupby]: The columns were grouped by {}".format(
                ", ".join([str(elem) for elem in columns])
            ),
        )

    # ---#
    def hchart(
        self,
        x: (str, list) = None,
        y: (str, list) = None,
        z: (str, list) = None,
        c: (str, list) = None,
        aggregate: bool = True,
        kind: str = "boxplot",
        width: int = 600,
        height: int = 400,
        options: dict = {},
        h: float = -1,
        max_cardinality: int = 10,
        limit: int = 10000,
        drilldown: bool = False,
        stock: bool = False,
        alpha: float = 0.25,
    ):
        """
    ---------------------------------------------------------------------------
    [Beta Version]
    Draws responsive charts using the High Chart API: 
    https://api.highcharts.com/highcharts/

    The returned object can be customized using the API parameters and the 
    'set_dict_options' method.

    \u26A0 Warning : This function uses the unsupported HighChart Python API. 
                     For more information, see python-hicharts repository:
                     https://github.com/kyper-data/python-highcharts

    Parameters
    ----------
    x / y / z / c: str / list
        The vColumns and aggregations used to draw the chart. These will depend 
        on the chart type. You can also specify an expression, but it must be a SQL 
        statement. For example: AVG(column1) + SUM(column2) AS new_name.

            area / area_ts / line / spline
                x: numerical or type date like vColumn.
                y: a single expression or list of expressions used to draw the plot
                z: [OPTIONAL] vColumn representing the different categories 
                    (only if y is a single vColumn)
            area_range
                x: numerical or date type vColumn.
                y: list of three expressions [expression, lower bound, upper bound]
            bar (single) / donut / donut3d / hist (single) / pie / pie_half / pie3d
                x: vColumn used to compute the categories.
                y: [OPTIONAL] numerical expression representing the categories values. 
                    If empty, COUNT(*) is used as the default aggregation.
            bar (double / drilldown) / hist (double / drilldown) / pie (drilldown) 
            / stacked_bar / stacked_hist
                x: vColumn used to compute the first category.
                y: vColumn used to compute the second category.
                z: [OPTIONAL] numerical expression representing the different categories values. 
                    If empty, COUNT(*) is used as the default aggregation.
            biserial / boxplot / pearson / kendall / pearson / spearman
                x: list of the vColumns used to draw the Chart.
            bubble / scatter
                x: numerical vColumn.
                y: numerical vColumn.
                z: numerical vColumn (bubble size in case of bubble plot, third 
                     dimension in case of scatter plot)
                c: [OPTIONAL] [OPTIONAL] vColumn used to compute the different categories.
            candlestick
                x: date type vColumn.
                y: Can be a numerical vColumn or list of 5 expressions 
                    [last quantile, maximum, minimum, first quantile, volume]
            negative_bar
                x: binary vColumn used to compute the first category.
                y: vColumn used to compute the second category.
                z: [OPTIONAL] numerical expression representing the categories values. 
                    If empty, COUNT(*) is used as the default aggregation.
            spider
                x: vColumn used to compute the different categories.
                y: [OPTIONAL] Can be a list of the expressions used to draw the Plot or a single expression. 
                    If empty, COUNT(*) is used as the default aggregation.
    aggregate: bool, optional
        If set to True, the input vColumns will be aggregated.
    kind: str, optional
        Chart Type.
            area         : Area Chart
            area_range   : Area Range Chart
            area_ts      : Area Chart with Time Series Design
            bar          : Bar Chart
            biserial     : Biserial Point Matrix (Correlation between binary
                             variables and numerical)
            boxplot      : Box Plot
            bubble       : Bubble Plot
            candlestick  : Candlestick and Volumes (Time Series Special Plot)
            cramer       : Cramer's V Matrix (Correlation between categories)
            donut        : Donut Chart
            donut3d      : 3D Donut Chart
            heatmap      : Heatmap
            hist         : Histogram
            kendall      : Kendall Correlation Matrix. The method will compute the Tau-B 
                           coefficients.
                           \u26A0 Warning : This method uses a CROSS JOIN during computation 
                                            and is therefore computationally expensive at 
                                            O(n * n), where n is the total count of the 
                                            vDataFrame.
            line         : Line Plot
            negative_bar : Multi Bar Chart for binary classes
            pearson      : Pearson Correlation Matrix
            pie          : Pie Chart
            pie_half     : Half Pie Chart
            pie3d        : 3D Pie Chart
            scatter      : Scatter Plot
            spider       : Spider Chart
            spline       : Spline Plot
            stacked_bar  : Stacked Bar Chart
            stacked_hist : Stacked Histogram
            spearman     : Spearman's Correlation Matrix
    width: int, optional
        Chart Width.
    height: int, optional
        Chart Height.
    options: dict, optional
        High Chart Dictionary to use to customize the Chart. Look at the API 
        documentation to know the different options.
    h: float, optional
        Interval width of the bar. If empty, an optimized value will be used.
    max_cardinality: int, optional
        Maximum number of the vColumn distinct elements.
    limit: int, optional
        Maximum number of elements to draw.
    drilldown: bool, optional
        Drilldown Chart: Only possible for Bars, Histograms, donuts and pies.
                          Instead of drawing 2D charts, this option allows you
                          to add a drilldown effect to 1D Charts.
    stock: bool, optional
        Stock Chart: Only possible for Time Series. The design of the Time
                     Series is dragable and have multiple options.
    alpha: float, optional
        Value used to determine the position of the upper and lower quantile 
        (Used when kind is set to 'candlestick')

    Returns
    -------
    Highchart
        Chart Object
        """
        check_types([("kind", kind, [str],)])
        kind = kind.lower()
        check_types(
            [
                ("aggregate", aggregate, [bool],),
                (
                    "kind",
                    kind,
                    [
                        "area",
                        "area_range",
                        "area_ts",
                        "bar",
                        "boxplot",
                        "bubble",
                        "candlestick",
                        "donut",
                        "donut3d",
                        "heatmap",
                        "hist",
                        "line",
                        "negative_bar",
                        "pie",
                        "pie_half",
                        "pie3d",
                        "scatter",
                        "spider",
                        "spline",
                        "stacked_bar",
                        "stacked_hist",
                        "pearson",
                        "kendall",
                        "cramer",
                        "biserial",
                        "spearman",
                    ],
                ),
                ("options", options, [dict],),
                ("width", width, [int, float],),
                ("height", height, [int, float],),
                ("drilldown", drilldown, [bool],),
                ("stock", stock, [bool],),
                ("limit", limit, [int, float],),
                ("max_cardinality", max_cardinality, [int, float],),
                ("h", h, [int, float],),
                ("alpha", alpha, [float],),
            ]
        )
        from verticapy.hchart import hchart_from_vdf

        try:
            return hchart_from_vdf(
                self,
                x,
                y,
                z,
                c,
                aggregate,
                kind,
                width,
                height,
                options,
                h,
                max_cardinality,
                limit,
                drilldown,
                stock,
                alpha,
            )
        except:
            return hchart_from_vdf(
                self,
                x,
                y,
                z,
                c,
                not (aggregate),
                kind,
                width,
                height,
                options,
                h,
                max_cardinality,
                limit,
                drilldown,
                stock,
                alpha,
            )

    # ---#
    def head(self, limit: int = 5):
        """
    ---------------------------------------------------------------------------
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

    # ---#
    def heatmap(
        self,
        columns: list,
        method: str = "count",
        of: str = "",
        h: tuple = (None, None),
        ax=None,
        **style_kwds,
    ):
        """
    ---------------------------------------------------------------------------
    Draws the Heatmap of the two input vColumns.

    Parameters
    ----------
    columns: list
        List of the vColumns names. The list must have two elements.
    method: str, optional
        The method to use to aggregate the data.
            count   : Number of elements.
            density : Percentage of the distribution.
            mean    : Average of the vColumn 'of'.
            min     : Minimum of the vColumn 'of'.
            max     : Maximum of the vColumn 'of'.
            sum     : Sum of the vColumn 'of'.
            q%      : q Quantile of the vColumn 'of (ex: 50% to get the median).
        It can also be a cutomized aggregation (ex: AVG(column1) + 5).
    of: str, optional
        The vColumn to use to compute the aggregation.
    h: tuple, optional
        Interval width of the vColumns 1 and 2 bars. Optimized h will be computed 
        if the parameter is empty or invalid.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object

    See Also
    --------
    vDataFrame.pivot_table  : Draws the pivot table of vColumns based on an aggregation.
        """
        if isinstance(columns, str):
            columns = [columns]
        check_types(
            [
                ("columns", columns, [list],),
                ("method", method, [str],),
                ("of", of, [str],),
                ("h", h, [list],),
            ]
        )
        columns_check(columns, self, [2])
        columns = vdf_columns_names(columns, self)
        if of:
            columns_check([of], self)
            of = vdf_columns_names([of], self)[0]
        for column in columns:
            assert self[column].isnum(), TypeError(f"vColumn {column} must be numerical to draw the Heatmap.")
        from verticapy.plot import pivot_table

        min_max = self.agg(func=["min", "max"], columns=columns).transpose()

        ax = pivot_table(
            self,
            columns,
            method,
            of,
            h,
            (0, 0),
            True,
            False,
            ax,
            True,
            min_max[columns[0]] + min_max[columns[1]],
            **style_kwds,
        )
        return ax

    # ---#
    def hexbin(
        self,
        columns: list,
        method: str = "count",
        of: str = "",
        bbox: list = [],
        img: str = "",
        ax=None,
        **style_kwds,
    ):
        """
    ---------------------------------------------------------------------------
    Draws the Hexbin of the input vColumns based on an aggregation.

    Parameters
    ----------
    columns: list
        List of the vColumns names. The list must have two elements.
    method: str, optional
        The method to use to aggregate the data.
            count   : Number of elements.
            density : Percentage of the distribution.
            mean    : Average of the vColumn 'of'.
            min     : Minimum of the vColumn 'of'.
            max     : Maximum of the vColumn 'of'.
            sum     : Sum of the vColumn 'of'.
    of: str, optional
        The vColumn to use to compute the aggregation.
    bbox: list, optional
        List of 4 elements to delimit the boundaries of the final Plot. 
        It must be similar the following list: [xmin, xmax, ymin, ymax]
    img: str, optional
         Path to the image to display as background.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object

    See Also
    --------
    vDataFrame.pivot_table : Draws the pivot table of vColumns based on an aggregation.
        """
        if isinstance(method, str):
            method = method.lower()
        if isinstance(columns, str):
            columns = [columns]
        check_types(
            [
                ("columns", columns, [list],),
                ("method", method, ["density", "count", "avg", "min", "max", "sum"],),
                ("of", of, [str],),
                ("bbox", bbox, [list],),
                ("img", img, [str],),
            ]
        )
        columns_check(columns, self, [2])
        columns = vdf_columns_names(columns, self)
        if of:
            columns_check([of], self)
            of = vdf_columns_names([of], self)[0]
        from verticapy.plot import hexbin

        return hexbin(self, columns, method, of, bbox, img, ax=ax, **style_kwds,)

    # ---#
    def hist(
        self,
        columns: list,
        method: str = "density",
        of: str = "",
        max_cardinality: tuple = (6, 6),
        h: (int, float, tuple) = (None, None),
        hist_type: str = "auto",
        ax=None,
        **style_kwds,
    ):
        """
    ---------------------------------------------------------------------------
    Draws the histogram of the input vColumns based on an aggregation.

    Parameters
    ----------
    columns: list
        List of the vColumns names. The list must have less than 5 elements.
    method: str, optional
        The method to use to aggregate the data.
            count   : Number of elements.
            density : Percentage of the distribution.
            mean    : Average of the vColumn 'of'.
            min     : Minimum of the vColumn 'of'.
            max     : Maximum of the vColumn 'of'.
            sum     : Sum of the vColumn 'of'.
            q%      : q Quantile of the vColumn 'of' (ex: 50% to get the median).
        It can also be a cutomized aggregation (ex: AVG(column1) + 5).
    of: str, optional
        The vColumn to use to compute the aggregation.
    h: int/float/tuple, optional
        Interval width of the vColumns 1 and 2 bars. It is only valid if the 
        vColumns are numerical. Optimized h will be computed if the parameter 
        is empty or invalid.
    max_cardinality: tuple, optional
        Maximum number of distinct elements for vColumns 1 and 2 to be used as 
        categorical (No h will be picked or computed)
    hist_type: str, optional
        The Histogram Type.
            auto    : Regular Histogram based on 1 or 2 vColumns.
            multi   : Multiple Regular Histograms based on 1 to 5 vColumns.
            stacked : Stacked Histogram based on 2 vColumns.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object

    See Also
    --------
    vDataFrame.bar         : Draws the bar chart of the input vColumns based on an aggregation.
    vDataFrame.boxplot     : Draws the Box Plot of the input vColumns.
    vDataFrame.pivot_table : Draws the pivot table of vColumns based on an aggregation.
        """
        if isinstance(columns, str):
            columns = [columns]
        check_types(
            [
                ("columns", columns, [list],),
                ("method", method, [str],),
                ("of", of, [str],),
                ("max_cardinality", max_cardinality, [list],),
                ("h", h, [list, float, int],),
                ("hist_type", hist_type, ["auto", "multi", "stacked"],),
            ]
        )
        columns_check(columns, self, [1, 2, 3, 4, 5])
        columns = vdf_columns_names(columns, self)
        if of:
            columns_check([of], self)
            of = vdf_columns_names([of], self)[0]
        stacked = True if (hist_type.lower() == "stacked") else False
        multi = True if (hist_type.lower() == "multi") else False
        if len(columns) == 1:
            return self[columns[0]].hist(method, of, 6, 0, 0, **style_kwds,)
        else:
            if multi:
                from verticapy.plot import multiple_hist

                if isinstance(h, (int, float)):
                    h_0 = h
                else:
                    h_0 = h[0] if (h[0]) else 0
                return multiple_hist(
                    self, columns, method, of, h_0, ax=ax, **style_kwds,
                )
            else:
                from verticapy.plot import hist2D

                return hist2D(
                    self,
                    columns,
                    method,
                    of,
                    max_cardinality,
                    h,
                    stacked,
                    ax=ax,
                    **style_kwds,
                )

    # ---#
    def iloc(self, limit: int = 5, offset: int = 0, columns: list = []):
        """
    ---------------------------------------------------------------------------
    Returns a part of the vDataFrame (delimited by an offset and a limit).

    Parameters
    ----------
    limit: int, optional
        Number of elements to display.
    offset: int, optional
        Number of elements to skip.
    columns: list, optional
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
        if isinstance(columns, str):
            columns = [columns]
        check_types(
            [
                ("limit", limit, [int, float],),
                ("offset", offset, [int, float],),
                ("columns", columns, [list],),
            ]
        )
        if offset < 0:
            offset = max(0, self.shape()[0] - limit)
        columns = vdf_columns_names(columns, self)
        if not (columns):
            columns = self.get_columns()
        all_columns = []
        for column in columns:
            all_columns += [
                "{} AS {}".format(
                    convert_special_type(self[column].category(), True, column), column
                )
            ]
        title = "Reads the final relation using a limit of {} and an offset of {}.".format(
            limit, offset
        )
        result = to_tablesample(
            "SELECT {} FROM {}{} LIMIT {} OFFSET {}".format(
                ", ".join(all_columns),
                self.__genSQL__(),
                last_order_by(self),
                limit,
                offset,
            ),
            self._VERTICAPY_VARIABLES_["cursor"],
            title=title,
        )
        pre_comp = self.__get_catalog_value__("VERTICAPY_COUNT")
        if pre_comp != "VERTICAPY_NOT_PRECOMPUTED":
            result.count = pre_comp
        result.offset = offset
        result.name = self._VERTICAPY_VARIABLES_["input_relation"]
        columns = self.get_columns()
        all_percent = True
        for column in columns:
            if not ("percent" in self[column].catalog):
                all_percent = False
        all_percent = (all_percent or (verticapy.options["percent_bar"] == True)) and (
            verticapy.options["percent_bar"] != False
        )
        if all_percent:
            percent = self.aggregate(["percent"], columns).transpose().values
        for column in result.values:
            result.dtype[column] = self[column].ctype()
            if all_percent:
                result.percent[column] = percent[vdf_columns_names([column], self)[0]][
                    0
                ]
        return result

    # ---#
    def info(self):
        """
    ---------------------------------------------------------------------------
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

    # ---#
    def isin(self, val: dict):
        """
    ---------------------------------------------------------------------------
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
        check_types([("val", val, [dict],)])
        columns_check([elem for elem in val], self)
        n = len(val[list(val.keys())[0]])
        result = []
        for i in range(n):
            tmp_query = []
            for column in val:
                if val[column][i] == None:
                    tmp_query += [str_column(column) + " IS NULL"]
                else:
                    tmp_query += [
                        str_column(column)
                        + " = '{}'".format(str(val[column][i]).replace("'", "''"))
                    ]
            result += [" AND ".join(tmp_query)]
        return self.search(" OR ".join(result))

    # ---#
    def join(
        self,
        input_relation,
        on: dict = {},
        on_interpolate: dict = {},
        how: str = "natural",
        expr1: list = ["*"],
        expr2: list = ["*"],
    ):
        """
    ---------------------------------------------------------------------------
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
    on: dict, optional
        Dictionary of all different keys. The dict must be similar to the following:
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
    expr1: list, optional
        List of the different columns in pure SQL to select from the current 
        vDataFrame, optionally as aliases. Aliases are recommended to avoid 
        ambiguous names. For example: 'column' or 'column AS my_new_alias'. 
    expr2: list, optional
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
        check_types(
            [
                ("on", on, [dict],),
                (
                    "how",
                    how.lower(),
                    ["left", "right", "cross", "full", "natural", "self", "inner", ""],
                ),
                ("expr1", expr1, [list],),
                ("expr2", expr2, [list],),
                ("input_relation", input_relation, [vDataFrame, str],),
            ]
        )
        how = how.lower()
        columns_check([elem for elem in on], self)
        if isinstance(input_relation, vDataFrame):
            columns_check([on[elem] for elem in on], input_relation)
            vdf_cols = []
            for elem in on:
                vdf_cols += [on[elem]]
            columns_check(vdf_cols, input_relation)
            relation = input_relation.__genSQL__()
            if (
                ("SELECT" in relation.upper())
                and ("FROM" in relation.upper())
                and ("(" in relation)
                and (")" in relation)
            ):
                second_relation = "(SELECT * FROM {}) AS y".format(relation)
            else:
                second_relation = "{} AS y".format(relation)
        elif isinstance(input_relation, str):
            if (
                ("SELECT" in input_relation.upper())
                and ("FROM" in input_relation.upper())
                and ("(" in input_relation)
                and (")" in input_relation)
            ):
                second_relation = "(SELECT * FROM {}) AS y".format(input_relation)
            else:
                second_relation = "{} AS y".format(input_relation)
        on_join = " AND ".join(
            [
                'x."'
                + elem.replace('"', "")
                + '" = y."'
                + on[elem].replace('"', "")
                + '"'
                for elem in on
            ]
            + [
                'x."'
                + elem.replace('"', "")
                + '" INTERPOLATE PREVIOUS VALUE y."'
                + on_interpolate[elem].replace('"', "")
                + '"'
                for elem in on_interpolate
            ]
        )
        on_join = " ON {}".format(on_join) if (on_join) else ""
        relation = self.__genSQL__()
        if (
            ("SELECT" in relation.upper())
            and ("FROM" in relation.upper())
            and ("(" in relation)
            and (")" in relation)
        ):
            first_relation = "(SELECT * FROM {}) AS x".format(relation)
        else:
            first_relation = "{} AS x".format(relation)
        expr1, expr2 = (
            ["x.{}".format(elem) for elem in expr1],
            ["y.{}".format(elem) for elem in expr2],
        )
        expr = expr1 + expr2
        expr = "*" if not (expr) else ", ".join(expr)
        table = "SELECT {} FROM {} {} JOIN {} {}".format(
            expr, first_relation, how.upper(), second_relation, on_join
        )
        return self.__vdf_from_relation__(
            "({}) VERTICAPY_SUBTABLE".format(table),
            "join",
            "[Join]: Two relations were joined together",
        )

    # ---#
    def kurtosis(self, columns: list = []):
        """
    ---------------------------------------------------------------------------
    Aggregates the vDataFrame using 'kurtosis'.

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        return self.aggregate(func=["kurtosis"], columns=columns)

    kurt = kurtosis
    # ---#
    def last(
        self, ts: str, offset: str,
    ):
        """
    ---------------------------------------------------------------------------
    Filters the vDataFrame by only keeping the last records.

    Parameters
    ----------
    ts: str
        TS (Time Series) vColumn to use to filter the data. The vColumn type must be
        date like (date, datetime, timestamp...)
    offset: str
        Interval offset. For example, to filter and keep only the last 6 months of
        records, offset should be set to '6 months'.

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame.at_time      : Filters the data at the input time.
    vDataFrame.between_time : Filters the data between two time ranges.
    vDataFrame.first        : Filters the data by only keeping the first records.
    vDataFrame.filter       : Filters the data using the input expression.
        """
        check_types(
            [("ts", ts, [str],), ("offset", offset, [str],),]
        )
        ts = vdf_columns_names([ts], self)[0]
        query = "SELECT (MAX({}) - '{}'::interval)::varchar FROM {}".format(
            ts, offset, self.__genSQL__()
        )
        self.__executeSQL__(query, title="Gets the vDataFrame last values.")
        last_date = self._VERTICAPY_VARIABLES_["cursor"].fetchone()[0]
        self.filter("{} >= '{}'".format(ts, last_date),)
        return self

    # ---#
    def load(self, offset: int = -1):
        """
    ---------------------------------------------------------------------------
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
        check_types([("offset", offset, [int, float],)])
        save = self._VERTICAPY_VARIABLES_["saving"][offset]
        vdf = pickle.loads(save)
        vdf._VERTICAPY_VARIABLES_["cursor"] = self._VERTICAPY_VARIABLES_["cursor"]
        return vdf

    # ---#
    def mad(self, columns: list = []):
        """
    ---------------------------------------------------------------------------
    Aggregates the vDataFrame using 'mad' (Median Absolute Deviation).

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        return self.aggregate(func=["mad"], columns=columns)

    # ---#
    def max(self, columns: list = []):
        """
    ---------------------------------------------------------------------------
    Aggregates the vDataFrame using 'max' (Maximum).

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        return self.aggregate(func=["max"], columns=columns)

    # ---#
    def median(self, columns: list = []):
        """
    ---------------------------------------------------------------------------
    Aggregates the vDataFrame using 'median'.

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        return self.aggregate(func=["median"], columns=columns)

    # ---#
    def memory_usage(self):
        """
    ---------------------------------------------------------------------------
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
        import sys

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

    # ---#
    def min(self, columns: list = []):
        """
    ---------------------------------------------------------------------------
    Aggregates the vDataFrame using 'min' (Minimum).

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        return self.aggregate(func=["min"], columns=columns)

    # ---#
    def narrow(
        self,
        index: (str, list),
        columns: list = [],
        col_name: str = "column",
        val_name: str = "value",
    ):
        """
    ---------------------------------------------------------------------------
    Returns the Narrow Table of the vDataFrame using the input vColumns.

    Parameters
    ----------
    index: str/list
        Index(es) used to identify the Row.
    columns: list, optional
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
        if isinstance(columns, str):
            columns = [columns]
        check_types(
            [("index", index, [str, list],), ("columns", columns, [list],),]
        )
        if isinstance(index, str):
            index = vdf_columns_names([index], self)
        else:
            index = vdf_columns_names(index, self)
        columns = vdf_columns_names(columns, self)
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
            query += [
                "(SELECT {}, '{}' AS {}, {}{} AS {} FROM {})".format(
                    ", ".join(index),
                    column.replace("'", "''")[1:-1],
                    col_name,
                    column,
                    conv,
                    val_name,
                    self.__genSQL__(),
                )
            ]
        query = " UNION ALL ".join(query)
        query = "({}) VERTICAPY_SUBTABLE".format(query)
        return self.__vdf_from_relation__(
            query, "narrow", "[Narrow]: Narrow table using index = {}".format(index),
        )

    # ---#
    def normalize(self, columns: list = [], method: str = "zscore"):
        """
    ---------------------------------------------------------------------------
    Normalizes the input vColumns using the input method.

    Parameters
    ----------
    columns: list, optional
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
        if isinstance(method, str):
            method = method.lower()
        if isinstance(columns, str):
            columns = [columns]
        check_types(
            [
                ("columns", columns, [list],),
                ("method", method, ["zscore", "robust_zscore", "minmax"],),
            ]
        )
        columns_check(columns, self)
        no_cols = True if not (columns) else False
        columns = self.numcol() if not (columns) else vdf_columns_names(columns, self)
        for column in columns:
            if self[column].isnum() and not (self[column].isbool()):
                self[column].normalize(method=method)
            elif (no_cols) and (self[column].isbool()):
                pass
            elif verticapy.options["print_info"]:
                warning_message = "The vColumn {} was skipped.\nNormalize only accept numerical data types.".format(
                    column
                )
                warnings.warn(warning_message, Warning)
        return self

    # ---#
    def numcol(self, exclude_columns: list = []):
        """
    ---------------------------------------------------------------------------
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

    # ---#
    def nunique(self, columns: list = []):
        """
    ---------------------------------------------------------------------------
    Aggregates the vDataFrame using 'unique' (cardinality).

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all vColumns will be used.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        return self.aggregate(func=["unique"], columns=columns)

    # ---#
    def outliers(
        self,
        columns: list = [],
        name: str = "distribution_outliers",
        threshold: float = 3.0,
        robust: bool = False,
    ):
        """
    ---------------------------------------------------------------------------
    Adds a new vColumn labeled with 0 and 1. 1 means that the record is a global 
    outlier.

    Parameters
    ----------
    columns: list, optional
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
        check_types(
            [
                ("columns", columns, [list],),
                ("name", name, [str],),
                ("threshold", threshold, [int, float],),
            ]
        )
        columns_check(columns, self)
        columns = vdf_columns_names(columns, self) if (columns) else self.numcol()
        if not (robust):
            result = self.aggregate(func=["std", "avg"], columns=columns).values
        else:
            result = self.aggregate(func=["mad", "median"], columns=columns).values
        conditions = []
        for idx, elem in enumerate(result["index"]):
            if not (robust):
                conditions += [
                    "ABS({} - {}) / NULLIFZERO({}) > {}".format(
                        elem, result["avg"][idx], result["std"][idx], threshold
                    )
                ]
            else:
                conditions += [
                    "ABS({} - {}) / NULLIFZERO({} * 1.4826) > {}".format(
                        elem, result["median"][idx], result["mad"][idx], threshold
                    )
                ]
        self.eval(
            name, "(CASE WHEN {} THEN 1 ELSE 0 END)".format(" OR ".join(conditions))
        )
        return self

    # ---#
    def outliers_plot(
        self,
        columns: list,
        threshold: float = 3.0,
        color: str = "orange",
        outliers_color: str = "black",
        inliers_color: str = "white",
        inliers_border_color: str = "red",
        max_nb_points: int = 500,
        ax=None,
        **style_kwds,
    ):
        """
    ---------------------------------------------------------------------------
    Draws the global outliers plot of one or two columns based on their ZSCORE.

    Parameters
    ----------
    columns: list
        List of one or two vColumn names.
    threshold: float, optional
        ZSCORE threshold used to detect outliers.
    color: str, optional
        Inliers Area color.
    outliers_color: str, optional
        Outliers color.
    inliers_color: str, optional
        Inliers color.
    inliers_border_color: str, optional
        Inliers border color.
    max_nb_points: int, optional
        Maximum number of points to display.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax: Matplotlib axes object, optional
        The axes to plot on.
        """
        if isinstance(columns, str):
            columns = [columns]
        check_types(
            [
                ("columns", columns, [list],),
                ("color", color, [str],),
                ("threshold", threshold, [int, float,],),
                ("outliers_color", outliers_color, [str],),
                ("inliers_color", inliers_color, [str],),
                ("inliers_border_color", inliers_border_color, [str],),
                ("max_nb_points", max_nb_points, [int],),
            ]
        )
        columns_check(columns, self, [1, 2])
        columns = vdf_columns_names(columns, self)
        from verticapy.plot import outliers_contour_plot

        return outliers_contour_plot(
            self,
            columns,
            color=color,
            threshold=threshold,
            outliers_color=outliers_color,
            inliers_color=inliers_color,
            inliers_border_color=inliers_border_color,
            max_nb_points=max_nb_points,
            ax=ax,
            **style_kwds,
        )

    # ---#
    def pacf(
        self,
        column: str,
        ts: str,
        by: list = [],
        p: (int, list) = 5,
        unit: str = "rows",
        confidence: bool = True,
        alpha: float = 0.95,
        show: bool = True,
        ax=None,
        **style_kwds,
    ):
        """
    ---------------------------------------------------------------------------
    Computes the partial autocorrelations of the input vColumn.

    Parameters
    ----------
    column: str
        Input vColumn to use to compute the partial autocorrelation plot.
    ts: str
        TS (Time Series) vColumn to use to order the data. It can be of type date
        or a numerical vColumn.
    by: list, optional
        vColumns used in the partition.
    p: int/list, optional
        Int equals to the maximum number of lag to consider during the computation
        or List of the different lags to include during the computation.
        p must be positive or a list of positive integers.
    unit: str, optional
        Unit to use to compute the lags.
            rows: Natural lags
            else : Any time unit, for example you can write 'hour' to compute the hours
                lags or 'day' to compute the days lags.
    confidence: bool, optional
        If set to True, the confidence band width is drawn.
    alpha: float, optional
        Significance Level. Probability to accept H0. Only used to compute the confidence
        band width.
    show: bool, optional
        If set to True, the Partial Auto Correlation Plot will be drawn using Matplotlib.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.acf    : Computes the correlations between a vColumn and its lags.
    vDataFrame.asfreq : Interpolates and computes a regular time interval vDataFrame.
    vDataFrame.corr   : Computes the correlation matrix of a vDataFrame.
    vDataFrame.cov    : Computes the covariance matrix of the vDataFrame.
        """
        if isinstance(by, str):
            by = [by]
        check_types(
            [
                ("by", by, [list],),
                ("ts", ts, [str],),
                ("column", column, [str],),
                ("p", p, [int, float, list],),
                ("unit", unit, [str],),
                ("confidence", confidence, [bool],),
                ("alpha", alpha, [int, float],),
                ("show", show, [bool],),
            ]
        )
        if isinstance(p, Iterable) and (len(p) == 1):
            p = p[0]
            if p == 0:
                return 1.0
            elif p == 1:
                return self.acf(ts=ts, column=column, by=by, p=[1], unit=unit)
            columns_check([column, ts] + by, self)
            by = vdf_columns_names(by, self)
            column = vdf_columns_names([column], self)[0]
            ts = vdf_columns_names([ts], self)[0]
            if unit == "rows":
                table = self.__genSQL__()
            else:
                table = self.asfreq(
                    ts=ts, rule="1 {}".format(unit), method={column: "linear"}, by=by
                ).__genSQL__()
            by = "PARTITION BY {} ".format(", ".join(by)) if (by) else ""
            columns = [
                "LAG({}, {}) OVER ({}ORDER BY {}) AS lag_{}_{}".format(
                    column, i, by, ts, i, gen_name([column])
                )
                for i in range(1, p + 1)
            ]
            relation = "(SELECT {} FROM {}) pacf".format(
                ", ".join([column] + columns), table
            )
            schema = self._VERTICAPY_VARIABLES_["schema_writing"]
            if not (schema):
                schema = "public"

            def drop_temp_elem(self, schema):
                try:
                    with warnings.catch_warnings(record=True) as w:
                        drop(
                            "{}.VERTICAPY_TEMP_MODEL_LINEAR_REGRESSION_{}".format(
                                schema,
                                get_session(self._VERTICAPY_VARIABLES_["cursor"]),
                            ),
                            cursor=self._VERTICAPY_VARIABLES_["cursor"],
                            method="model",
                        )
                        drop(
                            "{}.VERTICAPY_TEMP_MODEL_LINEAR_REGRESSION2_{}".format(
                                schema,
                                get_session(self._VERTICAPY_VARIABLES_["cursor"]),
                            ),
                            cursor=self._VERTICAPY_VARIABLES_["cursor"],
                            method="model",
                        )
                        drop(
                            "{}.VERTICAPY_TEMP_MODEL_LINEAR_REGRESSION_VIEW_{}".format(
                                schema,
                                get_session(self._VERTICAPY_VARIABLES_["cursor"]),
                            ),
                            cursor=self._VERTICAPY_VARIABLES_["cursor"],
                            method="view",
                        )
                except:
                    pass

            try:
                drop_temp_elem(self, schema)
                query = "CREATE VIEW {}.VERTICAPY_TEMP_MODEL_LINEAR_REGRESSION_VIEW_{} AS SELECT * FROM {}".format(
                    schema, get_session(self._VERTICAPY_VARIABLES_["cursor"]), relation
                )
                self._VERTICAPY_VARIABLES_["cursor"].execute(query)
                vdf = vDataFrame(
                    "{}.VERTICAPY_TEMP_MODEL_LINEAR_REGRESSION_VIEW_{}".format(
                        schema, get_session(self._VERTICAPY_VARIABLES_["cursor"])
                    ),
                    self._VERTICAPY_VARIABLES_["cursor"],
                )

                from verticapy.learn.linear_model import LinearRegression

                model = LinearRegression(
                    name="{}.VERTICAPY_TEMP_MODEL_LINEAR_REGRESSION_{}".format(
                        schema, get_session(self._VERTICAPY_VARIABLES_["cursor"])
                    ),
                    cursor=self._VERTICAPY_VARIABLES_["cursor"],
                    solver="Newton",
                )
                model.fit(
                    input_relation="{}.VERTICAPY_TEMP_MODEL_LINEAR_REGRESSION_VIEW_{}".format(
                        schema, get_session(self._VERTICAPY_VARIABLES_["cursor"])
                    ),
                    X=["lag_{}_{}".format(i, gen_name([column])) for i in range(1, p)],
                    y=column,
                )
                model.predict(vdf, name="prediction_0")
                model = LinearRegression(
                    name="{}.VERTICAPY_TEMP_MODEL_LINEAR_REGRESSION2_{}".format(
                        schema, get_session(self._VERTICAPY_VARIABLES_["cursor"])
                    ),
                    cursor=self._VERTICAPY_VARIABLES_["cursor"],
                    solver="Newton",
                )
                model.fit(
                    input_relation="{}.VERTICAPY_TEMP_MODEL_LINEAR_REGRESSION_VIEW_{}".format(
                        schema, get_session(self._VERTICAPY_VARIABLES_["cursor"])
                    ),
                    X=["lag_{}_{}".format(i, gen_name([column])) for i in range(1, p)],
                    y="lag_{}_{}".format(p, gen_name([column])),
                )
                model.predict(vdf, name="prediction_p")
                vdf.eval(expr="{} - prediction_0".format(column), name="eps_0")
                vdf.eval(
                    expr="{} - prediction_p".format(
                        "lag_{}_{}".format(p, gen_name([column]))
                    ),
                    name="eps_p",
                )
                result = vdf.corr(["eps_0", "eps_p"])
                drop_temp_elem(self, schema)
            except:
                drop_temp_elem(self, schema)
                raise
            return result
        else:
            if isinstance(p, (float, int)):
                p = range(0, p + 1)
            if verticapy.options["tqdm"]:
                from tqdm.auto import tqdm

                loop = tqdm(p)
            else:
                loop = p
            pacf = []
            for i in loop:
                pacf += [self.pacf(ts=ts, column=column, by=by, p=[i], unit=unit)]
            columns = [elem for elem in p]
            pacf_band = []
            if confidence:
                from scipy.special import erfinv

                for k in range(1, len(pacf) + 1):
                    pacf_band += [
                        math.sqrt(2)
                        * erfinv(alpha)
                        / math.sqrt(self[column].count() - k + 1)
                        * math.sqrt((1 + 2 * sum([pacf[i] ** 2 for i in range(1, k)])))
                    ]
            result = tablesample({"index": columns, "value": pacf})
            if pacf_band:
                result.values["confidence"] = pacf_band
            if show:
                from verticapy.plot import acf_plot

                acf_plot(
                    result.values["index"],
                    result.values["value"],
                    title="Partial Autocorrelation",
                    confidence=pacf_band,
                    type_bar=True,
                    ax=ax,
                    **style_kwds,
                )
            return result

    # ---#
    def pie(
        self,
        columns: list,
        max_cardinality: (int, tuple) = None,
        h: (float, tuple) = None,
        ax=None,
        **style_kwds,
    ):
        """
    ---------------------------------------------------------------------------
    Draws the nested density pie chart of the input vColumns.

    Parameters
    ----------
    columns: list
        List of the vColumns names.
    max_cardinality: int/tuple, optional
        Maximum number of the vColumn distinct elements to be used as categorical 
        (No h will be picked or computed).
        If of type tuple, it must represent each column 'max_cardinality'.
    h: float/tuple, optional
        Interval width of the bar. If empty, an optimized h will be computed.
        If of type tuple, it must represent each column 'h'.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object

    See Also
    --------
    vDataFrame[].pie : Draws the Pie Chart of the vColumn based on an aggregation.
        """
        if isinstance(columns, str):
            columns = [columns]
        check_types(
            [
                ("max_cardinality", max_cardinality, [int, tuple, list],),
                ("h", h, [list, tuple, float],),
            ]
        )
        columns_check(columns, self)
        columns = vdf_columns_names(columns, self)
        from verticapy.plot import nested_pie

        return nested_pie(self, columns, max_cardinality, h, ax=None, **style_kwds,)

    # ---#
    def pivot(
        self,
        index: str,
        columns: str,
        values: str,
        aggr: str = "sum",
        prefix: str = "",
    ):
        """
    ---------------------------------------------------------------------------
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
        check_types(
            [
                ("index", index, [str],),
                ("columns", columns, [str],),
                ("values", values, [str],),
                ("aggr", aggr, [str],),
                ("prefix", prefix, [str],),
            ]
        )
        index = vdf_columns_names([index], self)[0]
        columns = vdf_columns_names([columns], self)[0]
        values = vdf_columns_names([values], self)[0]
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
                        "(CASE WHEN {} IS NULL THEN {} ELSE NULL END)".format(
                            columns, values
                        ),
                    )
                    + "AS '{}NULL'".format(prefix)
                ]
            else:
                new_cols_trans += [
                    aggr.replace(
                        "{}",
                        "(CASE WHEN {} = '{}' THEN {} ELSE NULL END)".format(
                            columns, elem, values
                        ),
                    )
                    + "AS '{}{}'".format(prefix, elem)
                ]
        relation = "(SELECT {}, {} FROM {} GROUP BY 1) VERTICAPY_SUBTABLE".format(
            index, ", ".join(new_cols_trans), self.__genSQL__()
        )
        return self.__vdf_from_relation__(
            relation,
            "pivot",
            "[Pivot]: Pivot table using index = {} & columns = {} & values = {}".format(
                index, columns, values
            ),
        )

    # ---#
    def pivot_table(
        self,
        columns: list,
        method: str = "count",
        of: str = "",
        max_cardinality: tuple = (20, 20),
        h: tuple = (None, None),
        show: bool = True,
        with_numbers: bool = True,
        ax=None,
        **style_kwds,
    ):
        """
    ---------------------------------------------------------------------------
    Draws the pivot table of one or two columns based on an aggregation.

    Parameters
    ----------
    columns: list
        List of the vColumns names. The list must have one or two elements.
    method: str, optional
        The method to use to aggregate the data.
            count   : Number of elements.
            density : Percentage of the distribution.
            mean    : Average of the vColumn 'of'.
            min     : Minimum of the vColumn 'of'.
            max     : Maximum of the vColumn 'of'.
            sum     : Sum of the vColumn 'of'.
            q%      : q Quantile of the vColumn 'of (ex: 50% to get the median).
        It can also be a cutomized aggregation (ex: AVG(column1) + 5).
    of: str, optional
        The vColumn to use to compute the aggregation.
    max_cardinality: tuple, optional
        Maximum number of distinct elements for vColumns 1 and 2 to be used as 
        categorical (No h will be picked or computed)
    h: tuple, optional
        Interval width of the vColumns 1 and 2 bars. It is only valid if the 
        vColumns are numerical. Optimized h will be computed if the parameter 
        is empty or invalid.
    show: bool, optional
        If set to True, the result will be drawn using Matplotlib.
    with_numbers: bool, optional
        If set to True, no number will be displayed in the final drawing.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.hexbin : Draws the Hexbin Plot of 2 vColumns based on an aggregation.
    vDataFrame.pivot  : Returns the Pivot of the vDataFrame using the input aggregation.
        """
        if isinstance(columns, str):
            columns = [columns]
        check_types(
            [
                ("columns", columns, [list],),
                ("method", method, [str],),
                ("of", of, [str],),
                ("max_cardinality", max_cardinality, [list],),
                ("h", h, [list],),
                ("show", show, [bool],),
                ("with_numbers", with_numbers, [bool],),
            ]
        )
        columns_check(columns, self, [1, 2])
        columns = vdf_columns_names(columns, self)
        if of:
            columns_check([of], self)
            of = vdf_columns_names([of], self)[0]
        from verticapy.plot import pivot_table

        return pivot_table(
            self,
            columns,
            method,
            of,
            h,
            max_cardinality,
            show,
            with_numbers,
            ax=ax,
            **style_kwds,
        )

    # ---#
    def plot(
        self,
        ts: str,
        columns: list = [],
        start_date: (str, datetime.datetime, datetime.date,) = "",
        end_date: (str, datetime.datetime, datetime.date,) = "",
        step: bool = False,
        ax=None,
        **style_kwds,
    ):
        """
    ---------------------------------------------------------------------------
    Draws the time series.

    Parameters
    ----------
    ts: str
        TS (Time Series) vColumn to use to order the data. The vColumn type must be
        date like (date, datetime, timestamp...) or numerical.
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.
    start_date: str / date, optional
        Input Start Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is lesser than November 1993 the 3rd.
    end_date: str / date, optional
        Input End Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is greater than November 1993 the 3rd.
    step: bool, optional
        If set to True, draw a Step Plot.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object

    See Also
    --------
    vDataFrame[].plot : Draws the Time Series of one vColumn.
        """
        if isinstance(columns, str):
            columns = [columns]
        check_types(
            [
                ("columns", columns, [list],),
                ("ts", ts, [str],),
                (
                    "start_date",
                    start_date,
                    [str, datetime.datetime, datetime.date, int, float,],
                ),
                (
                    "end_date",
                    end_date,
                    [str, datetime.datetime, datetime.date, int, float,],
                ),
            ]
        )
        columns_check(columns + [ts], self)
        columns = vdf_columns_names(columns, self)
        ts = vdf_columns_names([ts], self)[0]
        kind = "step" if step else "line"
        from verticapy.plot import multi_ts_plot

        return multi_ts_plot(
            self, ts, columns, start_date, end_date, kind, ax=ax, **style_kwds,
        )

    # ---#
    def polynomial_comb(
        self, columns: list = [], r: int = 2,
    ):
        """
    ---------------------------------------------------------------------------
    Returns a vDataFrame containing different product combination of the 
    input vColumns. This function is ideal for bivariate analysis.

    Parameters
    ----------
    columns: list, optional
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
        check_types(
            [("columns", columns, [list],), ("r", r, [int],),]
        )
        columns_check(columns, self)
        if not (columns):
            numcol = self.numcol()
        else:
            numcol = vdf_columns_names(columns, self)
        vdf = self.copy()
        all_comb = combinations_with_replacement(numcol, r=r)
        for elem in all_comb:
            name = "_".join(elem)
            vdf.eval(name.replace('"', ""), expr=" * ".join(elem))
        return vdf

    # ---#
    def product(self, columns: list = []):
        """
    ---------------------------------------------------------------------------
    Aggregates the vDataFrame using 'product'.

    Parameters
    ----------
    columns: list, optional
        List of the vColumn names. If empty, all numerical vColumns will be used.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        return self.aggregate(func=["prod"], columns=columns)

    prod = product

    # ---#
    def quantile(self, q: list, columns: list = []):
        """
    ---------------------------------------------------------------------------
    Aggregates the vDataFrame using a list of 'quantiles'.

    Parameters
    ----------
    q: list
        List of the different quantiles. They must be numbers between 0 and 1.
        For example [0.25, 0.75] will return Q1 and Q3.
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        if isinstance(q, str):
            q = [q]
        return self.aggregate(
            func=["{}%".format(float(item) * 100) for item in q], columns=columns
        )

    # ---#
    def recommend(
        self,
        unique_id: str,
        item_id: str,
        method: str = "count",
        rating: (str, tuple) = "",
        ts: str = "",
        start_date: (str, datetime.datetime, datetime.date,) = "",
        end_date: (str, datetime.datetime, datetime.date,) = "",
    ):
        """
    ---------------------------------------------------------------------------
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
    start_date: str / date, optional
        Input Start Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is lesser than November 1993 the 3rd.
    end_date: str / date, optional
        Input End Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is greater than November 1993 the 3rd.

    Returns
    -------
    vDataFrame
        The vDataFrame of the recommendation.
        """
        if isinstance(method, str):
            method = method.lower()
        check_types(
            [
                ("unique_id", unique_id, [str],),
                ("item_id", item_id, [str],),
                ("method", method, ["count", "avg", "median",],),
                ("rating", rating, [str, list, tuple],),
                ("ts", ts, [str],),
                (
                    "start_date",
                    start_date,
                    [str, datetime.datetime, datetime.date, int, float,],
                ),
                (
                    "end_date",
                    end_date,
                    [str, datetime.datetime, datetime.date, int, float,],
                ),
            ]
        )
        columns_check([unique_id, item_id], self)
        unique_id, item_id = vdf_columns_names([unique_id, item_id], self)
        vdf = self.copy()
        assert method == "count" or rating, f"Method '{method}' can not be used if parameter 'rating' is empty."
        if rating:
            assert isinstance(rating, str) or len(rating) == 3, ParameterError(f"Parameter 'rating' must be of type str or composed of exactly 3 elements: (r_vdf, r_item_id, r_name).")
            assert method != "count", "Method 'count' can not be used if parameter 'rating' is defined."
            columns_check([rating], self)
            rating = vdf_columns_names([rating], self)[0]
        if ts:
            columns_check([ts], self)
            ts = vdf_columns_names([ts], self)[0]
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

    # ---#
    def regexp(
        self,
        column: str,
        pattern: str,
        method: str = "substr",
        position: int = 1,
        occurrence: int = 1,
        replacement: str = "",
        return_position: int = 0,
        name: str = "",
    ):
        """
    ---------------------------------------------------------------------------
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
        check_types(
            [
                ("column", column, [str],),
                ("pattern", pattern, [str],),
                (
                    "method",
                    method,
                    [
                        "count",
                        "ilike",
                        "instr",
                        "like",
                        "not_ilike",
                        "not_like",
                        "replace",
                        "substr",
                    ],
                ),
                ("position", position, [int],),
                ("occurrence", occurrence, [int],),
                ("replacement", replacement, [str],),
                ("return_position", return_position, [int],),
            ]
        )
        columns_check([column], self)
        column = vdf_columns_names([column], self)[0]
        expr = "REGEXP_{}({}, '{}'".format(
            method.upper(), column, pattern.replace("'", "''")
        )
        if method in ("replace"):
            expr += ", '{}'".format(replacement.replace("'", "''"))
        if method in ("count", "instr", "replace", "substr"):
            expr += ", {}".format(position)
        if method in ("instr", "replace", "substr"):
            expr += ", {}".format(occurrence)
        if method in ("instr"):
            expr += ", {}".format(return_position)
        expr += ")"
        gen_name([method, column])
        return self.eval(name=name, expr=expr)

    # ---#
    def regr(
        self,
        columns: list = [],
        method: str = "r2",
        show: bool = True,
        ax=None,
        **style_kwds,
    ):
        """
    ---------------------------------------------------------------------------
    Computes the regression matrix of the vDataFrame.

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.
    method: str, optional
        Method to use to compute the regression matrix.
            avgx  : Average of the independent expression in an expression pair.
            avgy  : Average of the dependent expression in an expression pair.
            count : Count of all rows in an expression pair.
            alpha : Intercept of the regression line determined by a set of 
                expression pairs.
            r2    : Square of the correlation coefficient of a set of expression 
                pairs.
            beta  : Slope of the regression line, determined by a set of expression 
                pairs.
            sxx   : Sum of squares of the independent expression in an expression 
                pair.
            sxy   : Sum of products of the independent expression multiplied by the 
                dependent expression in an expression pair.
            syy   : Returns the sum of squares of the dependent expression in an 
                expression pair.
    show: bool, optional
        If set to True, the Regression Matrix will be drawn using Matplotlib.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.acf   : Computes the correlations between a vColumn and its lags.
    vDataFrame.cov   : Computes the covariance matrix of the vDataFrame.
    vDataFrame.corr  : Computes the Correlation Matrix of the vDataFrame.
    vDataFrame.pacf  : Computes the partial autocorrelations of the input vColumn.
        """
        if isinstance(columns, str):
            columns = [columns]
        check_types(
            [
                ("columns", columns, [list],),
                (
                    "method",
                    method,
                    [
                        "avgx",
                        "avgy",
                        "count",
                        "intercept",
                        "r2",
                        "slope",
                        "sxx",
                        "sxy",
                        "syy",
                        "beta",
                        "alpha",
                    ],
                ),
                ("show", show, [bool],),
            ]
        )
        if method == "beta":
            method = "slope"
        elif method == "alpha":
            method = "intercept"
        method = "regr_{}".format(method)
        if not (columns):
            columns = self.numcol()
            assert columns, EmptyParameter("No numerical column found in the vDataFrame.")
        columns_check(columns, self)
        columns = vdf_columns_names(columns, self)
        columns = vdf_columns_names(columns, self)
        for column in columns:
            assert self[column].isnum(), TypeError(f"vColumn {column} must be numerical to compute the Regression Matrix.")
        n = len(columns)
        all_list, nb_precomputed = [], 0
        for i in range(0, n):
            for j in range(0, n):
                cast_i = "::int" if (self[columns[i]].isbool()) else ""
                cast_j = "::int" if (self[columns[j]].isbool()) else ""
                pre_comp_val = self.__get_catalog_value__(
                    method=method, columns=[columns[i], columns[j]]
                )
                if pre_comp_val == None or pre_comp_val != pre_comp_val:
                    pre_comp_val = "NULL"
                if pre_comp_val != "VERTICAPY_NOT_PRECOMPUTED":
                    all_list += [str(pre_comp_val)]
                    nb_precomputed += 1
                else:
                    all_list += [
                        "{}({}{}, {}{})".format(
                            method.upper(), columns[i], cast_i, columns[j], cast_j
                        )
                    ]
        try:
            if nb_precomputed == n * n:
                self._VERTICAPY_VARIABLES_["cursor"].execute(
                    "SELECT {}".format(", ".join(all_list))
                )
            else:
                self.__executeSQL__(
                    query="SELECT {} FROM {}".format(
                        ", ".join(all_list), self.__genSQL__()
                    ),
                    title="Computes the {} Matrix.".format(method.upper()),
                )
            result = self._VERTICAPY_VARIABLES_["cursor"].fetchone()
            if n == 1:
                return result[0]
        except:
            n = len(columns)
            result = []
            for i in range(0, n):
                for j in range(0, n):
                    self.__executeSQL__(
                        query="SELECT {}({}{}, {}{}) FROM {}".format(
                            method.upper(),
                            columns[i],
                            cast_i,
                            columns[j],
                            cast_j,
                            self.__genSQL__(),
                        ),
                        title="Computes the {} aggregation, one at a time.".format(
                            method.upper()
                        ),
                    )
                    result += [self._VERTICAPY_VARIABLES_["cursor"].fetchone()[0]]
        matrix = [[1 for i in range(0, n + 1)] for i in range(0, n + 1)]
        matrix[0] = [""] + columns
        for i in range(0, n + 1):
            matrix[i][0] = columns[i - 1]
        k = 0
        for i in range(0, n):
            for j in range(0, n):
                current = result[k]
                k += 1
                if current == None:
                    current = float("nan")
                matrix[i + 1][j + 1] = current
        if show:
            from verticapy.plot import cmatrix

            if method == "slope":
                method_title = "Beta"
            elif method == "intercept":
                method_title = "Alpha"
            else:
                method_title = method
            cmatrix(
                matrix,
                columns,
                columns,
                n,
                n,
                vmax=None,
                vmin=None,
                title="{} Matrix".format(method_title),
                ax=ax,
                **style_kwds,
            )
        values = {"index": matrix[0][1 : len(matrix[0])]}
        del matrix[0]
        for column in matrix:
            values[column[0]] = column[1 : len(column)]
        for elem in values:
            if elem != "index":
                for idx in range(len(values[elem])):
                    if isinstance(values[elem][idx], decimal.Decimal):
                        values[elem][idx] = float(values[elem][idx])
        for column1 in values:
            if column1 != "index":
                val = {}
                for idx, column2 in enumerate(values["index"]):
                    val[column2] = values[column1][idx]
                self.__update_catalog__(values=val, matrix=method, column=column1)
        return tablesample(values=values)

    # ---#
    def rolling(
        self,
        func: str,
        window: (list, tuple),
        columns: (str, list),
        by: list = [],
        order_by: (dict, list) = [],
        name: str = "",
    ):
        """
    ---------------------------------------------------------------------------
    Adds a new vColumn to the vDataFrame by using an advanced analytical window 
    function on one or two specific vColumns.

    \u26A0 Warning : Some window functions can make the vDataFrame structure 
                     heavier. It is recommended to always check the current structure 
                     using the 'current_relation' method and to save it using the 
                     'to_db' method with the parameters 'inplace = True' and 
                     'relation_type = table'

    Parameters
    ----------
    func: str
        Function to use.
            aad         : average absolute deviation
            beta        : Beta Coefficient between 2 vColumns
            count       : number of non-missing elements
            corr        : Pearson correlation between 2 vColumns
            cov         : covariance between 2 vColumns
            kurtosis    : kurtosis
            jb          : Jarque-Bera index
            max         : maximum
            mean        : average
            min         : minimum
            prod        : product
            range       : difference between the max and the min
            sem         : standard error of the mean
            skewness    : skewness
            sum         : sum
            std         : standard deviation
            var         : variance
                Other window functions could work if it is part of 
                the DB version you are using.
    window: list / tuple
        Window Frame Range.
        If two integers, it will compute a Row Window, otherwise it will compute
        a Time Window. For example, if set to (-5, 1), the moving windows will
        take 5 rows preceding and one following. If set to ('- 5 minutes', '0 minutes'),
        the moving window will take all elements of the last 5 minutes.
    columns: list
        Input vColumns. It can be a list of one or two elements.
    by: list, optional
        vColumns used in the partition.
    order_by: dict / list, optional
        List of the vColumns to use to sort the data using asc order or
        dictionary of all sorting methods. For example, to sort by "column1"
        ASC and "column2" DESC, write {"column1": "asc", "column2": "desc"}
    name: str, optional
        Name of the new vColumn. If empty, a default name will be generated.

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame.eval     : Evaluates a customized expression.
    vDataFrame.analytic : Adds a new vColumn to the vDataFrame by using an advanced 
        analytical function on a specific vColumn.
        """
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(by, str):
            by = [by]
        if isinstance(order_by, str):
            order_by = [order_by]
        check_types(
            [
                ("func", func, [str],),
                ("columns", columns, [list, str],),
                ("window", window, [list, tuple],),
                ("by", by, [list],),
                ("order_by", order_by, [list, dict],),
                ("name", name, [str],),
            ]
        )
        assert len(window) == 2, ParameterError(
            "The window must be composed of exactly 2 elements."
        )
        window = list(window)
        rule = [0, 0]
        unbounded, method = False, "rows"
        for idx, w in enumerate(window):
            if isinstance(w, (int, float)) and abs(w) == float("inf"):
                w = "unbounded"
            if isinstance(w, (str)):
                if w.lower() == "unbounded":
                    rule[idx] = "PRECEDING" if idx == 0 else "FOLLOWING"
                    window[idx] = "UNBOUNDED"
                else:
                    nb_min = 0
                    for i, char in enumerate(window[idx]):
                        if char == "-":
                            nb_min += 1
                        elif char != " ":
                            break
                    rule[idx] = "PRECEDING" if nb_min % 2 == 1 else "FOLLOWING"
                    window[idx] = "'" + window[idx][i:] + "'"
                    method = "range"
            elif isinstance(w, (datetime.timedelta)):
                rule[idx] = (
                    "PRECEDING" if window[idx] < datetime.timedelta(0) else "FOLLOWING"
                )
                window[idx] = "'" + str(abs(window[idx])) + "'"
                method = "range"
            else:
                rule[idx] = "PRECEDING" if int(window[idx]) < 0 else "FOLLOWING"
                window[idx] = abs(int(window[idx]))
        if isinstance(columns, str):
            columns = [columns]
        columns_check(columns + by + [elem for elem in order_by], self)
        if not (name):
            name = "moving_{}".format(
                gen_name([func] + columns + [window[0], rule[0], window[1], rule[1],]),
            )
        columns = vdf_columns_names(columns, self)
        by = (
            "" if not (by) else "PARTITION BY " + ", ".join(vdf_columns_names(by, self))
        )
        order_by = (
            " ORDER BY {}".format(columns[0])
            if not (order_by)
            else sort_str(order_by, self)
        )
        func = str_function(func.lower(), method="vertica")
        windows_frame = " OVER ({}{} {} BETWEEN {} AND {})".format(
            by,
            order_by,
            method.upper(),
            "{} {}".format(window[0], rule[0]),
            "{} {}".format(window[1], rule[1]),
        )
        all_cols = [
            elem.replace('"', "").lower()
            for elem in self._VERTICAPY_VARIABLES_["columns"]
        ]
        if func in ("kurtosis", "skewness", "aad", "prod", "jb"):
            if func in ("skewness", "kurtosis", "aad", "jb"):
                mean_name = "{}_mean_{}".format(
                    columns[0].replace('"', ""), random.randint(0, 10000000)
                ).lower()
                std_name = "{}_std_{}".format(
                    columns[0].replace('"', ""), random.randint(0, 10000000)
                ).lower()
                count_name = "{}_count_{}".format(
                    columns[0].replace('"', ""), random.randint(0, 10000000)
                ).lower()
                self.eval(mean_name, "AVG({}){}".format(columns[0], windows_frame))
                if func != "aad":
                    self.eval(
                        std_name, "STDDEV({}){}".format(columns[0], windows_frame)
                    )
                    self.eval(
                        count_name, "COUNT({}){}".format(columns[0], windows_frame)
                    )
                if func == "kurtosis":
                    expr = "AVG(POWER(({} - {}) / NULLIFZERO({}), 4))# * POWER({}, 2) * ({} + 1) / NULLIFZERO(({} - 1) * ({} - 2) * ({} - 3)) - 3 * POWER({} - 1, 2) / NULLIFZERO(({} - 2) * ({} - 3))".format(
                        columns[0],
                        mean_name,
                        std_name,
                        count_name,
                        count_name,
                        count_name,
                        count_name,
                        count_name,
                        count_name,
                        count_name,
                        count_name,
                    )
                elif func == "skewness":
                    expr = "AVG(POWER(({} - {}) / NULLIFZERO({}), 3))# * POWER({}, 2) / NULLIFZERO(({} - 1) * ({} - 2))".format(
                        columns[0],
                        mean_name,
                        std_name,
                        count_name,
                        count_name,
                        count_name,
                    )
                elif func == "jb":
                    expr = "{} / 6 * (POWER(AVG(POWER(({} - {}) / NULLIFZERO({}), 3))# * POWER({}, 2) / NULLIFZERO(({} - 1) * ({} - 2)), 2) + POWER(AVG(POWER(({} - {}) / NULLIFZERO({}), 4))# * POWER({}, 2) * ({} + 1) / NULLIFZERO(({} - 1) * ({} - 2) * ({} - 3)) - 3 * POWER({} - 1, 2) / NULLIFZERO(({} - 2) * ({} - 3)), 2) / 4)".format(
                        count_name,
                        columns[0],
                        mean_name,
                        std_name,
                        count_name,
                        count_name,
                        count_name,
                        columns[0],
                        mean_name,
                        std_name,
                        count_name,
                        count_name,
                        count_name,
                        count_name,
                        count_name,
                        count_name,
                        count_name,
                        count_name,
                    )
                elif func == "aad":
                    expr = "AVG(ABS({} - {}))#".format(columns[0], mean_name)
            else:
                expr = "DECODE(ABS(MOD(SUM(CASE WHEN {} < 0 THEN 1 ELSE 0 END)#, 2)), 0, 1, -1) * POWER(10, SUM(LOG(ABS({})))#)".format(
                    columns[0], columns[0]
                )
        elif func in ("corr", "cov", "beta"):
            if columns[1] == columns[0]:
                if func == "cov":
                    expr = "VARIANCE({})#".format(columns[0])
                else:
                    expr = "1"
            else:
                if func == "corr":
                    den = " / (STDDEV({})# * STDDEV({})#)".format(
                        columns[0], columns[1]
                    )
                elif func == "beta":
                    den = " / (VARIANCE({})#)".format(columns[1])
                else:
                    den = ""
                expr = "(AVG({} * {})# - AVG({})# * AVG({})#) {}".format(
                    columns[0], columns[1], columns[0], columns[1], den
                )
        elif func == "range":
            expr = "MAX({})# - MIN({})#".format(columns[0], columns[0])
        elif func == "sem":
            expr = "STDDEV({})# / SQRT(COUNT({})#)".format(columns[0], columns[0])
        else:
            expr = "{}({})#".format(func.upper(), columns[0])
        expr = expr.replace("#", windows_frame)
        self.eval(name=name, expr=expr)
        if func in ("kurtosis", "skewness", "jb"):
            self._VERTICAPY_VARIABLES_["exclude_columns"] += [
                str_column(mean_name),
                str_column(std_name),
                str_column(count_name),
            ]
        elif func in ("aad"):
            self._VERTICAPY_VARIABLES_["exclude_columns"] += [str_column(mean_name)]
        return self

    # ---#
    def sample(
        self, n: int = None, x: float = None, method: str = "random", by: list = [],
    ):
        """
    ---------------------------------------------------------------------------
    Downsamples the input vDataFrame.

    \u26A0 Warning : The result may be inconsistent between attempts at SQL
                     code generation if the data is not ordered.

    Parameters
     ----------
     n: int, optional
        Approximate number of element to consider in the sample.
     x: float, optional
        The sample size. For example it has to be equal to 0.33 to downsample to 
        approximatively 33% of the relation.
    method: str, optional
        The Sample method.
            random     : random sampling.
            systematic : systematic sampling.
            stratified : stratified sampling.
    by: list, optional
        vColumns used in the partition.

    Returns
    -------
    vDataFrame
        sample vDataFrame
        """
        if x == 1:
            return self.copy()
        assert n != None or x != None, ParameterError(
            "One of the parameter 'n' or 'x' must not be empty."
        )
        assert n == None or x == None, ParameterError(
            "One of the parameter 'n' or 'x' must be empty."
        )
        if n != None:
            check_types(
                [("n", n, [int, float,],),]
            )
            x = float(n / self.shape()[0])
            if x >= 1:
                return self.copy()
        if isinstance(method, str):
            method = method.lower()
        if method in ("systematic", "random"):
            order_by = ""
            assert not(by), ParameterError(f"Parameter 'by' must be empty when using '{method}' sampling.")
        check_types(
            [
                ("method", method, ["random", "systematic", "stratified",],),
                ("x", x, [int, float],),
            ]
        )
        if isinstance(by, str):
            by = [by]
        columns_check(by, self)
        by = vdf_columns_names(by, self)
        name = "__verticapy_random_{}__".format(random.randint(0, 10000000))
        vdf = self.copy()
        assert (0 < x < 1), ParameterError("Parameter 'x' must be between 0 and 1")
        if method == "random":
            random_state = verticapy.options["random_state"]
            random_seed = (
                random_state
                if isinstance(random_state, int)
                else random.randint(-10e6, 10e6)
            )
            random_func = "SEEDED_RANDOM({})".format(random_seed)
            vdf.eval(name, random_func)
            print_info_init = verticapy.options["print_info"]
            verticapy.options["print_info"] = False
            vdf.filter("{} < {}".format(name, x),)
            verticapy.options["print_info"] = print_info_init
            vdf._VERTICAPY_VARIABLES_["exclude_columns"] += [name]
        elif method in ("stratified", "systematic"):
            assert method != "stratified" or (by), ParameterError("Parameter 'by' must include at least one column when using 'stratified' sampling.")
            if method == "stratified":
                order_by = "ORDER BY " + ", ".join(by)
            vdf.eval(name, "ROW_NUMBER() OVER({})".format(order_by))
            print_info_init = verticapy.options["print_info"]
            verticapy.options["print_info"] = False
            vdf.filter("MOD({}, {}) = 0".format(name, int(1 / x)))
            verticapy.options["print_info"] = print_info_init
            vdf._VERTICAPY_VARIABLES_["exclude_columns"] += [name]
        return vdf

    # ---#
    def save(self):
        """
    ---------------------------------------------------------------------------
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
        vdf._VERTICAPY_VARIABLES_["cursor"] = None
        self._VERTICAPY_VARIABLES_["saving"] += [pickle.dumps(vdf)]
        return self

    # ---#
    def scatter(
        self,
        columns: list,
        catcol: str = "",
        max_cardinality: int = 6,
        cat_priority: list = [],
        with_others: bool = True,
        max_nb_points: int = 20000,
        bbox: list = [],
        img: str = "",
        ax=None,
        **style_kwds,
    ):
        """
    ---------------------------------------------------------------------------
    Draws the scatter plot of the input vColumns.

    Parameters
    ----------
    columns: list
        List of the vColumns names. The list must have two or three elements.
    catcol: str, optional
        Categorical vColumn to use to label the data.
    max_cardinality: int, optional
        Maximum number of distinct elements for 'catcol' to be used as 
        categorical. The less frequent elements will be gathered together to 
        create a new category: 'Others'.
    cat_priority: list, optional
        List of the different categories to consider when labeling the data using
        the vColumn 'catcol'. The other categories will be filtered.
    with_others: bool, optional
        If set to false and the cardinality of the vColumn 'catcol' is too big then
        the less frequent element will not be merged to another category and they 
        will not be drawn.
    max_nb_points: int, optional
        Maximum number of points to display.
    bbox: list, optional
        List of 4 elements to delimit the boundaries of the final Plot. 
        It must be similar the following list: [xmin, xmax, ymin, ymax]
    img: str, optional
        Path to the image to display as background.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object

    See Also
    --------
    vDataFrame.bubble      : Draws the bubble plot of the input vColumns.
    vDataFrame.pivot_table : Draws the pivot table of vColumns based on an aggregation.
        """
        if isinstance(columns, str):
            columns = [columns]
        check_types(
            [
                ("columns", columns, [list],),
                ("catcol", catcol, [str],),
                ("max_cardinality", max_cardinality, [int, float],),
                ("cat_priority", cat_priority, [list],),
                ("with_others", with_others, [bool],),
                ("max_nb_points", max_nb_points, [int, float],),
                ("img", img, [str],),
                ("bbox", bbox, [list],),
            ]
        )
        columns_check(columns, self, [2, 3])
        columns = vdf_columns_names(columns, self)
        if catcol:
            columns_check([catcol], self)
            catcol = vdf_columns_names([catcol], self)
        else:
            catcol = []
        if len(columns) == 2:
            from verticapy.plot import scatter2D

            return scatter2D(
                self,
                columns + catcol,
                max_cardinality,
                cat_priority,
                with_others,
                max_nb_points,
                bbox,
                img,
                ax=ax,
                **style_kwds,
            )
        elif len(columns) == 3:
            from verticapy.plot import scatter3D

            return scatter3D(
                self,
                columns + catcol,
                max_cardinality,
                cat_priority,
                with_others,
                max_nb_points,
                ax=ax,
                **style_kwds,
            )

    # ---#
    def scatter_matrix(
        self, columns: list = [], **style_kwds,
    ):
        """
    ---------------------------------------------------------------------------
    Draws the scatter matrix of the vDataFrame.

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object

    See Also
    --------
    vDataFrame.scatter : Draws the scatter plot of the input vColumns.
        """
        if isinstance(columns, str):
            columns = [columns]
        check_types([("columns", columns, [list],)])
        columns_check(columns, self)
        columns = vdf_columns_names(columns, self)
        from verticapy.plot import scatter_matrix

        return scatter_matrix(self, columns, **style_kwds,)

    # ---#
    def search(
        self,
        conditions: (str, list) = "",
        usecols: list = [],
        expr: list = [],
        order_by: (dict, list) = [],
    ):
        """
    ---------------------------------------------------------------------------
    Searches the elements which matches with the input conditions.
    
    Parameters
    ----------
    conditions: str / list, optional
        Filters of the search. It can be a list of conditions or an expression.
    usecols: list, optional
        vColumns to select from the final vDataFrame relation. If empty, all
        vColumns will be selected.
    expr: list, optional
        List of customized expressions in pure SQL.
        For example: 'column1 * column2 AS my_name'.
    order_by: dict / list, optional
        List of the vColumns to use to sort the data using asc order or
        dictionary of all sorting methods. For example, to sort by "column1"
        ASC and "column2" DESC, write {"column1": "asc", "column2": "desc"}

    Returns
    -------
    vDataFrame
        vDataFrame of the search

    See Also
    --------
    vDataFrame.filter : Filters the vDataFrame using the input expressions.
    vDataFrame.select : Returns a copy of the vDataFrame with only the selected vColumns.
        """
        if isinstance(order_by, str):
            order_by = [order_by]
        if isinstance(usecols, str):
            usecols = [usecols]
        if isinstance(expr, str):
            expr = [expr]
        check_types(
            [
                ("conditions", conditions, [str, list],),
                ("usecols", usecols, [list],),
                ("expr", expr, [list],),
                ("order_by", order_by, [dict, list],),
            ]
        )
        if isinstance(conditions, Iterable) and not (isinstance(conditions, str)):
            conditions = " AND ".join(["({})".format(elem) for elem in conditions])
        conditions = " WHERE {}".format(conditions) if conditions else ""
        all_cols = ", ".join(["*"] + expr)
        table = "(SELECT {} FROM {}{}) VERTICAPY_SUBTABLE".format(
            all_cols, self.__genSQL__(), conditions
        )
        result = self.__vdf_from_relation__(table, "search", "")
        if usecols:
            result = result.select(usecols)
        return result.sort(order_by)

    # ---#
    def select(
        self, columns: list,
    ):
        """
    ---------------------------------------------------------------------------
    Returns a copy of the vDataFrame with only the selected vColumns.

    Parameters
    ----------
    columns: list
        List of the vColumns to select. It can also be customized expressions.

    Returns
    -------
    vDataFrame
        object with only the selected columns.

    See Also
    --------
    vDataFrame.search : Searches the elements which matches with the input conditions.
        """
        if isinstance(columns, str):
            columns = [columns]
        check_types([("columns", columns, [list],)])
        for i in range(len(columns)):
            column = vdf_columns_names([columns[i]], self)
            if column:
                columns[i] = column[0]
            else:
                columns[i] = str(columns[i])
        table = "(SELECT {} FROM {}) VERTICAPY_SUBTABLE".format(
            ", ".join(columns), self.__genSQL__()
        )
        return self.__vdf_from_relation__(
            table, self._VERTICAPY_VARIABLES_["input_relation"], ""
        )

    # ---#
    def sem(self, columns: list = []):
        """
    ---------------------------------------------------------------------------
    Aggregates the vDataFrame using 'sem' (Standard Error of the Mean).

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        return self.aggregate(func=["sem"], columns=columns)

    # ---#
    def sessionize(
        self,
        ts: str,
        by: list = [],
        session_threshold: str = "30 minutes",
        name: str = "session_id",
    ):
        """
    ---------------------------------------------------------------------------
    Adds a new vColumn to the vDataFrame which will correspond to sessions 
    (user activity during a specific time). A session ends when ts - lag(ts) 
    is greater than a specific threshold.

    Parameters
    ----------
    ts: str
        vColumn used as timeline. It will be to use to order the data. It can be
        a numerical or type date like (date, datetime, timestamp...) vColumn.
    by: list, optional
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
        check_types(
            [
                ("ts", ts, [str],),
                ("by", by, [list],),
                ("session_threshold", session_threshold, [str],),
                ("name", name, [str],),
            ]
        )
        columns_check(by + [ts], self)
        by = vdf_columns_names(by, self)
        ts = vdf_columns_names([ts], self)[0]
        partition = "PARTITION BY {}".format(", ".join(by)) if (by) else ""
        expr = "CONDITIONAL_TRUE_EVENT({} - LAG({}) > '{}') OVER ({} ORDER BY {})".format(
            ts, ts, session_threshold, partition, ts
        )
        return self.eval(name=name, expr=expr)

    # ---#
    def set_cursor(self, cursor):
        """
    ---------------------------------------------------------------------------
    Sets a new database cursor. This can be very useful if the connection to
    the database is lost.

    Parameters
    ----------
    cursor: DBcursor
        New cursor.

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame.set_dsn: Sets a new DSN.
        """
        check_cursor(cursor)
        cursor.execute("SELECT 1;")
        self._VERTICAPY_VARIABLES_["cursor"] = cursor
        return self

    # ---#
    def set_schema_writing(self, schema_writing: str):
        """
    ---------------------------------------------------------------------------
    Sets a new writing schema used to create temporary tables when necessary.


    Parameters
    ----------
    schema_writing: str
        New schema writing name.

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame.set_cursor : Sets a new database cursor.
    vDataFrame.set_dsn    : Sets a new database DSN.
        """
        check_types([("schema_writing", schema_writing, [str],)])
        self._VERTICAPY_VARIABLES_["cursor"].execute(
            "SELECT table_schema FROM columns WHERE table_schema = '{}'".format(
                schema_writing.replace("'", "''")
            )
        )
        assert (self._VERTICAPY_VARIABLES_["cursor"].fetchone()) or (schema_writing.lower() not in ['"v_temp_schema"', "v_temp_schema", '"public"', "public"]), MissingSchema(f"The schema '{schema_writing}' doesn't exist or is not accessible.\nThe attribute of the vDataFrame 'schema_writing' did not change.")
        self._VERTICAPY_VARIABLES_["schema_writing"] = schema_writing
        return self

    # ---#
    def score(
        self, y_true: str, y_score: str, method: str, nbins: int = 30,
    ):
        """
    ---------------------------------------------------------------------------
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

    Returns
    -------
    float / tablesample
        score / tablesample of the curve

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        check_types(
            [
                ("y_true", y_true, [str],),
                ("y_score", y_score, [str],),
                ("method", method, [str],),
            ]
        )
        columns_check([y_true, y_score], self)
        if method in ("r2", "rsquared"):
            from verticapy.learn.metrics import r2_score

            return r2_score(
                y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]
            )
        elif method in ("mae", "mean_absolute_error"):
            from verticapy.learn.metrics import mean_absolute_error

            return mean_absolute_error(
                y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]
            )
        elif method in ("mse", "mean_squared_error"):
            from verticapy.learn.metrics import mean_squared_error

            return mean_squared_error(
                y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]
            )
        elif method in ("msle", "mean_squared_log_error"):
            from verticapy.learn.metrics import mean_squared_log_error

            return mean_squared_log_error(
                y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]
            )
        elif method in ("max", "max_error"):
            from verticapy.learn.metrics import max_error

            return max_error(
                y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]
            )
        elif method in ("median", "median_absolute_error"):
            from verticapy.learn.metrics import median_absolute_error

            return median_absolute_error(
                y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]
            )
        elif method in ("var", "explained_variance"):
            from verticapy.learn.metrics import explained_variance

            return explained_variance(
                y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]
            )
        elif method in ("accuracy", "acc"):
            from verticapy.learn.metrics import accuracy_score

            return accuracy_score(
                y_true,
                y_score,
                self.__genSQL__(),
                self._VERTICAPY_VARIABLES_["cursor"],
                pos_label=None,
            )
        elif method == "auc":
            from verticapy.learn.metrics import auc

            return auc(
                y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]
            )
        elif method == "prc_auc":
            from verticapy.learn.metrics import prc_auc

            return prc_auc(
                y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]
            )
        elif method in ("best_cutoff", "best_threshold"):
            from verticapy.learn.model_selection import roc_curve

            return roc_curve(
                y_true,
                y_score,
                self.__genSQL__(),
                self._VERTICAPY_VARIABLES_["cursor"],
                best_threshold=True,
                nbins=nbins,
            )
        elif method in ("recall", "tpr"):
            from verticapy.learn.metrics import recall_score

            return recall_score(
                y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]
            )
        elif method in ("precision", "ppv"):
            from verticapy.learn.metrics import precision_score

            return precision_score(
                y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]
            )
        elif method in ("specificity", "tnr"):
            from verticapy.learn.metrics import specificity_score

            return specificity_score(
                y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]
            )
        elif method in ("negative_predictive_value", "npv"):
            from verticapy.learn.metrics import precision_score

            return precision_score(
                y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]
            )
        elif method in ("log_loss", "logloss"):
            from verticapy.learn.metrics import log_loss

            return log_loss(
                y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]
            )
        elif method == "f1":
            from verticapy.learn.metrics import f1_score

            return f1_score(
                y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]
            )
        elif method == "mcc":
            from verticapy.learn.metrics import matthews_corrcoef

            return matthews_corrcoef(
                y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]
            )
        elif method in ("bm", "informedness"):
            from verticapy.learn.metrics import informedness

            return informedness(
                y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]
            )
        elif method in ("mk", "markedness"):
            from verticapy.learn.metrics import markedness

            return markedness(
                y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]
            )
        elif method in ("csi", "critical_success_index"):
            from verticapy.learn.metrics import critical_success_index

            return critical_success_index(
                y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]
            )
        elif method in ("roc_curve", "roc"):
            from verticapy.learn.model_selection import roc_curve

            return roc_curve(
                y_true,
                y_score,
                self.__genSQL__(),
                self._VERTICAPY_VARIABLES_["cursor"],
                nbins=nbins,
            )
        elif method in ("prc_curve", "prc"):
            from verticapy.learn.model_selection import prc_curve

            return prc_curve(
                y_true,
                y_score,
                self.__genSQL__(),
                self._VERTICAPY_VARIABLES_["cursor"],
                nbins=nbins,
            )
        elif method in ("lift_chart", "lift"):
            from verticapy.learn.model_selection import lift_chart

            return lift_chart(
                y_true,
                y_score,
                self.__genSQL__(),
                self._VERTICAPY_VARIABLES_["cursor"],
                nbins=nbins,
            )
        else:
            raise ParameterError(
                "The parameter 'method' must be in roc|prc|lift|accuracy|auc|prc_auc|best_cutoff|recall|precision|log_loss|negative_predictive_value|specificity|mcc|informedness|markedness|critical_success_index|r2|mae|mse|msle|max|median|var"
            )

    # ---#
    def shape(self):
        """
    ---------------------------------------------------------------------------
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
        query = "SELECT COUNT(*) FROM {} LIMIT 1".format(self.__genSQL__())
        self.__executeSQL__(
            query, title="Computes the total number of elements (COUNT(*))"
        )
        self._VERTICAPY_VARIABLES_["count"] = self._VERTICAPY_VARIABLES_[
            "cursor"
        ].fetchone()[0]
        return (self._VERTICAPY_VARIABLES_["count"], m)

    # ---#
    def skewness(self, columns: list = []):
        """
    ---------------------------------------------------------------------------
    Aggregates the vDataFrame using 'skewness'.

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        return self.aggregate(func=["skewness"], columns=columns)

    skew = skewness
    # ---#
    def sort(self, columns: (dict, list)):
        """
    ---------------------------------------------------------------------------
    Sorts the vDataFrame using the input vColumns.

    Parameters
    ----------
    columns: dict / list
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
        check_types([("columns", columns, [dict, list],)])
        columns_check([elem for elem in columns], self)
        max_pos = 0
        columns_tmp = [elem for elem in self._VERTICAPY_VARIABLES_["columns"]]
        for column in columns_tmp:
            max_pos = max(max_pos, len(self[column].transformations) - 1)
        self._VERTICAPY_VARIABLES_["order_by"][max_pos] = sort_str(columns, self)
        return self

    # ---#
    def stacked_area(
        self,
        ts: str,
        columns: list = [],
        start_date: (str, datetime.datetime, datetime.date,) = "",
        end_date: (str, datetime.datetime, datetime.date,) = "",
        fully: bool = False,
        ax=None,
        **style_kwds,
    ):
        """
    ---------------------------------------------------------------------------
    Draws the stacked area chart of the time series.

    Parameters
    ----------
    ts: str
        TS (Time Series) vColumn to use to order the data. The vColumn type must be
        date like (date, datetime, timestamp...) or numerical.
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used. They must all include only positive values.
    start_date: str / date, optional
        Input Start Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is lesser than November 1993 the 3rd.
    end_date: str / date, optional
        Input End Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is greater than November 1993 the 3rd.
    fully: bool, optional
        If set to True, a Fully Stacked Area Chart will be drawn.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object
        """
        if isinstance(columns, str):
            columns = [columns]
        check_types(
            [
                ("columns", columns, [list],),
                ("ts", ts, [str],),
                (
                    "start_date",
                    start_date,
                    [str, datetime.datetime, datetime.date, int, float,],
                ),
                (
                    "end_date",
                    end_date,
                    [str, datetime.datetime, datetime.date, int, float,],
                ),
            ]
        )
        if fully:
            kind = "area_percent"
        else:
            kind = "area_stacked"
        assert min(self.min(columns)["min"]) >= 0, ValueError("Columns having negative values can not be processed by the 'stacked_area' method.")
        columns_check(columns + [ts], self)
        columns = vdf_columns_names(columns, self)
        ts = vdf_columns_names([ts], self)[0]
        from verticapy.plot import multi_ts_plot

        return multi_ts_plot(
            self, ts, columns, start_date, end_date, kind=kind, ax=ax, **style_kwds,
        )

    # ---#
    def std(self, columns: list = []):
        """
    ---------------------------------------------------------------------------
    Aggregates the vDataFrame using 'std' (Standard Deviation).

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        return self.aggregate(func=["stddev"], columns=columns)

    stddev = std
    # ---#
    def sum(self, columns: list = []):
        """
    ---------------------------------------------------------------------------
    Aggregates the vDataFrame using 'sum'.

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        return self.aggregate(func=["sum"], columns=columns)

    # ---#
    def swap(self, column1: (int, str), column2: (int, str)):
        """
    ---------------------------------------------------------------------------
    Swap the two input vColumns.

    Parameters
    ----------
    column1: str/int
        The first vColumn or its index to swap.
    column2: str/int
        The second vColumn or its index to swap.

    Returns
    -------
    vDataFrame
        self
        """
        check_types(
            [("column1", column1, [str, int],), ("column2", column2, [str, int],),]
        )
        if isinstance(column1, int):
            assert column1 < self.shape()[1], ParameterError("The parameter 'column1' is incorrect, it is greater or equal to the vDataFrame number of columns: {}>={}\nWhen this parameter type is 'integer', it must represent the index of the column to swap.".format(column1, self.shape()[1]))
            column1 = self.get_columns()[column1]
        if isinstance(column2, int):
            assert column2 < self.shape()[1], ParameterError("The parameter 'column2' is incorrect, it is greater or equal to the vDataFrame number of columns: {}>={}\nWhen this parameter type is 'integer', it must represent the index of the column to swap.".format(column2, self.shape()[1]))
            column2 = self.get_columns()[column2]
        columns_check([column1, column2], self)
        column1 = vdf_columns_names([column1], self)[0]
        column2 = vdf_columns_names([column2], self)[0]
        columns = self._VERTICAPY_VARIABLES_["columns"]
        all_cols = {}
        for idx, elem in enumerate(columns):
            all_cols[elem] = idx
        columns[all_cols[column1]], columns[all_cols[column2]] = (
            columns[all_cols[column2]],
            columns[all_cols[column1]],
        )
        return self

    # ---#
    def tail(self, limit: int = 5):
        """
    ---------------------------------------------------------------------------
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

    # ---#
    def to_csv(
        self,
        name: str,
        path: str = "",
        sep: str = ",",
        na_rep: str = "",
        quotechar: str = '"',
        usecols: list = [],
        header: bool = True,
        new_header: list = [],
        order_by: (list, dict) = [],
        limit: int = 0,
    ):
        """
    ---------------------------------------------------------------------------
    Creates a CSV file of the current vDataFrame relation.

    Parameters
    ----------
    name: str
        Name of the CSV file. Be careful: if a CSV file with the same name exists, 
        it will over-write it.
    path: str, optional
        Absolute path where the CSV file will be created.
    sep: str, optional
        Column separator.
    na_rep: str, optional
        Missing values representation.
    quotechar: str, optional
        Char which will enclose the str values.
    usecols: list, optional
        vColumns to select from the final vDataFrame relation. If empty, all
        vColumns will be selected.
    header: bool, optional
        If set to False, no header will be written in the CSV file.
    new_header: list, optional
        List of columns to use to replace vColumns name in the CSV.
    order_by: dict / list, optional
        List of the vColumns to use to sort the data using asc order or
        dictionary of all sorting methods. For example, to sort by "column1"
        ASC and "column2" DESC, write {"column1": "asc", "column2": "desc"}
    limit: int, optional
        If greater than 0, the maximum number of elements to write at the same time 
        in the CSV file. It can be to use to minimize memory impacts. Be sure to keep
        the same order to avoid unexpected results.

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame.to_db   : Saves the vDataFrame current relation to the Vertica database.
    vDataFrame.to_json : Creates a JSON file of the current vDataFrame relation.
        """
        if isinstance(order_by, str):
            order_by = [order_by]
        if isinstance(usecols, str):
            usecols = [usecols]
        check_types(
            [
                ("name", name, [str],),
                ("path", path, [str],),
                ("sep", sep, [str],),
                ("na_rep", na_rep, [str],),
                ("quotechar", quotechar, [str],),
                ("usecols", usecols, [list],),
                ("header", header, [bool],),
                ("new_header", new_header, [list],),
                ("order_by", order_by, [list, dict],),
                ("limit", limit, [int, float],),
            ]
        )
        file = open("{}{}.csv".format(path, name), "w+")
        columns = (
            self.get_columns()
            if not (usecols)
            else [str_column(column) for column in usecols]
        )
        assert not(new_header) or len(new_header) == len(columns), ParsingError("The header has an incorrect number of columns")
        if new_header:
            file.write(sep.join(new_header))
        elif header:
            file.write(sep.join([column.replace('"', "") for column in columns]))
        total = self.shape()[0]
        current_nb_rows_written = 0
        if limit <= 0:
            limit = total
        order_by = sort_str(order_by, self)
        if not (order_by):
            order_by = last_order_by(self)
        while current_nb_rows_written < total:
            self._VERTICAPY_VARIABLES_["cursor"].execute(
                "SELECT {} FROM {}{} LIMIT {} OFFSET {}".format(
                    ", ".join(columns),
                    self.__genSQL__(),
                    order_by,
                    limit,
                    current_nb_rows_written,
                )
            )
            result = self._VERTICAPY_VARIABLES_["cursor"].fetchall()
            for row in result:
                tmp_row = []
                for item in row:
                    if isinstance(item, str):
                        tmp_row += [quotechar + item + quotechar]
                    elif item == None:
                        tmp_row += [na_rep]
                    else:
                        tmp_row += [str(item)]
                file.write("\n" + sep.join(tmp_row))
            current_nb_rows_written += limit
        file.close()
        return self

    # ---#
    def to_db(
        self,
        name: str,
        usecols: list = [],
        relation_type: str = "view",
        inplace: bool = False,
        db_filter: (str, list) = "",
        nb_split: int = 0,
    ):
        """
    ---------------------------------------------------------------------------
    Saves the vDataFrame current relation to the Vertica database.

    Parameters
    ----------
    name: str
        Name of the relation. To save the relation in a specific schema you can
        write '"my_schema"."my_relation"'. Use double quotes '"' to avoid errors
        due to special characters.
    usecols: list, optional
        vColumns to select from the final vDataFrame relation. If empty, all
        vColumns will be selected.
    relation_type: str, optional
        Type of the relation.
            view      : View
            table     : Table
            temporary : Temporary Table
            local     : Local Temporary Table
    inplace: bool, optional
        If set to True, the vDataFrame will be replaced using the new relation.
    db_filter: str / list, optional
        Filter used before creating the relation in the DB. It can be a list of
        conditions or an expression. This parameter is very useful to create train 
        and test sets on TS.
    nb_split: int, optional
        If this parameter is greater than 0, it will add to the final relation a
        new column '_verticapy_split_' which will contain values in 
        [0;nb_split - 1] where each category will represent 1 / nb_split
        of the entire distribution. 

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame.to_csv : Creates a csv file of the current vDataFrame relation.
        """
        if isinstance(usecols, str):
            usecols = [usecols]
        check_types(
            [
                ("name", name, [str],),
                ("usecols", usecols, [list],),
                (
                    "relation_type",
                    relation_type,
                    ["view", "temporary", "table", "local"],
                ),
                ("inplace", inplace, [bool],),
                ("db_filter", db_filter, [str, list],),
                ("nb_split", nb_split, [int, float],),
            ]
        )
        relation_type = relation_type.lower()
        columns_check(usecols, self)
        usecols = vdf_columns_names(usecols, self)
        commit = (
            " ON COMMIT PRESERVE ROWS"
            if (relation_type in ("local", "temporary"))
            else ""
        )
        if relation_type == "temporary":
            relation_type += " table"
        elif relation_type == "local":
            relation_type += " temporary table"
        usecols = (
            "*"
            if not (usecols)
            else ", ".join([str_column(column) for column in usecols])
        )
        random_func = random_function(nb_split)
        nb_split = (
            ", {} AS _verticapy_split_".format(random_func) if (nb_split > 0) else ""
        )
        if isinstance(db_filter, Iterable) and not (isinstance(db_filter, str)):
            db_filter = " AND ".join(["({})".format(elem) for elem in db_filter])
        db_filter = " WHERE {}".format(db_filter) if (db_filter) else ""
        query = "CREATE {} {}{} AS SELECT {}{} FROM {}{}{}".format(
            relation_type.upper(),
            name,
            commit,
            usecols,
            nb_split,
            self.__genSQL__(),
            db_filter,
            last_order_by(self),
        )
        self.__executeSQL__(
            query=query,
            title="Creates a new {} to save the vDataFrame.".format(relation_type),
        )
        self.__add_to_history__(
            "[Save]: The vDataFrame was saved into a {} named '{}'.".format(
                relation_type, name
            )
        )
        if inplace:
            history, saving = (
                self._VERTICAPY_VARIABLES_["history"],
                self._VERTICAPY_VARIABLES_["saving"],
            )
            catalog_vars, columns = {}, self.get_columns()
            for column in columns:
                catalog_vars[column] = self[column].catalog
            self.__init__(name, self._VERTICAPY_VARIABLES_["cursor"])
            self._VERTICAPY_VARIABLES_["history"] = history
            for column in columns:
                self[column].catalog = catalog_vars[column]
        return self

    # ---#
    def to_geopandas(self, geometry: str):
        """
    ---------------------------------------------------------------------------
    Converts the vDataFrame to a Geopandas DataFrame.

    \u26A0 Warning : The data will be loaded in memory.

    Parameters
    ----------
    geometry: str
        Geometry object used to create the GeoDataFrame.
        It can also be a Geography object but it will be casted to Geometry.

    Returns
    -------
    geopandas.GeoDataFrame
        The geopandas.GeoDataFrame of the current vDataFrame relation.
        """
        try:
            from geopandas import GeoDataFrame
            from shapely import wkt
        except:
            raise ImportError(
                "The geopandas module seems to not be installed in your environment.\nTo be able to use this method, you'll have to install it.\n[Tips] Run: 'pip3 install geopandas' in your terminal to install the module."
            )
        try:
            import pandas as pd
        except:
            raise ImportError(
                "The pandas module seems to not be installed in your environment.\nTo be able to use this method, you'll have to install it.\n[Tips] Run: 'pip3 install pandas' in your terminal to install the module."
            )
        columns = self.get_columns(exclude_columns=[geometry])
        columns = ", ".join(columns)
        if columns:
            columns += ", "
        columns += "ST_AsText({}) AS {}".format(geometry, geometry)
        query = "SELECT {} FROM {}{}".format(
            columns, self.__genSQL__(), last_order_by(self)
        )
        self.__executeSQL__(query, title="Gets the vDataFrame values.")
        column_names = [
            column[0] for column in self._VERTICAPY_VARIABLES_["cursor"].description
        ]
        data = self._VERTICAPY_VARIABLES_["cursor"].fetchall()
        df = pd.DataFrame(data)
        df.columns = column_names
        if len(geometry) > 2 and geometry[0] == geometry[-1] == '"':
            geometry = geometry[1:-1]
        df[geometry] = df[geometry].apply(wkt.loads)
        df = GeoDataFrame(df, geometry=geometry)
        return df

    # ---#
    def to_json(
        self,
        name: str,
        path: str = "",
        usecols: list = [],
        order_by: (list, dict) = [],
        limit: int = 0,
    ):
        """
    ---------------------------------------------------------------------------
    Creates a JSON file of the current vDataFrame relation.

    Parameters
    ----------
    name: str
        Name of the JSON file. Be careful: if a JSON file with the same name exists, 
        it will over-write it.
    path: str, optional
        Absolute path where the JSON file will be created.
    usecols: list, optional
        vColumns to select from the final vDataFrame relation. If empty, all
        vColumns will be selected.
    order_by: dict / list, optional
        List of the vColumns to use to sort the data using asc order or
        dictionary of all sorting methods. For example, to sort by "column1"
        ASC and "column2" DESC, write {"column1": "asc", "column2": "desc"}
    limit: int, optional
        If greater than 0, the maximum number of elements to write at the same time 
        in the JSON file. It can be to use to minimize memory impacts. Be sure to keep
        the same order to avoid unexpected results.

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame.to_csv : Creates a CSV file of the current vDataFrame relation.
    vDataFrame.to_db  : Saves the vDataFrame current relation to the Vertica database.
        """
        if isinstance(order_by, str):
            order_by = [order_by]
        if isinstance(usecols, str):
            usecols = [usecols]
        check_types(
            [
                ("name", name, [str],),
                ("path", path, [str],),
                ("usecols", usecols, [list],),
                ("order_by", order_by, [list, dict],),
                ("limit", limit, [int, float],),
            ]
        )
        file = open("{}{}.json".format(path, name), "w+")
        columns = (
            self.get_columns()
            if not (usecols)
            else [str_column(column) for column in usecols]
        )
        total = self.shape()[0]
        current_nb_rows_written = 0
        if limit <= 0:
            limit = total
        file.write("[\n")
        order_by = sort_str(order_by, self)
        if not (order_by):
            order_by = last_order_by(self)
        while current_nb_rows_written < total:
            self._VERTICAPY_VARIABLES_["cursor"].execute(
                "SELECT {} FROM {}{} LIMIT {} OFFSET {}".format(
                    ", ".join(columns),
                    self.__genSQL__(),
                    order_by,
                    limit,
                    current_nb_rows_written,
                )
            )
            result = self._VERTICAPY_VARIABLES_["cursor"].fetchall()
            for row in result:
                tmp_row = []
                for i, item in enumerate(row):
                    if isinstance(item, str):
                        tmp_row += ['{}: "{}"'.format(str_column(columns[i]), item)]
                    elif item != None:
                        tmp_row += ["{}: {}".format(str_column(columns[i]), item)]
                file.write("{" + ", ".join(tmp_row) + "},\n")
            current_nb_rows_written += limit
        file.write("]")
        file.close()
        return self

    # ---#
    def to_list(self):
        """
    ---------------------------------------------------------------------------
    Converts the vDataFrame to a Python list.

    \u26A0 Warning : The data will be loaded in memory.

    Returns
    -------
    List
        The list of the current vDataFrame relation.
        """
        query = "SELECT * FROM {}{}".format(self.__genSQL__(), last_order_by(self))
        self.__executeSQL__(query, title="Gets the vDataFrame values.")
        result = self._VERTICAPY_VARIABLES_["cursor"].fetchall()
        final_result = []
        for elem in result:
            final_result += [
                [
                    float(item) if isinstance(item, decimal.Decimal) else item
                    for item in elem
                ]
            ]
        return final_result

    # ---#
    def to_numpy(self):
        """
    ---------------------------------------------------------------------------
    Converts the vDataFrame to a Numpy array.

    \u26A0 Warning : The data will be loaded in memory.

    Returns
    -------
    numpy.array
        The numpy array of the current vDataFrame relation.
        """
        import numpy as np

        return np.array(self.to_list())

    # ---#
    def to_pandas(self):
        """
    ---------------------------------------------------------------------------
    Converts the vDataFrame to a pandas DataFrame.

    \u26A0 Warning : The data will be loaded in memory.

    Returns
    -------
    pandas.DataFrame
        The pandas.DataFrame of the current vDataFrame relation.
        """
        try:
            import pandas as pd
        except:
            raise ImportError(
                "The pandas module seems to not be installed in your environment.\nTo be able to use this method, you'll have to install it.\n[Tips] Run: 'pip3 install pandas' in your terminal to install the module."
            )
        query = "SELECT * FROM {}{}".format(self.__genSQL__(), last_order_by(self))
        self.__executeSQL__(query, title="Gets the vDataFrame values.")
        column_names = [
            column[0] for column in self._VERTICAPY_VARIABLES_["cursor"].description
        ]
        data = self._VERTICAPY_VARIABLES_["cursor"].fetchall()
        df = pd.DataFrame(data)
        df.columns = column_names
        return df

    # ---#
    def to_pickle(self, name: str):
        """
    ---------------------------------------------------------------------------
    Saves the vDataFrame to a Python pickle file.

    Parameters
    ----------
    name: str
        Name of the file. Be careful: if a file with the same name exists, it 
        will over-write it.

    Returns
    -------
    vDataFrame
        self
        """
        vdf = self.copy()
        vdf._VERTICAPY_VARIABLES_["cursor"] = None
        pickle.dump(vdf, open(name, "wb"))
        return self

    # ---#
    def to_shp(
        self,
        name: str,
        path: str,
        usecols: list = [],
        overwrite: bool = True,
        shape: str = "Polygon",
    ):
        """
    ---------------------------------------------------------------------------
    Creates a SHP file of the current vDataFrame relation. For the moment, 
    files will be exported in the Vertica server.

    Parameters
    ----------
    name: str
        Name of the SHP file.
    path: str
        Absolute path where the SHP file will be created.
    usecols: list, optional
        vColumns to select from the final vDataFrame relation. If empty, all
        vColumns will be selected.
    overwrite: bool, optional
        If set to True, the function will overwrite the index if an index exists.
    shape: str, optional
        Must be one of the following spatial classes: 
            Point, Polygon, Linestring, Multipoint, Multipolygon, Multilinestring. 
        Polygons and Multipolygons always have a clockwise orientation.

    Returns
    -------
    vDataFrame
        self
        """
        if isinstance(usecols, str):
            usecols = [usecols]
        check_types(
            [
                ("name", name, [str],),
                ("path", path, [str],),
                ("usecols", usecols, [list],),
                ("overwrite", overwrite, [bool],),
                (
                    "shape",
                    shape,
                    [
                        "Point",
                        "Polygon",
                        "Linestring",
                        "Multipoint",
                        "Multipolygon",
                        "Multilinestring",
                    ],
                ),
            ]
        )
        query = (
            f"SELECT STV_SetExportShapefileDirectory(USING PARAMETERS path = '{path}');"
        )
        self.__executeSQL__(query=query, title="Setting SHP Export directory.")
        columns = (
            self.get_columns()
            if not (usecols)
            else [str_column(column) for column in usecols]
        )
        columns = ", ".join(columns)
        query = f"SELECT STV_Export2Shapefile({columns} USING PARAMETERS shapefile = '{name}.shp', overwrite = {overwrite}, shape = '{shape}') OVER() FROM {self.__genSQL__()};"
        self.__executeSQL__(query=query, title="Exporting the SHP.")
        return self

    # ---#
    def train_test_split(
        self,
        test_size: float = 0.33,
        order_by: (list, dict) = {},
        random_state: int = None,
    ):
        """
    ---------------------------------------------------------------------------
    Creates 2 vDataFrame (train/test) which can be to use to evaluate a model.
    The intersection between the train and the test is empty only if a unique
    order is specified.

    Parameters
    ----------
    test_size: float, optional
        Proportion of the test set comparint to the training set.
    order_by: dict / list, optional
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
        check_types(
            [
                ("test_size", test_size, [float],),
                ("order_by", order_by, [list, dict],),
                ("random_state", random_state, [int],),
            ]
        )
        order_by = sort_str(order_by, self)
        random_state = verticapy.options["random_state"]
        random_seed = (
            random_state
            if isinstance(random_state, int)
            else random.randint(-10e6, 10e6)
        )
        random_func = "SEEDED_RANDOM({})".format(random_seed)
        test_table = "(SELECT * FROM {} WHERE {} < {}{}) x".format(
            self.__genSQL__(), random_func, test_size, order_by
        )
        train_table = "(SELECT * FROM {} WHERE {} > {}{}) x".format(
            self.__genSQL__(), random_func, test_size, order_by
        )
        return (
            vdf_from_relation(
                relation=train_table, cursor=self._VERTICAPY_VARIABLES_["cursor"]
            ),
            vdf_from_relation(
                relation=test_table, cursor=self._VERTICAPY_VARIABLES_["cursor"]
            ),
        )

    # ---#
    def var(self, columns: list = []):
        """
    ---------------------------------------------------------------------------
    Aggregates the vDataFrame using 'variance'.

    Parameters
    ----------
    columns: list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        return self.aggregate(func=["variance"], columns=columns)

    variance = var
    # ---#
    def version(self):
        """
    ---------------------------------------------------------------------------
    Returns the version of Vertica.

    Returns
    -------
    list
        List containing the version information.
        [MAJOR, MINOR, PATCH, POST]
        """
        from verticapy.utilities import version as vertica_version

        return vertica_version(cursor=self._VERTICAPY_VARIABLES_["cursor"])

    # ---#
    def iv_woe(
        self, y: str, columns: list = [], bins: int = 10, show: bool = True, ax=None,
    ):
        """
    ---------------------------------------------------------------------------
    Computes the Information Value (IV) Table. It tells the predictive power of 
    an independent variable in relation to the dependent variable.

    Parameters
    ----------
    y: str
        Response vColumn.
    columns: list, optional
        List of the vColumns names. If empty, all vColumns except the response 
        will be used.
    bins: int, optional
        Maximum number of bins used for the discretization (must be > 1).
    show: bool, optional
        If set to True, the IV Plot will be drawn using Matplotlib.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame[].iv_woe : Computes the Information Value (IV) / 
        Weight Of Evidence (WOE) Table.
        """
        if isinstance(columns, str):
            columns = [columns]
        check_types(
            [
                ("y", y, [str],),
                ("columns", columns, [list],),
                ("bins", bins, [int],),
                ("show", show, [bool],),
            ]
        )
        columns_check(columns + [y], self)
        columns = vdf_columns_names(columns, self)
        y = vdf_columns_names([y], self)[0]
        if not (columns):
            columns = self.get_columns(exclude_columns=[y])
        coeff_importances = {}
        for elem in columns:
            coeff_importances[elem] = self[elem].iv_woe(y=y, bins=bins,)["iv"][-1]
        if show:
            from verticapy.learn.mlplot import plot_importance

            ax = plot_importance(coeff_importances, print_legend=False, ax=ax,)
            ax.set_xlabel("IV")
        index = [elem for elem in coeff_importances]
        iv = [coeff_importances[elem] for elem in coeff_importances]
        data = [(index[i], iv[i]) for i in range(len(iv))]
        data = sorted(data, key=lambda tup: tup[1], reverse=True)
        return tablesample(
            {
                "index": [elem[0] for elem in data],
                "iv": [elem[1] for elem in data],
            }
        )
