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
import copy, re, time, warnings
from typing import Union

from verticapy._config.config import current_random, OPTIONS
from verticapy._utils._cast import to_varchar
from verticapy._utils._collect import save_verticapy_logs
from verticapy._utils._sql._execute import _executeSQL
from verticapy._utils._sql._format import indentSQL, quote_ident

from verticapy.core._utils._map import verticapy_agg_name
from verticapy.core.tablesample.base import tablesample

from verticapy.sql.flex import isvmap


class vDFSYS:
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
    computations. This method stores the input aggregation in the vDataColumn catalog.
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

    #
    # Methods
    #

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
        if reindent:
            return indentSQL(self.__genSQL__())
        else:
            return self.__genSQL__()

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

    def empty(self):
        """
    Returns True if the vDataFrame is empty.

    Returns
    -------
    bool
        True if the vDataFrame has no vDataColumns.
        """
        return not (self.get_columns())

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
    def swap(self, column1: Union[int, str], column2: Union[int, str]):
        """
    Swap the two input vDataColumns.

    Parameters
    ----------
    column1: str / int
        The first vDataColumn or its index to swap.
    column2: str / int
        The second vDataColumn or its index to swap.

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


class vDCSYS:
    def add_copy(self, name: str):
        """
    Adds a copy vDataColumn to the parent vDataFrame.

    Parameters
    ----------
    name: str
        Name of the copy.

    Returns
    -------
    vDataFrame
        self.parent

    See Also
    --------
    vDataFrame.eval : Evaluates a customized expression.
        """
        from verticapy.core.vdataframe.base import vDataColumn

        name = quote_ident(name.replace('"', "_"))
        assert name.replace('"', ""), EmptyParameter(
            "The parameter 'name' must not be empty"
        )
        assert not (self.parent.is_colname_in(name)), NameError(
            f"A vDataColumn has already the alias {name}.\nBy changing "
            "the parameter 'name', you'll be able to solve this issue."
        )
        new_vDataColumn = vDataColumn(
            name,
            parent=self.parent,
            transformations=[item for item in self.transformations],
            catalog=self.catalog,
        )
        setattr(self.parent, name, new_vDataColumn)
        setattr(self.parent, name[1:-1], new_vDataColumn)
        self.parent._VERTICAPY_VARIABLES_["columns"] += [name]
        self.parent.__add_to_history__(
            f"[Add Copy]: A copy of the vDataColumn {self.alias} "
            f"named {name} was added to the vDataFrame."
        )
        return self.parent

    @save_verticapy_logs
    def memory_usage(self):
        """
    Returns the vDataColumn memory usage. 

    Returns
    -------
    float
        vDataColumn memory usage (byte)

    See Also
    --------
    vDataFrame.memory_usage : Returns the vDataFrame memory usage.
        """
        total = (
            sys.getsizeof(self)
            + sys.getsizeof(self.alias)
            + sys.getsizeof(self.transformations)
            + sys.getsizeof(self.catalog)
        )
        for elem in self.catalog:
            total += sys.getsizeof(elem)
        return total

    @save_verticapy_logs
    def store_usage(self):
        """
    Returns the vDataColumn expected store usage (unit: b).

    Returns
    -------
    int
        vDataColumn expected store usage.

    See Also
    --------
    vDataFrame.expected_store_usage : Returns the vDataFrame expected store usage.
        """
        pre_comp = self.parent.__get_catalog_value__(self.alias, "store_usage")
        if pre_comp != "VERTICAPY_NOT_PRECOMPUTED":
            return pre_comp
        alias_sql_repr = to_varchar(self.category(), self.alias)
        store_usage = _executeSQL(
            query=f"""
                SELECT 
                    /*+LABEL('vDataColumn.storage_usage')*/ 
                    ZEROIFNULL(SUM(LENGTH({alias_sql_repr}::varchar))) 
                FROM {self.parent.__genSQL__()}""",
            title=f"Computing the Store Usage of the vDataColumn {self.alias}.",
            method="fetchfirstelem",
            sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
            symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
        )
        self.parent.__update_catalog__(
            {"index": ["store_usage"], self.alias: [store_usage]}
        )
        return store_usage

    def rename(self, new_name: str):
        """
    Renames the vDataColumn by dropping the current vDataColumn and creating a copy with 
    the specified name.

    \u26A0 Warning : SQL code generation will be slower if the vDataFrame has been 
                     transformed multiple times, so it's better practice to use 
                     this method when first preparing your data.

    Parameters
    ----------
    new_name: str
        The new vDataColumn alias.

    Returns
    -------
    vDataFrame
        self.parent

    See Also
    --------
    vDataFrame.add_copy : Creates a copy of the vDataColumn.
        """
        old_name = quote_ident(self.alias)
        new_name = new_name.replace('"', "")
        assert not (self.parent.is_colname_in(new_name)), NameError(
            f"A vDataColumn has already the alias {new_name}.\n"
            "By changing the parameter 'new_name', you'll "
            "be able to solve this issue."
        )
        self.add_copy(new_name)
        parent = self.drop(add_history=False)
        parent.__add_to_history__(
            f"[Rename]: The vDataColumn {old_name} was renamed '{new_name}'."
        )
        return parent
