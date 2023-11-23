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
import copy
import sys
import time
import warnings
from typing import Any, Optional, Union, TYPE_CHECKING

import verticapy._config.config as conf
from verticapy._typing import NoneType, SQLColumns
from verticapy._utils._map import verticapy_agg_name
from verticapy._utils._object import create_new_vdc
from verticapy._utils._sql._cast import to_varchar
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import format_type, indent_vpy_sql, quote_ident
from verticapy._utils._sql._random import _current_random
from verticapy._utils._sql._sys import _executeSQL

from verticapy.core.tablesample.base import TableSample

from verticapy.core.vdataframe._typing import vDFTyping, vDCTyping

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame


class vDFSystem(vDFTyping):
    def __format__(self, format_spec: Any) -> str:
        return format(self._genSQL(), format_spec)

    def _add_to_history(self, message: str) -> "vDataFrame":
        """
        VERTICAPY stores the user modification and helps the user
        to look at what they did. This method is used to add a
        customized message in the vDataFrame history attribute.
        """
        self._vars["history"] += ["{" + time.strftime("%c") + "}" + " " + message]
        return self

    def _genSQL(
        self,
        split: bool = False,
        transformations: Optional[dict] = None,
        force_columns: Optional[SQLColumns] = None,
    ) -> str:
        """
        Method used to generate the SQL final relation. It
        looks at all transformations and builds a nested query where
        each transformation is associated to a specific floor.

        Parameters
        ----------
        split: bool, optional
            Adds a split  column __verticapy_split__ in the relation,
            which can be used to downsample the data.
        transformations: dict, optional
            Dictionary of columns and their respective transformation.
            It can be used to test if an expression is correct and
            can be added in the final relation.
        force_columns: SQLColumns, optional
            Columns used to generate the final relation.

        Returns
        -------
        str
            The SQL final relation.
        """
        # The First step is to find the Max Floor
        all_imputations_grammar = []
        transformations = format_type(transformations, dtype=dict)
        force_columns = format_type(force_columns, dtype=list)
        force_columns_copy = copy.deepcopy(force_columns)
        if len(force_columns) == 0:
            force_columns = copy.deepcopy(self._vars["columns"])
        for column in force_columns:
            all_imputations_grammar += [
                [transformation[0] for transformation in self[column]._transf]
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
        where_positions = [item[1] for item in self._vars["where"]]
        max_where_pos = max(where_positions + [0])
        all_where = [[] for item in range(max_where_pos + 1)]
        for i in range(0, len(self._vars["where"])):
            all_where[where_positions[i]] += [self._vars["where"][i][0]]
        all_where = [
            " AND ".join([f"({c})" for c in condition]) for condition in all_where
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
        all_undefined = True
        for v in first_values:
            if v != "___VERTICAPY_UNDEFINED___":
                all_undefined = False
                break
        if (
            (transformations_first_floor)
            or (self._vars["allcols_ind"] != len(first_values))
        ) and not all_undefined:
            table = f"""
                SELECT 
                    {', '.join(first_values)} 
                FROM {self._vars['main_relation']}"""
        else:
            table = f"""SELECT * FROM {self._vars["main_relation"]}"""
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
            if (i - 1) in self._vars["order_by"]:
                table += self._vars["order_by"][i - 1]
        where_final = (
            all_where[max_transformation_floor - 1]
            if (len(all_where) > max_transformation_floor - 1)
            else ""
        )
        # Only the last order_by matters as the order_by will never change
        # the final relation
        try:
            order_final = self._vars["order_by"][max_transformation_floor - 1]
        except (IndexError, KeyError):
            order_final = ""
        for vml_undefined in [
            ", ___VERTICAPY_UNDEFINED___",
            "___VERTICAPY_UNDEFINED___, ",
            "___VERTICAPY_UNDEFINED___",
        ]:
            table = table.replace(vml_undefined, "")
        random_func = _current_random()
        split = f", {random_func} AS __verticapy_split__" if (split) else ""
        if (where_final == "") and (order_final == ""):
            if split:
                table = f"SELECT *{split} FROM ({table}) VERTICAPY_SUBTABLE"
            table = f"({table}) VERTICAPY_SUBTABLE"
        else:
            table = f"({table}) VERTICAPY_SUBTABLE{where_final}{order_final}"
            table = f"(SELECT *{split} FROM {table}) VERTICAPY_SUBTABLE"
        if (self._vars["exclude_columns"]) and not split:
            if not force_columns_copy:
                force_columns_copy = self.get_columns()
            force_columns_copy = ", ".join(force_columns_copy)
            table = f"""
                (SELECT 
                    {force_columns_copy}{split} 
                FROM {table}) VERTICAPY_SUBTABLE"""
        main_relation = self._vars["main_relation"]
        all_main_relation = f"(SELECT * FROM {main_relation}) VERTICAPY_SUBTABLE"
        return table.replace(all_main_relation, main_relation)

    def _get_catalog_value(
        self,
        column: Optional[str] = None,
        key: Optional[str] = None,
        method: Optional[str] = None,
        columns: Optional[SQLColumns] = None,
    ) -> Optional[str]:
        """
        VERTICAPY  stores  the  already  computed aggregations to  avoid
        useless computations. This method returns the stored aggregation
        if it was already computed.
        """
        if not conf.get_option("cache"):
            return "VERTICAPY_NOT_PRECOMPUTED"
        if column == "VERTICAPY_COUNT":
            if self._vars["count"] < 0:
                return "VERTICAPY_NOT_PRECOMPUTED"
            total = self._vars["count"]
            if not isinstance(total, (int, float)):
                return "VERTICAPY_NOT_PRECOMPUTED"
            return total
        elif method:
            method = verticapy_agg_name(method.lower())
            if columns[1] in self[columns[0]]._catalog[method]:
                return self[columns[0]]._catalog[method][columns[1]]
            else:
                return "VERTICAPY_NOT_PRECOMPUTED"
        key = verticapy_agg_name(key.lower())
        column = self.format_colnames(column)
        try:
            if (key == "approx_unique") and ("unique" in self[column]._catalog):
                key = "unique"
            result = (
                "VERTICAPY_NOT_PRECOMPUTED"
                if key not in self[column]._catalog
                else self[column]._catalog[key]
            )
        except AttributeError:
            result = "VERTICAPY_NOT_PRECOMPUTED"
        if result != result:
            result = None
        if ("top" not in key) and isinstance(result, NoneType):
            return "VERTICAPY_NOT_PRECOMPUTED"
        return result

    def _get_last_order_by(self) -> str:
        """
        Returns the last column used to sort the data.
        """
        max_pos, order_by = 0, ""
        columns_tmp = copy.deepcopy(self.get_columns())
        for column in columns_tmp:
            max_pos = max(max_pos, len(self[column]._transf) - 1)
        if max_pos in self._vars["order_by"]:
            order_by = self._vars["order_by"][max_pos]
        return order_by

    def _get_hash_syntax(self, columns: Union[dict, SQLColumns]) -> str:
        """
        Returns the SQL syntax used to segment using the input columns.
        """
        if not columns:
            return ""
        segment_cols = quote_ident(columns)
        return f" SEGMENTED BY HASH({', '.join(segment_cols)}) ALL NODES"

    def _get_sort_syntax(self, columns: Union[dict, SQLColumns]) -> str:
        """
        Returns the SQL syntax used to sort the input columns.
        """
        if not columns:
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
            order_by = quote_ident(columns)
        return f" ORDER BY {', '.join(order_by)}"

    def _update_catalog(
        self,
        values: Optional[dict] = None,
        erase: bool = False,
        columns: Optional[SQLColumns] = None,
        matrix: Optional[str] = None,
        column: Optional[str] = None,
    ) -> None:
        """
        VERTICAPY stores the already computed aggregations to
        avoid useless  computations.  This  method stores the
        input aggregation in the vDataColumn catalog.
        """
        values = format_type(values, dtype=dict)
        columns = format_type(columns, dtype=list)
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
            if not columns:
                columns = self.get_columns()
            for column in columns:
                self[column]._catalog = copy.deepcopy(agg_dict)
            self._vars["count"] = -1
        elif matrix:
            matrix = verticapy_agg_name(matrix.lower())
            if matrix in agg_dict:
                for elem in values:
                    val = values[elem]
                    try:
                        val = float(val)
                    except (TypeError, ValueError):
                        pass
                    self[column]._catalog[matrix][elem] = val
        else:
            columns = list(values)
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
                        except (OverflowError, TypeError, ValueError):
                            pass
                        if val != val:
                            val = None
                        self[column]._catalog[key] = val

    def current_relation(self, reindent: bool = True, split: bool = False) -> str:
        """
        Returns the current vDataFrame relation.

        Parameters
        ----------
        reindent: bool, optional
            Reindent the text to be more readable.
        split: bool, optional
            Adds a split  column __verticapy_split__
            in the  relation, which can be used to
            downsample the data.

        Returns
        -------
        str
            The formatted current vDataFrame relation.
        """
        if reindent:
            return indent_vpy_sql(self._genSQL(split=split))
        else:
            return self._genSQL(split=split)

    def del_catalog(self) -> "vDataFrame":
        """
        Deletes the current vDataFrame catalog.

        Returns
        -------
        vDataFrame
            self
        """
        self._update_catalog(erase=True)
        return self

    def empty(self) -> bool:
        """
        Returns True if the vDataFrame is empty.

        Returns
        -------
        bool
            True if the vDataFrame has no vDataColumns.
        """
        return not self.get_columns()

    @save_verticapy_logs
    def expected_store_usage(self, unit: str = "b") -> TableSample:
        """
        Returns the vDataFrame expected store usage.

        Parameters
        ----------
        unit: str, optional
            Unit used for the computation.
            b : byte
            kb: kilo byte
            gb: giga byte
            tb: tera byte

        Returns
        -------
        TableSample
            result.
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
            if ctype.startswith(("date", "time", "interval", "smalldatetime")):
                maxsize, expsize = 8, 8
            elif "int" in ctype:
                maxsize, expsize = 8, self[column].store_usage()
            elif ctype.startswith("bool"):
                maxsize, expsize = 1, 1
            elif ctype.startswith(("float", "double", "real")):
                maxsize, expsize = 8, 8
            elif ctype.startswith(("decimal", "number", "numeric", "money")):
                try:
                    size = sum(
                        int(item)
                        for item in ctype.split("(")[1].split(")")[0].split(",")
                    )
                except IndexError:
                    size = 38
                maxsize, expsize = size, size
            elif ctype.startswith("varchar"):
                try:
                    size = int(ctype.split("(")[1].split(")")[0])
                except IndexError:
                    size = 80
                maxsize, expsize = size, self[column].store_usage()
            elif ctype.startswith("geo") or ctype.endswith(
                ("binary", "bytea", "char", "raw")
            ):
                try:
                    size = int(ctype.split("(")[1].split(")")[0])
                    maxsize, expsize = size, size
                except IndexError:
                    if ctype.startswith("geo"):
                        maxsize, expsize = 10000000, 10000
                    elif ctype.startswith("long"):
                        maxsize, expsize = 32000000, 10000
                    else:
                        maxsize, expsize = 65000, 1000
            elif ctype.startswith("uuid"):
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
            (sum(len(col) for col in columns) + len(columns)) / div_unit,
            (sum(len(col) for col in columns) + len(columns)) / div_unit,
            "",
        ]
        total += values["header"][0]
        total_expected += values["header"][0]
        values["rawsize"] = [total_expected, total, ""]
        return TableSample(values=values).transpose()

    @save_verticapy_logs
    def explain(self, digraph: bool = False) -> str:
        """
        Provides information on how Vertica is computing the current
        vDataFrame relation.

        Parameters
        ----------
        digraph: bool, optional
            If set to True,  returns only the digraph of the explain
            plan.

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
                FROM {self}""",
            title="Explaining the Current Relation",
            method="fetchall",
            sql_push_ext=self._vars["sql_push_ext"],
            symbol=self._vars["symbol"],
        )
        result = [elem[0] for elem in result]
        result = "\n".join(result)
        if not digraph:
            result = result.replace("------------------------------\n", "")
            result = result.replace("\\n", "\n\t")
            result = result.replace(", ", ",").replace(",", ", ").replace("\n}", "}")
        else:
            result = "digraph G {" + result.split("digraph G {")[1]
        return result

    def info(self) -> str:
        """
        Displays information about the different vDataFrame
        transformations.

        Returns
        -------
        str
            information on the vDataFrame modifications
        """
        if len(self._vars["history"]) == 0:
            result = "The vDataFrame was never modified."
        elif len(self._vars["history"]) == 1:
            result = "The vDataFrame was modified with only one action: "
            result += "\n * " + self._vars["history"][0]
        else:
            result = "The vDataFrame was modified many times: "
            for modif in self._vars["history"]:
                result += "\n * " + modif
        return result

    @save_verticapy_logs
    def memory_usage(self) -> TableSample:
        """
        Returns the vDataFrame memory usage.

        Returns
        -------
        TableSample
            result.
        """
        total = sum(sys.getsizeof(v) for v in self._vars) + sys.getsizeof(self)
        values = {"index": ["object"], "value": [total]}
        columns = copy.deepcopy(self._vars["columns"])
        for column in columns:
            values["index"] += [column]
            values["value"] += [self[column].memory_usage()]
            total += self[column].memory_usage()
        values["index"] += ["total"]
        values["value"] += [total]
        return TableSample(values=values)

    @save_verticapy_logs
    def swap(self, column1: Union[int, str], column2: Union[int, str]) -> "vDataFrame":
        """
        Swap the two input vDataColumns.

        Parameters
        ----------
        column1: str / int
            The first  vDataColumn or its index to swap.
        column2: str / int
            The second vDataColumn or its index to swap.

        Returns
        -------
        vDataFrame
            self
        """
        if isinstance(column1, int):
            assert column1 < self.shape()[1], ValueError(
                "The parameter 'column1' is incorrect, it is greater or equal "
                f"to the vDataFrame number of columns: {column1}>={self.shape()[1]}"
                "\nWhen this parameter type is 'integer', it must represent the index "
                "of the column to swap."
            )
            column1 = self.get_columns()[column1]
        if isinstance(column2, int):
            assert column2 < self.shape()[1], ValueError(
                "The parameter 'column2' is incorrect, it is greater or equal "
                f"to the vDataFrame number of columns: {column2}>={self.shape()[1]}"
                "\nWhen this parameter type is 'integer', it must represent the "
                "index of the column to swap."
            )
            column2 = self.get_columns()[column2]
        column1, column2 = self.format_colnames(column1, column2)
        columns = self._vars["columns"]
        all_cols = {}
        for idx, elem in enumerate(columns):
            all_cols[elem] = idx
        columns[all_cols[column1]], columns[all_cols[column2]] = (
            columns[all_cols[column2]],
            columns[all_cols[column1]],
        )
        return self


class vDCSystem(vDCTyping):
    def __format__(self, format_spec) -> str:
        return format(self._alias, format_spec)

    def add_copy(self, name: str) -> "vDataFrame":
        """
        Adds a copy vDataColumn to the parent vDataFrame.

        Parameters
        ----------
        name: str
            Name of the copy.

        Returns
        -------
        vDataFrame
            self._parent
        """
        if name == "":
            raise ValueError("The parameter 'name' must not be empty")
        elif self._parent.is_colname_in(name):
            raise ValueError(
                f"A vDataColumn has already the alias {name}.\nBy changing "
                "the parameter 'name', you'll be able to solve this issue."
            )
        name = quote_ident(name.replace('"', "_"))
        new_vDataColumn = create_new_vdc(
            name,
            parent=self._parent,
            transformations=list(self._transf),
            catalog=self._catalog,
        )
        setattr(self._parent, name, new_vDataColumn)
        setattr(self._parent, name[1:-1], new_vDataColumn)
        self._parent._vars["columns"] += [name]
        self._parent._add_to_history(
            f"[Add Copy]: A copy of the vDataColumn {self} "
            f"named {name} was added to the vDataFrame."
        )
        return self._parent

    @save_verticapy_logs
    def memory_usage(self) -> float:
        """
        Returns the vDataColumn memory usage.

        Returns
        -------
        float
            vDataColumn memory usage (byte)
        """
        total = (
            sys.getsizeof(self)
            + sys.getsizeof(self._alias)
            + sys.getsizeof(self._transf)
            + sys.getsizeof(self._catalog)
        )
        for elem in self._catalog:
            total += sys.getsizeof(elem)
        return total

    @save_verticapy_logs
    def store_usage(self) -> int:
        """
        Returns the vDataColumn expected store usage (unit: b).

        Returns
        -------
        int
            vDataColumn expected store usage.
        """
        pre_comp = self._parent._get_catalog_value(self._alias, "store_usage")
        if pre_comp != "VERTICAPY_NOT_PRECOMPUTED":
            return pre_comp
        alias_sql_repr = to_varchar(self.category(), self._alias)
        store_usage = _executeSQL(
            query=f"""
                SELECT 
                    /*+LABEL('vDataColumn.storage_usage')*/ 
                    ZEROIFNULL(SUM(LENGTH({alias_sql_repr}::varchar))) 
                FROM {self._parent}""",
            title=f"Computing the Store Usage of the vDataColumn {self}.",
            method="fetchfirstelem",
            sql_push_ext=self._parent._vars["sql_push_ext"],
            symbol=self._parent._vars["symbol"],
        )
        self._parent._update_catalog(
            {"index": ["store_usage"], self._alias: [store_usage]}
        )
        return store_usage

    def rename(self, new_name: str) -> "vDataFrame":
        """
        Renames the vDataColumn by dropping the current vDataColumn
        and creating a copy with the specified name.

        \u26A0 Warning : SQL code generation  will be slower if the
                         vDataFrame  has been transformed  multiple
                         times, so it's better practice to use this
                         method when first preparing your data.

        Parameters
        ----------
        new_name: str
            The new vDataColumn alias.

        Returns
        -------
        vDataFrame
            self._parent
        """
        old_name = quote_ident(self._alias)
        new_name = quote_ident(new_name)[1:-1]
        if self._parent.is_colname_in(new_name):
            raise NameError(
                f"A vDataColumn has already the alias {new_name}.\n"
                "By changing the parameter 'new_name', you'll "
                "be able to solve this issue."
            )
        self._parent.eval(name=new_name, expr=old_name)
        parent = self.drop(add_history=False)
        parent._add_to_history(
            f"[Rename]: The vDataColumn {old_name} was renamed '{new_name}'."
        )
        return parent
