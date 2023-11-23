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
from typing import Any, Optional, TYPE_CHECKING
from collections.abc import Iterable

from vertica_python.errors import QueryError

import verticapy._config.config as conf
from verticapy._typing import SQLColumns
from verticapy._utils._object import create_new_vdf
from verticapy._utils._sql._cast import to_varchar
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import (
    clean_query,
    extract_subquery,
    format_type,
    quote_ident,
)
from verticapy._utils._sql._sys import _executeSQL
from verticapy._utils._sql._vertica_version import vertica_version

from verticapy.core.string_sql.base import StringSQL
from verticapy.core.tablesample.base import TableSample

from verticapy.core.vdataframe._utils import vDFUtils

if conf.get_import_success("IPython"):
    from IPython.display import HTML, display

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame


class vDFRead(vDFUtils):
    def __iter__(self) -> tuple:
        columns = self.get_columns()
        return (col for col in columns)

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            assert index.step in (1, None), ValueError(
                "vDataFrame doesn't allow slicing having steps different than 1."
            )
            index_stop = index.stop
            index_start = index.start
            if not isinstance(index_start, int):
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
                SELECT * 
                FROM {self}
                {self._get_last_order_by()} 
                OFFSET {index_start}{limit}"""
            return create_new_vdf(query)

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
                    FROM {self}
                    {self._get_last_order_by()} 
                    OFFSET {index} LIMIT 1""",
                title="Getting the vDataFrame element.",
                method="fetchrow",
                sql_push_ext=self._vars["sql_push_ext"],
                symbol=self._vars["symbol"],
            )

        elif isinstance(index, (str, StringSQL)):
            is_sql = False
            if hasattr(index, "object_type") and index.object_type == "vDataColumn":
                index = index._alias
            elif isinstance(index, StringSQL):
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

    def __repr__(self) -> str:
        return self._repr_object().__repr__()

    def _repr_html_(self, interactive: bool = False) -> str:
        return self._repr_object()._repr_html_(interactive)

    def _repr_object(self, interactive: bool = False) -> TableSample:
        if self._vars["sql_magic_result"]:
            self._vars["sql_magic_result"] = False
            query = extract_subquery(self._genSQL())
            query = clean_query(query)
            sql_on_init = conf.get_option("sql_on")
            limit = conf.get_option("max_rows")
            conf.set_option("sql_on", False)
            try:
                res = TableSample().read_sql(f"{query} LIMIT {limit}")
            except QueryError:
                res = TableSample().read_sql(query)
            finally:
                conf.set_option("sql_on", sql_on_init)
            if conf.get_option("count_on"):
                res.count = self.shape()[0]
            else:
                res.count = -1
            if conf.get_option("percent_bar"):
                percent = self.agg(["percent"]).transpose().values
                cnt = self.shape()[0]
                for column in res.values:
                    res.dtype[column] = self[column].ctype()
                    if cnt == 0:
                        res.percent[column] = 100.0
                    else:
                        res.percent[column] = percent[self.format_colnames(column)][0]
            return res
        max_rows = self._vars["max_rows"]
        if max_rows <= 0:
            max_rows = conf.get_option("max_rows")
        max_cols = conf.get_option("max_columns")
        colums = self.get_columns()
        n = len(colums)
        cols = None
        if n > max_cols:
            if n % 2 == 0:
                s = int(max_cols / 2)
                cols = colums[: s + 1] + colums[-s:]
            else:
                s = int((max_cols + 1) / 2)
                cols = colums[:s] + colums[-s:]
        return self.iloc(limit=max_rows, columns=cols)

    def idisplay(self) -> None:
        """
        This  method  displays  the interactive  table. It is used  when
        you don't want to activate interactive tables for all vDataFrames.
        """
        return display(HTML(self.copy()._repr_html_(interactive=True)))

    def get_columns(self, exclude_columns: Optional[SQLColumns] = None) -> list[str]:
        """
        Returns the vDataFrame vDataColumns.

        Parameters
        ----------
        exclude_columns: SQLColumns, optional
            List of the vDataColumns names to exclude from
            the final list.

        Returns
        -------
        List
            List of all vDataFrame columns.
        """
        exclude_columns = format_type(exclude_columns, dtype=list)
        exclude_columns_ = [
            c.replace('"', "").lower()
            for c in exclude_columns + self._vars["exclude_columns"]
        ]
        res = []
        for column in self._vars["columns"]:
            if column.replace('"', "").lower() not in exclude_columns_:
                res += [column]
        return res

    def head(self, limit: int = 5) -> TableSample:
        """
        Returns the vDataFrame head.

        Parameters
        ----------
        limit: int, optional
            Number of elements to display.

        Returns
        -------
        TableSample
            result.
        """
        return self.iloc(limit=limit, offset=0)

    def iloc(
        self, limit: int = 5, offset: int = 0, columns: Optional[SQLColumns] = None
    ) -> TableSample:
        """
        Returns a part of the vDataFrame (delimited by an offset
        and a limit).

        Parameters
        ----------
        limit: int, optional
            Number of elements to display.
        offset: int, optional
            Number of elements to skip.
        columns: SQLColumns, optional
            A list containing the names of the vDataColumns to
            include in the result.  If empty, all vDataColumns
            are selected.

        Returns
        -------
        TableSample
            result.
        """
        columns = format_type(columns, dtype=list, na_out=self.get_columns())
        columns = self.format_colnames(columns)
        if offset < 0:
            offset = max(0, self.shape()[0] - limit)
        all_columns = []
        for column in columns:
            cast = to_varchar(self[column].category(), column)
            all_columns += [f"{cast} AS {column}"]
        title = (
            "Reads the final relation using a limit "
            f"of {limit} and an offset of {offset}."
        )
        result = TableSample.read_sql(
            query=f"""
                SELECT 
                    {', '.join(all_columns)} 
                FROM {self}
                {self._get_last_order_by()} 
                LIMIT {limit} OFFSET {offset}""",
            title=title,
            max_columns=self._vars["max_columns"],
            sql_push_ext=self._vars["sql_push_ext"],
            symbol=self._vars["symbol"],
        )
        pre_comp = self._get_catalog_value("VERTICAPY_COUNT")
        if pre_comp != "VERTICAPY_NOT_PRECOMPUTED":
            result.count = pre_comp
        elif conf.get_option("count_on"):
            result.count = self.shape()[0]
        result.offset = offset
        all_percent = True
        for column in columns:
            if "percent" not in self[column]._catalog:
                all_percent = False
        all_percent = (all_percent) or (conf.get_option("percent_bar"))
        if all_percent:
            percent = self.aggregate(["percent"], columns).transpose().values
            for column in result.values:
                result.dtype[column] = self[column].ctype()
                if result.count == 0:
                    result.percent[column] = 100.0
                elif all_percent:
                    result.percent[column] = percent[self.format_colnames(column)][0]
        return result

    def shape(self) -> tuple[int, int]:
        """
        Returns the number of rows and columns of the vDataFrame.

        Returns
        -------
        tuple
            (number of lines, number of columns)
        """
        m = len(self.get_columns())
        pre_comp = self._get_catalog_value("VERTICAPY_COUNT")
        if pre_comp != "VERTICAPY_NOT_PRECOMPUTED":
            return (pre_comp, m)
        self._vars["count"] = _executeSQL(
            query=f"""
                SELECT 
                    /*+LABEL('vDataframe.shape')*/ COUNT(*) 
                FROM {self} LIMIT 1
            """,
            title="Computing the total number of elements (COUNT(*))",
            method="fetchfirstelem",
            sql_push_ext=self._vars["sql_push_ext"],
            symbol=self._vars["symbol"],
        )
        return (self._vars["count"], m)

    def tail(self, limit: int = 5) -> TableSample:
        """
        Returns the tail of the vDataFrame.

        Parameters
        ----------
        limit: int, optional
            Number of elements to display.

        Returns
        -------
        TableSample
            result.
        """
        return self.iloc(limit=limit, offset=-1)

    @save_verticapy_logs
    def select(self, columns: SQLColumns) -> "vDataFrame":
        """
        Returns a copy of the vDataFrame with only the selected
        vDataColumns.

        Parameters
        ----------
        columns: SQLColumns
            List of the vDataColumns to  select. You can also
            provide customized expressions.

        Returns
        -------
        vDataFrame
            object with only the selected columns.
        """
        columns = format_type(columns, dtype=list)
        for i in range(len(columns)):
            column = self.format_colnames(columns[i], raise_error=False)
            if column:
                dtype = ""
                if self._vars["isflex"]:
                    dtype = self[column].ctype().lower()
                    if (
                        "array" in dtype
                        or "map" in dtype
                        or "row" in dtype
                        or "set" in dtype
                    ):
                        dtype = ""
                    else:
                        dtype = f"::{dtype}"
                columns[i] = column + dtype
            else:
                columns[i] = str(columns[i])
        query = f"SELECT  {', '.join(columns)} FROM {self}"
        return create_new_vdf(query)


class vDCRead:
    def __init__(self):
        """Must be overridden in final class"""
        self._parent = create_new_vdf(_empty=True)
        self._alias = ""
        self._transf = []
        self._catalog = {}
        self._init_transf = ""
        self._init = False

    def __getitem__(self, index) -> Any:
        if isinstance(index, slice):
            assert index.step in (1, None), ValueError(
                "vDataColumn doesn't allow slicing having steps different than 1."
            )
            index_stop = index.stop
            index_start = index.start
            if not isinstance(index_start, int):
                index_start = 0
            if self.isarray():
                vertica_version(condition=[10, 0, 0])
                if index_start < 0:
                    index_start_str = f"{index_start} + APPLY_COUNT_ELEMENTS({{}})"
                else:
                    index_start_str = str(index_start)
                if isinstance(index_stop, int):
                    if index_stop < 0:
                        index_stop_str = f"{index_stop} + APPLY_COUNT_ELEMENTS({{}})"
                    else:
                        index_stop_str = str(index_stop)
                else:
                    index_stop_str = "1 + APPLY_COUNT_ELEMENTS({})"
                elem_to_select = f"{self}[{index_start_str}:{index_stop_str}]"
                elem_to_select = elem_to_select.replace("{}", self._alias)
                new_alias = quote_ident(
                    f"{self._alias[1:-1]}.{index_start}:{index_stop}"
                )
                query = f"""
                    (SELECT 
                        {elem_to_select} AS {new_alias} 
                    FROM {self._parent}) VERTICAPY_SUBTABLE"""
                vcol = create_new_vdf(query)[new_alias]
                vcol._transf[-1] = (
                    new_alias,
                    self.ctype(),
                    self.category(),
                )
                vcol._init_transf = (
                    f"{self._init_transf}[{index_start_str}:{index_stop_str}]"
                )
                vcol._init_transf = vcol._init_transf.replace("{}", self._init_transf)
                return vcol
            else:
                if index_start < 0:
                    index_start += self._parent.shape()[0]
                if isinstance(index_stop, int):
                    if index_stop < 0:
                        index_stop += self._parent.shape()[0]
                    limit = index_stop - index_start
                    if limit <= 0:
                        limit = 0
                    limit = f" LIMIT {limit}"
                else:
                    limit = ""
                query = f"""
                    SELECT 
                        {self} 
                    FROM {self._parent}
                    {self._parent._get_last_order_by()} 
                    OFFSET {index_start} {limit}"""
                return create_new_vdf(query)
        elif isinstance(index, int):
            if self.isarray():
                vertica_version(condition=[9, 3, 0])
                elem_to_select = f"{self}[{index}]"
                new_alias = quote_ident(f"{self._alias[1:-1]}.{index}")
                query = f"""
                    SELECT 
                        {elem_to_select} AS {new_alias} 
                    FROM {self._parent}"""
                vcol = create_new_vdf(query)[new_alias]
                vcol._init_transf = f"{self._init_transf}[{index}]"
                return vcol
            else:
                cast = "::float" if self.category() == "float" else ""
                if index < 0:
                    index += self._parent.shape()[0]
                return _executeSQL(
                    query=f"""
                        SELECT 
                            /*+LABEL('vDataColumn.__getitem__')*/ 
                            {self}{cast} 
                        FROM {self._parent}
                        {self._parent._get_last_order_by()} 
                        OFFSET {index} 
                        LIMIT 1""",
                    title="Getting the vDataColumn element.",
                    method="fetchfirstelem",
                    sql_push_ext=self._parent._vars["sql_push_ext"],
                    symbol=self._parent._vars["symbol"],
                )
        elif isinstance(index, str):
            if self.category() == "vmap":
                index_str = index.replace("'", "''")
                elem_to_select = f"MAPLOOKUP({self}, '{index_str}')"
                init_transf = f"MAPLOOKUP({self._init_transf}, '{index_str}')"
            else:
                vertica_version(condition=[10, 0, 0])
                elem_to_select = f"{self}.{quote_ident(index)}"
                init_transf = f"{self._init_transf}.{quote_ident(index)}"
            query = f"""
                SELECT 
                    {elem_to_select} AS {quote_ident(index)} 
                FROM {self._parent}"""
            vcol = create_new_vdf(query)[index]
            vcol._init_transf = init_transf
            return vcol
        else:
            return getattr(self, index)

    def __repr__(self) -> str:
        return self.head(limit=conf.get_option("max_rows")).__repr__()

    def _repr_html_(self) -> str:
        return self.head(limit=conf.get_option("max_rows"))._repr_html_()

    def head(self, limit: int = 5) -> TableSample:
        """
        Returns the head of the vDataColumn.

        Parameters
        ----------
        limit: int, optional
            Number of elements to display.

        Returns
        -------
        TableSample
            result.
        """
        return self.iloc(limit=limit)

    def iloc(self, limit: int = 5, offset: int = 0) -> TableSample:
        """
        Returns a part of the vDataColumn (delimited by an offset
        and a limit).

        Parameters
        ----------
        limit: int, optional
            Number of elements to display.
        offset: int, optional
            Number of elements to skip.

        Returns
        -------
        TableSample
            result.
        """
        if offset < 0:
            offset = max(0, self._parent.shape()[0] - limit)
        title = f"Reads {self}."
        alias_sql_repr = to_varchar(self.category(), self._alias)
        tail = TableSample.read_sql(
            query=f"""
                SELECT 
                    {alias_sql_repr} AS {self} 
                FROM {self._parent}
                {self._parent._get_last_order_by()} 
                LIMIT {limit} 
                OFFSET {offset}""",
            title=title,
            sql_push_ext=self._parent._vars["sql_push_ext"],
            symbol=self._parent._vars["symbol"],
        )
        tail.count = self._parent.shape()[0]
        tail.offset = offset
        tail.dtype[self._alias] = self.ctype()
        return tail

    @save_verticapy_logs
    def nlargest(self, n: int = 10) -> TableSample:
        """
        Returns the n largest vDataColumn elements.

        Parameters
        ----------
        n: int, optional
            Offset.

        Returns
        -------
        TableSample
            result.
        """
        query = f"""
            SELECT 
                * 
            FROM {self._parent} 
            WHERE {self} IS NOT NULL 
            ORDER BY {self} DESC LIMIT {n}"""
        title = f"Reads {self} {n} largest elements."
        return TableSample.read_sql(
            query,
            title=title,
            sql_push_ext=self._parent._vars["sql_push_ext"],
            symbol=self._parent._vars["symbol"],
        )

    @save_verticapy_logs
    def nsmallest(self, n: int = 10) -> TableSample:
        """
        Returns the n smallest elements in the vDataColumn.

        Parameters
        ----------
        n: int, optional
            Offset.

        Returns
        -------
        TableSample
            result.
        """
        return TableSample.read_sql(
            f"""
            SELECT 
                * 
            FROM {self._parent} 
            WHERE {self} IS NOT NULL 
            ORDER BY {self} ASC LIMIT {n}""",
            title=f"Reads {n} {self} smallest elements.",
            sql_push_ext=self._parent._vars["sql_push_ext"],
            symbol=self._parent._vars["symbol"],
        )

    def tail(self, limit: int = 5) -> TableSample:
        """
        Returns the tail of the vDataColumn.

        Parameters
        ----------
        limit: int, optional
            Number of elements to display.

        Returns
        -------
        TableSample
            result.
        """
        return self.iloc(limit=limit, offset=-1)
