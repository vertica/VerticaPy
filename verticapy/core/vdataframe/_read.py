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
from typing import Optional, Union
from collections.abc import Iterable

from vertica_python.errors import QueryError

import verticapy._config.config as conf
from verticapy._typing import SQLColumns
from verticapy._utils._sql._cast import to_varchar
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import clean_query, extract_subquery, quote_ident
from verticapy._utils._sql._sys import _executeSQL
from verticapy._utils._sql._vertica_version import vertica_version

from verticapy.core.string_sql.base import StringSQL
from verticapy.core.tablesample.base import TableSample

if conf._get_import_success("jupyter"):
    from IPython.display import HTML, display


class vDFRead:
    def __iter__(self):
        columns = self.get_columns()
        return (col for col in columns)

    def __getitem__(self, index):
        from verticapy.core.vdataframe.base import vDataColumn

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
                SELECT * 
                FROM {self._genSQL()}
                {self._get_last_order_by()} 
                OFFSET {index_start}{limit}"""
            return self._new_vdataframe(query)

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
                    FROM {self._genSQL()}
                    {self._get_last_order_by()} 
                    OFFSET {index} LIMIT 1""",
                title="Getting the vDataFrame element.",
                method="fetchrow",
                sql_push_ext=self._vars["sql_push_ext"],
                symbol=self._vars["symbol"],
            )

        elif isinstance(index, (str, StringSQL)):
            is_sql = False
            if isinstance(index, vDataColumn):
                index = index._alias
            elif isinstance(index, StringSQL):
                index = str(index)
                is_sql = True
            try:
                new_index = self._format_colnames(index)
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

    def __repr__(self):
        return self._repr_object().__repr__()

    def _repr_html_(self, interactive: bool = False):
        return self._repr_object()._repr_html_(interactive)

    def _repr_object(self, interactive: bool = False):
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
                        res.percent[column] = percent[self._format_colnames(column)][0]
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

    def idisplay(self):
        """This method displays the interactive table. It is used when 
        you don't want to activate interactive table for all vDataFrames."""
        return display(HTML(self.copy()._repr_html_(interactive=True)))

    def get_columns(self, exclude_columns: SQLColumns = []):
        """
    Returns the vDataFrame vDataColumns.

    Parameters
    ----------
    exclude_columns: SQLColumns, optional
        List of the vDataColumns names to exclude from the final list. 

    Returns
    -------
    List
        List of all vDataFrame columns.

    See Also
    --------
    vDataFrame.catcol  : Returns all categorical vDataFrame vDataColumns.
    vDataFrame.datecol : Returns all vDataFrame vDataColumns of type date.
    vDataFrame.numcol  : Returns all numerical vDataFrame vDataColumns.
        """
        # -#
        if isinstance(exclude_columns, str):
            exclude_columns = [columns]
        columns = [elem for elem in self._vars["columns"]]
        result = []
        exclude_columns = [elem for elem in exclude_columns]
        exclude_columns += [elem for elem in self._vars["exclude_columns"]]
        exclude_columns = [elem.replace('"', "").lower() for elem in exclude_columns]
        for column in columns:
            if column.replace('"', "").lower() not in exclude_columns:
                result += [column]
        return result

    def head(self, limit: int = 5):
        """
    Returns the vDataFrame head.

    Parameters
    ----------
    limit: int, optional
        Number of elements to display.

    Returns
    -------
    TableSample
        An object containing the result. For more information, see
        utilities.TableSample.

    See Also
    --------
    vDataFrame.tail : Returns the vDataFrame tail.
        """
        return self.iloc(limit=limit, offset=0)

    def iloc(
        self, limit: int = 5, offset: int = 0, columns: Optional[SQLColumns] = None
    ):
        """
    Returns a part of the vDataFrame (delimited by an offset and a limit).

    Parameters
    ----------
    limit: int, optional
        Number of elements to display.
    offset: int, optional
        Number of elements to skip.
    columns: SQLColumns, optional
        A list containing the names of the vDataColumns to include in the result. 
        If empty, all vDataColumns will be selected.


    Returns
    -------
    TableSample
        An object containing the result. For more information, see
        utilities.TableSample.

    See Also
    --------
    vDataFrame.head : Returns the vDataFrame head.
    vDataFrame.tail : Returns the vDataFrame tail.
        """
        if isinstance(columns, str):
            columns = [columns]
        if offset < 0:
            offset = max(0, self.shape()[0] - limit)
        columns = self._format_colnames(columns)
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
        result = TableSample.read_sql(
            query=f"""
                SELECT 
                    {', '.join(all_columns)} 
                FROM {self._genSQL()}
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
            if not ("percent" in self[column]._catalog):
                all_percent = False
        all_percent = (all_percent) or (conf.get_option("percent_bar"))
        if all_percent:
            percent = self.aggregate(["percent"], columns).transpose().values
        for column in result.values:
            result.dtype[column] = self[column].ctype()
            if result.count == 0:
                result.percent[column] = 100.0
            elif all_percent:
                result.percent[column] = percent[self._format_colnames(column)][0]
        return result

    def shape(self):
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
                FROM {self._genSQL()} LIMIT 1
            """,
            title="Computing the total number of elements (COUNT(*))",
            method="fetchfirstelem",
            sql_push_ext=self._vars["sql_push_ext"],
            symbol=self._vars["symbol"],
        )
        return (self._vars["count"], m)

    def tail(self, limit: int = 5):
        """
    Returns the tail of the vDataFrame.

    Parameters
    ----------
    limit: int, optional
        Number of elements to display.

    Returns
    -------
    TableSample
        An object containing the result. For more information, see
        utilities.TableSample.

    See Also
    --------
    vDataFrame.head : Returns the vDataFrame head.
        """
        return self.iloc(limit=limit, offset=-1)

    @save_verticapy_logs
    def select(self, columns: SQLColumns):
        """
    Returns a copy of the vDataFrame with only the selected vDataColumns.

    Parameters
    ----------
    columns: SQLColumns
        List of the vDataColumns to select. It can also be customized expressions.

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
        for i in range(len(columns)):
            column = self._format_colnames(columns[i], raise_error=False)
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
        query = f"SELECT  {', '.join(columns)} FROM {self._genSQL()}"
        return self._new_vdataframe(query)


class vDCRead:
    def __getitem__(self, index):
        if isinstance(index, slice):
            assert index.step in (1, None), ValueError(
                "vDataColumn doesn't allow slicing having steps different than 1."
            )
            index_stop = index.stop
            index_start = index.start
            if not (isinstance(index_start, int)):
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
                elem_to_select = f"{self._alias}[{index_start_str}:{index_stop_str}]"
                elem_to_select = elem_to_select.replace("{}", self._alias)
                new_alias = quote_ident(
                    f"{self._alias[1:-1]}.{index_start}:{index_stop}"
                )
                query = f"""
                    (SELECT 
                        {elem_to_select} AS {new_alias} 
                    FROM {self._parent._genSQL()}) VERTICAPY_SUBTABLE"""
                vcol = self._parent._new_vdataframe(query)[new_alias]
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
                        {self._alias} 
                    FROM {self._parent._genSQL()}
                    {self._parent._get_last_order_by()} 
                    OFFSET {index_start} {limit}"""
                return self._parent._new_vdataframe(query)
        elif isinstance(index, int):
            if self.isarray():
                vertica_version(condition=[9, 3, 0])
                elem_to_select = f"{self._alias}[{index}]"
                new_alias = quote_ident(f"{self._alias[1:-1]}.{index}")
                query = f"""
                    SELECT 
                        {elem_to_select} AS {new_alias} 
                    FROM {self._parent._genSQL()}"""
                vcol = self._parent._new_vdataframe(query)[new_alias]
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
                            {self._alias}{cast} 
                        FROM {self._parent._genSQL()}
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
                elem_to_select = f"MAPLOOKUP({self._alias}, '{index_str}')"
                init_transf = f"MAPLOOKUP({self._init_transf}, '{index_str}')"
            else:
                vertica_version(condition=[10, 0, 0])
                elem_to_select = f"{self._alias}.{quote_ident(index)}"
                init_transf = f"{self._init_transf}.{quote_ident(index)}"
            query = f"""
                SELECT 
                    {elem_to_select} AS {quote_ident(index)} 
                FROM {self._parent._genSQL()}"""
            vcol = self._parent._new_vdataframe(query)[index]
            vcol._init_transf = init_transf
            return vcol
        else:
            return getattr(self, index)

    def __repr__(self):
        return self.head(limit=conf.get_option("max_rows")).__repr__()

    def _repr_html_(self):
        return self.head(limit=conf.get_option("max_rows"))._repr_html_()

    def head(self, limit: int = 5):
        """
    Returns the head of the vDataColumn.

    Parameters
    ----------
    limit: int, optional
        Number of elements to display.

    Returns
    -------
    TableSample
        An object containing the result. For more information, see
        utilities.TableSample.

    See Also
    --------
    vDataFrame[].tail : Returns the a part of the vDataColumn.
        """
        return self.iloc(limit=limit)

    def iloc(self, limit: int = 5, offset: int = 0):
        """
    Returns a part of the vDataColumn (delimited by an offset and a limit).

    Parameters
    ----------
    limit: int, optional
        Number of elements to display.
    offset: int, optional
        Number of elements to skip.

    Returns
    -------
    TableSample
        An object containing the result. For more information, see
        utilities.TableSample.

    See Also
    --------
    vDataFrame[].head : Returns the head of the vDataColumn.
    vDataFrame[].tail : Returns the tail of the vDataColumn.
        """
        if offset < 0:
            offset = max(0, self._parent.shape()[0] - limit)
        title = f"Reads {self._alias}."
        alias_sql_repr = to_varchar(self.category(), self._alias)
        tail = TableSample.read_sql(
            query=f"""
                SELECT 
                    {alias_sql_repr} AS {self._alias} 
                FROM {self._parent._genSQL()}
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
    def nlargest(self, n: int = 10):
        """
    Returns the n largest vDataColumn elements.

    Parameters
    ----------
    n: int, optional
        Offset.

    Returns
    -------
    TableSample
        An object containing the result. For more information, see
        utilities.TableSample.

    See Also
    --------
    vDataFrame[].nsmallest : Returns the n smallest elements in the vDataColumn.
        """
        query = f"""
            SELECT 
                * 
            FROM {self._parent._genSQL()} 
            WHERE {self._alias} IS NOT NULL 
            ORDER BY {self._alias} DESC LIMIT {n}"""
        title = f"Reads {self._alias} {n} largest elements."
        return TableSample.read_sql(
            query,
            title=title,
            sql_push_ext=self._parent._vars["sql_push_ext"],
            symbol=self._parent._vars["symbol"],
        )

    @save_verticapy_logs
    def nsmallest(self, n: int = 10):
        """
    Returns the n smallest elements in the vDataColumn.

    Parameters
    ----------
    n: int, optional
        Offset.

    Returns
    -------
    TableSample
        An object containing the result. For more information, see
        utilities.TableSample.

    See Also
    --------
    vDataFrame[].nlargest : Returns the n largest vDataColumn elements.
        """
        return TableSample.read_sql(
            f"""
            SELECT 
                * 
            FROM {self._parent._genSQL()} 
            WHERE {self._alias} IS NOT NULL 
            ORDER BY {self._alias} ASC LIMIT {n}""",
            title=f"Reads {n} {self._alias} smallest elements.",
            sql_push_ext=self._parent._vars["sql_push_ext"],
            symbol=self._parent._vars["symbol"],
        )

    def tail(self, limit: int = 5):
        """
    Returns the tail of the vDataColumn.

    Parameters
    ----------
    limit: int, optional
        Number of elements to display.

    Returns
    -------
    TableSample
        An object containing the result. For more information, see
        utilities.TableSample.

    See Also
    --------
    vDataFrame[].head : Returns the head of the vDataColumn.
        """
        return self.iloc(limit=limit, offset=-1)
