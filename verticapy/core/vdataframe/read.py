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
from typing import Union
from collections.abc import Iterable

from verticapy._config.config import ISNOTEBOOK, OPTIONS
from verticapy._utils._cast import to_varchar
from verticapy._utils._collect import save_verticapy_logs
from verticapy._utils._sql._format import quote_ident
from verticapy._utils._sql._execute import _executeSQL
from verticapy._version import vertica_version

from verticapy.core.str_sql.base import str_sql

from verticapy.sql.read import readSQL, to_tablesample

if ISNOTEBOOK:
    from IPython.display import HTML, display


class vDFREAD:
    def __iter__(self):
        columns = self.get_columns()
        return (col for col in columns)

    def __getitem__(self, index):
        from verticapy.core.vdataframe.base import vDataFrame, vDataColumn

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
            return vDataFrame(query)

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
                sql_push_ext=self._VARS["sql_push_ext"],
                symbol=self._VARS["symbol"],
            )

        elif isinstance(index, (str, str_sql)):
            is_sql = False
            if isinstance(index, vDataColumn):
                index = index._ALIAS
            elif isinstance(index, str_sql):
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
        if self._VARS["sql_magic_result"] and (
            self._VARS["main_relation"][-10:] == "VSQL_MAGIC"
        ):
            return readSQL(
                self._VARS["main_relation"][1:-12],
                OPTIONS["time_on"],
                OPTIONS["max_rows"],
            ).__repr__()
        max_rows = self._VARS["max_rows"]
        if max_rows <= 0:
            max_rows = OPTIONS["max_rows"]
        return self.head(limit=max_rows).__repr__()

    def _repr_html_(self, interactive=False):
        if self._VARS["sql_magic_result"] and (
            self._VARS["main_relation"][-10:] == "VSQL_MAGIC"
        ):
            self._VARS["sql_magic_result"] = False
            return readSQL(
                self._VARS["main_relation"][1:-12],
                OPTIONS["time_on"],
                OPTIONS["max_rows"],
            )._repr_html_(interactive)
        max_rows = self._VARS["max_rows"]
        if max_rows <= 0:
            max_rows = OPTIONS["max_rows"]
        return self.head(limit=max_rows)._repr_html_(interactive)

    def idisplay(self):
        """This method displays the interactive table. It is used when 
        you don't want to activate interactive table for all vDataFrames."""
        return display(HTML(self.copy()._repr_html_(interactive=True)))

    def get_columns(self, exclude_columns: Union[str, list] = []):
        """
    Returns the vDataFrame vDataColumns.

    Parameters
    ----------
    exclude_columns: str / list, optional
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
        columns = [elem for elem in self._VARS["columns"]]
        result = []
        exclude_columns = [elem for elem in exclude_columns]
        exclude_columns += [elem for elem in self._VARS["exclude_columns"]]
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
        result = to_tablesample(
            query=f"""
                SELECT 
                    {', '.join(all_columns)} 
                FROM {self._genSQL()}
                {self._get_last_order_by()} 
                LIMIT {limit} OFFSET {offset}""",
            title=title,
            max_columns=self._VARS["max_columns"],
            sql_push_ext=self._VARS["sql_push_ext"],
            symbol=self._VARS["symbol"],
        )
        pre_comp = self._get_catalog_value("VERTICAPY_COUNT")
        if pre_comp != "VERTICAPY_NOT_PRECOMPUTED":
            result.count = pre_comp
        elif OPTIONS["count_on"]:
            result.count = self.shape()[0]
        result.offset = offset
        columns = self.get_columns()
        all_percent = True
        for column in columns:
            if not ("percent" in self[column]._CATALOG):
                all_percent = False
        all_percent = (all_percent or (OPTIONS["percent_bar"] == True)) and (
            OPTIONS["percent_bar"] != False
        )
        if all_percent:
            percent = self.aggregate(["percent"], columns).transpose().values
        for column in result.values:
            result.dtype[column] = self[column].ctype()
            if all_percent:
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
        self._VARS["count"] = _executeSQL(
            query=f"""
                SELECT 
                    /*+LABEL('vDataframe.shape')*/ COUNT(*) 
                FROM {self._genSQL()} LIMIT 1
            """,
            title="Computing the total number of elements (COUNT(*))",
            method="fetchfirstelem",
            sql_push_ext=self._VARS["sql_push_ext"],
            symbol=self._VARS["symbol"],
        )
        return (self._VARS["count"], m)

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
    def select(self, columns: Union[str, list]):
        """
    Returns a copy of the vDataFrame with only the selected vDataColumns.

    Parameters
    ----------
    columns: str / list
        List of the vDataColumns to select. It can also be customized expressions.

    Returns
    -------
    vDataFrame
        object with only the selected columns.

    See Also
    --------
    vDataFrame.search : Searches the elements which matches with the input conditions.
        """
        from verticapy.core.vdataframe.base import vDataFrame

        if isinstance(columns, str):
            columns = [columns]
        for i in range(len(columns)):
            column = self._format_colnames(columns[i], raise_error=False)
            if column:
                dtype = ""
                if self._VARS["isflex"]:
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
        return vDataFrame(query)


class vDCREAD:
    def __getitem__(self, index):
        from verticapy.core.vdataframe.base import vDataFrame

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
                elem_to_select = f"{self._ALIAS}[{index_start_str}:{index_stop_str}]"
                elem_to_select = elem_to_select.replace("{}", self._ALIAS)
                new_alias = quote_ident(
                    f"{self._ALIAS[1:-1]}.{index_start}:{index_stop}"
                )
                query = f"""
                    (SELECT 
                        {elem_to_select} AS {new_alias} 
                    FROM {self._PARENT._genSQL()}) VERTICAPY_SUBTABLE"""
                vcol = vDataFrame(query)[new_alias]
                vcol._TRANSF[-1] = (
                    new_alias,
                    self.ctype(),
                    self.category(),
                )
                vcol._INIT_TRANSF = (
                    f"{self._INIT_TRANSF}[{index_start_str}:{index_stop_str}]"
                )
                vcol._INIT_TRANSF = vcol._INIT_TRANSF.replace("{}", self._INIT_TRANSF)
                return vcol
            else:
                if index_start < 0:
                    index_start += self._PARENT.shape()[0]
                if isinstance(index_stop, int):
                    if index_stop < 0:
                        index_stop += self._PARENT.shape()[0]
                    limit = index_stop - index_start
                    if limit <= 0:
                        limit = 0
                    limit = f" LIMIT {limit}"
                else:
                    limit = ""
                query = f"""
                    SELECT 
                        {self._ALIAS} 
                    FROM {self._PARENT._genSQL()}
                    {self._PARENT._get_last_order_by()} 
                    OFFSET {index_start} {limit}"""
                return vDataFrame(query)
        elif isinstance(index, int):
            if self.isarray():
                vertica_version(condition=[9, 3, 0])
                elem_to_select = f"{self._ALIAS}[{index}]"
                new_alias = quote_ident(f"{self._ALIAS[1:-1]}.{index}")
                query = f"""
                    SELECT 
                        {elem_to_select} AS {new_alias} 
                    FROM {self._PARENT._genSQL()}"""
                vcol = vDataFrame(query)[new_alias]
                vcol._INIT_TRANSF = f"{self._INIT_TRANSF}[{index}]"
                return vcol
            else:
                cast = "::float" if self.category() == "float" else ""
                if index < 0:
                    index += self._PARENT.shape()[0]
                return _executeSQL(
                    query=f"""
                        SELECT 
                            /*+LABEL('vDataColumn.__getitem__')*/ 
                            {self._ALIAS}{cast} 
                        FROM {self._PARENT._genSQL()}
                        {self._PARENT._get_last_order_by()} 
                        OFFSET {index} 
                        LIMIT 1""",
                    title="Getting the vDataColumn element.",
                    method="fetchfirstelem",
                    sql_push_ext=self._PARENT._VARS["sql_push_ext"],
                    symbol=self._PARENT._VARS["symbol"],
                )
        elif isinstance(index, str):
            if self.category() == "vmap":
                index_str = index.replace("'", "''")
                elem_to_select = f"MAPLOOKUP({self._ALIAS}, '{index_str}')"
                init_transf = f"MAPLOOKUP({self._INIT_TRANSF}, '{index_str}')"
            else:
                vertica_version(condition=[10, 0, 0])
                elem_to_select = f"{self._ALIAS}.{quote_ident(index)}"
                init_transf = f"{self._INIT_TRANSF}.{quote_ident(index)}"
            query = f"""
                SELECT 
                    {elem_to_select} AS {quote_ident(index)} 
                FROM {self._PARENT._genSQL()}"""
            vcol = vDataFrame(query)[index]
            vcol._INIT_TRANSF = init_transf
            return vcol
        else:
            return getattr(self, index)

    def __repr__(self):
        return self.head(limit=OPTIONS["max_rows"]).__repr__()

    def _repr_html_(self):
        return self.head(limit=OPTIONS["max_rows"])._repr_html_()

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
            offset = max(0, self._PARENT.shape()[0] - limit)
        title = f"Reads {self._ALIAS}."
        alias_sql_repr = to_varchar(self.category(), self._ALIAS)
        tail = to_tablesample(
            query=f"""
                SELECT 
                    {alias_sql_repr} AS {self._ALIAS} 
                FROM {self._PARENT._genSQL()}
                {self._PARENT._get_last_order_by()} 
                LIMIT {limit} 
                OFFSET {offset}""",
            title=title,
            sql_push_ext=self._PARENT._VARS["sql_push_ext"],
            symbol=self._PARENT._VARS["symbol"],
        )
        tail.count = self._PARENT.shape()[0]
        tail.offset = offset
        tail.dtype[self._ALIAS] = self.ctype()
        tail.name = self._ALIAS
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
            FROM {self._PARENT._genSQL()} 
            WHERE {self._ALIAS} IS NOT NULL 
            ORDER BY {self._ALIAS} DESC LIMIT {n}"""
        title = f"Reads {self._ALIAS} {n} largest elements."
        return to_tablesample(
            query,
            title=title,
            sql_push_ext=self._PARENT._VARS["sql_push_ext"],
            symbol=self._PARENT._VARS["symbol"],
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
        return to_tablesample(
            f"""
            SELECT 
                * 
            FROM {self._PARENT._genSQL()} 
            WHERE {self._ALIAS} IS NOT NULL 
            ORDER BY {self._ALIAS} ASC LIMIT {n}""",
            title=f"Reads {n} {self._ALIAS} smallest elements.",
            sql_push_ext=self._PARENT._VARS["sql_push_ext"],
            symbol=self._PARENT._VARS["symbol"],
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
