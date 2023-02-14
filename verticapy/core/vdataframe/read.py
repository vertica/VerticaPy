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
from verticapy._utils._cast import to_varchar
from verticapy.sql.read import to_tablesample
from verticapy._utils._sql import _executeSQL
from verticapy._utils._collect import save_verticapy_logs
from verticapy._config.config import OPTIONS
from verticapy.sql.read import readSQL, vDataFrameSQL
from verticapy.core.str_sql import str_sql
from verticapy.sql._utils._format import quote_ident
from verticapy._version import vertica_version


# Jupyter - Optional
try:
    from IPython.display import HTML, display
except:
    pass


class vDFREAD:
    def __iter__(self):
        columns = self.get_columns()
        return (col for col in columns)

    def __getitem__(self, index):
        from verticapy.core.vcolumn import vColumn

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

    def idisplay(self):
        """This method displays the interactive table. It is used when 
        you don't want to activate interactive table for all vDataFrames."""
        return display(HTML(self.copy()._repr_html_(interactive=True)))

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
    def select(self, columns: Union[str, list]):
        """
    Returns a copy of the vDataFrame with only the selected vColumns.

    Parameters
    ----------
    columns: str / list
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
        for i in range(len(columns)):
            column = self.format_colnames(columns[i], raise_error=False)
            if column:
                dtype = ""
                if self._VERTICAPY_VARIABLES_["isflex"]:
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
        table = f"""
            (SELECT 
                {', '.join(columns)} 
            FROM {self.__genSQL__()}) VERTICAPY_SUBTABLE"""
        return self.__vDataFrameSQL__(
            table, self._VERTICAPY_VARIABLES_["input_relation"], ""
        )


class vDCREAD:
    def __getitem__(self, index):
        if isinstance(index, slice):
            assert index.step in (1, None), ValueError(
                "vColumn doesn't allow slicing having steps different than 1."
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
                elem_to_select = f"{self.alias}[{index_start_str}:{index_stop_str}]"
                elem_to_select = elem_to_select.replace("{}", self.alias)
                new_alias = quote_ident(
                    f"{self.alias[1:-1]}.{index_start}:{index_stop}"
                )
                query = f"""
                    (SELECT 
                        {elem_to_select} AS {new_alias} 
                    FROM {self.parent.__genSQL__()}) VERTICAPY_SUBTABLE"""
                vcol = vDataFrameSQL(query)[new_alias]
                vcol.transformations[-1] = (
                    new_alias,
                    self.ctype(),
                    self.category(),
                )
                vcol.init_transf = (
                    f"{self.init_transf}[{index_start_str}:{index_stop_str}]"
                )
                vcol.init_transf = vcol.init_transf.replace("{}", self.init_transf)
                return vcol
            else:
                if index_start < 0:
                    index_start += self.parent.shape()[0]
                if isinstance(index_stop, int):
                    if index_stop < 0:
                        index_stop += self.parent.shape()[0]
                    limit = index_stop - index_start
                    if limit <= 0:
                        limit = 0
                    limit = f" LIMIT {limit}"
                else:
                    limit = ""
                query = f"""
                    (SELECT 
                        {self.alias} 
                    FROM {self.parent.__genSQL__()}
                    {self.parent.__get_last_order_by__()} 
                    OFFSET {index_start}
                    {limit}) VERTICAPY_SUBTABLE"""
                return vDataFrameSQL(query)
        elif isinstance(index, int):
            if self.isarray():
                vertica_version(condition=[9, 3, 0])
                elem_to_select = f"{self.alias}[{index}]"
                new_alias = quote_ident(f"{self.alias[1:-1]}.{index}")
                query = f"""
                    (SELECT 
                        {elem_to_select} AS {new_alias} 
                    FROM {self.parent.__genSQL__()}) VERTICAPY_SUBTABLE"""
                vcol = vDataFrameSQL(query)[new_alias]
                vcol.init_transf = f"{self.init_transf}[{index}]"
                return vcol
            else:
                cast = "::float" if self.category() == "float" else ""
                if index < 0:
                    index += self.parent.shape()[0]
                return _executeSQL(
                    query=f"""
                        SELECT 
                            /*+LABEL('vColumn.__getitem__')*/ 
                            {self.alias}{cast} 
                        FROM {self.parent.__genSQL__()}
                        {self.parent.__get_last_order_by__()} 
                        OFFSET {index} 
                        LIMIT 1""",
                    title="Getting the vColumn element.",
                    method="fetchfirstelem",
                    sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
                    symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
                )
        elif isinstance(index, str):
            if self.category() == "vmap":
                index_str = index.replace("'", "''")
                elem_to_select = f"MAPLOOKUP({self.alias}, '{index_str}')"
                init_transf = f"MAPLOOKUP({self.init_transf}, '{index_str}')"
            else:
                vertica_version(condition=[10, 0, 0])
                elem_to_select = f"{self.alias}.{quote_ident(index)}"
                init_transf = f"{self.init_transf}.{quote_ident(index)}"
            query = f"""
                (SELECT 
                    {elem_to_select} AS {quote_ident(index)} 
                FROM {self.parent.__genSQL__()}) VERTICAPY_SUBTABLE"""
            vcol = vDataFrameSQL(query)[index]
            vcol.init_transf = init_transf
            return vcol
        else:
            return getattr(self, index)

    def __repr__(self):
        return self.head(limit=OPTIONS["max_rows"]).__repr__()

    def _repr_html_(self):
        return self.head(limit=OPTIONS["max_rows"])._repr_html_()

    def head(self, limit: int = 5):
        """
    Returns the head of the vColumn.

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
    vDataFrame[].tail : Returns the a part of the vColumn.
        """
        return self.iloc(limit=limit)

    def iloc(self, limit: int = 5, offset: int = 0):
        """
    Returns a part of the vColumn (delimited by an offset and a limit).

    Parameters
    ----------
    limit: int, optional
        Number of elements to display.
    offset: int, optional
        Number of elements to skip.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame[].head : Returns the head of the vColumn.
    vDataFrame[].tail : Returns the tail of the vColumn.
        """
        if offset < 0:
            offset = max(0, self.parent.shape()[0] - limit)
        title = f"Reads {self.alias}."
        alias_sql_repr = to_varchar(self.category(), self.alias)
        tail = to_tablesample(
            query=f"""
                SELECT 
                    {alias_sql_repr} AS {self.alias} 
                FROM {self.parent.__genSQL__()}
                {self.parent.__get_last_order_by__()} 
                LIMIT {limit} 
                OFFSET {offset}""",
            title=title,
            sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
            symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
        )
        tail.count = self.parent.shape()[0]
        tail.offset = offset
        tail.dtype[self.alias] = self.ctype()
        tail.name = self.alias
        return tail

    @save_verticapy_logs
    def nlargest(self, n: int = 10):
        """
    Returns the n largest vColumn elements.

    Parameters
    ----------
    n: int, optional
        Offset.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame[].nsmallest : Returns the n smallest elements in the vColumn.
        """
        query = f"""
            SELECT 
                * 
            FROM {self.parent.__genSQL__()} 
            WHERE {self.alias} IS NOT NULL 
            ORDER BY {self.alias} DESC LIMIT {n}"""
        title = f"Reads {self.alias} {n} largest elements."
        return to_tablesample(
            query,
            title=title,
            sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
            symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
        )

    @save_verticapy_logs
    def nsmallest(self, n: int = 10):
        """
    Returns the n smallest elements in the vColumn.

    Parameters
    ----------
    n: int, optional
        Offset.

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.

    See Also
    --------
    vDataFrame[].nlargest : Returns the n largest vColumn elements.
        """
        return to_tablesample(
            f"""
            SELECT 
                * 
            FROM {self.parent.__genSQL__()} 
            WHERE {self.alias} IS NOT NULL 
            ORDER BY {self.alias} ASC LIMIT {n}""",
            title=f"Reads {n} {self.alias} smallest elements.",
            sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
            symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
        )

    def tail(self, limit: int = 5):
        """
    Returns the tail of the vColumn.

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
    vDataFrame[].head : Returns the head of the vColumn.
        """
        return self.iloc(limit=limit, offset=-1)
