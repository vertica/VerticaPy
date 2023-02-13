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
from verticapy._utils._cast import to_varchar
from verticapy.sql.read import to_tablesample
from verticapy._utils._sql import _executeSQL


class vDFREAD:
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
