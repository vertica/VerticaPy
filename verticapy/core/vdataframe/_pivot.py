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
from typing import Optional, Union, TYPE_CHECKING

from verticapy._typing import NoneType, SQLColumns, SQLExpression
from verticapy._utils._object import create_new_vdf
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import format_type, quote_ident
from verticapy._utils._sql._merge import gen_coalesce, group_similar_names
from verticapy.errors import EmptyParameter

from verticapy.core.vdataframe._join_union_sort import vDFJoinUnionSort

from verticapy.sql.flex import compute_vmap_keys

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame


class vDFPivot(vDFJoinUnionSort):
    @save_verticapy_logs
    def flat_vmap(
        self,
        vmap_col: Optional[SQLExpression] = None,
        limit: int = 100,
        exclude_columns: Optional[SQLColumns] = None,
    ) -> "vDataFrame":
        """
        Flatten the selected VMap. A new vDataFrame is returned.

        \u26A0 Warning : This  function  might  have  a  long  runtime
                         and can make your  vDataFrame less performant.
                         It makes many calls to the MAPLOOKUP function,
                         which can be slow if your VMap is large.

        Parameters
        ----------
        vmap_col: SQLColumns, optional
            List of VMap columns to flatten.
        limit: int, optional
            Maximum number of keys to consider for each VMap. Only the
            most occurent keys are used.
        exclude_columns: SQLColumns, optional
            List of VMap columns to exclude.

        Returns
        -------
        vDataFrame
            object with the flattened VMaps.
        """
        vmap_col = format_type(vmap_col, dtype=list)
        if not vmap_col:
            vmap_col = []
            all_cols = self.get_columns()
            for col in all_cols:
                if self[col].isvmap():
                    vmap_col += [col]
        exclude_columns = format_type(exclude_columns, dtype=list)
        exclude_columns_final = quote_ident(exclude_columns, lower=True)
        vmap_col_final = []
        for col in vmap_col:
            if quote_ident(col).lower() not in exclude_columns_final:
                vmap_col_final += [col]
        if not vmap_col:
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

    @save_verticapy_logs
    def merge_similar_names(self, skip_word: Union[str, list[str]]) -> "vDataFrame":
        """
        Merges  columns with  similar names.  The function  generates
        a COALESCE  statement that  merges the columns into a  single
        column that excludes  the input words. Note that the order of
        the variables in the COALESCE statement is based on the order
        of the 'get_columns' method.

        Parameters
        ----------
        skip_word: str / list, optional
            List  of words to  exclude  from  the provided column  names.
            For example,     if      two      columns      are     named
            'age.information.phone'  and  'age.phone' AND  skip_word  is
            set  to  ['.information'],  then  the  two  columns are
            merged  together  with  the   following  COALESCE  statement:
            COALESCE("age.phone", "age.information.phone") AS "age.phone"

        Returns
        -------
        vDataFrame
            An object containing the merged element.
        """
        columns = self.get_columns()
        skip_word = format_type(skip_word, dtype=list)
        group_dict = group_similar_names(columns, skip_word=skip_word)
        sql = f"SELECT {gen_coalesce(group_dict)} FROM {self}"
        return create_new_vdf(sql)

    @save_verticapy_logs
    def narrow(
        self,
        index: SQLColumns,
        columns: Optional[SQLColumns] = None,
        col_name: str = "column",
        val_name: str = "value",
    ) -> "vDataFrame":
        """
        Returns the Narrow Table of the vDataFrame using the input
        vDataColumns.

        Parameters
        ----------
        index: SQLColumns
            Index(es) used to identify the Row.
        columns: SQLColumns, optional
            List of the vDataColumns names. If empty, all vDataColumns
            except the index(es) are used.
        col_name: str, optional
            Alias of the vDataColumn  representing the different input
            vDataColumns names as categories.
        val_name: str, optional
            Alias of the vDataColumn  representing the different input
            vDataColumns values.

        Returns
        -------
        vDataFrame
            the narrow table object.
        """
        index, columns = format_type(index, columns, dtype=list, na_out=self.numcol())
        index, columns = self.format_colnames(index, columns)
        for idx in index:
            if idx in columns:
                columns.remove(idx)
        query = []
        all_are_num, all_are_date = True, True
        for column in columns:
            if not self[column].isnum():
                all_are_num = False
            if not self[column].isdate():
                all_are_date = False
        for column in columns:
            conv = ""
            if not all_are_num and not all_are_date:
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
                FROM {self})"""
            ]
        query = " UNION ALL ".join(query)
        return create_new_vdf(query)

    melt = narrow

    @save_verticapy_logs
    def pivot(
        self,
        index: str,
        columns: str,
        values: str,
        aggr: str = "sum",
        prefix: Optional[str] = None,
    ) -> "vDataFrame":
        """
        Returns the Pivot of the vDataFrame using the input aggregation.

        Parameters
        ----------
        index: str
            vDataColumn used to group the elements.
        columns: str
            The vDataColumn used to compute the different categories,
            which then act as the columns in the pivot table.
        values: str
            The vDataColumn whose values populate the new vDataFrame.
        aggr: str, optional
            Aggregation to use on 'values'.  To use complex aggregations,
            you must use braces: {}. For example, to aggregate using the
            aggregation: x -> MAX(x) - MIN(x), write "MAX({}) - MIN({})".
        prefix: str, optional
            The prefix for the pivot table's column names.

        Returns
        -------
        vDataFrame
            the pivot table object.
        """
        if isinstance(prefix, NoneType):
            prefix = ""
        index, columns, values = self.format_colnames(index, columns, values)
        aggr = aggr.upper()
        if "{}" not in aggr:
            aggr += "({})"
        new_cols = self[columns].distinct()
        new_cols_trans = []
        for col in new_cols:
            if isinstance(col, NoneType):
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
                        f"(CASE WHEN {columns} = '{col}' THEN {values} ELSE NULL END)",
                    )
                    + f"AS '{prefix}{col}'"
                ]
        return create_new_vdf(
            f"""
            SELECT 
                {index},
                {", ".join(new_cols_trans)}
            FROM {self}
            GROUP BY 1""",
        )
