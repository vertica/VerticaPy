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

from verticapy._typing import SQLColumns
from verticapy._utils._parsers import guess_sep
from verticapy._utils._sql._cast import to_sql_dtype, to_category
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import clean_query
from verticapy._utils._sql._sys import _executeSQL
from verticapy._utils._sql._vertica_version import vertica_version

from verticapy.errors import ConversionError

from verticapy.core.tablesample.base import TableSample

from verticapy.core.vdataframe._read import vDFRead, vDCRead

from verticapy.sql.flex import isvmap

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame


class vDFTyping(vDFRead):
    @save_verticapy_logs
    def astype(self, dtype: dict) -> "vDataFrame":
        """
        Converts the vDataColumns to the input types.

        Parameters
        ----------
        dtype: dict
            Dictionary of the different types. Each key
            of   the   dictionary  must   represent   a
            vDataColumn. The dictionary must be similar
            to the following:
            {"column1": "type1", ... "columnk": "typek"}

        Returns
        -------
        vDataFrame
            self
        """
        for column in dtype:
            self[self.format_colnames(column)].astype(dtype=dtype[column])
        return self

    @save_verticapy_logs
    def bool_to_int(self) -> "vDataFrame":
        """
        Converts all booleans vDataColumns to integers.

        Returns
        -------
        vDataFrame
            self
        """
        columns = self.get_columns()
        for column in columns:
            if self[column].isbool():
                self[column].astype("int")
        return self

    def catcol(self, max_cardinality: int = 12) -> list:
        """
        Returns the vDataFrame categorical vDataColumns.

        Parameters
        ----------
        max_cardinality: int, optional
            Maximum number of unique values to consider
            integer vDataColumns as categorical.

        Returns
        -------
        List
            List of the categorical vDataColumns names.
        """
        columns = []
        for column in self.get_columns():
            if (self[column].category() == "int") and not self[column].isbool():
                is_cat = _executeSQL(
                    query=f"""
                        SELECT 
                            /*+LABEL('vDataframe.catcol')*/ 
                            (APPROXIMATE_COUNT_DISTINCT({column}) < {max_cardinality}) 
                        FROM {self}""",
                    title="Looking at columns with low cardinality.",
                    method="fetchfirstelem",
                    sql_push_ext=self._vars["sql_push_ext"],
                    symbol=self._vars["symbol"],
                )
            elif self[column].category() == "float":
                is_cat = False
            else:
                is_cat = True
            if is_cat:
                columns += [column]
        return columns

    def datecol(self) -> list:
        """
        Returns a list of the vDataColumns of type
        date in the vDataFrame.

        Returns
        -------
        List
            List of all vDataColumns of type date.
        """
        columns = []
        cols = self.get_columns()
        for column in cols:
            if self[column].isdate():
                columns += [column]
        return columns

    @save_verticapy_logs
    def dtypes(self) -> TableSample:
        """
        Returns the different vDataColumns types.

        Returns
        -------
        TableSample
            result.
        """
        values = {"index": [], "dtype": []}
        for column in self.get_columns():
            values["index"] += [column]
            values["dtype"] += [self[column].ctype()]
        return TableSample(values)

    def numcol(self, exclude_columns: Optional[SQLColumns] = None) -> list:
        """
        Returns a list of names of the numerical vDataColumns
        in the vDataFrame.

        Parameters
        ----------
        exclude_columns: SQLColumns, optional
            List  of the  vDataColumns names to exclude  from
            the final list.

        Returns
        -------
        List
            List of numerical vDataColumns names.
        """
        columns, cols = [], self.get_columns(exclude_columns=exclude_columns)
        for column in cols:
            if self[column].isnum():
                columns += [column]
        return columns


class vDCTyping(vDCRead):
    @save_verticapy_logs
    def astype(self, dtype: Union[str, type]) -> "vDataFrame":
        """
        Converts the vDataColumn to the input type.

        Parameters
        ----------
        dtype: str or Python data type
            New type. One of the following values:
                'json' : Converts to a JSON string.
                'array': Converts to an array.
                'vmap' : Converts to a VMap.  If converting a
                         delimited  string,  you can add  the
                         header_names  as   follows:  dtype =
                         'vmap(age,name,date)',   where   the
                         header_names are age, name, and date.

        Returns
        -------
        vDataFrame
            self._parent
        """
        dtype = to_sql_dtype(dtype)
        try:
            if (
                dtype == "array" or str(dtype).startswith("vmap")
            ) and self.category() == "text":
                if dtype == "array":
                    vertica_version(condition=[10, 0, 0])
                query = f"""
                    SELECT 
                        {self} 
                    FROM {self._parent} 
                    ORDER BY LENGTH({self}) DESC 
                    LIMIT 1"""
                biggest_str = _executeSQL(
                    query,
                    title="getting the biggest string",
                    method="fetchfirstelem",
                )
                biggest_str = biggest_str.strip()
                sep = guess_sep(biggest_str)
                if str(dtype).startswith("vmap"):
                    if len(biggest_str) > 2 and (
                        (biggest_str.startswith("{") and biggest_str.endswith("}"))
                    ):
                        transformation_2 = """MAPJSONEXTRACTOR({} 
                                                    USING PARAMETERS flatten_maps=false)"""
                    else:
                        header_names = ""
                        if (
                            len(dtype) > 4
                            and dtype.startswith("vmap(")
                            and dtype.endswith(")")
                        ):
                            header_names = f", header_names='{dtype[5:-1]}'"
                        transformation_2 = f"""MAPDELIMITEDEXTRACTOR({{}} 
                                                            USING PARAMETERS 
                                                            delimiter='{sep}'
                                                            {header_names})"""
                    dtype = "vmap"
                elif dtype == "array":
                    if biggest_str.replace(" ", "").count(sep + sep) > 0:
                        collection_null_element = ", collection_null_element=''"
                    else:
                        collection_null_element = ""
                    if len(biggest_str) > 2 and (
                        (biggest_str.startswith("(") and biggest_str.endswith(")"))
                        or (biggest_str.startswith("{") and biggest_str.endswith("}"))
                    ):
                        collection_open = f", collection_open='{biggest_str[0]}'"
                        collection_close = f", collection_close='{biggest_str[-1]}'"
                    else:
                        collection_open, collection_close = "", ""
                    transformation_2 = f"""
                        STRING_TO_ARRAY({{}} 
                                        USING PARAMETERS 
                                        collection_delimiter='{sep}'
                                        {collection_open}
                                        {collection_close}
                                        {collection_null_element})"""
            elif dtype.startswith(("char", "varchar")) and self.category() == "vmap":
                transformation_2 = f"""MAPTOSTRING({{}} 
                                                   USING PARAMETERS 
                                                   canonical_json=false)::{dtype}"""
            elif dtype == "json":
                if self.category() == "vmap":
                    transformation_2 = (
                        "MAPTOSTRING({} USING PARAMETERS canonical_json=true)"
                    )
                else:
                    vertica_version(condition=[10, 1, 0])
                    transformation_2 = "TO_JSON({})"
                dtype = "varchar"
            else:
                transformation_2 = f"{{}}::{dtype}"
            transformation_2 = clean_query(transformation_2)
            transformation = (transformation_2.format(self._alias), transformation_2)
            query = f"""
                SELECT 
                    /*+LABEL('vDataColumn.astype')*/ 
                    {transformation[0]} AS {self} 
                FROM {self._parent} 
                WHERE {self} IS NOT NULL 
                LIMIT 20"""
            _executeSQL(
                query,
                title="Testing the Type casting.",
                sql_push_ext=self._parent._vars["sql_push_ext"],
                symbol=self._parent._vars["symbol"],
            )
            self._transf += [
                (
                    transformation[1],
                    dtype,
                    to_category(ctype=dtype),
                )
            ]
            self._parent._add_to_history(
                f"[AsType]: The vDataColumn {self} was converted to {dtype}."
            )
            return self._parent
        except Exception as e:
            raise ConversionError(
                f"{e}\nThe vDataColumn {self} can not be converted to {dtype}"
            )

    def category(self) -> str:
        """
        Returns the category of the vDataColumn. The category
        will be one of the following:
        date / int / float / text / binary / spatial / uuid
        / undefined

        Returns
        -------
        str
            vDataColumn category.
        """
        return self._transf[-1][2]

    def ctype(self) -> str:
        """
        Returns the vDataColumn DB type.

        Returns
        -------
        str
            vDataColumn DB type.
        """
        return self._transf[-1][1].lower()

    dtype = ctype

    def isarray(self) -> bool:
        """
        Returns True if the vDataColumn is an array,
        False otherwise.

        Returns
        -------
        bool
            True if the vDataColumn is an array.
        """
        return self.ctype()[0:5].lower() == "array"

    def isbool(self) -> bool:
        """
        Returns True if the vDataColumn is boolean,
        False otherwise.

        Returns
        -------
        bool
            True if the vDataColumn is boolean.
        """
        return self.ctype().startswith("bool")

    def isdate(self) -> bool:
        """
        Returns True if the vDataColumn category is date,
        False otherwise.

        Returns
        -------
        bool
            True if the vDataColumn category is date.
        """
        return self.category() == "date"

    def isnum(self) -> bool:
        """
        Returns True if the vDataColumn is numerical,
        False otherwise.

        Returns
        -------
        bool
            True if the vDataColumn is numerical.
        """
        return self.category() in ("float", "int")

    def isvmap(self) -> bool:
        """
        Returns True if the vDataColumn category is VMap,
        False otherwise.

        Returns
        -------
        bool
            True if the vDataColumn category is VMap.
        """
        return self.category() == "vmap" or isvmap(
            column=self._alias, expr=self._parent._genSQL()
        )
