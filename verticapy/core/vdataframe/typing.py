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
from verticapy._utils._collect import save_verticapy_logs
from verticapy._utils._sql import _executeSQL
from verticapy.core.tablesample import tablesample


class vDFTYPING:
    @save_verticapy_logs
    def astype(self, dtype: dict):
        """
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
        for column in dtype:
            self[self.format_colnames(column)].astype(dtype=dtype[column])
        return self

    @save_verticapy_logs
    def bool_to_int(self):
        """
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

    def catcol(self, max_cardinality: int = 12):
        """
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
        # -#
        columns = []
        for column in self.get_columns():
            if (self[column].category() == "int") and not (self[column].isbool()):
                is_cat = _executeSQL(
                    query=f"""
                        SELECT 
                            /*+LABEL('vDataframe.catcol')*/ 
                            (APPROXIMATE_COUNT_DISTINCT({column}) < {max_cardinality}) 
                        FROM {self.__genSQL__()}""",
                    title="Looking at columns with low cardinality.",
                    method="fetchfirstelem",
                    sql_push_ext=self._VERTICAPY_VARIABLES_["sql_push_ext"],
                    symbol=self._VERTICAPY_VARIABLES_["symbol"],
                )
            elif self[column].category() == "float":
                is_cat = False
            else:
                is_cat = True
            if is_cat:
                columns += [column]
        return columns

    def datecol(self):
        """
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

    @save_verticapy_logs
    def dtypes(self):
        """
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

    def numcol(self, exclude_columns: list = []):
        """
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


class vDCTYPING:
    @save_verticapy_logs
    def astype(self, dtype: Union[str, type]):
        """
    Converts the vColumn to the input type.

    Parameters
    ----------
    dtype: str or Python data type
        New type. One of the following values:
        'json': Converts to a JSON string.
        'array': Converts to an array.
        'vmap': Converts to a VMap. If converting a delimited string, you can 
        add the header_names as follows: dtype = 'vmap(age,name,date)', where 
        the header_names are age, name, and date.

    Returns
    -------
    vDataFrame
        self.parent

    See Also
    --------
    vDataFrame.astype : Converts the vColumns to the input type.
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
                        {self.alias} 
                    FROM {self.parent.__genSQL__()} 
                    ORDER BY LENGTH({self.alias}) DESC 
                    LIMIT 1"""
                biggest_str = _executeSQL(
                    query, title="getting the biggest string", method="fetchfirstelem",
                )
                biggest_str = biggest_str.strip()
                from verticapy.sql.parsers.csv import guess_sep

                sep = guess_sep(biggest_str)
                if str(dtype).startswith("vmap"):
                    if len(biggest_str) > 2 and (
                        (biggest_str[0] == "{" and biggest_str[-1] == "}")
                    ):
                        transformation_2 = """MAPJSONEXTRACTOR({} 
                                                    USING PARAMETERS flatten_maps=false)"""
                    else:
                        header_names = ""
                        if len(dtype) > 4 and dtype[:5] == "vmap(" and dtype[-1] == ")":
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
                        (biggest_str[0] == "(" and biggest_str[-1] == ")")
                        or (biggest_str[0] == "{" and biggest_str[-1] == "}")
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
            elif (
                dtype[0:7] == "varchar" or dtype[0:4] == "char"
            ) and self.category() == "vmap":
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
            transformation = (transformation_2.format(self.alias), transformation_2)
            query = f"""
                SELECT 
                    /*+LABEL('vColumn.astype')*/ 
                    {transformation[0]} AS {self.alias} 
                FROM {self.parent.__genSQL__()} 
                WHERE {self.alias} IS NOT NULL 
                LIMIT 20"""
            _executeSQL(
                query,
                title="Testing the Type casting.",
                sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
                symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
            )
            self.transformations += [
                (transformation[1], dtype, to_category(ctype=dtype),)
            ]
            self.parent.__add_to_history__(
                f"[AsType]: The vColumn {self.alias} was converted to {dtype}."
            )
            return self.parent
        except Exception as e:
            raise ConversionError(
                f"{e}\nThe vColumn {self.alias} can not be converted to {dtype}"
            )

    def category(self):
        """
    Returns the category of the vColumn. The category will be one of the following:
    date / int / float / text / binary / spatial / uuid / undefined

    Returns
    -------
    str
        vColumn category.

    See Also
    --------
    vDataFrame[].ctype : Returns the vColumn database type.
        """
        return self.transformations[-1][2]

    def ctype(self):
        """
    Returns the vColumn DB type.

    Returns
    -------
    str
        vColumn DB type.
        """
        return self.transformations[-1][1].lower()

    dtype = ctype

    def isarray(self):
        """
    Returns True if the vColumn is an array, False otherwise.

    Returns
    -------
    bool
        True if the vColumn is an array.
        """
        return self.ctype()[0:5].lower() == "array"

    def isbool(self):
        """
    Returns True if the vColumn is boolean, False otherwise.

    Returns
    -------
    bool
        True if the vColumn is boolean.

    See Also
    --------
    vDataFrame[].isdate : Returns True if the vColumn category is date.
    vDataFrame[].isnum  : Returns True if the vColumn is numerical.
        """
        return self.ctype()[0:4] == "bool"

    def isdate(self):
        """
    Returns True if the vColumn category is date, False otherwise.

    Returns
    -------
    bool
        True if the vColumn category is date.

    See Also
    --------
    vDataFrame[].isbool : Returns True if the vColumn is boolean.
    vDataFrame[].isnum  : Returns True if the vColumn is numerical.
        """
        return self.category() == "date"

    def isnum(self):
        """
    Returns True if the vColumn is numerical, False otherwise.

    Returns
    -------
    bool
        True if the vColumn is numerical.

    See Also
    --------
    vDataFrame[].isbool : Returns True if the vColumn is boolean.
    vDataFrame[].isdate : Returns True if the vColumn category is date.
        """
        return self.category() in ("float", "int")

    def isvmap(self):
        """
    Returns True if the vColumn category is VMap, False otherwise.

    Returns
    -------
    bool
        True if the vColumn category is VMap.
        """
        return self.category() == "vmap" or isvmap(
            column=self.alias, expr=self.parent.__genSQL__()
        )
