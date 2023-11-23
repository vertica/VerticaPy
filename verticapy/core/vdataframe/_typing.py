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

        Examples
        ---------

        We import :py:mod:`verticapy`:

        .. ipython:: python

            import verticapy as vp

        .. hint::

            By assigning an alias to :py:mod:`verticapy`, we mitigate the risk
            of code collisions with other libraries. This precaution is
            necessary because verticapy uses commonly known function names
            like "average" and "median", which can potentially lead to naming
            conflicts. The use of an alias ensures that the functions from
            verticapy are used as intended without interfering with functions
            from other libraries.

        For this example, we will use the Titanic dataset.

        .. code-block:: python

            import verticapy.datasets as vpd

            data = vpd.load_titanic()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_titanic.html

        .. note::

            VerticaPy offers a wide range of sample datasets that are
            ideal for training and testing purposes. You can explore
            the full list of available datasets in the :ref:`api.datasets`,
            which provides detailed information on each dataset
            and how to use them effectively. These datasets are invaluable
            resources for honing your data analysis and machine learning
            skills within the VerticaPy environment.

        .. ipython:: python
            :suppress:

            import verticapy.datasets as vpd

            data = vpd.load_titanic()

        Let's check the data types of various vcolumns.

        .. code-block:: python

            data.dtypes()

        .. ipython:: python
            :suppress:

            res = data.dtypes()
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_typing_astype1.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_typing_astype1.html

        Let's change the data type of few vcolumns.

        .. code-block:: python

            data.astype({"fare": "int", "cabin": "varchar(1)"})

        Let's check the data type of various vcolumns again.

        .. code-block:: python

            data.dtypes()

        .. ipython:: python
            :suppress:

            data.astype({"fare": "int", "cabin": "varchar(1)"})
            res = data.dtypes()
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_typing_astype2.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_typing_astype2.html
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

        Examples
        ---------

        We import :py:mod:`verticapy`:

        .. ipython:: python

            import verticapy as vp

        .. hint::

            By assigning an alias to :py:mod:`verticapy`, we mitigate the risk
            of code collisions with other libraries. This precaution is
            necessary because verticapy uses commonly known function names
            like "average" and "median", which can potentially lead to naming
            conflicts. The use of an alias ensures that the functions from
            verticapy are used as intended without interfering with functions
            from other libraries.

        Let's create a small dataset:

        .. code-block:: python

            data = vp.vDataFrame(
                {
                    "empid": ['1', '2', '3', '4'],
                    "is_temp": [True, False, False, True],
                }
            )
            data

        .. ipython:: python
            :suppress:

            data = vp.vDataFrame(
                {
                    "empid": ['1', '2', '3', '4'],
                    "is_temp": [True, False, False, True],
                }
            )
            res = data
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_typing_booltoint1.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_typing_booltoint1.html

        Let's change the data type from bool to int.

        .. code-block:: python

            data.bool_to_int()

        .. ipython:: python
            :suppress:

            res = data.bool_to_int()
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_typing_booltoint2.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_typing_booltoint2.html
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

        Examples
        ---------

        We import :py:mod:`verticapy`:

        .. ipython:: python

            import verticapy as vp

        .. hint::

            By assigning an alias to :py:mod:`verticapy`, we mitigate the risk
            of code collisions with other libraries. This precaution is
            necessary because verticapy uses commonly known function names
            like "average" and "median", which can potentially lead to naming
            conflicts. The use of an alias ensures that the functions from
            verticapy are used as intended without interfering with functions
            from other libraries.

        For this example, we will use the Titanic dataset.

        .. code-block:: python

            import verticapy.datasets as vpd

            data = vpd.load_titanic()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_titanic.html

        .. note::

            VerticaPy offers a wide range of sample datasets that are
            ideal for training and testing purposes. You can explore
            the full list of available datasets in the :ref:`api.datasets`,
            which provides detailed information on each dataset
            and how to use them effectively. These datasets are invaluable
            resources for honing your data analysis and machine learning
            skills within the VerticaPy environment.

        .. ipython:: python
            :suppress:

            import verticapy.datasets as vpd

            data = vpd.load_titanic()

        Let's check the categorical vcolumns considering maximum
        cardinality as 10.

        .. ipython:: python

            data.catcol(max_cardinality = 10)

        Let's again check the categorical vcolumns considering
        maximum cardinality as 6.

        .. ipython:: python

            data.catcol(max_cardinality = 6)

        Notice that parch and sibsp are not considered because
        their cardinalities are greater than 6.

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.datecol` : Returns all vDataColumns with date-type values.
            | :py:meth:`verticapy.vDataFrame.numcol` : Returns all vDataColumns with numerical values.
            | :py:meth:`verticapy.vDataFrame.get_columns` : Returns all vDataColumns.
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

        Examples
        ---------

        We import :py:mod:`verticapy`:

        .. ipython:: python

            import verticapy as vp

        .. hint::

            By assigning an alias to :py:mod:`verticapy`, we mitigate the risk
            of code collisions with other libraries. This precaution is
            necessary because verticapy uses commonly known function names
            like "average" and "median", which can potentially lead to naming
            conflicts. The use of an alias ensures that the functions from
            verticapy are used as intended without interfering with functions
            from other libraries.

        Let's create a small dataset:

        .. code-block:: python

            data = vp.vDataFrame(
                {
                    "empid": ['1', '2', '3', '4'],
                    "dob": ['1993-01-01', '1988-01-01', '1992-01-01', '1989-01-01'],
                    "doj": ['2022-01-01', '2023-01-01', '2022-01-01', '2023-01-01'],
                    "emp_cat":[933, 945, 723, 799],
                }
            )
            data

        .. ipython:: python
            :suppress:

            data = vp.vDataFrame(
                {
                    "empid": ['1', '2', '3', '4'],
                    "dob": ['1993-01-01', '1988-01-01', '1992-01-01', '1989-01-01'],
                    "doj": ['2022-01-01', '2023-01-01', '2022-01-01', '2023-01-01'],
                    "emp_cat":[933, 945, 723, 799],
                }
            )
            res = data
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_typing_datecol.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_typing_datecol.html

        Let's set the data type of dob and doj to *date*.

        .. code-block:: python

            data["dob"].astype("date")
            data["doj"].astype("date")

        .. ipython:: python
            :suppress:

            data["dob"].astype("date")
            data["doj"].astype("date")

        Let's retrieve the date type vcolumns in the dataset.

        .. ipython:: python

            data.datecol()

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.catcol` : Returns all vDataColumns with categorical values.
            | :py:meth:`verticapy.vDataFrame.numcol` : Returns all vDataColumns with numerical values.
            | :py:meth:`verticapy.vDataFrame.get_columns` : Returns all vDataColumns.
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

        Examples
        ---------

        We import :py:mod:`verticapy`:

        .. ipython:: python

            import verticapy as vp

        .. hint::

            By assigning an alias to :py:mod:`verticapy`, we mitigate the risk
            of code collisions with other libraries. This precaution is
            necessary because verticapy uses commonly known function names
            like "average" and "median", which can potentially lead to naming
            conflicts. The use of an alias ensures that the functions from
            verticapy are used as intended without interfering with functions
            from other libraries.

        For this example, we will use the Titanic dataset.

        .. code-block:: python

            import verticapy.datasets as vpd

            data = vpd.load_titanic()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_titanic.html

        .. note::

            VerticaPy offers a wide range of sample datasets that are
            ideal for training and testing purposes. You can explore
            the full list of available datasets in the :ref:`api.datasets`,
            which provides detailed information on each dataset
            and how to use them effectively. These datasets are invaluable
            resources for honing your data analysis and machine learning
            skills within the VerticaPy environment.

        .. ipython:: python
            :suppress:

            import verticapy.datasets as vpd

            data = vpd.load_titanic()

        Let's check the data type of various vcolumns.

        .. code-block:: python

            data.dtypes()

        .. ipython:: python
            :suppress:

            res = data.dtypes()
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_typing_dtypes.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_typing_dtypes.html
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

        Examples
        ---------

        We import :py:mod:`verticapy`:

        .. ipython:: python

            import verticapy as vp

        .. hint::

            By assigning an alias to :py:mod:`verticapy`, we mitigate the risk
            of code collisions with other libraries. This precaution is
            necessary because verticapy uses commonly known function names
            like "average" and "median", which can potentially lead to naming
            conflicts. The use of an alias ensures that the functions from
            verticapy are used as intended without interfering with functions
            from other libraries.

        Let's create a small dataset:

        .. code-block:: python

            data = vp.vDataFrame(
                {
                    "empid": ['1', '2', '3', '4'],
                    "weight": [140.5, 175, 156.5, 178],
                    "height": [168.5, 175, 178.5, 170],
                    "emp_cat":[933, 945, 723, 799],
                }
            )
            data

        .. ipython:: python
            :suppress:

            data = vp.vDataFrame(
                {
                    "empid": ['1', '2', '3', '4'],
                    "weight": [140.5, 175, 156.5, 178],
                    "height": [168.5, 175, 178.5, 170],
                    "emp_cat":[933, 945, 723, 799],
                }
            )
            res = data
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_typing_numcol.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_typing_numcol.html

        Let's retrieve the numeric type vcolumns in the dataset.

        .. ipython:: python

            data.numcol()

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.catcol` : Returns all vDataColumns with categorical values.
            | :py:meth:`verticapy.vDataFrame.datecol` : Returns all vDataColumns with date-type values.
            | :py:meth:`verticapy.vDataFrame.get_columns` : Returns all vDataColumns.
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

            - 'json' : Converts to a JSON string.
            - 'array': Converts to an array.
            - 'vmap' : Converts to a VMap. If converting a
                delimited string, you can add the header_names
                as follows: dtype = 'vmap(age,name,date)',
                where the header_names are age, name, and date.

        Returns
        -------
        vDataFrame
            self._parent

        Examples
        ---------

        We import :py:mod:`verticapy`:

        .. ipython:: python

            import verticapy as vp

        .. hint::

            By assigning an alias to :py:mod:`verticapy`, we mitigate the risk
            of code collisions with other libraries. This precaution is
            necessary because verticapy uses commonly known function names
            like "average" and "median", which can potentially lead to naming
            conflicts. The use of an alias ensures that the functions from
            verticapy are used as intended without interfering with functions
            from other libraries.

        For this example, we will use the Titanic dataset.

        .. code-block:: python

            import verticapy.datasets as vpd

            data = vpd.load_titanic()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_titanic.html

        .. note::

            VerticaPy offers a wide range of sample datasets that are
            ideal for training and testing purposes. You can explore
            the full list of available datasets in the :ref:`api.datasets`,
            which provides detailed information on each dataset
            and how to use them effectively. These datasets are invaluable
            resources for honing your data analysis and machine learning
            skills within the VerticaPy environment.

        .. ipython:: python
            :suppress:

            import verticapy.datasets as vpd

            data = vpd.load_titanic()

        Let's check the data type of fare vcolumn.

        .. ipython:: python

            data["fare"].dtype()

        Let's change the data type of fare to integer.

        .. code-block:: python

            data["fare"].astype(int)

        .. ipython:: python
            :suppress:

            data["fare"].astype(int)

        Let's check the data type of fare vcolumn again.

        .. ipython:: python

            data["fare"].dtype()

        Now, let's see how we can change the data type from
        string to array. Let's create a small dataset.

        .. code-block:: python

            data = vp.vDataFrame(
                {
                    "artists": ["Inna, Alexandra, Reea", "Rihanna, Beyonce"]
                }
            )
            data["artists"].astype("array")

        .. ipython:: python
            :suppress:

            data = vp.vDataFrame(
                {
                    "artists": ["Inna, Alexandra, Reea", "Rihanna, Beyonce"]
                }
            )
            res = data["artists"].astype("array")
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_typing_astypecol1.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_typing_astypecol1.html

        Let's change the datatype of artists to json.

        .. code-block:: python

            data["artists"].astype("json")

        .. ipython:: python
            :suppress:

            res = data["artists"].astype("json")
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_typing_astypecol2.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_typing_astypecol2.html
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

        Examples
        ---------

        We import :py:mod:`verticapy`:

        .. ipython:: python

            import verticapy as vp

        .. hint::

            By assigning an alias to :py:mod:`verticapy`, we mitigate the risk
            of code collisions with other libraries. This precaution is
            necessary because verticapy uses commonly known function names
            like "average" and "median", which can potentially lead to naming
            conflicts. The use of an alias ensures that the functions from
            verticapy are used as intended without interfering with functions
            from other libraries.

        For this example, we will use the Titanic dataset.

        .. code-block:: python

            import verticapy.datasets as vpd

            data = vpd.load_titanic()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_titanic.html

        .. note::

            VerticaPy offers a wide range of sample datasets that are
            ideal for training and testing purposes. You can explore
            the full list of available datasets in the :ref:`api.datasets`,
            which provides detailed information on each dataset
            and how to use them effectively. These datasets are invaluable
            resources for honing your data analysis and machine learning
            skills within the VerticaPy environment.

        .. ipython:: python
            :suppress:

            import verticapy.datasets as vpd

            data = vpd.load_titanic()

        Let's check the category of "fare" and "name" vcolumns.

        .. ipython:: python

            data["fare"].category()

        .. ipython:: python

            data["name"].category()
        """
        return self._transf[-1][2]

    def ctype(self) -> str:
        """
        Returns the vDataColumn DB type.

        Returns
        -------
        str
            vDataColumn DB type.

        Examples
        ---------

        We import :py:mod:`verticapy`:

        .. ipython:: python

            import verticapy as vp

        .. hint::

            By assigning an alias to :py:mod:`verticapy`, we mitigate the risk
            of code collisions with other libraries. This precaution is
            necessary because verticapy uses commonly known function names
            like "average" and "median", which can potentially lead to naming
            conflicts. The use of an alias ensures that the functions from
            verticapy are used as intended without interfering with functions
            from other libraries.

        For this example, we will use the Titanic dataset.

        .. code-block:: python

            import verticapy.datasets as vpd

            data = vpd.load_titanic()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_titanic.html

        .. note::

            VerticaPy offers a wide range of sample datasets that are
            ideal for training and testing purposes. You can explore
            the full list of available datasets in the :ref:`api.datasets`,
            which provides detailed information on each dataset
            and how to use them effectively. These datasets are invaluable
            resources for honing your data analysis and machine learning
            skills within the VerticaPy environment.

        .. ipython:: python
            :suppress:

            import verticapy.datasets as vpd

            data = vpd.load_titanic()

        Let's check the DB type of "fare" and "name" vcolumns.

        .. ipython:: python

            data["fare"].ctype()

        .. ipython:: python

            data["name"].ctype()
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

        Examples
        ---------

        We import :py:mod:`verticapy`:

        .. ipython:: python

            import verticapy as vp

        .. hint::

            By assigning an alias to :py:mod:`verticapy`, we mitigate the risk
            of code collisions with other libraries. This precaution is
            necessary because verticapy uses commonly known function names
            like "average" and "median", which can potentially lead to naming
            conflicts. The use of an alias ensures that the functions from
            verticapy are used as intended without interfering with functions
            from other libraries.

        Let's create a small dataset.

        .. code-block:: python

            data = vp.vDataFrame(
                {"artists": ["Inna, Alexandra, Reea", "Rihanna, Beyonce"]}
            )
            data["artists"].astype("array")

        .. ipython:: python
            :suppress:

            data = vp.vDataFrame(
                {"artists": ["Inna, Alexandra, Reea", "Rihanna, Beyonce"]}
            )
            res = data["artists"].astype("array")
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_typing_isarray.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_typing_isarray.html

        Let's check if data type of "artists" vcolumn is array or not.

        .. ipython:: python

            data["artists"].isarray()
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

        Examples
        ---------

        We import :py:mod:`verticapy`:

        .. ipython:: python

            import verticapy as vp

        .. hint::

            By assigning an alias to :py:mod:`verticapy`, we mitigate the risk
            of code collisions with other libraries. This precaution is
            necessary because verticapy uses commonly known function names
            like "average" and "median", which can potentially lead to naming
            conflicts. The use of an alias ensures that the functions from
            verticapy are used as intended without interfering with functions
            from other libraries.

        Let's create a small dataset:

        .. code-block:: python

            data = vp.vDataFrame(
                {
                    "empid": [1, 2, 3, 4],
                    "is_temp": [True, False, False, True],
                }
            )
            data

        .. ipython:: python
            :suppress:

            data = vp.vDataFrame(
                {
                    "empid": [1, 2, 3, 4],
                    "is_temp": [True, False, False, True],
                }
            )
            res = data
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_typing_isbool.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_typing_isbool.html

        Let's check if data type of "is_temp" vcolumn is bool or not.

        .. ipython:: python

            data["is_temp"].isbool()

        Let's check if data type of "empid" vcolumn is bool or not.

        .. ipython:: python

            data["empid"].isbool()
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

        Examples
        ---------

        We import :py:mod:`verticapy`:

        .. ipython:: python

            import verticapy as vp

        .. hint::

            By assigning an alias to :py:mod:`verticapy`, we mitigate the risk
            of code collisions with other libraries. This precaution is
            necessary because verticapy uses commonly known function names
            like "average" and "median", which can potentially lead to naming
            conflicts. The use of an alias ensures that the functions from
            verticapy are used as intended without interfering with functions
            from other libraries.

        For this example, we will use the Amazon dataset.

        .. code-block:: python

            import verticapy.datasets as vpd

            amazon = vpd.load_amazon()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_amazon.html

        .. note::

            VerticaPy offers a wide range of sample datasets that are
            ideal for training and testing purposes. You can explore
            the full list of available datasets in the :ref:`api.datasets`,
            which provides detailed information on each dataset
            and how to use them effectively. These datasets are invaluable
            resources for honing your data analysis and machine learning
            skills within the VerticaPy environment.

        .. ipython:: python
            :suppress:

            import verticapy.datasets as vpd
            amazon = vpd.load_amazon()

        Let's check if the category of "date" vcolumn is date or not.

        .. ipython:: python

            amazon["date"].isdate()

        Let's check if the category of "state" vcolumn is date or not

        .. ipython:: python

            amazon["state"].isdate()
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

        Examples
        ---------

        We import :py:mod:`verticapy`:

        .. ipython:: python

            import verticapy as vp

        .. hint::

            By assigning an alias to :py:mod:`verticapy`, we mitigate the risk
            of code collisions with other libraries. This precaution is
            necessary because verticapy uses commonly known function names
            like "average" and "median", which can potentially lead to naming
            conflicts. The use of an alias ensures that the functions from
            verticapy are used as intended without interfering with functions
            from other libraries.

        Let's create a small dataset:

        .. code-block:: python

            data = vp.vDataFrame(
                {
                    "empid": [1, 2, 3, 4],
                    "is_temp": [True, False, False, True],
                }
            )
            data

        .. ipython:: python
            :suppress:

            data = vp.vDataFrame(
                {
                    "empid": [1, 2, 3, 4],
                    "is_temp": [True, False, False, True],
                }
            )
            res = data
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_typing_isbool.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_typing_isbool.html

        Let's check if data type of "empid" vcolumn is numerical or not.

        .. ipython:: python

            data["empid"].isnum()

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

        Examples
        ---------

        We import :py:mod:`verticapy`:

        .. ipython:: python

            import verticapy as vp

        .. hint::

            By assigning an alias to :py:mod:`verticapy`, we mitigate the risk
            of code collisions with other libraries. This precaution is
            necessary because verticapy uses commonly known function names
            like "average" and "median", which can potentially lead to naming
            conflicts. The use of an alias ensures that the functions from
            verticapy are used as intended without interfering with functions
            from other libraries.

        Let's create a small dataset.

        .. code-block:: python

            data = vp.vDataFrame(
                {
                    "empid": ['1'],
                    "mgr": ['{"0.country.id": "214", "0.country.name": "Spain", "0.name": "Luis Enrique Martínez García"}'],
                }
            )
            data["mgr"].astype("vmap")

        .. ipython:: python
            :suppress:

            import verticapy as vp
            data = vp.vDataFrame(
                            {
                                "empid": ['1'],
                                "mgr": ['{"0.country.id": "214", "0.country.name": "Spain", "0.name": "Luis Enrique Martínez García"}'],
                            }
                        )
            res = data["mgr"].astype("vmap")
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_typing_isvmap.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_typing_isvmap.html

        Let's check if data type of "mgr" vcolumn is vmap or not.

        .. ipython:: python

            data["mgr"].isvmap()

        Let's check if data type of "empid" vcolumn is vmap or not.

        .. ipython:: python

            data["empid"].isvmap()
        """
        return self.category() == "vmap" or isvmap(
            column=self._alias, expr=self._parent._genSQL()
        )
