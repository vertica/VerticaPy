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

from verticapy._typing import NoneType, SQLColumns, SQLExpression
from verticapy._utils._object import create_new_vdf
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import format_type, quote_ident
from verticapy._utils._sql._merge import gen_coalesce, group_similar_names
from verticapy._utils._gen import gen_col_name
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

        .. warning::

            This  function  might  have  a  long  runtime
            and can make your  vDataFrame less performant.
            It makes many calls to the MAPLOOKUP function,
            which can be slow if your VMap is large.

        Parameters
        ----------
        vmap_col: SQLColumns, optional
            List of VMap columns to flatten.
        limit: int, optional
            Maximum number of keys to consider for each VMap.
            Only the most occurent keys are used.
        exclude_columns: SQLColumns, optional
            List of VMap columns to exclude.

        Returns
        -------
        vDataFrame
            object with the flattened VMaps.

        Examples
        --------
        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        .. hint::

            By assigning an alias to :py:mod:`verticapy`,
            we mitigate the risk of code collisions with
            other libraries. This precaution is necessary
            because verticapy uses commonly known function
            names like "average" and "median", which can
            potentially lead to naming conflicts. The use
            of an alias ensures that the functions from
            :py:mod:`verticapy` are used as intended
            without interfering with functions from other
            libraries.

        For this example, let's generate a dataset
        that has a VMAP in one column:

        .. ipython:: python

            vdf = vp.vDataFrame(
                {
                    "id": [1],
                    "team": ['{"country" : "France", "region" : "IDF"}'],
                }
            )

        .. ipython:: python
            :suppress:

            result = vdf
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_flat_vmap_1.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_flat_vmap_1.html

        .. note::

            We can observe that our string follows the structure of a
            JSON. VerticaPy will automatically parse it and determine
            how to extract the elements.

        In order to utilize Vertica Flex Table auto-parsing, it is
        necessary to convert the string column 'team' to a vmap.

        .. code-block:: python

            vdf["team"].astype('vmap')

        .. ipython:: python
            :suppress:

            vdf["team"].astype('vmap')
            result = vdf
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_flat_vmap.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_flat_vmap.html

        Now we can flatten the vmap:

        .. code-block::

            vdf.flat_vmap()

        .. ipython:: python
            :suppress:

            result = vdf.flat_vmap()
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_flat_vmap_2.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_flat_vmap_2.html

        .. note::

            This function is applicable for flattening Flex tables VMAP.
            However, it is advisable to store the final result in a table,
            as the computations involved can be resource-intensive.

        .. seealso::
            | ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.merge_similar_names` : Merges
                columns with similar names.
            | ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.pivot` : Pivots the vDataFrame.
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
        skip_word: str | list, optional
            List  of words to  exclude  from  the provided column  names.
            For example,     if      two      columns      are     named
            'age.information.phone'  and  'age.phone' AND  ``skip_word``  is
            set  to  ``['.information']``,  then  the  two  columns are
            merged  together  with  the   following  COALESCE  statement:
            ``COALESCE("age.phone", "age.information.phone") AS "age.phone"``

        Returns
        -------
        vDataFrame
            An object containing the merged element.

        Examples
        --------
        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        .. hint::

            By assigning an alias to :py:mod:`verticapy`,
            we mitigate the risk of code collisions with
            other libraries. This precaution is necessary
            because verticapy uses commonly known function
            names like "average" and "median", which can
            potentially lead to naming conflicts. The use
            of an alias ensures that the functions from
            :py:mod:`verticapy` are used as intended
            without interfering with functions from other
            libraries.

        For this example, let's generate a dataset
        which has two columns that are duplicates
        with slight change in spelling and some
        missing values:

        .. ipython:: python

            vdf = vp.vDataFrame(
                {
                    "user.id": [12, None, 13],
                    "id": [12, 11, None],
                }
            )

        .. ipython:: python
            :suppress:

            result = vdf
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_merge_similar_names_1.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_merge_similar_names_1.html

        In order to remove the redundant column, we
        can combine them using ``merge_similar_names``:

        .. code-block:: python

            vdf.merge_similar_names(skip_word = "user.")

        .. ipython:: python
            :suppress:

            result = vdf.merge_similar_names(skip_word = "user.")
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_merge_similar_names.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_merge_similar_names.html

        .. note::

            This function is particularly useful when flattening highly
            nested JSON files. Such files may contain redundant features
            and inconsistencies. The function is designed to merge these
            features, ensuring consistent information.

        .. seealso::
            | ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.pivot` : Pivots the vDataFrame.
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

        Examples
        --------
        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        .. hint::

            By assigning an alias to :py:mod:`verticapy`,
            we mitigate the risk of code collisions with
            other libraries. This precaution is necessary
            because verticapy uses commonly known function
            names like "average" and "median", which can
            potentially lead to naming conflicts. The use
            of an alias ensures that the functions from
            :py:mod:`verticapy` are used as intended
            without interfering with functions from other
            libraries.

        For this example, let's generate a dataset
        which has multiple columns:

        .. ipython:: python

            vdf = vp.vDataFrame(
                {
                    "id": [12, 11, 13],
                    "state": [12, 11, 13],
                    "size":[100, 120, 140],
                    "score": [9, 9.5, 4],
                    "extra_info": ['Grey', 'Black', 'White'],
                }
            )

        .. ipython:: python
            :suppress:

            result = vdf
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_narrow_1.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_narrow_1.html

        To focus only on the quantities of interest, we can
        utilize the ``narrow`` function:

        .. code-block:: python

            vdf.narrow("id", col_name = "state", val_name = "score")

        .. ipython:: python
            :suppress:

            result = vdf.narrow("id", col_name = "state", val_name = "score")
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_narrow.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_narrow.html

        .. note::

            The inverse function of ``pivot`` is ``narrow``. With
            both, you can preprocess the table either vertically
            or horizontally. These functions utilize pure SQL
            statements to perform the job.

        .. seealso::
            | ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.pivot` : Pivots the vDataFrame.
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
        Returns the Pivot of the vDataFrame using the
        input aggregation.

        Parameters
        ----------
        index: str
            :py:class:`~vDataColumn` used to group the
            elements.
        columns: str
            The :py:class:`~vDataColumn` used to compute
            the different categories, which then act as
            the columns in the pivot table.
        values: str
            The vDataColumn whose values populate the
            new :py:class:`~vDataFrame`.
        aggr: str, optional
            Aggregation to use on 'values'.  To use complex
            aggregations, you must use braces: ``{}``. For
            example, to aggregate using the aggregation:
            ``x -> MAX(x) - MIN(x)``, write ``MAX({}) - MIN({})``.
        prefix: str, optional
            The prefix for the pivot table's column names.

        Returns
        -------
        vDataFrame
            the pivot table object.

        Examples
        --------
        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        .. hint::

            By assigning an alias to :py:mod:`verticapy`,
            we mitigate the risk of code collisions with
            other libraries. This precaution is necessary
            because verticapy uses commonly known function
            names like "average" and "median", which can
            potentially lead to naming conflicts. The use
            of an alias ensures that the functions from
            :py:mod:`verticapy` are used as intended
            without interfering with functions from other
            libraries.

        For this example, let's generate a dummy dataset
        representing sales of two items for different dates:

        .. ipython:: python

            vdf = vp.vDataFrame(
                {
                    "date": [
                        "2014-01-01",
                        "2014-01-02",
                        "2014-01-01",
                        "2014-01-02",
                    ],
                    "cat": ["A", "A", "B", "B"],
                    "sale": [100, 120, 120, 110],
                }
            )

        .. ipython:: python
            :suppress:

            result = vdf
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_pivot_1.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_pivot_1.html

        To better view the data, we can create a
        pivot table:

        .. code-block:: python

            vdf.pivot(
                index = "date",
                columns = "cat",
                values = "sale",
                aggr = "avg",
            )

        .. ipython:: python
            :suppress:

            result = vdf.pivot(
                index = "date",
                columns = "cat",
                values = "sale",
                aggr = "avg",
            )
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_pivot.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_pivot.html

        .. note::

            The inverse function of ``pivot`` is ``narrow``. With
            both, you can preprocess the table either vertically
            or horizontally. These functions utilize pure SQL
            statements to perform the job.

        .. seealso::
            | ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.narrow` :
                Narrow Table for a :py:class:`~vDataFrame`.
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

    @save_verticapy_logs
    def explode_array(
        self,
        index: str,
        column: str,
        prefix: Optional[str] = "col_",
        delimiter: Optional[bool] = True,
    ) -> "vDataFrame":
        """
        Returns exploded vDataFrame of array-like
        columns in a vDataFrame.

        .. versionadded:: 10.0.0

        Parameters
        ----------
        index: str
            Index used to identify the Row.
        column: str
            The name of the array-like column to
            explode.
        prefix: str, optional
            The prefix for the column names of
            exploded values defaults to "col_".
        delimeter: str, optional
            Specify if array-like data is separated
            by a comma defaults value is ``True``.

        Returns
        -------
        vDataFrame
            horizontal exploded vDataFrame.

        Examples
        --------
        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        .. hint::

            By assigning an alias to :py:mod:`verticapy`,
            we mitigate the risk of code collisions with
            other libraries. This precaution is necessary
            because verticapy uses commonly known function
            names like "average" and "median", which can
            potentially lead to naming conflicts. The use
            of an alias ensures that the functions from
            :py:mod:`verticapy` are used as intended
            without interfering with functions from other
            libraries.

        For this example, let's generate a dataset:

        .. ipython:: python

            data = vp.vDataFrame(
                {
                    "id": [1, 2, 3, 4],
                    "values": [
                        [70, 80, 90, 5],
                        [47, 34, 93, 20, 13, 16],
                        [1, 45, 56, 21, 10, 35, 56, 8, 39],
                        [89],
                    ]
                }
            )

        We can compute the exploded vDataFrame.

        .. code-block:: python

            data.explode_array(index = "id", column = "values")

        .. ipython:: python
            :suppress:

            result = data.explode_array(index = "id", column = "values")
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_explode_array_table.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_explode_array_table.html

        .. note::

            This function operates on various data types, including
            arrays and varchar representations of arrays.
            For arrays with elements separated by commas, as well as
            varchar representations of arrays with no delimiter (in
            which case you must specify ``delimiter`` as ``False``).

        .. seealso::
            | ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.pivot` : Pivots the vDataFrame.
        """

        # Type check
        if not isinstance(index, str):
            raise TypeError("The 'index' parameter must be a string.")
        if not isinstance(column, str):
            raise TypeError("The 'column' parameter must be a string.")

        # Creating a vDataFrame copy
        vdf = self.copy()

        # Replace tabs with commas if specified
        if not delimiter:
            vdf[column].str_replace(to_replace="[\t]+", value=",")

        # check if the column type is an array
        if vdf[column].dtype().find("array"):
            # Apply the STRING_TO_ARRAY function to convert str array to array of strings
            vdf[column].apply(func="STRING_TO_ARRAY({})")

        # To avoid any name conflict
        position_col_name = gen_col_name(n=6)
        value_col_name = gen_col_name(n=6)

        # Create a new vDataFrame with exploded values
        vdf = create_new_vdf(
            f"""
            SELECT 
                /*+LABEL('vDataframe.explode')*/ 
                {index}, 
                {column}, 
                EXPLODE({column} 
                        USING PARAMETERS skip_partitioning=True) 
                        AS ({position_col_name}, {value_col_name}) 
            FROM {vdf};"""
        )
        # Convert the "value" column to float and pivot the data
        return vdf.astype({value_col_name: "float"}).pivot(
            index=index, columns=position_col_name, values=value_col_name, prefix=prefix
        )
