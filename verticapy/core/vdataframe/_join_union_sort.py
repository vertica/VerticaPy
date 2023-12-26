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
import copy
from typing import Literal, Optional, Union, TYPE_CHECKING

from verticapy._typing import SQLColumns, SQLExpression, SQLRelation
from verticapy._utils._object import create_new_vdf
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import (
    extract_and_rename_subquery,
    format_type,
    quote_ident,
)
from verticapy._utils._sql._vertica_version import vertica_version

from verticapy.core.vdataframe._math import vDFMath

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame


class vDFJoinUnionSort(vDFMath):
    @save_verticapy_logs
    def append(
        self,
        input_relation: SQLRelation,
        expr1: Optional[SQLExpression] = None,
        expr2: Optional[SQLExpression] = None,
        union_all: bool = True,
    ) -> "vDataFrame":
        """
        Merges the vDataFrame with another vDataFrame or an input
        relation, and returns a new vDataFrame.

        .. warning::

            Appending datasets can potentially increase the structural
            weight; exercise caution when performing this operation.

        Parameters
        ----------
        input_relation: SQLRelation
            Relation to merge with.
        expr1: SQLExpression, optional
            List of pure-SQL expressions from the current
            :py:class:`~vDataFrame` to use during merging.
            For example,  ``CASE WHEN "column" > 3 THEN 2 ELSE
            NULL END`` and  ``POWER("column", 2)`` will work.
            If empty, all vDataFrame vDataColumns are used.
            Aliases are recommended to avoid auto-naming.
        expr2: SQLExpression, optional
            List of pure-SQL  expressions from the input relation to
            use during the merging.
            For example, ``CASE WHEN "column" > 3 THEN 2 ELSE NULL END``
            and ``POWER("column", 2)``  will work.  If empty, all input
            relation columns are  used. Aliases  are  recommended to
            avoid auto-naming.
        union_all: bool, optional
            If  set to True, the  vDataFrame is merged with the input
            relation using an 'UNION ALL' instead of an 'UNION'.

        Returns
        -------
        vDataFrame
           vDataFrame of the Union

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

        Let us create two :py:class:`~vDataFrame` which we can
        merge for this example:

        .. ipython:: python

                vdf = vp.vDataFrame(
                    {
                        "score": [12, 11, 13],
                        "cat": ['A', 'B', 'A'],
                    }
                )

                vdf_2 = vp.vDataFrame(
                    {
                        "score": [11, 1, 23],
                        "cat": ['A', 'B', 'B'],
                    }
                )

        We can conveniently append the the first
        :py:class:`~vDataFrame` with the second one:

        .. code-block:: python

            vdf.append(vdf_2)

        .. ipython:: python
            :suppress:

            result = vdf.append(vdf_2)
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_join_union_sort_append.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_join_union_sort_append.html

        We can also apply some SQL expressions on the append
        using ``expr1`` and ``expr2``. Let us try to
        limit the maximum value of the second
        :py:class:`~vDataFrame` to 20.

        .. code-block:: python

            vdf.append(
                vdf_2,
                expr1 = [
                    'CASE WHEN "score" > 20 THEN 20 ELSE "score" END',
                    '"cat"',
                ],
            )

        .. ipython:: python
            :suppress:

            result = vdf.append(vdf_2, expr1=['CASE WHEN "score" > 20 THEN 20 ELSE "score" END', '"cat"'])
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_join_union_sort_append_2.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_join_union_sort_append_2.html

        .. note::

            VerticaPy offers the flexibility to use UNION ALL or simple UNION
            based on your specific use case. The former includes duplicates,
            while the latter handles them. Refer to ``union_all`` for more
            information.

        .. seealso::

            | ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.join` : Joins the
                :py:class:`~vDataFrame` with another one or an
                ``input_relation``.
        """
        expr1, expr2 = format_type(expr1, expr2, dtype=list)
        columns = ", ".join(self.get_columns()) if not expr1 else ", ".join(expr1)
        columns2 = columns if not expr2 else ", ".join(expr2)
        union = "UNION" if not union_all else "UNION ALL"
        query = f"""
            (SELECT 
                {columns} 
             FROM {self}) 
             {union} 
            (SELECT 
                {columns2} 
             FROM {input_relation})"""
        return create_new_vdf(query)

    @save_verticapy_logs
    def join(
        self,
        input_relation: SQLRelation,
        on: Union[None, tuple, dict, list] = None,
        on_interpolate: Optional[dict] = None,
        how: Literal[
            "left", "right", "cross", "full", "natural", "self", "inner", None
        ] = "natural",
        expr1: Optional[SQLExpression] = None,
        expr2: Optional[SQLExpression] = None,
    ) -> "vDataFrame":
        """
        Joins the :py:class:`~vDataFrame` with another
        one or an ``input_relation``.

        .. warning::

            Joins  can  make  the  vDataFrame  structure
            heavier.  It is recommended that you check
            the    current     structure    using    the
            ``current_relation``  method  and  save  it
            with the ``to_db`` method, using the parameters
            ``inplace = True`` and ``relation_type = table``.

        Parameters
        ----------
        input_relation: SQLRelation
            Relation to join with.
        on: tuple | dict | list, optional
            If using a list:
            List of 3-tuples. Each tuple must include
            (key1, key2, operator) — where ``key1`` is
            the key of the :py:class:`~vDataFrame`,
            ``key2`` is the key of the ``input_relation``,
            and ``operator`` is one of the following:

            - '=':
                exact match
            - '<':
                key1  < key2
            - '>':
                key1  > key2
            - '<=':
                key1 <= key2
            - '>=':
                key1 >= key2
            - 'llike':
                key1 LIKE '%' || key2 || '%'
            - 'rlike':
                key2 LIKE '%' || key1 || '%'
            - 'linterpolate':
                key1 INTERPOLATE key2
            - 'rinterpolate':
                key2 INTERPOLATE key1

            Some operators need 5-tuples:
            ``(key1, key2, operator, operator2, x)``
            where  ``operator2`` is  a simple operator
            ``(=, >, <, <=, >=)``, x is a ``float`` or
            an ``integer``, and ``operator`` is one of the
            following:

            - 'jaro':
                ``JARO(key1, key2) operator2 x``
            - 'jarow':
                ``JARO_WINCKLER(key1, key2) operator2 x``
            - 'lev':
                ``LEVENSHTEIN(key1, key2) operator2 x``

            If using a dictionary:
            This parameter must include all the different
            keys. It must be similar to the following:
            ``{"relationA_key1": "relationB_key1" ...,"relationA_keyk": "relationB_keyk"}``
            where ``relationA`` is the current :py:class:`~vDataFrame`
            and ``relationB`` is the ``input_relation`` or
            the input :py:class:`~vDataFrame`.

        on_interpolate: dict, optional
            Dictionary of all unique keys. This is used
            to join two event series together using some
            ordered attribute. Event series joins let you
            compare values from two series directly, rather
            than having to normalize the series to the same
            measurement interval. The dict must be similar
            to the following:
            ``{"relationA_key1": "relationB_key1" ...,"relationA_keyk": "relationB_keyk"}``
            where ``relationA`` is the current :py:class:`~vDataFrame`
            and ``relationB`` is the ``input_relation`` or the
            input :py:class:`~vDataFrame`.

        how: str, optional
            Join Type.

            - left:
                Left Join.
            - right:
                Right Join.
            - cross:
                Cross Join.
            - full:
                Full Outer Join.
            - natural:
                Natural Join.
            - inner:
                Inner Join.

        expr1: SQLExpression, optional
            List of the different columns in pure SQL
            to select from the current :py:class:`~vDataFrame`,
            optionally as aliases. Aliases are recommended
            to avoid ambiguous names. For example: ``column``
            or ``column AS my_new_alias``.
        expr2: SQLExpression, optional
            List of the different columns in pure SQL
            to select from the current :py:class:`~vDataFrame`,
            optionally as aliases. Aliases are recommended
            to avoid ambiguous names. For example: ``column``
            or ``column AS my_new_alias``.

        Returns
        -------
        vDataFrame
            object result of the join.

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

        Let us create two :py:class:`~vDataFrame` which we
        can JOIN for this example:

        .. ipython:: python

            employees_data = vp.vDataFrame(
                {
                    "employee_id": [1, 2, 3, 4],
                    "employee_name": ['Alice', 'Bob', 'Charlie', 'David'],
                    "department_id": [101, 102, 101, 103],
                },
            )

            departments_data = vp.vDataFrame(
                {
                    "department_id": [101, 102, 104],
                    "department_name": ['HR', 'Finance', 'Marketing'],
                }
            )

        .. ipython:: python
            :suppress:

            result = employees_data
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_join_union_sort_join_table1.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_join_union_sort_join_table1.html

        .. ipython:: python
            :suppress:

            result = departments_data
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_join_union_sort_join_table2.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_join_union_sort_join_table2.html

        Let us look at the different type of JOINs
        available below:

        - INNER JOIN
        - LEFT JOIN
        - RIGHT JOIN
        - FULL JOIN

        After that we will also have a look at:

        - Other operators.
        - Special operators like Jaro-Winkler.

        INNER JOIN
        ^^^^^^^^^^^

        We can conveniently JOIN the two :py:class:`~vDataFrame`
        using the key column. Let us perform an INNER JOIN.
        INNER JOIN is executed to combine rows from both
        the main table and the ``input_relation`` based on
        a specified condition. Only the rows with matching
        values in the specified column are included in the
        result. If there is no match, those rows are
        excluded from the output.

        .. ipython:: python

            result = employees_data.join(
                input_relation = departments_data,
                on = [("department_id", "department_id", "=")],
                how = "inner",
                expr1 = [
                    "employee_id AS ID",
                    "employee_name AS Name",
                ],
                expr2 = ["department_name AS Dep"],
            )

        .. ipython:: python
            :suppress:

            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_join_union_sort_join.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_join_union_sort_join.html

        LEFT JOIN
        ^^^^^^^^^^

        Similarly we can perform a LEFT JOIN which ensures that
        all rows from the main table are included in the
        result, and matching rows from the ``input_relation``
        are included if they exist. If there is no match,
        the columns from the input relation will contain
        ``NULL`` values for the corresponding rows in the
        result.

        .. ipython:: python

            left_join_result = employees_data.join(
                input_relation = departments_data,
                on = [("department_id", "department_id", "=")],
                how = "left",
                expr1 = ["employee_id AS ID", "employee_name AS Name"],
                expr2 = ["department_name AS Dep"],
            )

        .. ipython:: python
            :suppress:

            result = left_join_result
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_join_union_sort_join_left_join.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_join_union_sort_join_left_join.html

        RIGHT JOIN
        ^^^^^^^^^^^

        A RIGHT JOIN is employed to include all rows
        from the ``input_relation`` in the result,
        regardless of whether there are matching values
        in the main table. Rows from the main table are
        included if there are matching values, and for
        non-matching rows, the columns from the main
        table will contain NULL values in the result.

        .. ipython:: python

            right_join_result = employees_data.join(
                input_relation = departments_data,
                on = [("department_id", "department_id", "=")],
                how = "right",
                expr1 = ["employee_id AS ID", "employee_name AS Name"],
                expr2 = ["department_name AS Dep"],
            )

        .. ipython:: python
            :suppress:

            result = right_join_result
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_join_union_sort_join_right_join.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_join_union_sort_join_right_join.html

        FULL JOIN
        ^^^^^^^^^^

        A FULL JOIN is utilized to include all rows
        from both the main table and the ``input_relation``
        in the result. Matching rows are included based
        on the specified condition, and for non-matching
        rows in either table, the columns from the non-matching
        side will contain NULL values in the result.
        This ensures that all rows from both tables are
        represented in the output.

        .. ipython:: python

            full_join_result = employees_data.join(
                input_relation = departments_data,
                on = [("department_id", "department_id", "=")],
                how = "full",
                expr1 = ["employee_id AS ID", "employee_name AS Name"],
                expr2 = ["department_name AS Dep"],
            )

        .. ipython:: python
            :suppress:

            result = full_join_result
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_join_union_sort_join_full_join.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_join_union_sort_join_full_join.html

        OTHER OPERATORS
        ^^^^^^^^^^^^^^^^

        Let us explore some additional features of joins.
        For that let us create another table:

        .. ipython:: python

            additional_departments_data = vp.vDataFrame(
                {
                    "department_size": [12, 8, 8, 10],
                    "department": ['HR', 'Fin', 'Mar', 'IT'],
                }
            )

        .. ipython:: python
            :suppress:

            result = additional_departments_data
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_join_union_sort_join_table_3.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_join_union_sort_join_table_3.html

        Notice the names are a bit different than the "department_name"
        column in the previous ``department_data`` table. In such cases
        we can utilize the ``llike`` operator:

        .. ipython:: python

            department_join = departments_data.join(
                input_relation = additional_departments_data,
                on = [("department_name", "department", "llike")],
                how = "inner",
                expr1 = ["department_id AS ID", "department_name AS Dep"],
                expr2 = ["department_size AS Size"],
            )

        .. ipython:: python
            :suppress:

            result = full_join_result
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_join_union_sort_join_llike.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_join_union_sort_join_llike.html

        .. note::

            VerticaPy provides an array of join options and diverse
            operators, delivering an exceptional user experience.

        JARO-WINKLER
        ^^^^^^^^^^^^

        VerticaPy also allows you to JOIN tables using the
        Jaro-Winkler method. It is a string similarity metric
        used to compare the similarity between two strings.
        This method can be particularly useful in scenarios
        where slight spelling mistakes are expected between
        keys of different tables.

        Let us create two tables for this example:

        .. ipython:: python

            users_data = vp.vDataFrame(
                {
                    "user_id": [1, 2, 3],
                    "email": ['alice@email.com', 'bob@email.com', 'charlie@email.com'],
                    "username": ['Ali', 'Bob', 'Charlie'],
                    "age": [25, 30, 22],
                },
            )

            orders_data = vp.vDataFrame(
                {
                    "order_id": [101, 102, 103],
                    "email": ['Alice@email.com', 'bob@email.com', 'charlee@email.com'],
                    "product_name": ['Laptop', 'Headphones', 'Smartphone'],
                    "quantity": [2, 1, 3],
                }
            )

        .. ipython:: python
            :suppress:

            result = users_data
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_join_union_sort_join_jarow_1.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_join_union_sort_join_jarow_1.html

        .. ipython:: python
            :suppress:

            result = orders_data
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_join_union_sort_join_jarow_2.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_join_union_sort_join_jarow_2.html

        Notice that some emails are not correctly spelled,
        so we can use the ``jarow`` option to JOIN them:

        .. ipython:: python

            result = users_data.join(
                input_relation = orders_data,
                on = [("email", "email", "jarow", ">=", 0.9)],
                how = "inner",
                expr1 = [
                    "user_id AS ID",
                    "username AS Name",
                    "email",
                ],
                expr2 = ["product_name AS Item", "quantity AS Qty"],
            )

        .. ipython:: python
            :suppress:

            result = orders_data
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_join_union_sort_join_jarow_2_result.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_join_union_sort_join_jarow_2_result.html

        .. seealso::

            | ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.append` : Append a
                :py:class:`~vDataFrame` with another one or an
                ``input_relation``.
        """
        on, on_interpolate = format_type(on, on_interpolate, dtype=dict)
        expr1, expr2 = format_type(expr1, expr2, dtype=list, na_out="*")
        if isinstance(on, tuple):
            on = [on]
        # List with the operators
        if str(how).lower() == "natural" and (on or on_interpolate):
            raise ValueError(
                "Natural Joins cannot be computed if any of "
                "the parameters 'on' or 'on_interpolate' are "
                "defined."
            )
        on_list = []
        if isinstance(on, dict):
            on_list += [(key, on[key], "=") for key in on]
        else:
            on_list += copy.deepcopy(on)
        on_list += [
            (key, on_interpolate[key], "linterpolate") for key in on_interpolate
        ]
        # Checks
        self.format_colnames([x[0] for x in on_list])
        object_type = None
        if hasattr(input_relation, "object_type"):
            object_type = input_relation.object_type
        if object_type == "vDataFrame":
            input_relation.format_colnames([x[1] for x in on_list])
        # Relations
        first_relation = extract_and_rename_subquery(self._genSQL(), alias="x")
        second_relation = extract_and_rename_subquery(f"{input_relation}", alias="y")
        # ON
        on_join = []
        all_operators = [
            "=",
            ">",
            ">=",
            "<",
            "<=",
            "llike",
            "rlike",
            "linterpolate",
            "rinterpolate",
            "jaro",
            "jarow",
            "lev",
        ]
        simple_operators = all_operators[0:5]
        for x in on_list:
            key1, key2, op = quote_ident(x[0]), quote_ident(x[1]), x[2]
            if op not in all_operators:
                raise ValueError(
                    f"Incorrect operator: '{op}'.\nCorrect values: {', '.join(simple_operators)}."
                )
            if op in ("=", ">", ">=", "<", "<="):
                on_join += [f"x.{key1} {op} y.{key2}"]
            elif op == "llike":
                on_join += [f"x.{key1} LIKE '%' || y.{key2} || '%'"]
            elif op == "rlike":
                on_join += [f"y.{key2} LIKE '%' || x.{key1} || '%'"]
            elif op == "linterpolate":
                on_join += [f"x.{key1} INTERPOLATE PREVIOUS VALUE y.{key2}"]
            elif op == "rinterpolate":
                on_join += [f"y.{key2} INTERPOLATE PREVIOUS VALUE x.{key1}"]
            elif op in ("jaro", "jarow", "lev"):
                if op in ("jaro", "jarow"):
                    vertica_version(condition=[12, 0, 2])
                else:
                    vertica_version(condition=[10, 1, 0])
                op2, x = x[3], x[4]
                if op2 not in simple_operators:
                    raise ValueError(
                        f"Incorrect operator: '{op2}'.\nCorrect values: {', '.join(simple_operators)}."
                    )
                map_to_fun = {
                    "jaro": "JARO_DISTANCE",
                    "jarow": "JARO_WINKLER_DISTANCE",
                    "lev": "EDIT_DISTANCE",
                }
                fun = map_to_fun[op]
                on_join += [f"{fun}(x.{key1}, y.{key2}) {op2} {x}"]
        # Final
        on_join = " ON " + " AND ".join(on_join) if on_join else ""
        expr = [f"x.{key}" for key in expr1] + [f"y.{key}" for key in expr2]
        expr = "*" if not expr else ", ".join(expr)
        if how:
            how = " " + how.upper() + " "
        query = (
            f"SELECT {expr} FROM {first_relation}{how}JOIN {second_relation} {on_join}"
        )
        return create_new_vdf(query)

    @save_verticapy_logs
    def sort(self, columns: Union[SQLColumns, dict]) -> "vDataFrame":
        """
        Sorts the :py:class:`~vDataFrame` using the input
        :py:class:`~vDataColumn`.

        Parameters
        ----------
        columns: SQLColumns | dict
            List  of the  :py:class:`~vDataColumn`  used to sort
            the data, using asc order or dictionary of all sorting
            methods. For example, to sort by "column1" ASC and
            "column2" DESC, write:
            ``{"column1": "asc", "column2": "desc"}``

        Returns
        -------
        vDataFrame
            self

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

        Let us create a :py:class:`~vDataFrame` which
        we can sort:

        .. ipython:: python

            vdf = vp.vDataFrame(
                {
                    "sales": [10, 11, 9, 20, 6],
                    "cat": ['C', 'B', 'A', 'A', 'B'],
                },
            )

        .. ipython:: python
            :suppress:

            result = vdf
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_join_union_sort_sort_data.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_join_union_sort_sort_data.html

        We can conveniently sort the :py:class:`~vDataFrame`
        using a particular column:

        .. ipython:: python

            vdf.sort({"sales": "asc"})

        .. ipython:: python
            :suppress:

            result = vdf
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_join_union_sort_sort.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_join_union_sort_sort.html

        The same operation can also be performed in descending
        order.

        .. ipython:: python

            vdf.sort({"sales": "desc"})

        .. ipython:: python
            :suppress:

            result = vdf
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_join_union_sort_sort_2.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_join_union_sort_sort_2.html

        .. note::

            Sorting the data is crucial to ensure consistent output.
            While Vertica forgoes the use of indexes for enhanced
            performance, it does not guarantee a specific order of
            data retrieval.

        .. seealso::

            | ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.append` : Append a
                :py:class:`~vDataFrame` with another one or an
                ``input_relation``.
        """
        columns = format_type(columns, dtype=list)
        columns = self.format_colnames(columns)
        max_pos = 0
        for column in self._vars["columns"]:
            max_pos = max(max_pos, len(self[column]._transf) - 1)
        self._vars["order_by"][max_pos] = self._get_sort_syntax(columns)
        return self
