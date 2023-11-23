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

        Parameters
        ----------
        input_relation: SQLRelation
            Relation to merge with.
        expr1: SQLExpression, optional
            List of pure-SQL expressions from the current vDataFrame
            to use during merging. For example,  'CASE WHEN "column"
            > 3 THEN 2 ELSE NULL END' and  'POWER("column", 2)' will
            work. If empty, all vDataFrame vDataColumns are used.
            Aliases are recommended to avoid auto-naming.
        expr2: SQLExpression, optional
            List of pure-SQL  expressions from the input relation to
            use during the merging.
            For example, 'CASE WHEN "column" > 3 THEN 2 ELSE NULL END'
            and 'POWER("column", 2)'  will work.  If empty, all input
            relation columns are  used. Aliases  are  recommended to
            avoid auto-naming.
        union_all: bool, optional
            If  set to True, the  vDataFrame is merged with the input
            relation using an 'UNION ALL' instead of an 'UNION'.

        Returns
        -------
        vDataFrame
           vDataFrame of the Union
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
        Joins the vDataFrame with another one or an input relation.

        \u26A0 Warning : Joins  can  make  the  vDataFrame  structure
                         heavier.  It is recommended that you check
                         the    current     structure    using    the
                         'current_relation'  method  and  save  it
                         with the 'to_db' method, using the parameters
                         'inplace = True' and 'relation_type = table'.

        Parameters
        ----------
        input_relation: SQLRelation
            Relation to join with.
        on: tuple / dict / list, optional
            If using a list:
            List of 3-tuples. Each tuple must include (key1, key2, operator)
            —where key1 is the key of the vDataFrame, key2 is the key of the
            input relation, and operator is one of the following:
                         '=' : exact match
                         '<' : key1  < key2
                         '>' : key1  > key2
                        '<=' : key1 <= key2
                        '>=' : key1 >= key2
                     'llike' : key1 LIKE '%' || key2 || '%'
                     'rlike' : key2 LIKE '%' || key1 || '%'
               'linterpolate': key1 INTERPOLATE key2
               'rinterpolate': key2 INTERPOLATE key1
            Some operators need 5-tuples: (key1, key2, operator, operator2, x)
            —where  operator2 is  a simple operator (=, >, <, <=, >=), x is  a
            float or an integer, and operator is one of the following:
                     'jaro' : JARO(key1, key2) operator2 x
                    'jarow' : JARO_WINCKLER(key1, key2) operator2 x
                      'lev' : LEVENSHTEIN(key1, key2) operator2 x

            If using a dictionary:
            This parameter must include all the different keys. It must be
            similar to the following:
            {"relationA_key1": "relationB_key1" ...,
             "relationA_keyk": "relationB_keyk"}
            where relationA is the current vDataFrame and relationB is the
            input relation or the input vDataFrame.
        on_interpolate: dict, optional
            Dictionary of all unique keys. This is used to join two event
            series together using some ordered attribute. Event series
            joins let you compare values from two series directly, rather
            than having to normalize the series  to the same measurement
            interval. The dict must be similar to the following:
            {"relationA_key1": "relationB_key1" ...,
             "relationA_keyk": "relationB_keyk"}
            where relationA is the  current vDataFrame and relationB is the
            input relation or the input vDataFrame.
        how: str, optional
            Join Type.
                left    : Left Join.
                right   : Right Join.
                cross   : Cross Join.
                full    : Full Outer Join.
                natural : Natural Join.
                inner   : Inner Join.
        expr1: SQLExpression, optional
            List  of the different columns in pure  SQL to select  from the
            current   vDataFrame,   optionally  as  aliases.   Aliases  are
            recommended to avoid  ambiguous names. For example: 'column' or
            'column AS my_new_alias'.
        expr2: SQLExpression, optional
            List  of the different  columns in pure SQL to select from  the
            input  relation optionally as aliases. Aliases are  recommended
            to avoid  ambiguous names.  For example: 'column' or 'column AS
            my_new_alias'.

        Returns
        -------
        vDataFrame
            object result of the join.
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
        Sorts the vDataFrame using the input vDataColumns.

        Parameters
        ----------
        columns: SQLColumns / dict
            List  of the  vDataColumns  used to sort  the data,
            using asc order or dictionary of all sorting methods.
            For example,  to sort by  "column1" ASC and "column2"
            DESC,  write  {"column1": "asc", "column2": "desc"}

        Returns
        -------
        vDataFrame
            self
        """
        columns = format_type(columns, dtype=list)
        columns = self.format_colnames(columns)
        max_pos = 0
        for column in self._vars["columns"]:
            max_pos = max(max_pos, len(self[column]._transf) - 1)
        self._vars["order_by"][max_pos] = self._get_sort_syntax(columns)
        return self
