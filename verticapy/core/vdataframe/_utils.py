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
import copy
from abc import abstractmethod
from typing import Optional, Union

from verticapy._utils._sql._format import format_type, quote_ident
from verticapy._typing import NoneType, SQLExpression
from verticapy.errors import MissingColumn

from verticapy.plotting._utils import PlottingUtils


class vDFUtils(PlottingUtils):
    def __init__(self):
        """Must be overridden in final class"""
        self._vars = {}

    @staticmethod
    def _levenshtein(s: str, t: str) -> int:
        rows = len(s) + 1
        cols = len(t) + 1
        dist = [[0 for x in range(cols)] for x in range(rows)]
        for i in range(1, rows):
            dist[i][0] = i
        for i in range(1, cols):
            dist[0][i] = i
        for col in range(1, cols):
            for row in range(1, rows):
                if s[row - 1] == t[col - 1]:
                    cost = 0
                else:
                    cost = 1
                dist[row][col] = min(
                    dist[row - 1][col] + 1,
                    dist[row][col - 1] + 1,
                    dist[row - 1][col - 1] + cost,
                )
        return dist[row][col]

    def format_colnames(
        self,
        *args,
        columns: Union[None, str, list, dict] = None,
        expected_nb_of_cols: Union[None, int, list] = None,
        raise_error: bool = True,
    ) -> SQLExpression:
        """
        Method used to format the input columns by using the
        vDataFrame columns' names.

        Parameters
        ----------
        *args: str / list / dict, optional
            List of columns' names to format. This allows you
            to use multiple objects as input and to format all
            of them.
            Example:  self.format_colnames(x0, x1, x2) returns
            x0_f, x1_f, x2_f
            where xi_f represents xi correctly formatted.
        columns: SQLColumns / dict, optional
            List of columns' names to format.
        expected_nb_of_cols: int | list
            [Only used for the function first argument]
            List of the expected number of columns.
            Example: If  expected_nb_of_cols is set to [2, 3],
            the parameters 'columns' or the first argument of
            args  should   have   exactly  2  or  3  elements.
            Otherwise, the function will raise an error.
        raise_error: bool, optional
            If set to True and there is an error, the function
            raises the error.

        Returns
        -------
        SQLExpression
            Formatted columns' names.
        """
        if len(args) > 0:
            result = []
            for arg in args:
                result += [self.format_colnames(columns=arg, raise_error=raise_error)]
            if len(args) == 1:
                result = result[0]
        else:
            if isinstance(columns, NoneType):
                return None
            if not columns or isinstance(columns, (int, float)):
                return copy.deepcopy(columns)
            if raise_error:
                if isinstance(columns, str):
                    cols_to_check = [columns]
                else:
                    cols_to_check = copy.deepcopy(columns)
                all_columns = self.get_columns()
                for column in cols_to_check:
                    result = []
                    if column not in all_columns:
                        min_distance, min_distance_op = 1000, ""
                        is_error = True
                        for col in all_columns:
                            if quote_ident(column).lower() == quote_ident(col).lower():
                                is_error = False
                                break
                            else:
                                ldistance = self._levenshtein(column, col)
                                if ldistance < min_distance:
                                    min_distance, min_distance_op = ldistance, col
                        if is_error:
                            error_message = (
                                f"The Virtual Column '{column}' doesn't exist."
                            )
                            if min_distance < 10:
                                error_message += f"\nDid you mean '{min_distance_op}' ?"
                            raise MissingColumn(error_message)

            if isinstance(columns, str):
                result = columns
                vdf_columns = self.get_columns()
                for col in vdf_columns:
                    if quote_ident(columns).lower() == quote_ident(col).lower():
                        result = col
                        break
            elif isinstance(columns, dict):
                result = {}
                for col in columns:
                    key = self.format_colnames(col, raise_error=raise_error)
                    result[key] = columns[col]
            else:
                result = []
                for col in columns:
                    result += [self.format_colnames(col, raise_error=raise_error)]
        if raise_error:
            expected_nb_of_cols = format_type(expected_nb_of_cols, dtype=list)
            if len(expected_nb_of_cols) > 0:
                if len(args) > 0:
                    columns = args[0]
                n = len(columns)
                if n not in expected_nb_of_cols:
                    x = "|".join([str(nb) for nb in expected_nb_of_cols])
                    raise ValueError(
                        f"The number of Virtual Columns expected is [{x}], found {n}."
                    )
        return result

    @abstractmethod
    def get_columns(self) -> str:
        """Must be overridden in child class"""
        raise NotImplementedError

    @staticmethod
    def get_match_index(
        x: str, col_list: list, str_check: bool = True
    ) -> Optional[int]:
        """
        Returns the matching index.
        """
        for idx, col in enumerate(col_list):
            if (str_check and quote_ident(x.lower()) == quote_ident(col.lower())) or (
                x == col
            ):
                return idx

    def is_colname_in(self, column: str) -> bool:
        """
        Method used to check if the input column name is used by
        the vDataFrame.

        Parameters
        ----------
        column: str
            Input column.

        Returns
        -------
        bool
            True if the  column is used by the vDataFrame; false
            otherwise.
        """
        columns = self.get_columns()
        column = quote_ident(column).lower()
        for col in columns:
            if column == quote_ident(col).lower():
                return True
        return False
