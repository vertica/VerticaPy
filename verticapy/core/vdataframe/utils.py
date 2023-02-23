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
from typing import Union

from verticapy._utils._sql._format import quote_ident
from verticapy._utils.math import levenshtein
from verticapy.errors import MissingColumn, ParameterError


class vDFUTILS:
    def _format_colnames(
        self,
        *argv,
        columns: Union[str, list, dict] = [],
        expected_nb_of_cols: Union[int, list] = [],
        raise_error: bool = True,
    ):
        """
    Method used to format the input columns by using the vDataFrame columns' names.

    Parameters
    ----------
    *argv: str / list / dict, optional
        List of columns' names to format. It allows to use as input multiple
        objects and to get all of them formatted.
        Example: self._format_colnames(x0, x1, x2) will return x0_f, x1_f, 
        x2_f where xi_f represents xi correctly formatted.
    columns: str / list / dict, optional
        List of columns' names to format.
    expected_nb_of_cols: int / list
        [Only used for the function first argument]
        List of the expected number of columns.
        Example: If expected_nb_of_cols is set to [2, 3], the parameters
        'columns' or the first argument of argv should have exactly 2 or
        3 elements. Otherwise, the function will raise an error.
    raise_error: bool, optional
        If set to True and if there is an error, it will be raised.

    Returns
    -------
    str / list
        Formatted columns' names.
        """
        if argv:
            result = []
            for arg in argv:
                result += [self._format_colnames(columns=arg, raise_error=raise_error)]
            if len(argv) == 1:
                result = result[0]
        else:
            if not (columns) or isinstance(columns, (int, float)):
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
                                ldistance = levenshtein(column, col)
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
                    key = self._format_colnames(col, raise_error=raise_error)
                    result[key] = columns[col]
            else:
                result = []
                for col in columns:
                    result += [self._format_colnames(col, raise_error=raise_error)]
        if raise_error:
            if isinstance(expected_nb_of_cols, int):
                expected_nb_of_cols = [expected_nb_of_cols]
            if len(expected_nb_of_cols) > 0:
                if len(argv) > 0:
                    columns = argv[0]
                n = len(columns)
                if n not in expected_nb_of_cols:
                    x = "|".join([str(nb) for nb in expected_nb_of_cols])
                    raise ParameterError(
                        f"The number of Virtual Columns expected is [{x}], found {n}."
                    )
        return result

    def is_colname_in(self, column: str):
        """
    Method used to check if the input column name is used by the vDataFrame.
    If not, the function raises an error.

    Parameters
    ----------
    column: str
        Input column.

    Returns
    -------
    bool
        True if the column is used by the vDataFrame
        False otherwise.
        """
        columns = self.get_columns()
        column = quote_ident(column).lower()
        for col in columns:
            if column == quote_ident(col).lower():
                return True
        return False

    def get_nearest_column(self, column: str):
        """
    Method used to find the nearest column's name to the input one.

    Parameters
    ----------
    column: str
        Input column.

    Returns
    -------
    tuple
        (nearest column, levenstein distance)
        """
        columns = self.get_columns()
        col = column.replace('"', "").lower()
        result = (columns[0], levenshtein(col, columns[0].replace('"', "").lower()))
        if len(columns) == 1:
            return result
        for col in columns:
            if col != result[0]:
                current_col = col.replace('"', "").lower()
                d = levenshtein(current_col, col)
                if result[1] > d:
                    result = (col, d)
        return result
