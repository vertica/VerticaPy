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
from typing import Literal

from verticapy._utils._collect import save_verticapy_logs
from verticapy._utils._gen import gen_name


class vDFTEXT:
    @save_verticapy_logs
    def regexp(
        self,
        column: str,
        pattern: str,
        method: Literal[
            "count",
            "ilike",
            "instr",
            "like",
            "not_ilike",
            "not_like",
            "replace",
            "substr",
        ] = "substr",
        position: int = 1,
        occurrence: int = 1,
        replacement: str = "",
        return_position: int = 0,
        name: str = "",
    ):
        """
    Computes a new vDataColumn based on regular expressions. 

    Parameters
    ----------
    column: str
        Input vDataColumn to use to compute the regular expression.
    pattern: str
        The regular expression.
    method: str, optional
        Method to use to compute the regular expressions.
            count     : Returns the number times a regular expression matches 
                each element of the input vDataColumn. 
            ilike     : Returns True if the vDataColumn element contains a match 
                for the regular expression.
            instr     : Returns the starting or ending position in a vDataColumn 
                element where a regular expression matches. 
            like      : Returns True if the vDataColumn element matches the regular 
                expression.
            not_ilike : Returns True if the vDataColumn element does not match the 
                case-insensitive regular expression.
            not_like  : Returns True if the vDataColumn element does not contain a 
                match for the regular expression.
            replace   : Replaces all occurrences of a substring that match a 
                regular expression with another substring.
            substr    : Returns the substring that matches a regular expression 
                within a vDataColumn.
    position: int, optional
        The number of characters from the start of the string where the function 
        should start searching for matches.
    occurrence: int, optional
        Controls which occurrence of a pattern match in the string to return.
    replacement: str, optional
        The string to replace matched substrings.
    return_position: int, optional
        Sets the position within the string to return.
    name: str, optional
        New feature name. If empty, a name will be generated.

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame.eval : Evaluates a customized expression.
        """
        column = self._format_colnames(column)
        pattern_str = pattern.replace("'", "''")
        expr = f"REGEXP_{method.upper()}({column}, '{pattern_str}'"
        if method == "replace":
            replacement_str = replacement.replace("'", "''")
            expr += f", '{replacement_str}'"
        if method in ("count", "instr", "replace", "substr"):
            expr += f", {position}"
        if method in ("instr", "replace", "substr"):
            expr += f", {occurrence}"
        if method == "instr":
            expr += f", {return_position}"
        expr += ")"
        gen_name([method, column])
        return self.eval(name=name, expr=expr)


class vDCTEXT:
    @save_verticapy_logs
    def str_contains(self, pat: str):
        """
    Verifies if the regular expression is in each of the vDataColumn records. 
    The vDataColumn will be transformed.

    Parameters
    ----------
    pat: str
        Regular expression.

    Returns
    -------
    vDataFrame
        self._parent

    See Also
    --------
    vDataFrame[].str_count   : Computes the number of matches for the regular expression
        in each record of the vDataColumn.
    vDataFrame[].extract     : Extracts the regular expression in each record of the 
        vDataColumn.
    vDataFrame[].str_replace : Replaces the regular expression matches in each of the 
        vDataColumn records by an input value.
    vDataFrame[].str_slice   : Slices the vDataColumn.
        """
        pat = pat.replace("'", "''")
        return self.apply(func=f"REGEXP_COUNT({{}}, '{pat}') > 0")

    @save_verticapy_logs
    def str_count(self, pat: str):
        """
    Computes the number of matches for the regular expression in each record of 
    the vDataColumn. The vDataColumn will be transformed.

    Parameters
    ----------
    pat: str
        regular expression.

    Returns
    -------
    vDataFrame
        self._parent

    See Also
    --------
    vDataFrame[].str_contains : Verifies if the regular expression is in each of the 
        vDataColumn records. 
    vDataFrame[].extract      : Extracts the regular expression in each record of the 
        vDataColumn.
    vDataFrame[].str_replace  : Replaces the regular expression matches in each of the 
        vDataColumn records by an input value.
    vDataFrame[].str_slice    : Slices the vDataColumn.
        """
        pat = pat.replace("'", "''")
        return self.apply(func=f"REGEXP_COUNT({{}}, '{pat}')")

    @save_verticapy_logs
    def str_extract(self, pat: str):
        """
    Extracts the regular expression in each record of the vDataColumn.
    The vDataColumn will be transformed.

    Parameters
    ----------
    pat: str
        regular expression.

    Returns
    -------
    vDataFrame
        self._parent

    See Also
    --------
    vDataFrame[].str_contains : Verifies if the regular expression is in each of the 
        vDataColumn records. 
    vDataFrame[].str_count    : Computes the number of matches for the regular expression
        in each record of the vDataColumn.
    vDataFrame[].str_replace  : Replaces the regular expression matches in each of the 
        vDataColumn records by an input value.
    vDataFrame[].str_slice    : Slices the vDataColumn.
        """
        pat = pat.replace("'", "''")
        return self.apply(func=f"REGEXP_SUBSTR({{}}, '{pat}')")

    @save_verticapy_logs
    def str_replace(self, to_replace: str, value: str = ""):
        """
    Replaces the regular expression matches in each of the vDataColumn record by an
    input value. The vDataColumn will be transformed.

    Parameters
    ----------
    to_replace: str
        Regular expression to replace.
    value: str, optional
        New value.

    Returns
    -------
    vDataFrame
        self._parent

    See Also
    --------
    vDataFrame[].str_contains : Verifies if the regular expression is in each of the 
        vDataColumn records. 
    vDataFrame[].str_count    : Computes the number of matches for the regular expression
        in each record of the vDataColumn.
    vDataFrame[].extract      : Extracts the regular expression in each record of the 
        vDataColumn.
    vDataFrame[].str_slice    : Slices the vDataColumn.
        """
        to_replace = to_replace.replace("'", "''")
        value = value.replace("'", "''")
        return self.apply(func=f"REGEXP_REPLACE({{}}, '{to_replace}', '{value}')")

    @save_verticapy_logs
    def str_slice(self, start: int, step: int):
        """
    Slices the vDataColumn. The vDataColumn will be transformed.

    Parameters
    ----------
    start: int
        Start of the slicing.
    step: int
        Size of the slicing.

    Returns
    -------
    vDataFrame
        self._parent

    See Also
    --------
    vDataFrame[].str_contains : Verifies if the regular expression is in each of the 
        vDataColumn records. 
    vDataFrame[].str_count    : Computes the number of matches for the regular expression
        in each record of the vDataColumn.
    vDataFrame[].extract      : Extracts the regular expression in each record of the 
        vDataColumn.
    vDataFrame[].str_replace  : Replaces the regular expression matches in each of the 
        vDataColumn records by an input value.
        """
        return self.apply(func=f"SUBSTR({{}}, {start}, {step})")
