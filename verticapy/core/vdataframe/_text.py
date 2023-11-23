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
from typing import Literal, Optional, TYPE_CHECKING

from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._gen import gen_name

from verticapy.core.vdataframe._rolling import vDFRolling
from verticapy.core.vdataframe._corr import vDCCorr

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame


class vDFText(vDFRolling):
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
        replacement: Optional[str] = None,
        return_position: int = 0,
        name: Optional[str] = None,
    ) -> "vDataFrame":
        """
        Computes a new vDataColumn based on regular expressions.

        Parameters
        ----------
        column: str
            Input vDataColumn  used  to compute the  regular
            expression.
        pattern: str
            The regular expression.
        method: str, optional
            Method used to compute the regular  expressions.

                 - count:
                    Returns the number of times a
                    regular expression matches each
                    element of the input vDataColumn.
                 - ilike:
                    Returns  True if  the  vDataColumn
                    element  contains a match for  the
                    regular expression.
                 - instr:
                    Returns  the  starting  or  ending
                    position in  a vDataColumn element
                    where a regular expression matches.
                 - like:
                    Returns  True  if the  vDataColumn
                    element    matches   the   regular
                    expression.
                 - not_ilike :
                    Returns  True  if the  vDataColumn
                    element  does  not match the  case
                    -insensitive  regular   expression.
                 - not_like:
                    Returns  True if  the  vDataColumn
                    element  does not contain a  match
                    for the regular expression.
                 - replace:
                    Replaces   all  occurrences  of  a
                    substring  that  match  a  regular
                    expression  with another substring.
                 - substr:
                    Returns the substring that matches
                    a  regular   expression  within  a
                    vDataColumn.

        position: int, optional
            The number of characters from the start of the string
            where the function should start searching for matches.
        occurrence: int, optional
            Controls  which occurrence of a pattern match in  the
            string to return.
        replacement: str, optional
            The string to replace matched substrings.
        return_position: int, optional
            Sets the position within the string to return.
        name: str, optional
            New feature name. If empty, a name is generated.

        Returns
        -------
        vDataFrame
            self

        Examples
        ---------

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp


        Let's generate a small dataset using the following data:

        .. ipython:: python

            data = vp.vDataFrame(
                {
                    "rollno": ['1', '2', '3', '4'],
                    "subjects": [
                        'English, Math',
                        'English, Math, Computer',
                        'Math, Computer, Science',
                        'Math, Science',
                    ],
                }
            )

        Let's retrieve the second subject.

        .. code-block:: python

            data.regexp(
                column = "subjects",
                pattern = "[^,]+",
                method = "substr",
                occurrence = 2,
                name = "subject_2").select(
                    [
                        "subjects",
                        "subject_2",
                    ]
                )

        .. ipython:: python
            :suppress:

            res = data.regexp(
                column = "subjects",
                pattern = "[^,]+",
                method = "substr",
                occurrence = 2,
                name = "subject_2").select(
                    [
                        "subjects",
                        "subject_2",
                    ]
                )
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_text_regex1.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_text_regex1.html

        Let's count the number of subjects.

        .. code-block:: python

            data.regexp(
                column = "subjects",
                pattern = ",",
                method = "count",
                name = "nb_subjects",
            )
            data["nb_subjects"].add(1)
            data.select(["subjects", "nb_subjects"])

        .. ipython:: python
            :suppress:

            data.regexp(
                column = "subjects",
                pattern = ",",
                method = "count",
                name = "nb_subjects",
            )
            data["nb_subjects"].add(1)
            res = data.select(["subjects", "nb_subjects"])
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_text_regex2.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_text_regex2.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.eval` : Evaluates an expression.

        """
        column = self.format_colnames(column)
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


class vDCText(vDCCorr):
    @save_verticapy_logs
    def str_contains(self, pat: str) -> "vDataFrame":
        """
        Verifies  if the  regular expression  is in each of  the
        vDataColumn records. The vDataColumn will be transformed.

        Parameters
        ----------
        pat: str
            Regular expression.

        Returns
        -------
        vDataFrame
            self._parent

        Examples
        ---------

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp


        Let's generate a small dataset using the following data:

        .. ipython:: python

            data = vp.vDataFrame(
                {
                    "rollno": ['1', '2', '3', '4'],
                    "subjects": [
                        'English, Math',
                        'English, Math, Computer',
                        'Math, Computer, Science',
                        'Math, Science',
                    ],
                }
            )

        Let's retrieve the second subject.

        .. code-block:: python

            data["subjects"].str_contains(pat = "English").select(
                [
                    "rollno",
                    "subjects as has_english",
                ]
            )

        .. ipython:: python
            :suppress:

            res = data["subjects"].str_contains(pat = "English").select(
                [
                    "rollno",
                    "subjects as has_english",
                ]
            )
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_text_str_contains.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_text_str_contains.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.str_count` : Counts occurrences matching the regular expression.
            | :py:meth:`verticapy.vDataFrame.str_extract` : Extracts the Regular Expression.
            | :py:meth:`verticapy.vDataFrame.str_replace` : Replaces the Regular Expression.
            | :py:meth:`verticapy.vDataFrame.str_slice` : Slices the Regular Expression.
        """
        pat = pat.replace("'", "''")
        return self.apply(func=f"REGEXP_COUNT({{}}, '{pat}') > 0")

    @save_verticapy_logs
    def str_count(self, pat: str) -> "vDataFrame":
        """
        Computes the number of matches for the regular expression in
        each  record  of  the vDataColumn.  The vDataColumn will  be
        transformed.

        Parameters
        ----------
        pat: str
            regular expression.

        Returns
        -------
        vDataFrame
            self._parent

        Examples
        ---------

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        Let's generate a small dataset using the following data:

        .. ipython:: python

            data = vp.vDataFrame(
                {
                    "rollno": ['1', '2', '3', '4'],
                    "subjects": [
                        'English, Math',
                        'English, Math, Computer',
                        'Math, Computer, Science',
                        'Math, Science',
                    ],
                }
            )

        Let's count number of times "English" appears in "subjects"
        vDataColumn.

        .. code-block:: python

            data["subjects"].str_count(pat = "English").select(
                [
                    "rollno",
                    "subjects as english_count",
                ]
            )

        .. ipython:: python
            :suppress:

            res = data["subjects"].str_count(pat = "English").select(
                [
                    "rollno",
                    "subjects as english_count",
                ]
            )
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_text_str_count.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_text_str_count.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.str_contains` : Validates the regular expression.
            | :py:meth:`verticapy.vDataFrame.str_extract` : Extracts the Regular Expression.
            | :py:meth:`verticapy.vDataFrame.str_replace` : Replaces the Regular Expression.
            | :py:meth:`verticapy.vDataFrame.str_slice` : Slices the Regular Expression.
        """
        pat = pat.replace("'", "''")
        return self.apply(func=f"REGEXP_COUNT({{}}, '{pat}')")

    @save_verticapy_logs
    def str_extract(self, pat: str) -> "vDataFrame":
        """
        Extracts  the regular  expression in  each record of
        the vDataColumn. The vDataColumn will be transformed.

        Parameters
        ----------
        pat: str
            regular expression.

        Returns
        -------
        vDataFrame
            self._parent

        Examples
        ---------

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp


        Let's generate a small dataset using the following data:

        .. ipython:: python

            data = vp.vDataFrame(
                {
                    "name": [
                        'Mr. Steve Smith',
                        'Mr. Charlie Dickens',
                        'Mrs. Helen Ross',
                        'Dr. Jack Smith',
                    ]
                }
            )

        Let's extract the name prefix.

        .. code-block:: python

            data["name"].str_extract(pat = "([A-Za-z])+\\.")

        .. ipython:: python
            :suppress:
            :okwarning:

            res = data["name"].str_extract(pat = "([A-Za-z])+\\.")
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_text_str_extract.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_text_str_extract.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.str_contains` : Validates the regular expression.
            | :py:meth:`verticapy.vDataFrame.str_count` : Counts occurrences matching the regular expression.
            | :py:meth:`verticapy.vDataFrame.str_replace` : Replaces the Regular Expression.
            | :py:meth:`verticapy.vDataFrame.str_slice` : Slices the Regular Expression.
        """
        pat = pat.replace("'", "''")
        return self.apply(func=f"REGEXP_SUBSTR({{}}, '{pat}')")

    @save_verticapy_logs
    def str_replace(self, to_replace: str, value: Optional[str] = None) -> "vDataFrame":
        """
        Replaces  the  regular expression matches in each  of  the
        vDataColumn record by an input value. The vDataColumn will
        be transformed.

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

        Examples
        ---------

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        Let's generate a small dataset using the following data:

        .. ipython:: python

            data = vp.vDataFrame(
                {
                    "name": [
                        'Mr. Steve Smith',
                        'Mr. Charlie Dickens',
                        'Mrs. Helen Ross',
                        'Dr. Jack Smith',
                    ]
                }
            )

        Let's replace the name prefix with static text
        "[Name_Prefix]".

        .. code-block:: python

            data["name"].str_replace(
                to_replace  = "([A-Za-z])+\\.",
                value = "[Name_Prefix]"
            )

        .. ipython:: python
            :suppress:
            :okwarning:

            res = data["name"].str_replace(
                to_replace  = "([A-Za-z])+\\.",
                value = "[Name_Prefix]"
            )
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_text_str_replace.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_text_str_replace.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.str_contains` : Validates the regular expression.
            | :py:meth:`verticapy.vDataFrame.str_count` : Counts occurrences matching the regular expression.
            | :py:meth:`verticapy.vDataFrame.str_extract` : Extracts the Regular Expression.
            | :py:meth:`verticapy.vDataFrame.str_slice` : Slices the Regular Expression.
        """
        to_replace = to_replace.replace("'", "''")
        value = value.replace("'", "''")
        return self.apply(func=f"REGEXP_REPLACE({{}}, '{to_replace}', '{value}')")

    @save_verticapy_logs
    def str_slice(self, start: int, step: int) -> "vDataFrame":
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

        Examples
        ---------

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp


        Let's generate a small dataset using the following data:

        .. ipython:: python

            data = vp.vDataFrame(
                {
                    "name": [
                        'Mr. Steve Smith',
                        'Mr. Charlie Dickens',
                        'Mrs. Helen Ross',
                        'Dr. Jack Smith',
                    ]
                }
            )

        Let's extract the first 3 alphabets of name.

        .. code-block:: python

            data["name"].str_slice(start = 0, step =3)

        .. ipython:: python
            :suppress:

            res = data["name"].str_slice(start = 0, step =3)
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_text_str_slice.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_text_str_slice.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.str_contains` : Validates the regular expression.
            | :py:meth:`verticapy.vDataFrame.str_count` : Counts occurrences matching the regular expression.
            | :py:meth:`verticapy.vDataFrame.str_replace` : Replaces the Regular Expression.
            | :py:meth:`verticapy.vDataFrame.str_extract` : Extracts the Regular Expression.
        """
        return self.apply(func=f"SUBSTR({{}}, {start}, {step})")
