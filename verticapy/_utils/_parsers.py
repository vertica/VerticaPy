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
import io
import os
from typing import List
import warnings

import verticapy._config.config as conf
from verticapy._utils._sql._format import list_strip

if conf.get_import_success("graphviz"):
    import graphviz
    from graphviz import Source

if conf.get_import_success("IPython"):
    from IPython.display import display

# CSV


def get_header_names(
    path: str, sep: str, record_terminator: str = os.linesep
) -> list[str]:
    """
    Returns the input CSV file's
    header columns' names.

    Parameters
    ----------
    path: str
        File's path.
    sep: str
        CSV separator.

    Returns
    -------
    list
        header columns' names.

    Examples
    --------
    The following code demonstrates
    the usage of the function.

    .. ipython:: python

        # Import the function.
        from verticapy._utils._parsers import get_header_names

        # Creating a CSV example.
        file_name = 'verticapy_test_parsers.csv'
        f = open(file_name, 'a')
        f.write("A;B;C;D&1;2;3;4&")
        f.close()

        # Example.
        get_header_names(file_name, sep = ';',
                        record_terminator='&')

        # Deleting the CSV file.
        import os

        os.remove(file_name)

    .. note::

        These functions serve as utilities to
        construct others, simplifying the overall
        code.
    """
    file_header = get_first_record_as_list(path, sep, record_terminator)

    for idx, col in enumerate(file_header):
        if col == "":
            if idx == 0:
                position = "beginning"
            elif idx == len(file_header) - 1:
                position = "end"
            else:
                position = "middle"
            file_header[idx] = f"col{idx}"
            warning_message = (
                f"An inconsistent name was found in the {position} of the "
                "file header (isolated separator). It will be replaced "
                f"by col{idx}."
            )
            if idx == 0:
                warning_message += (
                    "\nThis can happen when exporting a pandas DataFrame "
                    "to CSV while retaining its indexes.\nTip: Use "
                    "index=False when exporting with pandas.DataFrame.to_csv."
                )
            warnings.warn(warning_message, Warning)
    return list_strip(file_header)


def guess_sep(file_str: str) -> str:
    """
    Guesses the file's separator.

    Parameters
    ----------
    file_str: str
        Any lines of the CSV file.

    Returns
    -------
    str
        the separator.

    Examples
    --------
    The following code demonstrates
    the usage of the function.

    .. ipython:: python

        # Import the function.
        from verticapy._utils._parsers import guess_sep

        # ',' separator.
        guess_sep('col1, col2,col3,  col4')

        # ';' separator.
        guess_sep('col1; col2;col3;  col4')

    .. note::

        These functions serve as utilities to
        construct others, simplifying the overall
        code.
    """
    sep = ","
    max_occur = file_str.count(",")
    for s in ("|", ";"):
        total_occurences = file_str.count(s)
        if total_occurences > max_occur:
            max_occur = total_occurences
            sep = s
    return sep


def get_first_record_as_list(path: str, sep: str, record_terminator: str) -> List[str]:
    """
    Reads the first record from a file and splits it into a list.

    Parameters
    ------------
    path: str
        The name of the file to open
    sep: str
        The field seperator in the delimited file. Example: `','`
    record_terminator: str
        The string marking the end of a record. Example: `'\\n'`

    Returns
    ----------
    A list of strings. Each item in the list is a value in the first record of the file.

    Examples
    -------------------

    .. code-block:: python

        # import the function
        from verticapy._utils._parsers import get_first_record_as_list

        # If you have a file called 'test.csv'
        # with a first line that looks like
        # col1,col2,col3
        cols = get_first_record_as_list('test.csv', ',', '\\n')
        print(cols)
        # Should print
        #    ['col1', 'col2', 'col3']

    """
    first_line = read_first_record(path, record_terminator)
    file_header = first_line.replace(record_terminator, "").replace('"', "")
    if not sep:
        sep = guess_sep(file_header)
    return file_header.split(sep)


def read_first_record(path: str, record_terminator: str) -> str:
    """
    Reads the first record of a file and return it as a string. Includes
    the record terminator in the string returned.

    Parameters
    ------------
    path: str
        The path to the file. The file will be opened.
    record_terminator: str
        The string that marks the end of a record.

    Returns
    ----------
    A string that is the full first record.

    Examples
    ------------
    .. code-block:: python

        # import the function
        from verticapy._utils._parsers import read_first_record

        # If you have a file called 'test.csv'
        # with a first two lines that looks like
        # col1,col2,col3;
        # 100,abc,3.14;
        r = read_first_record('test.csv', ',', ';')
        print(r)
        # Should print
        #    'col1,col2,col3;'
    """
    with open(path, "r", encoding="utf-8") as file_obj:
        # readline will read up to a end of line. The value that determines the end of line
        # is set by open(). open() takes an argument of newline, but will only accept
        # certain common values like \n and \r\n. We use readline when we can.
        if record_terminator == os.linesep:
            return file_obj.readline()

        # record separator is special
        # need manual handling
        buf = io.StringIO()

        # We could pick any size chunk to read... 1024 seems
        # reasonably large for csv files. We expect that there
        # is some other buffering happening within the file_obj
        characters_per_read = 1024
        total_characters_read = 0
        while True:
            # Read some bytes, look for end of line
            line = file_obj.read(characters_per_read)
            total_characters_read += characters_per_read
            if line == "":
                raise ValueError(
                    f"Unable to find record terminator "
                    f"{record_terminator} in {total_characters_read}"
                    f"characters of input file {path}"
                )
            buf.write(line)
            current_value = buf.getvalue()
            pos_of_term = current_value.find(record_terminator)
            if pos_of_term > 0:
                # Slice 0:m returns characters from position 0 to m exclusive
                # so we need len(sep) more to include the terminator
                # because readline includes the terminator
                return current_value[0 : (pos_of_term + len(record_terminator))]


# Query Plan


def parse_explain_graphviz_text(rows: list[str]) -> list[str]:
    """
    Parses the explain plan and returns a ``list``
    of text elements.

    Parameters
    ------------
    rows: list
        Result of the explain plan.

    Returns
    ----------
    A list of text elements.
    """
    rows = [row[0] for row in rows]
    rows = "\n".join(rows)
    splits = rows.split("digraph G {")
    result = []
    for row in splits:
        if row.startswith("\ngraph ["):
            tmp_result = row.split("}")
            graphviz_tree = "digraph G {" + tmp_result[0] + "}"
            result += [graphviz_tree]
            result += ["}".join(tmp_result[1:])]
        else:
            result += [row]
    return result


def parse_explain_graphviz(rows: list[str], display_trees: bool = True) -> list:
    """
    Parses the explain plan and returns a ``list``
    of elements including trees and titles.

    Parameters
    ------------
    rows: list
        Result of the explain plan.

    Returns
    ----------
    A list of trees and titles.
    """
    rows = parse_explain_graphviz_text(rows)
    result = []
    for row in rows:
        if conf.get_import_success("graphviz") and row.startswith("digraph G {"):
            result += [Source(row)]
        else:
            result += [row]
    if display_trees:
        for row in result:
            if isinstance(row, str) or not (conf.get_import_success("IPython")):
                print(row)
            else:
                display(row)
    return result
