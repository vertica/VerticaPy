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
from verticapy.errors import ParsingError


def get_magic_options(line: str) -> dict:
    """
    Parses the input line and returns the dictionary
    of options.
    """

    # parsing the line
    i, n, splits = 0, len(line), []
    while i < n:
        while i < n and line[i] == " ":
            i += 1
        if i < n:
            k = i
            op = line[i]
            if op in ('"', "'"):
                i += 1
                while i < n - 1:
                    if line[i] == op and line[i + 1] != op:
                        break
                    i += 1
                i += 1
                quote_in = True
            else:
                while i < n and line[i] != " ":
                    i += 1
                quote_in = False
            if quote_in:
                splits += [line[k + 1 : i - 1]]
            else:
                splits += [line[k:i]]

    # Creating the dictionary
    n, i, options_dict = len(splits), 0, {}
    while i < n:
        if splits[i][0] != "-":
            raise ParsingError(
                f"Can not parse option '{splits[i][0]}'. "
                "Options must start with '-'."
            )
        options_dict[splits[i]] = splits[i + 1]
        i += 2

    return options_dict
