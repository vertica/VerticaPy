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
from typing import Literal
import warnings

import verticapy._config.config as conf

if conf.get_import_success("IPython"):
    from IPython.display import display, HTML


def print_message(
    message: str, mtype: Literal["print", "warning", "display"] = "print"
) -> None:
    """
    Prints the input message or warning.
    This function is used to manage the
    verbosity.
    """
    mtype = mtype.lower().strip()
    if mtype == "warning":
        warnings.warn(message, Warning)
    elif mtype == "print" and conf.get_option("print_info"):
        print(message)
    elif mtype == "display" and conf.get_option("print_info"):
        display(HTML(message))
