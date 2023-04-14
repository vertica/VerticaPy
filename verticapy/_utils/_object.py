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

This file was created to avoid circular import and to
locate all  the  inner functions imports in only  one
single file.  No other file should have inner imports.
"""
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame
    import verticapy.machine_learning.vertica as vml


def _get_mllib() -> Literal["vml"]:
    import verticapy.machine_learning.vertica as vml

    return vml


def _get_vdf(*args, **kwargs) -> "vDataFrame":
    from verticapy.core.vdataframe.base import vDataFrame

    return vDataFrame(*args, **kwargs)


def _read_pandas(*args, **kwargs) -> "vDataFrame":
    from verticapy.core.parsers.pandas import read_pandas

    return read_pandas(*args, **kwargs)
