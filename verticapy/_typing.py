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
import datetime, decimal
from typing import Annotated, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame
    from verticapy.core.string_sql.base import StringSQL

# Pythonic data types.

ArrayLike = Annotated[Union[list, np.ndarray], "Array Like Structure"]
PythonNumber = Annotated[Union[None, int, float, decimal.Decimal], "Python Numbers"]
PythonScalar = Annotated[
    Union[None, bool, float, str, datetime.timedelta, datetime.datetime],
    "Python Scalar",
]
TimeInterval = Annotated[Union[None, str, datetime.timedelta], "Time Interval"]
Datetime = Annotated[Union[None, str, datetime.datetime], ""]

# SQL data types.

SQLColumns = Annotated[
    Union[str, list[str]], "STRING representing one column or a list of columns"
]
SQLExpression = Annotated[Union[str, list[str], "StringSQL", list["StringSQL"]], ""]
SQLRelation = Annotated[Union[str, "vDataFrame"], ""]
