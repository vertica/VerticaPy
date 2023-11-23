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
import datetime
import decimal
from typing import Annotated, Literal, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.pyplot import Figure as mFigure
    from plotly.graph_objs._figure import Figure
    from vertica_highcharts import Highchart, Highstock

    from verticapy.core.vdataframe.base import vDataFrame
    from verticapy.core.string_sql.base import StringSQL
    from verticapy.core.tablesample.base import TableSample

    from verticapy.plotting.base import PlottingBase


# Pythonic data types.

ArrayLike = Annotated[Union[list, np.ndarray], "Array Like Structure"]
NoneType = type(None)
PythonNumber = Annotated[Union[int, float, decimal.Decimal], "Python Numbers"]
PythonScalar = Annotated[
    Union[bool, float, str, datetime.timedelta, datetime.datetime],
    "Python Scalar",
]
TimeInterval = Annotated[Union[str, datetime.timedelta], "Time Interval"]
Datetime = Annotated[Union[str, datetime.datetime], ""]

# SQL data types.

SQLColumns = Annotated[
    Union[str, list[str]], "STRING representing one column or a list of columns"
]
SQLExpression = Annotated[Union[str, list[str], "StringSQL", list["StringSQL"]], ""]
SQLRelation = Annotated[Union[str, "vDataFrame"], ""]

# Plotting data types.
HChart = Union["Highchart", "Highstock"]
PlottingObject = Union[
    "PlottingBase", "TableSample", "Axes", "mFigure", "Highchart", "Highstock", "Figure"
]
PlottingMethod = Union[Literal["density", "count", "avg", "min", "max", "sum"], str]
ColorType = str
