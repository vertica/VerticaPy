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
from verticapy._utils._sql._format import clean_query, format_magic

from verticapy.core.string_sql.base import StringSQL


def date(expr):
    """
Converts the input value to a DATE data type.

Parameters
----------
expr: object
    Expression.

Returns
-------
StringSQL
    SQL expression.
    """
    expr = format_magic(expr)
    return StringSQL(f"DATE({expr})", "date")


def day(expr):
    """
Returns as an integer the day of the month from the input expression. 

Parameters
----------
expr: object
    Expression.

Returns
-------
StringSQL
    SQL expression.
    """
    expr = format_magic(expr)
    return StringSQL(f"DAY({expr})", "float")


def dayofweek(expr):
    """
Returns the day of the week as an integer, where Sunday is day 1.

Parameters
----------
expr: object
    Expression.

Returns
-------
StringSQL
    SQL expression.
    """
    expr = format_magic(expr)
    return StringSQL(f"DAYOFWEEK({expr})", "float")


def dayofyear(expr):
    """
Returns the day of the year as an integer, where January 1 is day 1.

Parameters
----------
expr: object
    Expression.

Returns
-------
StringSQL
    SQL expression.
    """
    expr = format_magic(expr)
    return StringSQL(f"DAYOFYEAR({expr})", "float")


def extract(expr, field: str):
    """
Extracts a sub-field such as year or hour from a date/time expression.

Parameters
----------
expr: object
    Expression.
field: str
    The field to extract. It must be one of the following: 
 		CENTURY / DAY / DECADE / DOQ / DOW / DOY / EPOCH / HOUR / ISODOW / ISOWEEK /
 		ISOYEAR / MICROSECONDS / MILLENNIUM / MILLISECONDS / MINUTE / MONTH / QUARTER / 
 		SECOND / TIME ZONE / TIMEZONE_HOUR / TIMEZONE_MINUTE / WEEK / YEAR

Returns
-------
StringSQL
    SQL expression.
    """
    expr = format_magic(expr)
    return StringSQL(f"DATE_PART('{field}', {expr})", "int")


def getdate():
    """
Returns the current statement's start date and time as a TIMESTAMP value.

Returns
-------
StringSQL
    SQL expression.
    """
    return StringSQL("GETDATE()", "date")


def getutcdate():
    """
Returns the current statement's start date and time at TIME ZONE 'UTC' 
as a TIMESTAMP value.

Returns
-------
StringSQL
    SQL expression.
    """
    return StringSQL("GETUTCDATE()", "date")


def hour(expr):
    """
Returns the hour portion of the specified date as an integer, where 0 is 
00:00 to 00:59.

Parameters
----------
expr: object
    Expression.

Returns
-------
StringSQL
    SQL expression.
    """
    expr = format_magic(expr)
    return StringSQL(f"HOUR({expr})", "int")


def interval(expr):
    """
Converts the input value to a INTERVAL data type.

Parameters
----------
expr: object
    Expression.

Returns
-------
StringSQL
    SQL expression.
    """
    expr = format_magic(expr)
    return StringSQL(f"({expr})::interval", "interval")


def minute(expr):
    """
Returns the minute portion of the specified date as an integer.

Parameters
----------
expr: object
    Expression.

Returns
-------
StringSQL
    SQL expression.
    """
    expr = format_magic(expr)
    return StringSQL(f"MINUTE({expr})", "int")


def microsecond(expr):
    """
Returns the microsecond portion of the specified date as an integer.

Parameters
----------
expr: object
    Expression.

Returns
-------
StringSQL
    SQL expression.
    """
    expr = format_magic(expr)
    return StringSQL(f"MICROSECOND({expr})", "int")


def month(expr):
    """
Returns the month portion of the specified date as an integer.

Parameters
----------
expr: object
    Expression.

Returns
-------
StringSQL
    SQL expression.
    """
    expr = format_magic(expr)
    return StringSQL(f"MONTH({expr})", "int")


def overlaps(start0, end0, start1, end1):
    """
Evaluates two time periods and returns true when they overlap, false 
otherwise.

Parameters
----------
start0: object
    DATE, TIME, or TIMESTAMP/TIMESTAMPTZ value that specifies the beginning 
    of a time period.
end0: object
    DATE, TIME, or TIMESTAMP/TIMESTAMPTZ value that specifies the end of a 
    time period.
start1: object
    DATE, TIME, or TIMESTAMP/TIMESTAMPTZ value that specifies the beginning 
    of a time period.
end1: object
    DATE, TIME, or TIMESTAMP/TIMESTAMPTZ value that specifies the end of a 
    time period.

Returns
-------
StringSQL
    SQL expression.
    """
    expr = f"""
        ({format_magic(start0)},
         {format_magic(end0)})
         OVERLAPS
        ({format_magic(start1)},
         {format_magic(end1)})"""
    return StringSQL(clean_query(expr), "int")


def quarter(expr):
    """
Returns calendar quarter of the specified date as an integer, where the 
January-March quarter is 1.

Parameters
----------
expr: object
    Expression.

Returns
-------
StringSQL
    SQL expression.
    """
    expr = format_magic(expr)
    return StringSQL(f"QUARTER({expr})", "int")


def round_date(expr, precision: str = "DD"):
    """
Rounds the specified date or time.

Parameters
----------
expr: object
    Expression.
precision: str, optional
    A string constant that specifies precision for the rounded value, 
    one of the following:
	    Century: CC | SCC
	    Year: SYYY | YYYY | YEAR | YYY | YY | Y
	    ISO Year: IYYY | IYY | IY | I
	    Quarter: Q
	    Month: MONTH | MON | MM | RM
	    Same weekday as first day of year: WW
	    Same weekday as first day of ISO year: IW
	    Same weekday as first day of month: W
	    Day (default): DDD | DD | J
	    First weekday: DAY | DY | D
	    Hour: HH | HH12 | HH24
	    Minute: MI
	    Second: SS

Returns
-------
StringSQL
    SQL expression.
    """
    expr = format_magic(expr)
    return StringSQL(f"ROUND({expr}, '{precision}')", "date")


def second(expr):
    """
Returns the seconds portion of the specified date as an integer.

Parameters
----------
expr: object
    Expression.

Returns
-------
StringSQL
    SQL expression.
    """
    expr = format_magic(expr)
    return StringSQL(f"SECOND({expr})", "int")


def timestamp(expr):
    """
Converts the input value to a TIMESTAMP data type.

Parameters
----------
expr: object
    Expression.

Returns
-------
StringSQL
    SQL expression.
    """
    expr = format_magic(expr)
    return StringSQL(f"({expr})::timestamp", "date")


def week(expr):
    """
Returns the week of the year for the specified date as an integer, where the 
first week begins on the first Sunday on or preceding January 1.

Parameters
----------
expr: object
    Expression.

Returns
-------
StringSQL
    SQL expression.
    """
    expr = format_magic(expr)
    return StringSQL(f"WEEK({expr})", "int")


def year(expr):
    """
Returns an integer that represents the year portion of the specified date.

Parameters
----------
expr: object
    Expression.

Returns
-------
StringSQL
    SQL expression.
    """
    expr = format_magic(expr)
    return StringSQL(f"YEAR({expr})", "int")
