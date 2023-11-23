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
from verticapy._typing import SQLExpression
from verticapy._utils._sql._format import clean_query, format_magic

from verticapy.core.string_sql.base import StringSQL


def date(expr: SQLExpression) -> StringSQL:
    """
    Converts the input value to a DATE data type.

    Parameters
    ----------
    expr: SQLExpression
        Expression.

    Returns
    -------
    StringSQL
        SQL string.

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": ['11-03-1993 12:05:10.23', '09-05-1959 03:10:20.12']})
        # apply the date function, creating a "date_x" column
        df["date_x"] = vpf.date(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": ['11-03-1993 12:05:10.23', '09-05-1959 03:10:20.12']})
        df["date_x"] = vpf.date(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_date_date.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_date_date.html
    """
    expr = format_magic(expr)
    return StringSQL(f"DATE({expr})", "date")


def day(expr: SQLExpression) -> StringSQL:
    """
    Returns the day of the month as an integer.

    Parameters
    ----------
    expr: SQLExpression
        Expression.

    Returns
    -------
    StringSQL
        SQL string.

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": ['11-03-1993 12:00:00', '09-05-1959 03:00:00']})
        df["x"].astype("timestamp")
        # apply the day function, creating a "day_x" column
        df["day_x"] = vpf.day(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": ['11-03-1993 12:00:00', '09-05-1959 03:00:00']})
        df["x"].astype("timestamp")
        df["day_x"] = vpf.day(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_date_day.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_date_day.html
    """
    expr = format_magic(expr)
    return StringSQL(f"DAY({expr})", "float")


def dayofweek(expr: SQLExpression) -> StringSQL:
    """
    Returns the day of the week as an integer,
    where Sunday is day 1.

    Parameters
    ----------
    expr: SQLExpression
        Expression.

    Returns
    -------
    StringSQL
        SQL string.

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": ['11-03-1993 12:00:00', '09-05-1959 03:00:00']})
        df["x"].astype("timestamp")
        # apply the dayofweek function, creating a "dayofweek_x" column
        df["dayofweek_x"] = vpf.dayofweek(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": ['11-03-1993 12:00:00', '09-05-1959 03:00:00']})
        df["x"].astype("timestamp")
        df["dayofweek_x"] = vpf.dayofweek(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_date_dayofweek.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_date_dayofweek.html
    """
    expr = format_magic(expr)
    return StringSQL(f"DAYOFWEEK({expr})", "float")


def dayofyear(expr: SQLExpression) -> StringSQL:
    """
    Returns the day of the year as an integer,
    where January 1 is day 1.

    Parameters
    ----------
    expr: SQLExpression
        Expression.

    Returns
    -------
    StringSQL
        SQL string.

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": ['11-03-1993 12:00:00', '09-05-1959 03:00:00']})
        df["x"].astype("timestamp")
        # apply the dayofyear function, creating a "dayofyear_x" column
        df["dayofyear_x"] = vpf.dayofyear(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": ['11-03-1993 12:00:00', '09-05-1959 03:00:00']})
        df["x"].astype("timestamp")
        df["dayofyear_x"] = vpf.dayofyear(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_date_dayofyear.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_date_dayofyear.html
    """
    expr = format_magic(expr)
    return StringSQL(f"DAYOFYEAR({expr})", "float")


def extract(expr: SQLExpression, field: str) -> StringSQL:
    """
    Extracts a sub-field, such as year or hour, from
    a date/time expression.

    Parameters
    ----------
    expr: SQLExpression
        Expression.
    field: str
        The field to extract. It must be one of the
        following:
        CENTURY, DAY, DECADE, DOQ, DOW,
        DOY,   EPOCH,  HOUR,  ISODOW,   ISOWEEK,
        ISOYEAR,    MICROSECONDS,    MILLENNIUM,
        MILLISECONDS,  MINUTE,  MONTH,  QUARTER,
        SECOND,    TIME ZONE,     TIMEZONE_HOUR,
        TIMEZONE_MINUTE,       WEEK,       YEAR

    Returns
    -------
    StringSQL
        SQL string.

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": ['11-03-1993', '12-03-1993']})
        df["x"].astype("timestamp")
        # apply the extract function, creating a "months" column
        df["months"] = vpf.extract(df["x"], "MONTH")
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": ['11-03-1993', '12-03-1993']})
        df["x"].astype("timestamp")
        df["months"] = vpf.extract(df["x"], "MONTHS")
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_date_extract.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_date_extract.html
    """
    expr = format_magic(expr)
    return StringSQL(f"DATE_PART('{field}', {expr})", "int")


def getdate() -> StringSQL:
    """
    Returns the current statement's start date and time
    as a TIMESTAMP value.

    Returns
    -------
    StringSQL
        SQL string.

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [1, 2, 3, 4]})
        # apply the getdate function, creating a "date" column
        df["date"] = vpf.getdate()
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [1, 2, 3, 4]})
        df["date"] = vpf.getdate()
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_date_getdate.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_date_getdate.html
    """
    return StringSQL("GETDATE()", "date")


def getutcdate() -> StringSQL:
    """
    Returns the current statement's start date and time
    at TIME ZONE 'UTC' as a TIMESTAMP value.

    Returns
    -------
    StringSQL
        SQL string.

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [1, 2, 3, 4]})
        # apply the getutcdate function, creating a "date" column
        df["date"] = vpf.getutcdate()
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [1, 2, 3, 4]})
        df["date"] = vpf.getutcdate()
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_date_getutcdate.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_date_getutcdate.html
    """
    return StringSQL("GETUTCDATE()", "date")


def hour(expr: SQLExpression) -> StringSQL:
    """
    Returns the hour portion of the specified date as
    an integer, where 0 is 00:00 to 00:59.

    Parameters
    ----------
    expr: SQLExpression
        Expression.

    Returns
    -------
    StringSQL
        SQL string.

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": ['11-03-1993 12:00:00', '09-05-1959 03:00:00']})
        df["x"].astype("timestamp")
        # apply the hour function, creating a "hour_x" column
        df["hour_x"] = vpf.hour(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": ['11-03-1993 12:00:00', '09-05-1959 03:00:00']})
        df["x"].astype("timestamp")
        df["hour_x"] = vpf.hour(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_date_hour.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_date_hour.html
    """
    expr = format_magic(expr)
    return StringSQL(f"HOUR({expr})", "int")


def interval(expr: SQLExpression) -> StringSQL:
    """
    Converts the input value to a INTERVAL data type.

    Parameters
    ----------
    expr: SQLExpression
        Expression.

    Returns
    -------
    StringSQL
        SQL string.

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": ['1 day', '2 hours']})
        # apply the interval function, creating a "interval_x" column
        df["interval_x"] = vpf.interval(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": ['1 day', '2 hours']})
        df["interval_x"] = vpf.interval(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_date_interval.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_date_interval.html
    """
    expr = format_magic(expr)
    return StringSQL(f"({expr})::interval", "interval")


def minute(expr: SQLExpression) -> StringSQL:
    """
    Returns the minute portion of the specified date
    as an integer.

    Parameters
    ----------
    expr: SQLExpression
        Expression.

    Returns
    -------
    StringSQL
        SQL string.

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": ['11-03-1993 12:05:00', '09-05-1959 03:10:00']})
        df["x"].astype("timestamp")
        # apply the minute function, creating a "minute_x" column
        df["minute_x"] = vpf.minute(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": ['11-03-1993 12:05:00', '09-05-1959 03:10:00']})
        df["x"].astype("timestamp")
        df["minute_x"] = vpf.minute(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_date_minute.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_date_minute.html
    """
    expr = format_magic(expr)
    return StringSQL(f"MINUTE({expr})", "int")


def microsecond(expr: SQLExpression) -> StringSQL:
    """
    Returns the microsecond portion of the specified
    date as an integer.

    Parameters
    ----------
    expr: SQLExpression
        Expression.

    Returns
    -------
    StringSQL
        SQL string.

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": ['11-03-1993 12:05:10.23', '09-05-1959 03:10:20.12']})
        df["x"].astype("timestamp")
        # apply the microsecond function, creating a "microsecond_x" column
        df["microsecond_x"] = vpf.microsecond(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": ['11-03-1993 12:05:10.23', '09-05-1959 03:10:20.12']})
        df["x"].astype("timestamp")
        df["microsecond_x"] = vpf.microsecond(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_date_microsecond.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_date_microsecond.html
    """
    expr = format_magic(expr)
    return StringSQL(f"MICROSECOND({expr})", "int")


def month(expr: SQLExpression) -> StringSQL:
    """
    Returns the month portion of the specified date
    as an integer.

    Parameters
    ----------
    expr: SQLExpression
        Expression.

    Returns
    -------
    StringSQL
        SQL string.

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": ['11-03-1993 12:05:10.23', '09-05-1959 03:10:20.12']})
        df["x"].astype("timestamp")
        # apply the month function, creating a "month_x" column
        df["month_x"] = vpf.month(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": ['11-03-1993 12:05:10.23', '09-05-1959 03:10:20.12']})
        df["x"].astype("timestamp")
        df["month_x"] = vpf.month(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_date_month.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_date_month.html
    """
    expr = format_magic(expr)
    return StringSQL(f"MONTH({expr})", "int")


def overlaps(
    start0: SQLExpression,
    end0: SQLExpression,
    start1: SQLExpression,
    end1: SQLExpression,
) -> StringSQL:
    """
    Evaluates  two time  periods and returns true  when
    they overlap, false otherwise.

    Parameters
    ----------
    start0: SQLExpression
        DATE, TIME, or TIMESTAMP/TIMESTAMPTZ value that
        specifies the beginning of a time period.
    end0: SQLExpression
        DATE, TIME, or TIMESTAMP/TIMESTAMPTZ value that
        specifies the end of a time period.
    start1: SQLExpression
        DATE, TIME, or TIMESTAMP/TIMESTAMPTZ value that
        specifies the beginning of a time period.
    end1: SQLExpression
        DATE, TIME, or TIMESTAMP/TIMESTAMPTZ value that
        specifies the end of a time period.

    Returns
    -------
    StringSQL
        SQL string.

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"start0": ['11-03-1993'],
                          "end0": ['12-03-1993'],
                          "start1": ['11-30-1993'],
                          "end1": ['11-30-1994']})
        df["start0"].astype("timestamp")
        df["start1"].astype("timestamp")
        df["end0"].astype("timestamp")
        df["end1"].astype("timestamp")
        # apply the overlaps function, creating a "overlaps" column
        df["overlaps"] = vpf.overlaps(df["start0"], df["end0"], df["start1"], df["end1"])

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
                df = vDataFrame({"start0": ['11-03-1993'],
                          "end0": ['12-03-1993'],
                          "start1": ['11-30-1993'],
                          "end1": ['11-30-1994']})
        df["start0"].astype("timestamp")
        df["start1"].astype("timestamp")
        df["end0"].astype("timestamp")
        df["end1"].astype("timestamp")
        df["overlaps"] = vpf.overlaps(df["start0"], df["end0"], df["start1"], df["end1"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_date_overlaps.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_date_overlaps.html
    """
    expr = f"""
        ({format_magic(start0)},
         {format_magic(end0)})
         OVERLAPS
        ({format_magic(start1)},
         {format_magic(end1)})"""
    return StringSQL(clean_query(expr), "int")


def quarter(expr: SQLExpression) -> StringSQL:
    """
    Returns calendar quarter of the specified date
    as an integer, where the January-March quarter
    is 1.

    Parameters
    ----------
    expr: SQLExpression
        Expression.

    Returns
    -------
    StringSQL
        SQL string.

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": ['11-03-1993 12:05:10.23', '09-05-1959 03:10:20.12']})
        df["x"].astype("timestamp")
        # apply the quarter function, creating a "quarter_x" column
        df["quarter_x"] = vpf.quarter(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": ['11-03-1993 12:05:10.23', '09-05-1959 03:10:20.12']})
        df["x"].astype("timestamp")
        df["quarter_x"] = vpf.quarter(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_date_quarter.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_date_quarter.html
    """
    expr = format_magic(expr)
    return StringSQL(f"QUARTER({expr})", "int")


def round_date(expr: SQLExpression, precision: str = "DD") -> StringSQL:
    """
    Rounds the specified date or time.

    Parameters
    ----------
    expr: SQLExpression
        Expression.
    precision: str, optional
        A string  constant that  specifies precision
        for the rounded value, one of the following:

        **Century**:
                    CC | SCC

        **Year**:
                    SYYY | YYYY | YEAR | YYY | YY | Y

        **ISO Year**:
                    IYYY | IYY | IY | I

        **Quarter**:
                    Q

        **Month**:
                    MONTH | MON | MM | RM

        **Same weekday as first day of year**:
                    WW

        **Same weekday as first day of ISO year**:
                    IW

        **Same weekday as first day of month**:
                    W

        **Day (default)**:
                    DDD | DD | J

        **First weekday**:
                    DAY | DY | D

        **Hour**:
                    HH | HH12 | HH24

        **Minute**:
                    MI

        **Second**:
                    SS

    Returns
    -------
    StringSQL
        SQL string.

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": ['11/03/1993', '09/05/1959']})
        df["x"].astype("date")
        # apply the round_date function, creating a "round_x" column
        df["round_x"] = vpf.round_date(df["x"], 'MM')
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": ['11/03/1993', '09/05/1959']})
        df["x"].astype("date")
        df["round_x"] = vpf.round_date(df["x"], 'MM')
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_date_round_date.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_date_round_date.html
    """
    expr = format_magic(expr)
    return StringSQL(f"ROUND({expr}, '{precision}')", "date")


def second(expr: SQLExpression) -> StringSQL:
    """
    Returns the seconds portion of the specified
    date as an integer.

    Parameters
    ----------
    expr: SQLExpression
        Expression.

    Returns
    -------
    StringSQL
        SQL string.

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": ['11-03-1993 12:05:10.23', '09-05-1959 03:10:20.12']})
        df["x"].astype("timestamp")
        # apply the second function, creating a "second_x" column
        df["second_x"] = vpf.second(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": ['11-03-1993 12:05:10.23', '09-05-1959 03:10:20.12']})
        df["x"].astype("timestamp")
        df["second_x"] = vpf.second(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_date_second.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_date_second.html
    """
    expr = format_magic(expr)
    return StringSQL(f"SECOND({expr})", "int")


def timestamp(expr: SQLExpression) -> StringSQL:
    """
    Converts the input value to a TIMESTAMP
    data type.

    Parameters
    ----------
    expr: SQLExpression
        Expression.

    Returns
    -------
    StringSQL
        SQL string.

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": ['11-03-1993 12:05:10.23', '09-05-1959 03:10:20.12']})
        # apply the timestamp function, creating a "timestamp_x" column
        df["timestamp_x"] = vpf.timestamp(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": ['11-03-1993 12:05:10.23', '09-05-1959 03:10:20.12']})
        df["timestamp_x"] = vpf.timestamp(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_date_timestamp.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_date_timestamp.html
    """
    expr = format_magic(expr)
    return StringSQL(f"({expr})::timestamp", "date")


def week(expr: SQLExpression) -> StringSQL:
    """
    Returns the week of the year for the
    specified date  as an integer, where
    the  first week begins on the  first
    Sunday on or preceding January 1.

    Parameters
    ----------
    expr: SQLExpression
        Expression.

    Returns
    -------
    StringSQL
        SQL string.

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": ['11-03-1993 12:05:10.23', '09-05-1959 03:10:20.12']})
        df["x"].astype("timestamp")
        # apply the week function, creating a "week_x" column
        df["week_x"] = vpf.week(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": ['11-03-1993 12:05:10.23', '09-05-1959 03:10:20.12']})
        df["x"].astype("timestamp")
        df["week_x"] = vpf.week(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_date_week.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_date_week.html
    """
    expr = format_magic(expr)
    return StringSQL(f"WEEK({expr})", "int")


def year(expr: SQLExpression) -> StringSQL:
    """
    Returns an integer that represents the
    year  portion  of the  specified date.

    Parameters
    ----------
    expr: SQLExpression
        Expression.

    Returns
    -------
    StringSQL
        SQL string.

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": ['11-03-1993 12:05:10.23', '09-05-1959 03:10:20.12']})
        df["x"].astype("timestamp")
        # apply the year function, creating a "year_x" column
        df["year_x"] = vpf.year(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": ['11-03-1993 12:05:10.23', '09-05-1959 03:10:20.12']})
        df["x"].astype("timestamp")
        df["year_x"] = vpf.year(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_date_year.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_date_year.html
    """
    expr = format_magic(expr)
    return StringSQL(f"YEAR({expr})", "int")
