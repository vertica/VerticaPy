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
import re
import warnings
from typing import Any, Iterable, Literal, Optional

import numpy as np

import pandas as pd

import verticapy._config.config as conf
from verticapy._utils._object import read_pd
from verticapy._utils._sql._cast import to_dtype_category
from verticapy._typing import NoneType, SQLColumns, SQLExpression
from verticapy.errors import ParsingError

if conf.get_import_success("IPython"):
    from IPython.display import display, Markdown

"""
SQL KEYWORDS
"""

SQL_KEYWORDS = {
    "ADD CONSTRAINT": {"l": (" ",), "r": (" ",)},
    "ADD": {"l": (" ",), "r": (" ",)},
    "ALL": {"l": (" ",), "r": (" ",)},
    "ALTER COLUMN": {"l": (" ",), "r": (" ",)},
    "ALTER TABLE": {"l": (" ",), "r": (" ",)},
    "AND": {"l": (" ",), "r": (" ",)},
    "ANY": {"l": (" ",), "r": (" ",)},
    "AS": {"l": (" ",), "r": (" ",)},
    "ASC": {"l": (" ",), "r": (" ", ")")},
    "BACKUP DATABASE": {"l": (" ",), "r": (" ",)},
    "BETWEEN": {"l": (" ",), "r": (" ",)},
    "CASE": {"l": (" ",), "r": (" ",)},
    "CHECK": {"l": (" ",), "r": (" ",)},
    "COLUMN": {"l": (" ",), "r": (" ",)},
    "CONSTRAINT": {"l": (" ",), "r": (" ",)},
    "CREATE DATABASE": {"l": (" ",), "r": (" ",)},
    "CREATE INDEX": {"l": (" ",), "r": (" ",)},
    "CREATE OR REPLACE VIEW": {"l": (" ",), "r": (" ",)},
    "CREATE TABLE": {"l": (" ",), "r": (" ",)},
    "CREATE PROCEDURE": {"l": (" ",), "r": (" ",)},
    "CREATE UNIQUE INDEX": {"l": (" ",), "r": (" ",)},
    "CREATE VIEW": {"l": (" ",), "r": (" ",)},
    "DEFAULT": {"l": (" ",), "r": (" ",)},
    "DELETE": {"l": (" ",), "r": (" ",)},
    "DESC": {"l": (" ",), "r": (" ", ")")},
    "DISTINCT": {"l": (" ",), "r": (" ",)},
    "DROP COLUMN": {"l": (" ",), "r": (" ",)},
    "DROP CONSTRAINT": {"l": (" ",), "r": (" ",)},
    "DROP DATABASE": {"l": (" ",), "r": (" ",)},
    "DROP DEFAULT": {"l": (" ",), "r": (" ",)},
    "DROP INDEX": {"l": (" ",), "r": (" ",)},
    "DROP TABLE": {"l": (" ",), "r": (" ",)},
    "DROP VIEW": {"l": (" ",), "r": (" ",)},
    "EXEC": {"l": (" ",), "r": (" ",)},
    "EXISTS": {"l": (" ",), "r": (" ",)},
    "FOREIGN KEY": {"l": (" ",), "r": (" ",)},
    "FROM": {"l": (" ",), "r": (" ",)},
    "GROUP BY": {"l": (" ",), "r": (" ",)},
    "HAVING": {"l": (" ",), "r": (" ",)},
    "IN": {"l": (" ",), "r": (" ",)},
    "INDEX": {"l": (" ",), "r": (" ",)},
    "INSERT INTO": {"l": (" ",), "r": (" ",)},
    "IS NULL": {"l": (" ",), "r": (" ",)},
    "IS NOT NULL": {"l": (" ",), "r": (" ",)},
    "FULL OUTER JOIN": {"l": (" ",), "r": (" ",)},
    "INNER JOIN": {"l": (" ",), "r": (" ",)},
    "LEFT JOIN": {"l": (" ",), "r": (" ",)},
    "JOIN": {"l": (" ",), "r": (" ",)},
    "LIKE": {"l": (" ",), "r": (" ",)},
    "LIMIT": {"l": (" ",), "r": (" ",)},
    "NOT NULL": {"l": (" ",), "r": (" ",)},
    "NOT": {"l": (" ",), "r": (" ",)},
    "OR": {"l": (" ",), "r": (" ",)},
    "ORDER BY": {"l": (" ",), "r": (" ",)},
    "OUTER JOIN": {"l": (" ",), "r": (" ",)},
    "PRIMARY KEY": {"l": (" ",), "r": (" ",)},
    "PROCEDURE": {"l": (" ",), "r": (" ",)},
    "RIGHT JOIN": {"l": (" ",), "r": (" ",)},
    "ROWNUM": {"l": (" ",), "r": (" ",)},
    "SELECT": {"l": ("", " "), "r": (" ",)},
    "SET": {"l": (" ",), "r": (" ",)},
    "TABLE": {"l": (" ",), "r": (" ",)},
    "TOP": {"l": (" ",), "r": (" ",)},
    "TRUNCATE TABLE": {"l": (" ",), "r": (" ",)},
    "UNION ALL": {"l": (" ",), "r": (" ",)},
    "UNION": {"l": (" ",), "r": (" ",)},
    "UNIQUE": {"l": (" ",), "r": (" ",)},
    "UPDATE": {"l": (" ",), "r": (" ",)},
    "VALUES": {"l": (" ",), "r": (" ",)},
    "VIEW": {"l": (" ",), "r": (" ",)},
    "WHEN": {"l": (" ",), "r": (" ",)},
    "WHERE": {"l": (" ",), "r": (" ",)},
}

"""
SQL DATA TYPES
"""
SQL_DT = {
    "BINARY": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
    "LONG VARBINARY": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
    "VARBINARY": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
    "BYTEA": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
    "RAW": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
    "BOOLEAN": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
    "CHAR": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
    "LONG VARCHAR": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
    "VARCHAR": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
    "DATETIME": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
    "DATE": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
    "TIME WITH TIMEZONE": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
    "TIME": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
    "SMALLDATETIME": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
    "TIMESTAMP WITH TIMEZONE": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
    "TIMESTAMP": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
    "INTERVAL DAY TO SECOND": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
    "INTERVAL YEAR TO MONTH": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
    "INTERVAL": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
    "DOUBLE PRECISION": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
    "FLOAT": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
    "FLOAT8": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
    "REAL": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
    "INTEGER": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
    "INT": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
    "BIGINT": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
    "INT8": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
    "SMALLINT": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
    "TINYINT": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
    "DECIMAL": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
    "NUMERIC": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
    "NUMBER": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
    "MONEY": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
    "GEOMETRY": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
    "GEOGRAPHY": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
    "UUID": {
        "l": (" ", ":", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
        "r": (" ", "(", "'", "-", "+", "*", "/", "^", "|", "<", ">"),
    },
}

"""
SQL SPECIAL KEYWORDS
"""
SQL_SPECIAL_KEYWORDS = {
    "PROFILE": {"l": ("", " "), "r": (" ",)},
    "SHOW": {"l": ("", " "), "r": (" ",)},
}

"""
SQL AGGREGATE FUNCTIONS
"""

SQL_AGGREGATE = {
    "APPROXIMATE_COUNT_DISTINCT_OF_SYNOPSIS": {"l": (" ", ","), "r": (" ", "(")},
    "APPROXIMATE_COUNT_DISTINCT_SYNOPSIS": {"l": (" ", ","), "r": (" ", "(")},
    "APPROXIMATE_COUNT_DISTINCT_SYNOPSIS_MERGE": {"l": (" ", ","), "r": (" ", "(")},
    "APPROXIMATE_COUNT_DISTINCT": {"l": (" ", ","), "r": (" ", "(")},
    "APPROXIMATE_MEDIAN": {"l": (" ", ","), "r": (" ", "(")},
    "APPROXIMATE_PERCENTILE": {"l": (" ", ","), "r": (" ", "(")},
    "APPROXIMATE_QUANTILES": {"l": (" ", ","), "r": (" ", "(")},
    "ARGMAX_AGG": {"l": (" ", ","), "r": (" ", "(")},
    "ARGMIN_AGG": {"l": (" ", ","), "r": (" ", "(")},
    "AVG": {"l": (" ", ","), "r": (" ", "(")},
    "BIT_AND": {"l": (" ", ","), "r": (" ", "(")},
    "BIT_OR": {"l": (" ", ","), "r": (" ", "(")},
    "BIT_XOR": {"l": (" ", ","), "r": (" ", "(")},
    "BOOL_AND": {"l": (" ", ","), "r": (" ", "(")},
    "BOOL_OR": {"l": (" ", ","), "r": (" ", "(")},
    "BOOL_XOR": {"l": (" ", ","), "r": (" ", "(")},
    "CORR": {"l": (" ", ","), "r": (" ", "(")},
    "COUNT": {"l": (" ", ","), "r": (" ", "(")},
    "COVAR_POP": {"l": (" ", ","), "r": (" ", "(")},
    "COVAR_SAMP": {"l": (" ", ","), "r": (" ", "(")},
    "GROUP_ID": {"l": (" ", ","), "r": (" ", "(")},
    "GROUPING_ID": {"l": (" ", ","), "r": (" ", "(")},
    "GROUPING": {"l": (" ", ","), "r": (" ", "(")},
    "LISTAGG": {"l": (" ", ","), "r": (" ", "(")},
    "MAX": {"l": (" ", ","), "r": (" ", "(")},
    "MIN": {"l": (" ", ","), "r": (" ", "(")},
    "OVER": {"l": (" ", ")"), "r": (" ", "(")},
    "PARTITION BY": {"l": (" ", "("), "r": (" ",)},
    "REGR_AVGX": {"l": (" ", ","), "r": (" ", "(")},
    "REGR_AVGY": {"l": (" ", ","), "r": (" ", "(")},
    "REGR_COUNT": {"l": (" ", ","), "r": (" ", "(")},
    "REGR_INTERCEPT": {"l": (" ", ","), "r": (" ", "(")},
    "REGR_R2": {"l": (" ", ","), "r": (" ", "(")},
    "REGR_SLOPE": {"l": (" ", ","), "r": (" ", "(")},
    "REGR_SXX": {"l": (" ", ","), "r": (" ", "(")},
    "REGR_SXY": {"l": (" ", ","), "r": (" ", "(")},
    "REGR_SYY": {"l": (" ", ","), "r": (" ", "(")},
    "STDDEV_POP": {"l": (" ", ","), "r": (" ", "(")},
    "STDDEV_SAMP": {"l": (" ", ","), "r": (" ", "(")},
    "STDDEV": {"l": (" ", ","), "r": (" ", "(")},
    "SUM_FLOAT": {"l": (" ", ","), "r": (" ", "(")},
    "SUM": {"l": (" ", ","), "r": (" ", "(")},
    "TS_FIRST_VALUE": {"l": (" ", ","), "r": (" ", "(")},
    "TS_LAST_VALUE": {"l": (" ", ","), "r": (" ", "(")},
    "VAR_POP": {"l": (" ", ","), "r": (" ", "(")},
    "VAR_SAMP": {"l": (" ", ","), "r": (" ", "(")},
    "VARIANCE": {"l": (" ", ","), "r": (" ", "(")},
}

"""
OTHER SQL FUNCTIONS
"""

SQL_FUN = {
    # MATH
    "ABS": {
        "l": (
            " ",
            ",",
        ),
        "r": (" ", "("),
    },
    "ACOS": {"l": (" ", ","), "r": (" ", "(")},
    "ACOSH": {"l": (" ", ","), "r": (" ", "(")},
    "ASIN": {"l": (" ", ","), "r": (" ", "(")},
    "ASINH": {"l": (" ", ","), "r": (" ", "(")},
    "ATAN": {"l": (" ", ","), "r": (" ", "(")},
    "ATAN2": {"l": (" ", ","), "r": (" ", "(")},
    "ATANH": {"l": (" ", ","), "r": (" ", "(")},
    "CBRT": {"l": (" ", ","), "r": (" ", "(")},
    "CEILING": {"l": (" ", ","), "r": (" ", "(")},
    "COS": {"l": (" ", ","), "r": (" ", "(")},
    "COSH": {"l": (" ", ","), "r": (" ", "(")},
    "COT": {"l": (" ", ","), "r": (" ", "(")},
    "DEGREES": {"l": (" ", ","), "r": (" ", "(")},
    "DISTANCE": {"l": (" ", ","), "r": (" ", "(")},
    "DISTANCEV": {"l": (" ", ","), "r": (" ", "(")},
    "EXP": {"l": (" ", ","), "r": (" ", "(")},
    "FLOOR": {"l": (" ", ","), "r": (" ", "(")},
    "HASH": {"l": (" ", ","), "r": (" ", "(")},
    "LN": {"l": (" ", ","), "r": (" ", "(")},
    "LOG": {"l": (" ", ","), "r": (" ", "(")},
    "LOG10": {"l": (" ", ","), "r": (" ", "(")},
    "MOD": {"l": (" ", ","), "r": (" ", "(")},
    "PI": {"l": (" ", ","), "r": (" ", "(")},
    "POWER": {"l": (" ", ","), "r": (" ", "(")},
    "RADIANS": {"l": (" ", ","), "r": (" ", "(")},
    "RANDOM": {"l": (" ", ","), "r": (" ", "(")},
    "RANDOMINT": {"l": (" ", ","), "r": (" ", "(")},
    "RANDOMINT_CRYPTO": {"l": (" ", ","), "r": (" ", "(")},
    "ROUND": {"l": (" ", ","), "r": (" ", "(")},
    "SIGN": {"l": (" ", ","), "r": (" ", "(")},
    "SIN": {"l": (" ", ","), "r": (" ", "(")},
    "SINH": {"l": (" ", ","), "r": (" ", "(")},
    "SQRT": {"l": (" ", ","), "r": (" ", "(")},
    "TAN": {"l": (" ", ","), "r": (" ", "(")},
    "TANH": {"l": (" ", ","), "r": (" ", "(")},
    "TRUNC": {"l": (" ", ","), "r": (" ", "(")},
    "WIDTH_BUCKET": {"l": (" ", ","), "r": (" ", "(")},
    # DATE/DATETIME
    "ADD_MONTHS": {"l": (" ", ","), "r": (" ", "(")},
    "AGE_IN_MONTHS": {"l": (" ", ","), "r": (" ", "(")},
    "AGE_IN_YEARS": {"l": (" ", ","), "r": (" ", "(")},
    "CLOCK_TIMESTAMP": {"l": (" ", ","), "r": (" ", "(")},
    "CURRENT_DATE": {"l": (" ", ","), "r": (" ", "(")},
    "CURRENT_TIME": {"l": (" ", ","), "r": (" ", "(")},
    "CURRENT_TIMESTAMP": {"l": (" ", ","), "r": (" ", "(")},
    "DATE": {"l": (" ", ","), "r": (" ", "(")},
    "DATE_PART": {"l": (" ", ","), "r": (" ", "(")},
    "DATE_TRUNC": {"l": (" ", ","), "r": (" ", "(")},
    "DATEDIFF": {"l": (" ", ","), "r": (" ", "(")},
    "DAY": {"l": (" ", ","), "r": (" ", "(")},
    "DAYOFMONTH": {"l": (" ", ","), "r": (" ", "(")},
    "DAYOFWEEK": {"l": (" ", ","), "r": (" ", "(")},
    "DAYOFWEEK_ISO": {"l": (" ", ","), "r": (" ", "(")},
    "DAYOFYEAR": {"l": (" ", ","), "r": (" ", "(")},
    "DAYS": {"l": (" ", ","), "r": (" ", "(")},
    "EXTRACT": {"l": (" ", ","), "r": (" ", "(")},
    "GETDATE": {"l": (" ", ","), "r": (" ", "(")},
    "GETUTCDATE": {"l": (" ", ","), "r": (" ", "(")},
    "HOUR": {"l": (" ", ","), "r": (" ", "(")},
    "ISFINITE": {"l": (" ", ","), "r": (" ", "(")},
    "JULIAN_DAY": {"l": (" ", ","), "r": (" ", "(")},
    "LAST_DAY": {"l": (" ", ","), "r": (" ", "(")},
    "LOCALTIME": {"l": (" ", ","), "r": (" ", "(")},
    "LOCALTIMESTAMP": {"l": (" ", ","), "r": (" ", "(")},
    "MICROSECOND": {"l": (" ", ","), "r": (" ", "(")},
    "MIDNIGHT_SECONDS": {"l": (" ", ","), "r": (" ", "(")},
    "MINUTE": {"l": (" ", ","), "r": (" ", "(")},
    "MONTH": {"l": (" ", ","), "r": (" ", "(")},
    "MONTHS_BETWEEN": {"l": (" ", ","), "r": (" ", "(")},
    "NEW_TIME": {"l": (" ", ","), "r": (" ", "(")},
    "NEXT_DAY": {"l": (" ", ","), "r": (" ", "(")},
    "NOW": {"l": (" ", ","), "r": (" ", "(")},
    "OVERLAPS": {"l": (" ", ","), "r": (" ", "(")},
    "QUARTER": {"l": (" ", ","), "r": (" ", "(")},
    "ROUND": {"l": (" ", ","), "r": (" ", "(")},
    "SECOND": {"l": (" ", ","), "r": (" ", "(")},
    "STATEMENT_TIMESTAMP": {"l": (" ", ","), "r": (" ", "(")},
    "SYSDATE": {"l": (" ", ","), "r": (" ", "(")},
    "TIME_SLICE": {"l": (" ", ","), "r": (" ", "(")},
    "TIMEOFDAY": {"l": (" ", ","), "r": (" ", "(")},
    "TIMESTAMP_ROUND": {"l": (" ", ","), "r": (" ", "(")},
    "TIMESTAMP_TRUNC": {"l": (" ", ","), "r": (" ", "(")},
    "TIMESTAMPADD": {"l": (" ", ","), "r": (" ", "(")},
    "TIMESTAMPDIFF": {"l": (" ", ","), "r": (" ", "(")},
    "TRANSACTION_TIMESTAMP": {"l": (" ", ","), "r": (" ", "(")},
    "TRUNC": {"l": (" ", ","), "r": (" ", "(")},
    "WEEK": {"l": (" ", ","), "r": (" ", "(")},
    "WEEK_ISO": {"l": (" ", ","), "r": (" ", "(")},
    "YEAR": {"l": (" ", ","), "r": (" ", "(")},
    "YEAR_ISO": {"l": (" ", ","), "r": (" ", "(")},
}


"""
OPERATORS
"""
OPERATORS = {
    "*": {
        "l": (" ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "(", ")"),
        "r": (" ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "(", ")"),
    },
    "-": {
        "l": (" ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "(", ")"),
        "r": (" ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "(", ")"),
    },
    "+": {
        "l": (" ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "(", ")"),
        "r": (" ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "(", ")"),
    },
    "^": {
        "l": (" ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "(", ")"),
        "r": (" ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "(", ")"),
    },
    "/": {
        "l": (" ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "(", ")"),
        "r": (" ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "(", ")"),
    },
    "||": {
        "l": (" ", "'"),
        "r": (" ", "'"),
    },
}

"""
TAGS
"""
KEYWORDS_TAG_L = '<b style="color: #C695C6;">'
KEYWORDS_TAG_R = "</b>"
SKEYWORDS_TAG_L = '<b style="color: #FF0000;">'
SKEYWORDS_TAG_R = "</b>"
DT_TAG_L = '<b style="color: #FFC0CB; font-weight: normal;">'
DT_TAG_R = "</b>"
AGGREGATE_TAG_L = '<i style="color: #6699CB;">'
AGGREGATE_TAG_R = "</i>"
FUN_TAG_L = '<i style="color: #00008B;">'
FUN_TAG_R = "</i>"
OPERATOR_TAG_L = '<b style="color: #EA5E65; font-weight: normal;">'
OPERATOR_TAG_R = "</b>"
DIGIT_TAG_L = '<strong style="color: #F7AC57;">'
DIGIT_TAG_R = "</strong>"
STRING_TAG_L = '<b style="color: #89B285; font-weight: normal;">'
STRING_TAG_R = "</b>"
COL_TAG_L = '<b style="color: #5CADAD; font-weight: normal;">'
COL_TAG_R = "</b>"
COMMENT_TAG_L = '<b style="color: #9DA3AF; font-weight: normal;">'
COMMENT_TAG_R = "</b>"

"""
Main function
"""


def _format_keys(
    d: dict, sql: str, mkd: str, tag_l: str, tag_r: str
) -> tuple[str, str]:
    """
    Function to simplify the code.
    """
    for key in d:
        for l in d[key]["l"]:
            for r in d[key]["r"]:
                w = l + key + r
                sql = re.sub(re.escape(w), w.upper(), sql, flags=re.IGNORECASE)
                if not (isinstance(mkd, NoneType)):
                    mkd = re.sub(
                        re.escape(w),
                        l + tag_l + key.upper() + tag_r + r,
                        mkd,
                        flags=re.IGNORECASE,
                    )
    return sql, mkd


def format_query(
    query: SQLExpression, indent_sql: bool = True, print_sql: bool = True
) -> SQLExpression:
    """
    Query Formatter.
    """
    display_success = print_sql and conf.get_import_success("IPython")
    res = clean_query(query)
    if display_success:
        html_res = res
    else:
        html_res = None
    # STRINGS
    if display_success:
        html_res = re.sub(
            r"(\"(.)+\")",
            COL_TAG_L + r" \1 " + COL_TAG_R,
            html_res,
        )
        html_res = re.sub(
            r"(\'(.)+\')",
            STRING_TAG_L + r" \1 " + STRING_TAG_R,
            html_res,
        )
        html_res = re.sub(
            r"(--.+(\n|\Z))", COMMENT_TAG_L + r" \1 " + COMMENT_TAG_R, html_res
        )
        html_res = re.sub(
            r"(/\*(.+?)\*/)", COMMENT_TAG_L + r" \1 " + COMMENT_TAG_R, html_res
        )
    # SQL KEY WORDS
    res, html_res = _format_keys(
        SQL_KEYWORDS, res, html_res, KEYWORDS_TAG_L, KEYWORDS_TAG_R
    )
    # SQL SPECIAL KEY WORDS
    res, html_res = _format_keys(
        SQL_SPECIAL_KEYWORDS, res, html_res, SKEYWORDS_TAG_L, SKEYWORDS_TAG_R
    )
    # SQL DATA TYPES
    res, html_res = _format_keys(SQL_DT, res, html_res, DT_TAG_L, DT_TAG_R)
    # SQL AGGREGATE FUNCTIONS
    res, html_res = _format_keys(
        SQL_AGGREGATE, res, html_res, AGGREGATE_TAG_L, AGGREGATE_TAG_R
    )
    # OTHER SQL FUNCTIONS
    res, html_res = _format_keys(SQL_FUN, res, html_res, FUN_TAG_L, FUN_TAG_R)
    # OPERATORS
    res, html_res = _format_keys(
        OPERATORS, res, html_res, OPERATOR_TAG_L, OPERATOR_TAG_R
    )
    # DIGITS
    if display_success:
        html_res = re.sub(
            r"(\s|\+|\-|\\|\*|\/)(\d+)(\s|\+|\-|\\|\*|\/|$)",
            r"\1" + DIGIT_TAG_L + r" \2 " + DIGIT_TAG_R + r"\3",
            html_res,
        )

    if indent_sql:
        res = indent_vpy_sql(res)
    if display_success:
        html_res = html_res.replace("*", "&ast;")
        if indent_sql:
            html_res = (
                indent_vpy_sql(html_res.strip())
                .replace("\n", "<br>")
                .replace("    ", "&nbsp;&nbsp;&nbsp;&nbsp;")
            )
        display(Markdown(html_res))
    elif print_sql:
        print(res)
    return res, html_res


"""
Utils
"""


def clean_query(query: SQLExpression) -> SQLExpression:
    """
    Cleans the input query by erasing comments, spaces,
    and other unnecessary characters.
    Comments using '/*' and '*/' are left in the query.
    """
    if isinstance(query, list):
        return [clean_query(q) for q in query]
    else:
        query = re.sub(r"--.+(\n|\Z)", "", query)
        query = query.replace("\t", " ").replace("\n", " ")
        query = re.sub(" +", " ", query)

        while len(query) > 0 and query.endswith((";", " ")):
            query = query[0:-1]

        while len(query) > 0 and query.startswith((";", " ")):
            query = query[1:]

        return query.strip().replace("\xa0", " ")


def erase_comment(query: str) -> str:
    """
    Removes comments from the input query.
    """
    query = re.sub(r"--.+(\n|\Z)", "", query)
    query = re.sub(r"/\*(.+?)\*/", "", query)
    return query.strip()


def erase_label(query: str) -> str:
    """
    Removes labels from the input query.
    """
    labels = re.findall(r"\/\*\+LABEL(.*?)\*\/", query)
    for label in labels:
        query = query.replace(f"/*+LABEL{label}*/", "")
    return query.strip()


def extract_subquery(query: str) -> str:
    """
    Extracts the SQL subquery from the input query.
    """
    query_tmp = clean_query(query)
    query_tmp = erase_comment(query_tmp)
    if query_tmp[0] == "(" and query_tmp[-1] != ")":
        query = ")".join("(".join(query_tmp.split("(")[1:]).split(")")[:-1])
    return query.strip()


def extract_and_rename_subquery(query: str, alias: str) -> str:
    """
    Extracts the SQL subquery from the input query
    and renames it.
    """
    query_tmp = extract_subquery(query)
    query_clean = clean_query(query)
    query_clean = erase_comment(query)
    if query != query_tmp or query_clean[0:6].lower() == "select":
        query = f"({query_tmp})"
    return f"{query} AS {quote_ident(alias)}"


def extract_precision_scale(ctype: str) -> tuple:
    """
    Extracts the precision and scale from the
    input SQL type.
    """
    if "(" not in ctype:
        return (0, 0)
    else:
        precision_scale = ctype.split("(")[1].split(")")[0].split(",")
        if len(precision_scale) == 1:
            precision, scale = int(precision_scale[0]), 0
        else:
            precision, scale = precision_scale
        return int(precision), int(scale)


def format_magic(
    x: Any, return_cat: bool = False, cast_float_int_to_str: bool = False
) -> Any:
    """
    Formats  the input element using SQL rules.
    Ex: None values are represented by NULL and
        string are enclosed by single quotes "'"
    """
    object_type = None
    if hasattr(x, "object_type"):
        object_type = x.object_type
    if object_type == "vDataColumn":
        val = x._alias
    elif (isinstance(x, (int, float, np.int_)) and not cast_float_int_to_str) or (
        object_type == "StringSQL"
    ):
        val = x
    elif isinstance(x, NoneType):
        val = "NULL"
    elif isinstance(x, (int, float, np.int_)) or not cast_float_int_to_str:
        x_str = str(x).replace("'", "''")
        val = f"'{x_str}'"
    else:
        val = x
    if return_cat:
        return (val, to_dtype_category(x))
    else:
        return val


def format_schema_table(schema: str, table_name: str) -> str:
    """
    Returns the formatted relation. If the schema is not
    defined, the 'public' schema is used.
    """
    if not schema:
        schema = conf.get_option("temp_schema")
    return f"{quote_ident(schema)}.{quote_ident(table_name)}"


def format_type(*args, dtype: Literal[NoneType, dict, list], na_out: Any = None) -> Any:
    """
    Format the input objects by using the input type. This
    simplifies the code because many functions check
    types and instantiate the corresponding object.
    """
    res = ()
    for arg in args:
        if isinstance(arg, NoneType):
            if not isinstance(na_out, NoneType):
                r = na_out
            elif dtype == list:
                r = []
            elif dtype == dict:
                r = {}
            else:
                r = None
        elif isinstance(arg, (float, int, str)):
            if dtype == list:
                r = [arg]
            else:
                r = arg
        else:
            r = arg
        res += (r,)
    if len(res) == 1:
        return res[0]
    else:
        return res


def indent_vpy_sql(query: str) -> str:
    """
    Indents the input SQL query.
    """
    query = (
        query.replace("SELECT", "\n   SELECT\n    ")
        .replace("FROM", "\n   FROM\n")
        .replace("ORDER BY", "\n   ORDER BY")
        .replace("GROUP BY", "\n   GROUP BY")
        .replace("LIMIT", "\n   LIMIT")
        .replace("OFFSET", "\n   OFFSET")
        .replace("WHERE", "\n   WHERE")
        .replace(",", ",\n    ")
    )
    query = query.replace("VERTICAPY_SUBTABLE", "\nVERTICAPY_SUBTABLE")
    n = len(query)
    return_l = []
    j = 1
    while j < n - 9:
        if (
            query[j] == "("
            and (query[j - 1].isalnum() or query[j - 5 : j] == "OVER ")
            and query[j + 1 : j + 7] != "SELECT"
        ):
            k = 1
            while k > 0 and j < n - 9:
                j += 1
                if query[j] == "\n":
                    return_l += [j]
                elif query[j] == ")":
                    k -= 1
                elif query[j] == "(":
                    k += 1
        else:
            j += 1
    query_print = ""
    i = 0 if query[0] != "\n" else 1
    while return_l:
        j = return_l[0]
        query_print += query[i:j]
        if query[j] != "\n":
            query_print += query[j]
        else:
            i = j + 1
            while query[i] == " " and i < n - 9:
                i += 1
            query_print += " "
        del return_l[0]
    query_print += query[i:n]
    return query_print


def list_strip(L: list) -> list:
    """
    Erases all the start / end spaces from the
    input list.
    """
    return [val.strip() for val in L]


def quote_ident(column: Optional[SQLColumns], lower: bool = False) -> SQLColumns:
    """
    Returns the specified string argument in the format
    that is required in order to use that string as an
    identifier in an SQL statement.

    Parameters
    ----------
    column: str
        Column's name.

    Returns
    -------
    str
        Formatted column name.
    """
    if isinstance(column, str):
        tmp_column = str(column)
        if len(tmp_column) >= 2 and (tmp_column[0] == tmp_column[-1] == '"'):
            tmp_column = tmp_column[1:-1]
        temp_column_str = str(tmp_column).replace('"', '""')
        temp_column_str = f'"{temp_column_str}"'
        if temp_column_str == '""':
            return ""
        elif lower:
            temp_column_str = temp_column_str.lower()
        return temp_column_str
    elif isinstance(column, NoneType):
        return ""
    elif isinstance(column, Iterable):
        return [quote_ident(x) for x in column]
    else:
        return str(column)


def replace_label(
    query: str,
    new_label: Optional[str] = None,
    separator: Optional[str] = None,
    suffix: Optional[str] = None,
) -> str:
    """
    Replace the current query's label by a new one.
    """
    if isinstance(separator, NoneType):
        separator = ""
    if isinstance(suffix, NoneType):
        suffix = ""
    labels = re.findall(r"\/\*\+LABEL\(\'(.*?)\'\)\*\/", query)
    for label in labels:
        if isinstance(new_label, NoneType):
            nlabel = label
        else:
            nlabel = new_label
        query = query.replace(
            f"/*+LABEL('{label}')*/", f"/*+LABEL('{nlabel}{separator}{suffix}')*/"
        )
    return query.strip()


def replace_vars_in_query(query: str, locals_dict: dict) -> str:
    """
    Replaces the input variables with their respective SQL
    representations. If a input variable does not have a
    SQL representation, it is materialised by a temporary
    local table.
    """
    variables, query_tmp = re.findall(r"(?<!:):[A-Za-z0-9_\[\]]+", query), query
    for v in variables:
        fail = True
        if len(v) > 1 and not v[1].isdigit():
            try:
                var = v[1:]
                n, splits = var.count("["), []
                if var.count("]") == n and n > 0:
                    i, size = 0, len(var)
                    while i < size:
                        if var[i] == "[":
                            k = i + 1
                            while i < size and var[i] != "]":
                                i += 1
                            splits += [(k, i)]
                        i += 1
                    var = var[: splits[0][0] - 1]
                val = locals_dict[var]
                if splits:
                    for s in splits:
                        val = val[int(v[s[0] + 1 : s[1] + 1])]
                fail = False
            except Exception as e:
                warning_message = (
                    f"Failed to replace variables in the query.\nError: {e}"
                )
                warnings.warn(warning_message, Warning)
                fail = True
        if not fail:
            object_type = None
            if hasattr(val, "object_type"):
                object_type = val.object_type
            if object_type == "vDataFrame":
                val = val.current_relation()
            elif object_type == "TableSample":
                val = f"({val.to_sql()}) VERTICAPY_SUBTABLE"
            elif isinstance(val, pd.DataFrame):
                val = read_pd(val).current_relation()
            elif isinstance(val, list):
                val = ", ".join(["NULL" if x is None else str(x) for x in val])
            query_tmp = query_tmp.replace(v, str(val))
    return query_tmp


def schema_relation(relation: Any, do_quote: bool = True) -> tuple[str, str]:
    """
    Extracts the schema and the table from the input
    relation. If the input relation does not have a schema,
    the temporary schema is used.
    """
    if isinstance(relation, str):
        rel_transf = relation.replace('""', "__verticapy_doublequotes_")
        quote_nb = rel_transf.count('"')
        dot_nb = rel_transf.count(".")
        if (quote_nb == 0 and dot_nb == 0) or (
            quote_nb == 2 and rel_transf.startswith('"') and rel_transf.endswith('"')
        ):
            schema, relation = conf.get_option("temp_schema"), relation
        elif quote_nb == 0 and dot_nb == 1:
            schema, relation = relation.split(".")
        elif quote_nb == 2 and rel_transf.startswith('"'):
            schema, relation = rel_transf.split('"')[1:]
            schema = schema.replace("__verticapy_doublequotes_", '""')
            relation = relation[1:].replace("__verticapy_doublequotes_", '""')
        elif quote_nb == 2 and rel_transf.endswith('"'):
            schema, relation = rel_transf.split('"')[:-1]
            schema = schema[:-1].replace("__verticapy_doublequotes_", '""')
            relation = relation.replace("__verticapy_doublequotes_", '""')
        elif quote_nb == 4 and rel_transf.endswith('"') and rel_transf.startswith('"'):
            schema = rel_transf.split('"')[1].replace("__verticapy_doublequotes_", '""')
            relation = rel_transf.split('"')[3].replace(
                "__verticapy_doublequotes_", '""'
            )
            if rel_transf.split('"')[2] != ".":
                raise ParsingError("The format of the input relation is incorrect.")
        else:
            raise ParsingError("The format of the input relation is incorrect.")
    else:
        schema, relation = conf.get_option("temp_schema"), ""

    if do_quote:
        return (quote_ident(schema), quote_ident(relation))
    else:
        return (schema, relation)
