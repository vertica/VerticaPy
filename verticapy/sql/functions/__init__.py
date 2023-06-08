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
from verticapy.sql.functions.analytic import (
    avg,
    bool_and,
    bool_or,
    bool_xor,
    conditional_change_event,
    conditional_true_event,
    count,
    lag,
    lead,
    max,
    median,
    min,
    nth_value,
    quantile,
    rank,
    row_number,
    std,
    sum,
    var,
)
from verticapy.sql.functions.conditional import case_when, decode
from verticapy.sql.functions.date import (
    date,
    day,
    dayofweek,
    dayofyear,
    extract,
    getdate,
    getutcdate,
    hour,
    interval,
    minute,
    microsecond,
    month,
    overlaps,
    quarter,
    round_date,
    second,
    timestamp,
    week,
    year,
)
from verticapy.sql.functions.math import E, INF, NAN, PI, TAU
from verticapy.sql.functions.math import (
    apply,
    abs,
    acos,
    asin,
    atan,
    atan2,
    cbrt,
    ceil,
    comb,
    cos,
    cosh,
    cot,
    degrees,
    distance,
    exp,
    factorial,
    floor,
    gamma,
    hash,
    isfinite,
    isinf,
    isnan,
    lgamma,
    ln,
    log,
    radians,
    round,
    sign,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
    trunc,
)
from verticapy.sql.functions.null_handling import coalesce, nullifzero, zeroifnull
from verticapy.sql.functions.random import random, randomint, seeded_random
from verticapy.sql.functions.regexp import (
    regexp_count,
    regexp_ilike,
    regexp_instr,
    regexp_like,
    regexp_replace,
    regexp_substr,
)
from verticapy.sql.functions.string import (
    length,
    lower,
    substr,
    upper,
    edit_distance,
    soundex,
    soundex_matches,
    jaro_distance,
    jaro_winkler_distance,
)
