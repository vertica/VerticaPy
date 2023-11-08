.. _api.sql.functions:

==============
Functions
==============

______

Constants
------------

.. currentmodule:: verticapy.sql.functions

.. autosummary:: 
   :toctree: api/

   E
   INF
   NAN
   PI
   TAU

_____

Analytics
------------

.. currentmodule:: verticapy.sql.functions

.. autosummary:: 
   :toctree: api/

   avg
   bool_and
   bool_or
   bool_xor
   conditional_change_event
   conditional_true_event
   count
   lag
   lead
   max
   median
   min
   nth_value
   quantile
   rank
   row_number
   std
   sum
   var
______

Conditional
--------------------------

.. currentmodule:: verticapy.sql.functions

.. autosummary:: 
   :toctree: api/

   case_when
   decode

______

Date
------

.. currentmodule:: verticapy.sql.functions

.. autosummary:: 
   :toctree: api/

   date
   day
   dayofweek
   dayofyear
   extract
   getdate
   getutcdate
   hour
   interval
   microsecond
   minute
   month
   overlaps
   quarter
   round_date
   second
   timestamp
   week
   year

_____

Math
------

.. currentmodule:: verticapy.sql.functions

.. autosummary:: 
   :toctree: api/

   apply
   abs
   acos
   asin
   atan
   atan2
   cbrt
   ceil
   comb
   cos
   cosh
   cot
   degrees
   distance
   exp
   factorial
   floor
   gamma
   hash
   isfinite
   isinf
   isnan
   lgamma
   ln
   log
   radians
   round
   sign
   sin
   sinh
   sqrt
   tan
   tanh
   trunc





_____

Null Handling
---------------

.. currentmodule:: verticapy.sql.functions

.. autosummary:: 
   :toctree: api/

   coalesce
   nullifzero
   zeroifnull


_____

Random
---------------

.. currentmodule:: verticapy.sql.functions

.. autosummary:: 
   :toctree: api/

   random
   randomint
   seeded_random


_____

Regular Expression
-------------------

.. currentmodule:: verticapy.sql.functions

.. autosummary:: 
   :toctree: api/

   regexp_count
   regexp_ilike
   regexp_instr
   regexp_like
   regexp_replace
   regexp_substr


_____

String
--------

.. currentmodule:: verticapy.sql.functions

.. autosummary:: 
   :toctree: api/

   length
   lower
   substr
   upper
   edit_distance
   soundex
   soundex_matches
   jaro_distance
   jaro_winkler_distance