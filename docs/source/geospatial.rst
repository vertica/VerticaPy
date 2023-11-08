.. _api.geospatial:

==============
Geospatial
==============

______

Geospatial Function
--------------------------

.. currentmodule:: verticapy.sql.geo

.. autosummary:: 
   :toctree: api/

   coordinate_converter
   create_index
   describe_index
   intersect
   rename_index
   split_polygon_n

______

Mathematical Functions
------------------------

.. currentmodule:: verticapy

.. autosummary:: 
   :toctree: api/

   read_shp

.. tab:: vDataFrame

   ``vDataFrame.func(...)``

   .. currentmodule:: verticapy.vDataFrame

   .. autosummary::
      :toctree: api/

      to_geopandas
      to_shp

______

Plotting & Graohics
------------------------


.. tab:: vDataColumn

   ``vDataFrame[].func(...)``

   .. currentmodule:: verticapy.vDataColumn

   .. autosummary::
      :toctree: 

      geo_plot

______

Generic Functions
------------------------


.. tab:: vDataColumn

   ``vDataFrame[].func(...)``

   .. currentmodule:: verticapy.vDataColumn

   .. autosummary::
      :toctree: 

      apply