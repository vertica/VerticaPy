.. _api.sql.geospatial:

==============
Geospatial
==============

______

Geospatial Functions
--------------------------

.. currentmodule:: verticapy.sql.geo

.. autosummary:: 
   :toctree: api/

   coordinate_converter
   intersect
   split_polygon_n


_____

Index Functions
--------------------------

.. currentmodule:: verticapy.sql.geo

.. autosummary:: 
   :toctree: api/

   create_index
   describe_index
   rename_index



______

Import/Export
--------------

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