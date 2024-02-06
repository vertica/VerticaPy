.. _api.vdataframe:

============
vDataFrame
============


.. tab:: vDataFrame

   vDataFrame
   -----------

   .. currentmodule:: verticapy

   .. autoclass:: vDataFrame
      :members:

.. tab:: vDataColumn

   vDataColumn
   ------------

   .. currentmodule:: verticapy

   .. autoclass:: vDataColumn
      :members:

______


Plotting
---------


There are three main plotting libraries available in VerticaPy:

- Plotly
- Highcharts
- Matplotlib

To access the bases classes of all the plotting libraries click 
the dropdown button below.

.. note:: 

   The documentation for these classes is provided solely to enhance the 
   user's understanding of the implementations. Users are not required to 
   interact directly with these classes, and we do not recommend doing so.

.. dropdown:: Plotting Base Classes

   .. tab:: Plotly

      .. currentmodule:: verticapy.plotting._plotly

      .. autosummary:: 
         :toctree: api/

         ACFPlot
         BarChart
         BarChart2D
         BoxPlot
         CandleStick
         ChampionChallengerPlot
         ContourPlot
         CutoffCurve
         DensityPlot
         ElbowCurve
         HeatMap
         Histogram
         HorizontalBarChart
         HorizontalBarChart2D
         ImportanceBarChart
         LiftChart
         LinePlot
         LogisticRegressionPlot
         LOFPlot
         MultiDensityPlot
         MultiLinePlot
         NestedPieChart
         OutliersPlot
         PCACirclePlot
         PieChart
         PlotlyBase
         PRCCurve
         RangeCurve
         RegressionPlot
         RegressionTreePlot
         ROCCurve
         ScatterPlot
         SpiderChart
         StepwisePlot
         SVMClassifierPlot
         TSPlot
         VoronoiPlot

   .. tab:: Highcharts

      .. currentmodule:: verticapy.plotting._highcharts

      .. autosummary:: 
         :toctree: api/

         ACFPlot
         ACFPACFPlot
         BarChart
         BarChart2D
         BoxPlot
         CandleStick
         ChampionChallengerPlot
         ContourPlot
         CutoffCurve
         DensityPlot
         ElbowCurve
         HeatMap
         Histogram
         HighchartsBase
         HorizontalBarChart
         HorizontalBarChart2D
         ImportanceBarChart
         LiftChart
         LinePlot
         LogisticRegressionPlot
         LOFPlot
         MultiDensityPlot
         MultiLinePlot
         NestedPieChart
         OutliersPlot
         PCACirclePlot
         PieChart
         PRCCurve
         RangeCurve
         RegressionPlot
         RegressionTreePlot
         ROCCurve
         ScatterPlot
         SpiderChart
         StepwisePlot
         SVMClassifierPlot
         TSPlot

   .. tab:: Matplotlib

      .. currentmodule:: verticapy.plotting._matplotlib

      .. autosummary:: 
         :toctree: api/

         ACFPlot
         ACFPACFPlot
         AnimatedBarChart
         AnimatedBase
         AnimatedBubblePlot
         AnimatedLinePlot
         AnimatedPieChart
         BarChart
         BarChart2D
         BoxPlot
         CandleStick
         ChampionChallengerPlot
         ContourPlot
         CutoffCurve
         DensityPlot
         DensityPlot2D
         ElbowCurve
         HeatMap
         Histogram
         HorizontalBarChart
         HorizontalBarChart2D
         ImportanceBarChart
         LiftChart
         LinePlot
         LogisticRegressionPlot
         LOFPlot
         MatplotlibBase
         MultiDensityPlot
         MultiLinePlot
         NestedPieChart
         OutliersPlot
         PCACirclePlot
         PieChart
         PRCCurve
         RangeCurve
         RegressionPlot
         RegressionTreePlot
         ROCCurve
         ScatterMatrix
         ScatterPlot
         SpiderChart
         StepwisePlot
         SVMClassifierPlot
         TSPlot
         VoronoiPlot

General
~~~~~~~~


.. tab:: vDataFrame

   ``vDataFrame.func(...)``

   .. currentmodule:: verticapy.vDataFrame

   .. autosummary:: 
      :toctree: api/

      bar
      barh
      boxplot
      contour
      density
      heatmap
      hexbin
      hist
      outliers_plot
      pie
      pivot_table
      plot
      scatter
      scatter_matrix
      pivot_table_chi2
      range_plot
      

.. tab:: vDataColumn

   ``vDataFrame[].func(...)``

   .. currentmodule:: verticapy.vDataColumn

   .. autosummary:: 
      :toctree: api/

      bar
      barh
      candlestick
      boxplot
      density
      hist
      pie
      plot
      range_plot
      spider

Animated
~~~~~~~~~
.. tab:: vDataFrame

   ``vDataFrame.func(...)``

   .. currentmodule:: verticapy.vDataFrame

   .. autosummary:: 
      :toctree: api/

      animated_bar
      animated_pie
      animated_plot
      animated_scatter

______


Descriptive Statistics
-----------------------

.. tab:: vDataFrame

   ``vDataFrame.func(...)``

   .. currentmodule:: verticapy.vDataFrame

   .. autosummary::
      :toctree: api/

      aad
      aggregate
      all
      any
      avg
      count
      count_percent
      describe
      duplicated
      kurtosis
      mad
      max
      median
      min
      nunique
      product
      quantile
      score
      sem
      skewness
      std
      sum
      var


   

.. tab:: vDataColumn


   ``vDataFrame[].func(...)``

   .. currentmodule:: verticapy.vDataColumn

   .. autosummary::
      :toctree: api/

      aad
      aggregate
      avg
      count
      describe
      distinct
      kurtosis
      mad
      max
      median
      min
      mode
      nlargest
      nsmallest
      nunique
      product
      quantile
      sem
      skewness
      std
      sum
      topk
      value_counts
      var



______


Correlation & Dependency
-------------------------

General
~~~~~~~~

.. tab:: vDataFrame

   ``vDataFrame.func(...)``

   .. currentmodule:: verticapy.vDataFrame

   .. autosummary:: 
      :toctree: api/

      acf
      corr
      corr_pvalue
      cov
      iv_woe
      pacf
      regr



.. tab:: vDataColumn

   ``vDataFrame[].func(...)``

   .. currentmodule:: verticapy.vDataColumn

   .. autosummary::
      :toctree: 

      iv_woe

Time-series
~~~~~~~~~~~


.. tab:: vDataFrame

   ``vDataFrame.func(...)``

   .. currentmodule:: verticapy.vDataFrame

   .. autosummary:: 
      :toctree: api/

      acf
      pacf


______


Preprocessing
--------------

Encoding
~~~~~~~~~



.. tab:: vDataFrame


   ``vDataFrame.func(...)``

   .. currentmodule:: verticapy.vDataFrame

   .. autosummary:: 
      :toctree: 

      case_when
      one_hot_encode


.. tab:: vDataColumn

   ``vDataFrame[].func(...)``

   .. currentmodule:: verticapy.vDataColumn

   .. autosummary::
      :toctree: 


      cut
      decode
      discretize
      label_encode
      mean_encode
      one_hot_encode



Dealing With Missing Values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. tab:: vDataFrame

   ``vDataFrame.func(...)``

   .. currentmodule:: verticapy.vDataFrame

   .. autosummary:: 
      :toctree: 

      dropna
      fillna
      interpolate
      


.. tab:: vDataColumn

   ``vDataFrame[].func(...)``

   .. currentmodule:: verticapy.vDataColumn

   .. autosummary::
      :toctree: 


      dropna
      fillna


Duplicate Values
~~~~~~~~~~~~~~~~~~


.. tab:: vDataFrame

   ``vDataFrame.func(...)``

   .. currentmodule:: verticapy.vDataFrame

   .. autosummary:: 
      :toctree: 

      drop_duplicates

Normalization and Global Outliers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab:: vDataFrame

   ``vDataFrame.func(...)``

   .. currentmodule:: verticapy.vDataFrame

   .. autosummary:: 
      :toctree: 

      outliers
      scale
      


.. tab:: vDataColumn

   ``vDataFrame[].func(...)``

   .. currentmodule:: verticapy.vDataColumn

   .. autosummary::
      :toctree: 

      clip
      fill_outliers
      normalize 





Data Types Conversion
~~~~~~~~~~~~~~~~~~~~~~~~

.. tab:: vDataFrame

   ``vDataFrame.func(...)``

   .. currentmodule:: verticapy.vDataFrame

   .. autosummary:: 
      :toctree: 

      astype
      bool_to_int


.. tab:: vDataColumn

   ``vDataFrame[].func(...)``

   .. currentmodule:: verticapy.vDataColumn

   .. autosummary::
      :toctree: 

      astype

Formatting
~~~~~~~~~~~


.. tab:: vDataFrame

   ``vDataFrame.func(...)``

   .. currentmodule:: verticapy.vDataFrame

   .. autosummary::
      :toctree: 

      format_colnames
      get_match_index
      is_colname_in
      merge_similar_names
      explode_array

.. tab:: vDataColumn

   ``vDataFrame[].func(...)``

   .. currentmodule:: verticapy.vDataColumn

   .. autosummary::
      :toctree: 

      astype
      rename


Splitting into Train/Test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. tab:: vDataFrame

   ``vDataFrame.func(...)``

   .. currentmodule:: verticapy.vDataFrame

   .. autosummary:: 
      :toctree: 

      train_test_split

Working with Weights
~~~~~~~~~~~~~~~~~~~~~

.. tab:: vDataFrame

   ``vDataFrame.func(...)``

   .. currentmodule:: verticapy.vDataFrame

   .. autosummary::
      :toctree: 

      add_duplicates

Complete Disjunctive Table
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab:: vDataFrame

   ``vDataFrame.func(...)``

   .. currentmodule:: verticapy.vDataFrame

   .. autosummary::
      :toctree: 

      cdt

______



.. _api.vdataframe.features_engineering:

Features Engineering
---------------------

Analytic Functions
~~~~~~~~~~~~~~~~~~

.. tab:: vDataFrame

   ``vDataFrame.func(...)``

   .. currentmodule:: verticapy.vDataFrame

   .. autosummary::
      :toctree: 

      analytic
      interpolate
      sessionize


Custom Features Creation
~~~~~~~~~~~~~~~~~~~~~~~~

.. tab:: vDataFrame

   ``vDataFrame.func(...)``

   .. currentmodule:: verticapy.vDataFrame

   .. autosummary::
      :toctree: 

      case_when
      
   .. currentmodule:: verticapy.vDataFrame

   .. autosummary::
      :toctree:    
      
      eval


Features Transformations
~~~~~~~~~~~~~~~~~~~~~~~~

.. tab:: vDataFrame

   ``vDataFrame.func(...)``

   .. currentmodule:: verticapy.vDataFrame

   .. autosummary:: 
      :toctree: 

      abs
      apply
      applymap
      polynomial_comb
      swap


.. tab:: vDataColumn

   ``vDataFrame[].func(...)``

   .. currentmodule:: verticapy.vDataColumn

   .. autosummary::
      :toctree: 

      abs
      add
      apply
      apply_fun
      date_part
      div
      mul
      round
      slice
      sub



Moving Windows
~~~~~~~~~~~~~~~~

.. tab:: vDataFrame

   ``vDataFrame.func(...)``

   .. currentmodule:: verticapy.vDataFrame

   .. autosummary::
      :toctree: 

      cummax
      cummin
      cumprod
      cumsum
      rolling





Working with Text
~~~~~~~~~~~~~~~~~~~~~~~~

.. tab:: vDataFrame

   ``vDataFrame.func(...)``

   .. currentmodule:: verticapy.vDataFrame

   .. autosummary:: 
      :toctree: 

      regexp


.. tab:: vDataColumn

   ``vDataFrame[].func(...)``

   .. currentmodule:: verticapy.vDataColumn

   .. autosummary::
      :toctree: 

      str_contains
      str_count
      str_extract
      str_replace
      str_slice



Binary Operator Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab:: vDataColumn

   ``vDataFrame[].func(...)``

   .. currentmodule:: verticapy.vDataColumn

   .. autosummary:: 
      :toctree: 

      add
      div
      mul
      sub



Basic Feature Selection
~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab:: vDataFrame

   ``vDataFrame.func(...)``

   .. currentmodule:: verticapy.vDataFrame

   .. autosummary:: 
      :toctree: 

      chaid
      chaid_columns
______




Join, sort and transform
-------------------------

.. tab:: vDataFrame

   ``vDataFrame.func(...)``

   .. currentmodule:: verticapy.vDataFrame

   .. autosummary:: 
      :toctree: 

      append
      copy
      flat_vmap
      groupby
      join
      narrow
      pivot
      recommend
      sort


.. tab:: vDataColumn

   ``vDataFrame[].func(...)``

   .. currentmodule:: verticapy.vDataColumn

   .. autosummary::
      :toctree: 

      add_copy
______


Filter and Sample
------------------

Search
~~~~~~

.. tab:: vDataFrame

   ``vDataFrame.func(...)``

   .. currentmodule:: verticapy.vDataFrame

   .. autosummary:: 
      :toctree: 

      search

Sample
~~~~~~~~~~~~~~

.. tab:: vDataFrame

   ``vDataFrame.func(...)``

   .. currentmodule:: verticapy.vDataFrame

   .. autosummary:: 
      :toctree: 

      sample

Balance
~~~~~~~~

.. tab:: vDataFrame

   ``vDataFrame.func(...)``

   .. currentmodule:: verticapy.vDataFrame

   .. autosummary:: 
      :toctree: 

      balance



Filter Columns
~~~~~~~~~~~~~~

.. tab:: vDataFrame

   ``vDataFrame.func(...)``

   .. currentmodule:: verticapy.vDataFrame

   .. autosummary:: 
      :toctree: 

      drop
      select



.. tab:: vDataColumn

   ``vDataFrame[].func(...)``

   .. currentmodule:: verticapy.vDataColumn

   .. autosummary::
      :toctree: 

      drop
      drop_outliers


Filter Records
~~~~~~~~~~~~~~

.. tab:: vDataFrame

   ``vDataFrame.func(...)``

   .. currentmodule:: verticapy.vDataFrame

   .. autosummary:: 
      :toctree: 

      at_time
      between
      between_time
      filter
      first
      isin
      last


.. tab:: vDataColumn

   ``vDataFrame[].func(...)``

   .. currentmodule:: verticapy.vDataColumn

   .. autosummary::
      :toctree: 

      isin



______


Serialization
--------------

General Format
~~~~~~~~~~~~~~

.. tab:: vDataFrame

   ``vDataFrame.func(...)``

   .. currentmodule:: verticapy.vDataFrame

   .. autosummary:: 
      :toctree: 

      to_csv
      to_json
      to_shp


In-memory Object
~~~~~~~~~~~~~~~~~~


.. tab:: vDataFrame

   ``vDataFrame.func(...)``

   .. currentmodule:: verticapy.vDataFrame

   .. autosummary:: 
      :toctree: 

      to_numpy
      to_pandas
      to_list
      to_geopandas


Databases
~~~~~~~~~~

.. tab:: vDataFrame

   ``vDataFrame.func(...)``

   .. currentmodule:: verticapy.vDataFrame

   .. autosummary:: 
      :toctree: 

      to_db

Binary Format
~~~~~~~~~~~~~~~

.. tab:: vDataFrame

   ``vDataFrame.func(...)``

   .. currentmodule:: verticapy.vDataFrame

   .. autosummary:: 
      :toctree: 

      to_pickle
      


Utilities
----------

Information
~~~~~~~~~~~~

.. tab:: vDataFrame

   ``vDataFrame.func(...)``

   .. currentmodule:: verticapy.vDataFrame

   .. autosummary:: 
      :toctree: 

      catcol
      current_relation
      datecol
      dtypes
      empty
      explain
      get_columns
      head
      idisplay
      iloc
      info
      memory_usage
      expected_store_usage
      numcol
      shape
      tail



.. tab:: vDataColumn

   ``vDataFrame[].func(...)``

   .. currentmodule:: verticapy.vDataColumn

   .. autosummary::
      :toctree: 

      category
      ctype
      dtype
      get_len
      head
      iloc
      isarray
      isbool
      isdate
      isnum
      isvmap
      memory_usage
      store_usage
      tail




Management
~~~~~~~~~~~~~~~~~~~~~~

.. tab:: vDataFrame

   ``vDataFrame.func(...)``

   .. currentmodule:: verticapy.vDataFrame

   .. autosummary:: 
      :toctree: 

      del_catalog
      load
      save
      
