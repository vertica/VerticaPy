.. _user_guide.data_exploration.descriptive_statistics:

=======================
Descriptive Statistics
=======================


The easiest way to understand data is to aggregate it. 
An aggregation is a number or a category which summarizes the data. 
VerticaPy lets you compute all well-known aggregation in a single line.

The :py:func:`~verticapy.vDataFrame.agg` method is the best way to compute multiple aggregations on multiple columns at the same time.

.. ipython:: python

  import verticapy as vp
  help(vp.vDataFrame.agg)

This is a tremendously useful function for understanding your data. 
Let's use the `churn dataset <https://github.com/vertica/VerticaPy/tree/master/docs/source/notebooks/data_exploration/correlations/data>`_

.. code-block::

  vdf = vp.read_csv("data/churn.csv")
  vdf.agg(func = ["min", "10%", "median", "90%", "max", "kurtosis", "unique"])

.. ipython:: python
  :suppress:

  vdf = vp.read_csv("/project/data/VerticaPy/docs/source/_static/website/user_guides/data_exploration/churn.csv")
  res = vdf.agg(func = ["min", "10%", "median", "90%", "max", "kurtosis", "unique"])
  html_file = open("/project/data/VerticaPy/docs/figures/user_guides_data_exploration_descriptive_stats_vdf_agg.html", "w")
  html_file.write(res._repr_html_())
  html_file.close()

.. raw:: html
  :file: /project/data/VerticaPy/docs/figures/user_guides_data_exploration_descriptive_stats_vdf_agg.html

Some methods, like :py:func:`~verticapy.vDataFrame.describe`, are 
abstractions of the :py:func:`~verticapy.vDataFrame.agg` method; 
they simplify the call to computing specific aggregations.

.. code-block::

  vdf.describe()

.. ipython:: python
  :suppress:

  res = vdf.describe()
  html_file = open("/project/data/VerticaPy/docs/figures/user_guides_data_exploration_descriptive_stats_vdf_describe.html", "w")
  html_file.write(res._repr_html_())
  html_file.close()

.. raw:: html
  :file: /project/data/VerticaPy/docs/figures/user_guides_data_exploration_descriptive_stats_vdf_describe.html


.. code-block::

  vdf.describe(method = "all")

.. ipython:: python
  :suppress:

  res = vdf.describe(method = "all")
  html_file = open("/project/data/VerticaPy/docs/figures/user_guides_data_exploration_descriptive_stats_vdf_describe_all.html", "w")
  html_file.write(res._repr_html_())
  html_file.close()

.. raw:: html
  :file: /project/data/VerticaPy/docs/figures/user_guides_data_exploration_descriptive_stats_vdf_describe_all.html


.. code-block::

  vdf.describe(method = "categorical")

.. ipython:: python
  :suppress:

  res = vdf.describe(method = "categorical")
  html_file = open("/project/data/VerticaPy/docs/figures/user_guides_data_exploration_descriptive_stats_vdf_describe_categorical.html", "w")
  html_file.write(res._repr_html_())
  html_file.close()

.. raw:: html
  :file: /project/data/VerticaPy/docs/figures/user_guides_data_exploration_descriptive_stats_vdf_describe_categorical.html


Multi-column aggregations can also be called with many built-in methods. For example, you can compute the :py:func:`~verticapy.vDataFrameavg` of all the numerical columns in just one line.

.. code-block::

  vdf.avg()

.. ipython:: python
  :suppress:

  res = vdf.avg()
  html_file = open("/project/data/VerticaPy/docs/figures/user_guides_data_exploration_descriptive_stats_vdf_avg.html", "w")
  html_file.write(res._repr_html_())
  html_file.close()

.. raw:: html
  :file: /project/data/VerticaPy/docs/figures/user_guides_data_exploration_descriptive_stats_vdf_avg.html

Or just the 'median' of a specific column.

.. ipython:: python

  vdf["tenure"].median()


The approximate median is automatically computed. Set the parameter 'approx' to False to get the exact median.

.. ipython:: python

  vdf["tenure"].median(approx=False)

You can also use the 'groupby' method to compute customized aggregations.

.. code-block:: python

  # SQL way
  vdf.groupby(
      [
          "gender",
          "Contract"
      ],
      ["AVG(DECODE(Churn, 'Yes', 1, 0)) AS Churn"]
  )

.. ipython:: python
  :suppress:

  res = vdf.groupby(
      [
          "gender",
          "Contract"
      ],
      ["AVG(DECODE(Churn, 'Yes', 1, 0)) AS Churn"]
  )
  html_file = open("/project/data/VerticaPy/docs/figures/user_guides_data_exploration_descriptive_stats_group_by.html", "w")
  html_file.write(res._repr_html_())
  html_file.close()

.. raw:: html
  :file: /project/data/VerticaPy/docs/figures/user_guides_data_exploration_descriptive_stats_group_by.html


.. code-block:: python

  # Pythonic way
  import verticapy.sql.functions as st
  vdf.groupby(
      ["gender", "Contract"],
      [
          st.min(vdf["tenure"])._as("min_tenure"),
          st.max(vdf["tenure"])._as("max_tenure")
      ]
  )

.. ipython:: python
  :suppress:

  import verticapy.sql.functions as st
  res = vdf.groupby(
      ["gender", "Contract"],
      [
          st.min(vdf["tenure"])._as("min_tenure"),
          st.max(vdf["tenure"])._as("max_tenure")
      ]
  )
  html_file = open("/project/data/VerticaPy/docs/figures/user_guides_data_exploration_descriptive_stats_group_by_python.html", "w")
  html_file.write(res._repr_html_())
  html_file.close()

.. raw:: html
  :file: /project/data/VerticaPy/docs/figures/user_guides_data_exploration_descriptive_stats_group_by_python.html

Computing many aggregations at the same time can be resource intensive. 
You can use the parameters `ncols_block` and `processes` to manage the ressources.

For example, the parameter `ncols_block` will divide the main query into 
smaller using a specific number of columns. The parameter `processes` 
allows you to manage the number of queries you want to send at the same time. 

An entire example is available in the :py:func:`~verticapy.vDataFrame.agg` documentation.