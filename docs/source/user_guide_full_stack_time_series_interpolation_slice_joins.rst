.. _user_guide.full_stack.time_series_interpolation_slice_joins:

=============================================
Time Series Interpolation, Slices, and Joins
=============================================

One of the major problems with working with time series models is cleaning the data. Most time series models need to have equally sliced data, and most tools don't offer an easy way to do this.

Not only that, but missing values can distort predictions. You can fill these gaps with various interpolation methods.
Luckily, VerticaPy can easily slice and interpolate time series data. We'll demonstrate these functions with the `Smart Meter <https://github.com/vertica/VerticaPy/blob/master/examples/business/smart_meters/>`_ datasets.

.. code-block:: python
    
    import verticapy as vp

    sm_consumption = vp.read_csv(
        "sm_consumption.csv",
        dtype = {
            "meterID": "Integer",
            "dateUTC": "Timestamp(6)",
            "value": "Float(22)",
        }
    )
    sm_weather = vp.read_csv(
        "sm_weather.csv",
        dtype = {
            "dateUTC": "Timestamp(6)",
            "temperature": "Float(22)",
            "humidity": "Float(22)",
        }
    )
    sm_meters = vp.read_csv("sm_meters.csv")

.. note:: You can let Vertica automatically decide the data type, or you can manually force the data type on any column as seen above.

.. code-block:: python

    sm_consumption.head(100)

.. ipython:: python
    :suppress:

    import verticapy as vp
    sm_consumption = vp.read_csv(
        "SPHINX_DIRECTORY/source/_static/website/examples/data/smart_meters/sm_consumption.csv",
        dtype = {
            "meterID": "Integer",
            "dateUTC": "Timestamp(6)",
            "value": "Float(22)",
        }
    )
    sm_weather = vp.read_csv(
        "SPHINX_DIRECTORY/source/_static/website/examples/data/smart_meters/sm_weather.csv",
        dtype = {
            "dateUTC": "Timestamp(6)",
            "temperature": "Float(22)",
            "humidity": "Float(22)",
        }
    )
    sm_meters = vp.read_csv("SPHINX_DIRECTORY/source/_static/website/examples/data/smart_meters/sm_meters.csv")
    res = sm_consumption.head(100)
    html_file = open("SPHINX_DIRECTORY/figures/ug_fs_table_tsisj_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_table_tsisj_1.html

.. code-block:: python

    sm_weather.head(100)

.. ipython:: python
    :suppress:

    res = sm_weather.head(100)
    html_file = open("SPHINX_DIRECTORY/figures/ug_fs_table_tsisj_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_table_tsisj_2.html

Our first dataset has a lot of gaps, so let's slice and interpolate the energy consumption every 30 minutes.

.. code-block:: python

    interpolate = sm_consumption.interpolate(
        ts = "dateUTC",
        rule = "30 minutes",
        method = {"value": "linear"},
        by = ["meterID"],
    )
    interpolate.head(100)

.. ipython:: python
    :suppress:

    interpolate = sm_consumption.interpolate(
        ts = "dateUTC",
        rule = "30 minutes",
        method = {"value": "linear"},
        by = ["meterID"],
    )
    res = interpolate.head(100)
    html_file = open("SPHINX_DIRECTORY/figures/ug_fs_table_tsisj_3.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_table_tsisj_3.html

VerticaPy achieves this with its close integration with Vertica; by leveraging Vertica's comutational power and the ``TIMESERIES`` clause, slicing and interpolation is easy.

.. ipython:: python

    print(interpolate.current_relation())

Having sliced data on regular interval of time can often make it easier to join your the data with other datasets. For example, let's join the ``weather`` dataset with the ``smart_meters_consumption`` dataset on ``dateUTC``.

.. code-block:: python

    interpolate.join(
        sm_weather,
        how = "left",
        on = {"dateUTC": "dateUTC"},
        expr2 = ["temperature", "humidity"],
    )

.. ipython:: python
    :suppress:

    res = interpolate.join(
        sm_weather,
        how = "left",
        on = {"dateUTC": "dateUTC"},
        expr2 = ["temperature", "humidity"],
    )
    html_file = open("SPHINX_DIRECTORY/figures/ug_fs_table_tsisj_4.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_table_tsisj_4.html

Keep in mind that slicing, interpolating, and joins can be computationally expensive.

Thanks to Vertica's built-in clauses and options, VerticaPy can perform joins based on interpolated data. In the following example, we'll have Vertica identify the closest time series to our key and merge the two datasets.

.. code-block:: python

    sm_consumption.join(
        sm_weather,
        how = "left",
        on_interpolate = {"dateUTC": "dateUTC"},
        expr2 = ["temperature", "humidity"],
    )

.. ipython:: python
    :suppress:

    res = sm_consumption.join(
        sm_weather,
        how = "left",
        on_interpolate = {"dateUTC": "dateUTC"},
        expr2 = ["temperature", "humidity"],
    )
    html_file = open("SPHINX_DIRECTORY/figures/ug_fs_table_tsisj_5.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_table_tsisj_5.html

.. ipython:: python

    print(
        sm_consumption.join(
            sm_weather,
            how = "left",
            on_interpolate = {"dateUTC": "dateUTC"},
            expr2 = ["temperature", "humidity"],
        ).current_relation()
    )

Vertica offers powerful methods for cleaning time series data, and you can leverage it all with the flexibility of Python.