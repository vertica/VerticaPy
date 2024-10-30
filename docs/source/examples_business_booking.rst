.. _examples.business.booking:

Booking
========

This example uses the ``expedia`` dataset to predict, based on site activity, whether a user is likely to make a booking. You can download the Jupyter Notebook of the study `here <https://github.com/vertica/VerticaPy/blob/master/examples/understand/business/booking/booking.ipynb>`_ and the dataset `here <https://www.kaggle.com/c/expedia-hotel-recommendations/data>`_.

- **cnt:** Number of similar events in the context of the same user session.
- **user_location_city:** The ID of the city in which the customer is located.
- **is_package:** 1 if the click/booking was generated as a part of a package (i.e. combined with a flight), 0 otherwise.
- **user_id:** ID of the user.
- **srch_children_cnt:** The number of (extra occupancy) children specified in the hotel room.
- **channel:** marketing ID of a marketing channel.
- **hotel_cluster:** ID of a hotel cluster.
- **srch_destination_id:** ID of the destination where the hotel search was performed.
- **is_mobile:** 1 if the user is on a mobile device, 0 otherwise.
- **srch_adults_cnt:** The number of adults specified in the hotel room.
- **user_location_country:** The ID of the country in which the customer is located.
- **srch_destination_type_id:** ID of the destination where the hotel search was performed.
- **srch_rm_cnt:** The number of hotel rooms specified in the search.
- **posa_continent:** ID of the continent associated with the site_name.
- **srch_ci:** Check-in date.
- **user_location_region:** The ID of the region in which the customer is located.
- **hotel_country:** Hotel's country.
- **srch_co:** Check-out date.
- **is_booking:** 1 if a booking, 0 if a click.
- **orig_destination_distance:** Physical distance between a hotel and a customer at the time of search. A null means the distance could not be calculated.
- **hotel_continent:** Hotel continent.
- **site_name:** ID of the Expedia point of sale (i.e. Expedia.com, Expedia.co.uk, Expedia.co.jp, ...).

We will follow the data science cycle (Data Exploration - Data Preparation - Data Modeling - Model Evaluation - Model Deployment) to solve this problem.

Initialization
---------------

This example uses the following version of VerticaPy:

.. ipython:: python
    
    import verticapy as vp

    vp.__version__

Connect to Vertica. This example uses an existing connection called ``VerticaDSN``. 
For details on how to create a connection, see the :ref:`connection` tutorial.
You can skip the below cell if you already have an established connection.

.. code-block:: python
    
    vp.connect("VerticaDSN")

Let's create a Virtual DataFrame of the dataset.

.. code-block:: python

    expedia = vp.read_csv("expedia.csv", parse_nrows = 1000)
    expedia.head(5)

.. ipython:: python
    :suppress:

    expedia = vp.read_csv("SPHINX_DIRECTORY/source/_static/website/examples/data/booking/expedia.csv")
    res = expedia.head(5)
    html_file = open("SPHINX_DIRECTORY/figures/examples_expedia_table_head.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_expedia_table_head.html

.. warning::
    
    This example uses a sample dataset. For the full analysis, you should consider using the complete dataset.

Data Exploration and Preparation
---------------------------------

Sessionization is the process of gathering clicks for a certain period of time. We usually consider that after 30 minutes of inactivity, the user session ends (``date_time - lag(date_time) > 30 minutes``). For these kinds of use cases, aggregating sessions with meaningful statistics is the key for making accurate predictions.

We start by using the :py:func:`~verticapy.vDataFrame.sessionize` method to create the variable ``session_id``. We can then use this variable to aggregate the data.

.. code-block:: python

    expedia.sessionize(
        ts = "date_time", 
        by = ["user_id"], 
        session_threshold = "30 minutes", 
        name = "session_id",
    )

.. ipython:: python
    :suppress:

    res = expedia.sessionize(
        ts = "date_time", 
        by = ["user_id"], 
        session_threshold = "30 minutes", 
        name = "session_id",
    )
    html_file = open("SPHINX_DIRECTORY/figures/examples_expedia_sessionize.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_expedia_sessionize.html

The duration of the trip should also influence/be indicative of the user's behavior on the site, so we'll take that into account.

.. ipython:: python

    expedia["trip_duration"] = expedia["srch_co"] - expedia["srch_ci"]

If a user looks at the same hotel several times, then it might mean that they're looking to book that hotel during the session.

.. code-block:: python

    expedia.analytic(
        "mode", 
        columns = "hotel_cluster", 
        by = [
            "user_id",
            "session_id",
        ], 
        name = "mode_hotel_cluster",
        add_count = True,
    )

.. ipython:: python
    :suppress:

    res = expedia.analytic(
        "mode", 
        columns = "hotel_cluster", 
        by = [
            "user_id",
            "session_id",
        ], 
        name = "mode_hotel_cluster",
        add_count = True,
    )
    html_file = open("SPHINX_DIRECTORY/figures/examples_expedia_analytic.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_expedia_analytic.html

We can now aggregate the session and get some useful statistics out of it:

 - **end_session_date_time:** Date and time when the session ends.
 - **session_duration:** Session duration.
 - **is_booking:** 1 if the user booked during the session, 0 otherwise.
 - **trip_duration:** Trip duration.
 - **orig_destination_distance:** Average of the physical distances between the hotels and the customer.
 - **srch_family_cnt:** The number of people specified in the hotel room.

.. ipython:: python

    import verticapy.sql.functions as fun

    expedia = expedia.groupby(
        columns = [
            "user_id",
            "session_id", 
            "mode_hotel_cluster_count",
        ], 
        expr = [
            fun.max(expedia["date_time"])._as("end_session_date_time"),
            ((fun.max(expedia["date_time"]) - fun.min(expedia["date_time"])) / fun.interval("1 second"))._as(
                "session_duration"
            ),
            fun.max(expedia["is_booking"])._as("is_booking"),
            fun.avg(expedia["trip_duration"])._as("trip_duration"),
            fun.avg(expedia["orig_destination_distance"])._as("avg_distance"),
            fun.sum(expedia["cnt"])._as("nb_click_session"),
            fun.median(expedia["srch_children_cnt"] + expedia["srch_adults_cnt"])._as("srch_family_cnt"),
        ],
    )

Let's look at the missing values.

.. code-block:: python

    expedia.count_percent()

.. ipython:: python
    :suppress:

    res = expedia.count_percent()
    html_file = open("SPHINX_DIRECTORY/figures/examples_expedia_count_percent.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_expedia_count_percent.html

Let's impute the missing values for ``avg_distance`` and ``trip_duration``.

.. code-block:: python

    expedia["avg_distance" ].fillna(method = "avg")
    expedia["trip_duration"].fillna(method = "avg")

.. ipython:: python
    :suppress:

    expedia["avg_distance" ].fillna(method = "avg")
    res = expedia["trip_duration"].fillna(method = "avg")
    html_file = open("SPHINX_DIRECTORY/figures/examples_expedia_fillna_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_expedia_fillna_1.html

We can then look at the links between the variables. We will use Spearman's rank correleation coefficient to get all the monotonic relationships.

.. code-block:: python

    expedia.corr(method = "spearman")

.. ipython:: python
    :suppress:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = expedia.corr(method = "spearman", width = 750, with_numbers = False)
    fig.write_html("SPHINX_DIRECTORY/figures/examples_expedia_corr.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_expedia_corr.html

We can see huge links between some of the variables (``mode_hotel_cluster_count`` and ``session_duration``) and our response variable (``is_booking``). A logistic regression would work well in this case because the response and predictors have a monotonic relationship.

Machine Learning
-----------------

Let's create our :py:mod:`~verticapy.machine_learning.vertica.linear_model.LogisticRegression` model.

.. ipython:: python

    from verticapy.machine_learning.vertica import LogisticRegression

    model_logit = LogisticRegression(
        max_iter = 1000, 
        solver = "BFGS",
    )
    model_logit.fit(
        expedia, 
        [
            "avg_distance", 
            "session_duration",
            "nb_click_session",
            "mode_hotel_cluster_count",
            "session_id",
            "srch_family_cnt",
            "trip_duration",
        ], 
        "is_booking",
    )

None of our coefficients are rejected (``pvalue = 0``). Let's look at their importance.

.. code-block:: python

    model_logit.features_importance()

.. ipython:: python
    :suppress:

    fig = model_logit.features_importance()
    fig.write_html("SPHINX_DIRECTORY/figures/examples_expedia_features_importance.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_expedia_features_importance.html

It looks like there are two main predictors: ``mode_hotel_cluster_count`` and ``trip_duration``. According to our model, users likely to make a booking during a particular session will tend to:

- look at the same hotel many times.
- look for a shorter trip duration.
- not click as much (spend more time at the same web page).

Let's add our prediction to the :py:mod:`~verticapy.vDataFrame`.

.. code-block:: python

    model_logit.predict_proba(
        expedia, 
        name = "booking_prob_logit",
        pos_label = 1,
    )

.. ipython:: python
    :suppress:

    res = model_logit.predict_proba(
        expedia, 
        name = "booking_prob_logit",
        pos_label = 1,
    )
    html_file = open("SPHINX_DIRECTORY/figures/examples_expedia_predict_proba_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_expedia_predict_proba_1.html

While analyzing the following boxplot (prediction partitioned by ``is_booking``), we can notice that the ``cutoff`` is around 0.22 because most of the positive predictions have a probability between 0.23 and 0.5. Most of the negative predictions are between 0.05 and 0.2.

.. code-block:: python

    expedia["booking_prob_logit"].boxplot(by = "is_booking")

.. ipython:: python
    :suppress:
    :okwarning:

    fig = expedia["booking_prob_logit"].boxplot(by = "is_booking")
    fig.write_html("SPHINX_DIRECTORY/figures/examples_expedia_predict_boxplot_1.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_expedia_predict_boxplot_1.html

Let's confirm our hypothesis by computing the best ``cutoff``.

.. ipython:: python

    model_logit.score(metric = "best_cutoff")

Let's look at the efficiency of our model with a cutoff of 0.22.

.. code-block:: python

    model_logit.report(cutoff = 0.22)

.. ipython:: python
    :suppress:

    res = model_logit.report(cutoff = 0.22)
    html_file = open("SPHINX_DIRECTORY/figures/examples_expedia_cutoff_best.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_expedia_cutoff_best.html

ROC Curve:
+++++++++++

.. code-block:: python

    model_logit.roc_curve()

.. ipython:: python
    :suppress:

    fig = model_logit.roc_curve()
    fig.write_html("SPHINX_DIRECTORY/figures/examples_expedia_roc_curve_1.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_expedia_roc_curve_1.html

We're left with an excellent model. With this, we can predict whether a user will book a hotel during a specific session and make adjustments to our site accordingly. For example, to influence a user to make a booking, we could propose new hotels.

Conclusion
-----------

We've solved our problem in a Pandas-like way, all without ever loading data into memory!