.. _user_guide.data_preparation.joins:

======
Joins
======

When working with datasets, we often need to merge data from different sources. To do this, we need keys on which to join our data.

Let's use the `US Flights 2015 datasets <https://www.kaggle.com/datasets/usdot/flight-delays>`_. We have three datasets.

First, we have information on each flight.

.. code-block:: python

    import verticapy as vp

    flights  = vp.read_csv("flights.csv")
    flights.head(100)

.. ipython:: python
    :suppress:

    import verticapy as vp
    flights = vp.read_csv("SPHINX_DIRECTORY/source/_static/website/examples/data/flights/flights.csv")
    res = flights.head(100)
    html_file = open("SPHINX_DIRECTORY/figures/ug_dp_table_join_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_dp_table_join_1.html

Second, we have information on each airport.

.. code-block:: python

    airports = vp.read_csv("airports.csv")
    airports.head(100)

.. ipython:: python
    :suppress:

    import verticapy as vp
    airports = vp.read_csv("SPHINX_DIRECTORY/source/_static/website/examples/data/flights/airports.csv",
                        dtype = {
                            "IATA_CODE": "Varchar(20)",
                            "AIRPORT": "Varchar(156)",
                            "CITY": "Varchar(60)",
                            "STATE": "Varchar(20)",
                            "COUNTRY": "Varchar(20)",
                            "LATITUDE": "float",
                            "LONGITUDE": "float",
                        }
    )
    res = airports.head(100)
    html_file = open("SPHINX_DIRECTORY/figures/ug_dp_table_join_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_dp_table_join_2.html

Third, we have the names of each airline.

.. code-block:: python

    airlines = vp.read_csv("airlines.csv")
    airlines.head(100)

.. ipython:: python
    :suppress:

    import verticapy as vp
    airlines = vp.read_csv("SPHINX_DIRECTORY/source/_static/website/examples/data/flights/airlines.csv")
    res = airlines.head(100)
    html_file = open("SPHINX_DIRECTORY/figures/ug_dp_table_join_3.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_dp_table_join_3.html

Notice that each dataset has a primary or secondary key on which to join the data. For example, we can join the 'flights' dataset to the 'airlines' and 'airport' datasets using the corresponding IATA code.

To join datasets in VerticaPy, use the vDataFrame's :py:func:`~verticapy.vDataFrame.join` method.

.. ipython:: python

    help(vp.vDataFrame.join)

Let's use a left join to merge the 'airlines' dataset and the 'flights' dataset.

.. code-block:: python

    flights = flights.join(
        airlines,
        how = "left",
        on = {"airline": "IATA_CODE"},
        expr2 = ["AIRLINE AS airline_long"],
    )
    flights.head(100)

.. ipython:: python
    :suppress:

    flights = flights.join(
        airlines,
        how = "left",
        on = {"airline": "IATA_CODE"},
        expr2 = ["AIRLINE AS airline_long"],
    )
    res = flights.head(100)
    html_file = open("SPHINX_DIRECTORY/figures/ug_dp_table_join_4.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_dp_table_join_4.html

Let's use two left joins to get the information on the origin and destination airports.

.. code-block:: python

    flights = flights.join(
        airports,
        how = "left",
        on = {"origin_airport": "IATA_CODE"},
        expr2 = [
            "LATITUDE AS origin_lat",
            "LONGITUDE AS origin_lon",
        ],
    )
    flights = flights.join(
        airports,
        how = "left",
        on = {"destination_airport": "IATA_CODE"},
        expr2 = [
            "LATITUDE AS destination_lat",
            "LONGITUDE AS destination_lon",
        ],
    )
    flights.head(100)

.. ipython:: python
    :suppress:

    flights = flights.join(
        airports,
        how = "left",
        on = {"origin_airport": "IATA_CODE"},
        expr2 = [
            "LATITUDE AS origin_lat",
            "LONGITUDE AS origin_lon",
        ],
    )
    flights = flights.join(
        airports,
        how = "left",
        on = {"destination_airport": "IATA_CODE"},
        expr2 = [
            "LATITUDE AS destination_lat",
            "LONGITUDE AS destination_lon",
        ],
    )
    res = flights.head(100)
    html_file = open("SPHINX_DIRECTORY/figures/ug_dp_table_join_5.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_dp_table_join_5.html

To avoid duplicate information, splitting the data into different tables is very important. Just imagine: what if we wrote the longitude and the latitude of the destination and origin airports for each flight? It would add way too many duplicates and drastically impact the volume of the data.

Cross joins are special: they don't need a key. Cross joins are used to perform mathematical operations.

Let's use a cross join of the 'airports' dataset on itself to compute the distance between every airport.

.. code-block:: python

    distances = airports.join(
        airports, 
        how = "cross", 
        expr1 = [
            "IATA_CODE AS airport1", 
            "LATITUDE AS airport1_latitude", 
            "LONGITUDE AS airport1_longitude"
        ],
        expr2 = [
            "IATA_CODE AS airport2", 
            "LATITUDE AS airport2_latitude", 
            "LONGITUDE AS airport2_longitude",
        ],
    )
    distances.filter("airport1 != airport2")

    import verticapy.sql.functions as fun

    distances["distance"] = fun.distance(
        distances["airport1_latitude"], 
        distances["airport1_longitude"],                                
        distances["airport2_latitude"],
        distances["airport2_longitude"],
    )

.. ipython:: python
    :suppress:

    distances = airports.join(
        airports, 
        how = "cross", 
        expr1 = [
            "IATA_CODE AS airport1", 
            "LATITUDE AS airport1_latitude", 
            "LONGITUDE AS airport1_longitude"
        ],
        expr2 = [
            "IATA_CODE AS airport2", 
            "LATITUDE AS airport2_latitude", 
            "LONGITUDE AS airport2_longitude",
        ],
    )
    distances.filter("airport1 != airport2")

    import verticapy.sql.functions as fun

    distances["distance"] = fun.distance(
        distances["airport1_latitude"], 
        distances["airport1_longitude"],                                
        distances["airport2_latitude"],
        distances["airport2_longitude"],
    )
    res = distances["distance"]
    html_file = open("SPHINX_DIRECTORY/figures/ug_dp_table_join_6.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_dp_table_join_6.html

VerticaPy offers many powerful options for joining datasets.

In the next lesson, we'll learn how to deal with duplicates.