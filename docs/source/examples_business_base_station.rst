.. _examples.business.base_station:

Base Station Positions
==================================

This example uses the Telecom Dataset, provided by Shanghai Telecom, to predict the optimal positions for base radio stations. 
This dataset contains more than ``7.2`` million records about people's 
Internet access through ``3,233`` base stations from ``9,481`` mobile phones 
over period of six months.

The dataset can be found `here <http://sguangwang.com/TelecomDataset.html>`_. It consists of:

- **user_id :** User's ID.
- **start_time :** When the record begins.
- **end_time :** When the record ends.
- **latitude :** Latitude of the base station.
- **longitude :** Longitude of the base station.

To complement the study, we'll also use the shanghai_districts dataset, which contains information on Shanghai's districts. Some of the columns include:

- **name :** Name of the district.
- **division_code :** Division code of the district.
- **area :** Area of the district in square kilometers.
- **population :** Population of the district.
- **density :** Density of the district.
- **geometry :** Polygon of type ``Geometry`` that contains the coordinates of the district.

You can download the Jupyter notebook of this study `here <https://github.com/vertica/VerticaPy/blob/master/examples/business/base_station/base_station.ipynb>`_.

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


Let's load the two datasets.

.. code-block:: python

    # Creating the schema
    vp.drop("shanghai", method = "schema")
    vp.create_schema("shanghai")

    # Libraries import
    import matplotlib

    import verticapy.sql.functions as fun
    from verticapy.datasets import load_world

    # Increasing video limit
    matplotlib.rcParams['animation.embed_limit'] = 2 ** 128

    #######
    # CDR #
    #######
    cdr = vp.read_csv(
        "shanghai_cdr.csv", 
        schema = "shanghai", 
        table_name = "cdr", 
        sep = ",",
        parse_nrows = 1000,
    )
    # Unique Row id: It will be used to compute the Polygons intersection
    cdr["row_id"] = "ROW_NUMBER() OVER(ORDER BY user_id, start_time)"

    ######################
    # Shanghai Districts #
    ######################
    shanghai_districts = vp.read_csv(
        "shanghai_districts.csv", 
        schema = "shanghai", 
        table_name = "districts", 
        sep = ",",
    )
    # Converting the districts to Geometry
    shanghai_districts = shanghai_districts["geometry"].apply("ST_GeomFromText({})")
    # Creating Shanghai 
    shanghai_districts["district_level"] = fun.case_when(
        shanghai_districts["number"] <= 7, 'Downtown',
        shanghai_districts["number"] <= 11, 'Suburb1',
        shanghai_districts["number"] <= 15, 'Suburb2',
        'Suburb3',
    )

.. ipython:: python
    :suppress:

    vp.drop("shanghai", method = "schema")
    vp.create_schema("shanghai")
    import matplotlib
    import verticapy.sql.functions as fun
    from verticapy.datasets import load_world
    matplotlib.rcParams['animation.embed_limit'] = 2 ** 128
    cdr = vp.read_csv(
        "SPHINX_DIRECTORY/source/_static/website/examples/data/base_station/shanghai_cdr.csv", 
        schema = "shanghai", 
        table_name = "cdr", 
        sep = ",",
        parse_nrows = 1000,
    )
    # Unique Row id: It will be used to compute the Polygons intersection
    cdr["row_id"] = "ROW_NUMBER() OVER(ORDER BY user_id, start_time)"
    shanghai_districts = vp.read_csv(
        "SPHINX_DIRECTORY/source/_static/website/examples/data/base_station/shanghai_districts.csv", 
        schema = "shanghai", 
        table_name = "districts", 
        sep = ",",
    )
    # Converting the districts to Geometry
    shanghai_districts = shanghai_districts["geometry"].apply("ST_GeomFromText({})")
    # Creating Shanghai 
    shanghai_districts["district_level"] = fun.case_when(
        shanghai_districts["number"] <= 7, 'Downtown',
        shanghai_districts["number"] <= 11, 'Suburb1',
        shanghai_districts["number"] <= 15, 'Suburb2',
        'Suburb3',
    )

These datasets contain the following:

.. code-block:: python

    cdr.head(100)

.. ipython:: python
    :suppress:

    res = cdr.head(100)
    html_file = open("SPHINX_DIRECTORY/figures/examples_base_station_cdr_head.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()


.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_base_station_cdr_head.html

.. code-block:: python

    shanghai_districts.head(100)

.. ipython:: python
    :suppress:

    res = shanghai_districts.head(100)
    html_file = open("SPHINX_DIRECTORY/figures/examples_base_station_shanghai_district_head.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()


.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_base_station_shanghai_district_head.html

Data Exploration
----------------

Detecting outliers
+++++++++++++++++++

Since we're only concerned about the base stations in Shanghai, 
let's begin by finding the global outliers in our our Shanghai 
Telecom dataset, ``cdr``. First, we load the "World" dataset, a 
predefined dataset in VerticaPy, and then plot on a map of China 
to see if any points fall outside of Shanghai. We can then drop 
these outliers using the z-score method.

.. ipython:: python
    :okwarning:

    # Setting up the plotting Library to Matplotlib
    vp.set_option("plotting_lib", "matplotlib")

    # Outliers
    world = load_world()
    china = world[world["country"] == "China"]
    ax = china["geometry"].geo_plot(
        color = "white",
        edgecolor = "black",
    )
    cdr.groupby(["longitude", "latitude"]).scatter(
        ["longitude", "latitude"],
        ax = ax,
    )
    @savefig examples_base_station_shanghai_outliers.png
    ax.set_title("Shanghai's Base Stations with Outliers")

.. ipython:: python
    :okwarning:

    # Dropping Outliers
    cdr["longitude"].drop_outliers(threshold = 2.0);
    cdr["latitude"].drop_outliers(threshold = 2.0);

    # Without Outliers
    ax = china["geometry"].geo_plot(
        color = "white",
        edgecolor = "black",
    )
    cdr.groupby(["longitude", "latitude"]).scatter(
        ["longitude", "latitude"],
        ax = ax,
    )
    @savefig examples_base_station_shanghai_outliers_without.png
    ax.set_title("Shanghai's Base Stations without Outliers")

As we can see from the second plot, we've discarded the base stations outside of Shanghai.

Understanding Shanghai's Districts
+++++++++++++++++++++++++++++++++++

Let's check the districts on the map. The Huangpu district is 
the urban ``hub`` of sorts and the most central of Shanghai's 
districts, so we'll pay it some special attention. We'll be 
referring to the Huangpu district as Shanghai's "downtown" 
in this study.

.. ipython:: python

    ax = shanghai_districts["geometry"].geo_plot(
        column = "district_level",
        edgecolor='white',
    )

    # Finding Centroids
    centroids = shanghai_districts.select(
        [
            "name", 
            "ST_X(ST_CENTROID(geometry))", 
            "ST_Y(ST_CENTROID(geometry))",
        ],
    ).to_list()

    # Plotting the suburb names
    for c in centroids[7:]:
        ax.text(c[1], c[2], c[0], va="center", ha="center")
    ax.set_title("Shanghai's Districts")
    @savefig examples_base_station_shanghai_downtown_center.png
    ax.text(121.43, 31.25, "Downtown", va="center", ha="center")

.. ipython:: python

    ax2 = shanghai_districts.search("number <= 7")["geometry"].geo_plot(
        color="#CCCCCC",
        edgecolor="white",
    )

    # Plotting the downtown names
    for c in centroids[:7]:
        ax2.text(c[1], c[2], c[0], va="center", ha="center")
    ax2.set_title("Shanghai's Downtown")
    @savefig examples_base_station_shanghai_downtown_center_grey.png
    ax2

Districts' Activity
++++++++++++++++++++

Let's examine the network activity of each of our districts. To do this, we need VerticaPy's Geospatial, which leverages Vertica geospatial functions. We begin by creating an index for the districts and then find the intersection between connections and districts. We'll visualize this with a bar chart race, which reflects each district's cumulative activity duration through time.

.. code-block:: python

    from verticapy.sql.geo import create_index, intersect

    # Finding the intersections between each ping and each district
    create_index(
        shanghai_districts, 
        gid = "number", 
        g = "geometry", 
        index = "shanghai_districts",
        overwrite = True,
    )
    intersect_districts_cdr = intersect(
        cdr,
        index = "shanghai_districts",
        gid = "row_id",
        x = "longitude",
        y = "latitude",
    )
    intersect_districts_cdr.head(100)

.. ipython:: python
    :suppress:

    from verticapy.sql.geo import create_index, intersect

    # Finding the intersections between each ping and each district
    create_index(
        shanghai_districts, 
        gid = "number", 
        g = "geometry", 
        index = "shanghai_districts",
        overwrite = True,
    )
    intersect_districts_cdr = intersect(
        cdr,
        index = "shanghai_districts",
        gid = "row_id",
        x = "longitude",
        y = "latitude",
    )
    res = intersect_districts_cdr.head(100)
    html_file = open("SPHINX_DIRECTORY/figures/examples_base_station_shanghai_district_activity.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. ipython:: python

    # Total Connection Duration
    cdr["total_duration"] = (cdr["end_time"] - cdr["start_time"]) / fun.interval("1 minute");

    # Features Engineering
    cdr["date"] = "DATE(start_time)";

    # Total Duration per Connection per district per day
    activity = intersect_districts_cdr.join(
        cdr,
        on = {"point_id": "row_id"},
        how = "left",
        expr1 = ["polygon_gid",],
        expr2 = ["start_time", "total_duration", "date",]
    ).groupby(
        [
            "polygon_gid",
            "date",
        ],
        "SUM(total_duration) AS total_duration",
    ).join(
        shanghai_districts,
        on = {"polygon_gid": "number"},
        how = "left",
        expr1 = [
            "date",
            "total_duration / area AS total_duration_km2",
        ],
        expr2 = ["name", "district_level"],
    );

    # Cumulative Duration per Connection per district
    activity.cumsum(
        "total_duration_km2", 
        by = ["name"],
        order_by = ["date"],
        name = "activity",
    );
    activity["activity"] = fun.round(activity["activity"], 2);

    # Formatting the date
    def date_f(x):
        return x.strftime("%b %d")

.. code-block:: python

    # Drawing the activity Bar Race
    activity.animated_bar(
        "date", 
        ["name", "activity"],
        by = "district_level",
        start_date = "2014-07-01", 
        end_date = "2014-08-01",
        limit_over = 13,
        date_f = date_f,
    )

.. ipython:: python
    :suppress:

    # Drawing the activity Bar Race
    fig = activity.animated_bar(
        "date", 
        ["name", "activity"],
        by = "district_level",
        start_date = "2014-07-01", 
        end_date = "2014-08-01",
        limit_over = 13,
        date_f = date_f,
    )
    with open("figures/examples_base_station_animated_bar_activity.html", "w") as file:
        file.write(fig.__html__())

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_base_station_animated_bar_activity.html

Like you might expect, Shanghai's downtown is the most active one for the selected period. 

Data Preparation
-----------------

Finding Clusters of Base Stations
++++++++++++++++++++++++++++++++++

We create virtual base stations by grouping existing base stations in 100 clusters. Clustering is performed using :py:mod:`~verticapy.machine_learning.vertica.cluster.KMeans` clustering on Euclidean coordinates of the base stations. Each cluster represents a wider coverage of connections.

.. ipython:: python

    from verticapy.sql.geo import coordinate_converter

    # Creating the Base Station Dataset
    bs = cdr.groupby(
        ["longitude", "latitude"],
        [
            "COUNT(DISTINCT user_id) AS total_distinct_users",
            "AVG((end_time - start_time) / '1 minute') AS avg_connection_duration",
            "SUM((end_time - start_time) / '1 minute') AS total_connection_duration",
            "COUNT(*) AS connection_number",
        ],
    );
    # Converting longitude, latitude to x, y
    bs_xy = coordinate_converter(bs, "longitude", "latitude");

    # Using Clustering
    from verticapy.machine_learning.vertica import KMeans

    model = KMeans("shanghai.kmeans_bs", n_cluster = 100)
    model.fit(bs_xy, ["longitude", "latitude"])

.. code-block:: python

    model.predict(bs_xy, name = "cluster")

.. ipython:: python
    :suppress:

    res = model.predict(bs_xy, name = "cluster")
    html_file = open("SPHINX_DIRECTORY/figures/examples_base_station_model_rediction.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()


.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_base_station_model_rediction.html


.. ipython:: python

    bs = coordinate_converter(bs_xy, "longitude", "latitude", reverse = True);
    vp.drop("shanghai.bs", method = "table");
    bs.to_db("shanghai.bs",relation_type = "table",inplace = True,);

.. code-block:: python

    model.plot_voronoi(plot_crosses = False)

.. ipython:: python
    :suppress:
    :okwarning:

    vp.set_option("plotting_lib","plotly")
    fig = model.plot_voronoi(plot_crosses = False)
    fig.write_html("SPHINX_DIRECTORY/figures/examples_base_station_voronoi_plotly.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_base_station_voronoi_plotly.html


In this figure, each Voronoi cell represents a base station cluster.

Identifying Base Station Workloads
+++++++++++++++++++++++++++++++++++

Workload is defined as the number of connections per time interval. To find the workloads of the base stations and base station clusters, we'll filter the data to get connections registered in a time frame of one week and then use time-series slicing to get records for every minute per user. 

.. ipython:: python

    # Filtering to get the first week of July
    cdr_sample = cdr.search(
        cdr["start_time"]._between(
            "2014-07-01 00:00:00", 
            "2014-07-08 00:00:00",
        ),
    );
    # Merging Start Time and End Time to use Time Series Slicing
    cdr_sample = cdr_sample.select(
        [
            "row_id", 
            "user_id", 
            "start_time AS datetime", 
            "latitude", 
            "longitude",
        ]
    ).append(
        cdr_sample.select(
            [
                "row_id", 
                "user_id", 
                "end_time AS datetime", 
                "latitude", 
                "longitude",
            ],
        ),
    );
    # Slicing the datetime to get one record per mn per user
    cdr_sample = cdr_sample.asfreq(
        ts = "datetime",
        rule = "1 minute",
        by = [
            "user_id", 
            "latitude", 
            "longitude", 
            "row_id",
        ],
    );




.. code-block:: python

    cdr_sample.head(100)

.. ipython:: python
    :suppress:

    res = cdr_sample.head(100)
    html_file = open("SPHINX_DIRECTORY/figures/examples_base_station_cdr_sample_head.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()


.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_base_station_cdr_sample_head.html



.. ipython:: python

    # Switching back to matplotlib
    vp.set_option("plotting_lib","matplotlib") 
    vp.drop("shanghai.bs_workload");
    bs_workload = cdr_sample.groupby(
        [
            "datetime",
            "latitude",
            "longitude",
        ], 
        ["COUNT(DISTINCT user_id) AS workload"]
    ).to_db(
        "shanghai.bs_workload",
        relation_type = "table",
        inplace = True,
    );
    ax = shanghai_districts["geometry"].geo_plot(
        color = "white",
        edgecolor = "black",
    );

.. code-block:: python

    bs_workload.animated_scatter(
        "datetime",
        ["longitude", "latitude",],
        start_date = "2014-07-01 15:00:00",
        end_date = "2014-07-01 20:00:00",
        limit_over = 10000,                                                                    
        fixed_xy_lim = True,
        date_in_title = True,
        ax = ax,
    )

.. ipython:: python
    :suppress:
    :okwarning:

    # Drawing the activity Bar Race
    fig = bs_workload.animated_scatter(
        "datetime",
        ["longitude", "latitude",],
        start_date = "2014-07-01 15:00:00",
        end_date = "2014-07-01 20:00:00",
        limit_over = 10000,                                                                    
        fixed_xy_lim = True,
        date_in_title = True,
        ax = ax,
    )
    with open("figures/examples_base_station_animated_scatter_longi.html", "w") as file:
        file.write(fig.__html__())

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_base_station_animated_scatter_longi.html

From the above animation, we can see that we'll typically have unconnected base stations and that the most overloaded base stations are located around the downtown.

Let's define the base station workload as the number of connections in one time point, that is, the 90-th percentile of the interval. 

We can then calculate the workload for each cluster.

.. ipython:: python

    # Base Station Workload 90%
    bs_workload_90 = bs_workload.groupby(
        ["latitude", "longitude"],
        "APPROXIMATE_PERCENTILE(workload USING PARAMETERS percentile=0.90) AS workload",
    );
    bs_workload_90.astype(
        {
            "longitude": "float",
            "latitude": "float",
            "workload": "int",
        },
    );
    vp.drop("shanghai.bs_workload_90", method = "table")
    bs_workload_90.to_db(
        "shanghai.bs_workload_90",
        relation_type = "table",
        inplace = True,
    );

.. ipython:: python
    :suppress:

    res = bs_workload_90
    html_file = open("SPHINX_DIRECTORY/figures/examples_base_station_bs_workload_90.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_base_station_bs_workload_90.html

.. ipython:: python

    # Clusters Workload 90%
    cworkload = coordinate_converter(
        bs_workload, 
        "longitude", 
        "latitude",
    );
    model.predict(cworkload, name = "cluster");
    cworkload = coordinate_converter(
        cworkload, 
        "longitude", 
        "latitude",
        reverse = True)
    cworkload_bs = cworkload.groupby(
        ["datetime", "cluster"],
        ["SUM(workload) AS workload",],
    ).groupby(
        "cluster",
        ["APPROXIMATE_PERCENTILE(workload USING PARAMETERS percentile=0.90) AS workload",],
    );
    cworkload_bs = cworkload_bs.join(
        bs.groupby("cluster", "COUNT(*) AS cnt"),
        how = "left",
        on = {"cluster": "cluster"},
        expr2 = ["cnt AS total_bs"],
    );
    cworkload_bs["workload_per_bs"] = cworkload_bs["workload"] / cworkload_bs["total_bs"]
    cworkload_bs.sort({"workload_per_bs": "desc"});

.. ipython:: python
    :suppress:

    res = cworkload_bs
    html_file = open("SPHINX_DIRECTORY/figures/examples_base_station_cworkload_bs.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_base_station_cworkload_bs.html

Data Modeling
--------------

Finding New Base Stations positions
++++++++++++++++++++++++++++++++++++

Let's find a suitable number of clusters using elbow curve.

.. ipython:: python

    # Finding a suitable number of base stations for the region
    most_active_cluster = cworkload_bs.search(
        "total_bs > 20"
    ).sort(
        {"workload_per_bs": "desc"},
    )["cluster"][0]
    bs_most_active_cluster = bs.search(bs["cluster"] == most_active_cluster)
    bs_most_active_cluster.astype(
        {
            "longitude": "float",
            "latitude": "float",
        },
    );
    bs_most_active_cluster = bs_most_active_cluster.join(
        bs_workload_90,
        how = "left",
        on_interpolate = {
            "longitude": "longitude",
        },
        expr2 = "workload",
    );
    bs_weight = bs_most_active_cluster.add_duplicates(weight = "workload")
    bs_xy = coordinate_converter(bs_weight, "longitude", "latitude")

.. code-block:: python

    from verticapy.machine_learning.model_selection import elbow

    # Switching back to Plotly
    vp.set_option("plotting_lib", "plotly")

    elbow(bs_xy, ["longitude", "latitude"])

.. ipython:: python
    :suppress:
    :okwarning:

    from verticapy.machine_learning.model_selection import elbow
    vp.set_option("plotting_lib", "plotly")
    fig = elbow(bs_xy, ["longitude", "latitude"])
    fig.write_html("SPHINX_DIRECTORY/figures/examples_base_station_elbow_longi_lati.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_base_station_elbow_longi_lati.html

The :py:func:`~verticapy.machine_learning.model_selection.elbow` curve seems to indicate that 4 would be a good number of clusters, so let's try k = 4 and view the weighted :py:mod:`~verticapy.machine_learning.vertica.cluster.KMeans` algorithm's suggested positions for new base stations based on the centers of the clusters.

.. ipython:: python
    :okwarning:

    # Switching back to Matplotlib
    vp.set_option("plotting_lib", "matplotlib")

    # Creating the model
    from verticapy.machine_learning.vertica import KMeans

    model = KMeans("shanghai.new_bs_kmeans", n_cluster = 4);
    model.fit(bs_xy, ["longitude", "latitude"])
    model.predict(bs_xy, name = "new_bs_center");
    bs_new = coordinate_converter(bs_xy, "longitude", "latitude", reverse = True);

    # Drawing the map
    ax = shanghai_districts["geometry"].geo_plot(
        color = "white",
        edgecolor = "black",
    );
    bs_new.scatter(
        [
            "longitude", 
            "latitude",
        ],
        by = "new_bs_center",
        ax = ax,
    );
    coordinate_converter(
        vp.vDataFrame(
            model.clusters_, 
            usecols = ["longitude", "latitude"],
        ), 
        "longitude", 
        "latitude",
        reverse = True
    ).scatter(
        [
            "longitude", 
            "latitude",
        ],
        marker = "x",
        color = "r",
        s = 220,
        linewidths = 3,
        ax = ax,
    );
    ax.set_xlim(bs_most_active_cluster["longitude"].min() - 0.02, bs_most_active_cluster["longitude"].max() + 0.02,)
    ax.set_ylim(bs_most_active_cluster["latitude"].min() - 0.02, bs_most_active_cluster["latitude"].max() + 0.02,)

    import matplotlib.pyplot as plt

    text = bs_most_active_cluster[["longitude", "latitude", "workload"]].to_list()
    for t in text:
        ax.text(t[0] + 0.001, t[1], str(t[2]),)
    @savefig examples_base_station_possible_new_base_stations.png
    ax.set_title("Possible New Base Stations")

Predicting Base Station Workload
+++++++++++++++++++++++++++++++++

With the predictive power of AutoML, we can predict the workload of the base stations. :py:mod:`~verticapy.machine_learning.vertica.automl.AutoML` is a powerful technique that tests multiple models to maximize the input score.

The features used to train our model will be longitude, latitude, total number of distinct users, average duration of the connections, total duration of connections, total number of connections, the cluster they belong to, total number of base stations in the cluster, and the workload of the clusters.

.. ipython:: python

    vp.drop("shanghai.bs_metrics", method = "table")
    bs_metrics = bs.join(
        cworkload_bs,
        how = "left",
        on = {"cluster": "cluster"},
        expr2 = [
            "total_bs AS cluster_total_bs",
            "workload AS cluster_workload",
        ],
    );
    bs_metrics.to_db(
        "shanghai.bs_metrics",
        relation_type = "table",
        inplace = True,
    );

.. ipython:: python
    :suppress:

    res = bs_metrics
    html_file = open("SPHINX_DIRECTORY/figures/examples_base_station_bs_metrics.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()


.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_base_station_bs_metrics.html


.. ipython:: python
    :okwarning:

    from verticapy.machine_learning.vertica.automl import AutoML

    model = AutoML(
        "shanghai.automl",
        estimator = "fast",
        lmax = 3, 
        stepwise_direction = "backward", 
        stepwise_x_order = "spearman", 
        preprocess_dict = {"identify_ts": False},
    )
    model.fit(
        bs_metrics, 
        [
            "total_distinct_users", 
            "avg_connection_duration", 
            "total_connection_duration", 
            "connection_number",
            "cluster_total_bs",
        ], 
        "cluster_workload",
    )


.. code-block:: python

    # Switching back to Plotly
    vp.set_option("plotting_lib", "plotly")

    model.plot()


.. ipython:: python
    :suppress:
    :okwarning:

    vp.set_option("plotting_lib","plotly")
    fig = model.plot()
    fig.write_html("SPHINX_DIRECTORY/figures/examples_base_station_auto_ml_plot.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_base_station_auto_ml_plot.html

Conclusion
-----------

We've solved our problem in a Pandas-like way, all without loading data into memory!