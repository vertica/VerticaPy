.. _examples.business.movies:

Movies Scoring and Clustering 
==============================

This example uses the ``filmtv_movies`` dataset to evaluate the quality of the movies and create clusters of similar movies. 
You can download the Jupyter notebook `here <https://github.com/vertica/VerticaPy/blob/master/examples/business/movies/movies.ipynb>`_.

The columns provided include:

- **year:** Movie's release year.
- **filmtv_id:** Movie ID.
- **title:** Movie title.
- **genre:** Movie genre.
- **country:** Movie's country of origin.
- **description:** Movie description.
- **notes:** Information about the movie.
- **duration:** Movie duration.
- **votes:** Number of votes.
- **avg_vote:** Average score.
- **director:** Movie director.
- **actors:** Actors in the movie.


We will follow the data science cycle (Data Exploration - Data Preparation - Data Modeling - Model Evaluation - Model Deployment) to solve this problem.

Initialization
----------------

This example uses the following version of VerticaPy:

.. ipython:: python
    
    import verticapy as vp
    
    vp.__version__

Connect to Vertica. This example uses an existing connection called "VerticaDSN." 
For details on how to create a connection, see the :ref:`connection` tutorial.
You can skip the below cell if you already have an established connection.

.. code-block:: python
    
    vp.connect("VerticaDSN")

Let's  create a new schema and assign the data to a :py:mod:`~verticapy.vDataFrame` object.

.. code-block:: ipython

    vp.drop("movies", method="schema")
    vp.create_schema("movies")
    filmtv_movies = vp.read_csv("movies.csv", schema = "movies")
    filmtv_movies.head(5)

Let's take a look at the first few entries in the dataset.

.. ipython:: python
    :suppress:

    vp.drop("movies", method="schema")
    vp.create_schema("movies")
    filmtv_movies = vp.read_csv("SPHINX_DIRECTORY/source/_static/website/examples/data/movies/movies.csv", schema = "movies")
    res = filmtv_movies.head(5)
    html_file = open("SPHINX_DIRECTORY/figures/examples_movies_table.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_movies_table.html

Data Exploration and Preparation
---------------------------------

One of the biggest challenges for any streaming platform is to find a good catalog of movies.

First, let's explore the dataset.

.. code-block:: python

    filmtv_movies.describe(method = "categorical", unique = True)

.. ipython:: python
    :suppress:

    res = filmtv_movies.describe(method = "categorical", unique = True)
    html_file = open("SPHINX_DIRECTORY/figures/examples_movies_describe_cat.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_movies_describe_cat.html

We can drop the 'description' and 'notes' columns since these fields are empty for most of our dataset.

.. code-block:: python

    filmtv_movies.drop(["description", "notes"])

.. ipython:: python
    :suppress:

    filmtv_movies.drop(["description", "notes"])
    res = filmtv_movies
    html_file = open("SPHINX_DIRECTORY/figures/examples_movies_drop.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_movies_drop.html

We have access to more than ``50000`` movies in ``27`` different genres. Let's organize our list by their average rating.

.. code-block:: python

    filmtv_movies.sort({"avg_vote" : "desc"})

.. ipython:: python
    :suppress:

    filmtv_movies.sort({"avg_vote" : "desc"})
    res = filmtv_movies.sort({"avg_vote" : "desc"})
    html_file = open("SPHINX_DIRECTORY/figures/examples_movies_avg_vote_sort.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_movies_avg_vote_sort.html

Since we want properly averaged scores, let's just consider the top 10 movies that have at least 10 votes.

.. code-block:: python

    filmtv_movies.search(
        conditions = [filmtv_movies["votes"] > 10], 
        order_by = {"avg_vote" : "desc" },
    )

.. ipython:: python
    :suppress:

    res = filmtv_movies.search(
        conditions = [filmtv_movies["votes"] > 10], 
        order_by = {"avg_vote" : "desc" },
    )
    html_file = open("SPHINX_DIRECTORY/figures/examples_movies_search_votes.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_movies_search_votes.html

We can see classic movies like ``The Godfather`` and ``Greed``. Let's smooth the avg_vote using a linear regression to make it more representative.

To create our model we could use the votes, the category, the duration, etc. but let's go with the director and main actors. 

We can extract the five main actors for each movie with regular expressions.

.. code-block:: python

    for i in range(1, 5):
        filmtv_movies2 = vp.read_csv("movies.csv")
        filmtv_movies2.regexp(
            column = "actors",
            method = "substr",
            pattern = '[^,]+',
            occurrence = i,
            name = "actor",
        )
        if i == 1:
            filmtv_movies = filmtv_movies2.copy()
        else:
            filmtv_movies = filmtv_movies.append(filmtv_movies2)
    filmtv_movies["actor"].describe()

.. ipython:: python
    :suppress:

    for i in range(1, 5):
        filmtv_movies2 = vp.read_csv("SPHINX_DIRECTORY/source/_static/website/examples/data/movies/movies.csv")
        filmtv_movies2.regexp(
            column = "actors",
            method = "substr",
            pattern = '[^,]+',
            occurrence = i,
            name = "actor",
        )
        if i == 1:
            filmtv_movies = filmtv_movies2.copy()
        else:
            filmtv_movies = filmtv_movies.append(filmtv_movies2)
    res = filmtv_movies["actor"].describe()
    html_file = open("SPHINX_DIRECTORY/figures/examples_movies_describe_actors.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examplexamples_movies_describe_actorses_movies_search_votes.html

By aggregating the data, we can find the number of actors and the number of votes by actor. 
We can then normalize the data using the min-max method and quantify the notoriety of the actors.

.. code-block:: python

    import verticapy.sql.functions as fun

    actors_stats = filmtv_movies.groupby(
        columns = ["actor"], 
        expr = [
            fun.sum(filmtv_movies["votes"])._as("notoriety_actors"),
            fun.count(filmtv_movies["actors"])._as("castings_actors"),
        ],
    )
    actors_stats["actor"].dropna()
    actors_stats["notoriety_actors"].normalize(method = "minmax")

.. ipython:: python
    :suppress:

    import verticapy.sql.functions as fun

    actors_stats = filmtv_movies.groupby(
        columns = ["actor"], 
        expr = [
            fun.sum(filmtv_movies["votes"])._as("notoriety_actors"),
            fun.count(filmtv_movies["actors"])._as("castings_actors"),
        ]
    )
    actors_stats["actor"].dropna()
    res = actors_stats["notoriety_actors"].normalize(method = "minmax")
    html_file = open("SPHINX_DIRECTORY/figures/examples_movies_normalize_actors.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_movies_normalize_actors.html

Let's look at the top ten actors by notoriety.

.. code-block:: python

    actors_stats.search(
        order_by = {
            "notoriety_actors" : "desc", 
            "castings_actors" : "desc",
        },
    ).head(10)

.. ipython:: python
    :suppress:

    res = actors_stats.search(
        order_by = {
            "notoriety_actors" : "desc", 
            "castings_actors" : "desc",
        },
    ).head(10)
    html_file = open("SPHINX_DIRECTORY/figures/examples_movies_actors_notr_head.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_movies_actors_notr_head.html

As expected, we get a list of very popular actors like Robert De Niro, Morgan Freeman, and Clint Eastwood.

Let's do the same for the directors.

.. code-block:: python

    director_stats = filmtv_movies.groupby(
        columns = ["director"], 
        expr = [
            fun.sum(filmtv_movies["votes"])._as("notoriety_director"),
            fun.count(filmtv_movies["director"])._as("castings_director"),
        ],
    )
    director_stats["notoriety_director"].normalize(method = "minmax")

.. ipython:: python
    :suppress:


    director_stats = filmtv_movies.groupby(
        columns = ["director"], 
        expr = [
            fun.sum(filmtv_movies["votes"])._as("notoriety_director"),
            fun.count(filmtv_movies["director"])._as("castings_director"),
        ],
    )
    res = director_stats["notoriety_director"].normalize(method = "minmax")
    html_file = open("SPHINX_DIRECTORY/figures/examples_movies_notoriety_director.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_movies_notoriety_director.html

Now let's look at the top 10 movie directors.

.. code-block:: python

    director_stats.search(
        order_by = {
            "notoriety_director" : "desc", 
            "castings_director" : "desc",
        },
    ).head(10)

.. ipython:: python
    :suppress:

    res = director_stats.search(
        order_by = {
            "notoriety_director" : "desc", 
            "castings_director" : "desc",
        },
    ).head(10)
    html_file = open("SPHINX_DIRECTORY/figures/examples_movies_notoriety_director_head_order.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_movies_notoriety_director_head_order.html

Again, we get a list of popular directors like Steven Spielberg, Woody Allen, and Clint Eastwood.

Let's join our notoriety metrics for actors and directors with the main dataset.

.. ipython:: python

    filmtv_movies_director = filmtv_movies.join(
        director_stats,
        on = {"director": "director"},
        how = "left",
        expr1 = ["*"],
        expr2 = [
            "notoriety_director", 
            "castings_director",
        ],
    )
    filmtv_movies_director_actors = filmtv_movies_director.join(
        actors_stats,
        on = {"actor": "actor"},
        how = "left",
        expr1 = ["*"],
        expr2 = [
            "notoriety_actors",
            "castings_actors",
        ],
    )

As we did many operation, it can be nice to save the :py:mod:`~verticapy.vDataFrame` as a table in the Vertica database.

.. code-block:: python

    vp.drop("filmtv_movies_director_actors", method = "table")
    filmtv_movies_director_actors.to_db(
        name = "filmtv_movies_director_actors", 
        relation_type = "table",
        inplace = True,
    )

.. ipython:: python
    :suppress:

    vp.drop("filmtv_movies_director_actors", method = "table")
    filmtv_movies_director_actors.to_db(
        name = "filmtv_movies_director_actors", 
        relation_type = "table",
        inplace = True,
    )

We can aggregate the data to get metrics on each movie.

.. ipython:: python

    filmtv_movies_complete = filmtv_movies_director_actors.groupby(
        columns = [
            "filmtv_id", 
            "title",
            "year",
            "genre",
            "country",
            "avg_vote",
            "votes", 
            "duration", 
            "director", 
            "notoriety_director",
            "castings_director",
        ],
        expr = [
            fun.sum(filmtv_movies_director_actors["notoriety_actors"])._as("notoriety_actors"),
            fun.sum(filmtv_movies_director_actors["castings_actors"])._as("castings_actors"),
        ],
    )

Let's compute some statistics on our dataset.

.. code-block:: python

    filmtv_movies_complete.describe(method = "all")

.. ipython:: python
    :suppress:

    res = filmtv_movies_complete.describe(method = "all")
    html_file = open("SPHINX_DIRECTORY/figures/examples_movies_filmtv_describe.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_movies_filmtv_describe.html

We can use the movie's release year to get create three categories.

.. code-block:: python

    filmtv_movies_complete.case_when(
        "period",
        filmtv_movies_complete["year"] < 1990, "Old",
        filmtv_movies_complete["year"] >= 2000, "Recent", "90s",
    ) 

.. ipython:: python
    :suppress:

    res = filmtv_movies_complete.case_when(
        "period",
        filmtv_movies_complete["year"] < 1990, "Old",
        filmtv_movies_complete["year"] >= 2000, "Recent", "90s",
    ) 
    html_file = open("SPHINX_DIRECTORY/figures/examples_movies_filmtv_casewhen.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_movies_filmtv_casewhen.html

Now, let's look at the countries that made the most movies.

.. code-block:: python

    filmtv_movies_complete.groupby(
        columns = ["country"], 
        expr = ["COUNT(*)"]
    ).sort({"count" : "desc"}).head(10)

.. ipython:: python
    :suppress:

    res = filmtv_movies_complete.groupby(
        columns = ["country"], 
        expr = ["COUNT(*)"],
    ).sort({"count" : "desc"}).head(10)
    html_file = open("SPHINX_DIRECTORY/figures/examples_movies_filmtv_country_head.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_movies_filmtv_country_head.html

We can use this variable to create language groups.

.. ipython:: python

    # Language Discretization
    Arabic_Middle_Est = [
        "Arab", "Iran", "Turkey", "Egypt", "Tunisia",
        "Lebanon", "Palestine", "Morocco", "Iraq",
        "Sudan", "Algeria", "Yemen", "Afghanistan",
        "Azerbaijan", "Kazakhstan", "Kyrgyzstan",
        "Kurdistan", "Syria", "Uzbekistan",
    ]
    Chinese_Japan_Asian = [
        "Japan", "Hong Kong", "China", "South Korea", 
        "Thailand", "Philippines", "Taiwan", "Indonesia",
        "Singapore", "Malaysia", "Vietnam", "Laos", "Cambodia",
        "Bhutan",
    ]
    Indian = ["India", "Pakistan", "Nepal", "Sri Lanka", "Bangladesh",]
    Hebrew = ["Israel",]
    Spanish_Portuguese = [
        "Spain", "Portugal", "Mexico", "Brasil", "Chile",
        "Argentina", "Colombia", "Cuba", "Venezuela", "Peru",
        "Uruguay", "Dominican Republic", "Ecuador", "Guatemala",
        "Costa Rica", "Paraguay", "Bolivia",
    ]
    English = [
        "United States", "England", "Great Britain", "Ireland",
        "Australia", "New Zealand", "South Africa",
    ]

    French = ["France", "Canada", "Belgium", "Switzerland", "Luxembourg",]
    Italian = ["Italy",]
    German_North_Europe = [
        "German", "Austria", "Holland", "Netherlands", "Denmark",
        "Norway", "Iceland", "Finland", "Sweden", "Greenland",
    ]

    Russian_Est_Europe = ["Russia", "Soviet Union", "Yugoslavia", "Czechoslovakia", "Poland", "Bulgaria", "Croatia", "Czech Republic", "Serbia", "Ukraine", "Slovenia", "Lithuania", "Latvia", "Estonia", "Bosnia and Herzegovina", "Georgia"]

    Grec_Balkan = [
        "Greece", "Macedonia", "Cyprus", "Romania", "Armenia", "Hungary",
        "Albania", "Malta",
    ]

.. code-block:: python

    # Creation of the new feature
    filmtv_movies_complete.case_when('language_area', 
            vp.StringSQL("REGEXP_LIKE(Country, '{}')".format("|".join(Arabic_Middle_Est))), 'Arabic_Middle_Est',
            vp.StringSQL("REGEXP_LIKE(Country, '{}')".format("|".join(Chinese_Japan_Asian))), 'Chinese_Japan_Asian', 
            vp.StringSQL("REGEXP_LIKE(Country, '{}')".format("|".join(Indian))), 'Indian', 
            vp.StringSQL("REGEXP_LIKE(Country, '{}')".format("|".join(Hebrew))), 'Hebrew', 
            vp.StringSQL("REGEXP_LIKE(Country, '{}')".format("|".join(Spanish_Portuguese))), 'Spanish_Portuguese', 
            vp.StringSQL("REGEXP_LIKE(Country, '{}')".format("|".join(English))), 'English',
            vp.StringSQL("REGEXP_LIKE(Country, '{}')".format("|".join(French))), 'French',
            vp.StringSQL("REGEXP_LIKE(Country, '{}')".format("|".join(Italian))), 'Italian',
            vp.StringSQL("REGEXP_LIKE(Country, '{}')".format("|".join(German_North_Europe))), 'German_North_Europe',
            vp.StringSQL("REGEXP_LIKE(Country, '{}')".format("|".join(Russian_Est_Europe))), 'Russian_Est_Europe',
            vp.StringSQL("REGEXP_LIKE(Country, '{}')".format("|".join(Grec_Balkan))), 'Grec_Balkan', 
            'Others') 

.. ipython:: python
    :suppress:

    res = filmtv_movies_complete.case_when('language_area', 
            vp.StringSQL("REGEXP_LIKE(Country, '{}')".format("|".join(Arabic_Middle_Est))), 'Arabic_Middle_Est',
            vp.StringSQL("REGEXP_LIKE(Country, '{}')".format("|".join(Chinese_Japan_Asian))), 'Chinese_Japan_Asian', 
            vp.StringSQL("REGEXP_LIKE(Country, '{}')".format("|".join(Indian))), 'Indian', 
            vp.StringSQL("REGEXP_LIKE(Country, '{}')".format("|".join(Hebrew))), 'Hebrew', 
            vp.StringSQL("REGEXP_LIKE(Country, '{}')".format("|".join(Spanish_Portuguese))), 'Spanish_Portuguese', 
            vp.StringSQL("REGEXP_LIKE(Country, '{}')".format("|".join(English))), 'English',
            vp.StringSQL("REGEXP_LIKE(Country, '{}')".format("|".join(French))), 'French',
            vp.StringSQL("REGEXP_LIKE(Country, '{}')".format("|".join(Italian))), 'Italian',
            vp.StringSQL("REGEXP_LIKE(Country, '{}')".format("|".join(German_North_Europe))), 'German_North_Europe',
            vp.StringSQL("REGEXP_LIKE(Country, '{}')".format("|".join(Russian_Est_Europe))), 'Russian_Est_Europe',
            vp.StringSQL("REGEXP_LIKE(Country, '{}')".format("|".join(Grec_Balkan))), 'Grec_Balkan', 
            'Others') 
    html_file = open("SPHINX_DIRECTORY/figures/examples_movies_filmtv_complete_language.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_movies_filmtv_complete_language.html

We can do the same for the genres.

.. code-block:: python

    filmtv_movies_complete.case_when(
            'Category', 
            vp.StringSQL("REGEXP_LIKE(Genre, 'Drama|Noir')"), 'Drama', 
            vp.StringSQL("REGEXP_LIKE(Genre, 'Comedy|Grotesque')"), 'Comedy', 
            vp.StringSQL("REGEXP_LIKE(Genre, 'Fantasy|Super-hero')"), 'Fantasy', 
            vp.StringSQL("REGEXP_LIKE(Genre, 'Romantic|Sperimental|Mélo')"), 'Romantic', 
            vp.StringSQL("REGEXP_LIKE(Genre, 'Thriller|Crime|Gangster')"), 'Thriller', 
            vp.StringSQL("REGEXP_LIKE(Genre, 'Action|Western|War|Spy')"), 'Action', 
            vp.StringSQL("REGEXP_LIKE(Genre, 'Adventure')"), 'Adventure', 
            vp.StringSQL("REGEXP_LIKE(Genre, 'Animation')"), 'Animation', 
            vp.StringSQL("REGEXP_LIKE(Genre, 'Horror')"), 'Horror', 
            'Others'
    ) 

.. ipython:: python
    :suppress:

    res = filmtv_movies_complete.case_when(
         'Category', 
         vp.StringSQL("REGEXP_LIKE(Genre, 'Drama|Noir')"), 'Drama', 
         vp.StringSQL("REGEXP_LIKE(Genre, 'Comedy|Grotesque')"), 'Comedy', 
         vp.StringSQL("REGEXP_LIKE(Genre, 'Fantasy|Super-hero')"), 'Fantasy', 
         vp.StringSQL("REGEXP_LIKE(Genre, 'Romantic|Sperimental|Mélo')"), 'Romantic', 
         vp.StringSQL("REGEXP_LIKE(Genre, 'Thriller|Crime|Gangster')"), 'Thriller', 
         vp.StringSQL("REGEXP_LIKE(Genre, 'Action|Western|War|Spy')"), 'Action', 
         vp.StringSQL("REGEXP_LIKE(Genre, 'Adventure')"), 'Adventure', 
         vp.StringSQL("REGEXP_LIKE(Genre, 'Animation')"), 'Animation', 
         vp.StringSQL("REGEXP_LIKE(Genre, 'Horror')"), 'Horror', 
         'Others'
    ) 
    html_file = open("SPHINX_DIRECTORY/figures/examples_movies_filmtv_complete_category_genre.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_movies_filmtv_complete_category_genre.html

Since we're more concerned with the ``Category`` at this point, we can drop ``genre``.

.. code-block:: python

    filmtv_movies_complete.drop(columns = ["genre"])

.. ipython:: python
    :suppress:

    filmtv_movies_complete.drop(columns = ["genre"])

Let's look at the missing values.

.. code-block:: python

    filmtv_movies_complete.count_percent()

.. ipython:: python
    :suppress:

    res = filmtv_movies_complete.count_percent()
    html_file = open("SPHINX_DIRECTORY/figures/examples_movies_filmtv_complete_missing_vals.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_movies_filmtv_complete_missing_vals.html

Let's impute the missing values for ``notoriety_actors`` and ``castings_actors`` using different techniques.

We can then drop the few remaining missing values.

.. code-block:: python

    filmtv_movies_complete["notoriety_actors"].fillna(
        method = "median",
        by = [
            "director",
            "Category",
        ],
    )
    filmtv_movies_complete["castings_actors"].fillna(
        method = "median",
        by = [
            "director",
            "Category",
        ],
    )
    filmtv_movies_complete.dropna()

.. ipython:: python
    :suppress:

    filmtv_movies_complete["notoriety_actors"].fillna(
        method = "median",
        by = [
            "director",
            "Category",
        ],
    )
    filmtv_movies_complete["castings_actors"].fillna(
        method = "median",
        by = [
            "director",
            "Category",
        ],
    )
    filmtv_movies_complete.dropna()
    res = filmtv_movies_complete
    html_file = open("SPHINX_DIRECTORY/figures/examples_movies_filmtv_complete_after_drop.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_movies_filmtv_complete_after_drop.html

Before we export the data, we should normalize the numerical columns to get the dummies of the different categories.

.. ipython:: python

    filmtv_movies_complete.normalize(
        method = "minmax",
        columns = [
            "votes", 
            "duration", 
            "notoriety_director",
            "castings_director",
            "notoriety_actors",
            "castings_actors",
        ],
    )
    for elem in ["category", "period", "language_area"]:
        filmtv_movies_complete[elem].get_dummies(drop_first = True)

We can export the results to our Vertica database.

.. code-block:: python

    filmtv_movies_complete.to_db(
        name = "filmtv_movies_complete",
        relation_type = "table",
        inplace = True,
    )
    filmtv_movies_complete.to_db(
        name = "filmtv_movies_mco",
        relation_type = "view",
        db_filter = "votes > 0.02",
    )

.. ipython:: python
    :suppress:

    vp.drop("filmtv_movies_complete")
    filmtv_movies_complete.to_db(
        name = "filmtv_movies_complete",
        relation_type = "table",
        inplace = True,
    )
    vp.drop("filmtv_movies_mco")
    filmtv_movies_complete.to_db(
        name = "filmtv_movies_mco",
        relation_type = "view",
        db_filter = "votes > 0.02",
    )

Machine Learning : Adjusting the Films Rates
---------------------------------------------

Let's create a model to evaluate an unbiased score for each different movie.

.. ipython:: python

    from verticapy.machine_learning.vertica.linear_model import LinearRegression

    predictors = filmtv_movies_complete.get_columns(
        exclude_columns = [
            "avg_vote",
            "period",
            "director",
            "language_area",
            "title", 
            "year",
            "country",
            "Category",
        ],
    )
    vp.drop("filmtv_movies_lr") # If model name already exists
    model = LinearRegression(
        "filmtv_movies_lr",
        max_iter = 1000,
        solver = "BFGS",
    )
    model.fit("filmtv_movies_mco", predictors, "avg_vote")

.. code-block:: python

    model.report()

.. ipython:: python
    :suppress:
    :okwarning:

    res = model.report()
    html_file = open("SPHINX_DIRECTORY/figures/examples_movies_filmtv_complete_model_report.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_movies_filmtv_complete_model_report.html

The model is good. Let's add it in our :py:mod:`~verticapy.vDataFrame`.

.. code-block:: python

    model.predict(
        filmtv_movies_complete,
        name = "unbiased_vote",
    )

.. ipython:: python
    :suppress:
    :okwarning:

    res = model.predict(
        filmtv_movies_complete,
        name = "unbiased_vote",
    )
    html_file = open("SPHINX_DIRECTORY/figures/examples_movies_filmtv_complete_model_predict.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_movies_filmtv_complete_model_predict.html

Since a score can't be greater than 10 or less than 0, we need to adjust the ``unbiased_vote``.

.. ipython:: python

    filmtv_movies_complete["unbiased_vote"] = fun.case_when(
        filmtv_movies_complete["unbiased_vote"] > 10, 10,
        filmtv_movies_complete["unbiased_vote"] < 0, 0,
        filmtv_movies_complete["unbiased_vote"],
    )

Let's look at the top movies.

.. code-block:: python

    filmtv_movies_complete.search(
        usecols = [
            "filmtv_id",
            "title",
            "year",
            "country",
            "avg_vote",
            "unbiased_vote",
            "votes",
            "duration",
            "director",
            "notoriety_director",
            "castings_director",
            "notoriety_actors",
            "castings_actors",
            "period",
            "language_area",
        ],
        order_by = {
            "unbiased_vote" : "desc", 
            "avg_vote" : "desc",
        },
    ).head(10)

.. ipython:: python
    :suppress:

    res = filmtv_movies_complete.search(
        usecols = [
            "filmtv_id",
            "title",
            "year",
            "country",
            "avg_vote",
            "unbiased_vote",
            "votes",
            "duration",
            "director",
            "notoriety_director",
            "castings_director",
            "notoriety_actors",
            "castings_actors",
            "period",
            "language_area",
        ],
        order_by = {
            "unbiased_vote" : "desc", 
            "avg_vote" : "desc",
        },
    ).head(10)
    html_file = open("SPHINX_DIRECTORY/figures/examples_movies_filmtv_top_movie_head.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_movies_filmtv_top_movie_head.html

Great, our results are more consistent. Psycho, Pulp Fiction, and The Godfather are among the top movies.

Machine Learning : Creating Movie Clusters
-------------------------------------------

Since ``k-means`` clustering is sensitive to unnormalized data, let's normalize our new predictors.

.. code-block:: python

    filmtv_movies_complete["unbiased_vote"].normalize(method = "minmax")

.. ipython:: python
    :suppress:

    res = filmtv_movies_complete["unbiased_vote"].normalize(method = "minmax")
    html_file = open("SPHINX_DIRECTORY/figures/examples_movies_filmtv_normalize_minmax.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_movies_filmtv_normalize_minmax.html

Let's compute the :py:func:`~verticapy.machine_learning.model_selection.elbow` curve to find a suitable number of clusters.

.. ipython:: python

    predictors = filmtv_movies_complete.get_columns(
        exclude_columns = [
            "avg_vote",
            "period",
            "director",
            "language_area",
            "title", 
            "year",
            "country",
            "Category",
            "filmtv_id",
        ],
    )

    from verticapy.machine_learning.model_selection import elbow
    import verticapy

    verticapy.set_option("plotting_lib", "plotly") # to switch plotting graphics to plotly
    elbow_chart = elbow(
        filmtv_movies_complete,
        predictors,
        n_cluster = (1, 60),
        show = True
    )

.. code-block:: python

    elbow_chart

.. ipython:: python
    :suppress:

    import verticapy

    verticapy.set_option("plotting_lib", "plotly")
    fig = elbow_chart
    fig.write_html("SPHINX_DIRECTORY/figures/examples_movies_filmtv_elbow_plot.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_movies_filmtv_elbow_plot.html

By looking at the elbow curve, we can choose 15 clusters. Let's create a ``k-means`` model.

.. ipython:: python

    from verticapy.machine_learning.vertica.cluster import KMeans

    model_kmeans = KMeans(n_cluster = 15)
    model_kmeans.fit(filmtv_movies_complete, predictors)
    model_kmeans.clusters_

Let's add the clusters in the :py:mod:`~verticapy.vDataFrame`.

.. code-block:: python

    model_kmeans.predict(
        filmtv_movies_complete, 
        name = "movies_cluster",
    )

.. ipython:: python
    :suppress:

    res = model_kmeans.predict(
        filmtv_movies_complete, 
        name = "movies_cluster",
    )
    html_file = open("SPHINX_DIRECTORY/figures/examples_movies_filmtv_movie_cluster_predict.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_movies_filmtv_movie_cluster_predict.html

Let's look at the different clusters.

.. code-block:: python

    filmtv_movies_complete.search(
        filmtv_movies_complete["movies_cluster"] == 0,
        usecols=[
            "avg_vote",
            "period",
            "director",
            "language_area",
            "title",
            "year",
            "country",
            "Category",
        ]
    )

.. ipython:: python
    :suppress:

    res = filmtv_movies_complete.search(
        filmtv_movies_complete["movies_cluster"] == 0,
        usecols=[
            "avg_vote",
            "period",
            "director",
            "language_area",
            "title",
            "year",
            "country",
            "Category",
        ],
    )
    html_file = open("SPHINX_DIRECTORY/figures/examples_movies_filmtv_movie_cluster_0_search.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_movies_filmtv_movie_cluster_0_search.html

.. code-block:: python

    filmtv_movies_complete.search(
        filmtv_movies_complete["movies_cluster"] == 1,
        usecols=[
            "avg_vote",
            "period",
            "director",
            "language_area",
            "title",
            "year",
            "country",
            "Category",
        ],
    )

.. ipython:: python
    :suppress:

    res = filmtv_movies_complete.search(
        filmtv_movies_complete["movies_cluster"] == 1,
        usecols=[
            "avg_vote",
            "period",
            "director",
            "language_area",
            "title",
            "year",
            "country",
            "Category",
        ],
    )
    html_file = open("SPHINX_DIRECTORY/figures/examples_movies_filmtv_movie_cluster_1_search.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_movies_filmtv_movie_cluster_1_search.html

.. code-block:: python

    filmtv_movies_complete.search(
        filmtv_movies_complete["movies_cluster"] == 2,
        usecols=[
            "avg_vote",
            "period",
            "director",
            "language_area",
            "title",
            "year",
            "country",
            "Category",
        ],
    )

.. ipython:: python
    :suppress:

    res = filmtv_movies_complete.search(
        filmtv_movies_complete["movies_cluster"] == 2,
        usecols=[
            "avg_vote",
            "period",
            "director",
            "language_area",
            "title",
            "year",
            "country",
            "Category",
        ],
    )
    html_file = open("SPHINX_DIRECTORY/figures/examples_movies_filmtv_movie_cluster_2_search.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_movies_filmtv_movie_cluster_2_search.html

.. code-block:: python

    filmtv_movies_complete.search(
        filmtv_movies_complete["movies_cluster"] == 3,
        usecols=[
            "avg_vote",
            "period",
            "director",
            "language_area",
            "title",
            "year",
            "country",
            "Category",
        ],
    )

.. ipython:: python
    :suppress:

    res = filmtv_movies_complete.search(
        filmtv_movies_complete["movies_cluster"] == 3,
        usecols = [
            "avg_vote",
            "period",
            "director",
            "language_area",
            "title",
            "year",
            "country",
            "Category",
        ],
    )
    html_file = open("SPHINX_DIRECTORY/figures/examples_movies_filmtv_movie_cluster_3_search.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_movies_filmtv_movie_cluster_3_search.html

Each cluster consists of similar movies. These clusters can be used to give movie recommendations or help streaming platforms group movies together.

Conclusion
----------

We've solved our problem in a Pandas-like way, all without ever loading data into memory!