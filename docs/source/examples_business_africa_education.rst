.. _examples.business.africa_education:

Africe Education
=================

This example uses the 'Africa Education' dataset to predict student performance. 
You can can download the Jupyter Notebook of the study `here <https://github.com/vertica/VerticaPy/blob/master/examples/understand/understand/africa_education/africa_education.ipynb>`_.

- **COUNTRY:** COUNTRY ID.
- **REGION:** REGION ID.
- **SCHOOL:** SCHOOL ID.
- **PUPIL:** STUDENT ID.
- **province:** School Province.
- **schoolname:** School Name.
- **lat:** School Latitude.
- **long:** School Longitude.
- **country_long:** Country Name.
- **zralocp:** Student's standardized reading score.
- **zmalocp:** Student's standardized mathematics score.
- **ZRALEVP:** Student's reading level.
- **ZMALEVP:** Student's mathematics competency level.
- **zraloct:** Teacher's standardized reading score.
- **ZRALEVT:** Student's reading competency level Teacher.
- **ZMALEVT:** Student's mathematics competency level Teacher.
- **zsdist:** School average distance from clinic, road, public, library, book shop & secondary school.
- **XNUMYRS:** Teacher's years of teaching.
- **numstu:** Number of students at each school.
- **PSEX:** Student's sex.
- **PNURSERY:** Student preschool.
- **PENGLISH:** Student speaks English at home.
- **PMALIVE:** Student's biological mother alive.
- **PFALIVE:** Student's biological father alive.
- **PTRAVEL:** Travels to school.
- **PTRAVEL2:** Means of transportation to school.
- **PMOTHER:** Mother's education.
- **PFATHER:** Father's education.
- **PLIGHT:** Source of lighting.
- **PABSENT:** Days absent.
- **PREPEAT:** Years repeated.
- **STYPE:** School type.
- **SLOCAT:** School location.
- **SQACADEM:** Academic qualifications.
- **XSEX:** Teacher's sex.
- **XAGE:** Teacher's age.
- **XQPERMNT:** Teacher's employment status.
- **XQPROFES:** Teacher's training.
- **zpsibs:** Student's number of siblings.
- **zpsit:** Seating location.
- **zpmealsc:** Free school meals.
- **zphmwkhl:** Homework help.
- **zpses:** Student's socioeconomic status.
- **PAGE:** Student's Age.
- **SINS2006:** School inspection.
- **SPUPPR04:** Student dropout.
- **SPUPPR06:** Student cheats.
- **SPUPPR07:** Student uses abusive language.
- **SPUPPR08:** Student vandalism.
- **SPUPPR09:** Student theft.
- **SPUPPR10:** Student bullies students.
- **SPUPPR11:** Student bullies staff.
- **SPUPPR12:** Student injures staff.
- **SPUPPR13:** Student sexually harrasses students.
- **SPUPPR14:** Student sexually harrasses teachers.
- **SPUPPR15:** Student drug abuse.
- **SPUPPR16:** Student alcohol abuse.
- **SPUPPR17:** Student fights.
- **STCHPR04:** Teacher bullies students.
- **STCHPR05:** Teacher sexually harasses teachers.
- **STCHPR06:** Teacher sexually harasses students.
- **STCHPR07:** Teacher uses abusive language.
- **STCHPR08:** Teacher drug abuse.
- **STCHPR09:** Teacher alcohol abuse.

We will follow the data science cycle (Data Exploration - Data Preparation - Data Modeling - Model Evaluation - Model Deployment) to solve this problem.

Initialization
---------------

This example uses the following version of VerticaPy:

.. ipython:: python
    
    import verticapy as vp

    vp.__version__

Connect to Vertica. This example uses an existing connection called "VerticaDSN." 
For details on how to create a connection, see the :ref:`connection` tutorial.
You can skip the below cell if you already have an established connection.

.. code-block:: python
    
    vp.connect("VerticaDSN")

Let's create a Virtual DataFrame of the dataset.

.. ipython:: python

    from verticapy.datasets import load_africa_education

    africa = load_africa_education()

.. warning::
    
    This example uses a sample dataset. For the full analysis, you should consider using the complete dataset.

Data Exploration and Preparation
---------------------------------

Let's look at the links between all the variables. 
Remember our goal: find a way to predict students' final scores ('zralocp' & 'zmalocp').

.. code-block:: python

    africa.corr()

.. ipython:: python
    :suppress:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    africa = africa.sample(x = 0.1)
    fig = africa.corr(width = 900)
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_africe_corr_matrix.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_africe_corr_matrix.html

Some variables are useless because they are categorizations of others. 
For example, most scores can go from 0 to 1000, and some variables are created by mapping these variables to a reduced interval (for example: 0 to 10), so we can drop them.

.. code-block:: python

    africa.drop(
        [
            "ZMALEVT", 
            "ZRALEVT", 
            "ZRALEVP", 
            "ZMALEVP",
            "COUNTRY",
            "SCHOOL",
            "PUPIL",
        ],
    )

.. ipython:: python
    :suppress:

    africa.drop(
        [
            "ZMALEVT", 
            "ZRALEVT", 
            "ZRALEVP", 
            "ZMALEVP",
            "COUNTRY",
            "SCHOOL",
            "PUPIL",
        ],
    )

Let's take a look at the missing values.

.. code-block:: python

    africa.count_percent()

.. ipython:: python
    :suppress:

    res = africa.count_percent()
    html_file = open("/project/data/VerticaPy/docs/figures/examples_africa_count_percent.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_africa_count_percent.html

Many values are missing for 'zraloct' which is the teachers' test score. We need to find a way to impute them as they represent more than 10% of the dataset. For the others that represent less than 5% of the dataset, our goal is to identify what improves student performance, so we can filter them.

We'll use two variables to impute the teachers' scores: TEACHER'S SEX (XSEX) and Teacher's Training (XQPROFES).


.. code-block:: python

    africa["zraloct"].fillna(
        method = "avg", 
        by = ["XSEX", "XQPROFES"],
    )
    africa.dropna()


.. ipython:: python
    :suppress:

    africa["zraloct"].fillna(
        method = "avg", 
        by = ["XSEX", "XQPROFES"],
    )
    africa.dropna()
    res = africa
    html_file = open("/project/data/VerticaPy/docs/figures/examples_africa_after_drop.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_africa_after_drop.html


Now that we have a clean dataset, we can use a Random Forest Regressor to understand what tends to influence the a student's final score.

Machine Learning: Finding Clusters using lat/long
-------------------------------------------------

Let's try to find some clusters between schools.

Since we have the school's location, a natural approach might be to find school clusters based on proximity. 
These clusters can be used as inputs by our model.



.. code-block:: python

    from verticapy.machine_learning.model_selection import elbow

    elbow(
        africa,
        X = ["lon", "lat"],
        n_cluster = (1, 30),
        show = True,
    )

.. ipython:: python
    :suppress:

    from verticapy.machine_learning.model_selection import elbow
    fig = elbow(
        africa,
        X = ["lon", "lat"],
        n_cluster = (1, 30),
        show = True,
    )
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_africa_elbow.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_africa_elbow.html

Eight seems to be a suitable number of clusters. Let's compute a ``k-means`` model.

.. code-block:: python

    from verticapy.machine_learning.vertica import KMeans

    model = KMeans(n_cluster = 8)
    model.fit(africa, X = ["lon", "lat"])

.. ipython:: python
    :suppress:

    from verticapy.machine_learning.vertica import KMeans
    model = KMeans(n_cluster = 8)
    model.fit(africa, X = ["lon", "lat"])

We can add the prediction to the :py:mod:`vDataFrame` and draw the scatter map.


.. code-block:: python

    # Change the plotting lib to matplotlib
    vp.set_option("plotting_lib", "matplotlib")

    # Adding the prediction to the vDataFrame
    model.predict(africa, name = "clusters")

    # Importing the World Data
    from verticapy.datasets import load_world

    africa_world = load_world()

    # Filtering and drawing Africa
    africa_world = africa_world[africa_world["continent"] == "Africa"]
    ax = africa_world["geometry"].geo_plot(color = "white", edgecolor='black',)

.. ipython:: python
    :suppress:

    vp.set_option("plotting_lib", "matplotlib")
    model.predict(africa, name = "clusters")
    from verticapy.datasets import load_world
    africa_world = load_world()
    africa_world = africa_world[africa_world["continent"] == "Africa"]
    

.. ipython:: python

    ax = africa_world["geometry"].geo_plot(color = "white", edgecolor='black',)
    @savefig examples_africa_geo_plot.png
    africa.scatter(
        ["lon", "lat"],
        by = "clusters",
        ax = ax,
    )

Machine Learning: Understanding the Students' Final Scores
-----------------------------------------------------------

A student's math score is strongly correlated their reading score, 
so we can use just one of the variables for our predictions. 
Let's use a cross validation to see if our variables have enough 
information to predict the students' scores.


.. ipython:: python
    :okwarning:

    from verticapy.machine_learning.vertica import RandomForestRegressor
    from verticapy.machine_learning.model_selection import cross_validate

    predictors = africa.get_columns(
        exclude_columns = [
            "zralocp", 
            "zmalocp",
            "lat", 
            "lon",
            "schoolname",
        ],
    )
    response = "zralocp"
    model = RandomForestRegressor(
        n_estimators = 40,
        max_depth = 20,
        min_samples_leaf = 4,
        nbins = 20,
        sample = 0.7,
    )
    cross_validate(
        model, 
        africa,
        X = predictors, 
        y = response,
    )

These scores are quite good! Let's fit all the data and keep the most important variables.

.. code-block:: python

    model.fit(
        africa, 
        X = predictors, 
        y = response,
    )

.. ipython:: python
    :okwarning:
    :suppress:

    model.fit(
        africa, 
        X = predictors, 
        y = response,
    )
    
.. ipython:: python

    predictors = model.features_importance(show = False)["index"]

We can see here that socioeconomic status and a student's country 
tend to strongly influence the students work quality. 
This makes sense: you would expect that having poor studying 
conditions (unstable government, difficulties at home, etc.) 
would lead to worse results. For now, let's just consider the 20 most important variables.

Let's do some tuning to find the best parameters for the use case. 
Our goal will be to optimize the 'median_absolute_error'.

.. code-block:: python

    from verticapy.machine_learning.model_selection import grid_search_cv

    gcv = grid_search_cv(
        model,
        {
            "min_samples_leaf": [1, 3],
            "max_leaf_nodes": [50],
            "max_depth": [5, 8],
        },
        metric = "median",
        input_relation = africa,
        X = predictors[:20], 
        y = response,
    )
    gcv

.. ipython:: python
    :suppress:

    gcv_params = {
        'n_estimators': 40,
        'max_features': 'auto',
        'max_leaf_nodes': 100,
        'sample': 0.7,
        'max_depth': 10,
        'min_samples_leaf': 3,
        'min_info_gain': 0.0,
        'nbins': 20,
    }

Our model is excellent. Let's create one for the students' standardized reading score ('zralocp').

.. code-block:: python

    response = "zralocp"
    model_africa_rf_zralocp = RandomForestRegressor(**gcv["parameters"][0])
    model_africa_rf_zralocp.fit(
        africa,
        predictors[0:20], 
        response,
    )
    model_africa_rf_zralocp.regression_report()

.. ipython:: python
    :suppress:
    :okwarning:

    response = "zralocp"
    model_africa_rf_zralocp = RandomForestRegressor(**gcv_params)
    model_africa_rf_zralocp.fit(
        africa,
        predictors[0:20], 
        response,
    )
    res = model_africa_rf_zralocp.regression_report()
    html_file = open("/project/data/VerticaPy/docs/figures/examples_africa_reg_report_zralocp.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_africa_reg_report_zralocp.html

We'll also create one for the students' standardized mathematics score ('zmalocp').

.. code-block:: python

    response = "zmalocp"
    model_africa_rf_zmalocp = RandomForestRegressor(**gcv["parameters"][0])
    model_africa_rf_zmalocp.fit(
        africa,
        predictors[0:20], 
        response,
    )
    model_africa_rf_zmalocp.regression_report()

.. ipython:: python
    :suppress:
    :okwarning:

    response = "zmalocp"
    model_africa_rf_zmalocp = RandomForestRegressor(**gcv_params)
    model_africa_rf_zmalocp.fit(
        africa,
        predictors[0:20], 
        response,
    )
    res = model_africa_rf_zmalocp.regression_report()
    html_file = open("/project/data/VerticaPy/docs/figures/examples_africa_reg_report_zmalocp.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_africa_reg_report_zmalocp.html

Let's look at the feature importance for each model.

.. code-block:: python

    model_africa_rf_zralocp.features_importance()

.. ipython:: python
    :suppress:
    :okwarning:

    vp.set_option("plotting_lib", "plotly")
    fig = model_africa_rf_zralocp.features_importance()
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_africa_feature_zralocp.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_africa_feature_zralocp.html

.. code-block:: python

    model_africa_rf_zmalocp.features_importance()

.. ipython:: python
    :suppress:
    :okwarning:

    fig = model_africa_rf_zmalocp.features_importance()
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_africa_feature_zmalocp.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_africa_feature_zmalocp.html

Feature importance between between math score and the reading score are almost identical.

We can add these predictions to the main :py:mod:`vDataFrame`.

.. code-block:: python

    africa = africa.select(predictors[0:23] + ["zralocp", "zmalocp"])
    model_africa_rf_zralocp.predict(africa, name = "pred_zralocp")
    model_africa_rf_zmalocp.predict(africa, name = "pred_zmalocp")

.. ipython:: python
    :suppress:
    :okwarning:

    africa = africa.select(predictors[0:23] + ["zralocp", "zmalocp"])
    model_africa_rf_zralocp.predict(africa, name = "pred_zralocp")
    model_africa_rf_zmalocp.predict(africa, name = "pred_zmalocp")
    res = model_africa_rf_zmalocp

Let's visualize our model. We begin by creating a bubble plot using the two scores. 

.. code-block:: python

    vp.set_option("plotting_lib", "plotly")
    africa.scatter(
        columns = ["zralocp", "zmalocp"],
        size = "zpses",
        by = "PENGLISH",
        max_nb_points = 2000,
    )

.. ipython:: python
    :suppress:
    :okwarning:

    vp.set_option("plotting_lib", "plotly")
    fig = africa.scatter(
        columns = ["zralocp", "zmalocp"],
        size = "zpses",
        by = "PENGLISH",
        max_nb_points = 2000,
    )
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_africa_scatter_bubble.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_africa_scatter_bubble.html

Notable influences are home language and the socioeconomic status. 
It seems like students that both speak Engish at home often (but not all the time) and have a comfortable standard of living tend to perform the best.

Now, let's see how a student's nationality might affect their performance.

.. code-block:: python

    africa["country_long"].bar(
        method = "90%", 
        of = "pred_zmalocp",
        max_cardinality = 50,
        width = 800,
    )

.. ipython:: python
    :suppress:
    :okwarning:

    fig = africa["country_long"].bar(
        method = "90%", 
        of = "pred_zmalocp",
        max_cardinality = 50,
        width = 800,
    )
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_africa_bar_90_country_long.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_africa_bar_90_country_long.html

.. code-block:: python

    africa["country_long"].bar(
        method = "10%", 
        of = "pred_zmalocp",
        max_cardinality = 50,
        width = 800,
    )

.. ipython:: python
    :suppress:
    :okwarning:
    :okexcept:

    fig = africa["country_long"].bar(
        method = "10%", 
        of = "pred_zmalocp",
        max_cardinality = 50,
        width = 800,
    )
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_africa_bar_10_country_long.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_africa_bar_10_country_long.html

The students' nationalities seem to have big impact. 
For example, Swaziland, Kenya, and Tanzanie are probably 
overrating the bad students (90% of the scores are greater 
than the average (500)) whereas some countries like Zambia, 
South Africa, and Malawi are underrating their students 
(90% of the scores are under 480). This could be related to the 
global education in the country: some education systems could 
be harder than the others. Let's break this down by region.

.. code-block:: python

    africa["district"].bar(
        method = "50%", 
        of = "pred_zmalocp",
        max_cardinality = 50,
        width = 1000,
    )

.. ipython:: python
    :suppress:
    :okwarning:
    :okexcept:

    fig = africa["district"].bar(
        method = "50%", 
        of = "pred_zmalocp",
        max_cardinality = 50,
        width = 1000,
    )
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_africa_bar_district.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_africa_bar_district.html

The same applies to the regions. Let's look at student age.

.. code-block:: python

    africa["PAGE"].bar(
        method = "50%", 
        of = "pred_zmalocp",
        max_cardinality = 50,
    )

.. ipython:: python
    :suppress:
    :okwarning:
    :okexcept:

    fig = africa["PAGE"].bar(
        method = "50%", 
        of = "pred_zmalocp",
        max_cardinality = 50,
    )
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_africa_bar_page.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_africa_bar_page.html

Let's look at the the variables 'PLIGHT' (a student's main lighting source) and 'PREPEAT' (repeated years).

.. code-block:: python

    africa.bar(
        columns = ["PREPEAT", "PLIGHT"],
        method = "avg", 
        of = "pred_zmalocp",
        width = 850,
    )

.. ipython:: python
    :suppress:
    :okwarning:

    fig = africa.bar(
        columns = ["PREPEAT", "PLIGHT"],
        method = "avg", 
        of = "pred_zmalocp",
        width = 850,
    )
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_africa_bar_prepeat_plight.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_africa_bar_prepeat_plight.html

We can see that students who never repeated a year and have light at home tend to do better in school than those who don't.

Another factor in a student's performance might be their method of transportation, so we'll look at the "ptravel2" variable.

.. code-block:: python

    africa["ptravel2"].bar(
        method = "50%", 
        of = "pred_zmalocp",
        width = 850,
    )

.. ipython:: python
    :suppress:
    :okwarning:
    :okexcept:

    fig = africa["ptravel2"].bar(
        method = "50%", 
        of = "pred_zmalocp",
        width = 850,
    )
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_africa_bar_ptravel2.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_africa_bar_ptravel2.html

We can clearly see that the more inconvenient it is to get to school, 
the worse students tend to perform.

Let's look at the influence of the 'district'.

Predictably, better teachers generally lead to better results. 
Let's look at the influence of the 'district'.

.. code-block:: python

    africa["district"].bar(
        method = "50%",
        of = "pred_zmalocp",
        h = 100,
    )

.. ipython:: python
    :suppress:
    :okexcept:
    :okwarning:

    fig = africa["district"].bar(
        method = "50%",
        of = "pred_zmalocp",
        h = 100,
    )
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_africa_bar_district_50_pred.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_africa_bar_district_50_pred.html

Here, we can see that Chicualacuala has a very high median score, so we can conclude that a students' district might impact their performance in school.

After assessing several predictors of student-performance, we can hypothesize some solutions. For example, we might suggest in investing in extracurricular activities, ensuring that students have adequate light sources at home, or improving public transportation.

Machine Learning: Finding the Best Students
---------------------------------------------

To find the best students we can use each school's ID (the SCHOOL variable) and compute the average score. 
We can then order these by descending average score and note the top five students at each school.

.. code-block:: python

    africa = load_africa_education()

    # Computing the averaged score
    africa["score"] = (africa["zralocp"] + africa["zmalocp"]) / 2

    # Computing the averaged student score
    africa.analytic(
        func = "row_number",
        by = ["schoolname"],
        order_by = {"score": "desc"},
        name = "student_class_position",
    )

    # Finding the 3 best students by class
    africa.case_when(
        "best",
        africa["student_class_position"] <= 5, 1,
        0,
    )

    # Selecting the main variables
    africa = africa[
        [
            "PENGLISH", 
            "PAGE", 
            "zpses", 
            "PREPEAT",
            "PTRAVEL2", 
            "PLIGHT",
            "SLOCAT",
            "best",
            "zpmealsc",
            "PFATHER",
            "SPUPPR04",
            "PNURSERY",
        ]
    ]

    # Getting the categories dummies for the Logistic Regression
    africa.one_hot_encode(
        columns = [
            "PLIGHT", 
            "PTRAVEL2",
            "PREPEAT",
            "PENGLISH",
            "SLOCAT",
            "PFATHER",
            "SPUPPR04",
            "PNURSERY",
            "zpmealsc"
        ],
        max_cardinality = 1000,
    )
    africa.dropna()

.. ipython:: python
    :suppress:
    :okwarning:

    africa = load_africa_education()
    africa["score"] = (africa["zralocp"] + africa["zmalocp"]) / 2 
    africa.analytic(
        func = "row_number",
        by = ["schoolname"],
        order_by = {"score": "desc"},
        name = "student_class_position",
    )
    africa.case_when(
        "best",
        africa["student_class_position"] <= 5, 1,
        0,
    )
    africa = africa[
        [
            "PENGLISH", 
            "PAGE", 
            "zpses", 
            "PREPEAT",
            "PTRAVEL2", 
            "PLIGHT",
            "SLOCAT",
            "best",
            "zpmealsc",
            "PFATHER",
            "SPUPPR04",
            "PNURSERY",
        ]
    ]
    africa.one_hot_encode(
        columns = [
            "PLIGHT", 
            "PTRAVEL2",
            "PREPEAT",
            "PENGLISH",
            "SLOCAT",
            "PFATHER",
            "SPUPPR04",
            "PNURSERY",
            "zpmealsc",
        ],
        max_cardinality = 1000,
    )
    africa.dropna()

Let's create a logistic regression to understand what circumstances allowed these students to perform as well as they have.

.. code-block:: ipython

    from verticapy.machine_learning.vertica import LogisticRegression

    response = "best"
    predictors = africa.get_columns(
        exclude_columns = [
            "PLIGHT", 
            "PTRAVEL2",
            "PREPEAT",
            "PENGLISH",
            "SLOCAT",
            "PFATHER",
            "SPUPPR04",
            "PNURSERY",
            "zpmealsc",
            "best",
        ]
    )
    model_africa_logit_best = LogisticRegression(
        name="africa_logit_best", 
        solver="BFGS",
    )
    model_africa_logit_best.fit(
        africa,
        predictors, 
        response,
    )
    model_africa_logit_best.features_importance()

.. ipython:: python
    :suppress:
    :okwarning:

    from verticapy.machine_learning.vertica import LogisticRegression
    predictors = africa.get_columns(exclude_columns = [
            "PLIGHT",
            "PTRAVEL2",
            "PREPEAT", 
            "PENGLISH",
            "SLOCAT",
            "PFATHER",
            "SPUPPR04",
            "PNURSERY",
            "zpmealsc",
            "best",
        ]
    )
    vp.drop("africa_logit_best")
    model_africa_logit_best = LogisticRegression(name="africa_logit_best",solver="BFGS")
    model_africa_logit_best.fit(africa,predictors,"best")
    fig = model_africa_logit_best.features_importance()
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_africa_feature_final.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_africa_feature_final.html

We can see that the best students tend to be young, speak English at home, come from a good socioeconomic background, have a father with a degree, and live relatively close to school.

Conclusion
-----------

We've solved our problem in a Pandas-like way, all without ever loading data into memory!