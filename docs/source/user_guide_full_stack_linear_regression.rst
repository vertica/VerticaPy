.. _user_guide.full_stack.linear_regression:

==================
Linear Regression
==================

Linear regression is one of the most popular regression algorithms and produces good predictions for well-prepared data. Its optimization function computes coefficients to express a response column as a linear relationship of its predictors.

You must verify the Gauss-Markov assumptions when using linear regression algorithms:

- **Linearity:** the parameters we are estimating using the OLS method must be linear.
- **Non-Collinearity:** the regressors being calculated aren’t perfectly correlated with each other.
- **Exogeneity:** the regressors aren’t correlated with the error term.
- **Homoscedasticity:** no matter what the values of our regressors might be, the error of the variance is constant.

To create a good linear regression model, it's important to:

- Impute missing values.
- Encode categorical features (linear regression only accepts numerical variables).
- Compute the correlation matrix to retrieve highly-correlated predictors.
- Decompose the data (optional).
- Normalize the data (optional, but recommended).

Example without decomposition
------------------------------

Let's use the 'africa_education' dataset to compute a linear regression model of students' performance in school.

.. code-block:: ipython
    
    from verticapy.datasets import load_africa_education

    africa = load_africa_education()
    africa = africa.select(
        [
            "(zralocp + zmalocp) / 2 AS student_score",
            "zraloct AS teacher_score",
            "XNUMYRS AS teacher_year_teaching",
            "numstu AS number_students_school",
            "PENGLISH AS english_at_home",
            "PTRAVEL AS travel_distance",
            "PTRAVEL2 AS means_of_travel",
            "PMOTHER AS m_education",
            "PFATHER AS f_education",
            "PLIGHT AS source_of_lighting",
            "PABSENT AS days_absent",
            "PREPEAT AS repeated_grades",
            "zpsit AS sitting_place",
            "PAGE AS age",
            "zpses AS socio_eco_statut",
            "country_long AS country",
        ],
    )
    africa.head(100)

.. ipython:: python
    :suppress:
    :okwarning:

    from verticapy.datasets import load_africa_education

    africa = load_africa_education()
    africa = africa.select(
        [
            "(zralocp + zmalocp) / 2 AS student_score",
            "zraloct AS teacher_score",
            "XNUMYRS AS teacher_year_teaching",
            "numstu AS number_students_school",
            "PENGLISH AS english_at_home",
            "PTRAVEL AS travel_distance",
            "PTRAVEL2 AS means_of_travel",
            "PMOTHER AS m_education",
            "PFATHER AS f_education",
            "PLIGHT AS source_of_lighting",
            "PABSENT AS days_absent",
            "PREPEAT AS repeated_grades",
            "zpsit AS sitting_place",
            "PAGE AS age",
            "zpses AS socio_eco_statut",
            "country_long AS country",
        ],
    )
    res = africa.head(100)
    html_file = open("/project/data/VerticaPy/docs/figures/ug_fs_table_lr_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_table_lr_1.html

First, let's look for missing values.

.. code-block:: ipython
    
    africa.count_percent()

.. ipython:: python
    :suppress:
    :okwarning:

    res = africa.count_percent()
    html_file = open("/project/data/VerticaPy/docs/figures/ug_fs_table_lr_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_table_lr_2.html

We'll simply drop the missing values to avoid adding bias to the data.

.. code-block:: ipython
    
    africa.dropna()

.. ipython:: python
    :suppress:
    :okwarning:

    res = africa.dropna()
    html_file = open("/project/data/VerticaPy/docs/figures/ug_fs_table_lr_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_table_lr_2.html

We need to encode the categorical columns to dummies to retain linearity.

.. code-block:: ipython
    
    africa.one_hot_encode(max_cardinality = 20)

.. ipython:: python
    :suppress:
    :okwarning:

    res = africa.one_hot_encode(max_cardinality = 20)
    html_file = open("/project/data/VerticaPy/docs/figures/ug_fs_table_lr_3.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_table_lr_3.html

Linear regression can only handle numerical columns, so we'll drop the categorical columns.

.. code-block:: ipython
    
    africa.drop(
        columns = [
            "english_at_home",
            "travel_distance",
            "means_of_travel",
            "m_education",
            "f_education",
            "source_of_lighting",
            "repeated_grades",
            "sitting_place",
            "country",
        ],
    )

.. ipython:: python
    :suppress:
    :okwarning:

    res = africa.drop(
        columns = [
            "english_at_home",
            "travel_distance",
            "means_of_travel",
            "m_education",
            "f_education",
            "source_of_lighting",
            "repeated_grades",
            "sitting_place",
            "country",
        ],
    )
    html_file = open("/project/data/VerticaPy/docs/figures/ug_fs_table_lr_4.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_table_lr_4.html

Let's look at the correlation between the response column and the predictors. We'll look to keep columns with correlations coefficients greater than 20% (the top 10 features).

.. code-block:: ipython
    
    x = africa.corr(focus = "student_score", show = False)
    africa = africa.select(columns = x["index"][0:12])
    africa.head(100)

.. ipython:: python
    :suppress:
    :okwarning:

    x = africa.corr(focus = "student_score", show = False)
    africa = africa.select(columns = x["index"][0:12])
    res = africa.head(100)
    html_file = open("/project/data/VerticaPy/docs/figures/ug_fs_table_lr_5.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_table_lr_5.html

Let's examine the correlation matrix to see if we have any independent predictors.

.. code-block:: python

    africa.corr()

.. ipython:: python
    :suppress:
    :okwarning:

    vp.set_option("plotting_lib","plotly")
    fig = africa.corr()
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_fs_plot_lr_6.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_plot_lr_6.html

Some of these features are highly-correlated, like socioeconomic status and having an electric lighting. We'll drop the lighting column to avoid unexpected results while computing the linear regression.

.. code-block:: ipython
    
    africa["source_of_lighting_ELECTRIC"].drop()

.. ipython:: python
    :suppress:
    :okwarning:

    res = africa["source_of_lighting_ELECTRIC"].drop()
    html_file = open("/project/data/VerticaPy/docs/figures/ug_fs_table_lr_7.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_table_lr_7.html

Let's normalize the dataset to follow the Gaussian-Markov assumptions.

.. code-block:: ipython
    
    africa.normalize(columns = africa.get_columns(exclude_columns = ["student_score"]))

.. ipython:: python
    :suppress:
    :okwarning:

    res = africa.normalize(columns = africa.get_columns(exclude_columns = ["student_score"]))
    html_file = open("/project/data/VerticaPy/docs/figures/ug_fs_table_lr_8.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_table_lr_8.html

We can use a cross-validation to test our model.

.. code-block:: ipython
    
    from verticapy.machine_learning.vertica import LinearRegression
    from verticapy.machine_learning.model_selection import cross_validate

    cross_validate(
        LinearRegression(solver = "BFGS"), 
        input_relation = africa,
        X = africa.get_columns(exclude_columns = ["student_score"]),
        y = "student_score",
    )

.. ipython:: python
    :suppress:
    :okwarning:

    from verticapy.machine_learning.vertica import LinearRegression
    from verticapy.machine_learning.model_selection import cross_validate
    res = cross_validate(
        LinearRegression(solver = "BFGS"), 
        input_relation = africa,
        X = africa.get_columns(exclude_columns = ["student_score"]),
        y = "student_score",
    )
    html_file = open("/project/data/VerticaPy/docs/figures/ug_fs_table_lr_9.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_table_lr_9.html

The model isn't bad. We're just using a few variables to get a median absolute error of 47; that is, our score has a distance of 47 from the true value. This seems high, but if we keep in mind that the final score is over 1000, our predictions are quite good.

Let's compare the importance of our features.

.. ipython:: python

    model = LinearRegression(solver = "BFGS")
    model.fit(
        input_relation = africa,
        X = africa.get_columns(exclude_columns = ["student_score"]),
        y = "student_score",
    )

.. code-block:: python

    model.features_importance()

.. ipython:: python
    :suppress:
    :okwarning:

    vp.set_option("plotting_lib","plotly")
    fig = model.features_importance()
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_fs_plot_lr_10.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_plot_lr_10.html

The following factors seem to have the greatest influence on a student's performance:
- Having a good teacher.
- Being of good socio-economic status.
- Tanzanian teachers tend to overrate their students.
- Age (younger students tend to perform better).
- Being able to get to school by car.
Let's add the prediction to the vDataFrame to see how our model performs its estimations.

.. code-block:: python

    model.predict(africa, name = "estimated_student_score")
    africa.boxplot(["estimated_student_score", "student_score"])

.. ipython:: python
    :suppress:
    :okwarning:

    model.predict(africa, name = "estimated_student_score")
    vp.set_option("plotting_lib","plotly")
    fig = africa.boxplot(["estimated_student_score", "student_score"])
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_fs_plot_lr_11.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_plot_lr_11.html

.. code-block:: ipython
    
    africa.describe(columns = ["student_score", "estimated_student_score"])

.. ipython:: python
    :suppress:
    :okwarning:

    res = africa.describe(columns = ["student_score", "estimated_student_score"])
    html_file = open("/project/data/VerticaPy/docs/figures/ug_fs_table_lr_12.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_table_lr_12.html

Our model has trouble catching outliers: exceptionally well-performing and struggling students.

Let's draw a residual plot.

.. code-block:: python

    africa["residual"] = africa["student_score"] - africa["estimated_student_score"]
    africa.scatter(["residual", "student_score"])

.. ipython:: python
    :suppress:
    :okwarning:

    africa["residual"] = africa["student_score"] - africa["estimated_student_score"]
    vp.set_option("plotting_lib","plotly")
    fig = africa.scatter(["residual", "student_score"])
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_fs_plot_lr_13.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_plot_lr_13.html

We see a high heteroscedasticity, indicating that we can't trust the ``p-value`` of the coefficients.

.. ipython:: python

    model.coef_

Let's look at the model's analysis of variance (ANOVA) table.

.. code-block:: ipython
    
    model.report("anova")

.. ipython:: python
    :suppress:
    :okwarning:

    res = model.report("anova")
    html_file = open("/project/data/VerticaPy/docs/figures/ug_fs_table_lr_14.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_table_lr_14.html

According to the ``ANOVA`` table, at least one of our variables is influencing the prediction.

We can also see that a student's estimated score and true score skew heavily from a normal distribution.

.. code-block:: python

    africa["estimated_student_score"].hist()

.. ipython:: python
    :suppress:
    :okwarning:

    vp.set_option("plotting_lib","plotly")
    fig = africa["estimated_student_score"].hist()
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_fs_plot_lr_15.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_plot_lr_15.html

.. ipython:: python

    from verticapy.machine_learning.model_selection.statistical_tests import jarque_bera

    jarque_bera(africa, "estimated_student_score")

Our model doesn't verify the basic hypothesis and therefore isn't stable enough to be put into production. Let's look at a second technique.

Example with decomposition
---------------------------

Let's look at the same dataset, but use decomposition techniques to filter out unimportant information. We don't have to normalize our data or look at correlations with these types of methods.

We'll begin by repeating the data preparation process of the previous section and export the resulting :py:mod:`~verticapy.vDataFrame` to Vertica.

.. code-block:: ipython
    
    africa = load_africa_education()
    africa = africa.select(
        [
            "(zralocp + zmalocp) / 2 AS student_score",
            "zraloct AS teacher_score",
            "XNUMYRS AS teacher_year_teaching",
            "numstu AS number_students_school",
            "PENGLISH AS english_at_home",
            "PTRAVEL AS travel_distance",
            "PTRAVEL2 AS means_of_travel",
            "PMOTHER AS m_education",
            "PFATHER AS f_education",
            "PLIGHT AS source_of_lighting",
            "PABSENT AS days_absent",
            "PREPEAT AS repeated_grades",
            "zpsit AS sitting_place",
            "PAGE AS age",
            "zpses AS socio_eco_statut",
            "country_long AS country",
        ],
    )
    africa.dropna()
    africa.one_hot_encode(max_cardinality = 20)
    africa.drop(
        columns = [
            "english_at_home",
            "travel_distance",
            "means_of_travel",
            "m_education",
            "f_education",
            "source_of_lighting",
            "repeated_grades",
            "sitting_place",
            "country",
        ],
    )

.. ipython:: python
    :suppress:
    :okwarning:

    africa = load_africa_education()
    africa = africa.select(
        [
            "(zralocp + zmalocp) / 2 AS student_score",
            "zraloct AS teacher_score",
            "XNUMYRS AS teacher_year_teaching",
            "numstu AS number_students_school",
            "PENGLISH AS english_at_home",
            "PTRAVEL AS travel_distance",
            "PTRAVEL2 AS means_of_travel",
            "PMOTHER AS m_education",
            "PFATHER AS f_education",
            "PLIGHT AS source_of_lighting",
            "PABSENT AS days_absent",
            "PREPEAT AS repeated_grades",
            "zpsit AS sitting_place",
            "PAGE AS age",
            "zpses AS socio_eco_statut",
            "country_long AS country",
        ],
    )
    africa.dropna()
    africa.one_hot_encode(max_cardinality = 20)
    res = africa.drop(
        columns = [
            "english_at_home",
            "travel_distance",
            "means_of_travel",
            "m_education",
            "f_education",
            "source_of_lighting",
            "repeated_grades",
            "sitting_place",
            "country",
        ],
    )
    html_file = open("/project/data/VerticaPy/docs/figures/ug_fs_table_lr_16.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_table_lr_16.html

Let's create our principal component analysis (:py:mod:`~verticapy.machine_learning.vertica.PCA`) model.

.. code-block:: ipython
    
    from verticapy.machine_learning.vertica import PCA

    model = PCA()
    model.fit(
        africa,
        africa.get_columns(exclude_columns = ["student_score"]),
    )
    africa_pca = model.transform()
    africa_pca.head(100)

.. ipython:: python
    :suppress:
    :okwarning:

    from verticapy.machine_learning.vertica import PCA

    model = PCA()
    model.fit(
        africa,
        africa.get_columns(exclude_columns = ["student_score"]),
    )
    africa_pca = model.transform()
    res = africa_pca.head(100)
    html_file = open("/project/data/VerticaPy/docs/figures/ug_fs_table_lr_17.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_table_lr_17.html

We can verify the Gauss-Markov assumptions with our :py:mod:`~verticapy.machine_learning.vertica.PCA` model.

.. code-block:: python

    africa_pca.corr()

.. ipython:: python
    :suppress:
    :okwarning:

    vp.set_option("plotting_lib","plotly")
    fig = africa_pca.corr()
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_fs_plot_lr_18.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_plot_lr_18.html

Let's use a cross-validation to test our linear regression model.

.. code-block:: ipython
    
    cross_validate(
        LinearRegression(solver = "BFGS"), 
        input_relation = africa_pca,
        X = africa_pca.get_columns(exclude_columns = ["student_score"]),
        y = "student_score",
    )

.. ipython:: python
    :suppress:
    :okwarning:

    res = cross_validate(
        LinearRegression(solver = "BFGS"), 
        input_relation = africa_pca,
        X = africa_pca.get_columns(exclude_columns = ["student_score"]),
        y = "student_score",
    )
    html_file = open("/project/data/VerticaPy/docs/figures/ug_fs_table_lr_19.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_table_lr_19.html

As you can see, we've created a much more accurate model here than in our first attempt. This example emphasizes the importance of filtering noise from the data.

Conclusion
-----------

We've seen two techniques that can help us create powerful linear regression models. While the first method normalized the data and looked for correlations, the second method applied a :py:mod:`~verticapy.machine_learning.vertica.PCA` model. The second one allows us to confirm the Gauss-Markov assumptions - an essential part of using linear models.