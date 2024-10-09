.. _examples.business.insurance:

Health Insurance Costs
=======================

In this example, we use a `dataset of personal medical costs <https://www.kaggle.com/mirichoi0218/insurance>`_ to create a model to estimate treatment costs.

You can download the Jupyter notebook `here <https://github.com/vertica/VerticaPy/blob/master/examples/business/insurance/insurance.ipynb>`_.
    
The columns provided include:

- **age:** age of the primary beneficiary.
- **sex:** insurance contractor's gender.
- **bmi:** body mass index.
- **children:** number of dependent children covered by health insurance.
- **smoker:** smoker on non-smoker.
- **region:** the beneficiary's residential area in the US: northeast, southeast, southwest, northwest.
- **charges:** individual medical costs billed by health insurance.

We will follow the data science cycle (Data Exploration - Data Preparation - Data Modeling - Model Evaluation - Model Deployment) to solve this problem.

Initialization
---------------

This example uses the following version of VerticaPy:

.. ipython:: python
    
    import verticapy as vp
    
    vp.__version__

Connect to Vertica. This example uses an existing connection called "VerticaDSN". 
For details on how to create a connection, see the :ref:`connection` tutorial.
You can skip the below cell if you already have an established connection.

.. code-block:: python
    
    vp.connect("VerticaDSN")

Let's create a new schema and assign the data to a ``vDataFrame`` object.

.. code-block:: ipython

    vp.drop("insurance", method="schema")
    vp.create_schema("insurance")
    data = vp.read_csv("insurance.csv", schema = "insurance")

Let's take a look at the first few entries in the dataset.

.. code-block:: ipython
    
    data.head(5)

.. ipython:: python
    :suppress:

    vp.drop("insurance", method="schema")
    vp.create_schema("insurance")
    data = vp.read_csv(
        "/project/data/VerticaPy/docs/source/_static/website/examples/data/insurance/insurance.csv",
        schema = "insurance",
    )
    res = data.head(5)
    html_file = open("/project/data/VerticaPy/docs/figures/examples_insurance_table.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_insurance_table.html

Data Exploration
-----------------

Let's check our dataset for missing values. If we find any, we'll have to impute them before we create any models.

.. code-block:: python

    data.count_percent()

.. ipython:: python
    :suppress:

    res = data.count_percent()
    html_file = open("/project/data/VerticaPy/docs/figures/examples_insurance_table_count.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_insurance_table_count.html

There aren't missing any values, so let's get a summary of the features.

.. code-block:: python

    data.describe(method = "all")

.. ipython:: python
    :suppress:

    res = data.describe(method = "all")
    html_file = open("/project/data/VerticaPy/docs/figures/examples_insurance_table_describe.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_insurance_table_describe.html

The dataset covers 1338 individuals up to age 64 from four different regions, each with up to six dependent children.

We might find some interesting patterns if we check age distribution, so let's create a histogram.

.. code-block:: python

    data["age"].hist(method = "count", h = 1)

.. ipython:: python
    :suppress:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = data["age"].hist(method = "count", h = 1)
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_insurance_hist_age.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_insurance_hist_age.html

We have a pretty obvious trend here: the 18 and 19 year old age groups are significantly more frequent than any other, older age group. The other ages range from 20 to 30 people.

Before we do anything else, let's discretize the age column using equal-width binning with a width of 5. Our goal is to see if there are any obvious patterns among the different age groups.

.. code-block:: python

    data["age"].discretize(method = "same_width", h = 5)

.. ipython:: python
    :suppress:

    data["age"].discretize(method = "same_width", h = 5)
    res = data
    html_file = open("/project/data/VerticaPy/docs/figures/examples_insurance_descretize.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_insurance_descretize.html


Age probably influences one's body mass index (BMI), so let's compare the average of 
body mass indexes of each age group and look for patterns there. We'll use a bar graph this time.

.. code-block:: python

    data.bar(
        ["age"], 
        method = "mean",
        of = "bmi",
    )

.. ipython:: python
    :suppress:

    fig = data.bar(
        ["age"], 
        method = "mean",
        of = "bmi", 
    )
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_insurance_bar_age.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_insurance_bar_age.html

There's a pretty clear trend here, and we can say that, in general, older individuals tend to have a greater BMIs.

Let's check the average number of smokers for each age-group. Before we do, we'll convert the 'yes' and 'no' 'smoker' values to more convenient boolean values.

.. ipython:: python

    import verticapy.sql.functions as fun

    # Applying the decode function
    data["smoker_int"] = fun.decode(data["smoker"], True, 1, 0)

Now we can plot the average number of smokers for each age group.

.. code-block:: python

    data.bar(
        ["age"], 
        method = "mean",
        of = "smoker_int",
    )

.. ipython:: python
    :suppress:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = data.bar(
        ["age"], 
        method = "mean",
        of = "smoker_int",
    )
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_insurance_bar_age_smoker.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_insurance_bar_age_smoker.html

Unfortunately, there's no obvious relationship between age and smoking habits - none that we can find from this graph, anyway.

Let's see if we can relate an individual's smoking habits with their sex.

.. code-block:: python

    data.bar(
        ["sex"], 
        method = "mean",
        of = "smoker_int",
    )   

.. ipython:: python
    :suppress:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = data.bar(
        ["sex"], 
        method = "mean",
        of = "smoker_int",
    )
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_insurance_bar_sex_smoker.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_insurance_bar_sex_smoker.html

Now we're getting somewhere! Looks like we have noticeably more male smokers than female ones.

Let's see how an individual's BMI relates to their sex.

.. code-block:: python

    data.bar(
        ["sex"], 
        method = "mean",
        of = "bmi",
    ) 

.. ipython:: python
    :suppress:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = data.bar(
        ["sex"], 
        method = "mean",
        of = "bmi",
    )
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_insurance_bar_sex_bmi.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_insurance_bar_sex_bmi.html

Males seem to have a slightly higher BMI, but it'd be hard to draw any conclusions from such a small difference.

Going back to our earlier patterns, let's check the distribution of sexes among age groups and see if the 
patterns we identified earlier skews toward one of the sexes.

.. code-block:: python

    data.pivot_table(["age", "sex"])

.. ipython:: python
    :suppress:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = data.pivot_table(["age", "sex"])
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_insurance_corr_age_sex.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_insurance_corr_age_sex.html

It seems that sex is pretty evenly distributed in each age group.

Let's move onto costs: how much do people tend to spend on medical treatments?

.. code-block:: python

    data["charges"].hist(method = "count")

.. ipython:: python
    :suppress:

    fig = data["charges"].hist(method = "count")
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_insurance_charges_hist.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_insurance_charges_hist.html

Based on this graph, the majority of insurance holders tend to spend less than 1500 and only a handful of people spend more than 5000.

Encoding
---------

Since our features vary in type, let's start by encoding our categorical features. 
Remember, we label-encoded 'smoker' from boolean. Let's label-encode some other features: sex, region, and age groups.

.. code-block:: python

    # encoding sex 
    data["sex"].label_encode()

    # encoding region
    data["region"].label_encode()

    # encoding age
    data["age"].label_encode()


.. ipython:: python
    :suppress:

    # encoding sex 
    data["sex"].label_encode()

    # encoding region
    data["region"].label_encode()

    # encoding age
    data["age"].label_encode()
    res = data
    html_file = open("/project/data/VerticaPy/docs/figures/examples_insurance_table_encoded_new.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_insurance_table_encoded_new.html

Before going further, let's check the correlation of the variables with the predictor 'charges'.

.. code-block:: python

    data.corr(focus = "charges")

.. ipython:: python
    :suppress:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = data.corr(focus = "charges")
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_insurance_charges_focus.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_insurance_charges_focus.html

.. code-block:: python

    data.to_db("insurance.final_ins_data", relation_type = "table")

________

Predicting insurance charges
-----------------------------

Since our response variable is continuous, we can use regression to predict it. 
For this example, let's use a ``Random Forest`` model.

.. ipython:: python
    :okwarning:

    from verticapy.machine_learning.vertica.ensemble import RandomForestRegressor

    # define the random forest model
    rf_model = RandomForestRegressor(
        n_estimators = 20,
        max_features = "auto",
        max_leaf_nodes = 32, 
        sample = 0.7,
        max_depth = 3,
        min_samples_leaf = 5,
        min_info_gain = 0.0,
        nbins = 32,
    )

    # train the model
    rf_model.fit(
        data,
        X = ["age", "sex", "bmi", "children", "smoker", "region"], 
        y = "charges",
    )

We can create a regression report to check our model's performance.

.. code-block:: python

    rf_model.report()

.. ipython:: python
    :suppress:
    :okwarning:

    res = rf_model.report()
    html_file = open("/project/data/VerticaPy/docs/figures/examples_insurance_table_report.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_insurance_table_report.html

The results seem to be quite good! We have an explained variance around 0.8. 
Let's plot the predicted values and compare them to the real ones.

.. code-block:: python

    # plot the predicted values and real ones
    result = rf_model.predict(
        data, 
        name = "pred_charges",
    )

    # add an index
    result["id"] = "ROW_NUMBER() OVER()"

    # plot them along the id
    result.plot(
        ts = "id",
        columns = ['charges', 'pred_charges'],
    )

.. ipython:: python
    :suppress:

    result = rf_model.predict(
        data, 
        name = "pred_charges",
    )
    result["id"] = "ROW_NUMBER() OVER()"
    fig = result.plot(
        ts = "id",
        columns = ["charges", "pred_charges"]
    )
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_insurance_rf_plot.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_insurance_rf_plot.html

.. code-block:: python

    data.to_db("insurance.final_ins_data", relation_type = "table")

Now, let's examine the importance of each feature for this model. 
Ours is a random forest model, so we can use the built-in Vertica function ``RF_PREDICTOR_IMPORTANCE()`` to calculate the importance of each predictor with Mean Decrease in Impurity (MDI).

.. code-block:: python

    # feature importance for our random forest model
    rf_model.features_importance()

.. ipython:: python
    :suppress:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    # feature importance for our random forest model
    fig = rf_model.features_importance()
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_insurance_rf_feature_importance.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_insurance_rf_feature_importance.html

.. code-block:: python

    data.to_db("insurance.final_ins_data", relation_type = "table")

.. code-block:: python

    rf_model.features_importance(show = False)

.. ipython:: python
    :suppress:

    res = rf_model.features_importance(show = False)
    html_file = open("/project/data/VerticaPy/docs/figures/examples_insurance_table_feature_importance_rf.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_insurance_table_feature_importance_rf.html

We can examine how our model works by visualizing one of the trees in our ``Random Forest``.

.. code-block::

    # plot one of the trees comprising the forest
    rf_model.plot_tree(tree_id = 3)

.. ipython:: python
    :suppress:

    res = rf_model.plot_tree(tree_id = 3)
    res.render(filename="figures/examples_insurance_table_rf_tree", format="png")

.. image:: /../figures/examples_insurance_table_rf_tree.png

What affects medical costs?
----------------------------

We have a couple ways to approach this question. First, let's see what features are linearly correlated with the cost.

It seems that smoking habits have a significant effect on medical costs. Next in line comes BMI, the number of dependents, and sex.

As one might expect, the correlation between charges and region is almost 0.

Now, let's see what we can learn from a stepwise model with forward elimination using Bayesian 
information criterion (BIC) as a selection criteria.

.. code-block:: python

    from verticapy.machine_learning.vertica.linear_model import LinearRegression

    model = LinearRegression()

    # backward
    from verticapy.machine_learning.model_selection import stepwise

    stepwise(
        model,
        input_relation = data, 
        direction = "forward",
        X = ["age","sex", "bmi", "children", "smoker", "region"], 
        y = "charges",
    )


.. ipython:: python
    :suppress:

    from verticapy.machine_learning.vertica.linear_model import LinearRegression

    model = LinearRegression()

    # backward
    from verticapy.machine_learning.model_selection import stepwise

    res = stepwise(
        model,
        input_relation = data, 
        direction = "forward",
        X = ["age","sex", "bmi", "children", "smoker", "region"], 
        y = "charges",
    )
    html_file = open("/project/data/VerticaPy/docs/figures/examples_insurance_lr_stepwise.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_insurance_lr_stepwise.html

From here we see that, again, the same features have similarly significant effects on medical costs.

Conclusion
------------

In this example, we used several methods to identify the primary factors that affect one's insurance costs.