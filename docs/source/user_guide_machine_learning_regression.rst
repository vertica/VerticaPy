.. _user_guide.machine_learning.regression:

===========
Regression
===========

Regressions are machine learning algorithms used to predict numerical response columns. Predicting the salaries of employees using their age or predicting the number of cyber attacks a website might face would be examples of regressions. The most popular regression algorithm is the linear regression.

You must always verify that all the assumptions of a given algorithm are met before using them. For example, to create a good linear regression model, we need to verify the Gauss-Markov assumptions.

- **Linearity:** the parameters we are estimating using the OLS method must be linear.
- **Non-Collinearity:** the regressors being calculated aren’t perfectly correlated with each other.
- **Exogeneity:** the regressors aren’t correlated with the error term.
- **Homoscedasticity:** no matter what the values of our regressors might be, the error of the variance is constant.

Most of regression models are sensitive to unnormalized data, so it's important to normalize and decompose your data before using them (though some models like random forest can handle unnormalized and correlated data). If we don't follow the assumptions, we might get unexpected results (example: negative R2).

Let's predict the total charges of the Telco customers using their tenure. We will start by importing `the telco dataset <https://github.com/vertica/VerticaPy/blob/master/examples/business/churn/customers.csv>`_.

.. code-block:: ipython
    
    import verticapy as vp

    churn = vp.read_csv("customers.csv")

.. ipython:: python
    :suppress:

    import verticapy as vp

    churn = vp.read_csv(
        "/project/data/VerticaPy/docs/source/_static/website/examples/data/churn/customers.csv",
    )

Next, we can import a linear regression model.

.. ipython:: python
    
    from verticapy.machine_learning.vertica import LinearRegression

Let's create a model object.

.. ipython:: python
    
    model = LinearRegression()

We can then fit the model with our data.

.. ipython:: python
    
    model.fit(churn, ["tenure"], "TotalCharges")

.. code-block:: python

    model.plot()

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = model.plot()
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_ml_plot_regression_1.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_ml_plot_regression_1.html

We have many metrics to evaluate the model.

.. code-block::

    model.report()

.. ipython:: python
    :suppress:
    :okwarning:

    res = model.report()
    html_file = open("/project/data/VerticaPy/docs/figures/ug_ml_table_regression_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_ml_table_regression_2.html

Our example forgoes splitting the data into training and testing, which is important for real-world work. Our main goal in this lesson is to look at the metrics used to evaluate regressions. The most famous metric is ``R2``: generally speaking, the closer ``R2`` is to 1, the better the model is.

In the next lesson, we'll go over classification models.