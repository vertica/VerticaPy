.. _examples.business.churn:

Telco Churn
============

This example uses the Telco Churn dataset to predict which Telco user is likely to churn; that is, customers that will likely stop using Telco. You can download the Jupyter Notebook of the study `here <https://github.com/vertica/VerticaPy/blob/master/examples/business/churn/churn.ipynb>`_.

- **Churn:** customers that left within the last month.
- **Services:** services of each customer (phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies).
- **Customer account information:** how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges.
- **Customer demographics:** gender, age range, and if they have partners and dependents.

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

Let's create a Virtual DataFrame of the dataset. The dataset is available `here <https://github.com/vertica/VerticaPy/blob/master/examples/business/churn/customers.csv>`_.

.. code-block:: ipython

    churn = vp.read_csv("customers.csv")

Let's take a look at the first few entries in the dataset.

.. code-block:: ipython
    
    churn.head(10)

.. ipython:: python
    :suppress:

    churn = vp.read_csv(
        "SPHINX_DIRECTORY/source/_static/website/examples/data/churn/customers.csv",
    )
    res = churn.head(10)
    html_file = open("SPHINX_DIRECTORY/figures/examples_churn_table.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_churn_table.html

Data Exploration and Preparation
---------------------------------

Let's examine our data.

.. code-block:: python

    churn.describe(method = "categorical", unique = True)

.. ipython:: python
    :suppress:

    res = churn.describe(method = "categorical", unique = True)
    html_file = open("SPHINX_DIRECTORY/figures/examples_churn_table_describe.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_churn_table_describe.html

Several variables are categorical, and since they all have low cardinalities, we can compute their dummies. We can also convert all booleans to numeric.

.. code-block:: python

    for column in [
        "DeviceProtection", 
        "MultipleLines",
        "PaperlessBilling",
        "Churn",
        "TechSupport",
        "Partner",
        "StreamingTV",
        "OnlineBackup",
        "Dependents",
        "OnlineSecurity",
        "PhoneService",
        "StreamingMovies",
    ]:
        churn[column].decode("Yes", 1, 0)
    churn.one_hot_encode().drop(
        [
            "customerID", 
            "gender", 
            "Contract", 
            "PaymentMethod", 
            "InternetService",
        ],
    )

.. ipython:: python
    :suppress:
    :okwarning:

    for column in [
        "DeviceProtection", 
        "MultipleLines",
        "PaperlessBilling",
        "Churn",
        "TechSupport",
        "Partner",
        "StreamingTV",
        "OnlineBackup",
        "Dependents",
        "OnlineSecurity",
        "PhoneService",
        "StreamingMovies",
    ]:
        churn[column].decode("Yes", 1, 0)
    res = churn.one_hot_encode().drop(
        [
            "customerID", 
            "gender", 
            "Contract", 
            "PaymentMethod", 
            "InternetService",
        ],
    )
    html_file = open("SPHINX_DIRECTORY/figures/examples_insurance_table_clean_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_insurance_table_clean_1.html

Let's compute the correlations between the different variables and the response column.

.. code-block:: python

    churn.corr(focus = "Churn")

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = churn.corr(focus = "Churn")
    fig.write_html("SPHINX_DIRECTORY/figures/examples_churn_corr.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_churn_corr.html

Many features have a strong correlation with the ``Churn`` variable. For example, the customers that have a ``Month to Month`` contract are more likely to churn. Having this type of contract gives customers a lot of flexibility and allows them to leave at any time. New customers are also likely to churn.

.. code-block:: python

    # No lock-in = Churn
    churn.barh(["Contract_Month-to-month", "tenure"], method = "avg", of = "Churn", height = 500)

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = churn.barh(["Contract_Month-to-month", "tenure"], method = "avg", of = "Churn", height = 500)
    fig.write_html("SPHINX_DIRECTORY/figures/examples_churn_barh.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_churn_barh.html

The following scatter plot shows that providing better tariff plans can prevent churning. Indeed, customers having high total charges are more likely to churn even if they've been with the company for a long time.

.. code-block:: python

    churn.scatter(["TotalCharges", "tenure"], by = "Churn")

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = churn.scatter(["TotalCharges", "tenure"], by = "Churn")
    fig.write_html("SPHINX_DIRECTORY/figures/examples_churn_scatter.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_churn_scatter.html

Let's move on to machine learning.

________

Machine Learning
-----------------

:py:mod:`~verticapy.machine_learning.vertica.linear_model.LogisticRegression` is a very powerful algorithm and we can use it to detect churns. Let's split our :py:mod:`~verticapy.vDataFrame` into training and testing set to evaluate our model.

.. ipython:: python

    train, test = churn.train_test_split(
        test_size = 0.2, 
        random_state = 0,
    )

Let's train and evaluate our model.

.. code-block:: python

    from verticapy.machine_learning.vertica import LogisticRegression

    model = LogisticRegression(
        penalty = "L2", 
        tol = 1e-6, 
        max_iter = 1000, 
        solver = "BFGS",
    )
    model.fit(
        train, 
        churn.get_columns(exclude_columns = ["churn"]), 
        "churn",
        test,
    )
    model.classification_report()

.. ipython:: python
    :suppress:
    :okwarning:

    from verticapy.machine_learning.vertica import LogisticRegression

    model = LogisticRegression(
        penalty = "L2", 
        tol = 1e-6, 
        max_iter = 1000, 
        solver = "BFGS",
    )
    model.fit(
        train, 
        churn.get_columns(exclude_columns = ["churn"]), 
        "churn",
        test,
    )
    res = model.classification_report()
    html_file = open("SPHINX_DIRECTORY/figures/examples_churn_table_report.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_churn_table_report.html

The model is excellent! Let's run some machine learning on the entire dataset and compute the importance of each feature.

.. code-block:: python

    model.drop()
    model.fit(
        churn, 
        churn.get_columns(exclude_columns = ["churn"]), 
        "churn",
    )
    model.features_importance()

.. ipython:: python
    :suppress:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    model.drop()
    model.fit(
        churn, 
        churn.get_columns(exclude_columns = ["churn"]), 
        "churn",
    )
    fig = model.features_importance()
    fig.write_html("SPHINX_DIRECTORY/figures/examples_churn_features_importance.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_churn_features_importance.html

Based on our model, most churning customers are at least one of the following:

- Paying higher bills
- New Telco customers
- Have a monthly contract

Notice that customers have a ``Fiber Optic`` option are also likely to churn. Let's check if this relationship is causal by computing some aggregations.

.. code-block:: python

    import verticapy.sql.functions as fun

    # Is Fiber optic a Bad Option? - VerticaPy
    churn.groupby(
        ["InternetService_Fiber_optic"], 
        [
            fun.avg(churn["tenure"])._as("tenure"),
            fun.avg(churn["totalcharges"])._as("totalcharges"),
            fun.avg(churn["contract_month-to-month"])._as("contract_month_to_month"),
            fun.avg(churn["monthlycharges"])._as("monthlycharges"),
        ]
    )

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy.sql.functions as fun

    # Is Fiber optic a Bad Option? - VerticaPy
    res = churn.groupby(
        ["InternetService_Fiber_optic"], 
        [
            fun.avg(churn["tenure"])._as("tenure"),
            fun.avg(churn["totalcharges"])._as("totalcharges"),
            fun.avg(churn["contract_month-to-month"])._as("contract_month_to_month"),
            fun.avg(churn["monthlycharges"])._as("monthlycharges"),
        ]
    )
    html_file = open("SPHINX_DIRECTORY/figures/examples_churn_table_groupby.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_churn_table_groupby.html

It seems like the ``Fiber Optic`` option in and of itself doesn't lead to churning, but customers that have this option tend to churn because their contract puts them into one of the three categories we listed before: they're paying more.

To retain these customers, we'll need to make some changes to what types of contracts we offer.

We'll use a lift chart to help us identify which of our customers are likely to churn.

.. code-block:: python

    model.lift_chart()

.. ipython:: python
    :suppress:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = model.lift_chart()
    fig.write_html("SPHINX_DIRECTORY/figures/examples_churn_lift_chart.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_churn_lift_chart.html

By targeting less than ``30%`` of the entire distribution, our predictions will be more than three times more accurate than the other ``70%``.

Conclusion
-----------

We've solved our problem in a Pandas-like way, all without ever loading data into memory!