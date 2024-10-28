.. _examples.business.credit_card_fraud:

Credit Card Fraud
==================

In this example, we use VerticaPy to detect fraudulent credit card transactions. You can download the Jupyter notebook `here <https://github.com/vertica/VerticaPy/blob/master/examples/understand/business/credit_card_fraud/credit-card-fraud.ipynb>`_ and the dataset `here <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>`_.

The Credit Card Fraud Detection dataset contains credit card transactions from September 2013 by European cardholders. It contains numerical input variables from a principal component analysis (:py:mod:`~verticapy.machine_learning.vertica.decomposition.PCA`) transformation.

To preserve the cardholders' confidentiality, we cannot access the original features and background information about the data.

``Time`` and ``Amount`` are the only features that have not been transformed with :py:mod:`~verticapy.machine_learning.vertica.decomposition.PCA`.

- **V1, V2,..., V28:** principal components from :py:mod:`~verticapy.machine_learning.vertica.decomposition.PCA`.
- **Time:** Number of seconds elapsed between this transaction and the first transaction in the dataset.
- **Amount:** Transaction amount.
- **Class:** Response variable, where a value of 1 indicates fraudulent activity.

``Amount`` will be useful for example-dependent cost-sensitive learning.

We will follow the entire Data Science cycle (Data Exploration, Data Preparation, Data Modeling, Model Evaluation, Model Deployment) to solve this problem.

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

.. code-block:: python

    creditcard = vp.read_csv(
        "creditcard.csv", 
        parse_nrows = 1000,
    )
    creditcard.head(5)

.. ipython:: python
    :suppress:

    creditcard = vp.read_csv("SPHINX_DIRECTORY/source/_static/website/examples/data/credit_card_fraud/creditcard.csv")
    res = creditcard.head(5)
    html_file = open("SPHINX_DIRECTORY/figures/examples_creditcardfraud_table_head.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_creditcardfraud_table_head.html

.. warning::
    
    This example uses a sample dataset. For the full analysis, you should consider using the complete dataset.

Data Exploration and Preparation
---------------------------------

Let's explore the data by displaying descriptive statistics of all the columns.

.. code-block:: python

    creditcard.describe()

.. ipython:: python
    :suppress:

    res = creditcard.describe()
    html_file = open("SPHINX_DIRECTORY/figures/examples_creditcardfraud_describe.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_creditcardfraud_describe.html

It'll be difficult to work on the principal components (V1 through V28) without knowing what they mean. The only features we can work on are ``Time`` and ``Amount``.

Let's convert the number of seconds elapsed to the correct date and time. We know that the records were ingested in September 2013, so we'll use that to create the new feature.

.. code-block:: python

    creditcard["Time"].apply("TIMESTAMPADD(second, {}::int, '2013-09-01 00:00:00'::timestamp)")

.. ipython:: python
    :suppress:

    res = creditcard["Time"].apply("TIMESTAMPADD(second, {}::int, '2013-09-01 00:00:00'::timestamp)")
    html_file = open("SPHINX_DIRECTORY/figures/examples_creditcardfraud_apply.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_creditcardfraud_apply.html

When performing machine learning, we'll take the data from two days and split it into a training set (first day) and a test set (second day).

.. code-block:: python

    creditcard["Time"].describe()

.. ipython:: python
    :suppress:

    res = creditcard["Time"].describe()
    html_file = open("SPHINX_DIRECTORY/figures/examples_creditcardfraud_describe_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_creditcardfraud_describe_2.html

Fraudulent activity probably isn't uniform across all hours of the day, so we'll extract the hour from the time and see how that influences the prediction.

.. code-block:: python

    import verticapy.sql.functions as fun

    creditcard["hour"] = fun.hour(creditcard["Time"])
    creditcard[["Time", "hour"]]

.. ipython:: python
    :suppress:

    import verticapy.sql.functions as fun

    creditcard["hour"] = fun.hour(creditcard["Time"])
    res = creditcard[["Time", "hour"]]
    html_file = open("SPHINX_DIRECTORY/figures/examples_creditcardfraud_sample_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_creditcardfraud_sample_1.html

We can visualize the frequency of fraudulent transactions throughout the day with a histogram.

.. code-block:: python

    creditcard["hour"].hist(method = "avg", of = "Class")

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = creditcard["hour"].hist(method = "avg", of = "Class")
    fig.write_html("SPHINX_DIRECTORY/figures/examples_creditcardfraud_hist.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_creditcardfraud_hist.html

It seems like most fraudulent activity happens at night.

The transaction amount also likely differs between fraudulent and genuine transactions, so we'll look at that relationship with a bar chart. Notice that fraudulent transactions tend to be larger purchases.

.. code-block:: python

    creditcard["Class"].bar(
        method = "avg", 
        of = "Amount",
    )

.. ipython:: python
    :suppress:
    :okwarning:

    fig = creditcard["Class"].bar(
        method = "avg", 
        of = "Amount",
    )
    fig.write_html("SPHINX_DIRECTORY/figures/examples_creditcardfraud_bar.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_creditcardfraud_bar.html

Let's create some new features and move forward from there.

Features Engineering
---------------------

Since all data (besides ``Time`` and ``Amount``) are encoded, we're somewhat limited in creating features.
One way to work with this limitation for time series is with moving windows.

In lieu of customer IDs, we'll aggregate on the transaction amount over some partitions. Let's compute some features to analyze the transaction amount and frequencies across different windows: 5 hours preceding, 5 minutes preceding, and 5 seconds preceding. Choosing these windows is pretty subjective, but we can close in on the most relevant windows after some more extensive testing.

.. code-block:: python

    creditcard.rolling(
        name = "nb_same_transactions_mn_5h", 
        func = "COUNT", 
        columns = "Amount",
        window = ("- 5 hours", "0 hour"),
        by = ["Amount"],
        order_by = ["Time"],
    )
    creditcard.rolling(
        name = "nb_same_transactions_mn_5m", 
        func = "COUNT", 
        columns = "Amount",
        window = ("- 5 minutes", "0 minute"),
        by = ["Amount"],
        order_by = ["Time"],
    )
    creditcard.rolling(
        name = "nb_same_transactions_mn_5s", 
        func = "COUNT", 
        columns = "Amount",
        window = ("- 5 seconds", "0 second"),
        by = ["Amount"],
        order_by = ["Time"],
    )

.. ipython:: python
    :suppress:

    creditcard.rolling(
        name = "nb_same_transactions_mn_5h", 
        func = "COUNT", 
        columns = "Amount",
        window = ("- 5 hours", "0 hour"),
        by = ["Amount"],
        order_by = ["Time"],
    )
    creditcard.rolling(
        name = "nb_same_transactions_mn_5m", 
        func = "COUNT", 
        columns = "Amount",
        window = ("- 5 minutes", "0 minute"),
        by = ["Amount"],
        order_by = ["Time"],
    )
    res = creditcard.rolling(
        name = "nb_same_transactions_mn_5s", 
        func = "COUNT", 
        columns = "Amount",
        window = ("- 5 seconds", "0 second"),
        by = ["Amount"],
        order_by = ["Time"],
    )
    html_file = open("SPHINX_DIRECTORY/figures/examples_creditcardfraud_rolling.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_creditcardfraud_rolling.html

As an aside, we could also create some features that represent different parts of the day, but won't be useful for our use case since we're only working with data for two days' worth of data.

Let's look at the correlation matrix and see which features influence our prediction.

.. code-block:: python

    creditcard.corr()

.. ipython:: python
    :suppress:
    :okwarning:

    fig = creditcard.corr(width = 850, with_numbers = False)
    fig.write_html("SPHINX_DIRECTORY/figures/examples_creditcardfraud_corr_2.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_creditcardfraud_corr_2.html

Our new features aren't linearly correlated with our response, but some of the components seem to have a large influence on our prediction. We'll use these when we create our model.

To simplify things, let's save the dataset into a new table.

.. code-block:: python

    vp.drop(
        "creditcard_clean",
        method = "table",
    )
    creditcard.to_db(
        "creditcard_clean", 
        relation_type = "table",
        inplace = True,
    )

.. ipython:: python
    :suppress:

    vp.drop(
        "creditcard_clean",
        method = "table",
    )
    res = creditcard.to_db(
        "creditcard_clean", 
        relation_type = "table",
        inplace = True,
    )
    html_file = open("SPHINX_DIRECTORY/figures/examples_creditcardfraud_to_db.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_creditcardfraud_to_db.html

Data Modeling
--------------

Train/Test sets
++++++++++++++++

Since we're dealing with time series data, we have to maintain time linearity. Our goal is to use the past to predict the future, so a k-fold cross-validation, for example, wouldn't make much sense here.

We will split the dataset into a train (day 1) and a test (day 2).

.. ipython:: python

    train = creditcard.search("Time  < '2013-09-02 00:00:00'")
    test  = creditcard.search("Time >= '2013-09-02 00:00:00'")

Supervision
++++++++++++

Supervising would make this pretty easy since it would just be a binary classification problem. We can use different algorithms to optimize the prediction. Our dataset is unbalanced, so the AUC might be a good metric to evaluate the model. The PRC AUC would also be a relevant metric.

:py:mod:`~verticapy.machine_learning.vertica.linear_model.LogisticRegression` works well with monotonic relationships. Since we have a lot of independent features that correlate with the response, it should be a good first model to use.

.. code-block:: python

    from verticapy.machine_learning.vertica import LogisticRegression

    predictors = creditcard.get_columns(exclude_columns = ["Class", "Time"])
    response = "Class"
    model = LogisticRegression(
        penalty = 'L2',
        tol = 1e-6, 
        max_iter = 1000,
        solver = "BFGS",
    )
    model.fit(train, predictors, response, test)
    model.classification_report()

.. ipython:: python
    :suppress:

    from verticapy.machine_learning.vertica import LogisticRegression

    predictors = creditcard.get_columns(exclude_columns = ["Class", "Time"])
    response = "Class"
    model = LogisticRegression(
        penalty = 'L2',
        tol = 1e-6, 
        max_iter = 1000,
        solver = "BFGS",
    )
    model.fit(train, predictors, response, test)
    res = model.classification_report()
    html_file = open("SPHINX_DIRECTORY/figures/examples_creditcardfraud_classification_report.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_creditcardfraud_classification_report.html

Based on the report, our model is very good at detecting non-fraudulent events; the AUC is high and the PRC AUC is very good. We can use this model to filter obvious events and to get some insight on the importance of each feature.

.. code-block:: python

    model.features_importance()

.. ipython:: python
    :suppress:
    :okwarning:

    fig = model.features_importance()
    fig.write_html("SPHINX_DIRECTORY/figures/examples_creditcardfraud_features_importance_1.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_creditcardfraud_features_importance_1.html

Some :py:mod:`~verticapy.machine_learning.vertica.decomposition.PCA` components seem to be very relevant and will be essential for finding anomalies.

Unsupervised Learning
++++++++++++++++++++++

There are many unsupervised learning techniques, but not all of them will be useful for detecting anomalies. Since there's no rigid mathematical definition for what an outlier is, finding anomalies becomes somewhat subjective.
To solve this problem, we have to evaluate our constraints and needs. Do we need to find anomalies in real-time? Do we have a time constraint?

- **Real-time:** We don't have access to historical data, so we need an easy way to preprocess the data that is wholly independent from historical data, and the model must be simple to deploy at the source of the data stream. For example, we might use simple preprocessing techniques like normalization, standardization or One-Hot Encoding instead of more complex ones like windows, interpolation, or intersection. Isolation forests, k-means, robust :py:mod:`~verticapy.machine_learning.vertica.decomposition.PCA`, or global outlier detection using z-score would be ideal, whereas local outlier factor, DBSCAN, or other hard-to-deploy methods cannot be used.
- **Near Real-time:** We have access to historical data and our preprocessing method must be fast. The model has to be simple to score with. We can use any preprocessing technique as long as it is fast enough, which of course varies. Since this is still a real-time use case, we should still avoid any hard-to-deploy algorithms like DBSCAN or local outlier factor.
- **No time constraint:** We can use any techniques we want.

Due to the complexity of the computations, anomalies are difficult to detect in the context of "Big Data." We have three efficient methods for that case:

- **Machine Learning:** We need to use easily-deployable algorithms to perform real-time fraud detection. Isolation forests and ``k-means`` can be easily deployed and they work well for detecting anomalies.
- **Rules & Thresholds:** The z-score can be an efficient solution for detecting global outliers.
- **Decomposition:** Robust :py:mod:`~verticapy.machine_learning.vertica.decomposition.PCA` is another technique for detecting outliers.

Before using these techniques, let's draw some scatter plots to get a better idea of what kind of anomalies we can expect.

.. code-block:: python

    creditcard.scatter(
        ["V12", "V17"], 
        by = "Class", 
        max_nb_points = 5000000,
    )

.. ipython:: python
    :suppress:
    :okwarning:
    

    fig = creditcard.scatter(
        ["V12", "V17"], 
        by = "Class", 
        max_nb_points = 5000000,
    )
    fig.write_html("SPHINX_DIRECTORY/figures/examples_creditcardfraud_ml_scatter_1.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_creditcardfraud_ml_scatter_1.html

.. code-block:: python

    creditcard.scatter(
        ["V12", "V17", "V10"], 
        by = "Class",
    )

.. ipython:: python
    :suppress:
    :okwarning:

    fig = creditcard.scatter(
        ["V12", "V17", "V10"], 
        by = "Class",
    )
    fig.write_html("SPHINX_DIRECTORY/figures/examples_creditcardfraud_ml_scatter_2.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_creditcardfraud_ml_scatter_2.html

In this case, the anomalies seem pretty clear global outliers of the distributions. When doing unsupervised learning, we don't have this information in advance.

For the rest of this example, we'll investigate labels and how they can help us understand the efficacy of each technique.

k-means Clustering
+++++++++++++++++++

We begin by examining ``k-means`` clustering, which partitions the data into k clusters.

We can use an elbow curve to find a suitable number of clusters. We can then add more clusters then the amount suggested by the :py:func:`~verticapy.machine_learning.model_selection.elbow` curve to create clusters mainly composed of anomalies. Clusters with relatively fewer elements can then be investigated by an expert to label the anomalies.

From there, we perform the following procedure:

- Label historical data by looking at unsupervised learning results.
- Use supervised learning models to learn on the labeled anomalies. This model will be brought to the source of the data stream.

Once we deploy the unsupervised model and can reliably detect suspicious transactions, we could block them and contact the cardholder about potential fraudulent activity on their card.

.. code-block:: python

    from verticapy.machine_learning.model_selection import elbow

    elbow(
        creditcard,
        ["V12", "V17", "V10", "V14", "V16"], 
        n_cluster = [1, 2, 10, 20, 30],
    )

.. ipython:: python
    :suppress:
    :okwarning:

    from verticapy.machine_learning.model_selection import elbow
    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = elbow(
        creditcard,
        ["V12", "V17", "V10", "V14", "V16"], 
        n_cluster = [1, 2, 10, 20, 30],
    )
    fig.write_html("SPHINX_DIRECTORY/figures/examples_creditcardfraud_ml_elbow.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_creditcardfraud_ml_elbow.html

10 seems to be a suitable number of clusters, so let's try out 20 clusters and see if the collective outliers cluster together. We can then then evaluate each cluster independently and see which clusters have the most anomalies.

.. ipython:: python

    from verticapy.machine_learning.vertica import KMeans

    model = KMeans(n_cluster = 20)
    model.fit(creditcard, ["V12", "V17", "V10"])

Let's direct our attention to the smallest clusters.

.. code-block:: python

    model.predict(creditcard, name = "cluster")
    creditcard.groupby(
        ["cluster"],
        [
            "COUNT(*) AS total", 
            "100 * AVG(Class) AS percent_fraud",
            "SUM(Class) / 492 AS total_fraud",
        ],
    ).sort("total")

.. ipython:: python
    :suppress:

    model.predict(creditcard, name = "cluster")
    res = creditcard.groupby(
        ["cluster"],
        [
            "COUNT(*) AS total", 
            "100 * AVG(Class) AS percent_fraud",
            "SUM(Class) / 492 AS total_fraud",
        ],
    ).sort("total")
    html_file = open("SPHINX_DIRECTORY/figures/examples_creditcardfraud_groupby_ml.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_creditcardfraud_groupby_ml.html

Notice that clusters with fewer elemenets tend to contain much more fraudulent events than the others. This methodology makes ``k-means`` a good algorithm for catching collective outliers. Combining ``k-means`` with other techniques like z-score, we can find most of the outliers of the distribution.

Outliers of the distribution
+++++++++++++++++++++++++++++

Let's use the ``Z-score`` to detect global outliers of the distribution.

.. code-block:: python

    creditcard.outliers(
        ["V12", "V17", "V10"], 
        name = "global_outliers", 
        threshold = 5.0,
    )
    creditcard.groupby(
        ["global_outliers"],
        [
            "COUNT(*) AS total", 
            "100 * AVG(Class) AS percent_fraud",
            "SUM(Class) / 492 AS total_fraud",
        ],
    ).sort("total")

.. ipython:: python
    :suppress:

    creditcard.outliers(
        ["V12", "V17", "V10"], 
        name = "global_outliers", 
        threshold = 5.0,
    )
    res = creditcard.groupby(
        ["global_outliers"],
        [
            "COUNT(*) AS total", 
            "100 * AVG(Class) AS percent_fraud",
            "SUM(Class) / 492 AS total_fraud",
        ],
    ).sort("total")
    html_file = open("SPHINX_DIRECTORY/figures/examples_creditcardfraud_groupby_2_ml.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_creditcardfraud_groupby_2_ml.html

.. code-block:: python

    creditcard.outliers_plot(
        ["V12", "V17",],
        threshold = 5.0,
    )

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = creditcard.outliers_plot(
        ["V12", "V17",],
        threshold = 5.0,
    )
    fig.write_html("SPHINX_DIRECTORY/figures/examples_creditcardfraud_ml_outliers_plot_3.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_creditcardfraud_ml_outliers_plot_3.html

We can see that we can caught more than 71% of the fraudulent activity in less than 1% of the dataset.

Neighbors
++++++++++

Other algorithms could be used to solve the problem with more precision if we could use a more powerful clustering method and didn't have a time constraint. Based on neighbors, these algorithms are very computationally expensive. An example of this kind of algorithm is the local outlier factor.

.. code-block:: python

    from verticapy.machine_learning.vertica import LocalOutlierFactor

    model = LocalOutlierFactor()
    model.fit(creditcard.sample(x = 0.01), ["V12", "V17", "V10"])
    lof_creditcard = model.predict()
    lof_creditcard["outliers"] = "(CASE WHEN lof_score > 2 THEN 1 ELSE 0 END)"
    lof_creditcard.scatter(["V12", "V17", "V10"], by = "outliers")

.. ipython:: python
    :suppress:
    :okwarning:

    from verticapy.machine_learning.vertica import LocalOutlierFactor
    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    model = LocalOutlierFactor()
    model.fit(creditcard.sample(x = 0.01), ["V12", "V17", "V10"])
    lof_creditcard = model.predict()
    lof_creditcard["outliers"] = "(CASE WHEN lof_score > 2 THEN 1 ELSE 0 END)"
    fig = lof_creditcard.scatter(["V12", "V17", "V10"], by = "outliers")
    fig.write_html("SPHINX_DIRECTORY/figures/examples_creditcardfraud_ml_lof_plot_1.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_creditcardfraud_ml_lof_plot_1.html

We can catch outliers with a neighbors score. Again, the main problem with these sorts of algorithms is that what they have in precision, they lack in speed, which makes them unsuitable for scoring new data. This is why it's important to focus on scalable techniques like ``k-means``.

Other Techniques
+++++++++++++++++

Other scalable techniques that can solve this problem are robust :py:mod:`~verticapy.machine_learning.vertica.decomposition.PCA` and isolation forest.

Conclusion
-----------

We've solved our problem in a Pandas-like way, all without ever loading data into memory!