.. _user_guide.data_exploration.correlation:

===========================
Correlation and Dependency
===========================

Finding links between variables is a very important task. The main purpose of data science is to find relationships between variables, and to understand how these relationships can help us make better decisions.

Machine learning models are also sensitive to the number of variables and how they relate and affect each other, so finding correlations and dependencies can help us make better use of our machine learning algorithms.

Let's use the `Telco Churn dataset <https://github.com/vertica/VerticaPy/blob/master/examples/business/churn/customers.csv>`_ to understand how we can find links between different variables in VerticaPy.

.. code-block:: ipython
    
    import verticapy as vp

    churn = vp.read_csv("customers.csv")
    churn.head(100)

.. ipython:: python
    :suppress:

    import verticapy as vp
    churn = vp.read_csv(
        "/project/data/VerticaPy/docs/source/_static/website/examples/data/churn/customers.csv",
    )
    res = churn.head(100)
    html_file = open("/project/data/VerticaPy/docs/figures/ug_de_table_corr_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_de_table_corr_1.html

The Pearson correlation coefficient is a very common correlation function. In this case, it helped us to find linear links between the variables. Having a strong Pearson relationship means that the two input variables are linearly correlated.

.. code-block:: python

    churn.corr(method = "pearson")

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = churn.corr(method = "pearson")
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_de_plot_corr_2.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_de_plot_corr_2.html

We can see that 'tenure' is well-correlated to the 'TotalCharges', which makes sense.

.. code-block:: python

    churn.scatter(["tenure", "TotalCharges"])

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = churn.scatter(["tenure", "TotalCharges"])
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_de_plot_corr_3.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_de_plot_corr_3.html

.. ipython:: python

    churn.corr(["tenure", "TotalCharges"], method = "pearson")

Note, however, that having a low Pearson relationship imply that the variables aren't correlated. For example, let's compute the Pearson correlation coefficient between 'tenure' and 'TotalCharges' to the power of 20.

.. ipython:: python

    churn["TotalCharges^20"] = churn["TotalCharges"] ** 20
    churn.corr(["tenure", "TotalCharges^20"], method = "pearson")

We know that the 'tenure' and 'TotalCharges' are strongly linearly correlated. However we can notice that the correlation between the 'tenure' and 'TotalCharges' to the power of 20 is not very high. Indeed, the Pearson correlation coefficient is not robust for monotonic relationships, but rank-based correlations are. Knowing this, we'll calculate the Spearman's rank correlation coefficient instead.

.. code-block:: ipython
    
    churn.corr(method = "spearman", show = False)

.. ipython:: python
    :suppress:

    res = churn.corr(method = "spearman", show = False)
    html_file = open("/project/data/VerticaPy/docs/figures/ug_de_table_corr_4.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_de_table_corr_4.html

.. code-block:: python

    churn.corr(method = "spearman")

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = churn.corr(method = "spearman")
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_de_plot_corr_5.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_de_plot_corr_5.html

The Spearman's rank correlation coefficient determines the monotonic relationships between the variables.

.. ipython:: python

    churn.corr(["tenure", "TotalCharges^20"], method = "spearman")

We can notice that Spearman's rank correlation coefficient stays the same if one of the variables can be expressed using a monotonic function on the other. The same applies to Kendall rank correlation coefficient.

.. code-block:: python

    churn.corr(method = "kendall")

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = churn.corr(method = "kendall")
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_de_plot_corr_6.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_de_plot_corr_6.html

Notice that the Kendall rank correlation coefficient will also detect the monotonic relationship.

.. ipython:: python

    churn.corr(["tenure", "TotalCharges^20"], method = "kendall")

However, the Kendall rank correlation coefficient is very computationally expensive, so we'll generally use Pearson and Spearman when dealing with correlations between numerical variables.

Binary features are considered numerical, but this isn't technically accurate. Since binary variables can only take two values, calculating correlations between a binary and numerical variable can lead to misleading results. To account for this, we'll want to use the 'Biserial Point' method to calculate the Point-Biserial correlation coefficient. This powerful method will help us understand the link between a binary variable and a numerical variable.

.. code-block:: python

    churn.corr(method = "biserial")

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = churn.corr(method = "biserial")
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_de_plot_corr_7.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_de_plot_corr_7.html

Lastly, we'll look at the relationship between categorical columns. In this case, the 'Cramer's V' method is very efficient. Since there is no position in the Euclidean space for those variables, the 'Cramer's V' coefficients cannot be negative (which is a sign of an opposite relationship) and they will range in the interval `[0,1]`.

.. code-block:: python

    churn.corr(method = "cramer")

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = churn.corr(method = "cramer")
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_de_plot_corr_8.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_de_plot_corr_8.html

Sometimes, we just need to look at the correlation between a response and other variables. The parameter `focus` will isolate and show us the specified correlation vector.

.. code-block:: python

    churn.corr(method = "cramer", focus = "Churn")

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = churn.corr(method = "cramer", focus = "Churn")
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_de_plot_corr_9.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_de_plot_corr_9.html

Sometimes a correlation coefficient can lead to incorrect assumptions, so we should always look at the coefficient `p-value`.

.. ipython:: python

    churn.corr_pvalue("Churn", "customerID", method = "cramer")

We can see that churning correlates to the type of contract (monthly, yearly, etc.) which makes sense: you would expect that different types of contracts differ in flexibility for the customer, and particularly restrictive contracts may make churning more likely.

The type of internet service also seems to correlate with churning. Let's split the different categories to binaries to understand which services can influence the global churning rate.

.. code-block:: python

    churn["InternetService"].one_hot_encode()
    churn.corr(
        method = "spearman", 
        focus = "Churn", 
        columns = [
            "InternetService_DSL", 
            "InternetService_Fiber_optic",
        ],
    )

.. ipython:: python
    :suppress:
    :okwarning:

    churn["InternetService"].one_hot_encode()
    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = churn.corr(
        method = "spearman", 
        focus = "Churn", 
        columns = [
            "InternetService_DSL", 
            "InternetService_Fiber_optic",
        ],
    )
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_de_plot_corr_10.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_de_plot_corr_10.html

We can see that the Fiber Optic option in particular seems to be directly linked to a customer's likelihood to churn. Let's compute some aggregations to find a causal relationship.

.. code-block:: ipython
    
    churn["contract"].one_hot_encode()
    churn.groupby(
        [
            "InternetService_Fiber_optic",
        ], 
        [
            "AVG(tenure) AS tenure", 
            "AVG(totalcharges) AS totalcharges",
            'AVG("contract_month-to-month") AS "contract_month-to-month"',
            'AVG("monthlycharges") AS "monthlycharges"',
        ],
    )

.. ipython:: python
    :suppress:

    churn["contract"].one_hot_encode()
    res = churn.groupby(
        [
            "InternetService_Fiber_optic",
        ], 
        [
            "AVG(tenure) AS tenure", 
            "AVG(totalcharges) AS totalcharges",
            'AVG("contract_month-to-month") AS "contract_month-to-month"',
            'AVG("monthlycharges") AS "monthlycharges"',
        ],
    )
    html_file = open("/project/data/VerticaPy/docs/figures/ug_de_table_corr_11.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_de_table_corr_11.html

It seems that users with the Fiber Optic option tend more to churn not because of the option itself, but probably because of the type of contracts and the monthly charges the users are paying to get it. Be careful when dealing with identifying correlations! Remember: correlation doesn't imply causation!

Another important type of correlation is the autocorrelation. Let's use the Amazon dataset to understand it.

.. code-block:: ipython
    
    from verticapy.datasets import load_amazon

    amazon = load_amazon()
    amazon.head(100)

.. ipython:: python
    :suppress:

    from verticapy.datasets import load_amazon

    amazon = load_amazon()
    res = amazon.head(100)
    html_file = open("/project/data/VerticaPy/docs/figures/ug_de_table_corr_12.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_de_table_corr_12.html

Our goal is to predict the number of forest fires in Brazil. To do this, we can draw an autocorrelation plot and a partial autocorrelation plot.

.. code-block:: python

    amazon.acf(
        column = "number",
        ts = "date",
        by = ["state"],
        p = 24,
        method = "pearson",
    )

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = amazon.acf(
        column = "number",
        ts = "date",
        by = ["state"],
        p = 24,
        method = "pearson",
    )
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_de_plot_corr_13.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_de_plot_corr_13.html

.. code-block:: python

    amazon.pacf(
        column = "number",
        ts = "date",
        by = ["state"],
        p = 8,
    )

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = amazon.pacf(
        column = "number",
        ts = "date",
        by = ["state"],
        p = 8,
    )
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_de_plot_corr_14.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_de_plot_corr_14.html

We can see the seasonality forest fires.

It's mathematically impossible to build the perfect correlation function, but we still have several powerful functions at our disposal for finding relationships in all kinds of datasets.