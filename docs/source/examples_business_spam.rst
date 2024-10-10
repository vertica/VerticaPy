.. _examples.business.spam:

Spam
=====

This example uses the 'Spam' dataset to detect SMS spam. You can download the Jupyter Notebook of the study `here <https://github.com/vertica/VerticaPy/blob/master/examples/business/spam/spam.ipynb>`_.

- **v1:** the SMS type (spam or ham).
- **v2:** SMS content.

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

Let's create a Virtual DataFrame of the dataset. The dataset is available `here <https://github.com/vertica/VerticaPy/blob/master/examples/business/spam/spam.csv>`_.

.. code-block:: ipython

    spam = vp.read_csv("spam.csv")

Let's take a look at the first few entries in the dataset.

.. code-block:: ipython
    
    spam.head(10)

.. ipython:: python
    :suppress:

    spam = vp.read_csv(
        "/project/data/VerticaPy/docs/source/_static/website/examples/data/spam/spam.csv",
    )
    res = spam.head(10)
    html_file = open("/project/data/VerticaPy/docs/figures/examples_spam_table.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_spam_table.html

Data Exploration and Preparation
---------------------------------

Our dataset relies on text analysis. First, we should create some features. For example, we can use the SMS length and label encoding on the 'type' to get a dummy (1 if the message is a SPAM, 0 otherwise). We should also convert the message content to lowercase to simplify our analysis.

.. code-block:: python

    import verticapy.sql.functions as fun

    spam["length"] = fun.length(spam["content"])
    spam["content"].apply("LOWER({})")
    spam["type"].decode('spam', 1, 0)

.. ipython:: python
    :suppress:

    import verticapy.sql.functions as fun

    spam["length"] = fun.length(spam["content"])
    spam["content"].apply("LOWER({})")
    res = spam["type"].decode('spam', 1, 0)
    html_file = open("/project/data/VerticaPy/docs/figures/examples_spam_table_clean.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_spam_table_clean.html

Let's compute some statistics using the length of the message.

.. code-block:: python

    spam['type'].describe(
        method = 'cat_stats', 
        numcol = 'length',
    )

.. ipython:: python
    :suppress:

    res = spam['type'].describe(
        method = 'cat_stats', 
        numcol = 'length',
    )
    html_file = open("/project/data/VerticaPy/docs/figures/examples_spam_table_describe.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_spam_table_describe.html

**Notice:** spam tends to be longer than a normal message. First, let's create a view with just spam. Then, we'll use the ``CountVectorizer`` to create a dictionary and identify keywords.

.. code-block:: python

    spams = spam.search(spam["type"] == 1)

    from verticapy.machine_learning.vertica import CountVectorizer

    dict_spams = CountVectorizer()
    dict_spams.fit(spams, ["content"])
    dict_spams = dict_spams.transform()
    dict_spams

.. ipython:: python
    :suppress:
    :okwarning:

    spams = spam.search(spam["type"] == 1)

    from verticapy.machine_learning.vertica import CountVectorizer

    dict_spams = CountVectorizer()
    dict_spams.fit(spams, ["content"])
    dict_spams = dict_spams.transform()
    res = dict_spams
    html_file = open("/project/data/VerticaPy/docs/figures/examples_spam_table_clean_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_spam_table_clean_2.html

Let's add the most occurent words in our ``vDataFrame`` and compute the correlation vector.

.. code-block:: python

    for word in dict_spams.head(200).values["token"]:
        if word not in ['content', 'length', 'type'] : # because there is already a column called content, length and type
            spam.regexp(
                name = word,
                pattern = word,
                method = "count",
                column = "content",
            )
    spam.corr(focus = "type")

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    for word in dict_spams.head(200).values["token"]:
        if word not in ['content', 'length', 'type'] : # because there is already a column called content, length and type
            spam.regexp(
                name = word,
                pattern = word,
                method = "count",
                column = "content",
            )
    fig = spam.corr(focus = "type")
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_spam_corr.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_spam_corr.html

Let's just keep the first 100-most correlated features and merge the numbers together.

.. code-block:: python

    words = spam.corr(focus = "type", show = False)
    spam.drop(columns = words["index"][101:])

    for word in words["index"][1:101]:
        if any(char.isdigit() for char in word):
            spam[word].drop()
            
    spam.regexp(
        column = "content",
        pattern = "([0-9])+",
        method = "count",
        name = "nb_numbers",
    )

.. ipython:: python
    :suppress:

    words = spam.corr(focus = "type", show = False)
    spam.drop(columns = words["index"][101:])

    for word in words["index"][1:101]:
        if any(char.isdigit() for char in word):
            spam[word].drop()
    res = spam.regexp(
        column = "content",
        pattern = "([0-9])+",
        method = "count",
        name = "nb_numbers",
    )
    html_file = open("/project/data/VerticaPy/docs/figures/examples_spam_table_regexp.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_spam_table_regexp.html

Let's narrow down our keyword list to words of more than two characters.

.. code-block:: python

    columns = spam.get_columns()
    for word in columns:
        if len(word.replace('"', '')) <= 2:
            spam[word].drop()

.. ipython:: python
    :suppress:

    columns = spam.get_columns()
    for word in columns:
        if len(word.replace('"', '')) <= 2:
            spam[word].drop()

Compute the correlation vector again using the response column.

.. code-block:: python

    spam.corr(focus = "type")

.. ipython:: python
    :suppress:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = spam.corr(focus = "type")
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_spam_corr_2.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_spam_corr_2.html

We have enough correlated features with our response to create a fantastic model.

________

Machine Learning
-----------------

The ``Naive Bayes`` classifier is a powerful and performant algorithm for text analytics and binary classification. Before using it on our data, let's use a ``cross-validation`` to test the efficiency of our model.

.. code-block:: python

    from verticapy.machine_learning.vertica import MultinomialNB

    model = MultinomialNB()

    from verticapy.machine_learning.model_selection import cross_validate

    cross_validate(
        model, 
        spam, 
        spam.get_columns(exclude_columns = ["type", "content"]), 
        "type", 
        cv = 5,
    )

.. ipython:: python
    :suppress:
    :okwarning:

    from verticapy.machine_learning.vertica import MultinomialNB

    model = MultinomialNB()

    from verticapy.machine_learning.model_selection import cross_validate

    res = cross_validate(
        model, 
        spam, 
        spam.get_columns(exclude_columns = ["type", "content"]), 
        "type", 
        cv = 5,
    )
    html_file = open("/project/data/VerticaPy/docs/figures/examples_spam_table_report.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_spam_table_report.html

We have an excellent model! Let's learn from the data.

.. ipython:: python

    model.fit(
        spam, 
        spam.get_columns(exclude_columns = ["type", "content"]), 
        "type",
    )
    model.confusion_matrix()

Our model can reliably identify spam.

Conclusion
-----------

We've solved our problem in a Pandas-like way, all without ever loading data into memory!