.. _user_guide.full_stack.to_json:

=========================
Example: XGBoost.to_json
=========================

Connect to Vertica
--------------------

For a demonstration on how to create a new connection to Vertica, 
see :ref:`connection`. In this example, we will use an 
existing connection named 'VerticaDSN'.

.. code-block:: python

    import verticapy as vp
    vp.connect("VerticaDSN")

Create a Schema (Optional)
---------------------------

Schemas allow you to organize database objects in a collection, similar to a namespace. If you create a database object 
without specifying a schema, Vertica uses the 'public' schema. For example, to specify the 'example_table' in 'example_schema', you would use: 'example_schema.example_table'.

To keep things organized, this example creates the 'xgb_to_json' schema and drops it (and its associated tables, views, etc.) at the end:

.. ipython:: python
    :suppress:

    import verticapy as vp

.. ipython:: python

    vp.drop("xgb_to_json", method = "schema")
    vp.create_schema("xgb_to_json")

Load Data
----------

VerticaPy lets you load many well-known datasets like Iris, Titanic, Amazon, etc.
For a full list, check out :ref:`datasets`.

.. ipython:: python

    from verticapy.datasets import load_titanic
    vdf = load_titanic(
        name = "titanic",
        schema = "xgb_to_json",
    )


You can also load your own data. To ingest data from a CSV file, 
use the :py:func:`~verticapy.read_csv` function.

Create a vDataFrame
--------------------

vDataFrames allow you to prepare and explore your data without modifying its representation in your Vertica database. Any changes you make are applied to the vDataFrame as modifications to the SQL query for the table underneath.

To create a vDataFrame out of a table in your Vertica database, specify its schema and table name with the standard SQL syntax. For example, to create a vDataFrame out of the 'titanic' table in the 'xgb_to_json' schema:

.. ipython:: python

    vdf = vp.vDataFrame("xgb_to_json.titanic")

Create an XGB model
-------------------

Create a :py:func:`~verticapy.machine_learning.vertica.ensemble.XGBClassifier` model.

Unlike a vDataFrame object, which simply queries the table it 
was created with, the VerticaPy :py:func:`~verticapy.machine_learning.vertica.ensemble.XGBClassifier` object creates 
and then references a model in Vertica, so it must be stored in a 
schema like any other database object.

This example creates the 'my_model' :py:func:`~verticapy.machine_learning.vertica.ensemble.XGBClassifier` model in 
the 'xgb_to_json' schema:

This example loads the Titanic dataset with the load_titanic function 
into a table called 'titanic' in the 'xgb_to_json' schema:

.. ipython:: python

    from verticapy.machine_learning.vertica.ensemble import XGBClassifier
    model = XGBClassifier(
        "xgb_to_json.my_model",
        max_ntree = 4,
        max_depth = 3,
    )

Prepare the Data
-----------------

While Vertica XGBoost supports columns of type VARCHAR, Python XGBoost does not, so you must encode the categorical 
columns you want to use. You must also drop or impute missing values.

This example drops 'age', 'fare', 'sex', 'embarked' and 'survived' columns from the vDataFrame and then encodes the 
'sex' and 'embarked' columns. These changes are applied to the vDataFrame's query and does not affect the main "xgb_to_json.titanic' table stored in Vertica:

.. ipython:: python

    vdf = vdf[["age", "fare", "sex", "embarked", "survived"]];
    vdf.dropna();
    vdf["sex"].label_encode();
    vdf["embarked"].label_encode();


.. ipython:: python
    :suppress:
    :okwarning:

    res = vdf
    html_file = open("/project/data/VerticaPy/docs/figures/ug_fs_to_json_vdf.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_to_json_vdf.html

Split your data into training and testing:

.. ipython:: python

    train, test = vdf.train_test_split(0.05);

Train the Model
----------------

Define the predictor and the response columns:

.. ipython:: python

    relation = train;
    X = ["age", "fare", "sex", "embarked"]
    y = "survived"

Train the model with fit():

.. ipython:: python
    :okwarning:

    model.fit(relation, X, y)

Evaluate the Model
--------------------

Evaluate the model with :py:func:`~verticapy.machine_learning.vertica.ensemble.XGBClassifier.report`:

.. code-block:: ipython

    model.report()

.. ipython:: python
    :suppress:
    :okwarning:

    res = model.report()
    html_file = open("/project/data/VerticaPy/docs/figures/ug_fs_to_json_report.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_to_json_report.html

Use to_json() to export the model to a JSON file. If you omit a filename, VerticaPy prints the model:

.. ipython:: python

    model.to_json()


To export and save the model as a JSON file, specify a filename:

.. ipython:: python

    model.to_json("exported_xgb_model.json");

Unlike Python XGBoost, Vertica does not store some information like 'sum_hessian' or 'loss_changes,' and the exported model from :py:func:`~verticapy.machine_learning.vertica.ensemble.XGBClassifier.to_json` replaces this information with a list of zeroes. These information are replaced by a list filled with zeros.

Make Predictions with an Exported Model
----------------------------------------

This exported model can be used with the Python XGBoost API right away, and exported models make identical predictions in Vertica and Python:

.. ipython:: python

    import pytest
    import xgboost as xgb
    model_python = xgb.XGBClassifier();
    model_python.load_model("exported_xgb_model.json");
    # Convert to numpy format
    X_test = test["age","fare","sex","embarked"].to_numpy() ;
    y_test_vertica = model.to_python(return_proba = True)(X_test);
    y_test_python = model_python.predict_proba(X_test);
    result = (y_test_vertica - y_test_python) ** 2;
    result = result.sum() / len(result);
    assert result == pytest.approx(0.0, abs = 1.0E-14)

For multiclass classifiers, the probabilities returned by the VerticaPy and the exported model may differ slightly because of normalization; while Vertica uses multinomial logistic regression, XGBoost Python uses Softmax. Again, this difference does not affect the model's final predictions. Categorical predictors must be encoded.


Clean the Example Environment
------------------------------

Drop the 'xgb_to_json' schema, using CASCADE to drop any 
database objects stored inside (the 'titanic' table, the 
:py:func:`~verticapy.machine_learning.vertica.ensemble.XGBClassifier` 
model, etc.), then delete the 'exported_xgb_model.json' file:

.. ipython:: python

    import os
    os.remove("exported_xgb_model.json")
    vp.drop("xgb_to_json", method = "schema")

Conclusion
-----------

VerticaPy lets you to create, train, evaluate, and export 
Vertica machine learning models. There are some notable 
nuances when importing a Vertica XGBoost model into 
Python XGBoost, but these do not affect the accuracy of the model or its predictions:

Some information computed during the training phase may not 
be stored (e.g. 'sum_hessian' and 'loss_changes').
The exact probabilities of multiclass classifiers in a 
Vertica model may differ from those in Python, but bot  h 
will make the same predictions.
Python XGBoost does not support categorical predictors, 
so you must encode them before training the model in VerticaPy.