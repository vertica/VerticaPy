.. _examples.learn.pokemon:

Pokemon
========

This example uses the ``pokemon`` and ``combats`` datasets to predict the winner of a 1-on-1 Pokemon battle. You can download the Jupyter Notebook of the study here and two datasets:

`pokemon <https://github.com/vertica/VerticaPy/tree/master/examples/learn/pokemon/pokemons.csv>`_

- **Name:** The name of the Pokemon.
- **Generation:** Pokemon's generation.
- **Legendary:** True if the Pokemon is legendary.
- **HP:** Number of hit points.
- **Attack:** Attack stat.
- **Sp_Atk:** Special attack stat.
- **Defense:** Defense stat.
- **Sp_Def:** Special defense stat.
- **Speed:** Speed stat.
- **Type_1:** Pokemon's first type.
- **Type_2:** Pokemon's second type.

`fights <https://github.com/vertica/VerticaPy/tree/master/examples/learn/pokemon/fights.csv>`_

- **First_pokemon:** Pokemon of trainer 1.
- **Second_pokemon:** Pokemon of trainer 2.
- **Winner:** Winner of the battle.

We will follow the data science cycle (Data Exploration - Data Preparation - Data Modeling - Model Evaluation - Model Deployment) to solve this problem.

Initialization
---------------

This example uses the following version of VerticaPy:

.. ipython:: python
    
    import verticapy as vp
    
    vp.__version__

Connect to Vertica. This example uses an existing connection called ``VerticaDSN`` . 
For details on how to create a connection, see the :ref:`connection` tutorial.
You can skip the below cell if you already have an established connection.

.. code-block:: python
    
    vp.connect("VerticaDSN")

Let's ingest the datasets.

.. code-block:: python
    
    import verticapy.sql.functions as fun

    combats = vp.read_csv("fights.csv")
    combats.head(5)

.. ipython:: python
    :suppress:

    import verticapy.sql.functions as fun

    combats = vp.read_csv("SPHINX_DIRECTORY/source/_static/website/examples/data/pokemon/fights.csv")
    res = combats.head(5)
    html_file = open("SPHINX_DIRECTORY/figures/examples_combats_table.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_combats_table.html

.. code-block:: python

    pokemon = vp.read_csv("pokemons.csv")
    pokemon.head(5)

.. ipython:: python
    :suppress:

    pokemon = vp.read_csv("SPHINX_DIRECTORY/source/_static/website/examples/data/pokemon/pokemons.csv")
    res = pokemon.head(5)
    html_file = open("SPHINX_DIRECTORY/figures/examples_pokemon_table_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_pokemon_table_2.html

Data Exploration and Preparation
---------------------------------

The table ``combats`` will be joined to the table ``pokemon`` to predict the winner.

The ``pokemon`` table contains the information on each Pokemon. Let's describe this table.

.. code-block:: python

    pokemon.describe(method = "categorical", unique = True)

.. ipython:: python
    :suppress:

    res = pokemon.describe(method = "categorical", unique = True)
    html_file = open("SPHINX_DIRECTORY/figures/examples_pokemon_table_describe.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_pokemon_table_describe.html

The pokemon's ``Name``, ``Generation``, and whether or not it's ``Legendary`` will never influence the outcome of the battle, so we can drop these columns.

.. code-block:: python

    pokemon.drop(
        [
            "Generation", 
            "Legendary", 
            "Name",
        ]
    )

.. ipython:: python
    :suppress:

    res = pokemon.drop(
        [
            "Generation", 
            "Legendary", 
            "Name",
        ]
    )
    html_file = open("SPHINX_DIRECTORY/figures/examples_pokemon_table_drop.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_pokemon_table_drop.html

The ``ID`` will be the key to join the data. By joining the data, we will be able to create more relevant features.

.. ipython:: python

    fights = pokemon.join(
        combats, 
        on = {"ID": "First_Pokemon"}, 
        how = "inner",
        expr1 = [
            "Sp_Atk AS Sp_Atk_1", 
            "Speed AS Speed_1", 
            "Sp_Def AS Sp_Def_1", 
            "Defense AS Defense_1", 
            "Type_1 AS Type_1_1", 
            "Type_2 AS Type_2_1", 
            "HP AS HP_1",  
            "Attack AS Attack_1",
        ],
        expr2 = [
            "First_Pokemon", 
            "Second_Pokemon", 
            "Winner",
        ]).join(pokemon, 
        on = {"Second_Pokemon": "ID"}, 
        how = "inner",
        expr2 = [
            "Sp_Atk AS Sp_Atk_2", 
            "Speed AS Speed_2", 
            "Sp_Def AS Sp_Def_2", 
            "Defense AS Defense_2", 
            "Type_1 AS Type_1_2", 
            "Type_2 AS Type_2_2", 
            "HP AS HP_2", 
            "Attack AS Attack_2",
        ],
        expr1 = 
            [
                "Sp_Atk_1", 
                "Speed_1", 
                "Sp_Def_1", 
                "Defense_1", 
                "Type_1_1", 
                "Type_2_1", 
                "HP_1", 
                "Attack_1", 
                "Winner", 
                "Second_pokemon",
            ]
    )

Features engineering is the key. Here, we can create features that describe the stat differences between the first and second Pokemon. We can also change ``winner`` to a binary value: 1 if the first pokemon won and 0 otherwise.

.. ipython:: python

    fights["Sp_Atk_diff"] = fights["Sp_Atk_1"] - fights["Sp_Atk_2"]
    fights["Speed_diff"] = fights["Speed_1"] - fights["Speed_2"]
    fights["Sp_Def_diff"] = fights["Sp_Def_1"] - fights["Sp_Def_2"]
    fights["Defense_diff"] = fights["Defense_1"] - fights["Defense_2"]
    fights["HP_diff"] = fights["HP_1"] - fights["HP_2"]
    fights["Attack_diff"] = fights["Attack_1"] - fights["Attack_2"]
    fights["Winner"] = fun.case_when(fights["Winner"] == fights["Second_pokemon"], 0, 1)
    fights = fights[
        [
            "Sp_Atk_diff",
            "Speed_diff",
            "Sp_Def_diff", 
            "Defense_diff",
            "HP_diff",
            "Attack_diff", 
            "Type_1_1",
            "Type_1_2",
            "Type_2_1",
            "Type_2_2", 
            "Winner",
        ]
    ]

Missing values can not be handled by most machine learning models. Let's see which features we should impute.

.. code-block:: python

    fights.count()

.. ipython:: python
    :suppress:

    res = fights.count()
    html_file = open("SPHINX_DIRECTORY/figures/examples_pokemon_table_clean_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_pokemon_table_clean_1.html

In terms of missing values, our only concern is the Pokemon's second type (``Type_2_1`` and ``Type_2_2``). Since some Pokemon only have one type, these features are MNAR (missing values not at random). We can impute the missing values by creating another category.

.. code-block:: python

    fights["Type_2_1"].fillna("No")
    fights["Type_2_2"].fillna("No")

.. ipython:: python
    :suppress:

    fights["Type_2_1"].fillna("No")
    res = fights["Type_2_2"].fillna("No")
    html_file = open("SPHINX_DIRECTORY/figures/examples_pokemon_table_clean_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_pokemon_table_clean_2.html

Let's use the current_relation method to see how our data preparation so far on the :py:mod:`~verticapy.vDataFrame` generates SQL code.

.. ipython:: python

    print(fights.current_relation())

VerticaPy will remember your modifications and always generate an up-to-date SQL query.

Let's look at the correlations between all the variables.

.. code-block:: python

    fights.corr(method = "spearman")

.. ipython:: python
    :suppress:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = fights.corr(method = "spearman")
    fig.write_html("SPHINX_DIRECTORY/figures/examples_pokemon_corr.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_pokemon_corr.html

Many variables are correlated to the response column. We have enough information to create our predictive model.

Machine Learning
-----------------

Some really important features are categorical. ``Random Forest`` can handle them. Besides, we need trees deep enough to compare all the different types.

.. code-block:: python

    from verticapy.machine_learning.vertica import RandomForestClassifier
    from verticapy.machine_learning.model_selection import cross_validate

    predictors = fights.get_columns(exclude_columns = ["Winner"])
    model = RandomForestClassifier(
        n_estimators = 50, 
        max_depth = 100, 
        max_leaf_nodes = 400, 
        nbins = 100,
    )
    cross_validate(model, fights, predictors, "Winner")

.. ipython:: python
    :suppress:
    :okwarning:

    from verticapy.machine_learning.vertica import RandomForestClassifier
    from verticapy.machine_learning.model_selection import cross_validate

    predictors = fights.get_columns(exclude_columns = ["Winner"])
    model = RandomForestClassifier(
        n_estimators = 50, 
        max_depth = 100, 
        max_leaf_nodes = 400, 
        nbins = 100,
    )
    res = cross_validate(model, fights, predictors, "Winner")
    html_file = open("SPHINX_DIRECTORY/figures/examples_pokemon_cv.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_pokemon_cv.html

We have an excellent model with an average ``AUC`` of more than ``99%``. Let's create a model with the entire dataset and look at the importance of each feature.

.. code-block:: python

    model.fit(
        fights,
        predictors, 
        "Winner",
    )
    model.features_importance()

.. ipython:: python
    :suppress:
    :okwarning:

    model.fit(
        fights,
        predictors, 
        "Winner",
    )
    fig = model.features_importance()
    fig.write_html("SPHINX_DIRECTORY/figures/examples_pokemon_features_importance_ml.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_pokemon_features_importance_ml.html

Based on our model, it seems that a Pokemon's speed and attack stats are the strongest predictors for the winner of a battle.

Conclusion
-----------

We've solved our problem in a Pandas-like way, all without ever loading data into memory!