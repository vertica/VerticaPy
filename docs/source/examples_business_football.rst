.. _examples.business.football:

Football
=========

In this example, we use the 'football' dataset to predict the outcomes of games between various teams. You can download the Jupyter Notebook of the study `here <https://github.com/vertica/VerticaPy/blob/master/examples/understand/business/football/football.ipynb>`_ and the dataset `here <https://github.com/vertica/VerticaPy/blob/master/examples/business/football/games.csv>`_.

- **date:** Date of the game.
- **home_team:** Home Team.
- **home_score:** Home Team number of goals.
- **away_team:** Away Team.
- **away_score:** Away Team number of goals.
- **tournament:** Game Type (World Cup, Friendly...).
- **city:** City where the game took place.
- **country:** Country where the game took place.
- **neutral:** If the event took place to a neutral location.

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

.. code-block:: python

    football = vp.read_csv("games.csv")
    football.head(5)

.. ipython:: python
    :suppress:

    football = vp.read_csv("SPHINX_DIRECTORY/source/_static/website/examples/data/football/games.csv")
    res = football.head(5)
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_table_head.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_table_head.html

Data Exploration and Preparation
---------------------------------

Let's explore the data by displaying descriptive statistics of all the columns.

.. code-block:: python

    football["date"].describe()

.. ipython:: python
    :suppress:

    res = football["date"].describe()
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_describe.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_describe.html

The dataset includes a total of 41,586 games, which take place between 1872 and 2020. Let's look at our game types and teams.

.. code-block:: python

    football["tournament"].describe()

.. ipython:: python
    :suppress:

    res = football["tournament"].describe()
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_describe_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_describe_2.html

Different types of tournaments took place (FIFA World Cup, UEFA Euro, etc.) aand most of the games in our data are friendlies or qualifiers for international tournaments.

.. code-block:: python

    football.describe()

.. ipython:: python
    :suppress:

    res = football.describe()
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_describe_3.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_describe_3.html

.. code-block:: python

    football.describe(method = "categorical")

.. ipython:: python
    :suppress:

    res = football.describe(method = "categorical")
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_describe_4.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_describe_4.html

The dataset includes 308 national teams. For most of the games, the home team scores better than the away team. Since some games take place in a neutral location, we can ensure this hypothesis using the variable 'neutral'. Notice also that the number of goals per match is pretty low (median of 1 for both away and home teams).

Goal
+++++

Our goal for the study will be to predict the outcomes of games after 2015.
Before doing the study, we can notice that some teams names have changed over time. We need to change the old names by the new names otherwise it will add too much bias in the data.

.. code-block:: python

    for team in ["home_team", "away_team"]:
        football[team].decode(
            'German DR', 'Germany',
            'Czechoslovakia', 'Czech Republic',
            'Yugoslavia', 'Serbia',
            'Yemen DPR', 'Yemen',
            football[team],
        )

.. ipython:: python
    :suppress:

    for team in ["home_team", "away_team"]:
        football[team].decode(
            'German DR', 'Germany',
            'Czechoslovakia', 'Czech Republic',
            'Yugoslavia', 'Serbia',
            'Yemen DPR', 'Yemen',
            football[team],
        )

Let's just consider teams that have played more than five home and away games.

.. code-block:: python

    football["cnt_games_1"] = "COUNT(*) OVER (PARTITION BY home_team)"
    football["cnt_games_2"] = "COUNT(*) OVER (PARTITION BY away_team)"
    football.filter((football["cnt_games_2"] > 5) & (football["cnt_games_1"] > 5))
    vp.drop("football_clean", method = "table")
    football.to_db(
        name = "football_clean",
        usecols = [
            "date", 
            "home_score", 
            "home_team", 
            "tournament", 
            "away_team", 
            "away_score", 
            "neutral", 
            "country",
            "city",
        ],
        relation_type = "table",
        inplace = True,
    )

.. ipython:: python
    :suppress:

    football["cnt_games_1"] = "COUNT(*) OVER (PARTITION BY home_team)"
    football["cnt_games_2"] = "COUNT(*) OVER (PARTITION BY away_team)"
    football.filter((football["cnt_games_2"] > 5) & (football["cnt_games_1"] > 5))
    vp.drop("football_clean", method = "table")
    football.to_db(
        name = "football_clean",
        usecols = [
            "date", 
            "home_score", 
            "home_team", 
            "tournament", 
            "away_team", 
            "away_score", 
            "neutral", 
            "country",
            "city",
        ],
        relation_type = "table",
        inplace = True,
    )
    res = football
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_to_db_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_to_db_1.html

A lot of things could influence the outcome of a game. Since we only have access to the score, teams, and type of game, we can't consider external factors like, weather or temperature, which would otherwise help our prediction.

To create a good model using this dataset, we could compute each team's key performance indicator (KPI), ranking (clusters computed using the number of games in important tournaments like the World Cup, the percentage of victory...), shape (moving windows using the last games information), and other factors.

Here's our plan:
- Identify cup winners
- Rank the teams with clustering
- Compute teams' KPIs
- Create a machine learning model

Data Preparation for Clustering
--------------------------------

To create clusters, we need to find which teams are the winners of main tournaments (mainly the World Cups and Continental Cups). Since all tournaments took place the same year, we could partition by tournament and year to identify the last game of the tournament.

We'll ignore ties for our analysis since there's no way to determine a winner.

Cup Winner
+++++++++++

Let's start by creating the feature 'winner' to indicate the winner of a game.

.. code-block:: python

    import verticapy.sql.functions as fun

    football.filter(fun.year(football["date"]) <= 2015)
    football.case_when(
        "winner",
        football["home_score"] > football["away_score"], football["home_team"],
        football["home_score"] < football["away_score"], football["away_team"],
        None,
    )

.. ipython:: python
    :suppress:

    import verticapy.sql.functions as fun

    football.filter(fun.year(football["date"]) <= 2015)
    res = football.case_when(
        "winner",
        football["home_score"] > football["away_score"], football["home_team"],
        football["home_score"] < football["away_score"], football["away_team"],
        None,
    )
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_case_when_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_case_when_1.html

Let's analyze the last game of each tournament.

.. code-block:: python

    football["year"] = fun.year(football["date"])
    football.analytic(
        "row_number", 
        order_by = {"date": "desc"}, 
        by = ["tournament", "year"] , 
        name = "order_tournament",
    )

.. ipython:: python
    :suppress:

    import verticapy.sql.functions as fun

    football["year"] = fun.year(football["date"])
    football.analytic(
        "row_number", 
        order_by = {"date": "desc"}, 
        by = ["tournament", "year"] , 
        name = "order_tournament",
    )
    res = football
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_analytic_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_analytic_2.html

We can filter the data by only considering the last games and top tournaments.

.. code-block:: python

    football.filter(
        conditions = [
            football["order_tournament"] == 1,
            football["winner"] != None,
            football["tournament"]._in(
                [
                    "FIFA World Cup", 
                    "UEFA Euro", 
                    "Copa América", 
                    "African Cup of Nations",
                    "AFC Asian Cup",
                    "Gold Cup",
                ]
            )
        ]
    )

.. ipython:: python
    :suppress:

    football.filter(
        conditions = [
            football["order_tournament"] == 1,
            football["winner"] != None,
            football["tournament"]._in(
                [
                    "FIFA World Cup", 
                    "UEFA Euro", 
                    "Copa América", 
                    "African Cup of Nations",
                    "AFC Asian Cup",
                    "Gold Cup",
                ]
            )
        ]
    )
    res = football
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_filter_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_filter_2.html

Let's consider the World Cup as a special tournament. It is the only one where the confrontations between the top teams is possible.

.. code-block:: python

    football["Word_Cup"] = fun.decode(
        football["tournament"], "FIFA World Cup", 
        1, 0,
    )

.. ipython:: python
    :suppress:

    football["Word_Cup"] = fun.decode(
        football["tournament"], "FIFA World Cup", 
        1, 0,
    )
    res = football["Word_Cup"]
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_decode_3.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_decode_3.html

We can compute all the number of cup-wins by team. As expected, Brazil and Germany are the top football teams.

.. code-block:: python

    agg = [
        fun.sum(football["Word_Cup"])._as("nb_World_Cup"),
        fun.sum(1 - football["Word_Cup"])._as("nb_Continental_Cup"),
    ]
    football_cup_winners = football.groupby(["winner"], agg)
    football_cup_winners.sort(
        {
            "nb_World_Cup": "desc",
            "nb_Continental_Cup": "desc",
        }
    ).head(10)

.. ipython:: python
    :suppress:

    agg = [
        fun.sum(football["Word_Cup"])._as("nb_World_Cup"),
        fun.sum(1 - football["Word_Cup"])._as("nb_Continental_Cup"),
    ]
    football_cup_winners = football.groupby(["winner"], agg)
    res = football_cup_winners.sort(
        {
            "nb_World_Cup": "desc",
            "nb_Continental_Cup": "desc",
        }
    ).head(10)
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_groupby_3.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_groupby_3.html

Let's export the result to our Vertica database.

.. code-block:: python

    vp.drop(
        "football_cup_winners",
        method = "table",
    )
    football_cup_winners.to_db(
        "football_cup_winners", 
        relation_type = "table",
    )

.. ipython:: python
    :suppress:

    vp.drop(
        "football_cup_winners",
        method = "table",
    )
    football_cup_winners.to_db(
        "football_cup_winners", 
        relation_type = "table",
    )
    res = football_cup_winners
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_to_db_4.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_to_db_4.html

Team Confederations
++++++++++++++++++++

Looking into team confederations could help our analysis. For example, this might help us quantify skill differences between different continents. A team that had played a qualification of a specific location can only belong to that tournament confederation.

First let's encode the different continents so we can compute the correct aggregations.

.. code-block:: python

    football = vp.read_csv("games.csv")
    football.case_when(
        'confederation', 
        football["tournament"] == 'UEFA Euro qualification', 5,
        football["tournament"] == 'African Cup of Nations qualification', 4,
        football["tournament"] == 'AFC Asian Cup qualification', 3,
        football["tournament"] == 'Copa América', 2,
        football["tournament"] == 'Gold Cup', 1, 0,
    )

.. ipython:: python
    :suppress:

    football = vp.read_csv("SPHINX_DIRECTORY/source/_static/website/examples/data/football/games.csv")
    res = football.case_when(
        'confederation', 
        football["tournament"] == 'UEFA Euro qualification', 5,
        football["tournament"] == 'African Cup of Nations qualification', 4,
        football["tournament"] == 'AFC Asian Cup qualification', 3,
        football["tournament"] == 'Copa América', 2,
        football["tournament"] == 'Gold Cup', 1, 0,
    )
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_table_confederation_case_when.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_table_confederation_case_when.html

We can aggregate the data and get each team's continent.

.. code-block:: python

    confederation = football.groupby(
        ["home_team"],
        [fun.max(football["confederation"])._as("confederation")],
    )
    confederation.head(100)

.. ipython:: python
    :suppress:

    confederation = football.groupby(
        ["home_team"],
        [fun.max(football["confederation"])._as("confederation")],
    )
    res = confederation.head(100)
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_confederation_6.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_confederation_6.html

We can decode the previous label encoding.

.. code-block:: python

    confederation["confederation"].decode(
        5, "UEFA",
        4, "CAF",
        3, "AFC",
        2, "CONMEBOL",
        1, "CONCACAF",
        "OFC",
    )

.. ipython:: python
    :suppress:

    res = confederation["confederation"].decode(
        5, "UEFA",
        4, "CAF",
        3, "AFC",
        2, "CONMEBOL",
        1, "CONCACAF",
        "OFC",
    )
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_confederation_8.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_confederation_8.html

Let's export the result to our Vertica database.

.. code-block:: python

    vp.drop("confederation")
    confederation["home_team"].rename("team")
    confederation.to_db(
        name = "confederation",
        relation_type = "table",
    )

.. ipython:: python
    :suppress:

    vp.drop("confederation")
    confederation["home_team"].rename("team")
    confederation.to_db(
        name = "confederation",
        relation_type = "table",
    )
    res = confederation
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_confederation_9.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_confederation_9.html

Team KPIs
++++++++++

We use just two variables to track teams: away_team and home_team. This makes it a bit difficult to compute new features. We need to duplicate the dataset and intervert the two teams. This way, we can compute KPIs using a partition by the first team to avoid double-counting any games.

.. code-block:: python

    football = vp.vDataFrame("football_clean")
    football.filter(fun.year(football["date"]) <= 2015)
    football["home_team"].rename("team1")
    football["home_score"].rename("team1_score")
    football["away_team"].rename("team2")
    football["away_score"].rename("team2_score")
    football["neutral"].decode(True, 0, 1)

    football2 = vp.vDataFrame("football_clean")
    football2.filter(fun.year(football["date"]) <= 2015)
    football2["home_team"].rename("team2")
    football2["home_score"].rename("team2_score")
    football2["away_team"].rename("team1")
    football2["away_score"].rename("team1_score")
    football2["neutral"].decode(True, 0, 2)

    # Merging the 2 interverted datasets
    all_matchs = football.append(football2)
    all_matchs["neutral"].rename("home_team_id")

.. ipython:: python
    :suppress:

    football = vp.vDataFrame("football_clean")
    football.filter(fun.year(football["date"]) <= 2015)
    football["home_team"].rename("team1")
    football["home_score"].rename("team1_score")
    football["away_team"].rename("team2")
    football["away_score"].rename("team2_score")
    football["neutral"].decode(True, 0, 1)

    football2 = vp.vDataFrame("football_clean")
    football2.filter(fun.year(football["date"]) <= 2015)
    football2["home_team"].rename("team2")
    football2["home_score"].rename("team2_score")
    football2["away_team"].rename("team1")
    football2["away_score"].rename("team1_score")
    football2["neutral"].decode(True, 0, 2)

    # Merging the 2 interverted datasets
    all_matchs = football.append(football2)
    res = all_matchs["neutral"].rename("home_team_id")
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_clean_10.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_clean_10.html

To compute the different aggregations, we need to add dummies which indicate the type of game and winner.

.. code-block:: python

    all_matchs["World_Tournament"] = fun.case_when(all_matchs["tournament"]._in(
        [
            "FIFA World Cup",
            "Confederations Cup"
        ],
    ), 1, 0)
    all_matchs["Continental_Tournament"] = fun.case_when(
        all_matchs["tournament"]._in(
            [
                "UEFA Euro", 
                "Copa América", 
                "African Cup of Nations",
                "AFC Asian Cup",
                "Gold Cup",
                "FIFA World Cup qualification",
            ]
        ), 1, 0)
    all_matchs["Victory_team1"] = (all_matchs["team1_score"] > all_matchs["team2_score"])
    all_matchs["Victory_team1"].astype("int")
    all_matchs["Draw"] = (all_matchs["team1_score"] == all_matchs["team2_score"])
    all_matchs["Draw"].astype("int")

.. ipython:: python
    :suppress:

    all_matchs["World_Tournament"] = fun.case_when(all_matchs["tournament"]._in(
        [
            "FIFA World Cup",
            "Confederations Cup"
        ],
    ), 1, 0)
    all_matchs["Continental_Tournament"] = fun.case_when(
        all_matchs["tournament"]._in(
            [
                "UEFA Euro", 
                "Copa América", 
                "African Cup of Nations",
                "AFC Asian Cup",
                "Gold Cup",
                "FIFA World Cup qualification",
            ]
        ), 1, 0)
    all_matchs["Victory_team1"] = (all_matchs["team1_score"] > all_matchs["team2_score"])
    all_matchs["Victory_team1"].astype("int")
    all_matchs["Draw"] = (all_matchs["team1_score"] == all_matchs["team2_score"])
    res = all_matchs["Draw"].astype("int")
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_clean_11.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_clean_11.html

Now we can compute each team's KPI.

.. code-block:: python

    teams_kpi = all_matchs.groupby(
        ["team1"],
        [
            fun.sum(all_matchs["World_Tournament"])._as("Number_Games_World_Tournament"),
            fun.sum(all_matchs["Continental_Tournament"])._as("Number_Games_Continental_Tournament"),
            fun.avg(fun.decode(all_matchs["World_Tournament"], 1, all_matchs["Victory_team1"]))._as("Percent_Victory_World_Tournament"),
            fun.avg(fun.decode(all_matchs["Continental_Tournament"], 1, all_matchs["Victory_team1"]))._as("Percent_Victory_Continental_Tournament"),
            fun.avg(fun.case_when((all_matchs["home_team_id"] == 1) & (all_matchs["World_Tournament"] == 0) & (all_matchs["Continental_Tournament"] == 0), all_matchs["Victory_team1"], None))._as("Percent_Victory_Home"),
            fun.avg(fun.case_when((all_matchs["home_team_id"] != 1) & (all_matchs["World_Tournament"] == 0) & (all_matchs["Continental_Tournament"] == 0), all_matchs["Victory_team1"], None))._as("Percent_Victory_Away"),
            fun.avg(all_matchs["Victory_team1"])._as("Percent_Victory"),
            fun.avg(all_matchs["Draw"])._as("Percent_Draw"),
            fun.avg(all_matchs["team1_score"])._as("Avg_goals"),
            fun.avg(all_matchs["team2_score"])._as("Avg_goals_conceded"),
        ],
    ).sort({"Number_Games_World_Tournament": "desc"})
    teams_kpi.head(100)

.. ipython:: python
    :suppress:

    teams_kpi = all_matchs.groupby(
        ["team1"],
        [
            fun.sum(all_matchs["World_Tournament"])._as("Number_Games_World_Tournament"),
            fun.sum(all_matchs["Continental_Tournament"])._as("Number_Games_Continental_Tournament"),
            fun.avg(fun.decode(all_matchs["World_Tournament"], 1, all_matchs["Victory_team1"]))._as("Percent_Victory_World_Tournament"),
            fun.avg(fun.decode(all_matchs["Continental_Tournament"], 1, all_matchs["Victory_team1"]))._as("Percent_Victory_Continental_Tournament"),
            fun.avg(fun.case_when((all_matchs["home_team_id"] == 1) & (all_matchs["World_Tournament"] == 0) & (all_matchs["Continental_Tournament"] == 0), all_matchs["Victory_team1"], None))._as("Percent_Victory_Home"),
            fun.avg(fun.case_when((all_matchs["home_team_id"] != 1) & (all_matchs["World_Tournament"] == 0) & (all_matchs["Continental_Tournament"] == 0), all_matchs["Victory_team1"], None))._as("Percent_Victory_Away"),
            fun.avg(all_matchs["Victory_team1"])._as("Percent_Victory"),
            fun.avg(all_matchs["Draw"])._as("Percent_Draw"),
            fun.avg(all_matchs["team1_score"])._as("Avg_goals"),
            fun.avg(all_matchs["team2_score"])._as("Avg_goals_conceded"),
        ],
    ).sort({"Number_Games_World_Tournament": "desc"})
    res = teams_kpi.head(100)
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_clean_12.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_clean_12.html

We can join the different information about the cup winners to enrich our dataset. We'll be using this later, so let's export it to our Vertica database.

.. code-block:: python

    vp.drop("teams_kpi", method = "table")
    teams_kpi = teams_kpi.join(
        football_cup_winners,
        on = {"team1": "winner"},
        how = "left",
        expr2 = [
            "nb_World_Cup", 
            "nb_Continental_Cup",
        ],
    ).to_db("teams_kpi", relation_type = "table")
    teams_kpi.head(100)

.. ipython:: python
    :suppress:

    vp.drop("teams_kpi", method = "table")
    teams_kpi = teams_kpi.join(
        football_cup_winners,
        on = {"team1": "winner"},
        how = "left",
        expr2 = [
            "nb_World_Cup", 
            "nb_Continental_Cup",
        ],
    ).to_db("teams_kpi", relation_type = "table")
    res = teams_kpi.head(100)
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_clean_kpi_final.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_clean_kpi_final.html

Let's add each team's confederation to our dataset.

.. code-block:: python

    teams_kpi = teams_kpi.join(
        confederation,
        how = "left",
        on = {"team1": "team"},
        expr2 = ["confederation"],
    )
    teams_kpi.head(100)

.. ipython:: python
    :suppress:

    teams_kpi = teams_kpi.join(
        confederation,
        how = "left",
        on = {"team1": "team"},
        expr2 = ["confederation"],
    )
    res = teams_kpi.head(100)
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_clean_kpi_final_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_clean_kpi_final_1.html

Since clustering will use different statistics, we need to normalize the data. We'll also create a dummy that will equal 1 if the team won at least one World Cup.

.. code-block:: python

    teams_kpi.normalize(
        columns = [
            "Number_Games_Continental_Tournament", 
            "Number_Games_World_Tournament",
            "nb_Continental_Cup",
        ],
        method = "minmax",
    )
    teams_kpi["Word_Cup_Victory"] = teams_kpi["nb_World_Cup"] > 0
    teams_kpi["Word_Cup_Victory"].astype("int")

.. ipython:: python
    :suppress:

    teams_kpi.normalize(
        columns = [
            "Number_Games_Continental_Tournament", 
            "Number_Games_World_Tournament",
            "nb_Continental_Cup",
        ],
        method = "minmax",
    )
    teams_kpi["Word_Cup_Victory"] = teams_kpi["nb_World_Cup"] > 0
    res = teams_kpi["Word_Cup_Victory"].astype("int")
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_clean_kpi_final_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_clean_kpi_final_2.html

Some data is missing; this is because only top teams won major tournaments. Besides, some non-professional teams may not have a stadium.

.. code-block:: python

    teams_kpi.count()

.. ipython:: python
    :suppress:

    res = teams_kpi.count()
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_clean_kpi_final_3.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_clean_kpi_final_3.html

Let's impute the missing values by 0.

.. code-block:: python

    teams_kpi.fillna(
        {
            "Percent_Victory_Away": 0,
            "Percent_Victory_Home": 0,
            "Percent_Victory_Continental_Tournament": 0,
            "Percent_Victory_World_Tournament": 0,
            "nb_World_Cup": 0,
            "Word_Cup_Victory": 0,
            "nb_Continental_Cup": 0,
            "confederation": "OFC",
        },
    )

.. ipython:: python
    :suppress:

    res = teams_kpi.fillna(
        {
            "Percent_Victory_Away": 0,
            "Percent_Victory_Home": 0,
            "Percent_Victory_Continental_Tournament": 0,
            "Percent_Victory_World_Tournament": 0,
            "nb_World_Cup": 0,
            "Word_Cup_Victory": 0,
            "nb_Continental_Cup": 0,
            "confederation": "OFC",
        },
    )
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_clean_kpi_final_4.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_clean_kpi_final_4.html

Let's export the result to our Vertica database.

.. code-block:: python

    vp.drop("football_clustering", method = "table")
    teams_kpi.to_db(
        "football_clustering", 
        relation_type = "table",
        inplace = True,
    )

.. ipython:: python
    :suppress:

    vp.drop("football_clustering", method = "table")
    teams_kpi.to_db(
        "football_clustering", 
        relation_type = "table",
        inplace = True,
    )
    res = teams_kpi
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_clean_kpi_football_clustering_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_clean_kpi_football_clustering_1.html

Team Rankings with k-means
---------------------------

To compute a ``k-means`` model, we need to find a value for ``k``. Let's draw an :py:func:`~verticapy.machine_learning.model_selection.elbow` curve to find a suitable number of clusters.

.. code-block:: python

    from verticapy.machine_learning.model_selection import elbow

    predictors = [
        'Word_Cup_Victory',  
        'nb_Continental_Cup',
        'Number_Games_World_Tournament',    
        'Number_Games_Continental_Tournament',   
        'Percent_Victory_World_Tournament',
        'Percent_Victory_Continental_Tournament',  
        'Percent_Victory_Home',
        'Percent_Victory_Away',
    ]
    elbow(
        "football_clustering",
        predictors,
        n_cluster = (1, 11),
    )

.. ipython:: python
    :suppress:
    :okwarning:

    from verticapy.machine_learning.model_selection import elbow

    predictors = [
        'Word_Cup_Victory',  
        'nb_Continental_Cup',
        'Number_Games_World_Tournament',    
        'Number_Games_Continental_Tournament',   
        'Percent_Victory_World_Tournament',
        'Percent_Victory_Continental_Tournament',  
        'Percent_Victory_Home',
        'Percent_Victory_Away',
    ]
    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = elbow(
        "football_clustering",
        predictors,
        n_cluster = (1, 11),
    )
    fig.write_html("SPHINX_DIRECTORY/figures/examples_football_elbow_1.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_elbow_1.html

6 seems to be a good number of clusters. To help the algorithm to converge to meaningful clusters, we can initialize the clusters with different types of centroid levels. For example, we can associate very good teams (champions) to World Cups Winners, good teams to continental Cup Winners, etc. This will let us to properly weigh the performance of each team relatve to the strength of their region.

.. ipython:: python

    from verticapy.machine_learning.vertica import KMeans

        # w_cup c_cup w_games c_games w_vict c_vict h_vict a_vict
    init =  [
        (0,    0,       0,  0.05,      0,    0,      0, 0.05), # very bad
        (0,    0,       0,  0.30,      0, 0.25,   0.30, 0.10), # bad
        (0,    0,    0.05,  0.40,   0.15, 0.35,   0.40, 0.20), # outsiders
        (0, 0.10,    0.15,  0.50,   0.20, 0.45,   0.50, 0.30), # good
        (0, 0.20,    0.30,  0.40,   0.40, 0.55,   0.60, 0.40), # strong
        (1,  0.5,       1,  0.80,   0.70, 0.65,   0.75, 0.55), # champions
    ]
    model_kmeans = KMeans(
        n_cluster = 6,
        init = init,
    )
    model_kmeans.fit("football_clustering", predictors)
    model_kmeans.clusters_

Let's add the prediction to the :py:mod:`~verticapy.vDataFrame`.

.. code-block:: python

    model_kmeans.predict(
        teams_kpi, 
        name = "fifa_rank",
    )

.. ipython:: python
    :suppress:

    res = model_kmeans.predict(
        teams_kpi, 
        name = "fifa_rank",
    )
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_clean_kpi_model_kmeans_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_clean_kpi_model_kmeans_1.html

Let's look at the strongest group, which includes well-known teams like Argentina, Brazil, and France.

.. code-block:: python

    teams_kpi.search(
        conditions = [teams_kpi["fifa_rank"] == 5],
        usecols = ["team1", "fifa_rank"],
        order_by = ["fifa_rank"],
    ).head(10)

.. ipython:: python
    :suppress:

    res = teams_kpi.search(
        conditions = [teams_kpi["fifa_rank"] == 5],
        usecols = ["team1", "fifa_rank"],
        order_by = ["fifa_rank"],
    ).head(10)
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_clean_kpi_kmeans_10.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_clean_kpi_kmeans_10.html

The weakest group includes less well-known teams.

.. code-block:: python

    teams_kpi.search(
        conditions = [teams_kpi["fifa_rank"] == 0],
        usecols = ["team1", "fifa_rank"],
        order_by = ["fifa_rank"],
    ).head(10)

.. ipython:: python
    :suppress:

    res = teams_kpi.search(
        conditions = [teams_kpi["fifa_rank"] == 0],
        usecols = ["team1", "fifa_rank"],
        order_by = ["fifa_rank"],
    ).head(10)
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_clean_kpi_kmeans_11.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_clean_kpi_kmeans_11.html

A bubble plot will let us visualize the differences in strength between each confederation.

We can see the strongest group at the top right of the graphic and weakest teams at the bottom left. Some teams may be very good in their location but very bad in World Tournaments. They are mainly at the bottom right of the graph.

.. code-block:: python

    teams_kpi.scatter(
        [
            "Percent_Victory_Continental_Tournament", 
            "Percent_Victory_World_Tournament",
        ],
        size = "fifa_rank",
        by = "confederation",
    )

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = teams_kpi.scatter(
        [
            "Percent_Victory_Continental_Tournament", 
            "Percent_Victory_World_Tournament",
        ],
        size = "fifa_rank",
        by = "confederation",
    )
    fig.write_html("SPHINX_DIRECTORY/figures/examples_football_scatter_1.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_scatter_1.html

We can also look at the Percent of Victory by rank to confirm our hypothesis.

.. code-block:: python

    teams_kpi.scatter(
        [
            "Percent_Victory_Continental_Tournament", 
            "Percent_Victory_World_Tournament",
        ],
        size = "Percent_Victory",
        by = "fifa_rank",
    )

.. ipython:: python
    :suppress:
    :okwarning:

    fig = teams_kpi.scatter(
        [
            "Percent_Victory_Continental_Tournament", 
            "Percent_Victory_World_Tournament",
        ],
        size = "Percent_Victory",
        by = "fifa_rank",
    )
    fig.write_html("SPHINX_DIRECTORY/figures/examples_football_scatter_2.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_scatter_2.html

A box plot can also show us the differences in skill between teams. We can look at rank 1, where the percent of victory is high because of the confederation.

Note that the best team in a weaker confederation might not be particularly strong, but still have a high Percent of Victory.

.. code-block:: python

    teams_kpi["Percent_Victory"].boxplot(by = "fifa_rank")

.. ipython:: python
    :suppress:
    :okwarning:

    fig = teams_kpi["Percent_Victory"].boxplot(by = "fifa_rank")
    fig.write_html("SPHINX_DIRECTORY/figures/examples_football_boxplot_2.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_boxplot_2.html

Let's export the KPIs to our Vertica database.

.. code-block:: python

    vp.drop(
        "team_kpi", 
        method = "table",
    )
    teams_kpi.to_db(
        name = "team_kpi",
        relation_type = "table",
        inplace = True,
    )

.. ipython:: python
    :suppress:

    vp.drop(
        "team_kpi", 
        method = "table",
    )
    teams_kpi.to_db(
        name = "team_kpi",
        relation_type = "table",
        inplace = True,
    )
    res = teams_kpi
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_clean_kpi_kmeans_13.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_clean_kpi_kmeans_13.html

Features Engineering
---------------------

Many very interesting features can be to use to evaluate each team. Moving windows of the previous games can drastically improve our model.

Since a team can by a home or away team, we'll intervert the away and home teams. By using this technique, we will never get twice the same game and we will get the proper moving windows.

.. ipython:: python

    football = vp.vDataFrame("football_clean")
    football["home_team"].rename("team1");
    football["home_score"].rename("team1_score");
    football["away_team"].rename("team2");
    football["away_score"].rename("team2_score");
    # will be to use to filter the data after the features engineering
    football["match_sample"] = "1";

    football2 = vp.vDataFrame("football_clean");
    football2["home_team"].rename("team2");
    football2["home_score"].rename("team2_score");
    football2["away_team"].rename("team1");
    football2["away_score"].rename("team1_score");
    # will be to use to filter the data after the features engineering
    football2["match_sample"] = "2";

    # Merging the 2 interverted datasets
    all_matchs = football.append(football2);

Let's add the different KPIs to our dataset.

.. ipython:: python

    all_matchs = all_matchs.join(
        teams_kpi,
        on = {"team1": "team1"},
        how = "left",
        expr2 = [
            "nb_World_Cup AS nb_World_Cup_1",
            "fifa_rank AS fifa_rank_1",
            "Avg_goals AS Avg_goals_1",
            "Percent_Draw AS Percent_Draw_1",
            "Number_Games_World_Tournament AS Number_Games_World_Tournament_1",
            "Percent_Victory_World_Tournament AS Percent_Victory_World_Tournament_1",
            "Percent_Victory_Away AS Percent_Victory_Away_1",
            "Percent_Victory_Continental_Tournament AS Percent_Victory_Continental_Tournament_1",
            "confederation AS confederation_1",
            "Percent_Victory_Home AS Percent_Victory_Home_1",
            "Avg_goals_conceded AS Avg_goals_conceded_1",
            "Number_Games_Continental_Tournament AS Number_Games_Continental_Tournament_1",
            "nb_Continental_Cup AS nb_Continental_Cup_1",
            "Percent_Victory AS Percent_Victory_1",
        ],
    )
    all_matchs = all_matchs.join(
        teams_kpi,
        on = {"team2": "team1"},
        how = "left",
        expr2 = [
            "nb_World_Cup AS nb_World_Cup_2",
            "fifa_rank AS fifa_rank_2",
            "Avg_goals AS Avg_goals_2",
            "Percent_Draw AS Percent_Draw_2",
            "Number_Games_World_Tournament AS Number_Games_World_Tournament_2",
            "Percent_Victory_World_Tournament AS Percent_Victory_World_Tournament_2",
            "Percent_Victory_Away AS Percent_Victory_Away_2",
            "Percent_Victory_Continental_Tournament AS Percent_Victory_Continental_Tournament_2",
            "confederation AS confederation_2",
            "Percent_Victory_Home AS Percent_Victory_Home_2",
            "Avg_goals_conceded AS Avg_goals_conceded_2",
            "Number_Games_Continental_Tournament AS Number_Games_Continental_Tournament_2",
            "nb_Continental_Cup AS nb_Continental_Cup_2",
            "Percent_Victory AS Percent_Victory_2",
        ],
    )

We can add dumies to do aggregations on the different games.

.. code-block:: python

    all_matchs["victory_team1"] = all_matchs["team1_score"] > all_matchs["team2_score"]
    all_matchs["victory_team1"].astype("int")
    all_matchs["draw"] = all_matchs["team1_score"] == all_matchs["team2_score"]
    all_matchs["draw"].astype("int")
    all_matchs["victory_team2"] = all_matchs["team1_score"] < all_matchs["team2_score"]
    all_matchs["victory_team2"].astype("int")

.. ipython:: python
    :suppress:

    all_matchs["victory_team1"] = all_matchs["team1_score"] > all_matchs["team2_score"]
    all_matchs["victory_team1"].astype("int")
    all_matchs["draw"] = all_matchs["team1_score"] == all_matchs["team2_score"]
    all_matchs["draw"].astype("int")
    all_matchs["victory_team2"] = all_matchs["team1_score"] < all_matchs["team2_score"]
    res = all_matchs["victory_team2"].astype("int")
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_clean_kpi_kmeans_15.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_clean_kpi_kmeans_15.html

Let's use moving windows to compute some additional features.

The teams' performance in their recent games
+++++++++++++++++++++++++++++++++++++++++++++

.. code-block:: python

    # TEAM 1

    # Victory 10 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-10, -1),
        columns = "victory_team1",
        by = ["team1"],
        order_by = ["date"],
        name = "avg_victory_team1_1_10",
    )
    # Victory 3 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-3, -1),
        columns = "victory_team1",
        by = ["team1"],
        order_by = ["date"],
        name = "avg_victory_team1_1_3",
    )
    # Draw 5 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-5, -1),
        columns = "draw",
        by = ["team1"],
        order_by = ["date"],
        name = "avg_draw_team1_1_5",
    )

    # TEAM 2

    # Victory 10 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-10, -1),
        columns = "victory_team2",
        by = ["team2"],
        order_by = ["date"],
        name = "avg_victory_team2_1_10",
    )
    # Victory 3 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-3, -1),
        columns = "victory_team2",
        by = ["team2"],
        order_by = ["date"],
        name = "avg_victory_team2_1_3",
    )
    # Draw 5 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-5, -1),
        columns = "draw",
        by = ["team2"],
        order_by = ["date"],
        name = "avg_draw_team2_1_5",
    )

.. ipython:: python
    :suppress:

    # TEAM 1

    # Victory 10 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-10, -1),
        columns = "victory_team1",
        by = ["team1"],
        order_by = ["date"],
        name = "avg_victory_team1_1_10",
    )
    # Victory 3 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-3, -1),
        columns = "victory_team1",
        by = ["team1"],
        order_by = ["date"],
        name = "avg_victory_team1_1_3",
    )
    # Draw 5 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-5, -1),
        columns = "draw",
        by = ["team1"],
        order_by = ["date"],
        name = "avg_draw_team1_1_5",
    )

    # TEAM 2

    # Victory 10 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-10, -1),
        columns = "victory_team2",
        by = ["team2"],
        order_by = ["date"],
        name = "avg_victory_team2_1_10",
    )
    # Victory 3 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-3, -1),
        columns = "victory_team2",
        by = ["team2"],
        order_by = ["date"],
        name = "avg_victory_team2_1_3",
    )
    # Draw 5 previous games
    res = all_matchs.rolling(
        func = "avg",
        window = (-5, -1),
        columns = "draw",
        by = ["team2"],
        order_by = ["date"],
        name = "avg_draw_team2_1_5",
    )
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_clean_kpi_kmeans_16.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_clean_kpi_kmeans_16.html

The teams' performance in the last same tournament
+++++++++++++++++++++++++++++++++++++++++++++++++++

.. code-block:: python

    # TEAM 1

    # Victory 10 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-10, -1),
        columns = "victory_team1",
        by = ["team1", "tournament"],
        order_by = ["date"],
        name = "avg_victory_same_tournament_team1_1_10",
    )
    # Victory 3 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-3, -1),
        columns = "victory_team1",
        by = ["team1", "tournament"],
        order_by = ["date"],
        name = "avg_victory_same_tournament_team1_1_3",
    )
    # Draw 5 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-5, -1),
        columns = "draw",
        by = ["team1", "tournament"],
        order_by = ["date"],
        name = "avg_draw_same_tournament_team1_1_5",
    )

    # TEAM 2

    # Victory 10 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-10, -1),
        columns = "victory_team2",
        by = ["team2", "tournament"],
        order_by = ["date"],
        name = "avg_victory_same_tournament_team2_1_10",
    )
    # Victory 3 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-3, -1),
        columns = "victory_team2",
        by = ["team2", "tournament"],
        order_by = ["date"],
        name = "avg_victory_same_tournament_team2_1_3",
    )
    # Draw 5 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-5, -1),
        columns = "draw",
        by = ["team2", "tournament"],
        order_by = ["date"],
        name = "avg_draw_same_tournament_team2_1_5",
    )

.. ipython:: python
    :suppress:

    # TEAM 1

    # Victory 10 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-10, -1),
        columns = "victory_team1",
        by = ["team1", "tournament"],
        order_by = ["date"],
        name = "avg_victory_same_tournament_team1_1_10",
    )
    # Victory 3 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-3, -1),
        columns = "victory_team1",
        by = ["team1", "tournament"],
        order_by = ["date"],
        name = "avg_victory_same_tournament_team1_1_3",
    )
    # Draw 5 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-5, -1),
        columns = "draw",
        by = ["team1", "tournament"],
        order_by = ["date"],
        name = "avg_draw_same_tournament_team1_1_5",
    )

    # TEAM 2

    # Victory 10 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-10, -1),
        columns = "victory_team2",
        by = ["team2", "tournament"],
        order_by = ["date"],
        name = "avg_victory_same_tournament_team2_1_10",
    )
    # Victory 3 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-3, -1),
        columns = "victory_team2",
        by = ["team2", "tournament"],
        order_by = ["date"],
        name = "avg_victory_same_tournament_team2_1_3",
    )
    # Draw 5 previous games
    res = all_matchs.rolling(
        func = "avg",
        window = (-5, -1),
        columns = "draw",
        by = ["team2", "tournament"],
        order_by = ["date"],
        name = "avg_draw_same_tournament_team2_1_5",
    )
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_clean_kpi_kmeans_17.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_clean_kpi_kmeans_17.html

Direct Confrontation
+++++++++++++++++++++

.. code-block:: python

    # Victory 5 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-5, -1),
        columns = "victory_team1",
        by = ["team1", "team2"],
        order_by = ["date"],
        name = "avg_victory_direct_team1_1_5",
    )
    # Victory 3 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-3, -1),
        columns = "victory_team1",
        by = ["team1", "team2"],
        order_by = ["date"],
        name = "avg_victory_direct_team1_1_3",
    )
    # Draw 5 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-5, -1),
        columns = "draw",
        by = ["team1", "team2"],
        order_by = ["date"],
        name = "avg_draw_direct_team1_1_5",
    )

.. ipython:: python
    :suppress:

    # Victory 5 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-5, -1),
        columns = "victory_team1",
        by = ["team1", "team2"],
        order_by = ["date"],
        name = "avg_victory_direct_team1_1_5",
    )
    # Victory 3 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-3, -1),
        columns = "victory_team1",
        by = ["team1", "team2"],
        order_by = ["date"],
        name = "avg_victory_direct_team1_1_3",
    )
    # Draw 5 previous games
    res = all_matchs.rolling(
        func = "avg",
        window = (-5, -1),
        columns = "draw",
        by = ["team1", "team2"],
        order_by = ["date"],
        name = "avg_draw_direct_team1_1_5",
    )
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_clean_kpi_kmeans_19.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_clean_kpi_kmeans_19.html

Games against an opponents with the same rank
++++++++++++++++++++++++++++++++++++++++++++++

.. code-block:: python

    # TEAM 1

    # Victory 5 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-5, -1),
        columns = "victory_team1",
        by = ["team1", "fifa_rank_2"],
        order_by = ["date"],
        name = "avg_victory_rank2_team1_1_5",
    )
    # Draw 5 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-5, -1),
        columns = "draw",
        by = ["team1", "fifa_rank_2"],
        order_by = ["date"],
        name = "avg_draw_rank2_team1_1_5",
    )

    # TEAM 2

    # Victory 5 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-5, -1),
        columns = "victory_team2",
        by = ["team2", "fifa_rank_1"],
        order_by = ["date"],
        name = "avg_victory_rank1_team2_1_5",
    )
    # Draw 5 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-5, -1),
        columns = "draw",
        by = ["team2", "fifa_rank_1"],
        order_by = ["date"],
        name = "avg_draw_rank1_team2_1_5",
    )

.. ipython:: python
    :suppress:

    # TEAM 1

    # Victory 5 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-5, -1),
        columns = "victory_team1",
        by = ["team1", "fifa_rank_2"],
        order_by = ["date"],
        name = "avg_victory_rank2_team1_1_5",
    )
    # Draw 5 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-5, -1),
        columns = "draw",
        by = ["team1", "fifa_rank_2"],
        order_by = ["date"],
        name = "avg_draw_rank2_team1_1_5",
    )

    # TEAM 2

    # Victory 5 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-5, -1),
        columns = "victory_team2",
        by = ["team2", "fifa_rank_1"],
        order_by = ["date"],
        name = "avg_victory_rank1_team2_1_5",
    )
    # Draw 5 previous games
    res = all_matchs.rolling(
        func = "avg",
        window = (-5, -1),
        columns = "draw",
        by = ["team2", "fifa_rank_1"],
        order_by = ["date"],
        name = "avg_draw_rank1_team2_1_5",
    )
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_clean_kpi_kmeans_21.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_clean_kpi_kmeans_21.html

Games between teams with rank 1 and rank 2
+++++++++++++++++++++++++++++++++++++++++++

.. code-block:: python

    # Victory 5 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-5, -1),
        columns = "victory_team1",
        by = ["fifa_rank_1", "fifa_rank_2"],
        order_by = ["date"],
        name = "avg_victory_rank1_rank2_team1_1_5",
    )
    # Draw 5 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-5, -1),
        columns = "draw",
        by = ["fifa_rank_1", "fifa_rank_2"],
        order_by = ["date"],
        name = "avg_draw_rank1_rank2_team1_1_5",
    )

.. ipython:: python
    :suppress:

    # Victory 5 previous games
    all_matchs.rolling(
        func = "avg",
        window = (-5, -1),
        columns = "victory_team1",
        by = ["fifa_rank_1", "fifa_rank_2"],
        order_by = ["date"],
        name = "avg_victory_rank1_rank2_team1_1_5",
    )
    # Draw 5 previous games
    res = all_matchs.rolling(
        func = "avg",
        window = (-5, -1),
        columns = "draw",
        by = ["fifa_rank_1", "fifa_rank_2"],
        order_by = ["date"],
        name = "avg_draw_rank1_rank2_team1_1_5",
    )
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_clean_kpi_kmeans_22.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_clean_kpi_kmeans_22.html

Before we use the 'neutral' variable with our model, we should convert it to an integer.

We need also to create our response column: the outcome of the game.

.. code-block:: python

    all_matchs["neutral"].astype("int")
    all_matchs.case_when(
        "result",
        all_matchs["team1_score"] > all_matchs["team2_score"], "1",
        all_matchs["team1_score"] < all_matchs["team2_score"], "2", 
        "X",
    )

.. ipython:: python
    :suppress:

    all_matchs["neutral"].astype("int")
    res = all_matchs.case_when(
        "result",
        all_matchs["team1_score"] > all_matchs["team2_score"], "1",
        all_matchs["team1_score"] < all_matchs["team2_score"], "2", 
        "X",
    )
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_clean_kpi_kmeans_23.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_clean_kpi_kmeans_23.html

We have some missing values here. This might be because the two teams never played together, the competition was one or both teams' first, etc.

.. code-block:: python

    all_matchs.count()

.. ipython:: python
    :suppress:

    res = all_matchs.count()
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_clean_kpi_count_final_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_clean_kpi_count_final_1.html

We need to impute these missing values.

.. code-block:: python

    all_matchs["avg_victory_direct_team1_1_5"] = fun.coalesce(
        all_matchs["avg_victory_direct_team1_1_5"],
        all_matchs["avg_victory_rank2_team1_1_5"],
        all_matchs["avg_victory_rank1_rank2_team1_1_5"],
    )
    all_matchs["avg_victory_direct_team1_1_3"] = fun.coalesce(
        all_matchs["avg_victory_direct_team1_1_3"],
        all_matchs["avg_victory_rank2_team1_1_5"],
        all_matchs["avg_victory_rank1_rank2_team1_1_5"],
    )
    all_matchs["avg_draw_direct_team1_1_5"] = fun.coalesce(
        all_matchs["avg_draw_direct_team1_1_5"],
        all_matchs["avg_draw_rank2_team1_1_5"],
        all_matchs["avg_draw_rank1_rank2_team1_1_5"],
    )
    all_matchs["avg_victory_same_tournament_team1_1_10"].fillna(expr = "avg_victory_team1_1_10")
    all_matchs["avg_victory_same_tournament_team1_1_3"].fillna(expr = "avg_victory_team1_1_3")
    all_matchs["avg_draw_same_tournament_team1_1_5"].fillna(expr = "avg_draw_team1_1_5")
    all_matchs["avg_victory_same_tournament_team2_1_10"].fillna(expr = "avg_victory_team2_1_10")
    all_matchs["avg_victory_same_tournament_team2_1_3"].fillna(expr = "avg_victory_team2_1_3")
    all_matchs["avg_draw_same_tournament_team2_1_5"].fillna(expr = "avg_draw_team2_1_5")

.. ipython:: python
    :suppress:

    all_matchs["avg_victory_direct_team1_1_5"] = fun.coalesce(
        all_matchs["avg_victory_direct_team1_1_5"],
        all_matchs["avg_victory_rank2_team1_1_5"],
        all_matchs["avg_victory_rank1_rank2_team1_1_5"],
    )
    all_matchs["avg_victory_direct_team1_1_3"] = fun.coalesce(
        all_matchs["avg_victory_direct_team1_1_3"],
        all_matchs["avg_victory_rank2_team1_1_5"],
        all_matchs["avg_victory_rank1_rank2_team1_1_5"],
    )
    all_matchs["avg_draw_direct_team1_1_5"] = fun.coalesce(
        all_matchs["avg_draw_direct_team1_1_5"],
        all_matchs["avg_draw_rank2_team1_1_5"],
        all_matchs["avg_draw_rank1_rank2_team1_1_5"],
    )
    all_matchs["avg_victory_same_tournament_team1_1_10"].fillna(expr = "avg_victory_team1_1_10")
    all_matchs["avg_victory_same_tournament_team1_1_3"].fillna(expr = "avg_victory_team1_1_3")
    all_matchs["avg_draw_same_tournament_team1_1_5"].fillna(expr = "avg_draw_team1_1_5")
    all_matchs["avg_victory_same_tournament_team2_1_10"].fillna(expr = "avg_victory_team2_1_10")
    all_matchs["avg_victory_same_tournament_team2_1_3"].fillna(expr = "avg_victory_team2_1_3")
    res = all_matchs["avg_draw_same_tournament_team2_1_5"].fillna(expr = "avg_draw_team2_1_5")
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_clean_kpi_all_matchs_final_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_clean_kpi_all_matchs_final_1.html

Let's export the result to our Vertica database using the variable ``match_sample`` to avoid counting the same game twice.

.. code-block:: python

    vp.drop("football_train", method = "table")
    all_matchs.to_db(
        name = "football_train",
        relation_type = "table",
        db_filter = (fun.year(all_matchs["date"]) <= 2015) & (fun.year(all_matchs["date"]) > 1980) & (all_matchs["match_sample"] == 1),
    )

    vp.drop("football_test", method = "table")
    all_matchs.to_db(
        name = "football_test",
        relation_type = "table",
        db_filter = (fun.year(all_matchs["date"]) > 2015) & (all_matchs["match_sample"] == 1),
    )

.. ipython:: python
    :suppress:

    vp.drop("football_train", method = "table")
    all_matchs.to_db(
        name = "football_train",
        relation_type = "table",
        db_filter = (fun.year(all_matchs["date"]) <= 2015) & (fun.year(all_matchs["date"]) > 1980) & (all_matchs["match_sample"] == 1),
    )

    vp.drop("football_test", method = "table")
    all_matchs.to_db(
        name = "football_test",
        relation_type = "table",
        db_filter = (fun.year(all_matchs["date"]) > 2015) & (all_matchs["match_sample"] == 1),
    )
    res = all_matchs
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_clean_kpi_all_matchs_final_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_clean_kpi_all_matchs_final_2.html

Machine Learning
-----------------

It's time to make predictions about the outcomes of games. We have a lot of variables, so we need trees deep enough to pick up the most important features. We also need to consider a minimum number of games in each leaf to avoid over-fitting.

.. ipython:: python
    :okwarning:

    predictors = all_matchs.get_columns(
        exclude_columns = [
            "match_sample", 
            "team2_score", 
            "team1_score", 
            "date",
            "city",
            "country",
            "result",
            "victory_team1",
            "victory_team2",
            "draw",
        ],
    )

    from verticapy.machine_learning.vertica import RandomForestClassifier

    model = RandomForestClassifier(
        max_depth = 25,
        n_estimators = 20,
        sample = 0.7,
        nbins = 50,
        max_leaf_nodes = 11000,
        min_samples_leaf = 3,
    )
    model.fit(
        "football_train",
        predictors,
        "result",
        "football_test",
    )

.. code-block:: python

    model.classification_report()

.. ipython:: python
    :suppress:

    res = model.classification_report()
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_clean_kpi_ml_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_clean_kpi_ml_1.html

Our model is excellent! 57% of accuracy on 3 categories - it's almost twice as good as a random model.

.. ipython:: python

    model.score(metric = "accuracy")

Looking at the importance of each feature, it seems like direct confrontations and victories against teams of another rank seem to be the strongest indicators of a team's success.

.. code-block:: python

    model.features_importance()

.. ipython:: python
    :suppress:

    fig = model.features_importance()
    fig.write_html("SPHINX_DIRECTORY/figures/examples_football_features_importance.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_features_importance.html

Let's add the predictions to the :py:mod:`~verticapy.vDataFrame`.

Draws are pretty rare, so we'll only consider them if a tie was very likely to occur.

.. code-block:: python

    test = vp.vDataFrame("football_test")
    model.predict_proba(test, name = "prob_1", pos_label = "1")
    model.predict_proba(test, name = "prob_X", pos_label = "X")
    model.predict_proba(test, name = "prob_2", pos_label = "2")
    test.case_when(
        "prediction",
        test["prob_1"] > test["prob_2"] + 0.05, "1",
        test["prob_2"] > test["prob_1"] + 0.05, "2",
        (test["prob_X"] > test["prob_1"]) & (test["prob_X"] > test["prob_2"]), "X",
        fun.abs(test["prob_1"] - test["prob_2"]) < 0.03, "X",
        test["prob_1"] > test["prob_2"], "1",
        test["prob_1"] < test["prob_2"], "2",
    )

.. ipython:: python
    :suppress:

    test = vp.vDataFrame("football_test")
    model.predict_proba(test, name = "prob_1", pos_label = "1")
    model.predict_proba(test, name = "prob_X", pos_label = "X")
    model.predict_proba(test, name = "prob_2", pos_label = "2")
    res = test.case_when(
        "prediction",
        test["prob_1"] > test["prob_2"] + 0.05, "1",
        test["prob_2"] > test["prob_1"] + 0.05, "2",
        (test["prob_X"] > test["prob_1"]) & (test["prob_X"] > test["prob_2"]), "X",
        fun.abs(test["prob_1"] - test["prob_2"]) < 0.03, "X",
        test["prob_1"] > test["prob_2"], "1",
        test["prob_1"] < test["prob_2"], "2",
    )
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_ml_case_when_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_ml_case_when_1.html

Let's look at our predictions for the 2018 World Cup.

.. code-block:: python

    test.search(
        conditions = [test["tournament"] == 'FIFA World Cup'], 
        usecols = [
            "date",
            "team1", 
            "result", 
            "prediction", 
            "team2", 
            "prob_1", 
            "prob_X", 
            "prob_2",
        ],
        order_by = ["date"],
    ).head(128)

.. ipython:: python
    :suppress:

    res = test.search(
        conditions = [test["tournament"] == 'FIFA World Cup'], 
        usecols = [
            "date",
            "team1", 
            "result", 
            "prediction", 
            "team2", 
            "prob_1", 
            "prob_X", 
            "prob_2",
        ],
        order_by = ["date"],
    ).head(128)
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_ml_search_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_ml_search_1.html

Fantastic: we built a very efficient model which predicted that France will win almost all of its games (except the game against Argentina which is really hard to predict). In reality, France did indeed win the 2018 World Cup!

.. code-block:: python

    test.search(
        conditions = [
            test["tournament"] == 'FIFA World Cup',
            (test["team1"] == 'France') | (test["team2"] == 'France'),
        ], 
        usecols = [
            "date",
            "team1", 
            "result", 
            "prediction", 
            "team2", 
            "prob_1", 
            "prob_X", 
            "prob_2",
        ],
        order_by = ["date"],
    ).head(128)

.. ipython:: python
    :suppress:

    res = test.search(
        conditions = [
            test["tournament"] == 'FIFA World Cup',
            (test["team1"] == 'France') | (test["team2"] == 'France'),
        ], 
        usecols = [
            "date",
            "team1", 
            "result", 
            "prediction", 
            "team2", 
            "prob_1", 
            "prob_X", 
            "prob_2",
        ],
        order_by = ["date"],
    ).head(128)
    html_file = open("SPHINX_DIRECTORY/figures/examples_football_ml_search_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_football_ml_search_2.html

Conclusion
-----------

We've solved our problem in a Pandas-like way, all without ever loading data into memory!