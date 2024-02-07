================
Pipelines (Beta)
================

Vertica Pipelines is an open source platform for managing data
scientists’ machine learning pipelines. They are built on a
human-readable data format: **YAML**.

==============================
Setting Up Your First Pipeline
==============================

Requirements
~~~~~~~~~~~~
To begin, you must:
 
* Have access to a machine that has Vertica installed
* Install Python on your machine
* Install Verticapy

Create Your First YAML files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. important::
   
   **THE BASIC RULES OF ANY YAML FILE:**
   
   - YAML is case sensitive
   - the files should have **.yaml** as the extension, 
   - YAML does not allow the use of tabs while creating YAML files

The information in connection.yaml will be the same you use in Verticapy.

.. code:: bash

   vim connection.yaml

.. code:: yaml

   # Replace with your information #
   host: "127.0.0.1"
   port: "5433"
   database: ""
   password: ""
   user: "dbadmin"

The information in your pipeline yaml file will outline the flow of data.

.. code:: bash

   vim gettingStarted.yaml

.. code:: yaml

   schema: "public"
   pipeline: "gettingStarted"
   table: public.example_table
   # steps:
       # ingest: 
       # transform: 
       # train:
       # test:

Run the Parser
~~~~~~~~~~~~~~
 
The parser will follow the general format:

.. code:: bash

    python -m verticapy.pipeline.parser [connection_file] [input_file] -o [output file]

Both of the following will generate a sql file: **gettingStarted.sql**

.. code:: bash

   python -m verticapy.pipeline.parser connection.yaml gettingStarted.yaml

.. code:: bash

   python -m verticapy.pipeline.parser connection.yaml gettingStarted.yaml -o gettingStarted.sql

Dropping the Pipeline
~~~~~~~~~~~~~~~~~~~~~
 
If you are done with the pipeline and want to drop all ingestions, views, models, or stored procedures associated with it, you can do either of the following:

**In VSQL terminal:**

.. code:: bash

   CALL drop_pipeline([schema name], [pipeline name]);
 
**In VerticaPy cell:**

.. code:: sql

   %%sql
   CALL drop_pipeline([schema name], [pipeline name]);

  
For the example above running the sql would drop the pipeline:

.. code:: bash

    CALL drop_pipeline('public', 'gettingStarted');

Now you should be ready to quickly build new pipelines!

=============
Documentation
=============

Ingestion
~~~~~~~~~

For more information on how to customize this step: `DataLoader Parameters <https://docs.vertica.com/latest/en/sql-reference/statements/create-statements/create-data-loader/#arguments>`__ and `Copy Parameters <https://docs.vertica.com/latest/en/sql-reference/statements/copy/>`__.

.. code:: yaml

     ingest:
         from: '~/data/bucket/*'
         delimiter: ','
         retry_limit: 'NONE'
         retention_interval: "'15 days'"

Transform
~~~~~~~~~

For more information on how to customize this step: :ref:`api.vdataframe.features_engineering`.

``public.winequality``

    .. ipython:: python
        :suppress:

        from verticapy.datasets import load_winequality
        html_file = open("SPHINX_DIRECTORY/figures/datasets_loaders_load_winequality.html", "w")
        html_file.write(load_winequality()["density", "pH", "color", "fixed_acidity"]._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_winequality.html
        
 

**Example**

.. code:: yaml

   transform:
       # 1. Existing Column
       col1:
           sql: fixed_acidity
       # 2. Column + Transform
       col2:
           sql: color
           transform_method:
               name: str_count
               params:
                   pat: white
       # 3. Create a new column with Method
       # Note: Don't specify 'name' in params
       col3:
           transform_method:
               name: cummax
               params:
                   column: fixed_acidity
       # 4. Complex Sql
       col4:
           sql: fixed_acidity * density
       # 5. Multiple Params
       col5:
           transform_method:
               name: regexp
               params:
                   column: color
                   pattern: "white"
                   method: "replace"
                   replacement: "NOT white"
       # 6. Multi-Stage Transforms
       col6:
           sql: color
           transform_method1:
               name: str_count
               params:
                   pat: white
           transform_method2:
               name: add
               params:
                   x: 0.5
       # 7. Using Previously Created Columns
       col7:
           sql: col2
           transform_method:
               name: add
               params:
                   x: 0.5

Train
~~~~~

For more information on how to customize this step: :ref:`api.machine_learning.vertica`. 

If you want to train a model, the default setting is to use all the previously created ``cols`` as predictors.
To subtract the specified columns from the default columns use ``exclude``.
To strictly choose subset to overide the default columns use ``include``.
The previous transform example is the basis for this train example:

**Example 1**

.. code:: yaml

     train:    
       method:
           name: RandomForestClassifier
           target: citric_acid
           params:
               n_estimators: 40
               max_depth: 4

**Example 2**

.. code:: yaml

     train:    
       method:
           name: LinearSVC
           target: col2
           exclude: ['col5', 'col2']

**Example 3**

.. code:: yaml

     train:    
       method:
           name: LinearSVC
           target: col2
           include: ['col1', 'col3', 'col4', 'col6', 'col7']

Test
~~~~
For more information on how to customize this step: :ref:`api.machine_learning.metrics`.

You may want to compute metrics for your newly created model. The results are stored in:

``[schema].[pipeline_name]_METRIC_TABLE``

**Example 1**

.. code:: yaml

     test:
       metric1: 
           name: accuracy_score
           y_true: quality
           y_score: prediction
       metric2: 
           name: r2_score
           y_true: quality
           y_score: prediction
       metric3: 
           name: max_error
           y_true: quality
           y_score: prediction


``public.example_METRIC_TABLE``

    .. ipython:: python
        :suppress:

        import verticapy as vp
	vdf = vp.vDataFrame(
		{
			"metric_name": ['accuracy_score', 'r2_score', 'max_error'],
			"metric": [0.0, 0.188352265031045, 3.49495733261932],
		},
	)
        html_file = open("SPHINX_DIRECTORY/figures/pipeline_metric_table.html", "w")
        html_file.write(vdf._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/pipeline_metric_table.html

Scheduler
~~~~~~~~~

For more information to on how to customize this step: `Cron Wiki <https://en.wikipedia.org/wiki/Cron>`__ and `Vertica Schedulers <https://docs.vertica.com/latest/en/sql-reference/statements/create-statements/create-schedule/>`__.

If you would like the ``ingestion`` or ``train`` steps to continously update on a set
schedule use the ``schedule`` key. The schedule follows the cron format.

**Example 1**

.. code:: yaml

     train:
       method:
           name: RandomForestClassifier
           target: survived
           params:
               n_estimators: 40
               max_depth: 4
       schedule: "* * * * *"

**Example 2**

.. code:: yaml

     ingest:
         from: '/bucket/*'
         delimiter: ','
         schedule: "* * * * *"
