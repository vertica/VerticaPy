.. _chart_gallery.tree:

=============================
Machine Learning - Tree Plots
=============================

.. Necessary Code Elements

.. ipython:: python
    :suppress:
    :okwarning:

    import random
    import verticapy as vp
    import verticapy.machine_learning.vertica as vml
    import numpy as np

    N = 100 # Number of Records
    k = 10 # step

    # Normal Distributions
    x = np.random.normal(5, 1, round(N / 2))
    y = np.random.normal(3, 1, round(N / 2))
    z = np.random.normal(3, 1, round(N / 2))

    # Creating a vDataFrame with two clusters
    data = vp.vDataFrame({"x": np.concatenate([x, x + k]), "y": np.concatenate([y, y + k]), "z": np.concatenate([z, z + k]), "c": [random.randint(0, 1) for _ in range(N)]})

    # Defining a Tree Based model
    model = vml.RandomForestClassifier()

    # Fitting the model
    model.fit(data, ["x", "y", "z"], "c")


General
-------

In this example, we will demonstrate how to create a model tree visualization using Graphviz. It's important to note that these plots are purely illustrative and are based on generated data.

Let's begin by importing `VerticaPy`.

.. ipython:: python

    import verticapy as vp

Let's also import `numpy` to create a random dataset.

.. ipython:: python

    import numpy as np

Let's generate a dataset using the following data.

.. code-block:: python
        
    N = 100 # Number of Records
    k = 10 # step

    # Normal Distributions
    x = np.random.normal(5, 1, round(N / 2))
    y = np.random.normal(3, 1, round(N / 2))
    z = np.random.normal(3, 1, round(N / 2))

    # Creating a vDataFrame with two clusters
    data = vp.vDataFrame({
        "x": np.concatenate([x, x + k]),
        "y": np.concatenate([y, y + k]),
        "z": np.concatenate([z, z + k]),
        "c": [random.randint(0, 1) for _ in range(N)]
    })

Let's proceed by creating a Random Forest Classifier model using the complete dataset.

.. code-block:: python
    
    # Importing the Vertica ML module
    import verticapy.machine_learning.vertica as vml

    # Defining the Models
    model = vml.RandomForestClassifier()

    # Fitting the models
    model.fit(data, ["x", "y", "z"], "c")

VerticaPy provides the option to create various types of geospatial plots, including scatter plots and heat maps. To leverage these capabilities, it's important to have geospatial data stored within Vertica, specifically in either GEOMETRY or GEOGRAPHY data types. This data forms the foundation for generating insightful geospatial visualizations using VerticaPy.

.. note::
    
    VerticaPy offers a straightforward method to generate tree visualizations using Graphviz, making it easy to interpret and analyze decision tree models. We have plans to further enhance this functionality by extending it to Plotly in the future, providing even more dynamic and interactive visualization options for decision trees.
            
.. tab:: Graphviz

    .. ipython:: python
        :okwarning:

        model.plot_tree(pic_path = "figures/plotting_graphviz_tree")


    .. raw:: html
        
        <!DOCTYPE html>
        <html>
        <head>
            <title>PDF Viewer</title>
            <style>
                #pdf-embed {
                    max-width: 800px; /* Set the maximum width */
                    width: 100%; /* Make it responsive */
                    min-height: 400px;
                    height: auto; /* Maintain aspect ratio */
                }
            </style>
        </head>
        <body>
            <embed id="pdf-embed" src="../../../docs/figures/plotting_graphviz_tree.pdf"/>
        </body>
        </html>




___________________


Chart Customization
-------------------

VerticaPy empowers users with a high degree of flexibility when it comes to tailoring the visual aspects of their plots. 
This customization extends to essential elements such as **color schemes**, **text labels**, and **plot sizes**, as well as a wide range of other attributes that can be fine-tuned to align with specific design preferences and analytical requirements. Whether you want to make your visualizations more visually appealing or need to convey specific insights with precision, VerticaPy's customization options enable you to craft graphics that suit your exact needs.

.. note:: VerticaPy's tree plots, generated using Graphviz, can be tailored to your preferences by utilizing Graphviz parameters. You can explore the full list of available parameters and their descriptions by visiting the following link: `graphviz <https://graphviz.org/doc/info/attrs.html>`_

Example
~~~~~~~

.. tab:: Graphviz

    **Changing different parameters**

    .. ipython:: python
        :okwarning:

        model.plot_tree(
            pic_path = "figures/plotting_graphviz_tree_custom",
            node_style={"shape": "box", "style": "filled"},
            edge_style={"color": "blue"},
            leaf_style={"shape": "circle", "style": "filled"},
        )


    .. raw:: html
        
        <!DOCTYPE html>
        <html>
        <head>
            <title>PDF Viewer</title>
            <style>
                #pdf-embed-2 {
                    max-width: 800px; /* Set the maximum width */
                    width: 100%; /* Make it responsive */
                    min-height: 400px;
                    height: auto; /* Maintain aspect ratio */
                }
            </style>
        </head>
        <body>
            <embed id="pdf-embed-2" src="../../../docs/figures/plotting_graphviz_tree_custom.pdf"/>
        </body>
        </html>


.. hint:: In VerticaPy, when a model consists of multiple trees, you can utilize the `tree_id` parameter to access and analyze specific trees within the model.
