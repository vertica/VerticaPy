.. _getting_started:

=================
Getting Started
=================

.. raw:: html

   <div style="text-align: center;">
     <a href="../../..">
       <img src="_static/vp_logo.png" alt="Clickable Image" style="width: 20%;">
     </a>
   </div>
.. raw:: html

    <div style="text-align: center; font-size: 26px; font-family: 'Inter', sans-serif; padding-top: 10px;">
        VerticaPy
    </div>
    <div style="text-align: center; font-size: 18px; font-weight: bold;">
      Python API for Vertica Data Science at Scale
    </div>

The following instructions provide steps to install VerticaPy, Juypter Lab, and Vertica. For information
about installing VerticaPyLab, a custom JuypterLab enviornment that already has VerticaPy installed and 
includes an easy-to-use interface and tools like :py:func:`~verticapy.performance.vertica.qprof`, 
see :ref:`verticapylab_gs`.

Installion Guide
-----------------

Prerequisite
^^^^^^^^^^^^

**Python 3.9+**


VerticaPy runs with Python 3.9 or higher. You can install Python directly from their `website <https://www.python.org/downloads>`_ .


**Jupyter Lab**

If you want to have a nice environment to play with, we recommend you to install the last Jupyter version. You can find all the information to install it in their `website <https://jupyter.org/install>`_

**Vertica 9+**

VerticaPy relies on Vertica 9 or a more recent version.

Vertica is the most advanced analytics data warehouse based on a massively scalable architecture. 
It features the broadest set of analytical functions spanning event and time series, geospatial, end-to-end in-database machine learning, and pattern matching. 

Vertica lets you to easily apply these powerful functions to the largest and most demanding analytical workloads, 
arming you and your customers with predictive business insights faster than any other analytics data warehouse on the market.

Vertica provides a unified analytics platform across major public clouds and on-premises data centers and integrates data in cloud object storage and 
HDFS without forcing you to move any of your data.

To learn more about the Vertica database, check out the `Vertica Official Website <https://www.vertica.com/about/>`_.

If you already have Vertica installed, you can skip this step. Otherwise, you have some options for trying out Vertica for free.

- The easiest way to install Vertica is to use containers on Docker. You can find all the needed information `here <https://hub.docker.com/r/vertica/vertica-k8s>`_.
- If you have a Linux machine, you can install Vertica Community Edition. Please see this `video <https://www.youtube.com/watch?v=D5SbzVVR_Ps&ab_channel=MicroFocusisnowOpenText>`_.
- If you don't have a Linux machine, you can use the Vertica Community Edition VM. In this case, follow the instructions of the `Vertica Community Edition Virtual Machine Installation Guide <https://www.vertica.com/docs/VMs/Vertica_CE_VM_Download_and_Startup_Instructions.pdf>`_.

.. hint::

    You can also install VerticaPyLab which has both Vertica and VerticaPy pre-installed in a docker environment. See :ref:`Install VerticaPyLab`.

Install VerticaPy
^^^^^^^^^^^^^^^^^

To install VerticaPy with all the dependencies, including some geospatial packages such as GeoPandas and Descartes, run the following **pip** command:

>>> pip3 install verticapy[all]

If you do not want to install extra dependencies, you can use the following command:

>>> pip3 install verticapy

To start playing with the API, create a new connection:

.. code-block:: python

    import verticapy as vp

    vp.new_connection(
        {
            "host": "10.211.55.14", 
            "port": "5433", 
            "database": "testdb", 
            "password": "XxX", 
            "user": "dbadmin",
        },
        name = "My_New_Vertica_Connection",
    )

.. note::

    For more information, see: :ref:`connection`.

After a connection is created, you can use the :py:func:`~verticapy.connection.connect` function to reconnect.

.. code-block:: python

    vp.connect("My_New_Vertica_Connection")

Create a VerticaPy schema for native VerticaPy models (that is, models available in VerticaPy, but not Vertica itself):

.. code-block:: python

    vp.create_verticapy_schema()

You can now start playing with the library! A good starting point is the `Quick Start guide <https://github.com/vertica/VerticaPy#quickstart>`_.


.. toctree::
    :hidden:
    :maxdepth: 1
    :titlesonly:

    verticapylab_gs