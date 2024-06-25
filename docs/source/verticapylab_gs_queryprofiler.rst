.. _verticapylab_gs.queryprofiler:

=============================
Query Profiler User Interface
=============================

If you already have a Vertica database running and just want to use the QueryProfiler tool 
to analyze your query, then you can do the minimalistic installaion of VerticaPyLab
to achieve this.

Prerequisites
=================

VerticaPyLab requires the following prerequisites:

**For Windows**

- Docker Desktop version (Docker version 18.09 or higher)
- Windows Subsystem for Linux  2 (WSL 2)

**For Mac/Linux**

- Docker version 18.09 or higher `(click here for installation instructions) <https://docs.docker.com/engine/install/>`_

.. note:: To learn how to get the above please look at :ref:`pre_reqs`.


VerticaPyLab Quickstart
========================

The following steps import and launch VerticaPyLab:

1. Start a Linux distribution (WSL 2 for Windows).
2. Clone the VerticaPyLab repository:  
    .. code-block::

      git clone https://github.com/vertica/VerticaPyLab.git

3. Navigate to the cloned directory:  
      .. code-block::
            
        cd VerticaPyLab

4.  Start the JupyterLab container:
      .. code-block::
        
        make verticapylab-start

.. note:: If ``make`` command is not there, you may need to install it:
      .. code-block::

        sudo apt install make

    
    A browser should pop up automatically. If it does not, then you can copy the URL link into any browser. 
    Additionally, if you are trying to connect to your container through a VM/remotely, then please follow the instructions of step 5 here: :ref:`url_issue`.


When you are done with using the container and want to shut it down, you can simple do:

.. code-block::
    
    make verticapylab-stop


Connect to Database
====================

Once the container launches click on the Connect button to navigate to the Connection page. 
Here enter your database credentials to connect to any Vertica database. 


.. image:: ../../docs/source/_static/getting_started_connection.png
   :width: 80%
   :align: center


Now you are ready to use the Query Profiler tool using the GUI.