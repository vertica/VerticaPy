.. _verticapylab_gs:

=================
VerticaPyLab
=================

The easiest way to try out VerticaPy is with `VerticaPyLab <https://github.com/vertica/VerticaPyLab>`_, a docker-based 
JupyterLab environement designed to provide users with a powerful, GUI-based platform for using Vertica and VerticaPy 
functionalites, including various no-code tools. Capitalizing on the convenicnece of docker, VerticaPyLab eliminates the need
for any intricate installations, configurations, and dependency management. VerticaPyLab extends the capabilities of JupyterLab
by integrating essential tools, extensions, and utilities that allow you to unlock the full potential of VerticaPy's machine 
learning and analytic funtionalities, all while utilizing the performance and scalability of the Vertica columnar database. 

VerticaPyLab utilizes two containers, one housing a Vertica database and the other a JupyterLab container with VerticaPy and 
necessary dependecies installed. Using these two components, VerticaPyLab offers a cohesive environment that bridges the power 
of Vertica with the capabilities of VerticaPy and the ease of Python syntax.

Some key features of VerticaPyLab include the following:

- **QueryProfiler:** use the QueryProfiler tool to investigate and identify the reasons why a query or set of queries are running slow.
- **Graphics and visualizations:** create an array of visualizations and plots directly in the JupyterLab environment using the power of Highcharts, Matplotlib, Plotly, and other libraries.
- **Seamless data ingestion:** VerticaPyLab offers streamlined data ingestion capabilities, allowing you to effortlessly load and process massive datasets from your Vertica database.
- **Interactive tutorials:** familiarize yourself with core VerticaPy concepts using the included VerticaPy lessons.

Install VerticaPyLab
=====================

The following guide provides instructions for installing VerticaPyLab and its prerequisites. 

Prerequisites
---------------

VerticaPyLab requires the following prerequisites:
- Docker Desktop version 
- (Windows only) Windows Subsystem for Linux  2 (WSL 2)

If you need to install any of the prerequisites, installation instructions are provided below. 

Windows Subsystem for Linux (WSL)
----------------------------------

WSL allows users to run a lightweight Linux environment on a Windows machine. WSL requires that you are running Windows 10 version 2004 and higher (Build 19041 and higher) or Windows 11. If you are running an older build, you can upgrade to a newer version by following the instructions on the `Microsoft documentation <https://docs.microsoft.com/en-us/windows/wsl/install-manual>`_.

Install WSL 2
~~~~~~~~~~~~~~

1. Open the command prompt or Windows PowerShell as **Administrator**.

2. Run the following command, which enables the required components, downloads the latest Linux kernel, sets WSL 2 as default, and installs the default Ubuntu distribution:
	
.. code-block:: 
    
  $ wsl --install   


.. note:: The above command works only if WSL is not already installed. If the above command returns the WSL help text, run `wsl --list --online` to see a list of available Linux distributions, then run `wsl --install -d `*`DistroName`*  to install one of the available distributions. To uninstall WSL, see `unregister or uninstall a Linux distribution <https://docs.microsoft.com/en-us/windows/wsl/basic-commands#unregister-or-uninstall-a-linux-distribution>`_.

3. Reboot your machine. The Linux distribution you installed above should start automatically. If it does not, you can start it manually from the Windows Start menu.

4. The first time you launch a newly installed Linux distribution, a console window opens asking you to wait for files to decompress and be stored on your machine. All future launches should take less than a second.

5. Set your username and password.
6. Install the **make** package:

.. code-block::

  $ sudo apt install make


Docker Desktop
---------------

Docker Desktop is an out-of-the-box containerization software that includes a graphical user interface for managing your containers, applications, and images. Docker Desktop requires that WSL 2 is installed on your machine.

Install and configure Docker Desktop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Download and run the `Docker Desktop installer <https://docs.docker.com/desktop/windows/install/>`_, making sure to download a version of Docker Desktop using a Docker version 18.09 or higher. 
2. Start Docker Desktop from the Windows Start menu.
3. Click the gear icon on the top right and navigate to the **General** tab.
4. Verify that the **Use WSL 2 based engine** option is turned on. If not, check it and click **Apply & Restart**.

VerticaPyLab Quickstart
========================

The following steps import and launch VerticaPyLab:

1. Start a Linux distribution.
2. Clone the VerticaPyLab repository:  
    .. code-block::

      $ git clone https://github.com/vertica/VerticaPyLab.git

3. Navigate to the cloned directory:  
      .. code-block::
            
        $ cd VerticaPyLab

4.  Start the VerticaPyLab service. There are two options, depending on whether you want to launch both the Vertica and JupyterLab containers or just the JupyterLab container:
	1. To launch both containers, run:
        
        .. code-block::
            
          $ make all

        This command automatically creates and connects you to a demo database.

	1. To launch only the JupyterLab container, run:

        .. code-block::
            
          $ make verticapylab-start

        You can connect to an existing Vertica database after you open VerticaPyLab.

5. Open the displayed link in a browser.
6. To stop the VerticaPyLab, run:
    .. code-block::
        
      $ make stop

7. To clean up the environment and delete all images, run:
    .. code-block::
        
      $ make uninstall

Getting started with VerticaPyLab
===================================

After you launch the service, you are taken to the VerticaPy UI homepage. From this page, you have access to a number of tools
and functionalites, including:

- Connect: connect to a Vertica database. You will need the host, username, password, and database name. Once you create a \
  connection, you can use it to reconnect to the database by selecting it in **Available connections**.

- QueryProfiler: profile a query or set of queries to investigate reasons for slow performance. You can either load \
  an existing QueryProfiler object or create one from scratch. After you load or create a QueryProfiler object, a window \ 
  opens with an interactive version of the query plan. 

- Data Science Essentials: explore a set of interactive data science tutorials that walk through some of the amazing capabilities \
  of VerticaPy.
