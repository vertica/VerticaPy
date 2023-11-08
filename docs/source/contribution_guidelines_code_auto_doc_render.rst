.. _contribution_guidelines.code.auto_doc.render:

=======================
Render your doc-string
=======================




You can either use the `sphinx documentation page <https://www.sphinx-doc.org/en/master/usage/quickstart.html>`_ 
to setup sphinx in your system. Or you can follow the below simple steps that give you the opportunity to test just one specific file. 

Download this file package and unzip it in the directory where you want to work:

Go in side the docs directory and install the requirements. 

.. tip:: It is always best practice to create a new environment for installing dependencies etc.

.. code-block:: shell

    pip install -r requirements.txt


Install make (if you don't have already):

.. code-block:: shell

    sudo apt install make (sudo is optional)


Place your python file inside docs and rename it `test_file.py`.


.. note:: There is already a test_file.py as an example. You can delete that and replace with yours.

Build the html pages by running the below command while inside the docs directory:
.. code-block:: shell

    make html

Once the build is complete, you can navigate to the build/html directory. There open the index.html in any browser to view your resulting documentation page.

If you want to make changes then make changes to your test_file.py and then clean the files using:
.. code-block:: shell
    
    make clean

Then to again display new results just build again

.. code-block:: shell

    make html