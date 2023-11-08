.. _contribution_guidelines.code.auto_doc:

========================
Automatic Documentation
========================

Sphinx
=======

To ensure a consistent and professional look for the documentation, we will use Sphinx, 
a widely-used documentation tool, in conjunction with reStructuredText (RST) as the 
markup language. RST allows for clear, simple, and readable documentation writing, 
and Sphinx offers robust features for generating HTML and other formats from RST source files.


To get started with reStructuredText (RST) and Sphinx, follow these steps:

* Read the reStructuredText (RST) Guide below: Before diving into Sphinx, familiarize 
yourself with the reStructuredText (RST) syntax and features. The RST Guide 
provides a comprehensive overview of the markup language, covering headings, 
lists, links, code blocks, and more. Understanding RST is crucial as it forms the foundation of Sphinx documentation.

* Explore the Sphinx Guide and Documentation: Once you're comfortable with RST, 
turn your attention to Sphinx. The Sphinx documentation offers detailed information 
on how to set up and use Sphinx to generate professional and well-structured documentation. 
Pay close attention to the "Sphinx Basics" section, which covers project initialization, 
configuration, and building the documentation.

* Practice and Experiment: The best way to learn RST and Sphinx is to practice. 
Create a small test project, write RST files, and generate the documentation using Sphinx. 
Experiment with different elements like cross-referencing, images, and code blocks. 
This hands-on approach will help solidify your understanding and boost your confidence. Use the 
:ref:`contribution_guidelines.code.auto_doc.example`.

.. important:: To easily render the results, you can use the page :ref:`contribution_guidelines.code.auto_doc.render`.

.. toctree::
    :hidden:

    contribution_guidelines_code_auto_doc_render
    contribution_guidelines_code_auto_doc_example
_______

Setting up environment
=======================

This section is about setting up sphinx environment to automate the documentation process. Currently we set up the environment such that the following procedures are followed:



(But in future we will integrate this into the CI/CD pipeline.)




Clone the VerticaPy repo 
Inside the repo create a docs folder
Inside the docs folder, setup the sphinx environment.
First download and unzip this zip file which has all the essential parts. Replace the docs folder with this such that all the contents are now in docs folder. Primarily the make file and source folder.
Edit the ``docs/source/conf.py`` file taking special care about:
- copyright year 
- release version
- rst_prolog (This ensures which contents should be put at the top of every rst file). It can have general imports and settings for verticapy.

(FILE GOES HERE)

Install the requirements by: 

.. code-block::
    
    pip install -r requirements.txt




Install Verticapy from the setup file using below in the VerticaPy directory

.. code-block::
    
    pip install .

Note: Make sure VerticaPy is not already installed. If it is, then you can first uninstall it using:

.. code-block::
    
    pip uninstall verticapy




Navigate inside the docs folder

Run below code to change the name of all the directory paths inside the VerticaPy code 

.. code-block::
    
    python3 replace_sphinx_dir.py




Build the html using sphinx.

.. code-block::
    
    make html

Note: You may have to install make if you don't already have it:

.. code-block::
    
    apt install make




Clean up the names inside the HTML files. This is done to remove the verticapy prefix from the functions. This converts verticapy.vDataFrame.bar → vDataFrame.bar

.. code-block::
    
    python3 remove_pattern.py




Now all the html files are built inside the "docs/build" directory. This can be imported and used.

For ease of use, a script file called "refresh.py" has also been placed inside the folder. If you make any changes in the VerticaPy code/docstring, then to build the updated html files this script can be run: 



.. code-block::
    
    ./refresh.sh

Note: You may need to change permissions if the file is not executable:

.. code-block::
    
    chmod +x refresh.sh

________

Useful Links
==============

reStructuredText (RST) Primer (Sphinx Documentation):

Link: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
Description: This is the official reStructuredText primer from the Sphinx documentation. It covers the basics of RST syntax, including headings, lists, links, code blocks, and more.
reStructuredText (RST) Directives (Sphinx Documentation):

Link: https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html
Description: This page provides information about various directives available in RST, which allow you to add additional formatting and functionality to your documentation.
RST Cheat Sheet (reStructuredText):

Link: https://docutils.sourceforge.io/docs/user/rst/quickref.html
Description: A concise cheat sheet for reStructuredText (RST) with examples of common markup elements and syntax.
RST Syntax Specification (reStructuredText):

Link: https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html
Description: The official syntax specification for reStructuredText, detailing all the elements and rules of the markup language.
Docutils Documentation:

Link: https://docutils.sourceforge.io/docs/index.html
Description: Docutils is the Python library that underlies the parsing and processing of reStructuredText. This documentation provides insights into the library and its features.
Sphinx Domains and Roles (Sphinx Documentation):

Link: https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html
Description: This page explains how to use domains and roles in Sphinx to add specialized markup for documenting specific subjects, such as Python code, references, and more.
Common RST Substitutions (Sphinx Documentation):

Link: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#substitutions
Description: This section covers common substitutions in reStructuredText, allowing you to define and reuse certain strings or pieces of content.
Sphinx Configuration (Sphinx Documentation):

Link: https://www.sphinx-doc.org/en/master/usage/configuration.html
Description: This page provides details on how to configure Sphinx and your RST documents to customize the output and behavior of your documentation.

_____

Common Elements of Doc
============================


In order to automatically produce professional and intelligible documentation, we must strictly follow the reStructuredText(rst) format while writing the Docstring inside the python modules. RST a lot of flexibility in displaying our documentation, but it also inhibits certain formatting. Below we will example of good practices and their results along with bad practices and their results.



Headers and Sub-headers
------------------------

By default we can use "--------" underneath a header to mark it is as a header. But if for a specific module, we need headings and subsections then we can do this using a combination of "=======" and "--------".

The example below demonstrates how sub-section can be created using the example of numh function's docstring:


.. code-block::

    # Normal headers    
    def numh(
            self, method: Literal["sturges", "freedman_diaconis", "fd", "auto"] = "auto"
    ) -> float:
        """
        Computes the optimal vDataColumn bar width.

        Parameters
        ----------
        method: str, optional

        Returns
        -------
        float
            optimal bar width.
        """

    Computes the optimal vDataColumn bar width.



**Subsections:**

.. code-block::

    # Subsections 
    def numh(
            self, method: Literal["sturges", "freedman_diaconis", "fd", "auto"] = "auto"
        ) -> float:
            """
            Computes the optimal vDataColumn bar width.

            Parameters
            ==========
            method: str, optional

            Subsection
            ----------
            Some text here.

            Returns
            =======
            float
                optimal bar width.
            """


We can even create some headings in the center to highlight some information using the centered directive as below:




Centered Info
--------------
.. code-block::

    .. centered:: Some Centered Information!


**Output:**

.. centered:: Some Centered Information!

Indentation and Line Spacing
-----------------------------

Line spacing and indentation have to be carefully set. Without a line-break, all the text is considered single line. In order to avoid a line-space and still go to next line, use "|". Below example demonstrates a few cases and outputs:

.. code-block::

    Parameters
    ----------
    method: str, optional
        Method used to compute the optimal h.
            | auto : Combination of Freedman Diaconis and Sturges.
            | freedman_diaconis : Freedman Diaconis 
            | sturges : Sturges 

**Output:**

    method: str, optional
        Method used to compute the optimal h.
            | auto : Combination of Freedman Diaconis and Sturges.
            | freedman_diaconis : Freedman Diaconis 
            | sturges : Sturges 

.. code-block::

    Parameters
    ----------
    method: str, optional
        Method used to compute the optimal h.
            auto : Combination of Freedman Diaconis and Sturges.

            freedman_diaconis : Freedman Diaconis 

            sturges : Sturges

**Output:**

    method: str, optional
        Method used to compute the optimal h.
            auto : Combination of Freedman Diaconis and Sturges.

            freedman_diaconis : Freedman Diaconis 

            sturges : Sturges

.. code-block::

    Parameters
    ----------
    method: str, optional
        Method used to compute the optimal h.
            auto              : Combination of Freedman Diaconis
                                and Sturges.
            freedman_diaconis : Freedman Diaconis
            sturges           : Sturges


**Output:**

    method: str, optional
        Method used to compute the optimal h.
            auto              : Combination of Freedman Diaconis
                                and Sturges.
            freedman_diaconis : Freedman Diaconis
            sturges           : Sturges



Code blocks
--------------

Code blocks can be added using the code-clock directive. Always remember to leave a line after the directive. Note that the python lines are not executed using code-block syntax. Following examples describe its use:

    .. code-block::

        .. code-block:: python

           import verticapy as vp
           vp.DataFrame({"a":[1,2,3]}) 

**Output:**

.. code-block:: python

    import verticapy as vp
    vp.DataFrame({"a":[1,2,3]}) 



Code execution

In order to execute and display certain, we can use the ipython directive. Note that you will have to install and import the two extensions inside the conf file while using with sphinx: IPython.sphinxext.ipython_directive, IPython.sphinxext.ipython_console_highlighting

To execute and display line of code use the following syntax:




Code execution block
---------------------

.. code-block::

    .. ipython:: python

        import verticapy as vp
        vp.vDataFrame({"a":[1,2,3]})


SUPPRESSED RESULTS

You can even choose to suppress the code execution. This could be useful when you want to hide all the imports. You could either suppress the output of the entire block with the directive option "suppress", or you could just skip input of a line using pseudo-directive "@suppress".

The following examples highlights this:

Code block with Suppressed code
-------------------------------

.. code-block::

    .. ipython:: python
        :suppress: 
        
        import verticapy as vp
        import pandas as pd
        import sys
        
    .. ipython:: python

        print(vp.vDataFrame({"a":[1,2,3]}))

**Output:**

    .. ipython:: python
        :suppress: 
        
        import verticapy as vp
        import pandas as pd
        import sys
        
    .. ipython:: python

        print(vp.vDataFrame({"a":[1,2,3]}))

Code block with Suppressed code
-------------------------------

.. code-block::

    .. ipython:: python

        @suppress      
        import verticapy as vpp
        print(vpp.vDataFrame({"a":[1,2,3]}))


Though both examples produce the same result but there are suggested use-cases for each. In cases where you have to import a lot of supporting libraries you can use the directive method, and in cases where you only need to skip one line, then pseudo-directive (@suppress) is preferred.


Plotting Using matplotlib
--------------------------

We can plot using the matplotlib library. An extra step that we need to do is to save the figure using the "@savefig" pseudo-directive. We also need to provide the unique name of the image file.


.. important:: We need to follow the following naming convention while creating image files. 
    The name should be the entire directory with _ between sub-directories. We should also add the class name if the function is inside a particular class. For example: 
    If I want to create an html/image for the boxplot function which is located in core/vDataFrame/_plottling.py and inside teh vDFPlot class. For this the file name should be: ``core_vDataFrame__plotting_vDFPlot_boxplot``
    If you want to create multiple plots inside one function that add numbering at the end for example:
    ``core_vDataFrame__plotting_vDFPlot_boxplot_1``
    ``core_vDataFrame__plotting_vDFPlot_boxplot_2``







**Example**

.. code-block::

    .. ipython:: python
        :suppress:

        import matplotlib.pyplot as plt
        x = [1, 2, 3, 4, 5];
        y = [2, 4, 6, 8, 10];
        plt.plot(x, y, marker='o', linestyle='-');
        plt.xlabel('X-axis');
        plt.ylabel('Y-axis');
        plt.title('Simple Linear Plot');
        @savefig simple_plot.png
        plt.show()

.. ipython:: python
    :suppress:

    import matplotlib.pyplot as plt
    x = [1, 2, 3, 4, 5];
    y = [2, 4, 6, 8, 10];
    plt.plot(x, y, marker='o', linestyle='-');
    plt.xlabel('X-axis');
    plt.ylabel('Y-axis');
    plt.title('Simple Linear Plot');
    @savefig simple_plot.png
    plt.show()

.. note:: We have suppressed the input code to avoid unnecessary distraction.


Using verticapy, it is pretty similar to matplotlib.

Plotting VerticaPy plot
------------------------
.. code-block::

    .. ipython:: python
        :suppress:

        import verticapy as vp
        data= vp.vDataFrame({"counts":[1,1,1,2,2,3]})
        @savefig verticapy_plot.png
        data.bar("counts")

.. ipython:: python
    :suppress:

    import verticapy as vp
    data= vp.vDataFrame({"counts":[1,1,1,2,2,3]})
    @savefig verticapy_plot.png
    data.bar("counts")


Advanced Output
----------------

If we want to produce advanced output like the VerticaPy vDataFrame output or some other advanced graphics then we can use the following methodology:

(1) Create the html representation of the image

(2) Save it as an html file.

(3) Load and display the html file




.. important:: We need to follow the following naming convention while creating html files similar to images. Details follow.

The name should be the entire directory with _ between sub-directories. We should also add the class name if the function is inside a particular class. For example:

If I want to create an html/image for the boxplot function which is located in ``core/vDataFrame/_plottling.py`` and inside teh vDFPlot class. For this the html name should be:

``core_vDataFrame__plotting_vDFPlot_boxplot.html``

.. important:: We need to add "figures/" before the file name. Additionally when retrieving the file, we should add the sphinx directory ("SPHINX_DIRECTORY/figures/").



VerticaPy Output
^^^^^^^^^^^^^^^^^^

.. code-block::

    .. ipython:: python
        :suppress:

        import verticapy as vp
        html_file = open("figures/filename.html", "w")
        html_file.write(vp.vDataFrame({"a":[1,2,3]})._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/filename.html






**Plotly Graphics using VerticaPy**

We can even plot advanced graphics using the similar way. First save the figure as an html file and then display it:

Note: You can use any plotting library to plot using the verticapy.set_option("plotting_library","plotly") syntax. But also note that matplotlib files can be saved and displayed as images. While the plotly and highcharts can only be displayed after beings saved as HTML files.

VerticaPy Plotly Plot
^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block::

    .. ipython:: python
        :suppress:

        import verticapy as vp
        vp.set_option("plotting_lib","plotly")
        data= vp.vDataFrame({"counts":[1,1,1,2,2,3]})
        fig=data.bar("counts")
        fig.write_html("figures/filename2.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/filename2.html



Let's look at another example of correlation table using plotly library:

VerticaPy Correlation Table
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block::

    .. ipython:: python
        :suppress:

        from verticapy.datasets import load_titanic
        titanic = load_titanic()
        fig=titanic.corr(method = "spearman")
        fig.write_html("figures/plotly_corr.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/plotly_corr.html



Tables
-------

We can input tables using multiple methods. The easiest method is using the csv-table directive.

.. code-block:: 

    .. csv-table:: **Table Title**
        :header: Column Header, Another Column
        :widths: 30, 70

        Row 1 Value, Value in Row 1
        Row 2 Value, Value in Row 2
        Row 3 Value, Value in Row 3

**Output:**

    .. csv-table:: **Table Title**
        :header: Column Header, Another Column
        :widths: 30, 70

        Row 1 Value, Value in Row 1
        Row 2 Value, Value in Row 2
        Row 3 Value, Value in Row 3


Another method is to input the entire table along with decorators which is a bit more cumbersome:

.. code-block::

    +---------------+------------------+
    | Column Header | Another Column   |
    +===============+==================+
    | Row 1 Value   | Value in Row 1   |
    +---------------+------------------+
    | Row 2 Value   | Value in Row 2   |
    +---------------+------------------+
    | Row 3 Value   | Value in Row 3   |
    +---------------+------------------+

Output:

+---------------+------------------+
| Column Header | Another Column   |
+===============+==================+
| Row 1 Value   | Value in Row 1   |
+---------------+------------------+
| Row 2 Value   | Value in Row 2   |
+---------------+------------------+
| Row 3 Value   | Value in Row 3   |
+---------------+------------------+



Miscellaneous
--------------

See Also
^^^^^^^^^

We can use the see also directive to show similar codes that can help the user.

.. code-block::

    .. seealso:: :mod:`plottt.vDCPlot.bar`
        Documentation of the :mod:`plottt.vDCPlot.bar` standard module.

Output:

.. seealso:: :mod:`plottt.vDCPlot.bar`
    Documentation of the :mod:`verticapy.vDCPlot.bar` standard module.




Deprecation Warning
^^^^^^^^^^^^^^^^^^^^

We can use the default syntax for highlighting deprecate functions:

.. code-block::

    .. deprecated:: 3.8

**Output:**

.. deprecated:: 3.8


Notes and Warnings
^^^^^^^^^^^^^^^^^^^^^

There are certain admonitions that we can use to make the documentation more professional. Below I give examples of a few:

.. code-block::

    .. note:: This is a note.

    .. DANGER:: This is a danger warning.

    .. Important:: This is a note.

    .. Hint:: This is a hint.

    .. Warning:: This is a warning.

**Output:**

.. note:: This is a note.

.. DANGER:: This is a danger warning.

.. Important:: This is a note.

.. Hint:: This is a hint.

.. Warning:: This is a warning.
