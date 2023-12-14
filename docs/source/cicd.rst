.. _cicd:

======
CI/CD
======

The VerticaPy project employs a robust Continuous 
Integration/Continuous Deployment (CI/CD) pipeline to 
ensure the quality, consistency, and reliability of its 
codebase. Here's an overview of the key stages in the pipeline:

.. image:: /_static/cicd.png

- Step 1: Code Formatting Check (black) 
    The pipeline checks if the code adheres to consistent 
    formatting standards using the black Python library. 
    This ensures a uniform and readable codebase.
    For more info click here: :ref:`cicd.black`.

- Step 2: Code Quality Check (pylint)
    The pipeline utilizes the pylint Python library to assess 
    the overall quality of the code. This step helps identify 
    potential issues and enforces coding standards.
    For more info click here: :ref:`cicd.pylint`.

- Step 3: Unit Testing (pytest)
    Various Python environments are employed to execute unit 
    tests using the pytest framework. This ensures that the 
    proposed changes do not introduce regressions and maintains 
    the stability of the code.
    For more info click here: :ref:`cicd.unittest`.

- Step 4: Code Coverage Analysis
    The pipeline calculates code coverage metrics to ensure 
    that a substantial percentage of the codebase is covered 
    by unit tests. This practice guarantees a comprehensive 
    and effective test suite.
    For more info click here: :ref:`cicd.codecov`.

- Step 5: Automated Documentation Update (Sphinx)
    After successful PR merging, the pipeline automatically 
    updates the project's documentation using Sphinx. This 
    ensures that the documentation remains synchronized with 
    the latest changes, providing users with accurate and 
    up-to-date information.
    For more info click here: :ref:`cicd.sphinx`.

This CI/CD pipeline not only accelerates the development 
process but also upholds code quality standards, minimizes 
errors, and enhances the overall reliability of VerticaPy. 
By automating these essential checks and processes, the 
project maintains a high level of consistency and ensures 
that each contribution aligns with the established guidelines 
and best practices.


.. toctree::
    :maxdepth: 1
    :titlesonly:

    cicd_black
    cicd_pylint
    cicd_unittest
    cicd_codecov
    cicd_sphinx