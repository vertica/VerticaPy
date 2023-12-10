.. _cicd.pylint:

=======
Pylint
=======

`Pylint <https://pypi.org/project/pylint/#:~:text=What%20is%20Pylint%3F,code%20without%20actually%20running%20it.>`_ 
is a powerful static code analysis tool for Python 
that evaluates code against a set of coding standards and 
identifies potential errors, stylistic issues, and other 
code quality concerns. 


Let's consider an example to illustrate how Pylint works \
in practice.

Suppose we have the following Python script named example.py:

.. code-block:: python

    # example.py
    def add_numbers(a, b):
        return a + b

    result = add_numbers(3, 4)
    print(result)

If we run pylint on this code using ``pylint example.py``
Pylint will generate a report providing feedback on the code. 
In this example, it might highlight certain issues such as 
missing docstrings, which are comments providing information 
about the purpose and usage of functions and modules. 
The output might look something like this:

.. code-block:: python

    ************* Module example
    example.py:1:0: C0111: Missing module docstring (missing-docstring)
    example.py:1:0: C0103: Function name "add_numbers" doesn't conform to snake_case naming style (invalid-name)
    example.py:1:0: C0111: Missing function docstring (missing-docstring)
    example.py:5:4: W0104: Statement seems to have no effect (pointless-statement)
    example.py:5:4: R1705: Unnecessary "else" after "return" (no-else-return)

    ------------------------------------
    Your code has been rated at 5.45/10

In this output, Pylint points out issues such as missing 
docstrings, a naming convention violation, and a statement 
that seems to have no effect. It also scores the code out
of 10.

.. important::

    VerticaPy has a strict threshold below which any code
    will not be accepted. Have a look at the GitHub action
    in order to confirm the threshold value. It is
    recommended to have a value closer to 10 to be certain.

By using Pylint in this manner, developers can identify 
and address potential problems in their code, ensuring 
adherence to coding standards, improving readability, and 
ultimately enhancing the overall quality of their Python 
projects.
