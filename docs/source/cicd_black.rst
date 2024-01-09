.. _cicd.black:

======
Black
======

`Black <https://pypi.org/project/black/>`_ is a highly 
popular and powerful code formatting tool 
for Python that automatically reformats Python code to comply 
with a standardized style guide, known as PEP 8. Developed 
with the goal of promoting code consistency and readability, 
Black eliminates the need for developers to manually adhere 
to formatting conventions, making the codebase visually 
consistent and more maintainable. The name "Black" is an 
acronym for "Blackened," emphasizing its approach of taking 
input Python code and producing output with a consistent and 
unambiguous style.

The operation of Black is straightforward: it parses Python 
code and rewrites it in a clean, standardized format. Unlike 
some other formatting tools, Black does not engage in 
discussions about formatting choices; it enforces a single, 
opinionated style, and users are encouraged to embrace it 
without configuration. This simplicity contributes to Black's 
popularity and its adoption across a wide range of projects.


Example
^^^^^^^^

Look at the code below. It has some formatting issues
including unwarranted blank spaces and blank lines.

.. code-block:: python
    
    def   calculate_total_price( quantity,price ):

     total_price =0
      for i in range(quantity ):
     total_price +=price
        return total_price
    def calculate_total_price_with_tax( quantity,price,tax_rate):
        total_price   =0
        for i in range( quantity):
            total_price+= price+ (price*tax_rate)
        return total_price

Now, if we apply black to it, observe how the formatting
is perfected:

.. code-block:: python

    def calculate_total_price(quantity, price):
        total_price = 0
        for i in range(quantity):
            total_price += price
        return total_price

    def calculate_total_price_with_tax(quantity, price, tax_rate):
        total_price = 0
        for i in range(quantity):
            total_price += price + (price * tax_rate)
        return total_price

All the unnecessary spaces and lines are removed.

.. important::

    It is expected that the contributors use black on their
    code before pushing it to the GitHub repo. Otherwise
    the checks will fail and the PR cannot be merged.

By integrating Black into the development workflow, teams 
can achieve a more streamlined and efficient process for 
maintaining consistent code formatting. It is often used 
in conjunction with version control systems and integrated 
into Continuous Integration (CI) pipelines to automatically 
format code contributions during the development process, 
ensuring that the entire codebase adheres to a uniform style 
and minimizing formatting-related code reviews.