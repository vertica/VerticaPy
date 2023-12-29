.. _cicd.codecov:

==============
Code Coverage
==============

Code coverage is a metric used in software development 
to quantify the proportion of a codebase that is 
exercised by a set of tests. It measures the 
percentage of lines, branches, or statements within 
the code that have been executed during the execution 
of a test suite. The primary goal of code coverage 
analysis is to assess the thoroughness of testing and 
identify areas of code that remain untested. A higher 
code coverage percentage generally indicates that a 
greater portion of the codebase has been scrutinized, 
reducing the likelihood of undiscovered bugs. 


For the latest statistics on code-coverage of 
``VerticaPy`` please look at 
`this link <https://app.codecov.io/gh/vertica/VerticaPy>`_.

.. image:: /_static/img_code_coverage_1.png
    :width: 50%
    :align: center


Not only does it provide a wholistic view of the 
``VerticaPy`` library, but it also gives the 
contributors and in-depth understanding of which 
folders/functions are lacking behind the most. 

.. image:: /_static/img_code_coverage_2.png

In the details, we can see the total number of lines
as well as the number of lines in each folder that have
been covered in unit tests.

Additionally, for each python file we can look
at the exact lines that are missing unit tests.
This can guide contributors in understanding the 
gaps in testing.

.. image:: /_static/img_code_coverage_3.png

In summary, developers use code coverage tools to gain 
insights into which parts of their code are well-tested 
and which may require additional test cases. It serves 
as a valuable quality assurance metric, guiding the 
improvement of test suites and ultimately contributing 
to the creation of more reliable and robust software 
systems. 