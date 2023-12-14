.. _cicd.unittest:

==========
Unit Tests
==========


Unit tests serve as a foundational element in the software 
development lifecycle, offering a systematic approach to 
validating the correctness and reliability of individual 
units of code, often corresponding to public functions 
within Python files. For VerticaPy we use
`Pytest <https://docs.pytest.org/en/7.4.x/>`_ to run unit 
tests.

A key practice is to create dedicated 
test files for each Python module containing public 
functions. These test files, written by developers, serve 
as a comprehensive suite of tests to scrutinize the 
behavior of the smallest units of code in isolation.

In our development process, we emphasize the importance of 
maintaining a one-to-one relationship between Python source 
files and their associated test files. This ensures that 
the tests remain tightly coupled with the code they are 
designed to verify. 

For example: 

If there is a file `verticapy/plotting/_matplotlib/hist.py`
then a corresponding test file will be created in the
test directory: 
`verticapy/tests/plotting/_matplotlib/test_hist.py`.

We adhere to the test-driven development 
(TDD) methodology, where developers craft unit tests before 
the actual code implementation. These tests are constructed 
to assess the correctness of individual functions, evaluating 
their behavior against a predefined set of inputs. As a result, 
we establish a continuous feedback loop, where modifications 
or enhancements to the code are validated against the 
corresponding unit tests, fostering a proactive approach to 
defect detection.

To enhance the efficiency of our testing procedures, we 
leverage pytest and specifically employ the pytest-xdist 
plugin. This plugin facilitates parallel execution of tests, 
enabling us to distribute the test suite across multiple 
processes or even machines. The parallelization capability 
significantly accelerates the testing process, particularly 
beneficial as software projects scale in complexity. By 
harnessing the power of pytest-xdist, we ensure that our 
unit tests can be performed concurrently, allowing us to 
achieve faster feedback cycles and promptly identify and 
address potential issues in our codebase. This strategic 
integration of unit testing practices, coupled with parallel 
execution capabilities, contributes to the development of 
robust, reliable, and maintainable software systems.

.. important::

    For more detailed information, please check out 
    the relevant section: 
    :ref:`contribution_guidelines.code.unit_tests`