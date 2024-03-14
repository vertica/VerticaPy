.. _contribution_guidelines.code.setting_up:

==========================
Setting-Up the Environment
==========================



Step 1: Git
=============

Fork the project on `Github <https://github.com/vertica/VerticaPy>`_ and check out your copy locally.

.. code-block:: shell

  git clone git@github.com:YOURUSERNAME/VerticaPy.git
  cd VerticaPy


Your GitHub repository **YOURUSERNAME/VerticaPy** will be called "origin" in
Git. You should also setup **vertica/VerticaPy** as an "upstream" remote.

.. code-block:: shell

  git remote add upstream git@github.com:vertica/VerticaPy.git
  git fetch upstream


Configure Git for the first time
----------------------------------

Make sure git knows your `name <https://help.github.com/articles/setting-your-username-in-git>`_  "Set commit username in Git") and [email address](https://help.github.com/articles/setting-your-commit-email-address-in-git/ "Set commit email address in Git"):

.. code-block:: shell

  git config --global user.name "John Smith"
  git config --global user.email "email@example.com"


Step 2: Branch
================

Create a new branch for the work with a descriptive name:

.. code-block:: shell

  git checkout -b my-fix-branch


Step 3: Install dependencies
===============================

Install the Python dependencies for development:

.. code-block:: shell
  
  pip3 install -r requirements-dev.txt


Step 4: Get the test suite running (Under development)
=======================================================

*VerticaPy* comes with its own test suite in the `verticapy/tests` directory. It’s our policy to make sure all tests pass at all times.

We appreciate any and all contributions to the test suite! These tests use a Python module: `pytest <https://docs.pytest.org/en/latest/>`_. You might want to check out the pytest documentation for more details.

You must have access to a Vertica database to run the tests. We recommend using a non-production database, because some tests may need the superuser permission to manipulate global settings and potentially break that database. Heres one way to go about it:
- Download docker kitematic: https://kitematic.com/
- Spin up a vertica container (e.g. sumitchawla/vertica)

Spin up your Vertica database for tests and then config test settings:

* Here are default settings:

  .. code-block:: python

    host: 'localhost'
    port: 5433
    user: <current OS login user>
    database: <same as the value of user>
    password: ''
    log_dir: 'vp_test_log'  # all test logs would write to files under this directory
    log_level: logging.WARNING

* Override with a configuration file called `verticapy/tests/vp_test.conf`. This is a file that would be ignored by git. We created an example `verticapy/tests/vp_test.conf.example` for your reference.
  
  .. code-block:: python

    # edit under [vp_test_config] section
    VP_TEST_HOST=10.0.0.2
    VP_TEST_PORT=5000
    VP_TEST_USER=dbadmin
    VP_TEST_DATABASE=vdb1
    VP_TEST_PASSWORD=abcdef1234
    VP_TEST_LOG_DIR=my_log/year/month/date
    VP_TEST_LOG_LEVEL=DEBUG
  

* Override again with VP_TEST_* environment variables

  .. code-block:: shell

    # Set environment variables in linux
    $ export VP_TEST_HOST=10.0.0.2
    $ export VP_TEST_PORT=5000
    $ export VP_TEST_USER=dbadmin
    $ export VP_TEST_DATABASE=vdb1
    $ export VP_TEST_PASSWORD=abcdef1234
    $ export VP_TEST_LOG_DIR=my_log/year/month/date
    $ export VP_TEST_LOG_LEVEL=DEBUG

  # Delete your environment variables after tests
  $ unset VP_TEST_PASSWORD
  ```

`Tox <https://tox.readthedocs.io>`_ is a tool for running those tests in different Python environments. *VerticaPy*
includes a `tox.ini` file that lists all Python versions we test. Tox is installed with the `requirements-dev.txt`,
discussed above.

Edit `tox.ini` envlist property to list the version(s) of Python you have installed. Then you can run the **tox** command from any place in the *verticapy* source tree. If `VP_TEST_LOG_DIR` sets to a relative path, it will be in the *verticapy* directory no matter where you run the **tox** command.

Examples of running tests:
----------------------------


.. code-block:: bash
  
  # Run all tests using tox:
  tox

  # Run tests on specified python versions with `tox -e ENV,ENV`
  tox -e py36,py37

  # Run specific tests by filename (e.g.) `test_vDF_combine_join_sort.py`
  tox -- verticapy/tests/vDataFrame/test_vDF_combine_join_sort.py

  # Run all tests on the python version 3.6:
  tox -e py36 -- verticapy/tests

  # Run all tests on the python version 3.7 with verbose result outputs:
  tox -e py37 -v -- verticapy/tests

  # Run an individual test on specified python versions.
  # e.g.: Run the test `test_vDF_append` under `test_vDF_combine_join_sort.py` on the python versions 3.7 and 3.8
  tox -e py37,py38 -- verticapy/tests/vDataFrame/test_vDF_combine_join_sort.py::TestvDFCombineJoinSort::test_vDF_append


The arguments after the `--` will be substituted everywhere where you specify `{posargs}` in your test *commands* of
`tox.ini`, which are sent to pytest. See `pytest --help` to see all arguments you can specify after the `--`.

You might also run `pytest` directly, which will evaluate tests in your current Python environment, rather than across
the Python environments/versions that are enumerated in `tox.ini`.

For more usages about `tox <https://tox.readthedocs.io>`_, see the Python documentation.

Step 5: Implement your fix or feature
==========================================

At this point, you're ready to make your changes! Feel free to ask for help; everyone is a beginner at first.

Have a look at an :ref:`contribution_guidelines.code.example`.


Commits
---------


Make some changes on your branch, then stage and commit as often as necessary:

.. code-block:: shell

  git add .
  git commit -m 'Added two more tests for #166'
```


When writing the commit message, try to describe precisely what the commit does. The commit message should be in lines of 72 chars maximum. Include the issue number `#N`, if the commit is related to an issue.

Step 6: Push and Rebase
========================

You can publish your work on GitHub just by doing:

.. code-block:: shell
  
  git push origin my-fix-branch


When you go to your GitHub page, you will notice commits made on your local branch is pushed to the remote repository.

When upstream (vertica/VerticaPy) has changed, you should rebase your work. The **rebase** command creates a linear history by moving your local commits onto the tip of the upstream commits.

You can rebase your branch locally and force-push to your GitHub repository by doing:

.. code-block:: shell

  git checkout my-fix-branch
  git fetch upstream
  git rebase upstream/master
  git push -f origin my-fix-branch



Step 7: Make a Pull Request
==============================

When you think your work is ready to be pulled into *VerticaPy*, you should create a pull request(PR) at GitHub.

A good pull request means:
 - a self-explanatory title (and the content of the PR should not go beyond the original title/scope)
 - commits with one logical change in each
 - well-formed messages for each commit
 - documentation and tests, if needed

Go to https://github.com/YOURUSERNAME/VerticaPy and `make a Pull Request <https://help.github.com/articles/creating-a-pull-request/>`_ to `vertica:master`. 

Sign the CLA
--------------
Before we can accept a pull request, we first ask people to sign a Contributor License Agreement (or CLA). We ask this so that we know that contributors have the right to donate the code. You should notice a comment from **CLAassistant** on your pull request page, follow this comment to sign the CLA electronically. 

Review
------------
Pull requests are usually reviewed within a few days. If there are comments to address, apply your changes in new commits, rebase your branch and force-push to the same branch, re-run the test suite to ensure tests are still passing. We care about quality, Vertica has internal test suites to run as well, so your pull request won't be merged until all internal tests pass. In order to produce a clean commit history, our maintainers would do squash merging once your PR is approved, which means combining all commits of your PR into a single commit in the master branch.

That's it! Thank you for your code contribution!

After your pull request is merged, you can safely delete your branch and pull the changes from the upstream repository.