
.. _contribution_guidelines.code.unit_tests:

============
Unit Tests
============

Functions must include unit tests, which are located in the 'tests' folder. The test files follow the same folder structure as the original veritcapy directory. For each file there is a test file. That means if a function is added in core/vdataframe/_aggregate.py file then its respective test will be added in the test file test/core/vdataframe/test_aggregate.py.

Unit tests can be tested with the default VerticaPy datasets or the smaller efficient datasets creted inside the tests/conftest.py file. Be sure to look at all the datasets before creating your own. All these datasets can be imported as fixtures:

.. code-block:: python

    def test_properties_type(self, titanic_vd):
        result=titanic_vd["survived"].bar()
        assert(type(result)==plotly.graph_objs._figure.Figure)

The above titanic_vd is a fixture that is defined in the conftest file. 

To make the code efficient, it is highly encouraged to use fixtures and to share results across the scope. This ensures that the result will be shared. For every test (i.e. assert), there should be a separate function. To combine multiple tests for the same method, feel free to use classes. 

For the function we just created, we would place the unit tests 'test_vDF_correlation.py' in the 'vDataFrame' directory.

A unit test might look like this:
.. code-block:: python

    # Example unit test function

    class TestPearson(self):
        def test_age_and_fare(titanic_vd):
            result= titanic_vd.pearson("age", "fare")
            assert result == pytest.approx(0.178575164117464, 1e-2)

        def test_age_and_survived(titanic_vd):
            result_2 = titanic_vd.pearson("age", "survived")
            assert result == pytest.approx(-0.0422446185581737, 1e-2)  


A fixture by the name of "schema_loader" has been defined in the conftest file that creates a schema with a random name. This schema is dropped at the end of the unit test. You are encouraged to make use of this fixture to name your models/datasets, if necessary. 
For example, the follwoing loads a dataset and gives it a name from the schema.

.. code-block:: python

    @pytest.fixture(scope="module")
    def titanic_vd(schema_loader):
        """
        Create a dummy vDataFrame for titanic dataset
        """
        titanic = load_titanic(schema_loader, "titanic")
        yield titanic
        drop(name=f"{schema_loader}.titanic")

Since we are using the "schema_loader" fixture, we do not necessarily have to drop the dataset schema because it is automatically dropped at the end of the unit test.

Lastly, double check to make sure that your test allows parallel execution by using the following pytest command:

.. code-block:: python

    pytest -n auto --dist=loadscope

Note that in order to use the above, you will have to install pytest-xdist.


Add appropriate tests for the bugs or features behavior, run the test suite again, and ensure that all tests pass. Here are additional guidelines for writing tests:
 - Tests should be easy for any contributor to run. Contributors may not get complete access to their Vertica database. For example, they may only have a non-admin user with write privileges to a single schema, and the database may not be the latest version. We encourage tests to use only what they need and nothing more.
 - If there are requirements to the database for running a test, the test should adapt to different situations and never report a failure. For example, if a test depends on a multi-node database, it should check the number of DB nodes first, and skip itself when it connects to a single-node database (see helper function `require_DB_nodes_at_least()` in `verticapy/tests/integration_tests/base.py`).