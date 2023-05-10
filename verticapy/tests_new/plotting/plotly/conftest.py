# Pytest
import pytest

# Standard Python Modules


# Other Modules


# VerticaPy
import verticapy._config.config as conf

DUMMY_TEST_SIZE = 100


@pytest.fixture(scope="session", autouse=True)
def load_plotly():
    conf.set_option("plotting_lib", "plotly")
    yield
    conf.set_option("plotting_lib", "matplotlib")
