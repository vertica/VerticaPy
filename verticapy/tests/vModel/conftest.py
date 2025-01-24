# Pytest
import pytest

# VerticaPy
import verticapy._config.config as conf

# Other Modules


@pytest.fixture(scope="module", autouse=True)
def load_matplotlib():
    """
    Set default plotting library to matplotlib
    """
    conf.set_option("plotting_lib", "matplotlib")
    yield
    conf.set_option("plotting_lib", "plotly")
