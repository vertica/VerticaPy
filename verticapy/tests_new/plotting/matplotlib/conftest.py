# Pytest
import pytest

# Standard Python Modules


# Other Modules
import matplotlib.pyplot as plt

# VerticaPy
import verticapy._config.config as conf


@pytest.fixture(scope="session", autouse=True)
def load_matplotlib():
    conf.set_option("plotting_lib", "matplotlib")
    yield
    conf.set_option("plotting_lib", "plotly")


@pytest.fixture(scope="session")
def matplotlib_figure_object():
    yield plt.Axes
