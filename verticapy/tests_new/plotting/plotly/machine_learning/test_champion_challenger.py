# Pytest
import pytest

# Standard Python Modules


# Other Modules
import numpy as np

# Verticapy
from verticapy.learn.delphi import AutoML

# Testing variables
col_name_1 = "binary"
col_name_2 = "0"


@pytest.fixture(scope="class")
def plot_result(dummy_dist_vd):
    model = AutoML("model_automl", lmax=10, print_info=False)
    model.fit(
        dummy_dist_vd,
        [
            col_name_1,
        ],
        col_name_2,
    )
    return model.plot()


class TestMachineLearningChampionChallengerPlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result

    def test_properties_output_type(self, plotly_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result) == plotly_figure_object, "Wrong object crated"
