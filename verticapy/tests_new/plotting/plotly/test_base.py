# # Pytest
# import pytest

# # Standard Python Modules


# # Other Modules
# import numpy as np

# # Testing variables
# col_name = "check 2"


# # Data
# [array([['0', '0', '0', '1', '1', '1'],
#         ['A', 'B', 'C', 'A', 'B', 'C'],
#         ['4', '13', '23', '16', '17', '27']], dtype='<U21')
#  array([['0', '1'],
#         ['40', '60']], dtype='<U21')]

# @pytest.fixture(scope="class")
# def plot_result(dummy_vd):
#     return dummy_vd[col_name].bar()


# class TestBarPlot:
#     @pytest.fixture(autouse=True)
#     def result(self, plot_result):
#         self.result = plot_result

#     def test__convert_labels_and_get_counts(self, plotly_figure_object):
#         # Arrange
#         # Act
#         # Assert - checking if correct object created
#         assert type(self.result) == plotly_figure_object, "wrong object crated"
