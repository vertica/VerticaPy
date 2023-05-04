# # Pytest
import pytest

# # Standard Python Modules
import numpy as np

# # Verticapy 
from verticapy.plotting._plotly.base import PlotlyBase

@pytest.fixture(scope="function")
def dummy_data_parents_children():
    parents = ['0', '1']
    children = ['A', 'B', 'C']
    all_counts = [4, 8, 12, 6, 12, 18]
    data_dict = {}
    for i, parent in enumerate(parents):
        for j, child in enumerate(children):
            count = all_counts[i*len(children) + j]
            data_dict[(parent, child)] = count

    data_dict[('', parents[0])] = sum(data_dict[(parents[0], child)] for child in children)
    data_dict[('', parents[1])] = sum(data_dict[(parents[1], child)] for child in children)

    data_shaped = np.array([    [parents[0], parents[0], parents[0], parents[1], parents[1], parents[1]],
        [children[0], children[1], children[2], children[0], children[1], children[2]],
        [data_dict[(parent, child)] for parent, child in data_dict.keys() if parent != '']
    ], dtype='<U21')
    return data_shaped,data_dict



def test_convert_labels_and_get_counts(dummy_data_parents_children):
    # Arrange
    reference=dummy_data_parents_children[1]
    func=PlotlyBase()
    # Act
    output=func._convert_labels_and_get_counts(dummy_data_parents_children[0])
    result = {}
    for i in range(len(output[0])):
        child_value = output[1][i]
        parent_value = output[2][i]
        count = output[3][i]
        key = (parent_value, child_value)
        result[key] = count
    # Assert
    assert result==reference
    


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
