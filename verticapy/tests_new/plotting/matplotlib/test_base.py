"""
(c)  Copyright  [2018-2023]  OpenText  or one of its
affiliates.  Licensed  under  the   Apache  License,
Version 2.0 (the  "License"); You  may  not use this
file except in compliance with the License.

You may obtain a copy of the License at:
http://www.apache.org/licenses/LICENSE-2.0

Unless  required  by applicable  law or  agreed to in
writing, software  distributed  under the  License is
distributed on an  "AS IS" BASIS,  WITHOUT WARRANTIES
OR CONDITIONS OF ANY KIND, either express or implied.
See the  License for the specific  language governing
permissions and limitations under the License.
"""
# Pytest
import pytest

# Standard Python Modules
import numpy as np
import matplotlib.pyplot as plt

# Verticapy
from verticapy.plotting._matplotlib.base import MatplotlibBase


@pytest.fixture(scope="class")
def dummy_data_parents_children():
    return data_shaped, data_dict


class Test_Get_Ax_Fig:
    @pytest.mark.parametrize("n, reference", [(1, 1), (2, 2)])
    def test_axes(self, n, reference):
        # Arrange
        func = MatplotlibBase()
        # Act
        result = func._get_ax_fig(n)
        # Assert
        assert result[0] == reference

    def test_figure(self):
        # Arrange
        func = MatplotlibBase()
        # Act
        result = func._get_ax_fig(3)
        # Assert
        assert result[1] == plt


class Test_Get_Matrix_Fig_Size:
    @pytest.mark.parametrize("n, reference", [(1, 3), (2, 4.5)])
    def test_result(self, n, reference):
        # Arrange
        func = MatplotlibBase()
        # Act
        result = func._get_matrix_fig_size(n)
        # Assert
        assert result == (reference, reference)


class Test_Format_String:
    long_string = [
        "123456789 123456789 123456789 123456789 123456789 123456789 123456789"
    ]

    @pytest.mark.parametrize(
        "n,string, reference",
        [
            (50, long_string, ["123456789 123456789 123456789 123456789 1234567..."]),
            (10, long_string, ["1234567..."]),
        ],
    )
    def test_result(self, n, string, reference):
        # Arrange
        func = MatplotlibBase()
        # Act
        result = func._format_string(
            string,
            th=n,
        )
        # Assert
        assert result == reference
