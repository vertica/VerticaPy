"""
Copyright  (c)  2018-2024 Open Text  or  one  of its
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

# Verticapy
from verticapy.plotting._matplotlib.base import MatplotlibBase

# Standard Python Modules
import matplotlib.pyplot as plt


@pytest.mark.parametrize("num, reference", [(1, 1), (2, 2)])
def test_get_ax_fig_axes(num, reference):
    """
    Test if correct object returned
    """
    # Arrange
    func = MatplotlibBase()
    # Act
    result = func._get_ax_fig(num)
    # Assert
    assert result[0] == reference and result[1] == plt


@pytest.mark.parametrize("num, reference", [(1, 3), (2, 4.5)])
def test_get_matrix_fig_size_result(num, reference):
    """
    Test matrix fig size
    """
    # Arrange
    func = MatplotlibBase()
    # Act
    result = func._get_matrix_fig_size(num)
    # Assert
    assert result == (reference, reference)


long_string = ["123456789 123456789 123456789 123456789 123456789 123456789 123456789"]


@pytest.mark.parametrize(
    "num,string, reference",
    [
        (50, long_string, ["123456789 123456789 123456789 123456789 1234567..."]),
        (10, long_string, ["1234567..."]),
    ],
)
def test_format_string_result(num, string, reference):
    """
    Test string formatting
    """
    # Arrange
    func = MatplotlibBase()
    # Act
    result = func._format_string(
        string,
        th=num,
    )
    # Assert
    assert result == reference
