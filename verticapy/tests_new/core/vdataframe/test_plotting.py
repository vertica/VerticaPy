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
import pytest

import verticapy as vp


class TestPlotting:
    """
    test class for Plotting functions test
    """

    @pytest.mark.parametrize(
        "columns, max_nb_points",
        [(["SepalLengthCm", "SepalWidthCm", "PetalWidthCm"], 5)],
    )
    def test_scatter_matrix(self, iris_vd, columns, max_nb_points):
        """
        test function - scatter_matrix
        """
        vp.set_option("plotting_lib", "plotly")
        print(vp.get_option("plotting_lib"))

        res = iris_vd.scatter_matrix(columns=columns, max_nb_points=max_nb_points)
        assert len(res) == len(columns)
