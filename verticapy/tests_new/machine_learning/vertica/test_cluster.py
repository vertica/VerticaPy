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
from matplotlib.axes._axes import Axes

import verticapy._config.config as conf


@pytest.mark.parametrize("model_class", ["KMeans"])
class TestCluster:
    """
    test class - TestCluster
    """

    @pytest.mark.parametrize("return_report", [True, False])
    def test_fit(
        self,
        get_vpy_model,
        get_py_model,
        model_class,
        iris_vd_fun,
        return_report,
    ):
        """
        test function - fit
        """
        vpy_model_obj, py_model_obj = get_vpy_model(model_class), get_py_model(
            model_class
        )
        vpy_model_obj.model.drop()
        vpy_res = vpy_model_obj.model.fit(
            iris_vd_fun, py_model_obj.X, return_report=return_report
        )

        assert (
            isinstance(vpy_res, str)
            if return_report
            else isinstance(vpy_res, type(None))
        )

    def test_plot_voronoi(self, model_class, get_vpy_model, get_py_model):
        """
        test function - plot_voronoi
        """
        conf.set_option("plotting_lib", "matplotlib")
        X = list(get_py_model(model_class).X.columns)
        # print(X)
        # print(get_vpy_model(model_class, X=X[:2])[0].plot_voronoi())

        vpy_res = get_vpy_model(model_class, X=X[:2])[0].plot_voronoi()

        print(vpy_res.__dict__.keys())
        assert isinstance(vpy_res, Axes)
        # assert isinstance(vpy_res, plotly.graph_objs.Figure)
        # assert isinstance(vpy_res, Highchart)
