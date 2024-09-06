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

# Other Modules
import matplotlib.pyplot as plt

# VerticaPy
from verticapy import (
    drop,
    set_option,
)
from verticapy.connection import current_cursor
from verticapy.datasets import load_titanic
from verticapy.learn.neighbors import KernelDensity

# Matplotlib skip
import matplotlib

matplotlib_version = matplotlib.__version__
skip_plt = pytest.mark.skipif(
    matplotlib_version > "3.5.2",
    reason="Test skipped on matplotlib version greater than 3.5.2",
)

set_option("print_info", False)


@pytest.fixture(scope="module")
def titanic_vd():
    titanic = load_titanic()
    yield titanic
    drop(
        name="public.titanic",
    )


@pytest.fixture(scope="module")
def model(titanic_vd):
    model_class = KernelDensity(
        "KernelDensity_model_test",
    )
    model_class.drop()
    model_class.fit("public.titanic", ["age", "fare"])
    yield model_class
    model_class.drop()


class TestKernelDensity:
    def test_repr(self, model):
        assert model.__repr__() == "<KernelDensity>"

    def test_get_params(self, model):
        assert model.get_params() == {
            "bandwidth": 1,
            "kernel": "gaussian",
            "max_depth": 5,
            "max_leaf_nodes": 1000000000,
            "min_samples_leaf": 1,
            "nbins": 5,
            "p": 2,
            "xlim": [],
        }

    def test_get_predict(self, titanic_vd, model):
        titanic_copy = model.predict(titanic_vd.copy(), name="kde")

        assert titanic_copy["kde"].mean() == pytest.approx(
            1.82115211838814e-06, abs=1e-6
        )

    def test_get_vertica_attributes(self, model):
        result = model.get_vertica_attributes()
        assert result["attr_name"][0] == "tree_count"

    @skip_plt
    def test_get_plot(self, model):
        result = model.plot()
        assert len(result.get_default_bbox_extra_artists()) == 8
        plt.close("all")
        model_test = KernelDensity("model_test_plot_kde_plot")
        model_test.drop()
        model_test.fit("public.titanic", ["age"])
        result = model_test.plot()
        assert len(result.get_default_bbox_extra_artists()) == 9
        model_test.drop()

    def test_set_params(self, model):
        model.set_params({"p": 1})

        assert model.get_params()["p"] == 1

    def test_model_from_vDF(self, titanic_vd):
        model_test = KernelDensity(
            "KernelDensity_from_vDF_tmp",
        )
        model_test.drop()
        model_test.fit(titanic_vd, ["age", "fare"])
        titanic_copy = model_test.predict(titanic_vd.copy(), name="kde")

        assert titanic_copy["kde"].mean() == pytest.approx(
            1.82115211838814e-06, abs=1e-6
        )
        model_test.drop()

    def test_overwrite_model(self, titanic_vd):
        model = KernelDensity("test_overwrite_model")
        model.drop()  # to isulate this test from any previous left over
        model.fit(titanic_vd, ["age", "fare"])

        # overwrite_model is false by default
        with pytest.raises(NameError) as exception_info:
            model.fit(titanic_vd, ["age", "fare"])
        assert exception_info.match("The model 'test_overwrite_model' already exists!")

        # overwriting the model when overwrite_model is specified true
        model = KernelDensity("test_overwrite_model", overwrite_model=True)
        model.fit(titanic_vd, ["age", "fare"])

        # cleaning up
        model.drop()
