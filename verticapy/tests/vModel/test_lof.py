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
from vertica_python.errors import DuplicateObject

# VerticaPy
from verticapy import (
    drop,
    set_option,
)
from verticapy.connection import current_cursor
from verticapy.datasets import load_titanic
from verticapy.learn.neighbors import LocalOutlierFactor

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
    model_class = LocalOutlierFactor(
        "lof_model_test",
    )
    model_class.drop()
    model_class.fit("public.titanic", ["age", "fare"])
    yield model_class
    model_class.drop()


class TestLocalOutlierFactor:
    def test_repr(self, model):
        assert model.__repr__() == "<LocalOutlierFactor>"

    def test_get_params(self, model):
        assert model.get_params() == {"n_neighbors": 20, "p": 2}

    def test_get_predict(self, titanic_vd, model):
        titanic_copy = model.predict()

        assert titanic_copy["lof_score"].mean() == pytest.approx(
            1.17226637499694, abs=1e-6
        )

    def test_get_attributes(self, model):
        result = model.get_attributes()
        assert result == ["n_neighbors_", "p_", "n_errors_", "cnt_"]

    @skip_plt
    def test_get_plot(self, model):
        result = model.plot(color=["r", "b"])
        assert len(result.get_default_bbox_extra_artists()) == 9
        plt.close("all")
        model_test = LocalOutlierFactor("model_test_plot_lof")
        model_test.drop()
        model_test.fit("public.titanic", ["age"])
        result = model_test.plot(color=["r", "b"])
        assert len(result.get_default_bbox_extra_artists()) == 9
        model_test.drop()

        model_test = LocalOutlierFactor("model_test_plot_lof_2")
        model_test.drop()
        model_test.fit("public.titanic", ["age", "fare", "pclass"])
        result = model_test.plot(color=["r", "b"])
        assert len(result.get_default_bbox_extra_artists()) == 3
        model_test.drop()

    def test_set_params(self, model):
        model.set_params({"p": 1})

        assert model.get_params()["p"] == 1

    def test_model_from_vDF(self, titanic_vd):
        model_test = LocalOutlierFactor(
            "lof_from_vDF_tmp",
        )
        model_test.drop()
        model_test.fit(titanic_vd, ["age", "fare"])
        assert model_test.predict()["lof_score"].mean() == pytest.approx(
            1.17226637499694, abs=1e-6
        )
        model_test.drop()

    def test_overwrite_model(self, titanic_vd):
        model = LocalOutlierFactor("test_overwrite_model")
        model.drop()  # to isulate this test from any previous left over
        model.fit(titanic_vd, ["age", "fare"])

        # overwrite_model is false by default
        with pytest.raises(DuplicateObject) as exception_info:
            model.fit(titanic_vd, ["age", "fare"])
        assert 'Object "test_overwrite_model" already exists' in str(
            exception_info.value
        )

        # overwriting the model when overwrite_model is specified true
        model = LocalOutlierFactor("test_overwrite_model", overwrite_model=True)
        model.fit(titanic_vd, ["age", "fare"])

        # cleaning up
        model.drop()
