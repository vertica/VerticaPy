# (c) Copyright [2018-2022] Micro Focus or one of its affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Pytest
import pytest

# Other Modules
import matplotlib.pyplot as plt

# VerticaPy
import verticapy
from verticapy import (
    drop,
    set_option,
    create_verticapy_schema,
)
from verticapy.connect import current_cursor
from verticapy.datasets import load_titanic
from verticapy.learn.cluster import DBSCAN

set_option("print_info", False)


@pytest.fixture(scope="module")
def titanic_vd():
    titanic = load_titanic()
    yield titanic
    drop(name="public.titanic",)


@pytest.fixture(scope="module")
def model(titanic_vd):
    create_verticapy_schema()
    model_class = DBSCAN("DBSCAN_model_test",)
    model_class.drop()
    model_class.fit("public.titanic", ["age", "fare"])
    yield model_class
    model_class.drop()


class TestDBSCAN:
    def test_repr(self, model):
        assert "Additional Info" in model.__repr__()
        model_repr = DBSCAN("model_repr")
        model_repr.drop()
        assert model_repr.__repr__() == "<DBSCAN>"

    def test_drop(self):
        model_test = DBSCAN("model_test_drop",)
        model_test.drop()
        model_test.fit("public.titanic", ["age", "fare"], "survived")
        current_cursor().execute(
            "SELECT model_name FROM verticapy.models WHERE model_name IN ('model_test_drop', '\"model_test_drop\"')"
        )
        assert current_cursor().fetchone()[0] in (
            "model_test_drop",
            '"model_test_drop"',
        )

        model_test.drop()
        current_cursor().execute(
            "SELECT model_name FROM verticapy.models WHERE model_name IN ('model_test_drop', '\"model_test_drop\"')"
        )
        assert current_cursor().fetchone() is None

    def test_get_params(self, model):
        assert model.get_params() == {"eps": 0.5, "min_samples": 5, "p": 2}

    def test_get_predict(self, titanic_vd, model):
        titanic_copy = model.predict()

        assert titanic_copy["dbscan_cluster"].min() == pytest.approx(-1, abs=1e-6)

    def test_get_attr(self, model):
        result = model.get_attr()
        assert result["attr_name"] == ["n_cluster", "n_noise"]

    def test_get_plot(self, model):
        result = model.plot()
        assert len(result.get_default_bbox_extra_artists()) == 27
        plt.close("all")

    def test_set_params(self, model):
        model.set_params({"p": 1})

        assert model.get_params()["p"] == 1

    def test_model_from_vDF(self, titanic_vd):
        model_test = DBSCAN("dbscan_from_vDF")
        model_test.drop()
        model_test.fit(titanic_vd, ["age", "fare"], "survived")
        assert model_test.predict()["dbscan_cluster"].min() == pytest.approx(
            -1, abs=1e-6
        )
        model_test.drop()
