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

# Standard Python Modules
from abc import abstractmethod
import warnings

# VerticaPy
from verticapy import drop, set_option

import verticapy.mlops.model_tracking as mt
import verticapy.sql.sys as sys

set_option("print_info", False)


class ExperimentBase:
    """
    A base class for model tracking tests with different experiment types
    """

    def test_repr(self, experiment, representation):
        assert experiment.__repr__() == representation
        assert experiment.__str__() == representation

    def test_creating_db_backed_experiment(
        self,
        vdf_data,
        new_model,
        predictors,
        response,
        standard_metrics_0,
        ud_metrics,
        exp_type,
    ):
        experiment = mt.vExperiment(
            experiment_name="my_exp",
            test_relation=vdf_data,
            X=predictors,
            y=response,
            experiment_type=exp_type,
            experiment_table="my_exp_table",
        )

        experiment.add_model(new_model, ud_metrics)

        assert experiment._model_name_list[0] == new_model.model_name
        assert experiment._model_type_list[0] == new_model._model_type
        assert experiment._parameters[0] == new_model.get_params()
        if len(experiment._measured_metrics[0]) > 0:
            assert experiment._measured_metrics[0][0] == pytest.approx(
                standard_metrics_0
            )
        if experiment._user_defined_metrics[0]:
            assert experiment._user_defined_metrics[0] == ud_metrics
        assert sys.does_table_exist("my_exp_table", "public")

        experiment.drop()
        assert not new_model.does_model_exists(new_model.model_name)
        assert not sys.does_table_exist("my_exp_table", "public")

    def test_creating_in_memory_experiment(
        self,
        vdf_data,
        new_model,
        predictors,
        response,
        standard_metrics_0,
        ud_metrics,
        exp_type,
    ):
        with pytest.warns(Warning) as record:
            experiment = mt.vExperiment(
                experiment_name="my_exp",
                test_relation=vdf_data,
                X=predictors,
                y=response,
                experiment_type=exp_type,
            )
        assert "not be backed up" in record[0].message.args[0]

        experiment.add_model(new_model, ud_metrics)

        assert experiment._model_name_list[0] == new_model.model_name
        assert experiment._model_type_list[0] == new_model._model_type
        assert experiment._parameters[0] == new_model.get_params()
        if len(experiment._measured_metrics[0]) > 0:
            assert experiment._measured_metrics[0][0] == pytest.approx(
                standard_metrics_0
            )
        if experiment._user_defined_metrics[0]:
            assert experiment._user_defined_metrics[0] == ud_metrics

        experiment.drop()
        assert not new_model.does_model_exists(new_model.model_name)

    def test_loading_experiment_from_db(
        self, experiment, vdf_data, predictors, response
    ):
        new_experiment = mt.vExperiment(
            experiment_name=experiment.experiment_name,
            test_relation=vdf_data,
            X=predictors,
            y=response,
            experiment_table=experiment.experiment_table,
        )

        assert new_experiment.experiment_type == experiment.experiment_type

        assert len(new_experiment._model_id_list) == len(experiment._model_id_list)
        for index, id in enumerate(new_experiment._model_id_list):
            assert id == new_experiment._model_id_list[index]

    def test_list_models(self, experiment, list_of_models):
        ts = experiment.list_models()
        assert ts.values["model_name"] == list_of_models

    def test_load_best_model(self, experiment, best_model_name, metric):
        best_model = experiment.load_best_model(metric)
        assert best_model.model_name == best_model_name


######################### Regressor  ##############################

from verticapy.learn.linear_model import LinearRegression
from verticapy.learn.linear_model import Ridge
from verticapy.learn.svm import LinearSVR


@pytest.fixture(scope="module")
def reg_model1(winequality_vpy):
    model = LinearRegression("reg_m1", solver="bfgs", max_iter=1)
    model.drop()

    model.fit(
        winequality_vpy,
        ["citric_acid", "residual_sugar", "alcohol"],
        "quality",
    )
    yield model
    model.drop()


@pytest.fixture(scope="module")
def reg_model2(winequality_vpy):
    model = LinearRegression("reg_m2", solver="bfgs", max_iter=3)
    model.drop()

    model.fit(
        winequality_vpy,
        ["citric_acid", "residual_sugar", "alcohol"],
        "quality",
    )
    yield model
    model.drop()


@pytest.fixture(scope="module")
def reg_model3(winequality_vpy):
    model = LinearSVR("reg_m3", max_iter=5)
    model.drop()

    model.fit(
        winequality_vpy,
        ["citric_acid", "residual_sugar", "alcohol"],
        "quality",
    )
    yield model
    model.drop()


@pytest.fixture(scope="module")
def regressor_experiment(winequality_vpy, reg_model1, reg_model2, reg_model3):
    experiment = mt.vExperiment(
        experiment_name="reg_exp",
        test_relation=winequality_vpy,
        X=["citric_acid", "residual_sugar", "alcohol"],
        y="quality",
        experiment_type="auto",
        experiment_table="reg_exp_table",
    )

    experiment.add_model(reg_model1)
    experiment.add_model(reg_model2)
    experiment.add_model(reg_model3)

    yield experiment
    experiment.drop()


class TestRegressorExperiment(ExperimentBase):
    def test_repr(self, regressor_experiment):
        representation = "<experiment_name: reg_exp, experiment_type: regressor>"
        super().test_repr(regressor_experiment, representation)

    def test_creating_db_backed_experiment(self, winequality_vpy):
        predictors = ["citric_acid", "residual_sugar", "alcohol"]
        response = "quality"

        model = Ridge("reg_m4", max_iter=5, overwrite_model=True)
        model.fit(winequality_vpy, predictors, response)

        standard_metrics_0 = 0.2198162448
        ud_metrics = None

        super().test_creating_db_backed_experiment(
            winequality_vpy,
            model,
            predictors,
            response,
            standard_metrics_0,
            ud_metrics,
            "regressor",
        )

    def test_creating_in_memory_experiment(self, winequality_vpy):
        predictors = ["citric_acid", "residual_sugar", "alcohol"]
        response = "quality"

        model = Ridge("reg_m4", max_iter=5, overwrite_model=True)
        model.fit(winequality_vpy, predictors, response)

        standard_metrics_0 = 0.2198162448
        ud_metrics = None

        super().test_creating_in_memory_experiment(
            winequality_vpy,
            model,
            predictors,
            response,
            standard_metrics_0,
            ud_metrics,
            "regressor",
        )

    def test_loading_experiment_from_db(self, regressor_experiment, winequality_vpy):
        predictors = ["citric_acid", "residual_sugar", "alcohol"]
        response = "quality"

        super().test_loading_experiment_from_db(
            regressor_experiment, winequality_vpy, predictors, response
        )

    def test_list_models(self, regressor_experiment):
        list_of_models = ["reg_m1", "reg_m2", "reg_m3"]
        super().test_list_models(regressor_experiment, list_of_models)

    def test_load_best_model(self, regressor_experiment):
        best_model_name = "reg_m3"
        super().test_load_best_model(
            regressor_experiment, best_model_name, "mean_squared_error"
        )


######################### Binary  ##############################

from verticapy.learn.linear_model import LogisticRegression
from verticapy.learn.svm import LinearSVC
from verticapy.learn.tree import DecisionTreeClassifier


@pytest.fixture(scope="module")
def bin_model1(winequality_vpy):
    model = LogisticRegression("bin_m1", solver="newton", max_iter=5, penalty=None)
    model.drop()

    model.fit(
        winequality_vpy,
        ["citric_acid", "residual_sugar", "alcohol"],
        "good",
    )
    yield model
    model.drop()


@pytest.fixture(scope="module")
def bin_model2(winequality_vpy):
    model = LogisticRegression("bin_m2", solver="bfgs", max_iter=5, penalty=None)
    model.drop()

    model.fit(
        winequality_vpy,
        ["citric_acid", "residual_sugar", "alcohol"],
        "good",
    )
    yield model
    model.drop()


@pytest.fixture(scope="module")
def bin_model3(winequality_vpy):
    model = LinearSVC("bin_m3", max_iter=5)
    model.drop()

    model.fit(
        winequality_vpy,
        ["citric_acid", "residual_sugar", "alcohol"],
        "good",
    )
    yield model
    model.drop()


@pytest.fixture(scope="module")
def binary_experiment(winequality_vpy, bin_model1, bin_model2, bin_model3):
    experiment = mt.vExperiment(
        experiment_name="bin_exp",
        test_relation=winequality_vpy,
        X=["citric_acid", "residual_sugar", "alcohol"],
        y="good",
        experiment_type="binary",
        experiment_table="bin_exp_table",
    )

    experiment.add_model(bin_model1)
    experiment.add_model(bin_model2)
    experiment.add_model(bin_model3)

    yield experiment
    experiment.drop()


class TestBinaryExperiment(ExperimentBase):
    def test_repr(self, binary_experiment):
        representation = "<experiment_name: bin_exp, experiment_type: binary>"
        super().test_repr(binary_experiment, representation)

    def test_creating_db_backed_experiment(self, winequality_vpy):
        predictors = ["citric_acid", "residual_sugar", "alcohol"]
        response = "good"

        model = DecisionTreeClassifier(
            "bin_m4", max_features="max", max_depth=3, overwrite_model=True
        )
        model.fit(winequality_vpy, predictors, response)

        standard_metrics_0 = 0.7844518552
        ud_metrics = None

        super().test_creating_db_backed_experiment(
            winequality_vpy,
            model,
            predictors,
            response,
            standard_metrics_0,
            ud_metrics,
            "binary",
        )

    def test_creating_in_memory_experiment(self, winequality_vpy):
        predictors = ["citric_acid", "residual_sugar", "alcohol"]
        response = "good"

        model = DecisionTreeClassifier(
            "reg_m4", max_features="max", max_depth=3, overwrite_model=True
        )
        model.fit(winequality_vpy, predictors, response)

        standard_metrics_0 = 0.7844518552
        ud_metrics = None

        super().test_creating_in_memory_experiment(
            winequality_vpy,
            model,
            predictors,
            response,
            standard_metrics_0,
            ud_metrics,
            "binary",
        )

    def test_loading_experiment_from_db(self, binary_experiment, winequality_vpy):
        predictors = ["citric_acid", "residual_sugar", "alcohol"]
        response = "good"

        super().test_loading_experiment_from_db(
            binary_experiment, winequality_vpy, predictors, response
        )

    def test_list_models(self, binary_experiment):
        list_of_models = ["bin_m1", "bin_m2", "bin_m3"]
        super().test_list_models(binary_experiment, list_of_models)

    def test_load_best_model(self, binary_experiment):
        best_model_name = "bin_m1"
        super().test_load_best_model(binary_experiment, best_model_name, "f1_score")


######################### Multi  ##############################
from verticapy.learn.tree import DecisionTreeClassifier
from verticapy.learn.ensemble import RandomForestClassifier


@pytest.fixture(scope="module")
def multi_model1(iris_vd):
    model = DecisionTreeClassifier("multi_m1", max_features="max", max_depth=3)
    model.drop()

    model.fit(
        iris_vd,
        ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
        "Species",
    )
    yield model
    model.drop()


@pytest.fixture(scope="module")
def multi_model2(iris_vd):
    model = DecisionTreeClassifier("multi_m2", max_features="max", max_depth=2)
    model.drop()

    model.fit(
        iris_vd,
        ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
        "Species",
    )
    yield model
    model.drop()


@pytest.fixture(scope="module")
def multi_model3(iris_vd):
    model = RandomForestClassifier(
        "multi_m3", max_features="max", max_depth=3, n_estimators=3, sample=1
    )
    model.drop()

    model.fit(
        iris_vd,
        ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
        "Species",
    )
    yield model
    model.drop()


@pytest.fixture(scope="module")
def multi_experiment(iris_vd, multi_model1, multi_model2, multi_model3):
    experiment = mt.vExperiment(
        experiment_name="multi_exp",
        test_relation=iris_vd,
        X=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
        y="Species",
        experiment_table="multi_exp_table",
    )

    experiment.add_model(multi_model1)
    experiment.add_model(multi_model2)
    experiment.add_model(multi_model3)

    yield experiment
    experiment.drop()


class TestMultiExperiment(ExperimentBase):
    def test_repr(self, multi_experiment):
        representation = "<experiment_name: multi_exp, experiment_type: multi>"
        super().test_repr(multi_experiment, representation)

    def test_creating_db_backed_experiment(self, iris_vd):
        predictors = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
        response = "Species"

        model = DecisionTreeClassifier(
            "multi_m4", max_features="max", max_depth=3, overwrite_model=True
        )
        model.fit(iris_vd, predictors, response)

        standard_metrics_0 = 0.9822222222
        ud_metrics = None

        super().test_creating_db_backed_experiment(
            iris_vd,
            model,
            predictors,
            response,
            standard_metrics_0,
            ud_metrics,
            "multi",
        )

    def test_creating_in_memory_experiment(self, iris_vd):
        predictors = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
        response = "Species"

        model = DecisionTreeClassifier(
            "multi_m4", max_features="max", max_depth=3, overwrite_model=True
        )
        model.fit(iris_vd, predictors, response)

        standard_metrics_0 = 0.9822222222
        ud_metrics = None

        super().test_creating_in_memory_experiment(
            iris_vd,
            model,
            predictors,
            response,
            standard_metrics_0,
            ud_metrics,
            "multi",
        )

    def test_loading_experiment_from_db(self, multi_experiment, iris_vd):
        predictors = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
        response = "Species"

        super().test_loading_experiment_from_db(
            multi_experiment, iris_vd, predictors, response
        )

    def test_list_models(self, multi_experiment):
        list_of_models = ["multi_m1", "multi_m2", "multi_m3"]
        super().test_list_models(multi_experiment, list_of_models)

    def test_load_best_model(self, multi_experiment):
        best_model_name = "multi_m1"
        super().test_load_best_model(
            multi_experiment, best_model_name, "weighted_precision"
        )


######################### Clustering  ##############################
from verticapy.learn.cluster import KMeans
from verticapy.learn.cluster import BisectingKMeans


@pytest.fixture(scope="module")
def clustering_model1(iris_vd):
    model = KMeans("clustering_m1", n_cluster=2, max_iter=5)
    model.drop()

    model.fit(
        iris_vd, ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    )
    yield model
    model.drop()


@pytest.fixture(scope="module")
def clustering_model2(iris_vd):
    model = KMeans("clustering_m2", n_cluster=3, max_iter=5)
    model.drop()

    model.fit(
        iris_vd, ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    )
    yield model
    model.drop()


@pytest.fixture(scope="module")
def clustering_model3(iris_vd):
    model = BisectingKMeans("clustering_m3", n_cluster=3, max_iter=5)
    model.drop()

    model.fit(
        iris_vd, ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    )
    yield model
    model.drop()


@pytest.fixture(scope="module")
def clustering_experiment(
    iris_vd, clustering_model1, clustering_model2, clustering_model3
):
    experiment = mt.vExperiment(
        experiment_name="clustering_exp",
        test_relation=iris_vd,
        X=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
        y=None,
        experiment_type="auto",
        experiment_table="clustering_exp_table",
    )

    experiment.add_model(clustering_model1, {"metric1": 1.1, "metric2": 2.1})
    experiment.add_model(clustering_model2, {"metric1": 1.2, "metric2": 2.2})
    experiment.add_model(clustering_model3, {"metric1": 1.3, "metric2": 2.3})

    yield experiment
    experiment.drop()


class TestClusteringExperiment(ExperimentBase):
    def test_repr(self, clustering_experiment):
        representation = (
            "<experiment_name: clustering_exp, experiment_type: clustering>"
        )
        super().test_repr(clustering_experiment, representation)

    def test_creating_db_backed_experiment(self, iris_vd):
        predictors = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
        response = None

        model = KMeans("clustering_m4", n_cluster=3, max_iter=5, overwrite_model=True)
        model.fit(iris_vd, predictors)

        standard_metrics_0 = None
        ud_metrics = {"metric1": 1.1, "metric2": 2.1}

        super().test_creating_db_backed_experiment(
            iris_vd,
            model,
            predictors,
            response,
            standard_metrics_0,
            ud_metrics,
            "clustering",
        )

    def test_creating_in_memory_experiment(self, iris_vd):
        predictors = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
        response = None

        model = KMeans("clustering_m4", n_cluster=3, max_iter=5, overwrite_model=True)
        model.fit(iris_vd, predictors)

        standard_metrics_0 = None
        ud_metrics = {"metric1": 1.1, "metric2": 2.1}

        super().test_creating_in_memory_experiment(
            iris_vd,
            model,
            predictors,
            response,
            standard_metrics_0,
            ud_metrics,
            "clustering",
        )

    def test_loading_experiment_from_db(self, clustering_experiment, iris_vd):
        predictors = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
        response = None

        super().test_loading_experiment_from_db(
            clustering_experiment, iris_vd, predictors, response
        )

    def test_list_models(self, clustering_experiment):
        list_of_models = ["clustering_m1", "clustering_m2", "clustering_m3"]
        super().test_list_models(clustering_experiment, list_of_models)

    def test_load_best_model(self, clustering_experiment):
        best_model_name = "clustering_m3"
        super().test_load_best_model(clustering_experiment, best_model_name, "metric2")
