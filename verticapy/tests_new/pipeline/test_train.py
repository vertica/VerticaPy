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

from verticapy import drop
from verticapy._utils._sql._sys import _executeSQL
from verticapy.datasets import (
    load_airline_passengers,
    load_iris,
    load_winequality,
)

from verticapy.machine_learning.vertica.cluster import (
    BisectingKMeans,
    DBSCAN,
    KMeans,
    KPrototypes,
    NearestCentroid,
)
from verticapy.machine_learning.vertica.decomposition import MCA, PCA, SVD
from verticapy.machine_learning.vertica.ensemble import (
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
    XGBClassifier,
    XGBRegressor,
)
from verticapy.machine_learning.vertica.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    PoissonRegressor,
    Ridge,
)
from verticapy.machine_learning.vertica.naive_bayes import (
    BernoulliNB,
    CategoricalNB,
    GaussianNB,
    MultinomialNB,
    NaiveBayes,
)
from verticapy.machine_learning.vertica.neighbors import (
    KNeighborsClassifier,
    KernelDensity,
    KNeighborsRegressor,
    LocalOutlierFactor,
)
from verticapy.machine_learning.vertica.svm import LinearSVC, LinearSVR
from verticapy.machine_learning.vertica.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    DummyTreeClassifier,
    DummyTreeRegressor,
)
from verticapy.machine_learning.vertica.tsa import ARIMA, ARMA, AR, MA

from verticapy.pipeline._train import training

from verticapy.tests_new.pipeline.conftest import pipeline_exists, pipeline_not_exists

SUPPORTED_FUNCTIONS = [
    BisectingKMeans,  # Winequality
    DBSCAN,  # [Not Tested]
    KMeans,  # Winequality
    KPrototypes,  # Winequality
    NearestCentroid,  # [FIXME: Non Native Function]
    MCA,  # [FIXME: Non Native Function]
    PCA,  # Winequality
    SVD,  # Winequality
    IsolationForest,  # Winequality
    RandomForestClassifier,  # Winequality
    RandomForestRegressor,  # Winequality
    XGBClassifier,  # Winequality
    XGBRegressor,  # Winequality
    ElasticNet,  # Winequality
    Lasso,  # Winequality
    LinearRegression,  # Winequality
    LogisticRegression,  # Winequality
    PoissonRegressor,  # Winequality
    Ridge,  # Winequality
    BernoulliNB,  # [Not Tested]
    CategoricalNB,  # Iris
    GaussianNB,  # [Not Tested]
    MultinomialNB,  # [Not Tested]
    NaiveBayes,  # Iris
    KNeighborsClassifier,  # [FIXME: Non Native Function]
    KernelDensity,  # [FIXME: Model_NAMING error]
    KNeighborsRegressor,  # Winequality
    LocalOutlierFactor,  # [FIXME: Non Native Function]
    LinearSVC,  # Winequality
    LinearSVR,  # Winequality
    DecisionTreeClassifier,  # Winequality
    DecisionTreeRegressor,  # Winequality
    DummyTreeClassifier,  # Winequality
    DummyTreeRegressor,  # Winequality
    ARIMA,  # Airline
    ARMA,  # Airline
    AR,  # Airline
    MA,  # Airline
]


class TestTrain:
    """
    Test winequality dependent models.
    """

    @pytest.mark.parametrize(
        "kwargs",
        [
            {
                "method": {
                    "name": "BisectingKMeans",
                    "params": {"n_cluster": 8, "max_iter": 3},
                },
                "cols": ["density", "sulphates"],
            },
            {
                "method": {
                    "name": "KMeans",
                    "params": {
                        "n_cluster": 8,
                        "init": "kmeanspp",
                        "max_iter": 300,
                        "tol": 1e-4,
                    },
                },
                "cols": ["density", "sulphates"],
            },
            {
                "method": {
                    "name": "KPrototypes",
                    "params": {"n_cluster": 8, "max_iter": 3},
                },
                "cols": ["density", "sulphates"],
            },
            {
                "method": {
                    "name": "PCA",
                    "params": {
                        "n_components": 3,
                    },
                },
                "cols": ["fixed_acidity", "citric_acid", "density", "sulphates"],
            },
            {
                "method": {
                    "name": "SVD",
                    "params": {
                        "n_components": 3,
                    },
                },
                "cols": ["fixed_acidity", "citric_acid", "density", "sulphates"],
            },
            {
                "method": {
                    "name": "IsolationForest",
                    "params": {"n_estimators": 10, "max_depth": 3, "nbins": 6},
                },
                "cols": ["density", "sulphates"],
            },
            {
                "method": {
                    "name": "RandomForestClassifier",
                    "target": "good",
                    "params": {
                        "max_features": "auto",
                        "max_leaf_nodes": 32,
                        "sample": 0.5,
                        "max_depth": 3,
                        "min_samples_leaf": 5,
                        "min_info_gain": 0.0,
                        "nbins": 32,
                    },
                },
                "cols": [
                    "fixed_acidity",
                    "volatile_acidity",
                    "citric_acid",
                    "residual_sugar",
                ],
            },
            {
                "method": {
                    "name": "RandomForestRegressor",
                    "target": "quality",
                    "params": {
                        "max_features": "auto",
                        "max_leaf_nodes": 32,
                        "sample": 0.5,
                        "max_depth": 3,
                        "min_samples_leaf": 5,
                        "min_info_gain": 0.0,
                        "nbins": 32,
                    },
                },
                "cols": [
                    "fixed_acidity",
                    "volatile_acidity",
                    "citric_acid",
                    "residual_sugar",
                ],
            },
            {
                "method": {
                    "name": "XGBClassifier",
                    "target": "good",
                    "params": {
                        "max_ntree": 3,
                        "max_depth": 3,
                        "nbins": 6,
                        "split_proposal_method": "global",
                        "tol": 0.001,
                        "learning_rate": 0.1,
                        "min_split_loss": 0,
                        "weight_reg": 0,
                        "sample": 0.7,
                        "col_sample_by_tree": 1,
                        "col_sample_by_node": 1,
                    },
                },
                "cols": [
                    "fixed_acidity",
                    "volatile_acidity",
                    "citric_acid",
                    "residual_sugar",
                ],
            },
            {
                "method": {
                    "name": "XGBRegressor",
                    "target": "quality",
                    "params": {
                        "max_ntree": 3,
                        "max_depth": 3,
                        "nbins": 6,
                        "split_proposal_method": "global",
                        "tol": 0.001,
                        "learning_rate": 0.1,
                        "min_split_loss": 0,
                        "weight_reg": 0,
                        "sample": 0.7,
                        "col_sample_by_tree": 1,
                        "col_sample_by_node": 1,
                    },
                },
                "cols": [
                    "fixed_acidity",
                    "volatile_acidity",
                    "citric_acid",
                    "residual_sugar",
                ],
            },
            {
                "method": {
                    "name": "ElasticNet",
                    "target": "quality",
                    "params": {
                        "tol": 1e-6,
                        "C": 1,
                        "max_iter": 100,
                        "solver": "CGD",
                        "l1_ratio": 0.5,
                        "fit_intercept": True,
                    },
                },
                "cols": [
                    "fixed_acidity",
                    "volatile_acidity",
                    "citric_acid",
                    "residual_sugar",
                ],
            },
            {
                "method": {
                    "name": "Lasso",
                    "target": "quality",
                    "params": {"tol": 1e-6, "C": 0.5, "max_iter": 100, "solver": "CGD"},
                },
                "cols": [
                    "fixed_acidity",
                    "volatile_acidity",
                    "citric_acid",
                    "residual_sugar",
                ],
            },
            {
                "method": {
                    "name": "LinearRegression",
                    "target": "quality",
                    "params": {
                        "tol": 1e-6,
                        "max_iter": 100,
                        "solver": "newton",
                        "fit_intercept": True,
                    },
                },
                "cols": [
                    "fixed_acidity",
                    "volatile_acidity",
                    "citric_acid",
                    "residual_sugar",
                ],
            },
            {
                "method": {
                    "name": "LogisticRegression",
                    "target": "good",
                    "params": {
                        "tol": 1e-6,
                        "max_iter": 100,
                        "solver": "newton",
                        "fit_intercept": True,
                    },
                },
                "cols": [
                    "fixed_acidity",
                    "volatile_acidity",
                    "citric_acid",
                    "residual_sugar",
                ],
            },
            {
                "method": {
                    "name": "PoissonRegressor",
                    "target": "quality",
                    "params": {
                        "tol": 1e-6,
                        "penalty": "L2",
                        "C": 1,
                        "max_iter": 100,
                        "fit_intercept": True,
                    },
                },
                "cols": [
                    "fixed_acidity",
                    "volatile_acidity",
                    "citric_acid",
                    "residual_sugar",
                ],
            },
            {
                "method": {
                    "name": "Ridge",
                    "target": "quality",
                    "params": {
                        "tol": 1e-6,
                        "C": 0.5,
                        "max_iter": 100,
                        "solver": "newton",
                    },
                },
                "cols": [
                    "fixed_acidity",
                    "volatile_acidity",
                    "citric_acid",
                    "residual_sugar",
                ],
            },
            {
                "method": {
                    "name": "LinearSVC",
                    "target": "good",
                    "params": {
                        "tol": 1e-4,
                        "C": 1.0,
                        "intercept_scaling": 1.0,
                        "intercept_mode": "regularized",
                        "max_iter": 100,
                        "class_weight": [1, 1],
                    },
                },
                "cols": [
                    "fixed_acidity",
                    "volatile_acidity",
                    "citric_acid",
                    "residual_sugar",
                ],
            },
            {
                "method": {
                    "name": "LinearSVR",
                    "target": "quality",
                    "params": {
                        "tol": 1e-4,
                        "C": 1.0,
                        "intercept_scaling": 1.0,
                        "intercept_mode": "regularized",
                        "acceptable_error_margin": 0.1,
                        "max_iter": 100,
                    },
                },
                "cols": [
                    "fixed_acidity",
                    "volatile_acidity",
                    "citric_acid",
                    "residual_sugar",
                ],
            },
            {
                "method": {
                    "name": "DecisionTreeClassifier",
                    "target": "good",
                    "params": {
                        "max_features": "auto",
                        "max_leaf_nodes": 32,
                        "max_depth": 3,
                        "min_info_gain": 0.0,
                        "nbins": 32,
                    },
                },
                "cols": [
                    "fixed_acidity",
                    "volatile_acidity",
                    "citric_acid",
                    "residual_sugar",
                ],
            },
            {
                "method": {
                    "name": "DecisionTreeRegressor",
                    "target": "quality",
                    "params": {
                        "max_features": "auto",
                        "max_leaf_nodes": 32,
                        "max_depth": 3,
                        "min_info_gain": 0.0,
                        "nbins": 32,
                    },
                },
                "cols": [
                    "fixed_acidity",
                    "volatile_acidity",
                    "citric_acid",
                    "residual_sugar",
                ],
            },
            {
                "method": {
                    "name": "DummyTreeClassifier",
                    "target": "good",
                },
                "cols": [
                    "fixed_acidity",
                    "volatile_acidity",
                    "citric_acid",
                    "residual_sugar",
                ],
            },
            {
                "method": {
                    "name": "DummyTreeRegressor",
                    "target": "quality",
                },
                "cols": [
                    "fixed_acidity",
                    "volatile_acidity",
                    "citric_acid",
                    "residual_sugar",
                ],
            },
        ],
    )
    def test_winequality(self, kwargs):
        _executeSQL("CALL drop_pipeline('public', 'test_pipeline');")
        table = load_winequality()
        print(kwargs)
        cols = kwargs["cols"]

        # return ``meta_sql, model, model_sql``
        _, model, _ = training(kwargs, table, "test_pipeline", cols)
        assert model

        assert pipeline_exists("test_pipeline", model=model)

        # drop pipeline
        _executeSQL("CALL drop_pipeline('public', 'test_pipeline');")

        assert pipeline_not_exists("test_pipeline", model=model)

        drop("public.winequality")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {
                "method": {
                    "name": "CategoricalNB",
                    "target": "Species",
                },
                "cols": [
                    "SepalLengthCm",
                    "SepalWidthCm",
                    "PetalLengthCm",
                    "PetalWidthCm",
                ],
            },
            {
                "method": {
                    "name": "NaiveBayes",
                    "target": "Species",
                },
                "cols": [
                    "SepalLengthCm",
                    "SepalWidthCm",
                    "PetalLengthCm",
                    "PetalWidthCm",
                ],
            },
        ],
    )
    def test_iris(self, kwargs):
        _executeSQL("CALL drop_pipeline('public', 'test_pipeline');")
        table = load_iris()
        print(kwargs)
        cols = kwargs["cols"]

        # return ``meta_sql, model, model_sql``
        _, model, _ = training(kwargs, table, "test_pipeline", cols)
        assert model

        assert pipeline_exists("test_pipeline", model=model)
        # drop pipeline
        _executeSQL("CALL drop_pipeline('public', 'test_pipeline');")

        assert pipeline_not_exists("test_pipeline", model=model)

        drop("public.iris")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {
                "method": {
                    "name": "ARIMA",
                    "target": "passengers",
                    "params": {"order": (12, 1, 2)},
                },
                "cols": "date",
            },
            {
                "method": {
                    "name": "ARMA",
                    "target": "passengers",
                    "params": {"order": (12, 1, 2)},
                },
                "cols": "date",
            },
            {
                "method": {"name": "AR", "target": "passengers", "params": {"p": 2}},
                "cols": "date",
            },
            {
                "method": {"name": "MA", "target": "passengers", "params": {"q": 2}},
                "cols": "date",
            },
        ],
    )
    def test_airline_passengers(self, kwargs):
        _executeSQL("CALL drop_pipeline('public', 'test_pipeline');")
        table = load_airline_passengers()
        print(kwargs)
        cols = kwargs["cols"]

        # return ``meta_sql, model, model_sql``
        _, model, _ = training(kwargs, table, "test_pipeline", cols)
        assert model

        assert pipeline_exists("test_pipeline", model=model)
        # drop pipeline
        _executeSQL("CALL drop_pipeline('public', 'test_pipeline');")
        assert pipeline_not_exists("test_pipeline", model=model)

        drop("public.airline_passengers")
