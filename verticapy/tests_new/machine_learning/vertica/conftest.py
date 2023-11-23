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
from collections import namedtuple
import math
import os
import sys
import pytest
import numpy as np
import sklearn.metrics as skl_metrics
import sklearn.linear_model as skl_linear_model
import sklearn.svm as skl_svm
import sklearn.ensemble as skl_ensemble
import sklearn.tree as skl_tree
import sklearn.dummy as skl_dummy
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import f
import verticapy as vp
import verticapy.machine_learning.vertica as vpy_linear_model
import verticapy.machine_learning.vertica.svm as vpy_svm
import verticapy.machine_learning.vertica.tree as vpy_tree
import verticapy.machine_learning.vertica.tsa as vpy_tsa
import verticapy.machine_learning.vertica.ensemble as vpy_ensemble
from verticapy.connection import current_cursor
from verticapy.tests_new.machine_learning.metrics.test_classification_metrics import (
    python_metrics,
)

if sys.version_info < (3, 12):
    import tensorflow as tf

le = LabelEncoder()


@pytest.fixture(autouse=True)
def set_plotting_lib():
    vp.set_option("plotting_lib", "matplotlib")


@pytest.fixture(name="get_vpy_model", scope="function")
def get_vpy_model_fixture(
    winequality_vpy_fun, titanic_vd_fun, airline_vd_fun, schema_loader
):
    """
    getter function for vertica tree model
    """

    def _get_vpy_model(model_class, X=None, y=None, **kwargs):
        schema_name, model_name = schema_loader, "vpy_model"

        tree_param_map = {}

        rf_params_map = {
            "ntree": 10,
            "mtry": 2,
            "max_breadth": 10,
            "sampling_size": 0.632,
            "max_depth": 10,
            "min_leaf_size": 1,
            "nbins": 32,
        }

        decision_params_map = {
            "ntree": 1,
            "mtry": 2,
            "max_breadth": 10,
            "sampling_size": 1,
            "max_depth": 10,
            "min_leaf_size": 1,
            "nbins": 32,
        }

        dummy_params_map = {
            "ntree": 1,
            "mtry": 2,
            "max_breadth": 1000000000,
            "sampling_size": 1,
            "max_depth": 100,
            "min_leaf_size": 1,
            "nbins": 1000,
        }

        xgb_params_map = {
            "max_ntree": 10,
            "max_depth": 10,
            "nbins": 150,
            "split_proposal_method": "'global'",
            "tol": 0.001,
            "learning_rate": 0.1,
            "min_split_loss": 0.0,
            "weight_reg": 0.0,
            "sample": 1.0,
            "col_sample_by_tree": 1.0,
            "col_sample_by_node": 1.0,
        }

        if kwargs.get("solver"):
            solver = kwargs.get("solver")
        else:
            if model_class in ["Lasso", "ElasticNet"]:
                solver = "cgd"
            else:
                solver = "newton"

        if model_class in ["RandomForestRegressor", "RandomForestClassifier"]:
            model = getattr(vpy_tree, model_class)(
                f"{schema_name}.{model_name}",
                overwrite_model=kwargs.get("overwrite_model")
                if kwargs.get("overwrite_model")
                else False,
                n_estimators=kwargs.get("n_estimators")
                if kwargs.get("n_estimators")
                else 10,
                max_features=kwargs.get("max_features")
                if kwargs.get("max_features")
                else 2,
                max_leaf_nodes=kwargs.get("max_leaf_nodes")
                if kwargs.get("max_leaf_nodes")
                else 10,
                sample=kwargs.get("sample") if kwargs.get("sample") else 0.632,
                max_depth=kwargs.get("max_depth") if kwargs.get("max_depth") else 10,
                min_samples_leaf=kwargs.get("min_samples_leaf")
                if kwargs.get("min_samples_leaf")
                else 1,
                min_info_gain=kwargs.get("min_info_gain")
                if kwargs.get("min_info_gain")
                else 0.0,
                nbins=kwargs.get("nbins") if kwargs.get("nbins") else 32,
            )
            tree_param_map = rf_params_map
        elif model_class in ["DecisionTreeRegressor", "DecisionTreeClassifier"]:
            model = getattr(vpy_tree, model_class)(
                f"{schema_name}.{model_name}",
                overwrite_model=kwargs.get("overwrite_model")
                if kwargs.get("overwrite_model")
                else False,
                max_features=kwargs.get("max_features")
                if kwargs.get("max_features")
                else 2,
                max_leaf_nodes=kwargs.get("max_leaf_nodes")
                if kwargs.get("max_leaf_nodes")
                else 10,
                max_depth=kwargs.get("max_depth") if kwargs.get("max_depth") else 10,
                min_samples_leaf=kwargs.get("min_samples_leaf")
                if kwargs.get("min_samples_leaf")
                else 1,
                min_info_gain=kwargs.get("min_info_gain")
                if kwargs.get("min_info_gain")
                else 0.0,
                nbins=kwargs.get("nbins") if kwargs.get("nbins") else 32,
            )
            tree_param_map = decision_params_map
        elif model_class in ["XGBRegressor", "XGBClassifier"]:
            model = getattr(vpy_ensemble, model_class)(
                f"{schema_name}.{model_name}",
                overwrite_model=kwargs.get("overwrite_model")
                if kwargs.get("overwrite_model")
                else False,
                max_ntree=kwargs.get("max_ntree") if kwargs.get("max_ntree") else 10,
                max_depth=kwargs.get("max_depth") if kwargs.get("max_depth") else 10,
                nbins=kwargs.get("nbins") if kwargs.get("nbins") else 150,
                split_proposal_method=kwargs.get("split_proposal_method")
                if kwargs.get("split_proposal_method")
                else "global",
                tol=kwargs.get("tol") if kwargs.get("tol") else 0.001,
                learning_rate=kwargs.get("learning_rate")
                if kwargs.get("learning_rate")
                else 0.1,
                min_split_loss=kwargs.get("min_split_loss")
                if kwargs.get("min_split_loss")
                else 0.0,
                weight_reg=kwargs.get("weight_reg")
                if kwargs.get("weight_reg")
                else 0.0,
                sample=kwargs.get("sample") if kwargs.get("sample") else 1.0,
                col_sample_by_tree=kwargs.get("col_sample_by_tree")
                if kwargs.get("col_sample_by_tree")
                else 1.0,
                col_sample_by_node=kwargs.get("col_sample_by_node")
                if kwargs.get("col_sample_by_node")
                else 1.0,
            )
            tree_param_map = xgb_params_map
        elif model_class in ["DummyTreeRegressor", "DummyTreeClassifier"]:
            model = getattr(vpy_tree, model_class)(f"{schema_name}.{model_name}")
            tree_param_map = dummy_params_map
        elif model_class == "LinearSVR":
            model = getattr(vpy_svm, model_class)(
                f"{schema_name}.{model_name}",
                overwrite_model=kwargs.get("overwrite_model")
                if kwargs.get("overwrite_model")
                else False,
                tol=kwargs.get("tol") if kwargs.get("tol") else 1e-4,
                C=kwargs.get("c") if kwargs.get("c") else 1.0,
                intercept_scaling=kwargs.get("intercept_scaling")
                if kwargs.get("intercept_scaling")
                else 1.0,
                intercept_mode=kwargs.get("intercept_mode")
                if kwargs.get("intercept_mode")
                else "regularized",
                acceptable_error_margin=kwargs.get("acceptable_error_margin")
                if kwargs.get("acceptable_error_margin")
                else 0.1,
                max_iter=kwargs.get("max_iter") if kwargs.get("max_iter") else 100,
            )
        elif model_class == "LinearRegression":
            model = getattr(vpy_linear_model, model_class)(
                f"{schema_name}.{model_name}",
                overwrite_model=kwargs.get("overwrite_model")
                if kwargs.get("overwrite_model")
                else False,
                tol=kwargs.get("tol") if kwargs.get("tol") else 1e-6,
                max_iter=kwargs.get("max_iter") if kwargs.get("max_iter") else 100,
                solver=solver,
                fit_intercept=kwargs.get("fit_intercept")
                if kwargs.get("fit_intercept")
                else True,
            )
        elif model_class == "ElasticNet":
            model = getattr(vpy_linear_model, model_class)(
                f"{schema_name}.{model_name}",
                overwrite_model=kwargs.get("overwrite_model")
                if kwargs.get("overwrite_model")
                else False,
                tol=kwargs.get("tol") if kwargs.get("tol") else 1e-6,
                C=kwargs.get("c") if kwargs.get("c") else 1.0,
                max_iter=kwargs.get("max_iter") if kwargs.get("max_iter") else 100,
                solver=solver,
                l1_ratio=kwargs.get("l1_ratio") if kwargs.get("l1_ratio") else 0.5,
                fit_intercept=kwargs.get("fit_intercept")
                if kwargs.get("fit_intercept")
                else True,
            )
        elif model_class == "PoissonRegressor":
            model = getattr(vpy_linear_model, model_class)(
                f"{schema_name}.{model_name}",
                overwrite_model=kwargs.get("overwrite_model")
                if kwargs.get("overwrite_model")
                else False,
                penalty=kwargs.get("penalty") if kwargs.get("penalty") else "l2",
                tol=kwargs.get("tol") if kwargs.get("tol") else 1e-6,
                C=kwargs.get("c") if kwargs.get("c") else 1.0,
                max_iter=kwargs.get("max_iter") if kwargs.get("max_iter") else 100,
                solver=solver,
                fit_intercept=kwargs.get("fit_intercept")
                if kwargs.get("fit_intercept")
                else True,
            )
        elif model_class == "AR":
            model = getattr(vpy_tsa, model_class)(
                f"{schema_name}.{model_name}",
                overwrite_model=kwargs.get("overwrite_model")
                if kwargs.get("overwrite_model")
                else False,
                p=kwargs.get("p") if kwargs.get("p") else 3,
                method=kwargs.get("method") if kwargs.get("method") else "ols",
                penalty=kwargs.get("penalty") if kwargs.get("penalty") else "none",
                C=kwargs.get("c") if kwargs.get("c") else 1.0,
                missing=kwargs.get("missing")
                if kwargs.get("missing")
                else "linear_interpolation",
                # compute_mse=kwargs.get("compute_mse")
                # if kwargs.get("compute_mse")
                # else True,
            )
        elif model_class == "MA":
            model = getattr(vpy_tsa, model_class)(
                f"{schema_name}.{model_name}",
                overwrite_model=kwargs.get("overwrite_model")
                if kwargs.get("overwrite_model")
                else False,
                q=kwargs.get("q") if kwargs.get("q") else 1,
                penalty=kwargs.get("penalty") if kwargs.get("penalty") else "none",
                C=kwargs.get("c") if kwargs.get("c") else 1.0,
                missing=kwargs.get("missing")
                if kwargs.get("missing")
                else "linear_interpolation",
            )
        elif model_class == "ARMA":
            model = getattr(vpy_tsa, model_class)(
                f"{schema_name}.{model_name}",
                overwrite_model=kwargs.get("overwrite_model")
                if kwargs.get("overwrite_model")
                else False,
                order=kwargs.get("order") if kwargs.get("order") else (2, 1),
                tol=kwargs.get("tol") if kwargs.get("tol") else 1e-06,
                max_iter=kwargs.get("max_iter") if kwargs.get("max_iter") else 100,
                init=kwargs.get("init") if kwargs.get("init") else "zero",
                missing=kwargs.get("missing")
                if kwargs.get("missing")
                else "linear_interpolation",
                # compute_mse=kwargs.get("compute_mse")
                # if kwargs.get("compute_mse")
                # else True,
            )
        elif model_class == "ARIMA":
            model = getattr(vpy_tsa, model_class)(
                f"{schema_name}.{model_name}",
                overwrite_model=kwargs.get("overwrite_model")
                if kwargs.get("overwrite_model")
                else False,
                order=kwargs.get("order") if kwargs.get("order") else (2, 1, 1),
                tol=kwargs.get("tol") if kwargs.get("tol") else 1e-06,
                max_iter=kwargs.get("max_iter") if kwargs.get("max_iter") else 100,
                init=kwargs.get("init") if kwargs.get("init") else "zero",
                missing=kwargs.get("missing")
                if kwargs.get("missing")
                else "linear_interpolation",
                # compute_mse=kwargs.get("compute_mse")
                # if kwargs.get("compute_mse")
                # else True,
            )
        else:
            model = getattr(vpy_linear_model, model_class)(
                f"{schema_name}.{model_name}",
                overwrite_model=kwargs.get("overwrite_model")
                if kwargs.get("overwrite_model")
                else False,
                tol=kwargs.get("tol") if kwargs.get("tol") else 1e-6,
                C=kwargs.get("c") if kwargs.get("c") else 1.0,
                max_iter=kwargs.get("max_iter") if kwargs.get("max_iter") else 100,
                solver=solver,
                fit_intercept=kwargs.get("fit_intercept")
                if kwargs.get("fit_intercept")
                else True,
            )

        print(f"VerticaPy Training Parameters: {model.get_params()}")
        model.drop()

        if model_class in [
            "RandomForestClassifier",
            "DecisionTreeClassifier",
            "DummyTreeClassifier",
            "XGBClassifier",
        ]:
            delete_sql = f"DELETE FROM {schema_name}.titanic WHERE AGE IS NULL OR FARE IS NULL OR SEX IS NULL OR SURVIVED IS NULL"
            print(f"Delete SQL: {delete_sql}")
            current_cursor().execute(delete_sql)

            # added to remove duplicate record with same name
            delete_name_sql = f"delete from {schema_name}.titanic where name in ('Kelly, Mr. James', 'Connolly, Miss. Kate')"
            print(f"Delete Name SQL: {delete_name_sql}")
            current_cursor().execute(delete_name_sql)

            if X is None:
                X = ["age", "fare", "sex"]
            if y is None:
                y = "survived"

            predictor_columns = ",".join(X)
            _X = [f'"{i}"' for i in X]

            if model_class == "XGBClassifier":
                train_sql = f"SELECT xgb_classifier('{schema_name}.{model_name}', '{schema_name}.titanic', '{y}', '{predictor_columns}' USING PARAMETERS exclude_columns='name', max_ntree={tree_param_map['max_ntree']}, max_depth={tree_param_map['max_depth']}, nbins={tree_param_map['nbins']}, split_proposal_method={tree_param_map['split_proposal_method']}, tol={tree_param_map['tol']}, learning_rate={tree_param_map['learning_rate']}, min_split_loss={tree_param_map['min_split_loss']}, weight_reg={tree_param_map['weight_reg']}, sample={tree_param_map['sample']}, col_sample_by_tree={tree_param_map['col_sample_by_tree']}, col_sample_by_node={tree_param_map['col_sample_by_node']}, seed=1, id_column='name')"
            else:
                train_sql = f"SELECT rf_classifier('{schema_name}.{model_name}', '{schema_name}.titanic', '{y}', '{predictor_columns}' USING PARAMETERS exclude_columns='name', ntree={tree_param_map['ntree']}, mtry={tree_param_map['mtry']}, max_breadth={tree_param_map['max_breadth']}, sampling_size={tree_param_map['sampling_size']}, max_depth={tree_param_map['max_depth']}, min_leaf_size={tree_param_map['min_leaf_size']}, nbins={tree_param_map['nbins']}, seed=1, id_column='name')"
            print(f"Tree Classifier Train SQL: {train_sql}")
            current_cursor().execute(train_sql)

            model.input_relation = f"{schema_name}.titanic"
            model.test_relation = model.input_relation
            model.X = _X
            model.y = f'"{y}"'
            model._compute_attributes()

            pred_vdf = model.predict(titanic_vd_fun, name=f"{y}_pred")[
                f"{y}_pred"
            ].astype("int")

            pred_prob_vdf = model.predict_proba(titanic_vd_fun, name=f"{y}_pred")

            y_class = titanic_vd_fun[y].distinct()
            pred_prob_vdf[f"{y}_pred"].astype("int")
            for i in y_class:
                pred_prob_vdf[f"{y}_pred_{i}"].astype("float")

        elif model_class in [
            "RandomForestRegressor",
            "DecisionTreeRegressor",
            "DummyTreeRegressor",
            "XGBRegressor",
        ]:
            # adding id column to winequality. id column is needed for seed parm for tree based model
            current_cursor().execute(
                f"ALTER TABLE {schema_name}.winequality ADD COLUMN id int"
            )
            seq_sql = f"CREATE SEQUENCE IF NOT EXISTS {schema_name}.sequence_auto_increment START 1"
            print(f"Sequence SQL: {seq_sql}")
            current_cursor().execute(seq_sql)
            current_cursor().execute(
                f"CREATE TABLE {schema_name}.winequality1 as select * from {schema_name}.winequality limit 0"
            )
            current_cursor().execute(
                f"insert into {schema_name}.winequality1 select fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol,quality,good,color, NEXTVAL('{schema_name}.sequence_auto_increment') from {schema_name}.winequality"
            )
            current_cursor().execute(f"DROP TABLE {schema_name}.winequality")
            current_cursor().execute(
                f"ALTER TABLE {schema_name}.winequality1 RENAME TO winequality"
            )

            if X is None:
                X = ["citric_acid", "residual_sugar", "alcohol"]
            if y is None:
                y = "quality"

            predictor_columns = ",".join(X)
            _X = [f'"{i}"' for i in X]

            if model_class == "XGBRegressor":
                train_sql = f"SELECT xgb_regressor('{schema_name}.{model_name}', '{schema_name}.winequality', '{y}', '{predictor_columns}' USING PARAMETERS exclude_columns='id', max_ntree={tree_param_map['max_ntree']}, max_depth={tree_param_map['max_depth']}, nbins={tree_param_map['nbins']}, split_proposal_method={tree_param_map['split_proposal_method']}, tol={tree_param_map['tol']}, learning_rate={tree_param_map['learning_rate']}, min_split_loss={tree_param_map['min_split_loss']}, weight_reg={tree_param_map['weight_reg']}, sample={tree_param_map['sample']}, col_sample_by_tree={tree_param_map['col_sample_by_tree']}, col_sample_by_node={tree_param_map['col_sample_by_node']}, seed=1, id_column='id')"
            else:
                train_sql = f"SELECT rf_regressor('{schema_name}.{model_name}', '{schema_name}.winequality', '{y}', '{predictor_columns}' USING PARAMETERS exclude_columns='id', ntree={tree_param_map['ntree']}, mtry={tree_param_map['mtry']}, max_breadth={tree_param_map['max_breadth']}, sampling_size={tree_param_map['sampling_size']}, max_depth={tree_param_map['max_depth']}, min_leaf_size={tree_param_map['min_leaf_size']}, nbins={tree_param_map['nbins']}, seed=1, id_column='id')"
            print(f"Tree Regressor Train SQL: {train_sql}")
            current_cursor().execute(train_sql)

            model.input_relation = f"{schema_name}.winequality"
            model.test_relation = model.input_relation
            model.X = _X
            model.y = f'"{y}"'
            model._compute_attributes()

            pred_vdf = model.predict(winequality_vpy_fun, name=f"{y}_pred")
            pred_prob_vdf = None
            current_cursor().execute(
                f"DROP SEQUENCE IF EXISTS {schema_name}.sequence_auto_increment"
            )
        elif model_class in [
            "AR",
            "MA",
            "ARMA",
            "ARIMA",
        ]:
            row_cnt = airline_vd_fun.describe()["count"][0]
            if model_class == "AR":
                p_val = kwargs.get("p", 3)
            elif model_class == "MA":
                p_val = kwargs.get("q", 1)
            elif model_class == "ARMA":
                p_val = kwargs.get("order", (2, 1))[0]
            elif model_class == "ARIMA":
                p_val = kwargs.get("order", (2, 1, 1))[0]
            else:
                p_val = 3

            if X is None:
                X = "date"
            if y is None:
                y = "passengers"

            model.fit(
                f"{schema_name}.airline",
                X,
                f"{y}",
            )
            pred_vdf = model.predict(
                airline_vd_fun,
                X,
                y,
                start=p_val,
                npredictions=kwargs.get("npredictions", row_cnt),
                output_estimated_ts=True,
            )
            pred_prob_vdf = None
        else:
            if X is None:
                X = ["citric_acid", "residual_sugar", "alcohol"]
            if y is None:
                y = "quality"

            model.fit(
                f"{schema_name}.winequality",
                X,
                f"{y}",
            )
            pred_vdf = model.predict(winequality_vpy_fun, name=f"{y}_pred")
            pred_prob_vdf = None

        vpy = namedtuple(
            "vertica_models",
            ["model", "pred_vdf", "pred_prob_vdf", "schema_name", "model_name"],
        )(model, pred_vdf, pred_prob_vdf, schema_name, model_name)

        return vpy

    yield _get_vpy_model


@pytest.fixture(name="get_py_model", scope="function")
def get_py_model_fixture(winequality_vpy_fun, titanic_vd_fun, airline_vd_fun):
    """
    getter function for python model
    """

    def _get_py_model(model_class, py_fit_intercept=None, py_tol=None, **kwargs):
        # sklearn
        if model_class in [
            "RandomForestClassifier",
            "DecisionTreeClassifier",
            "DummyTreeClassifier",
            "XGBClassifier",
        ]:
            # titanic_pdf = impute_dataset(titanic_vd_fun)
            # print(titanic_pdf.columns)
            titanic_pdf = titanic_vd_fun.to_pandas()
            titanic_pdf.dropna(subset=["age", "fare", "sex", "survived"], inplace=True)
            titanic_pdf.drop(
                titanic_pdf[
                    (titanic_pdf.name == "Kelly, Mr. James")
                    | (titanic_pdf.name == "Connolly, Miss. Kate")
                ].index,
                inplace=True,
            )

            titanic_pdf["sex"] = le.fit_transform(titanic_pdf["sex"])
            titanic_pdf["age"] = titanic_pdf["age"].astype(float)
            titanic_pdf["fare"] = titanic_pdf["fare"].astype(float)

            X = titanic_pdf[["age", "fare", "sex"]]
            y = titanic_pdf["survived"]
        elif model_class in [
            "AR",
            "MA",
            "ARMA",
            "ARIMA",
        ]:
            airline_pdf = airline_vd_fun.to_pandas()
            airline_pdf_ts = airline_pdf.set_index("date")

            X = airline_pdf[["date"]]
            y = airline_pdf["passengers"]
        elif model_class.upper() in ["TENSORFLOW", "TF"]:
            num_test_images = 500
            tftype = tf.float32
            nptype = np.float32

            (train_eval_data, train_eval_labels), (
                test_data,
                test_labels,
            ) = tf.keras.datasets.mnist.load_data()

            train_eval_labels = np.asarray(train_eval_labels, dtype=nptype)
            train_eval_labels = tf.keras.utils.to_categorical(train_eval_labels)

            test_labels = np.asarray(test_labels, dtype=nptype)
            test_labels = tf.keras.utils.to_categorical(test_labels)

            #  Split the training data into two parts, training and evaluation
            data_split = np.split(train_eval_data, [55000])
            labels_split = np.split(train_eval_labels, [55000])

            train_data = data_split[0]
            train_labels = labels_split[0]

            eval_data = data_split[1]
            eval_labels = labels_split[1]

            print("Size of train_data: ", train_data.shape[0])
            print("Size of eval_data: ", eval_data.shape[0])
            print("Size of test_data: ", test_data.shape[0])

            train_data = train_data.reshape((55000, 28, 28, 1))
            eval_data = eval_data.reshape((5000, 28, 28, 1))
            test_data = test_data.reshape((10000, 28, 28, 1))

            X = test_data[:num_test_images]
            y = test_labels[:num_test_images]
        else:
            winequality_pdf = winequality_vpy_fun.to_pandas()
            winequality_pdf["citric_acid"] = winequality_pdf["citric_acid"].astype(
                float
            )
            winequality_pdf["residual_sugar"] = winequality_pdf[
                "residual_sugar"
            ].astype(float)

            X = winequality_pdf[["citric_acid", "residual_sugar", "alcohol"]]
            y = winequality_pdf["quality"]

        if model_class in ["RandomForestRegressor", "RandomForestClassifier"]:
            model = getattr(skl_ensemble, model_class)(
                n_estimators=kwargs.get("n_estimators")
                if kwargs.get("n_estimators")
                else 10,
                max_features=kwargs.get("max_features")
                if kwargs.get("max_features")
                else 2,
                max_leaf_nodes=kwargs.get("max_leaf_nodes")
                if kwargs.get("max_leaf_nodes")
                else 10,
                max_samples=kwargs.get("sample") if kwargs.get("sample") else 0.632,
                max_depth=kwargs.get("max_depth") if kwargs.get("max_depth") else 10,
                min_samples_leaf=kwargs.get("min_samples_leaf")
                if kwargs.get("min_samples_leaf")
                else 1,
                random_state=1,
            )
        elif model_class in ["DecisionTreeRegressor", "DecisionTreeClassifier"]:
            model = getattr(skl_tree, model_class)(
                max_features=kwargs.get("max_features")
                if kwargs.get("max_features")
                else 2,
                max_leaf_nodes=kwargs.get("max_leaf_nodes")
                if kwargs.get("max_leaf_nodes")
                else 10,
                max_depth=kwargs.get("max_depth") if kwargs.get("max_depth") else 10,
                min_samples_leaf=kwargs.get("min_samples_leaf")
                if kwargs.get("min_samples_leaf")
                else 1,
                random_state=1,
            )
        elif model_class in ["XGBRegressor", "XGBClassifier"]:
            model = getattr(xgb, model_class)(
                n_estimators=kwargs.get("n_estimators")
                if kwargs.get("n_estimators")
                else 10,
                max_depth=kwargs.get("max_depth") if kwargs.get("max_depth") else 10,
                max_bin=kwargs.get("max_bin") if kwargs.get("max_bin") else 150,
                # split_proposal_method=kwargs.get("split_proposal_method") if kwargs.get("split_proposal_method") else 'global',
                # tol=kwargs.get("tol") if kwargs.get("tol") else 0.001,
                # learning_rate=kwargs.get("learning_rate")
                # if kwargs.get("learning_rate")
                # else 0.1,
                # gamma=kwargs.get("gamma") if kwargs.get("gamma") else 0.0,
                # reg_alpha=kwargs.get("reg_alpha") if kwargs.get("reg_alpha") else 0.0,
                # reg_lambda=kwargs.get("reg_lambda")
                # if kwargs.get("reg_lambda")
                # else 0.0,
                # subsample=kwargs.get("subsample") if kwargs.get("subsample") else 1.0,
                # colsample_bytree=kwargs.get("colsample_bytree")
                # if kwargs.get("colsample_bytree")
                # else 1.0,
                # colsample_bynode=kwargs.get("colsample_bynode")
                # if kwargs.get("colsample_bynode")
                # else 1.0,
                random_state=1,
                tree_method="exact",
            )
        elif model_class in ["DummyTreeRegressor"]:
            model = getattr(skl_dummy, "DummyRegressor")()
        elif model_class in ["DummyTreeClassifier"]:
            model = getattr(skl_dummy, "DummyClassifier")()
        elif model_class == "LinearSVR":
            model = getattr(skl_svm, model_class)(
                fit_intercept=py_fit_intercept if py_fit_intercept else True
            )
        elif model_class == "PoissonRegressor":
            model = getattr(skl_linear_model, model_class)(
                alpha=0.00005,
                fit_intercept=py_fit_intercept if py_fit_intercept else True,
                tol=py_tol if py_tol else 1e-06,
            )
        elif model_class in ["AR", "MA", "ARMA", "ARIMA"]:
            if model_class == "AR":
                order = (3, 0, 0)
                # model = AutoReg(
                #     airline_pdf_ts, lags=kwargs.get("p") if kwargs.get("p") else 3
                # ).fit()
            elif model_class == "MA":
                order = (0, 0, 1)
            elif model_class == "ARMA":
                order = (2, 0, 1)
            elif model_class == "ARIMA":
                order = (2, 1, 1)
            else:
                order = (3, 0, 0)

            model = ARIMA(
                airline_pdf_ts,
                order=kwargs.get("order") if kwargs.get("order") else order,
            ).fit()
            print(model.summary())
        elif model_class.upper() in ["TENSORFLOW", "TF"]:
            inputs = tf.keras.Input(shape=(28, 28, 1), name="image")
            x = tf.keras.layers.Conv2D(32, 5, activation="relu")(inputs)
            x = tf.keras.layers.MaxPooling2D(2)(x)
            x = tf.keras.layers.Conv2D(64, 5, activation="relu")(x)
            x = tf.keras.layers.MaxPooling2D(2)(x)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(10, activation="softmax", name="OUTPUT")(x)
            model = tf.keras.Model(inputs, x)
        else:
            model = getattr(skl_linear_model, model_class)(
                fit_intercept=py_fit_intercept if py_fit_intercept else True
            )

        if model_class in ["AR", "MA", "ARMA", "ARIMA"]:
            if model_class == "AR":
                p_val = kwargs.get("p", 3)
            elif model_class == "MA":
                p_val = kwargs.get("order", (0, 0, 1))[2]
            elif model_class == "ARMA":
                p_val = kwargs.get("order", (2, 0, 1))[0]
            elif model_class == "ARIMA":
                p_val = kwargs.get("order", (2, 1, 1))[0]
            else:
                p_val = 3

            npred = (
                kwargs.get("npredictions") + p_val
                if kwargs.get("npredictions")
                else None
            )
            pred = model.predict(start=p_val, end=npred, dynamic=False).values
            y = y[p_val : npred + 1 if npred else npred].values
        elif model_class.upper() in ["TENSORFLOW", "TF"]:
            batch_size = 100
            epochs = 5

            model.compile(
                loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"]
            )

            model.fit(
                train_data,
                train_labels,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
            )
            model.summary()
            loss, acc = model.evaluate(eval_data, eval_labels)
            print("Loss: ", loss, "  Accuracy: ", acc)
            pred = model.predict(X)
        else:
            print(f"Python Training Parameters: {model.get_params()}")
            model.fit(X, y)

            # num_params = len(skl_model.coef_) + 1
            pred = model.predict(X)

        if model_class in [
            "RandomForestClassifier",
            "DecisionTreeClassifier",
            "DummyTreeClassifier",
            "XGBClassifier",
        ]:
            pred_prob = model.predict_proba(X)
        else:
            pred_prob = None

        # statsmodels
        # add constant to predictor variables
        # X_sm = sm.add_constant(X)

        # fit linear regression model
        # sm_model = sm.OLS(y, X_sm).fit()

        py = namedtuple(
            "python_models", ["X", "y", "sm_model", "pred", "pred_prob", "model"]
        )(X, y, None, pred, pred_prob, model)

        return py

    return _get_py_model


@pytest.fixture(name="regression_metrics", scope="function")
def calculate_regression_metrics(get_py_model):
    """
    fixture to calculate python metrics
    """

    def _calculate_regression_metrics(model_class, model_obj=None, fit_intercept=True):
        if model_class in [
            "RandomForestRegressor",
            "RandomForestClassifier",
            "DecisionTreeRegressor",
            "DecisionTreeClassifier",
            "XGBRegressor",
            "DummyTreeRegressor",
            # "DummyTreeClassifier",
        ]:
            y, pred, model = model_obj.y, model_obj.pred, model_obj.model
            if model_class in ["RandomForestRegressor", "RandomForestClassifier"]:
                num_params = (
                    sum(tree.tree_.node_count for tree in model.estimators_) * 5
                    if model_class == model_class
                    else len(model.coef_) + 1
                )
                num_params = (
                    2  # setting it to 2 as per dev code(k+1 where, k=1), need to check
                )
            else:
                num_params = len(model.get_params()) + 1
        elif model_class in ["AR", "MA", "ARMA", "ARIMA"]:
            _, y, _, pred, _, model = get_py_model(model_class)
            num_params = len(list(model.params))
        else:
            _, y, _, pred, _, model = get_py_model(
                model_class, py_fit_intercept=fit_intercept
            )
            num_params = len(model.coef_) + 1

        regression_metrics_map = {}
        no_of_records = len(y)
        avg = sum(y) / no_of_records
        num_features = (
            3
            if model_class in ["DummyTreeRegressor"]
            else 1
            if model_class in ["AR", "MA", "ARMA", "ARIMA"]
            else (len(model.feature_names_in_))
        )
        # y_bar = y.mean()
        # ss_tot = ((y - y_bar) ** 2).sum()
        # ss_res = ((y - pred) ** 2).sum()

        regression_metrics_map["mse"] = getattr(skl_metrics, "mean_squared_error")(
            y, pred
        )
        regression_metrics_map["rmse"] = np.sqrt(regression_metrics_map["mse"])
        regression_metrics_map["ssr"] = sum(np.square(pred - avg))
        regression_metrics_map["sse"] = sum(np.square(y - pred))
        regression_metrics_map["dfr"] = num_features
        regression_metrics_map["dfe"] = no_of_records - num_features - 1
        regression_metrics_map["msr"] = (
            regression_metrics_map["ssr"] / regression_metrics_map["dfr"]
        )
        regression_metrics_map["_mse"] = (
            regression_metrics_map["sse"] / regression_metrics_map["dfe"]
        )
        regression_metrics_map["f"] = (
            regression_metrics_map["msr"] / regression_metrics_map["_mse"]
        )
        regression_metrics_map["p_value"] = f.sf(
            regression_metrics_map["f"], num_features, no_of_records
        )
        regression_metrics_map["mean_squared_log_error"] = (
            sum(
                pow(
                    (np.log10(pred + 1) - np.log10(y + 1)),
                    2,
                )
            )
            / no_of_records
        )
        regression_metrics_map["r2"] = regression_metrics_map[
            "r2_score"
        ] = skl_metrics.r2_score(y, pred)
        # regression_metrics_map["r2"] = regression_metrics_map["r2_score"] = 1 - (
        #     ss_res / ss_tot
        # )
        regression_metrics_map["rsquared_adj"] = 1 - (
            1 - regression_metrics_map["r2"]
        ) * (no_of_records - 1) / (no_of_records - num_features - 1)
        regression_metrics_map["aic"] = (
            no_of_records * math.log(regression_metrics_map["mse"]) + 2 * num_params
        )
        regression_metrics_map["bic"] = no_of_records * math.log(
            regression_metrics_map["mse"]
        ) + num_params * math.log(no_of_records)
        # regression_metrics_map["explained_variance_score"] = getattr(
        #     skl_metrics, "explained_variance_score"
        # )(y, pred)
        regression_metrics_map["explained_variance_score"] = 1 - np.var(
            (y - pred)
        ) / np.var(y)
        regression_metrics_map["max_error"] = getattr(skl_metrics, "max_error")(y, pred)
        regression_metrics_map["median_absolute_error"] = getattr(
            skl_metrics, "median_absolute_error"
        )(y, pred)
        regression_metrics_map["mean_absolute_error"] = getattr(
            skl_metrics, "mean_absolute_error"
        )(y, pred)
        regression_metrics_map["mean_squared_error"] = getattr(
            skl_metrics, "mean_squared_error"
        )(y, pred)
        regression_metrics_map[""] = ""

        return regression_metrics_map

    return _calculate_regression_metrics


@pytest.fixture(name="classification_metrics", scope="function")
def calculate_classification_metrics(get_py_model):
    """
    fixture to calculate python classification metrics
    """

    def _calculate_classification_metrics(model_class, model_obj=None):
        _model_obj = get_py_model(model_class)
        y, pred, pred_prob, _ = (
            model_obj.y.ravel() if model_obj else _model_obj.y.ravel(),
            model_obj.pred.ravel() if model_obj else _model_obj.pred.ravel(),
            model_obj.pred_prob[:, 1].ravel()
            if model_obj
            else _model_obj.pred_prob[:, 1].ravel(),
            model_obj.model if model_obj else _model_obj.model,
        )

        precision, recall, _ = skl_metrics.precision_recall_curve(
            y, pred_prob, pos_label=1
        )

        classification_metrics_map = {}
        # no_of_records = len(y)
        # avg = sum(y) / no_of_records
        # num_features = 3 if model_class in ["DummyTreeClassifier"] else len(model.feature_names_in_)

        classification_metrics_map["auc"] = skl_metrics.auc(recall, precision)
        classification_metrics_map["prc_auc"] = skl_metrics.auc(recall, precision)
        classification_metrics_map["accuracy_score"] = classification_metrics_map[
            "accuracy"
        ] = skl_metrics.accuracy_score(y, pred)
        classification_metrics_map["log_loss"] = -(
            (y * np.log10(pred + 1e-90)) + (1 - y) * np.log10(1 - pred + 1e-90)
        ).mean()
        classification_metrics_map["precision_score"] = classification_metrics_map[
            "precision"
        ] = python_metrics(y_true=y, y_pred=pred, metric_name="precision_score")
        classification_metrics_map["recall_score"] = classification_metrics_map[
            "recall"
        ] = python_metrics(y_true=y, y_pred=pred, metric_name="recall_score")
        classification_metrics_map["f1_score"] = classification_metrics_map[
            "f1"
        ] = python_metrics(y_true=y, y_pred=pred, metric_name="f1_score")
        classification_metrics_map["matthews_corrcoef"] = classification_metrics_map[
            "mcc"
        ] = skl_metrics.matthews_corrcoef(y_true=y, y_pred=pred)
        classification_metrics_map["informedness"] = python_metrics(
            y_true=y, y_pred=pred, metric_name="informedness"
        )
        classification_metrics_map["markedness"] = python_metrics(
            y_true=y, y_pred=pred, metric_name="markedness"
        )
        classification_metrics_map["critical_success_index"] = python_metrics(
            y_true=y, y_pred=pred, metric_name="critical_success_index"
        )
        classification_metrics_map["fpr"] = python_metrics(
            y_true=y, y_pred=pred, metric_name="fpr"
        )
        classification_metrics_map["tpr"] = python_metrics(
            y_true=y, y_pred=pred, metric_name="tpr"
        )

        return classification_metrics_map

    return _calculate_classification_metrics
