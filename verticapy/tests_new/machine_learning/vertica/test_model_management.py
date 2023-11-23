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
from itertools import chain
import sys
import subprocess
import numpy as np
import pytest
from verticapy.machine_learning.vertica import export_models, import_models, load_model
from verticapy.tests_new.machine_learning.vertica import rel_tolerance_map
import verticapy as vp

if sys.version_info < (3, 12):
    from verticapy.machine_learning.vertica.tensorflow.freeze_tf2_model import (
        freeze_model,
    )


def remove_model_dir(folder_path=""):
    """
    function to remove dir
    """
    print(f"Checking if model export path {folder_path} exists ..................")
    path_proc = subprocess.Popen(
        f"test -d {folder_path}",
        stdout=sys.stdout,
        stderr=sys.stderr,
        shell=True,
        universal_newlines=True,
    )
    _, _ = path_proc.communicate()
    if path_proc.returncode != 0:
        print(
            f"Model export output directory {folder_path} does not exists........................"
        )
    else:
        print(
            f"Model export output directory {folder_path} already exists. Hence, removing this folder."
        )
        rm_proc = subprocess.Popen(
            f"sudo rm -rf {folder_path}",
            stdout=sys.stdout,
            stderr=sys.stderr,
            shell=True,
            universal_newlines=True,
        )
        rm_code, _ = rm_proc.communicate()
        if rm_proc.returncode != 0:
            print(
                f"Error in removing {folder_path} Error code: {rm_proc.returncode}, {rm_code}"
            )
        else:
            print(f"Model export output directory {folder_path} removed successfully")


def _export(model_obj, category):
    """
    function to export
    """
    export_models(
        name=f"{model_obj.schema_name}.{model_obj.model_name}",
        path=f"/tmp/{model_obj.schema_name}",
        kind=category,
    )


def _import(vpy_model_obj, py_model_obj, category, schema_name):
    """
    function to import
    """
    if category in ["tf", "tensorflow"]:
        print(f"Saving frozen model to /tmp/{schema_name}/tf_frozen_model")
        freeze_model(py_model_obj.model, f"/tmp/{schema_name}/tf_frozen_model", "0")
        print("freeze_tf2_model code execution completed......................")

        import_models(
            path=f"/tmp/{schema_name}/tf_frozen_model",
            schema=schema_name,
            kind=category,
        )
    else:
        import_models(
            path=f"/tmp/{vpy_model_obj.schema_name}/{vpy_model_obj.model_name}",
            schema=vpy_model_obj.schema_name,
            kind=category,
        )


@pytest.mark.parametrize("category", ["pmml", "vertica", "vertica_models", None])
@pytest.mark.parametrize(
    "model_class",
    [
        "RandomForestRegressor",
        "RandomForestClassifier",
        "DecisionTreeRegressor",
        "DecisionTreeClassifier",
        "DummyTreeRegressor",
        # "DummyTreeClassifier", # fail
        "XGBRegressor",
        "XGBClassifier",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "LinearRegression",
        "LinearSVR",  # PMML is not yet supported
        "PoissonRegressor",  # PMML is not yet supported
        # "AR", # PMML is not yet supported
        # "MA", # PMML is not yet supported
        # "ARMA", # PMML is not yet supported
        # "ARIMA", # PMML is not yet supported
    ],
)
class TestModelManagement:
    """
    test class for import/export models
    """

    def test_export_models(self, get_vpy_model, model_class, category):
        """
        test function - export_models
        """
        if (
            model_class
            in ["LinearSVR", "PoissonRegressor", "AR", "MA", "ARMA", "ARIMA"]
            and category == "pmml"
        ):
            pytest.skip(f"PMML is not yet supported for {model_class}")

        vpy_model_obj = get_vpy_model(model_class)
        remove_model_dir(folder_path=f"/tmp/{vpy_model_obj.schema_name}")
        _export(vpy_model_obj, category)
        remove_model_dir(folder_path=f"/tmp/{vpy_model_obj.schema_name}")

    def test_import_models(self, get_vpy_model, model_class, category):
        """
        test function - import_models
        """
        if (
            model_class
            in ["LinearSVR", "PoissonRegressor", "AR", "MA", "ARMA", "ARIMA"]
            and category == "pmml"
        ):
            pytest.skip(f"PMML is not yet supported for {model_class}")

        vpy_model_obj = get_vpy_model(model_class)

        # model export
        remove_model_dir(folder_path=f"/tmp/{vpy_model_obj.schema_name}")
        _export(vpy_model_obj, category)

        # drop model
        vpy_model_obj.model.drop()

        # import model
        _import(vpy_model_obj, None, category, None)

        remove_model_dir(folder_path=f"/tmp/{vpy_model_obj.schema_name}")

    def test_load_model(
        self,
        get_vpy_model,
        get_py_model,
        winequality_vpy_fun,
        titanic_vd_fun,
        model_class,
        category,
    ):
        """
        test function - load_model
        """
        if (
            model_class
            in ["LinearSVR", "PoissonRegressor", "AR", "MA", "ARMA", "ARIMA"]
            and category == "pmml"
        ):
            pytest.skip(f"PMML is not yet supported for {model_class}")

        py_model_obj = get_py_model(model_class)

        # model export
        vpy_model_obj = get_vpy_model(model_class)

        remove_model_dir(folder_path=f"/tmp/{vpy_model_obj.schema_name}")
        _export(vpy_model_obj, category)
        # drop model
        vpy_model_obj.model.drop()

        # import model
        _import(vpy_model_obj, None, category, None)
        model = load_model(
            name=f"{vpy_model_obj.schema_name}.{vpy_model_obj.model_name}"
        )

        if (
            model_class
            in [
                "RandomForestRegressor",
                "DecisionTreeRegressor",
                "DummyTreeRegressor",
                "XGBRegressor",
            ]
            and category == "pmml"
        ):
            pred_vdf = model.predict(
                winequality_vpy_fun,
                ["citric_acid", "residual_sugar", "alcohol"],
                "prediction",
            )
            vpy_res = np.mean(
                [i[0]["predicted_value"] for i in pred_vdf[["prediction"]].to_list()]
            )
            py_res = py_model_obj.pred.mean()
        elif model_class in [
            "RandomForestClassifier",
            "DecisionTreeClassifier",
            "DummyTreeClassifier",
            "XGBClassifier",
        ]:
            pred_vdf = model.predict(
                titanic_vd_fun, ["age", "fare", "sex"], "prediction"
            )
            if category == "pmml":
                vpy_res = np.sum(
                    [
                        i[0]["predicted_value"]
                        for i in pred_vdf[["prediction"]].to_list()
                    ],
                    dtype=int,
                )
            else:
                vpy_res = np.mean(
                    list(
                        chain(*np.array(pred_vdf[["prediction"]].to_list(), dtype=int))
                    )
                )
            py_res = py_model_obj.pred.sum()
        else:
            pred_vdf = model.predict(
                winequality_vpy_fun,
                ["citric_acid", "residual_sugar", "alcohol"],
                "prediction",
            )
            vpy_res = np.mean(
                list(chain(*np.array(pred_vdf[["prediction"]].to_list(), dtype=float)))
            )
            py_res = py_model_obj.pred.mean()

        remove_model_dir(folder_path=f"/tmp/{vpy_model_obj.schema_name}")

        assert vpy_res == pytest.approx(py_res, rel=rel_tolerance_map[model_class])


@pytest.mark.parametrize(
    "category",
    [
        "tensorflow",
        "tf",
    ],
)
@pytest.mark.parametrize(
    "model_class",
    [
        "TENSORFLOW",
    ],
)
@pytest.mark.skipif(
    sys.version_info > (3, 11, 11), reason="requires python3.11 or lower"
)
class TestModelManagementTF:
    """
    test class for Tensorflow import/export
    """

    @pytest.mark.skipif(
        sys.version_info > (3, 11, 11), reason="keras is not supported for Python 3.12"
    )
    def test_tf_export(self, get_py_model, schema_loader, category, model_class):
        """
        test function - test_tf_import
        """
        vp.drop(name=f"{schema_loader}.tf_frozen_model")
        remove_model_dir(folder_path=f"/tmp/{schema_loader}")
        py_model_obj = get_py_model(model_class)
        _import(None, py_model_obj, category, schema_loader)
        remove_model_dir(folder_path=f"/tmp/{schema_loader}")
        tf_model_obj = namedtuple(
            "tf_model",
            ["schema_name", "model_name"],
        )(schema_loader, "tf_frozen_model")
        _export(tf_model_obj, category)
        remove_model_dir(folder_path=f"/tmp/{schema_loader}")

    @pytest.mark.skipif(
        sys.version_info > (3, 11, 11), reason="keras is not supported for Python 3.12"
    )
    def test_tf_import(self, get_py_model, schema_loader, category, model_class):
        """
        test function - test_tf_import
        """
        vp.drop(name=f"{schema_loader}.tf_frozen_model")
        py_model_obj = get_py_model(model_class)
        _import(None, py_model_obj, category, schema_loader)
        remove_model_dir(folder_path=f"/tmp/{schema_loader}")

    @pytest.mark.skip(reason="it needs more investigation")
    def test_tf_load_model(self, get_py_model, schema_loader, model_class, category):
        """
        test function - tf_load_model
        """
        vp.drop(name=f"{schema_loader}.tf_frozen_model")
        remove_model_dir(folder_path=f"/tmp/{schema_loader}")

        py_model_obj = get_py_model(model_class)

        _import(None, py_model_obj, category, schema_loader)
        model = load_model(name=f"{schema_loader}.tf_frozen_model")

        reshaped_2d_test_data = []
        for data in range(py_model_obj.X.shape[0]):
            reshaped_2d_test_data.append(np.squeeze(py_model_obj.X[data]).flatten())

        # explicitly adding column name as it gives error without it in prediction
        vdf = vp.vDataFrame(
            reshaped_2d_test_data,
            usecols=[f"x{i}" for i in range(len(reshaped_2d_test_data[0]))],
        )

        pred_raw_vdf = model.predict(
            vdf, vdf.get_columns(), "prediction", inplace=False
        )
        pred_vdf = pred_raw_vdf.select(columns=pred_raw_vdf.get_columns()[-10:])
        vpy_pred = pred_vdf.to_numpy()

        vpy_match_cnt = py_match_cnt = 0
        np.set_printoptions(threshold=sys.maxsize)

        for i in range(py_model_obj.y.shape[0]):
            # vpy
            if np.argmax(py_model_obj.y[i]) == np.argmax(vpy_pred[i]):
                vpy_match_cnt += 1

            # py
            if np.argmax(py_model_obj.y[i]) == np.argmax(py_model_obj.pred[i]):
                py_match_cnt += 1

        vpy_res = (vpy_match_cnt / py_model_obj.y.shape[0]) * 100
        print(f"vpy_score_pct: {vpy_res}")

        py_res = (py_match_cnt / py_model_obj.y.shape[0]) * 100
        print(f"py_score_pct: {py_res}")

        remove_model_dir(folder_path=f"/tmp/{schema_loader}")
        vp.drop(name=f"{schema_loader}.tf_frozen_model")

        assert vpy_res == pytest.approx(py_res, rel=rel_tolerance_map[model_class])
