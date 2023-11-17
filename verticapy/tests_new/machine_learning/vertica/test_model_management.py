"""
Copyright  (c)  2018-2023 Open Text  or  one  of its
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
from itertools import chain
import sys
import subprocess
import numpy as np
import git
import pytest
from verticapy.machine_learning.vertica import export_models, import_models, load_model
from verticapy.tests_new.machine_learning.vertica import (
    REL_TOLERANCE,
    ABS_TOLERANCE,
    rel_tolerance_map,
)
from verticapy.tests_new.machine_learning.vertica import freeze_tf2_model
from verticapy import vDataFrame


def remove_model_dir(folder_path=""):
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
            f"rm -rf {folder_path}",
            stdout=sys.stdout,
            stderr=sys.stderr,
            shell=True,
            universal_newlines=True,
        )
        rm_code, rm_errs = rm_proc.communicate()
        if rm_proc.returncode != 0:
            print(
                f"Error in removing {folder_path} Error code: {rm_proc.returncode}, {rm_code}"
            )
        else:
            print(f"Model export output directory {folder_path} removed successfully")


@pytest.mark.parametrize(
    "model_class",
    [
        # "RandomForestRegressor",
        # "RandomForestClassifier",
        # "DecisionTreeRegressor",
        # "DecisionTreeClassifier",
        # "DummyTreeRegressor",
        # # "DummyTreeClassifier", # fail
        # "XGBRegressor",
        # "XGBClassifier",
        # "Ridge",
        # "Lasso",
        # "ElasticNet",
        # "LinearRegression",
        "TENSORFLOW",

        # "LinearSVR", # PMML is not yet supported
        # "PoissonRegressor", # PMML is not yet supported
        # "AR", # PMML is not yet supported
        # "MA", # PMML is not yet supported
        # "ARMA", # PMML is not yet supported
        # "ARIMA", # PMML is not yet supported
    ],
)
class TestBaseModelMethods:
    """
    test class for import/export models
    """

    @pytest.mark.parametrize(
        "category", [
            "pmml",
            "vertica",
            "vertica_models",
            # "tensorflow",
            # "tf",
            None
        ]
    )
    def test_export_models(self, get_vpy_model, model_class, category):
        """
        test function - export_models
        """

        vpy_model_obj = get_vpy_model(model_class)
        remove_model_dir(folder_path=f"/tmp/{vpy_model_obj.model_name}")
        export_models(
            name=f"{vpy_model_obj.schema_name}.{vpy_model_obj.model_name}",
            path="/tmp/",
            kind=category,
        )

    @pytest.mark.parametrize(
        "category", [
            "pmml",
            "vertica",
            "vertica_models",
            # "tensorflow",
            # "tf",
            None
        ]
    )
    def test_import_models(self, get_vpy_model, model_class, category):
        """
        test function - import_models
        """

        # model export
        vpy_model_obj = get_vpy_model(model_class)
        remove_model_dir(folder_path=f"/tmp/{vpy_model_obj.model_name}")
        export_models(
            name=f"{vpy_model_obj.schema_name}.{vpy_model_obj.model_name}",
            path="/tmp/",
            kind=category,
        )

        # drop model
        vpy_model_obj.model.drop()

        # import model
        import_models(
            path=f"/tmp/{vpy_model_obj.model_name}",
            schema=vpy_model_obj.schema_name,
            kind=category,
        )

    @pytest.mark.parametrize(
        "category", [
            # "pmml",
            # "vertica",
            # "vertica_models",
            "tensorflow",
            # # "tf",
            # None
        ]
    )
    def test_load_model(self, get_vpy_model, get_py_model, winequality_vpy_fun, titanic_vd_fun, schema_loader,
                        model_class, category):
        """
        test function - load_model
        """
        py_model_obj = get_py_model(model_class)

        if model_class == "TENSORFLOW":
            print(f'Saving frozen model to /tmp/tf_frozen_model')
            freeze_tf2_model.freeze_model(py_model_obj.model, '/tmp/tf_frozen_model', '0')
            print('freeze_tf2_model code execution completed......................')

            import_models(
                path=f"/tmp/tf_frozen_model",
                schema=schema_loader,
                kind=category,
            )
            model = load_model(name=f"{schema_loader}.tf_frozen_model")
        else:
            # model export
            vpy_model_obj = get_vpy_model(model_class)

            remove_model_dir(folder_path=f"/tmp/{vpy_model_obj.model_name}")
            export_models(
                name=f"{vpy_model_obj.schema_name}.{vpy_model_obj.model_name}",
                path="/tmp/",
                kind=category,
            )
            # drop model
            vpy_model_obj.model.drop()

            # import model
            import_models(
                path=f"/tmp/{vpy_model_obj.model_name}",
                schema=vpy_model_obj.schema_name,
                kind=category,
            )
            model = load_model(name=f"{vpy_model_obj.schema_name}.{vpy_model_obj.model_name}")

        if model_class in [
            "RandomForestRegressor",
            "DecisionTreeRegressor",
            "DummyTreeRegressor",
            "XGBRegressor",
        ]:
            pred_vdf = model.predict(winequality_vpy_fun, ["citric_acid", "residual_sugar", "alcohol"], "prediction")
            vpy_res = np.mean([i[0]['predicted_value'] for i in pred_vdf[["prediction"]].to_list()])
            py_res = py_model_obj.pred.mean()
        elif model_class in [
            "RandomForestClassifier",
            "DecisionTreeClassifier",
            "DummyTreeClassifier",
            "XGBClassifier",
        ]:
            pred_vdf = model.predict(titanic_vd_fun, ["age", "fare", "sex"], "prediction")
            vpy_res = np.sum([i[0]['predicted_value'] for i in pred_vdf[["prediction"]].to_list()], dtype=int)
            py_res = py_model_obj.pred.sum()
        elif model_class == "TENSORFLOW":
            reshaped_2d_test_data = []
            for data in range(py_model_obj.X.shape[0]):
                reshaped_2d_test_data.append(np.squeeze(py_model_obj.X[data]).flatten())

            vdf = vDataFrame(reshaped_2d_test_data)
            cols = vdf.get_columns()

            pred_vdf = model.predict(vdf, cols, "prediction", inplace=False)
            vpy_pred = pred_vdf.to_numpy()

            vpy_match_cnt = py_match_cnt = 0

            print(vpy_pred.argmax(axis=0))
            print(py_model_obj.pred.argmax(axis=0))

            for i in range(py_model_obj.pred.shape[0]):
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

        else:
            pred_vdf = model.predict(winequality_vpy_fun, ["citric_acid", "residual_sugar", "alcohol"], "prediction")
            vpy_res = np.mean(list(chain(*np.array(pred_vdf[["prediction"]].to_list(), dtype=float))))
            py_res = py_model_obj.pred.mean()

        assert vpy_res == pytest.approx(py_res, rel=rel_tolerance_map[model_class])
