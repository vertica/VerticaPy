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

# VerticaPy
from verticapy import drop, set_option
from verticapy.learn.linear_model import LinearRegression
from verticapy.learn.linear_model import LogisticRegression

import verticapy.mlops.model_versioning as mv

set_option("print_info", False)


@pytest.fixture(scope="module")
def reg_model1(winequality_vpy):
    model = LinearRegression("reg_m1", solver="newton", max_iter=2)
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
    model = LinearRegression("reg_m2", solver="newton", max_iter=2)
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
    model = LinearRegression("reg_m3", solver="newton", max_iter=2)
    model.drop()

    model.fit(
        winequality_vpy,
        ["citric_acid", "residual_sugar", "alcohol"],
        "quality",
    )
    yield model
    model.drop()


@pytest.fixture(scope="module")
def bin_model1(winequality_vpy):
    model = LogisticRegression("bin_m1", solver="newton", max_iter=2, penalty=None)
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
    model = LogisticRegression("bin_m2", solver="newton", max_iter=2, penalty=None)
    model.drop()

    model.fit(
        winequality_vpy,
        ["citric_acid", "residual_sugar", "alcohol"],
        "good",
    )
    yield model
    model.drop()


####################################################
#                  IMPORTANT                       #
#                                                  #
# Due to the nature of the in-DB model versioning, #
# this test can pass only on a freshly created DB. #
####################################################
class TestModelVersioning:
    def test_register_models(self, reg_model1):
        reg_model1.register("regression_app1", raise_error=True)

        new_model = LinearRegression("new_m", solver="newton", max_iter=5)
        with pytest.raises(RuntimeError, match="Failed to register the model"):
            new_model.register("new_app", raise_error=True)

        assert not new_model.register("new_app")

    def test_list_models(self, bin_model1):
        with pytest.raises(ValueError):
            rm = mv.RegisteredModel("classification_app1")

        bin_model1.register("classification_app1")
        rm = mv.RegisteredModel("classification_app1")

        ts = rm.list_models()
        assert ts.values["registered_name"] == ["classification_app1"]
        assert ts.values["status"] == ["UNDER_REVIEW"]
        assert ts.values["schema_name"] == ["public"]
        assert ts.values["category"] == ["VERTICA_MODELS"]
        assert ts.values["model_name"] == ["bin_m1"]

    def test_reg_predict_change_status(self, winequality_vpy, reg_model2):
        reg_model2.register("regression_app2")
        rm = mv.RegisteredModel("regression_app2")

        data1 = winequality_vpy.copy()
        pred_vdf1 = rm.predict(
            data1,
            X=["citric_acid", "residual_sugar", "alcohol"],
            name="y_score",
            version=1,
        )
        assert pred_vdf1["y_score"].avg() == pytest.approx(5.8183777127)

        rm.change_status(version=1, new_status="staging")
        rm.change_status(version=1, new_status="production")
        data2 = winequality_vpy.copy()

        pred_vdf2 = rm.predict(
            data2, X=["citric_acid", "residual_sugar", "alcohol"], name="y_score"
        )

        assert pred_vdf2["y_score"].avg() == pytest.approx(5.8183777127)

    def test_bin_predict_change_status(self, winequality_vpy, bin_model2):
        bin_model2.register("classification_app2")
        rm = mv.RegisteredModel("classification_app2")

        data1 = winequality_vpy.copy()
        pred_vdf1 = rm.predict(
            data1,
            X=["citric_acid", "residual_sugar", "alcohol"],
            name="y_score",
            version=1,
        )
        assert pred_vdf1["y_score"].mode() == 0

        data2 = winequality_vpy.copy()
        pred_vdf2 = rm.predict_proba(
            data2,
            X=["citric_acid", "residual_sugar", "alcohol"],
            name="y_score",
            version=1,
        )
        assert pred_vdf2["y_score_1"].avg() == pytest.approx(0.2053748985)

        rm.change_status(version=1, new_status="staging")
        rm.change_status(version=1, new_status="production")

        data3 = winequality_vpy.copy()
        pred_vdf3 = rm.predict(
            data3, X=["citric_acid", "residual_sugar", "alcohol"], name="y_score"
        )
        assert pred_vdf3["y_score"].mode() == 0

        data4 = winequality_vpy.copy()
        pred_vdf4 = rm.predict_proba(
            data4, X=["citric_acid", "residual_sugar", "alcohol"], name="y_score"
        )
        assert pred_vdf4["y_score_1"].avg() == pytest.approx(0.2053748985)

    def test_list_status_history(self, reg_model3):
        reg_model3.register("regression_app3")
        rm = mv.RegisteredModel("regression_app3")

        ts = rm.list_status_history()

        assert ts.values["registered_name"][0] == "regression_app3"
        assert ts.values["new_status"][0] == "UNDER_REVIEW"
        assert ts.values["old_status"][0] == "UNREGISTERED"
