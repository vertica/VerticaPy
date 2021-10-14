# (c) Copyright [2018-2021] Micro Focus or one of its affiliates.
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

import pytest, warnings, os, verticapy
from verticapy.learn.memmodel import *
from verticapy import drop

@pytest.fixture(scope="module")
def titanic_vd(base):
    from verticapy.datasets import load_titanic

    titanic = load_titanic(cursor=base.cursor)
    yield titanic
    with warnings.catch_warnings(record=True) as w:
        drop(
            name="public.titanic", cursor=base.cursor,
        )

class Test_memModel:
    def test_LinearRegression(self,):
        model = memModel("LinearRegression", {"coefficients": [0.5, 0.6,], 
                                              "intercept": 0.8})
        assert model.predict([[0.4, 0.5]])[0] == pytest.approx(1.3)
        assert model.predict_sql([0.4, 0.5]) == '0.8 + 0.5 * 0.4 + 0.6 * 0.5'
        attributes = model.get_attributes()
        assert attributes["coefficients"][0] == 0.5
        assert attributes["coefficients"][1] == 0.6
        assert attributes["intercept"] == 0.8
        model.set_attributes({"coefficients": [0.4, 0.5]})
        attributes = model.get_attributes()
        assert attributes["coefficients"][0] == 0.4
        assert attributes["coefficients"][1] == 0.5
        assert attributes["intercept"] == 0.8
        assert model.model_type_ == "LinearRegression"

    def test_LinearSVR(self,):
        model = memModel("LinearSVR", {"coefficients": [0.5, 0.6,], 
                                       "intercept": 0.8})
        assert model.predict([[0.4, 0.5]])[0] == pytest.approx(1.3)
        assert model.predict_sql([0.4, 0.5]) == '0.8 + 0.5 * 0.4 + 0.6 * 0.5'
        attributes = model.get_attributes()
        assert attributes["coefficients"][0] == 0.5
        assert attributes["coefficients"][1] == 0.6
        assert attributes["intercept"] == 0.8
        model.set_attributes({"coefficients": [0.4, 0.5]})
        attributes = model.get_attributes()
        assert attributes["coefficients"][0] == 0.4
        assert attributes["coefficients"][1] == 0.5
        assert attributes["intercept"] == 0.8
        assert model.model_type_ == "LinearSVR"

    def test_LogisticRegression(self,):
        model = memModel("LogisticRegression", {"coefficients": [0.5, 0.6,], 
                                                "intercept": 0.8})
        assert model.predict([[0.4, 0.5]])[0] == pytest.approx(1)
        assert model.predict_sql([0.4, 0.5]) == '((1 / (1 + EXP(- (0.8 + 0.5 * 0.4 + 0.6 * 0.5)))) > 0.5)::int'
        predict_proba_val = model.predict_proba([[0.4, 0.5]])
        assert predict_proba_val[0][0] == pytest.approx(0.21416502)
        assert predict_proba_val[0][1] == pytest.approx(0.78583498)
        predict_proba_val_sql = model.predict_proba_sql([0.4, 0.5])
        assert predict_proba_val_sql[0] == '1 - (1 / (1 + EXP(- (0.8 + 0.5 * 0.4 + 0.6 * 0.5))))'
        assert predict_proba_val_sql[1] == '1 / (1 + EXP(- (0.8 + 0.5 * 0.4 + 0.6 * 0.5)))'
        attributes = model.get_attributes()
        assert attributes["coefficients"][0] == 0.5
        assert attributes["coefficients"][1] == 0.6
        assert attributes["intercept"] == 0.8
        model.set_attributes({"coefficients": [0.4, 0.5]})
        attributes = model.get_attributes()
        assert attributes["coefficients"][0] == 0.4
        assert attributes["coefficients"][1] == 0.5
        assert attributes["intercept"] == 0.8
        assert model.model_type_ == "LogisticRegression"

    def test_LinearSVC(self,):
        model = memModel("LinearSVC", {"coefficients": [0.5, 0.6,], 
                                       "intercept": 0.8})
        assert model.predict([[0.4, 0.5]])[0] == pytest.approx(1)
        assert model.predict_sql([0.4, 0.5]) == '((1 / (1 + EXP(- (0.8 + 0.5 * 0.4 + 0.6 * 0.5)))) > 0.5)::int'
        predict_proba_val = model.predict_proba([[0.4, 0.5]])
        assert predict_proba_val[0][0] == pytest.approx(0.21416502)
        assert predict_proba_val[0][1] == pytest.approx(0.78583498)
        predict_proba_val_sql = model.predict_proba_sql([0.4, 0.5])
        assert predict_proba_val_sql[0] == '1 - (1 / (1 + EXP(- (0.8 + 0.5 * 0.4 + 0.6 * 0.5))))'
        assert predict_proba_val_sql[1] == '1 / (1 + EXP(- (0.8 + 0.5 * 0.4 + 0.6 * 0.5)))'
        attributes = model.get_attributes()
        assert attributes["coefficients"][0] == 0.5
        assert attributes["coefficients"][1] == 0.6
        assert attributes["intercept"] == 0.8
        model.set_attributes({"coefficients": [0.4, 0.5]})
        attributes = model.get_attributes()
        assert attributes["coefficients"][0] == 0.4
        assert attributes["coefficients"][1] == 0.5
        assert attributes["intercept"] == 0.8
        assert model.model_type_ == "LinearSVC"

    def test_PCA(self,):
        model = memModel("PCA", {"principal_components": [[0.4, 0.5], [0.3, 0.2],],
                                 "mean": [0.1, 0.3]})
        transformation = model.transform([[0.4, 0.5]])
        assert transformation[0][0] == pytest.approx(0.18)
        assert transformation[0][1] == pytest.approx(0.19)
        transformation_sql = model.transform_sql([0.4, 0.5])
        assert transformation_sql[0] == '(0.4 - 0.1) * 0.4 + (0.5 - 0.3) * 0.3'
        assert transformation_sql[1] == '(0.4 - 0.1) * 0.5 + (0.5 - 0.3) * 0.2'
        attributes = model.get_attributes()
        assert attributes["principal_components"][0][0] == 0.4
        assert attributes["principal_components"][0][1] == 0.5
        assert attributes["principal_components"][1][0] == 0.3
        assert attributes["principal_components"][1][1] == 0.2
        assert attributes["mean"][0] == 0.1
        assert attributes["mean"][1] == 0.3
        model.set_attributes({"principal_components": [[0.1, 0.2], [0.7, 0.8],], "mean": [0.9, 0.8]})
        attributes = model.get_attributes()
        assert attributes["principal_components"][0][0] == 0.1
        assert attributes["principal_components"][0][1] == 0.2
        assert attributes["principal_components"][1][0] == 0.7
        assert attributes["principal_components"][1][1] == 0.8
        model = model.rotate()
        attributes = model.get_attributes()
        assert attributes["principal_components"][0][0] == pytest.approx(0.05887149)
        assert attributes["principal_components"][0][1] == pytest.approx(0.21571775)
        assert attributes["principal_components"][1][0] == pytest.approx(0.01194755)
        assert attributes["principal_components"][1][1] == pytest.approx(1.06294744)
        assert attributes["mean"][0] == 0.9
        assert attributes["mean"][1] == 0.8
        assert model.model_type_ == "PCA"

    def test_SVD(self,):
        model = memModel("SVD", {"vectors": [[0.4, 0.5], [0.3, 0.2],],
                                 "values": [0.1, 0.3]})
        transformation = model.transform([[0.4, 0.5]])
        assert transformation[0][0] == pytest.approx(3.1)
        assert transformation[0][1] == pytest.approx(1.)
        transformation_sql = model.transform_sql([0.4, 0.5])
        assert transformation_sql[0] == '0.4 * 0.4 / 0.1 + 0.5 * 0.3 / 0.1'
        assert transformation_sql[1] == '0.4 * 0.5 / 0.3 + 0.5 * 0.2 / 0.3'
        attributes = model.get_attributes()
        assert attributes["vectors"][0][0] == 0.4
        assert attributes["vectors"][0][1] == 0.5
        assert attributes["vectors"][1][0] == 0.3
        assert attributes["vectors"][1][1] == 0.2
        assert attributes["values"][0] == 0.1
        assert attributes["values"][1] == 0.3
        model.set_attributes({"vectors": [[0.1, 0.2], [0.7, 0.8],], "values": [0.9, 0.8]})
        attributes = model.get_attributes()
        assert attributes["vectors"][0][0] == 0.1
        assert attributes["vectors"][0][1] == 0.2
        assert attributes["vectors"][1][0] == 0.7
        assert attributes["vectors"][1][1] == 0.8
        assert attributes["values"][0] == 0.9
        assert attributes["values"][1] == 0.8
        assert model.model_type_ == "SVD"

    def test_Normalizer(self,):
        model = memModel("Normalizer", {"values": [(0.4, 0.5), (0.3, 0.2),],
                                        "method": "minmax"})
        transformation = model.transform([[0.4, 0.5]])
        assert transformation[0][0] == pytest.approx(0.)
        assert transformation[0][1] == pytest.approx(-2.)
        transformation_sql = model.transform_sql([0.4, 0.5])
        assert transformation_sql[0] == '(0.4 - 0.4) / 0.09999999999999998'
        assert transformation_sql[1] == '(0.5 - 0.3) / -0.09999999999999998'
        attributes = model.get_attributes()
        assert attributes["values"][0][0] == 0.4
        assert attributes["values"][0][1] == 0.5
        assert attributes["values"][1][0] == 0.3
        assert attributes["values"][1][1] == 0.2
        assert attributes["method"] == "minmax"
        model.set_attributes({"method": "zscore"})
        transformation = model.transform([[0.4, 0.5]])
        assert transformation[0][0] == pytest.approx(0.)
        assert transformation[0][1] == pytest.approx(1.)
        transformation_sql = model.transform_sql([0.4, 0.5])
        assert transformation_sql[0] == '(0.4 - 0.4) / 0.5'
        assert transformation_sql[1] == '(0.5 - 0.3) / 0.2'
        attributes = model.get_attributes()
        assert attributes["values"][0][0] == 0.4
        assert attributes["values"][0][1] == 0.5
        assert attributes["values"][1][0] == 0.3
        assert attributes["values"][1][1] == 0.2
        assert attributes["method"] == "zscore"
        model.set_attributes({"method": "robust_zscore"})
        transformation = model.transform([[0.4, 0.5]])
        assert transformation[0][0] == pytest.approx(0.)
        assert transformation[0][1] == pytest.approx(1.)
        transformation_sql = model.transform_sql([0.4, 0.5])
        assert transformation_sql[0] == '(0.4 - 0.4) / 0.5'
        assert transformation_sql[1] == '(0.5 - 0.3) / 0.2'
        attributes = model.get_attributes()
        assert attributes["values"][0][0] == 0.4
        assert attributes["values"][0][1] == 0.5
        assert attributes["values"][1][0] == 0.3
        assert attributes["values"][1][1] == 0.2
        assert attributes["method"] == "robust_zscore"
        model.set_attributes({"values": [(0.5, 0.6), (0.4, 0.3),]})
        attributes = model.get_attributes()
        assert attributes["values"][0][0] == 0.5
        assert attributes["values"][0][1] == 0.6
        assert attributes["values"][1][0] == 0.4
        assert attributes["values"][1][1] == 0.3
        assert model.model_type_ == "Normalizer"

    def test_OneHotEncoder(self,):
        model = memModel("OneHotEncoder", {"categories": [['male', 'female'], [1, 2, 3],],
                                           "drop_first": False,
                                           "column_naming": None})
        transformation = model.transform([['male', 1],
                                          ['female', 3],])
        assert transformation[0][0] == pytest.approx(1)
        assert transformation[0][1] == pytest.approx(0)
        assert transformation[0][2] == pytest.approx(1)
        assert transformation[0][3] == pytest.approx(0)
        assert transformation[0][4] == pytest.approx(0)
        assert transformation[1][0] == pytest.approx(0)
        assert transformation[1][1] == pytest.approx(1)
        assert transformation[1][2] == pytest.approx(0)
        assert transformation[1][3] == pytest.approx(0)
        assert transformation[1][4] == pytest.approx(1)
        transformation_sql = model.transform_sql(["'male'", 1])
        assert transformation_sql[0][0] == "(CASE WHEN 'male' = 'male' THEN 1 ELSE 0 END)"
        assert transformation_sql[0][1] == "(CASE WHEN 'male' = 'female' THEN 1 ELSE 0 END)"
        assert transformation_sql[1][0] == '(CASE WHEN 1 = 1 THEN 1 ELSE 0 END)'
        assert transformation_sql[1][1] == '(CASE WHEN 1 = 2 THEN 1 ELSE 0 END)'
        assert transformation_sql[1][2] == '(CASE WHEN 1 = 3 THEN 1 ELSE 0 END)'
        model.set_attributes({"drop_first": True})
        transformation = model.transform([['male', 1],
                                          ['female', 3],])
        assert transformation[0][0] == pytest.approx(0)
        assert transformation[0][1] == pytest.approx(0)
        assert transformation[0][2] == pytest.approx(0)
        assert transformation[1][0] == pytest.approx(1)
        assert transformation[1][1] == pytest.approx(0)
        assert transformation[1][2] == pytest.approx(1)
        transformation_sql = model.transform_sql(["'male'", 1])
        assert transformation_sql[0][0] == "(CASE WHEN 'male' = 'female' THEN 1 ELSE 0 END)"
        assert transformation_sql[1][0] == '(CASE WHEN 1 = 2 THEN 1 ELSE 0 END)'
        assert transformation_sql[1][1] == '(CASE WHEN 1 = 3 THEN 1 ELSE 0 END)'
        model.set_attributes({"column_naming": "indices"})
        transformation_sql = model.transform_sql(['sex', 'pclass'])
        assert transformation_sql[0][0] == "(CASE WHEN sex = 'female' THEN 1 ELSE 0 END) AS \"sex_1\""
        assert transformation_sql[1][0] == '(CASE WHEN pclass = 2 THEN 1 ELSE 0 END) AS \"pclass_1\"'
        assert transformation_sql[1][1] == '(CASE WHEN pclass = 3 THEN 1 ELSE 0 END) AS \"pclass_2\"'
        model.set_attributes({"column_naming": "values"})
        transformation_sql = model.transform_sql(['sex', 'pclass'])
        assert transformation_sql[0][0] == "(CASE WHEN sex = 'female' THEN 1 ELSE 0 END) AS \"sex_female\""
        assert transformation_sql[1][0] == '(CASE WHEN pclass = 2 THEN 1 ELSE 0 END) AS \"pclass_2\"'
        assert transformation_sql[1][1] == '(CASE WHEN pclass = 3 THEN 1 ELSE 0 END) AS \"pclass_3\"'
        assert model.model_type_ == "OneHotEncoder"

    def test_KMeans(self,):
        model = memModel("KMeans", {"clusters": [[0.5, 0.6,], [1, 2,], [100, 200,]], 
                         "p": 2})
        assert model.predict([[0.2, 0.3]])[0] == 0
        assert model.predict([[2, 2]])[0] == 1
        assert model.predict([[100, 201]])[0] == 2
        assert model.predict_sql([0.4, 0.5]) == 'CASE WHEN 0.4 IS NULL OR 0.5 IS NULL THEN NULL WHEN POWER(POWER(0.4 - 100.0, 2) + POWER(0.5 - 200.0, 2), 1 / 2) <= POWER(POWER(0.4 - 0.5, 2) + POWER(0.5 - 0.6, 2), 1 / 2) AND POWER(POWER(0.4 - 100.0, 2) + POWER(0.5 - 200.0, 2), 1 / 2) <= POWER(POWER(0.4 - 1.0, 2) + POWER(0.5 - 2.0, 2), 1 / 2) THEN 2 WHEN POWER(POWER(0.4 - 1.0, 2) + POWER(0.5 - 2.0, 2), 1 / 2) <= POWER(POWER(0.4 - 0.5, 2) + POWER(0.5 - 0.6, 2), 1 / 2) THEN 1 ELSE 0 END'
        predict_proba_val = model.predict_proba([[0.2, 0.3]])
        assert predict_proba_val[0][0] == pytest.approx(0.81452236)
        assert predict_proba_val[0][1] == pytest.approx(0.18392972)
        assert predict_proba_val[0][2] == pytest.approx(0.001547924158153152)
        predict_proba_sql = model.predict_proba_sql([0.2, 0.3])
        assert predict_proba_sql[0] == '(CASE WHEN POWER(POWER(0.2 - 0.5, 2) + POWER(0.3 - 0.6, 2), 1 / 2) = 0 THEN 1.0 ELSE 1 / (POWER(POWER(0.2 - 0.5, 2) + POWER(0.3 - 0.6, 2), 1 / 2)) / (1 / (POWER(POWER(0.2 - 0.5, 2) + POWER(0.3 - 0.6, 2), 1 / 2)) + 1 / (POWER(POWER(0.2 - 1.0, 2) + POWER(0.3 - 2.0, 2), 1 / 2)) + 1 / (POWER(POWER(0.2 - 100.0, 2) + POWER(0.3 - 200.0, 2), 1 / 2))) END)'
        assert predict_proba_sql[1] == '(CASE WHEN POWER(POWER(0.2 - 1.0, 2) + POWER(0.3 - 2.0, 2), 1 / 2) = 0 THEN 1.0 ELSE 1 / (POWER(POWER(0.2 - 1.0, 2) + POWER(0.3 - 2.0, 2), 1 / 2)) / (1 / (POWER(POWER(0.2 - 0.5, 2) + POWER(0.3 - 0.6, 2), 1 / 2)) + 1 / (POWER(POWER(0.2 - 1.0, 2) + POWER(0.3 - 2.0, 2), 1 / 2)) + 1 / (POWER(POWER(0.2 - 100.0, 2) + POWER(0.3 - 200.0, 2), 1 / 2))) END)'
        assert predict_proba_sql[2] == '(CASE WHEN POWER(POWER(0.2 - 100.0, 2) + POWER(0.3 - 200.0, 2), 1 / 2) = 0 THEN 1.0 ELSE 1 / (POWER(POWER(0.2 - 100.0, 2) + POWER(0.3 - 200.0, 2), 1 / 2)) / (1 / (POWER(POWER(0.2 - 0.5, 2) + POWER(0.3 - 0.6, 2), 1 / 2)) + 1 / (POWER(POWER(0.2 - 1.0, 2) + POWER(0.3 - 2.0, 2), 1 / 2)) + 1 / (POWER(POWER(0.2 - 100.0, 2) + POWER(0.3 - 200.0, 2), 1 / 2))) END)'
        transform_val = model.transform([[0.2, 0.3]])
        assert transform_val[0][0] == pytest.approx(0.42426407)
        assert transform_val[0][1] == pytest.approx(1.87882942)
        assert transform_val[0][2] == pytest.approx(223.24903135)
        transform_val_sql = model.transform_sql([0.2, 0.3])
        assert transform_val_sql[0] == 'POWER(POWER(0.2 - 0.5, 2) + POWER(0.3 - 0.6, 2), 1 / 2)'
        assert transform_val_sql[1] == 'POWER(POWER(0.2 - 1.0, 2) + POWER(0.3 - 2.0, 2), 1 / 2)'
        assert transform_val_sql[2] == 'POWER(POWER(0.2 - 100.0, 2) + POWER(0.3 - 200.0, 2), 1 / 2)'
        attributes = model.get_attributes()
        assert attributes["clusters"][0][0] == 0.5
        assert attributes["clusters"][0][1] == 0.6
        assert attributes["p"] == 2
        model.set_attributes({"clusters": [[0.1, 0.2]], "p": 3})
        attributes = model.get_attributes()
        assert attributes["clusters"][0][0] == 0.1
        assert attributes["clusters"][0][1] == 0.2
        assert attributes["p"] == 3
        assert model.model_type_ == "KMeans"

    def test_NearestCentroid(self,):
        model = memModel("NearestCentroid", {"clusters": [[0.5, 0.6,], [1, 2,], [100, 200,]], 
                                             "p": 2,
                                             "classes": ['a', 'b', 'c']})
        assert model.predict([[0.2, 0.3]])[0] == 'a'
        assert model.predict([[2, 2]])[0] == 'b'
        assert model.predict([[100, 201]])[0] == 'c'
        assert model.predict_sql([0.4, 0.5]) == "CASE WHEN 0.4 IS NULL OR 0.5 IS NULL THEN NULL WHEN POWER(POWER(0.4 - 100.0, 2) + POWER(0.5 - 200.0, 2), 1 / 2) <= POWER(POWER(0.4 - 0.5, 2) + POWER(0.5 - 0.6, 2), 1 / 2) AND POWER(POWER(0.4 - 100.0, 2) + POWER(0.5 - 200.0, 2), 1 / 2) <= POWER(POWER(0.4 - 1.0, 2) + POWER(0.5 - 2.0, 2), 1 / 2) THEN 'c' WHEN POWER(POWER(0.4 - 1.0, 2) + POWER(0.5 - 2.0, 2), 1 / 2) <= POWER(POWER(0.4 - 0.5, 2) + POWER(0.5 - 0.6, 2), 1 / 2) THEN 'b' ELSE 'a' END"
        predict_proba_val = model.predict_proba([[0.2, 0.3]])
        assert predict_proba_val[0][0] == pytest.approx(0.81452236)
        assert predict_proba_val[0][1] == pytest.approx(0.18392972)
        assert predict_proba_val[0][2] == pytest.approx(0.001547924158153152)
        predict_proba_sql = model.predict_proba_sql([0.2, 0.3])
        assert predict_proba_sql[0] == '(CASE WHEN POWER(POWER(0.2 - 0.5, 2) + POWER(0.3 - 0.6, 2), 1 / 2) = 0 THEN 1.0 ELSE 1 / (POWER(POWER(0.2 - 0.5, 2) + POWER(0.3 - 0.6, 2), 1 / 2)) / (1 / (POWER(POWER(0.2 - 0.5, 2) + POWER(0.3 - 0.6, 2), 1 / 2)) + 1 / (POWER(POWER(0.2 - 1.0, 2) + POWER(0.3 - 2.0, 2), 1 / 2)) + 1 / (POWER(POWER(0.2 - 100.0, 2) + POWER(0.3 - 200.0, 2), 1 / 2))) END)'
        assert predict_proba_sql[1] == '(CASE WHEN POWER(POWER(0.2 - 1.0, 2) + POWER(0.3 - 2.0, 2), 1 / 2) = 0 THEN 1.0 ELSE 1 / (POWER(POWER(0.2 - 1.0, 2) + POWER(0.3 - 2.0, 2), 1 / 2)) / (1 / (POWER(POWER(0.2 - 0.5, 2) + POWER(0.3 - 0.6, 2), 1 / 2)) + 1 / (POWER(POWER(0.2 - 1.0, 2) + POWER(0.3 - 2.0, 2), 1 / 2)) + 1 / (POWER(POWER(0.2 - 100.0, 2) + POWER(0.3 - 200.0, 2), 1 / 2))) END)'
        assert predict_proba_sql[2] == '(CASE WHEN POWER(POWER(0.2 - 100.0, 2) + POWER(0.3 - 200.0, 2), 1 / 2) = 0 THEN 1.0 ELSE 1 / (POWER(POWER(0.2 - 100.0, 2) + POWER(0.3 - 200.0, 2), 1 / 2)) / (1 / (POWER(POWER(0.2 - 0.5, 2) + POWER(0.3 - 0.6, 2), 1 / 2)) + 1 / (POWER(POWER(0.2 - 1.0, 2) + POWER(0.3 - 2.0, 2), 1 / 2)) + 1 / (POWER(POWER(0.2 - 100.0, 2) + POWER(0.3 - 200.0, 2), 1 / 2))) END)'
        transform_val = model.transform([[0.2, 0.3]])
        assert transform_val[0][0] == pytest.approx(0.42426407)
        assert transform_val[0][1] == pytest.approx(1.87882942)
        assert transform_val[0][2] == pytest.approx(223.24903135)
        transform_val_sql = model.transform_sql([0.2, 0.3])
        assert transform_val_sql[0] == 'POWER(POWER(0.2 - 0.5, 2) + POWER(0.3 - 0.6, 2), 1 / 2)'
        assert transform_val_sql[1] == 'POWER(POWER(0.2 - 1.0, 2) + POWER(0.3 - 2.0, 2), 1 / 2)'
        assert transform_val_sql[2] == 'POWER(POWER(0.2 - 100.0, 2) + POWER(0.3 - 200.0, 2), 1 / 2)'
        attributes = model.get_attributes()
        assert attributes["clusters"][0][0] == 0.5
        assert attributes["clusters"][0][1] == 0.6
        assert attributes["classes"][0] == 'a'
        assert attributes["classes"][1] == 'b'
        assert attributes["classes"][2] == 'c'
        assert attributes["p"] == 2
        model.set_attributes({"clusters": [[0.1, 0.2]], "p": 3})
        attributes = model.get_attributes()
        assert attributes["clusters"][0][0] == 0.1
        assert attributes["clusters"][0][1] == 0.2
        assert attributes["p"] == 3
        assert model.model_type_ == "NearestCentroid"

    def test_BisectingKMeans(self,):
        model = memModel("BisectingKMeans", {"clusters": [[0.5, 0.6,], [1, 2,], [100, 200,], [10, 700,], [-100, -200,]], 
                                             "p": 2,
                                             "left_child": [1, 3, None, None, None,],
                                             "right_child": [2, 4, None, None, None,],})
        assert model.predict([[0.2, 0.3]])[0] == 4
        assert model.predict([[2, 2]])[0] == 4
        assert model.predict([[100, 201]])[0] == 2
        assert model.predict_sql([0.4, 0.5]) == '(CASE WHEN 0.4 IS NULL OR 0.5 IS NULL THEN NULL ELSE (CASE WHEN POWER(POWER(0.4 - 1.0, 2) + POWER(0.5 - 2.0, 2), 1/2) < POWER(POWER(0.4 - 100.0, 2) + POWER(0.5 - 200.0, 2), 1/2) THEN (CASE WHEN POWER(POWER(0.4 - 10.0, 2) + POWER(0.5 - 700.0, 2), 1/2) < POWER(POWER(0.4 - -100.0, 2) + POWER(0.5 - -200.0, 2), 1/2) THEN 3 ELSE 4 END) ELSE 2 END) END)'
        transform_val = model.transform([[0.2, 0.3]])
        assert transform_val[0][0] == pytest.approx(0.42426407)
        assert transform_val[0][1] == pytest.approx(1.87882942)
        assert transform_val[0][2] == pytest.approx(223.24903135)
        transform_val_sql = model.transform_sql([0.2, 0.3])
        assert transform_val_sql[0] == 'POWER(POWER(0.2 - 0.5, 2) + POWER(0.3 - 0.6, 2), 1 / 2)'
        assert transform_val_sql[1] == 'POWER(POWER(0.2 - 1.0, 2) + POWER(0.3 - 2.0, 2), 1 / 2)'
        assert transform_val_sql[2] == 'POWER(POWER(0.2 - 100.0, 2) + POWER(0.3 - 200.0, 2), 1 / 2)'
        attributes = model.get_attributes()
        assert attributes["clusters"][0][0] == 0.5
        assert attributes["clusters"][0][1] == 0.6
        assert attributes["p"] == 2
        model.set_attributes({"clusters": [[0.1, 0.2]], "p": 3})
        attributes = model.get_attributes()
        assert attributes["clusters"][0][0] == 0.1
        assert attributes["clusters"][0][1] == 0.2
        assert attributes["p"] == 3
        assert model.model_type_ == "BisectingKMeans"

    def test_BinaryTreeRegressor(self,):
        model = memModel("BinaryTreeRegressor", {"children_left": [1, 3, None, None, None], 
                                                 "children_right": [2, 4, None, None, None],
                                                 "feature": [0, 1, None, None, None],
                                                 "threshold": ['female', 30, None, None, None],
                                                 "value": [None, None, 3, 11, 1993],})
        prediction = model.predict([['male', 100], ['female', 20] , ['female', 50]])
        assert prediction[0] == pytest.approx(3.0)
        assert prediction[1] == pytest.approx(11.0)
        assert prediction[2] == pytest.approx(1993.0)
        assert model.predict_sql(['sex', 'fare']) == "(CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 11 ELSE 1993 END) ELSE 3 END)"
        attributes = model.get_attributes()
        assert attributes["children_left"][0] == 1
        assert attributes["children_left"][1] == 3
        assert attributes["children_right"][0] == 2
        assert attributes["children_right"][1] == 4
        assert attributes["feature"][0] == 0
        assert attributes["feature"][1] == 1
        assert attributes["threshold"][0] == 'female'
        assert attributes["threshold"][1] == 30
        assert attributes["value"][2] == 3
        assert attributes["value"][3] == 11
        assert model.model_type_ == "BinaryTreeRegressor"

    def test_BinaryTreeClassifier(self,):
        model = memModel("BinaryTreeClassifier", {"children_left": [1, 3, None, None, None], 
                                                  "children_right": [2, 4, None, None, None],
                                                  "feature": [0, 1, None, None, None],
                                                  "threshold": ['female', 30, None, None, None],
                                                  "value": [None, None, [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]],
                                                  "classes": ['a', 'b', 'c',]})
        prediction = model.predict([['male', 100], ['female', 20] , ['female', 50]])
        assert prediction[0] == 'a'
        assert prediction[1] == 'b'
        assert prediction[2] == 'c'
        assert model.predict_sql(['sex', 'fare']) == "(CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 'b' ELSE 'c' END) ELSE 'a' END)"
        prediction = model.predict_proba([['male', 100], ['female', 20] , ['female', 50]])
        assert prediction[0][0] == 0.8
        assert prediction[0][1] == 0.1
        assert prediction[0][2] == 0.1
        assert prediction[1][0] == 0.1
        assert prediction[1][1] == 0.8
        assert prediction[1][2] == 0.1
        assert prediction[2][0] == 0.2
        assert prediction[2][1] == 0.2
        assert prediction[2][2] == 0.6
        attributes = model.get_attributes()
        assert attributes["children_left"][0] == 1
        assert attributes["children_left"][1] == 3
        assert attributes["children_right"][0] == 2
        assert attributes["children_right"][1] == 4
        assert attributes["feature"][0] == 0
        assert attributes["feature"][1] == 1
        assert attributes["threshold"][0] == 'female'
        assert attributes["threshold"][1] == 30
        assert attributes["value"][2][0] == 0.8
        assert attributes["value"][3][0] == 0.1
        model.set_attributes({"classes": [0, 1, 2],})
        attributes = model.get_attributes()
        assert attributes["classes"][0] == 0
        assert attributes["classes"][1] == 1
        assert attributes["classes"][2] == 2
        assert model.model_type_ == "BinaryTreeClassifier"

    def test_CHAID(self, titanic_vd,):
        tree = titanic_vd.chaid("survived", ["sex", "fare"]).attributes_["tree"]
        model = memModel("CHAID", {"tree": tree,
                                   "classes": ['a', 'b',]})
        prediction = model.predict([['male', 100], ['female', 20] , ['female', 50]])
        assert prediction[0] == 'a'
        assert prediction[1] == 'b'
        assert prediction[2] == 'b'
        assert model.predict_sql(['sex', 'fare']) == "(CASE WHEN sex = 'female' THEN (CASE WHEN fare <= 127.6 THEN 'b' WHEN fare <= 255.2 THEN 'b' WHEN fare <= 382.8 THEN 'b' WHEN fare <= 638.0 THEN 'b' ELSE NULL END) WHEN sex = 'male' THEN (CASE WHEN fare <= 129.36 THEN 'a' WHEN fare <= 258.72 THEN 'a' WHEN fare <= 388.08 THEN 'a' WHEN fare <= 517.44 THEN 'b' ELSE NULL END) ELSE NULL END)"
        prediction = model.predict_proba([['male', 100], ['female', 20] , ['female', 50]])
        assert prediction[0][0] == pytest.approx(0.82129278)
        assert prediction[0][1] == pytest.approx(0.17870722)
        assert prediction[1][0] == pytest.approx(0.3042328)
        assert prediction[1][1] == pytest.approx(0.6957672)
        assert prediction[2][0] == pytest.approx(0.3042328)
        assert prediction[2][1] == pytest.approx(0.6957672)
        attributes = model.get_attributes()
        assert attributes["tree"]["chi2"] == pytest.approx(38.3735130215037)
        assert not(attributes["tree"]["is_leaf"])
        assert not(attributes["tree"]["split_is_numerical"])
        assert attributes["tree"]["split_predictor"] == '"sex"'
        assert attributes["tree"]["split_predictor_idx"] == 0
        assert attributes["tree"]["children"]['female']["chi2"] == pytest.approx(23.265415918308957)
        model.set_attributes({"classes": [0, 1,],})
        attributes = model.get_attributes()
        assert attributes["classes"][0] == 0
        assert attributes["classes"][1] == 1
        assert model.model_type_ == "CHAID"

    def test_RandomForestRegressor(self,):
        model1 = memModel("BinaryTreeRegressor", {"children_left": [1, 3, None, None, None], 
                                                  "children_right": [2, 4, None, None, None],
                                                  "feature": [0, 1, None, None, None],
                                                  "threshold": ['female', 30, None, None, None],
                                                  "value": [None, None, 3, 11, 1993],})
        model2 = memModel("BinaryTreeRegressor", {"children_left": [1, 3, None, None, None], 
                                                  "children_right": [2, 4, None, None, None],
                                                  "feature": [0, 1, None, None, None],
                                                  "threshold": ['female', 30, None, None, None],
                                                  "value": [None, None, -3, -11, -1993],})
        model3 = memModel("BinaryTreeRegressor", {"children_left": [1, 3, None, None, None], 
                                                  "children_right": [2, 4, None, None, None],
                                                  "feature": [0, 1, None, None, None],
                                                  "threshold": ['female', 30, None, None, None],
                                                  "value": [None, None, 0, 3, 6],})
        model = memModel("RandomForestRegressor", {"trees": [model1, model2, model3]})
        prediction = model.predict([['male', 100], ['female', 20] , ['female', 50]])
        assert prediction[0] == pytest.approx(0.0)
        assert prediction[1] == pytest.approx(1.0)
        assert prediction[2] == pytest.approx(2.0)
        assert model.predict_sql(['sex', 'fare']) == "((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 11 ELSE 1993 END) ELSE 3 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN -11 ELSE -1993 END) ELSE -3 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 3 ELSE 6 END) ELSE 0 END)) / 3"
        attributes = model.get_attributes()["trees"][0].get_attributes()
        assert attributes["children_left"][0] == 1
        assert attributes["children_left"][1] == 3
        assert attributes["children_right"][0] == 2
        assert attributes["children_right"][1] == 4
        assert attributes["feature"][0] == 0
        assert attributes["feature"][1] == 1
        assert attributes["threshold"][0] == 'female'
        assert attributes["threshold"][1] == 30
        assert attributes["value"][2] == 3
        assert attributes["value"][3] == 11
        assert model.model_type_ == "RandomForestRegressor"

    def test_RandomForestClassifier(self,):
        model1 = memModel("BinaryTreeClassifier", {"children_left": [1, 3, None, None, None], 
                                                   "children_right": [2, 4, None, None, None],
                                                   "feature": [0, 1, None, None, None],
                                                   "threshold": ['female', 30, None, None, None],
                                                   "value": [None, None, [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
                                                   "classes": ['a', 'b', 'c'],})
        model2 = memModel("BinaryTreeClassifier", {"children_left": [1, 3, None, None, None], 
                                                   "children_right": [2, 4, None, None, None],
                                                   "feature": [0, 1, None, None, None],
                                                   "threshold": ['female', 30, None, None, None],
                                                   "value": [None, None, [0.7, 0.15, 0.15], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]],
                                                   "classes": ['a', 'b', 'c'],})
        model3 = memModel("BinaryTreeClassifier", {"children_left": [1, 3, None, None, None], 
                                                   "children_right": [2, 4, None, None, None],
                                                   "feature": [0, 1, None, None, None],
                                                   "threshold": ['female', 30, None, None, None],
                                                   "value": [None, None, [0.3, 0.7, 0.0], [0.0, 0.4, 0.6], [0.9, 0.1, 0.0]],
                                                   "classes": ['a', 'b', 'c'],})
        model = memModel("RandomForestClassifier", {"trees": [model1, model2, model3]})
        prediction = model.predict([['male', 100], ['female', 20] , ['female', 50]])
        assert prediction[0] == 'a'
        assert prediction[1] == 'b'
        assert prediction[2] == 'c'
        assert model.predict_sql(['sex', 'fare']) == "CASE WHEN sex IS NULL OR fare IS NULL THEN NULL WHEN ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 1.0 END) ELSE 0.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 1.0 END) ELSE 0.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 1.0 ELSE 0.0 END) ELSE 0.0 END)) / 3 >= ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 0.0 END) ELSE 1.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 0.0 END) ELSE 1.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 1.0 END) ELSE 0.0 END)) / 3 AND ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 1.0 END) ELSE 0.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 1.0 END) ELSE 0.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 1.0 ELSE 0.0 END) ELSE 0.0 END)) / 3 >= ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 1.0 ELSE 0.0 END) ELSE 0.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 1.0 ELSE 0.0 END) ELSE 0.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 0.0 END) ELSE 1.0 END)) / 3 THEN 'c' WHEN ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 1.0 ELSE 0.0 END) ELSE 0.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 1.0 ELSE 0.0 END) ELSE 0.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 0.0 END) ELSE 1.0 END)) / 3 >= ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 0.0 END) ELSE 1.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 0.0 END) ELSE 1.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 1.0 END) ELSE 0.0 END)) / 3 THEN 'b' ELSE 'a' END"
        prediction = model.predict_proba([['male', 100], ['female', 20] , ['female', 50]])
        assert prediction[0][0] == pytest.approx(0.66666667)
        assert prediction[0][1] == pytest.approx(0.33333333)
        assert prediction[0][2] == pytest.approx(0.0)
        assert prediction[1][0] == pytest.approx(0.0)
        assert prediction[1][1] == pytest.approx(0.66666667)
        assert prediction[1][2] == pytest.approx(0.33333333)
        assert prediction[2][0] == pytest.approx(0.33333333)
        assert prediction[2][1] == pytest.approx(0.0)
        assert prediction[2][2] == pytest.approx(0.66666667)
        prediction = model.predict_proba_sql(["sex", "fare"])
        assert prediction[0] == "((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 0.0 END) ELSE 1.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 0.0 END) ELSE 1.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 1.0 END) ELSE 0.0 END)) / 3"
        assert prediction[1] == "((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 1.0 ELSE 0.0 END) ELSE 0.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 1.0 ELSE 0.0 END) ELSE 0.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 0.0 END) ELSE 1.0 END)) / 3"
        assert prediction[2] == "((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 1.0 END) ELSE 0.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 1.0 END) ELSE 0.0 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 1.0 ELSE 0.0 END) ELSE 0.0 END)) / 3"
        attributes = model.get_attributes()["trees"][0].get_attributes()
        assert attributes["children_left"][0] == 1
        assert attributes["children_left"][1] == 3
        assert attributes["children_right"][0] == 2
        assert attributes["children_right"][1] == 4
        assert attributes["feature"][0] == 0
        assert attributes["feature"][1] == 1
        assert attributes["threshold"][0] == 'female'
        assert attributes["threshold"][1] == 30
        assert attributes["value"][2][0] == 0.8
        assert attributes["value"][3][0] == 0.1
        assert model.model_type_ == "RandomForestClassifier"

    def test_XGBoostRegressor(self,):
        model1 = memModel("BinaryTreeRegressor", {"children_left": [1, 3, None, None, None], 
                                                  "children_right": [2, 4, None, None, None],
                                                  "feature": [0, 1, None, None, None],
                                                  "threshold": ['female', 30, None, None, None],
                                                  "value": [None, None, 3, 11, 1993],})
        model2 = memModel("BinaryTreeRegressor", {"children_left": [1, 3, None, None, None], 
                                                  "children_right": [2, 4, None, None, None],
                                                  "feature": [0, 1, None, None, None],
                                                  "threshold": ['female', 30, None, None, None],
                                                  "value": [None, None, -3, -11, -1993],})
        model3 = memModel("BinaryTreeRegressor", {"children_left": [1, 3, None, None, None], 
                                                  "children_right": [2, 4, None, None, None],
                                                  "feature": [0, 1, None, None, None],
                                                  "threshold": ['female', 30, None, None, None],
                                                  "value": [None, None, 0, 3, 6],})
        model = memModel("XGBoostRegressor", {"trees": [model1, model2, model3],
                                              "learning_rate": 0.1,
                                              "mean": 1.0})
        prediction = model.predict([['male', 100], ['female', 20] , ['female', 50]])
        assert prediction[0] == pytest.approx(1.0)
        assert prediction[1] == pytest.approx(1.3)
        assert prediction[2] == pytest.approx(1.6)
        assert model.predict_sql(['sex', 'fare']) == "((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 11 ELSE 1993 END) ELSE 3 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN -11 ELSE -1993 END) ELSE -3 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 3 ELSE 6 END) ELSE 0 END)) * 0.1 + 1.0"
        attributes = model.get_attributes()["trees"][0].get_attributes()
        assert attributes["children_left"][0] == 1
        assert attributes["children_left"][1] == 3
        assert attributes["children_right"][0] == 2
        assert attributes["children_right"][1] == 4
        assert attributes["feature"][0] == 0
        assert attributes["feature"][1] == 1
        assert attributes["threshold"][0] == 'female'
        assert attributes["threshold"][1] == 30
        assert attributes["value"][2] == 3
        assert attributes["value"][3] == 11
        attributes = model.get_attributes()
        assert attributes["learning_rate"] == 0.1
        assert attributes["mean"] == 1.0
        model.set_attributes({"learning_rate": 0.2, "mean": 2.0})
        attributes = model.get_attributes()
        assert attributes["learning_rate"] == 0.2
        assert attributes["mean"] == 2.0
        assert model.model_type_ == "XGBoostRegressor"

    def test_XGBoostClassifier(self,):
        model1 = memModel("BinaryTreeClassifier", {"children_left": [1, 3, None, None, None], 
                                           "children_right": [2, 4, None, None, None],
                                           "feature": [0, 1, None, None, None],
                                           "threshold": ['female', 30, None, None, None],
                                           "value": [None, None, [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
                                           "classes": ['a', 'b', 'c'],})
        model2 = memModel("BinaryTreeClassifier", {"children_left": [1, 3, None, None, None], 
                                                   "children_right": [2, 4, None, None, None],
                                                   "feature": [0, 1, None, None, None],
                                                   "threshold": ['female', 30, None, None, None],
                                                   "value": [None, None, [0.7, 0.15, 0.15], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]],
                                                   "classes": ['a', 'b', 'c'],})
        model3 = memModel("BinaryTreeClassifier", {"children_left": [1, 3, None, None, None], 
                                                   "children_right": [2, 4, None, None, None],
                                                   "feature": [0, 1, None, None, None],
                                                   "threshold": ['female', 30, None, None, None],
                                                   "value": [None, None, [0.3, 0.7, 0.0], [0.0, 0.4, 0.6], [0.9, 0.1, 0.0]],
                                                   "classes": ['a', 'b', 'c'],})
        model = memModel("XGBoostClassifier", {"trees": [model1, model2, model3],
                                               "learning_rate": 0.1,
                                               "logodds": [0.1, 0.12, 0.15]})
        prediction = model.predict([['male', 100], ['female', 20] , ['female', 50]])
        assert prediction[0] == 'a'
        assert prediction[1] == 'b'
        assert prediction[2] == 'c'
        assert model.predict_sql(['sex', 'fare']) == "CASE WHEN sex IS NULL OR fare IS NULL THEN NULL WHEN (1 / (1 + EXP(- (0.15 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.8 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.6 END) ELSE 0.15 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.0 END) ELSE 0.0 END)))))) / ((1 / (1 + EXP(- (0.1 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.1 END) ELSE 0.8 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.2 END) ELSE 0.7 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 0.9 END) ELSE 0.3 END)))))) + (1 / (1 + EXP(- (0.12 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.8 ELSE 0.1 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.2 END) ELSE 0.15 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.4 ELSE 0.1 END) ELSE 0.7 END)))))) + (1 / (1 + EXP(- (0.15 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.8 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.6 END) ELSE 0.15 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.0 END) ELSE 0.0 END))))))) >= (1 / (1 + EXP(- (0.1 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.1 END) ELSE 0.8 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.2 END) ELSE 0.7 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 0.9 END) ELSE 0.3 END)))))) / ((1 / (1 + EXP(- (0.1 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.1 END) ELSE 0.8 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.2 END) ELSE 0.7 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 0.9 END) ELSE 0.3 END)))))) + (1 / (1 + EXP(- (0.12 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.8 ELSE 0.1 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.2 END) ELSE 0.15 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.4 ELSE 0.1 END) ELSE 0.7 END)))))) + (1 / (1 + EXP(- (0.15 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.8 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.6 END) ELSE 0.15 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.0 END) ELSE 0.0 END))))))) AND (1 / (1 + EXP(- (0.15 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.8 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.6 END) ELSE 0.15 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.0 END) ELSE 0.0 END)))))) / ((1 / (1 + EXP(- (0.1 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.1 END) ELSE 0.8 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.2 END) ELSE 0.7 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 0.9 END) ELSE 0.3 END)))))) + (1 / (1 + EXP(- (0.12 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.8 ELSE 0.1 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.2 END) ELSE 0.15 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.4 ELSE 0.1 END) ELSE 0.7 END)))))) + (1 / (1 + EXP(- (0.15 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.8 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.6 END) ELSE 0.15 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.0 END) ELSE 0.0 END))))))) >= (1 / (1 + EXP(- (0.12 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.8 ELSE 0.1 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.2 END) ELSE 0.15 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.4 ELSE 0.1 END) ELSE 0.7 END)))))) / ((1 / (1 + EXP(- (0.1 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.1 END) ELSE 0.8 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.2 END) ELSE 0.7 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 0.9 END) ELSE 0.3 END)))))) + (1 / (1 + EXP(- (0.12 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.8 ELSE 0.1 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.2 END) ELSE 0.15 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.4 ELSE 0.1 END) ELSE 0.7 END)))))) + (1 / (1 + EXP(- (0.15 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.8 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.6 END) ELSE 0.15 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.0 END) ELSE 0.0 END))))))) THEN 'c' WHEN (1 / (1 + EXP(- (0.12 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.8 ELSE 0.1 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.2 END) ELSE 0.15 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.4 ELSE 0.1 END) ELSE 0.7 END)))))) / ((1 / (1 + EXP(- (0.1 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.1 END) ELSE 0.8 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.2 END) ELSE 0.7 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 0.9 END) ELSE 0.3 END)))))) + (1 / (1 + EXP(- (0.12 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.8 ELSE 0.1 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.2 END) ELSE 0.15 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.4 ELSE 0.1 END) ELSE 0.7 END)))))) + (1 / (1 + EXP(- (0.15 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.8 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.6 END) ELSE 0.15 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.0 END) ELSE 0.0 END))))))) >= (1 / (1 + EXP(- (0.1 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.1 END) ELSE 0.8 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.2 END) ELSE 0.7 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 0.9 END) ELSE 0.3 END)))))) / ((1 / (1 + EXP(- (0.1 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.1 END) ELSE 0.8 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.2 END) ELSE 0.7 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 0.9 END) ELSE 0.3 END)))))) + (1 / (1 + EXP(- (0.12 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.8 ELSE 0.1 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.2 END) ELSE 0.15 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.4 ELSE 0.1 END) ELSE 0.7 END)))))) + (1 / (1 + EXP(- (0.15 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.8 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.6 END) ELSE 0.15 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.0 END) ELSE 0.0 END))))))) THEN 'b' ELSE 'a' END"
        prediction = model.predict_proba([['male', 100], ['female', 20] , ['female', 50]])
        assert prediction[0][0] == pytest.approx(0.34171499)
        assert prediction[0][1] == pytest.approx(0.33211396)
        assert prediction[0][2] == pytest.approx(0.32617105)
        assert prediction[1][0] == pytest.approx(0.31948336)
        assert prediction[1][1] == pytest.approx(0.34467713)
        assert prediction[1][2] == pytest.approx(0.33583951)
        assert prediction[2][0] == pytest.approx(0.33286283)
        assert prediction[2][1] == pytest.approx(0.32394435)
        assert prediction[2][2] == pytest.approx(0.34319282)
        prediction = model.predict_proba_sql(["sex", "fare"])
        assert prediction[0] == "(1 / (1 + EXP(- (0.1 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.1 END) ELSE 0.8 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.2 END) ELSE 0.7 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 0.9 END) ELSE 0.3 END)))))) / ((1 / (1 + EXP(- (0.1 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.1 END) ELSE 0.8 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.2 END) ELSE 0.7 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 0.9 END) ELSE 0.3 END)))))) + (1 / (1 + EXP(- (0.12 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.8 ELSE 0.1 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.2 END) ELSE 0.15 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.4 ELSE 0.1 END) ELSE 0.7 END)))))) + (1 / (1 + EXP(- (0.15 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.8 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.6 END) ELSE 0.15 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.0 END) ELSE 0.0 END)))))))"
        assert prediction[1] == "(1 / (1 + EXP(- (0.12 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.8 ELSE 0.1 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.2 END) ELSE 0.15 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.4 ELSE 0.1 END) ELSE 0.7 END)))))) / ((1 / (1 + EXP(- (0.1 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.1 END) ELSE 0.8 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.2 END) ELSE 0.7 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 0.9 END) ELSE 0.3 END)))))) + (1 / (1 + EXP(- (0.12 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.8 ELSE 0.1 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.2 END) ELSE 0.15 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.4 ELSE 0.1 END) ELSE 0.7 END)))))) + (1 / (1 + EXP(- (0.15 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.8 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.6 END) ELSE 0.15 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.0 END) ELSE 0.0 END)))))))"
        assert prediction[2] == "(1 / (1 + EXP(- (0.15 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.8 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.6 END) ELSE 0.15 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.0 END) ELSE 0.0 END)))))) / ((1 / (1 + EXP(- (0.1 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.1 END) ELSE 0.8 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.2 END) ELSE 0.7 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.0 ELSE 0.9 END) ELSE 0.3 END)))))) + (1 / (1 + EXP(- (0.12 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.8 ELSE 0.1 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.2 END) ELSE 0.15 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.4 ELSE 0.1 END) ELSE 0.7 END)))))) + (1 / (1 + EXP(- (0.15 + 0.1 * ((CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.1 ELSE 0.8 END) ELSE 0.1 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.2 ELSE 0.6 END) ELSE 0.15 END) + (CASE WHEN sex = 'female' THEN (CASE WHEN fare < 30 THEN 0.6 ELSE 0.0 END) ELSE 0.0 END)))))))"
        attributes = model.get_attributes()["trees"][0].get_attributes()
        assert attributes["children_left"][0] == 1
        assert attributes["children_left"][1] == 3
        assert attributes["children_right"][0] == 2
        assert attributes["children_right"][1] == 4
        assert attributes["feature"][0] == 0
        assert attributes["feature"][1] == 1
        assert attributes["threshold"][0] == 'female'
        assert attributes["threshold"][1] == 30
        assert attributes["value"][2][0] == 0.8
        assert attributes["value"][3][0] == 0.1
        assert model.model_type_ == "XGBoostClassifier"

    def test_NaiveBayes(self,):
        model = memModel("NaiveBayes",
             {"attributes": [{"type": "gaussian", 'C': {'mu': 63.9878308300395, 'sigma_sq': 7281.87598377196}, 'Q': {'mu': 13.0217386792453, 'sigma_sq': 211.626862330204}, 'S': {'mu': 27.6928120412844, 'sigma_sq': 1428.57067393938}},
                             {"type": "multinomial", 'C': 0.771666666666667, 'Q': 0.910714285714286, 'S': 0.878216123499142},
                             {"type": "bernoulli", 'C': 0.771666666666667, 'Q': 0.910714285714286, 'S': 0.878216123499142},
                             {'type': 'categorical', 'C': {'female': 0.407843137254902, 'male': 0.592156862745098}, 
                                                     'Q': {'female': 0.416666666666667, 'male': 0.583333333333333},
                                                     'S': {'female': 0.406666666666667, 'male': 0.593333333333333},},],
                      "classes": ["C", "Q", "S"],
                      "prior": [0.8, 0.1, 0.1],})
        prediction = model.predict([[40.0, 1, True, 'male'], [60.0, 3, True, 'male'] , [15.0, 2, False, 'female']])
        assert prediction[0] == 'C'
        assert prediction[1] == 'C'
        assert prediction[2] == 'Q'
        assert model.predict_sql(['age', 'pclass', 'survived', 'sex',]) == "CASE WHEN age IS NULL OR pclass IS NULL OR survived IS NULL OR sex IS NULL THEN NULL WHEN 0.010555023401917874 * EXP(- POWER(age - 27.6928120412844, 2) / 2857.14134787876) * POWER(0.878216123499142, pclass) * (CASE WHEN survived THEN 0.878216123499142 ELSE 0.12178387650085798 END) * DECODE(sex, 'female', 0.406666666666667, 'male', 0.593333333333333) * 0.1 >= 0.004675073323276673 * EXP(- POWER(age - 63.9878308300395, 2) / 14563.75196754392) * POWER(0.771666666666667, pclass) * (CASE WHEN survived THEN 0.771666666666667 ELSE 0.22833333333333306 END) * DECODE(sex, 'female', 0.407843137254902, 'male', 0.592156862745098) * 0.8 AND 0.010555023401917874 * EXP(- POWER(age - 27.6928120412844, 2) / 2857.14134787876) * POWER(0.878216123499142, pclass) * (CASE WHEN survived THEN 0.878216123499142 ELSE 0.12178387650085798 END) * DECODE(sex, 'female', 0.406666666666667, 'male', 0.593333333333333) * 0.1 >= 0.027423612860412977 * EXP(- POWER(age - 13.0217386792453, 2) / 423.253724660408) * POWER(0.910714285714286, pclass) * (CASE WHEN survived THEN 0.910714285714286 ELSE 0.08928571428571397 END) * DECODE(sex, 'female', 0.416666666666667, 'male', 0.583333333333333) * 0.1 THEN 'S' WHEN 0.027423612860412977 * EXP(- POWER(age - 13.0217386792453, 2) / 423.253724660408) * POWER(0.910714285714286, pclass) * (CASE WHEN survived THEN 0.910714285714286 ELSE 0.08928571428571397 END) * DECODE(sex, 'female', 0.416666666666667, 'male', 0.583333333333333) * 0.1 >= 0.004675073323276673 * EXP(- POWER(age - 63.9878308300395, 2) / 14563.75196754392) * POWER(0.771666666666667, pclass) * (CASE WHEN survived THEN 0.771666666666667 ELSE 0.22833333333333306 END) * DECODE(sex, 'female', 0.407843137254902, 'male', 0.592156862745098) * 0.8 THEN 'Q' ELSE 'C' END"
        prediction = model.predict_proba([[40.0, 1, True, 'male'], [60.0, 3, True, 'male'] , [15.0, 2, False, 'female']])
        assert prediction[0][0] == pytest.approx(0.64564673)
        assert prediction[0][1] == pytest.approx(0.12105224)
        assert prediction[0][2] == pytest.approx(0.23330103)
        assert prediction[1][0] == pytest.approx(0.74783083)
        assert prediction[1][1] == pytest.approx(0.00570541)
        assert prediction[1][2] == pytest.approx(0.24646376)
        assert prediction[2][0] == pytest.approx(0.34471925)
        assert prediction[2][1] == pytest.approx(0.49592024)
        assert prediction[2][2] == pytest.approx(0.15936051)
        prediction = model.predict_proba_sql(['age', 'pclass', 'survived', 'sex',])
        assert prediction[0] == "(0.004675073323276673 * EXP(- POWER(age - 63.9878308300395, 2) / 14563.75196754392) * POWER(0.771666666666667, pclass) * (CASE WHEN survived THEN 0.771666666666667 ELSE 0.22833333333333306 END) * DECODE(sex, 'female', 0.407843137254902, 'male', 0.592156862745098) * 0.8) / (0.004675073323276673 * EXP(- POWER(age - 63.9878308300395, 2) / 14563.75196754392) * POWER(0.771666666666667, pclass) * (CASE WHEN survived THEN 0.771666666666667 ELSE 0.22833333333333306 END) * DECODE(sex, 'female', 0.407843137254902, 'male', 0.592156862745098) * 0.8 + 0.027423612860412977 * EXP(- POWER(age - 13.0217386792453, 2) / 423.253724660408) * POWER(0.910714285714286, pclass) * (CASE WHEN survived THEN 0.910714285714286 ELSE 0.08928571428571397 END) * DECODE(sex, 'female', 0.416666666666667, 'male', 0.583333333333333) * 0.1 + 0.010555023401917874 * EXP(- POWER(age - 27.6928120412844, 2) / 2857.14134787876) * POWER(0.878216123499142, pclass) * (CASE WHEN survived THEN 0.878216123499142 ELSE 0.12178387650085798 END) * DECODE(sex, 'female', 0.406666666666667, 'male', 0.593333333333333) * 0.1)"
        assert prediction[1] == "(0.027423612860412977 * EXP(- POWER(age - 13.0217386792453, 2) / 423.253724660408) * POWER(0.910714285714286, pclass) * (CASE WHEN survived THEN 0.910714285714286 ELSE 0.08928571428571397 END) * DECODE(sex, 'female', 0.416666666666667, 'male', 0.583333333333333) * 0.1) / (0.004675073323276673 * EXP(- POWER(age - 63.9878308300395, 2) / 14563.75196754392) * POWER(0.771666666666667, pclass) * (CASE WHEN survived THEN 0.771666666666667 ELSE 0.22833333333333306 END) * DECODE(sex, 'female', 0.407843137254902, 'male', 0.592156862745098) * 0.8 + 0.027423612860412977 * EXP(- POWER(age - 13.0217386792453, 2) / 423.253724660408) * POWER(0.910714285714286, pclass) * (CASE WHEN survived THEN 0.910714285714286 ELSE 0.08928571428571397 END) * DECODE(sex, 'female', 0.416666666666667, 'male', 0.583333333333333) * 0.1 + 0.010555023401917874 * EXP(- POWER(age - 27.6928120412844, 2) / 2857.14134787876) * POWER(0.878216123499142, pclass) * (CASE WHEN survived THEN 0.878216123499142 ELSE 0.12178387650085798 END) * DECODE(sex, 'female', 0.406666666666667, 'male', 0.593333333333333) * 0.1)"
        assert prediction[2] == "(0.010555023401917874 * EXP(- POWER(age - 27.6928120412844, 2) / 2857.14134787876) * POWER(0.878216123499142, pclass) * (CASE WHEN survived THEN 0.878216123499142 ELSE 0.12178387650085798 END) * DECODE(sex, 'female', 0.406666666666667, 'male', 0.593333333333333) * 0.1) / (0.004675073323276673 * EXP(- POWER(age - 63.9878308300395, 2) / 14563.75196754392) * POWER(0.771666666666667, pclass) * (CASE WHEN survived THEN 0.771666666666667 ELSE 0.22833333333333306 END) * DECODE(sex, 'female', 0.407843137254902, 'male', 0.592156862745098) * 0.8 + 0.027423612860412977 * EXP(- POWER(age - 13.0217386792453, 2) / 423.253724660408) * POWER(0.910714285714286, pclass) * (CASE WHEN survived THEN 0.910714285714286 ELSE 0.08928571428571397 END) * DECODE(sex, 'female', 0.416666666666667, 'male', 0.583333333333333) * 0.1 + 0.010555023401917874 * EXP(- POWER(age - 27.6928120412844, 2) / 2857.14134787876) * POWER(0.878216123499142, pclass) * (CASE WHEN survived THEN 0.878216123499142 ELSE 0.12178387650085798 END) * DECODE(sex, 'female', 0.406666666666667, 'male', 0.593333333333333) * 0.1)"
        attributes = model.get_attributes()
        assert attributes["prior"][0] == 0.8
        assert attributes["prior"][1] == 0.1
        assert attributes["prior"][2] == 0.1
        assert attributes["classes"][0] == "C"
        assert attributes["classes"][1] == "Q"
        assert attributes["classes"][2] == "S"
        assert attributes["attributes"][0]["type"] == "gaussian"
        assert attributes["attributes"][1]["type"] == "multinomial"
        assert attributes["attributes"][2]["type"] == "bernoulli"
        assert attributes["attributes"][3]["type"] == "categorical"
        assert model.model_type_ == "NaiveBayes"
