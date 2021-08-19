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
        assert model.predict_sql([0.4, 0.5]) == '((1 - 1 / (1 + EXP(0.8 + 0.5 * 0.4 + 0.6 * 0.5))) > 0.5)::int'
        predict_proba_val = model.predict_proba([[0.4, 0.5]])
        assert predict_proba_val[0][0] == pytest.approx(0.21416502)
        assert predict_proba_val[0][1] == pytest.approx(0.78583498)
        predict_proba_val_sql = model.predict_proba_sql([0.4, 0.5])
        assert predict_proba_val_sql[0] == '1 - (1 - 1 / (1 + EXP(0.8 + 0.5 * 0.4 + 0.6 * 0.5)))'
        assert predict_proba_val_sql[1] == '1 - 1 / (1 + EXP(0.8 + 0.5 * 0.4 + 0.6 * 0.5))'
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
        assert model.predict_sql([0.4, 0.5]) == 'CASE WHEN 0.4 IS NULL OR 0.5 IS NULL THEN NULL WHEN POWER(POWER(0.4 - 100.0, 2) + POWER(0.5 - 200.0, 2), 2) <= POWER(POWER(0.4 - 0.5, 2) + POWER(0.5 - 0.6, 2), 2) AND POWER(POWER(0.4 - 100.0, 2) + POWER(0.5 - 200.0, 2), 2) <= POWER(POWER(0.4 - 1.0, 2) + POWER(0.5 - 2.0, 2), 2) THEN 2 WHEN POWER(POWER(0.4 - 1.0, 2) + POWER(0.5 - 2.0, 2), 2) <= POWER(POWER(0.4 - 0.5, 2) + POWER(0.5 - 0.6, 2), 2) THEN 1 ELSE 0 END'
        predict_proba_val = model.predict_proba([[0.2, 0.3]])
        assert predict_proba_val[0][0] == pytest.approx(0.81452236)
        assert predict_proba_val[0][1] == pytest.approx(0.18392972)
        assert predict_proba_val[0][2] == pytest.approx(0.001547924158153152)
        predict_proba_sql = model.predict_proba_sql([0.2, 0.3])
        assert predict_proba_sql[0] == '(CASE WHEN POWER(POWER(0.2 - 0.5, 2) + POWER(0.3 - 0.6, 2), 2) = 0 THEN 1.0 ELSE 1 / (POWER(POWER(0.2 - 0.5, 2) + POWER(0.3 - 0.6, 2), 2)) / (1 / (POWER(POWER(0.2 - 0.5, 2) + POWER(0.3 - 0.6, 2), 2)) + 1 / (POWER(POWER(0.2 - 1.0, 2) + POWER(0.3 - 2.0, 2), 2)) + 1 / (POWER(POWER(0.2 - 100.0, 2) + POWER(0.3 - 200.0, 2), 2))) END)'
        assert predict_proba_sql[1] == '(CASE WHEN POWER(POWER(0.2 - 1.0, 2) + POWER(0.3 - 2.0, 2), 2) = 0 THEN 1.0 ELSE 1 / (POWER(POWER(0.2 - 1.0, 2) + POWER(0.3 - 2.0, 2), 2)) / (1 / (POWER(POWER(0.2 - 0.5, 2) + POWER(0.3 - 0.6, 2), 2)) + 1 / (POWER(POWER(0.2 - 1.0, 2) + POWER(0.3 - 2.0, 2), 2)) + 1 / (POWER(POWER(0.2 - 100.0, 2) + POWER(0.3 - 200.0, 2), 2))) END)'
        assert predict_proba_sql[2] == '(CASE WHEN POWER(POWER(0.2 - 100.0, 2) + POWER(0.3 - 200.0, 2), 2) = 0 THEN 1.0 ELSE 1 / (POWER(POWER(0.2 - 100.0, 2) + POWER(0.3 - 200.0, 2), 2)) / (1 / (POWER(POWER(0.2 - 0.5, 2) + POWER(0.3 - 0.6, 2), 2)) + 1 / (POWER(POWER(0.2 - 1.0, 2) + POWER(0.3 - 2.0, 2), 2)) + 1 / (POWER(POWER(0.2 - 100.0, 2) + POWER(0.3 - 200.0, 2), 2))) END)'
        transform_val = model.transform([[0.2, 0.3]])
        assert transform_val[0][0] == pytest.approx(0.42426407)
        assert transform_val[0][1] == pytest.approx(1.87882942)
        assert transform_val[0][2] == pytest.approx(223.24903135)
        transform_val_sql = model.transform_sql([0.2, 0.3])
        assert transform_val_sql[0] == 'POWER(POWER(0.2 - 0.5, 2) + POWER(0.3 - 0.6, 2), 2)'
        assert transform_val_sql[1] == 'POWER(POWER(0.2 - 1.0, 2) + POWER(0.3 - 2.0, 2), 2)'
        assert transform_val_sql[2] == 'POWER(POWER(0.2 - 100.0, 2) + POWER(0.3 - 200.0, 2), 2)'
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

    def test_NearestCentroids(self,):
        model = memModel("NearestCentroids", {"clusters": [[0.5, 0.6,], [1, 2,], [100, 200,]], 
                                              "p": 2,
                                              "classes": ['a', 'b', 'c']})
        assert model.predict([[0.2, 0.3]])[0] == 'a'
        assert model.predict([[2, 2]])[0] == 'b'
        assert model.predict([[100, 201]])[0] == 'c'
        assert model.predict_sql([0.4, 0.5]) == 'CASE WHEN 0.4 IS NULL OR 0.5 IS NULL THEN NULL WHEN POWER(POWER(0.4 - 100.0, 2) + POWER(0.5 - 200.0, 2), 2) <= POWER(POWER(0.4 - 0.5, 2) + POWER(0.5 - 0.6, 2), 2) AND POWER(POWER(0.4 - 100.0, 2) + POWER(0.5 - 200.0, 2), 2) <= POWER(POWER(0.4 - 1.0, 2) + POWER(0.5 - 2.0, 2), 2) THEN \'c\' WHEN POWER(POWER(0.4 - 1.0, 2) + POWER(0.5 - 2.0, 2), 2) <= POWER(POWER(0.4 - 0.5, 2) + POWER(0.5 - 0.6, 2), 2) THEN \'b\' ELSE \'a\' END'
        predict_proba_val = model.predict_proba([[0.2, 0.3]])
        assert predict_proba_val[0][0] == pytest.approx(0.81452236)
        assert predict_proba_val[0][1] == pytest.approx(0.18392972)
        assert predict_proba_val[0][2] == pytest.approx(0.001547924158153152)
        predict_proba_sql = model.predict_proba_sql([0.2, 0.3])
        assert predict_proba_sql[0] == '(CASE WHEN POWER(POWER(0.2 - 0.5, 2) + POWER(0.3 - 0.6, 2), 2) = 0 THEN 1.0 ELSE 1 / (POWER(POWER(0.2 - 0.5, 2) + POWER(0.3 - 0.6, 2), 2)) / (1 / (POWER(POWER(0.2 - 0.5, 2) + POWER(0.3 - 0.6, 2), 2)) + 1 / (POWER(POWER(0.2 - 1.0, 2) + POWER(0.3 - 2.0, 2), 2)) + 1 / (POWER(POWER(0.2 - 100.0, 2) + POWER(0.3 - 200.0, 2), 2))) END)'
        assert predict_proba_sql[1] == '(CASE WHEN POWER(POWER(0.2 - 1.0, 2) + POWER(0.3 - 2.0, 2), 2) = 0 THEN 1.0 ELSE 1 / (POWER(POWER(0.2 - 1.0, 2) + POWER(0.3 - 2.0, 2), 2)) / (1 / (POWER(POWER(0.2 - 0.5, 2) + POWER(0.3 - 0.6, 2), 2)) + 1 / (POWER(POWER(0.2 - 1.0, 2) + POWER(0.3 - 2.0, 2), 2)) + 1 / (POWER(POWER(0.2 - 100.0, 2) + POWER(0.3 - 200.0, 2), 2))) END)'
        assert predict_proba_sql[2] == '(CASE WHEN POWER(POWER(0.2 - 100.0, 2) + POWER(0.3 - 200.0, 2), 2) = 0 THEN 1.0 ELSE 1 / (POWER(POWER(0.2 - 100.0, 2) + POWER(0.3 - 200.0, 2), 2)) / (1 / (POWER(POWER(0.2 - 0.5, 2) + POWER(0.3 - 0.6, 2), 2)) + 1 / (POWER(POWER(0.2 - 1.0, 2) + POWER(0.3 - 2.0, 2), 2)) + 1 / (POWER(POWER(0.2 - 100.0, 2) + POWER(0.3 - 200.0, 2), 2))) END)'
        transform_val = model.transform([[0.2, 0.3]])
        assert transform_val[0][0] == pytest.approx(0.42426407)
        assert transform_val[0][1] == pytest.approx(1.87882942)
        assert transform_val[0][2] == pytest.approx(223.24903135)
        transform_val_sql = model.transform_sql([0.2, 0.3])
        assert transform_val_sql[0] == 'POWER(POWER(0.2 - 0.5, 2) + POWER(0.3 - 0.6, 2), 2)'
        assert transform_val_sql[1] == 'POWER(POWER(0.2 - 1.0, 2) + POWER(0.3 - 2.0, 2), 2)'
        assert transform_val_sql[2] == 'POWER(POWER(0.2 - 100.0, 2) + POWER(0.3 - 200.0, 2), 2)'
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
        assert model.model_type_ == "NearestCentroids"

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
        assert transform_val_sql[0] == 'POWER(POWER(0.2 - 0.5, 2) + POWER(0.3 - 0.6, 2), 2)'
        assert transform_val_sql[1] == 'POWER(POWER(0.2 - 1.0, 2) + POWER(0.3 - 2.0, 2), 2)'
        assert transform_val_sql[2] == 'POWER(POWER(0.2 - 100.0, 2) + POWER(0.3 - 200.0, 2), 2)'
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


