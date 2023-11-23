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
REL_TOLERANCE = 1e-6
ABS_TOLERANCE = 1e-12

rel_tolerance_map = {
    "RandomForestRegressor": 1e-0,
    "RandomForestClassifier": 1e-0,
    "DecisionTreeRegressor": 1e-0,
    "DecisionTreeClassifier": 1e-0,
    "DummyTreeRegressor": 1e-01,
    "DummyTreeClassifier": 1e-01,
    "XGBRegressor": 1e-0,
    "XGBClassifier": 1e-0,
    "Ridge": 1e-02,
    "Lasso": 1e-02,
    "ElasticNet": 1e-02,
    "LinearRegression": 1e-02,
    "LinearSVR": 1e-01,
    "PoissonRegressor": 1e-01,
    "AR": 1e-01,
    "MA": 1e-01,
    "ARMA": 1e-01,
    "ARIMA": 1e-01,
    "TENSORFLOW": 1e-06,
}
