"""
(c)  Copyright  [2018-2023]  OpenText  or one of its
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
import numpy as np
from verticapy.core.vdataframe.base import vDataFrame
from verticapy.datasets import load_winequality, load_titanic
from verticapy import drop
import verticapy
import random
import string


@pytest.fixture(scope="module", autouse=True)
def load_test_schema():
    alphabet = string.ascii_letters
    random_string = "".join(random.choice(alphabet) for i in range(4))
    schema_name = f"test_{random_string}"
    verticapy.create_schema(schema_name)
    yield schema_name
    verticapy.drop(schema_name, method="schema")


@pytest.fixture(scope="module")
def pred_cl_dataset_multi():
    labels = np.array(["a", "b", "c"])
    y_true = np.array(["a", "a", "b", "c", "c", "a", "b", "c", "a", "b", "c", "a", "b", "a", "b", "c", "a", "b", "c"])
    y_pred = np.array(["a", "b", "b", "b", "c", "a", "b", "a", "a", "c", "a", "a", "b", "a", "b", "c", "b", "b", "a"])

    labels_num = np.array([0, 1, 2])
    y_true_num = np.array([random.randint(0, 2) for _ in range(len(y_true))])
    y_pred_num = np.array([random.randint(0, 2) for _ in range(len(y_true))])

    y_prob = np.array([[random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)] for _ in range(len(y_true))])
    input_relation = np.column_stack((y_true, y_pred, y_true_num, y_pred_num, y_prob))
    vdf = vDataFrame(input_relation, usecols=["y_true", "y_pred", "y_true_num", "y_pred_num", "y_prob"])
    yield vdf, y_true, y_pred, y_prob, labels, y_true_num, y_pred_num, labels_num


@pytest.fixture(scope="module")
def pred_cl_dataset_binary():
    labels = np.array(["a", "b"])
    y_true = np.array(["a", "a", "b", "b", "a", "a", "b", "b", "a", "b", "a", "a", "b", "a", "b", "b", "a", "b", "a"])
    y_pred = np.array(["a", "b", "b", "b", "a", "a", "b", "a", "a", "b", "a", "a", "b", "a", "b", "a", "b", "b", "a"])

    labels_num = np.array([0, 1])
    y_true_num = np.array([random.randint(0, 1) for _ in range(len(y_true))])
    y_pred_num = np.array([random.randint(0, 1) for _ in range(len(y_true))])

    y_prob = np.array([random.uniform(0, 1) for _ in range(len(y_true))])

    input_relation = np.column_stack((y_true, y_pred, y_true_num, y_pred_num, y_prob))
    vdf = vDataFrame(input_relation, usecols=["y_true", "y_pred", "y_true_num", "y_pred_num", "y_prob"])
    yield vdf, y_true, y_pred, y_prob, labels, y_true_num, y_pred_num, labels_num


@pytest.fixture(scope="module")
def pred_cl_dataset_multilevel():
    labels = np.array(["c1", "c2", "c3", "c4"])
    y_true = np.array([[0, 1, 1], [1, 1, 1], [0, 1, 0], [1, 0, 1]])
    y_pred = np.array([[0, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]])
    input_relation = np.column_stack((y_true, y_pred))
    vdf = vDataFrame(input_relation, usecols=["y_t", "y_s"])
    yield vdf, y_true, y_pred, labels


@pytest.fixture(scope="module")
def winequality_vpy(load_test_schema):
    winequality = load_winequality(load_test_schema, 'winequality')
    yield winequality
    drop(name=f"{load_test_schema}.winequality")


@pytest.fixture(scope="module")
def titanic_vdf(load_test_schema):
    titanic = load_titanic(load_test_schema, 'titanic')
    yield titanic
    drop(name=f"{load_test_schema}.titanic")