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
# Standard Python Modules
import random
import tempfile
import string
from contextlib import contextmanager

# Pytest
import pytest

# Other Modules
from scipy.special import erfinv
import numpy as np
import pandas as pd

# VerticaPy
import verticapy
from verticapy import drop
from verticapy.datasets import (
    load_titanic,
    load_iris,
    load_amazon,
    load_winequality,
    load_market,
    load_smart_meters,
    load_laliga,
    load_airline_passengers,
)
from verticapy.core.vdataframe.base import vDataFrame

DUMMY_TEST_SIZE = 100


@pytest.fixture(name="schema_loader", scope="session", autouse=True)
def load_test_schema():
    """
    Create a schema with a random name for test
    """
    alphabet = string.ascii_letters
    random_string = "".join(random.choice(alphabet) for i in range(4))
    schema_name = f"test_{random_string}"
    verticapy.create_schema(schema_name)
    yield schema_name
    verticapy.drop(schema_name, method="schema")


@pytest.fixture(scope="session")
def dummy_vd():
    """
    Create a dummy vDataFrame
    """
    arr1 = np.concatenate((np.ones(60), np.zeros(40))).astype(int)
    np.random.shuffle(arr1)
    arr2 = np.concatenate((np.repeat("A", 20), np.repeat("B", 30), np.repeat("C", 50)))
    np.random.shuffle(arr2)
    dummy = verticapy.vDataFrame(list(zip(arr1, arr2)), ["check 1", "check 2"])
    yield dummy


@pytest.fixture(scope="session")
def dummy_scatter_vd():
    """
    Create a dummy vDataFrame that is suitable for scatter plot
    """
    data_size = DUMMY_TEST_SIZE
    slopes_y_z = [10, 5]
    y_intercept = -20
    z_intercept = 20
    scatter_magnitude_y = 4
    scatter_magnitude_z = 40
    x_val = np.linspace(0, 10, data_size)
    y_val = (
        y_intercept
        + slopes_y_z[0] * x_val
        + np.random.randn(data_size) * scatter_magnitude_y
    )
    z_val = (
        z_intercept
        + slopes_y_z[1] * x_val
        + np.random.randn(data_size) * scatter_magnitude_z
    )
    percentage_a = 0.4
    percentage_b = 0.3
    percentage_c = 1 - (percentage_a + percentage_b)
    categories = ["A", "B", "C"]
    category_array = np.random.choice(
        categories, size=data_size, p=[percentage_a, percentage_b, percentage_c]
    )
    dummy = verticapy.vDataFrame(
        {"X": x_val, "Y": y_val, "Z": z_val, "Category": category_array.tolist()}
    )
    yield dummy


@pytest.fixture(scope="session")
def dummy_date_vd():
    """
    Create a dummy vDataFrame that has date data
    """
    data_size = DUMMY_TEST_SIZE
    median = 500
    qrtr_1 = 200
    qrtr_3 = 800
    std = (qrtr_3 - qrtr_1) / (2 * np.sqrt(2) * erfinv(0.5))
    data = np.random.normal(median, std, data_size)
    dummy = pd.DataFrame(
        {
            "date": [1910, 1920, 1930, 1940, 1950] * int(data_size / 5),
            "value": list(data),
        }
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        dummy = verticapy.read_pandas(dummy, temp_path=temp_dir)
    yield dummy


@pytest.fixture(scope="session")
def dummy_probability_data():
    """
    Create a dummy vDataFrame that has probability data
    """
    count = DUMMY_TEST_SIZE
    first_count = 10
    second_count = 40
    third_count = count - first_count - second_count
    prob_1_first = 0
    prob_1_second = 0.4
    prob_1_third = 0.9
    pred = list(
        np.random.choice([0, 1], size=first_count, p=[1 - prob_1_first, prob_1_first])
    )
    pred.extend(
        np.random.choice(
            [0, 1], size=second_count, p=[1 - prob_1_second, prob_1_second]
        )
    )
    pred.extend(
        np.random.choice([0, 1], size=third_count, p=[1 - prob_1_third, prob_1_third])
    )
    prob = np.linspace(0, 1, count)
    dummy = verticapy.vDataFrame({"y_true": prob, "y_score": pred})
    yield dummy


@pytest.fixture(scope="session")
def dummy_dist_vd():
    """
    Create a dummy vDataFrame
    """
    data_size = DUMMY_TEST_SIZE
    ones_percentage = 0.4
    median = 50
    qrtrs_1_3 = [40, 60]
    percentages = [0.4, 0.3, 0.3]
    categories = ["A", "B", "C"]
    category_array = np.random.choice(
        categories, size=data_size, p=[percentages[0], percentages[1], percentages[2]]
    )
    category_array = category_array.reshape(len(category_array), 1)
    result_array = np.concatenate(
        (
            np.zeros(int(data_size * (1 - ones_percentage))),
            np.ones(int(data_size * ones_percentage)),
        )
    )
    np.random.shuffle(result_array)
    result_array = result_array.reshape(len(result_array), 1)
    std = (qrtrs_1_3[1] - qrtrs_1_3[0]) / (2 * np.sqrt(2) * erfinv(0.5))
    data = np.random.normal(median, std, data_size)
    data_2 = np.random.normal(median + 10, std + 5, data_size)
    data = data.reshape(len(data), 1)
    data_2 = data_2.reshape(len(data_2), 1)
    data[-1] = data.max() + 15
    data_2[-1] = data_2.max() + 15 + 10
    data_all = pd.DataFrame(
        np.concatenate((data, data_2, result_array, category_array), axis=1),
        columns=["0", "1", "binary", "cats"],
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        dummy = verticapy.read_pandas(data_all, temp_path=temp_dir)
    dummy["binary"].astype("int")
    yield dummy


@pytest.fixture(scope="session")
def dummy_line_data_vd():
    """
    Create a dummy vDataFrame that has time data
    """
    data_size = int(DUMMY_TEST_SIZE / 10)
    start_year = 1900
    step = 10
    num_values = data_size
    years = [start_year + i * step for i in range(num_values)] * 2
    values = [random.randint(1, 100) for _ in range(data_size * 2)]
    category = ["A"] * data_size + ["B"] * data_size
    yield verticapy.vDataFrame({"date": years, "values": values, "category": category})


@pytest.fixture(scope="session")
def dummy_pred_data_vd():
    """
    Create a dummy vDataFrame that has prediction as a column
    """
    data_size = DUMMY_TEST_SIZE
    x_val = np.linspace(0, 10, data_size)
    y_val = np.linspace(0, 10, data_size)
    z_val = np.linspace(0, 10, data_size)
    pred = [0 if x_val[i] + y_val[i] + z_val[i] > 20 else 1 for i in range(len(x_val))]
    yield verticapy.vDataFrame({"X": x_val, "Y": y_val, "Z": z_val, "Category": pred})


@pytest.fixture(scope="module")
def titanic_vd(schema_loader):
    """
    Create a dummy vDataFrame for titanic dataset
    """
    titanic = load_titanic(schema_loader, "titanic")
    yield titanic
    drop(name=f"{schema_loader}.titanic")


@pytest.fixture(scope="module")
def iris_vd(schema_loader):
    """
    Create a dummy vDataFrame for iris dataset
    """
    iris = load_iris(schema_loader, "iris")
    yield iris
    drop(name=f"{schema_loader}.iris")


@pytest.fixture(scope="module")
def amazon_vd(schema_loader):
    """
    Create a dummy vDataFrame for amazon dataset
    """
    amazon = load_amazon(schema_loader, "amazon")
    yield amazon
    drop(name=f"{schema_loader}.amazon")


@pytest.fixture(scope="module")
def pred_cl_dataset_multi():
    """
    Create a dummy vDataFrame
    """
    labels = np.array(["a", "b", "c"])
    y_true = np.array(
        [
            "a",
            "a",
            "b",
            "c",
            "c",
            "a",
            "b",
            "c",
            "a",
            "b",
            "c",
            "a",
            "b",
            "a",
            "b",
            "c",
            "a",
            "b",
            "c",
        ]
    )
    y_pred = np.array(
        [
            "a",
            "b",
            "b",
            "b",
            "c",
            "a",
            "b",
            "a",
            "a",
            "c",
            "a",
            "a",
            "b",
            "a",
            "b",
            "c",
            "b",
            "b",
            "a",
        ]
    )
    labels_num = np.array([0, 1, 2])
    y_true_num = np.array([random.randint(0, 2) for _ in range(len(y_true))])
    y_pred_num = np.array([random.randint(0, 2) for _ in range(len(y_true))])

    _y_prob = np.array(
        [
            [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
            for _ in range(len(y_true))
        ]
    )
    # for some of the sklearn metrics (like roc_auc_score), probabilities should be sum up to 1 for multiclass
    y_prob = []
    for i in _y_prob:
        row = [j / sum(i) for j in i]
        y_prob.append(row)
    y_prob = np.array(y_prob)

    input_relation = np.column_stack(
        (
            y_true,
            y_pred,
            y_true_num,
            y_pred_num,
            y_prob[:, 0],
            y_prob[:, 1],
            y_prob[:, 2],
        )
    )
    vdf = vDataFrame(
        input_relation,
        usecols=[
            "y_true",
            "y_pred",
            "y_true_num",
            "y_pred_num",
            "y_prob0",
            "y_prob1",
            "y_prob2",
        ],
    )
    yield vdf, y_true, y_pred, y_prob, labels, y_true_num, y_pred_num, labels_num


@pytest.fixture(scope="module")
def pred_cl_dataset_binary():
    """
    Create a dummy vDataFrame
    """
    labels = np.array(["a", "b"])
    y_true = np.array(
        [
            "a",
            "a",
            "b",
            "b",
            "a",
            "a",
            "b",
            "b",
            "a",
            "b",
            "a",
            "a",
            "b",
            "a",
            "b",
            "b",
            "a",
            "b",
            "a",
        ]
    )
    y_pred = np.array(
        [
            "a",
            "b",
            "b",
            "b",
            "a",
            "a",
            "b",
            "a",
            "a",
            "b",
            "a",
            "a",
            "b",
            "a",
            "b",
            "a",
            "b",
            "b",
            "a",
        ]
    )

    labels_num = np.array([0, 1])
    y_true_num = np.array([random.randint(0, 1) for _ in range(len(y_true))])
    y_pred_num = np.array([random.randint(0, 1) for _ in range(len(y_true))])

    y_prob = np.array([random.uniform(0, 1) for _ in range(len(y_true))])

    input_relation = np.column_stack((y_true, y_pred, y_true_num, y_pred_num, y_prob))
    vdf = vDataFrame(
        input_relation,
        usecols=["y_true", "y_pred", "y_true_num", "y_pred_num", "y_prob"],
    )
    yield vdf, y_true, y_pred, y_prob, labels, y_true_num, y_pred_num, labels_num


@pytest.fixture(scope="module")
def pred_cl_dataset_multilevel():
    """
    Create a dummy vDataFrame
    """
    labels = np.array(["c1", "c2", "c3", "c4"])
    y_true = np.array([[0, 1, 1], [1, 1, 1], [0, 1, 0], [1, 0, 1]])
    y_pred = np.array([[0, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]])
    input_relation = np.column_stack((y_true, y_pred))
    vdf = vDataFrame(input_relation, usecols=["y_t", "y_s"])
    yield vdf, y_true, y_pred, labels


@pytest.fixture(scope="module")
def winequality_vpy(schema_loader):
    with winequality_vpy_main(schema_loader) as result:
        yield result


@pytest.fixture(scope="function")
def winequality_vpy_fun(schema_loader):
    with winequality_vpy_main(schema_loader) as result:
        yield result


# @pytest.fixture(scope="function")
@contextmanager
def winequality_vpy_main(schema_loader):
    """
    Create a dummy vDataFrame for winequality data
    """
    winequality = load_winequality(schema_loader, "winequality")
    yield winequality
    drop(
        name=f"{schema_loader}.winequality",
    )


@pytest.fixture(scope="function")
def titanic_vd_fun(schema_loader):
    """
    Create a dummy vDataFrame for titanic dataset
    """
    titanic = load_titanic(schema_loader, "titanic")
    yield titanic
    drop(name=f"{schema_loader}.titanic")


@pytest.fixture(scope="module")
def market_vd(schema_loader):
    """
    Create a dummy vDataFrame for market dataset
    """
    market = load_market(schema_loader, "market")
    yield market
    drop(name=f"{schema_loader}.market")


@pytest.fixture(scope="module")
def smart_meters_vd(schema_loader):
    """
    Create a dummy vDataFrame for smart_meters dataset
    """
    smart_meters = load_smart_meters(schema_loader, "smart_meters")
    yield smart_meters
    drop(name=f"{schema_loader}.smart_meters")


@pytest.fixture(scope="function")
def iris_vd_fun(schema_loader):
    """
    Create a dummy vDataFrame for iris dataset
    """
    iris = load_iris(schema_loader, "iris")
    yield iris
    drop(name=f"{schema_loader}.iris")


@pytest.fixture(scope="module")
def laliga_vd(schema_loader):
    """
    Create a dummy vDataFrame for laliga dataset
    """
    laliga = load_laliga(schema_loader, "laliga")
    yield laliga
    drop(name=f"{schema_loader}.laliga")


@pytest.fixture(scope="module")
def airline_vd(schema_loader):
    """
    Create a dummy vDataFrame for airline_passengers dataset
    """
    airline = load_airline_passengers(schema_loader, "airline")
    yield airline
    drop(name=f"{schema_loader}.airline")


@pytest.fixture(scope="function")
def airline_vd_fun(schema_loader):
    """
    Create a dummy vDataFrame for airline_passengers dataset
    """
    airline = load_airline_passengers(schema_loader, "airline")
    yield airline
    drop(name=f"{schema_loader}.airline")
