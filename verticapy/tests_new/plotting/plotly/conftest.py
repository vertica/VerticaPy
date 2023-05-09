# Pytest
import pytest

# Standard Python Modules
import random
import tempfile
import string

# Other Modules
import scipy
import numpy as np
import pandas as pd
import plotly

# VerticaPy
import verticapy
import verticapy._config.config as conf
from verticapy import drop
from verticapy.datasets import load_titanic, load_iris, load_amazon
from verticapy.learn.preprocessing import OneHotEncoder

# # TODO - CHECK IF THE BELOW IS OPTIMAL FOR ALL CODES OR SHALL I NOT JUST DROP THE SCHEMA
# @pytest.fixture(scope="session", autouse=True)
# def load_test_schema(tmp_path_factory, worker_id):
#     if worker_id == "master":
#         # not executing with multiple workers, just create the schema and let pytest's fixture caching do its job
#         verticapy.create_schema("test")
#         yield
#         verticapy.drop("test", method="schema")
#     else:
#         # get the temp directory shared by all workers
#         root_tmp_dir = tmp_path_factory.getbasetemp().parent

#         # acquire a file lock to ensure that only one worker creates the schema
#         fn = root_tmp_dir / "schema_created"
#         with FileLock(str(fn) + ".lock"):
#             if not fn.is_file():
#                 verticapy.create_schema("test")
#                 fn.write_text("created")
#             yield
#             # only the worker that created the schema should drop it
#             if fn.is_file():
#                 verticapy.drop("test", method="schema")
#                 fn.unlink()

# @pytest.fixture(scope="session", autouse=True)
# def load_test_schema(tmp_path_factory, worker_id):
#     # get the temp directory shared by all workers
#     root_tmp_dir = tmp_path_factory.getbasetemp().parent

#     # acquire a file lock to ensure that only one worker creates the schema
#     fn = root_tmp_dir / "schema_created"
#     with FileLock(str(fn) + ".lock"):
#         if not fn.is_file():
#             # only the first worker that acquires the lock creates the schema
#             verticapy.create_schema("test")
#             fn.write_text("created")

#     # wait for the schema to be created by the first worker
#     while not verticapy.schema.Schema("test").exists():
#         pass

#     yield

#     # only the worker that created the schema should drop it
#     if fn.is_file():
#         verticapy.drop("test", method="schema")
#         fn.unlink()

DUMMY_TEST_SIZE = 100


@pytest.fixture(scope="session", autouse=True)
def load_test_schema():
    alphabet = string.ascii_letters
    random_string = "".join(random.choice(alphabet) for i in range(4))
    schema_name = f"test_{random_string}"
    verticapy.create_schema(schema_name)
    yield schema_name
    verticapy.drop(schema_name, method="schema")


@pytest.fixture(scope="session", autouse=True)
def load_plotly():
    conf.set_option("plotting_lib", "plotly")
    yield
    conf.set_option("plotting_lib", "matplotlib")


@pytest.fixture(scope="session")
def plotly_figure_object():
    yield plotly.graph_objs._figure.Figure


@pytest.fixture(scope="session")
def dummy_vd():
    arr1 = np.concatenate((np.ones(60), np.zeros(40))).astype(int)
    np.random.shuffle(arr1)
    arr2 = np.concatenate((np.repeat("A", 20), np.repeat("B", 30), np.repeat("C", 50)))
    np.random.shuffle(arr2)
    dummy = verticapy.vDataFrame(list(zip(arr1, arr2)), ["check 1", "check 2"])
    yield dummy


@pytest.fixture(scope="session")
def dummy_scatter_vd():
    N = DUMMY_TEST_SIZE
    slope_y = 10
    slope_z = 5
    y_intercept = -20
    z_intercept = 20
    scatter_magnitude_y = 4
    scatter_magnitude_z = 40
    x = np.linspace(0, 10, N)
    y = y_intercept + slope_y * x + np.random.randn(N) * scatter_magnitude_y
    z = z_intercept + slope_z * x + np.random.randn(N) * scatter_magnitude_z
    percentage_A = 0.4
    percentage_B = 0.3
    percentage_C = 1 - (percentage_A + percentage_B)
    categories = ["A", "B", "C"]
    category_array = np.random.choice(
        categories, size=N, p=[percentage_A, percentage_B, percentage_C]
    )
    dummy = verticapy.vDataFrame(
        {"X": x, "Y": y, "Z": z, "Category": category_array.tolist()}
    )
    yield dummy


@pytest.fixture(scope="session")
def dummy_date_vd():
    N = DUMMY_TEST_SIZE
    years = [1910, 1920, 1930, 1940, 1950]
    median = 500
    q1 = 200
    q3 = 800
    std = (q3 - q1) / (2 * np.sqrt(2) * scipy.special.erfinv(0.5))
    data = np.random.normal(median, std, N)
    dummy = pd.DataFrame(
        {
            "date": [1910, 1920, 1930, 1940, 1950] * int(N / 5),
            "value": list(data),
        }
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        dummy = verticapy.read_pandas(dummy, temp_path=temp_dir)
    yield dummy


@pytest.fixture(scope="session")
def dummy_probability_data():
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
    N = DUMMY_TEST_SIZE
    ones_percentage = 0.4
    median = 50
    q1 = 40
    q3 = 60
    percentage_A = 0.4
    percentage_B = 0.3
    percentage_C = 1 - (percentage_A + percentage_B)
    categories = ["A", "B", "C"]
    category_array = np.random.choice(
        categories, size=N, p=[percentage_A, percentage_B, percentage_C]
    )
    category_array = category_array.reshape(len(category_array), 1)
    zeros_array = np.zeros(int(N * (1 - ones_percentage)))
    ones_array = np.ones(int(N * ones_percentage))
    result_array = np.concatenate((zeros_array, ones_array))
    np.random.shuffle(result_array)
    result_array = result_array.reshape(len(result_array), 1)
    std = (q3 - q1) / (2 * np.sqrt(2) * scipy.special.erfinv(0.5))
    data = np.random.normal(median, std, N)
    data_2 = np.random.normal(median + 10, std + 5, N)
    data = data.reshape(len(data), 1)
    data_2 = data_2.reshape(len(data_2), 1)
    data[-1] = data.max() + 15
    data_2[-1] = data_2.max() + 15 + 10
    cols_combined = np.concatenate((data, data_2, result_array, category_array), axis=1)
    data_all = pd.DataFrame(cols_combined, columns=["0", "1", "binary", "cats"])
    with tempfile.TemporaryDirectory() as temp_dir:
        dummy = verticapy.read_pandas(data_all, temp_path=temp_dir)
    dummy["binary"].astype("int")
    yield dummy


@pytest.fixture(scope="session")
def dummy_line_data_vd():
    N = int(DUMMY_TEST_SIZE / 10)
    start_year = 1900
    step = 10
    num_values = N
    import random

    years = [start_year + i * step for i in range(num_values)] * 2
    values = [random.randint(1, 100) for _ in range(N * 2)]
    category = ["A"] * N + ["B"] * N
    yield verticapy.vDataFrame({"date": years, "values": values, "category": category})


@pytest.fixture(scope="session")
def dummy_pred_data_vd():
    N = DUMMY_TEST_SIZE
    slope_y = 10
    slope_z = 5
    y_intercept = -20
    z_intercept = 20
    scatter_magnitude_y = 4
    scatter_magnitude_z = 40
    x = np.linspace(0, 10, N)
    y = np.linspace(0, 10, N)
    z = np.linspace(0, 10, N)
    pred = [0 if x[i] + y[i] + z[i] > 20 else 1 for i in range(len(x))]
    yield verticapy.vDataFrame({"X": x, "Y": y, "Z": z, "Category": pred})


@pytest.fixture(scope="session")
def titanic_vd(load_test_schema):
    titanic = load_titanic(load_test_schema, "titanic")
    yield titanic
    drop(name=f"{load_test_schema}.titanic")


@pytest.fixture(scope="session")
def iris_vd(load_test_schema):
    iris = load_iris(load_test_schema, "iris")
    yield iris
    drop(name=f"{load_test_schema}.iris")


@pytest.fixture(scope="session")
def amazon_vd(load_test_schema):
    amazon = load_amazon(load_test_schema, "amazon")
    yield amazon
    drop(name=f"{load_test_schema}.amazon")
