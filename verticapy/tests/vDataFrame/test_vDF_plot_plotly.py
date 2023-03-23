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

# Pytest
import pytest

# Standard Python Modules
import datetime, os, sys

# Other Modules
import plotly.express as px
import plotly
import pandas as pd
import numpy as np

# VerticaPy
import verticapy
import verticapy._config.config as conf
from verticapy import drop
from verticapy.datasets import load_titanic, load_iris

conf.set_option("print_info", False)


@pytest.fixture(scope="module")
def dummy_vd():
    arr1 = np.concatenate((np.ones(60), np.zeros(40))).astype(int)
    np.random.shuffle(arr1)
    arr2 = np.concatenate((np.repeat("A", 20), np.repeat("B", 30), np.repeat("C", 50)))
    np.random.shuffle(arr2)
    dummy = verticapy.vDataFrame(list(zip(arr1, arr2)), ["check 1", "check 2"])
    yield dummy


@pytest.fixture(scope="module")
def titanic_vd():
    titanic = load_titanic()
    yield titanic
    drop(name="public.titanic")


@pytest.fixture(scope="module")
def iris_vd():
    iris = load_iris()
    yield iris
    drop(name="public.iris")


@pytest.fixture(scope="module")
def load_plotly():
    conf.set_option("plotting_lib", "plotly")
    yield
    conf.set_option("plotting_lib", "matplotlib")


class TestvDFPlotPlotly:
    def test_vDF_bar(self, titanic_vd, load_plotly):
        # 1D bar charts

        ## Checking plotting library
        assert conf.get_option("plotting_lib") == "plotly"
        survived_values = titanic_vd.to_pandas()["survived"]

        ## Creating a test figure to compare
        test_fig = px.bar(
            x=[0, 1],
            y=[
                survived_values[survived_values == 0].count(),
                survived_values[survived_values == 1].count(),
            ],
        )
        test_fig = test_fig.update_xaxes(type="category")
        result = titanic_vd["survived"].bar()

        ## Testing Plot Properties
        ### Checking if correct object is created
        assert type(result) == plotly.graph_objs._figure.Figure
        ### Checking if the x-axis is a category instead of integer
        assert result.layout["xaxis"]["type"] == "category"

        ## Testing Data
        ### Comparing result with a test figure
        assert result.data[0]['y'][np.where(result.data[0]['x']=='0')[0][0]]/result.data[0]['y'][np.where(result.data[0]['x']=='1')[0][0]] == test_fig.data[0]['y'][np.where(test_fig.data[0]['x']==0)[0][0]]/test_fig.data[0]['y'][np.where(test_fig.data[0]['x']==1)[0][0]]
        
        ## Testing Additional Options
        ### Testing keyword arguments (kwargs)
        result = titanic_vd["survived"].bar(xaxis_title="Custom X Axis Title")
        assert result.layout["xaxis"]["title"]["text"] == "Custom X Axis Title"
        result = titanic_vd["survived"].bar(yaxis_title="Custom Y Axis Title")
        assert result.layout["yaxis"]["title"]["text"] == "Custom Y Axis Title"

    def test_vDF_pie(self, dummy_vd, load_plotly):
        # 1D pie charts

        ## Creating a pie chart
        result = dummy_vd["check 1"].pie()

        ## Testing Data
        ### check value corresponding to 0s
        assert (
            result.data[0]["values"][0]
            == dummy_vd[dummy_vd["check 1"] == 0]["check 1"].count()
            / dummy_vd["check 1"].count()
        )
        ### check value corresponding to 1s
        assert (
            result.data[0]["values"][1]
            == dummy_vd[dummy_vd["check 1"] == 1]["check 1"].count()
            / dummy_vd["check 1"].count()
        )

        ## Testing Plot Properties
        ### checking the label
        assert result.data[0]["labels"] == ("0", "1")
        ### check title
        assert result.layout["title"]["text"] == "check 1"

        ## Testing Additional Options
        ### check hole option
        result = dummy_vd["check 1"].pie(pie_type="donut")
        assert result.data[0]["hole"] == 0.2
        ### check exploded option
        result = dummy_vd["check 1"].pie(exploded=True)
        assert len(result.data[0]["pull"]) > 0

    def test_vDF_barh(self, dummy_vd, load_plotly):
        # 1D horizontal bar charts

        ## Creating horizontal bar chart
        result = dummy_vd["check 2"].barh()
        ## Testing Plot Properties
        ### Checking if correct object is created
        assert type(result) == plotly.graph_objs._figure.Figure
        ### Checking if the x-axis is a category instead of integer
        assert result.layout["yaxis"]["type"] == "category"

        ## Testing Data
        ### Comparing total adds up to 1
        assert sum(result.data[0]["x"]) == 1
        ### Checking if all labels are inlcuded
        assert set(result.data[0]["y"]).issubset(set(["A", "B", "C"]))
        ### Checking if the density was plotted correctly
        nums = dummy_vd.to_pandas()["check 2"].value_counts()
        total = len(dummy_vd)
        assert set(result.data[0]["x"]).issubset(
            set([nums["A"] / total, nums["B"] / total, nums["C"] / total])
        )

        ## Testing Additional Options
        ### Testing keyword arguments (kwargs)
        result = dummy_vd["check 2"].barh(xaxis_title="Custom X Axis Title")
        assert result.layout["xaxis"]["title"]["text"] == "Custom X Axis Title"
        result = dummy_vd["check 2"].barh(yaxis_title="Custom Y Axis Title")
        assert result.layout["yaxis"]["title"]["text"] == "Custom Y Axis Title"


class TestVDFNestedPieChart:
    def test_properties_type(self, load_plotly, dummy_vd):
        # Arrange
        # Act
        result = dummy_vd.pie(["check 1", "check 2"])
        # Assert - checking if correct object created
        assert type(result) == plotly.graph_objs._figure.Figure

    def test_properties_branch_values(self, load_plotly, dummy_vd):
        # Arrange
        # Act
        result = dummy_vd.pie(["check 1", "check 2"])
        # Assert - checking if the branch values are covering all
        assert result.data[0]["branchvalues"] == "total"

    def test_data_all_labels_for_nested(self, load_plotly, dummy_vd):
        # Arrange
        result = dummy_vd.pie(["check 1", "check 2"])
        # Act
        # Assert - checking if all the labels exist
        assert set(result.data[0]["labels"]) == {"1", "0", "A", "B", "C"}

    def test_data_all_labels_for_simple_pie(self, load_plotly, dummy_vd):
        # Arrange
        result = dummy_vd.pie(["check 1"])
        # Act
        # Assert - checking if all the labels exist for a simple pie plot
        assert set(result.data[0]["labels"]) == {"1", "0"}

    def test_data_check_parent_of_A(self, load_plotly, dummy_vd):
        # Arrange
        result = dummy_vd.pie(["check 1", "check 2"])
        # Act
        # Assert - checking the parent of 'A' which is an element of column "check 2"
        assert result.data[0]["parents"][result.data[0]["labels"].index("A")] in [
            "0",
            "1",
        ]

    def test_data_check_parent_of_0(self, load_plotly, dummy_vd):
        # Arrange
        result = dummy_vd.pie(["check 1", "check 2"])
        # Act
        # Assert - checking the parent of '0' which is an element of column "check 1"
        assert result.data[0]["parents"][result.data[0]["labels"].index("0")] in [""]

    def test_data_add_up_all_0s_from_children(self, load_plotly, dummy_vd):
        # Arrange
        # Act
        result = dummy_vd.pie(["check 1", "check 2"])
        zero_indices = [i for i, x in enumerate(result.data[0]["parents"]) if x == "0"]
        # Assert - checking if if all the children elements of 0 add up to its count
        assert sum([list(result.data[0]["values"])[i] for i in zero_indices]) == 40


class TestVDFScatterPlot:
    def test_properties_output_type(self, load_plotly, iris_vd):
        # Arrange
        # Act
        result = iris_vd.scatter(["SepalWidthCm", "SepalLengthCm"])
        # Assert - checking if correct object created
        assert type(result) == plotly.graph_objs._figure.Figure, "wrong object crated"

    def test_properties_xaxis_title(self, load_plotly, iris_vd):
        # Arrange
        # Act
        result = iris_vd.scatter(["SepalWidthCm", "SepalLengthCm"])
        # Assert
        assert (
            result.layout["xaxis"]["title"]["text"] == "SepalWidthCm"
        ), "X-axis title issue"

    def test_properties_yaxis_title(self, load_plotly, iris_vd):
        # Arrange
        # Act
        result = iris_vd.scatter(["SepalWidthCm", "SepalLengthCm"])
        # Assert
        assert (
            result.layout["yaxis"]["title"]["text"] == "SepalLengthCm"
        ), "Y-axis title issue"

    def test_properties_xaxis_title_3D_plot(self, load_plotly, iris_vd):
        # Arrange
        # Act
        result = iris_vd.scatter(["SepalWidthCm", "SepalLengthCm", "PetalLengthCm"])
        # Assert
        assert (
            result.layout["scene"]["xaxis"]["title"]["text"] == "SepalWidthCm"
        ), "X-axis title issue in 3D plot"

    def test_properties_yaxis_title_3D_plot(self, load_plotly, iris_vd):
        # Arrange
        # Act
        result = iris_vd.scatter(["SepalWidthCm", "SepalLengthCm", "PetalLengthCm"])
        # Assert
        assert (
            result.layout["scene"]["yaxis"]["title"]["text"] == "SepalLengthCm"
        ), "Y-axis title issue in 3D plot"

    def test_properties_zaxis_title_3D_plot(self, load_plotly, iris_vd):
        # Arrange
        # Act
        result = iris_vd.scatter(["SepalWidthCm", "SepalLengthCm", "PetalLengthCm"])
        # Assert
        assert (
            result.layout["scene"]["zaxis"]["title"]["text"] == "PetalLengthCm"
        ), "Z-axis title issue in 3D plot"

    def test_properties_all_unique_values_for_catcol(self, load_plotly, iris_vd):
        # Arrange
        # Act
        result = iris_vd.scatter(["PetalWidthCm", "PetalLengthCm",], catcol="Species")
        # Assert
        assert set(
            [result.data[0]["name"], result.data[1]["name"], result.data[2]["name"]]
        ).issubset(
            set(["Iris-virginica", "Iris-versicolor", "Iris-setosa"])
        ), "Some unique values were not found in the plot"

    def test_properties_all_unique_values_for_catcol_3D_plot(
        self, load_plotly, iris_vd
    ):
        # Arrange
        # Act
        result = iris_vd.scatter(
            ["PetalWidthCm", "PetalLengthCm", "SepalLengthCm"], catcol="Species"
        )
        # Assert
        assert set(
            [result.data[0]["name"], result.data[1]["name"], result.data[2]["name"]]
        ).issubset(
            set(["Iris-virginica", "Iris-versicolor", "Iris-setosa"])
        ), "Some unique values were not found in the 3D plot"

    def test_properties_colors_for_catcol(self, load_plotly, iris_vd):
        # Arrange
        # Act
        result = iris_vd.scatter(["PetalWidthCm", "PetalLengthCm",], catcol="Species")
        assert (
            len(
                set(
                    [
                        result.data[0]["marker"]["color"],
                        result.data[1]["marker"]["color"],
                        result.data[2]["marker"]["color"],
                    ]
                )
            )
            == 3
        ), "Colors are not unique for three different cat_col parameter"

    def test_properties_colors_for_catcol_3D_plot(self, load_plotly, iris_vd):
        # Arrange
        # Act
        result = iris_vd.scatter(
            ["PetalWidthCm", "PetalLengthCm", "SepalLengthCm"], catcol="Species"
        )
        assert (
            len(
                set(
                    [
                        result.data[0]["marker"]["color"],
                        result.data[1]["marker"]["color"],
                        result.data[2]["marker"]["color"],
                    ]
                )
            )
            == 3
        ), "Colors are not unique for three different cat_col parameter"

    def test_data_total_number_of_points(self, load_plotly, iris_vd):
        # Arrange
        # Act
        result = iris_vd.scatter(["SepalWidthCm", "SepalLengthCm"])
        # Assert - checking if correct object created
        assert len(result.data[0]["x"]) == len(
            iris_vd
        ), "Number of points not consistent with data"
        assert len(result.data[0]["y"]) == len(
            iris_vd
        ), "Number of points not consistent with data"

    def test_data_total_number_of_points_3D_plot(self, load_plotly, iris_vd):
        # Arrange
        # Act
        result = iris_vd.scatter(["PetalWidthCm", "PetalLengthCm", "SepalLengthCm"])
        # Assert - checking if correct object created
        assert len(result.data[0]["x"]) == len(
            iris_vd
        ), "Number of points not consistent with data"
        assert len(result.data[0]["y"]) == len(
            iris_vd
        ), "Number of points not consistent with data"
        assert len(result.data[0]["z"]) == len(
            iris_vd
        ), "Number of points not consistent with data"

    def test_data_random_point_from_plot_in_data(self, load_plotly, iris_vd):
        # Arrange
        # Act
        result = iris_vd.scatter(["SepalWidthCm", "SepalLengthCm"])
        # Assert -
        len_of_data = len(
            iris_vd.search(
                conditions=[
                    f"SepalWidthCm ={result.data[0]['x'][0]} and SepalLengthCm={result.data[0]['y'][0]}"
                ],
                usecols=["SepalWidthCm", "SepalLengthCm"],
            )
        )
        assert len_of_data > 0, "A wrong point was plotted"

    def test_data_random_point_from_data_in_plot(self, load_plotly, iris_vd):
        # Arrange
        sample = iris_vd.sample(n=1)
        x_test = sample["SepalWidthCm"][0]
        y_test = sample["SepalLengthCm"][0]
        # Act
        result = iris_vd.scatter(["SepalWidthCm", "SepalLengthCm"])
        # Assert -
        assert (
            y_test in result.data[0]["y"][np.where(result.data[0]["x"] == x_test)[0]]
        ), "A random sample datapoint was not plotted"
