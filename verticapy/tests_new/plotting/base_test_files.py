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

from abc import abstractmethod
import pytest

# Vertica
from verticapy.learn.model_selection import elbow
from verticapy.learn.ensemble import RandomForestClassifier
from verticapy.learn.neighbors import LocalOutlierFactor
from verticapy.learn.linear_model import LogisticRegression
from verticapy.learn.model_selection import lift_chart, prc_curve
from verticapy.learn.decomposition import PCA
from verticapy.learn.tree import DecisionTreeRegressor
from verticapy.learn.linear_model import LinearRegression
from verticapy.learn.model_selection import stepwise
from verticapy.learn.svm import LinearSVC
from verticapy.learn.cluster import KMeans
from vertica_highcharts.highcharts.highcharts import Highchart

# Standard Python Modules
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def get_xaxis_label(obj):
    """
    Get x-axis label for given plotting object
    """
    if isinstance(obj, plt.Axes):
        return obj.get_xlabel()
    if isinstance(obj, go.Figure):
        return obj.layout["xaxis"]["title"]["text"]
    if isinstance(obj, Highchart):
        return obj.options["xAxis"].title.text
    return None


def get_yaxis_label(obj):
    """
    Get y-axis label for given plotting object
    """
    if isinstance(obj, plt.Axes):
        return obj.get_ylabel()
    if isinstance(obj, go.Figure):
        return obj.layout["yaxis"]["title"]["text"]
    if isinstance(obj, Highchart):
        return obj.options["yAxis"].title.text
    return None


def get_zaxis_label(obj):
    """
    Get z-axis label for given plotting object
    """
    if isinstance(obj, plt.Axes):
        return obj.get_zlabel()
    if isinstance(obj, go.Figure):
        return obj.layout["scene"]["zaxis"]["title"]["text"]
    if isinstance(obj, Highchart):
        return obj.options["zAxis"].title.text
    return None


def get_width(obj):
    """
    Get width for given plotting object
    """
    if isinstance(obj, plt.Axes):
        return obj.get_figure().get_size_inches()[0]
    if isinstance(obj, go.Figure):
        return obj.layout["width"]
    if isinstance(obj, Highchart):
        return obj.options["chart"].width
    return None


def get_height(obj):
    """
    Get height for given plotting object
    """
    if isinstance(obj, plt.Axes):
        return obj.get_figure().get_size_inches()[1]
    if isinstance(obj, go.Figure):
        return obj.layout["height"]
    if isinstance(obj, Highchart):
        return obj.options["chart"].height
    return None


def get_title(obj):
    """
    Get title for given plotting object
    """
    if isinstance(obj, plt.Axes):
        return obj.get_title()
    if isinstance(obj, go.Figure):
        return obj.layout["title"]["text"]
    if isinstance(obj, Highchart):
        return obj.options["title"].text
    return None


class BasicPlotTests:
    """
    Basic Tests for all plots
    """

    cols = []

    # @property
    @abstractmethod
    def create_plot(self):
        """
        Abstract method to create the plot
        """

    @property
    def result(self):
        """
        Create the plot
        """
        func, arg = self.create_plot()
        return func(**arg)

    def test_properties_output_type(self, plotting_library_object):
        """
        Test if correct object created
        """
        assert isinstance(self.result, plotting_library_object), "wrong object crated"

    def test_properties_xaxis_label(self):
        """
        Testing x-axis label
        """
        test_title = self.cols[0]
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_label(self):
        """
        Testing y-axis title
        """
        test_title = self.cols[1]
        assert get_yaxis_label(self.result) == test_title, "Y axis label incorrect"

    def test_properties_zaxis_label(self):
        """
        Testing y-axis title
        """
        if len(self.cols) > 2:
            test_title = self.cols[2]
            assert get_zaxis_label(self.result) == test_title, "Z axis label incorrect"
        else:
            pass

    def test_additional_options_custom_width_and_height(self):
        """
        Test custom width and height
        """
        func, arg = self.create_plot()
        custom_width = 300
        custom_height = 400
        result = func(width=custom_width, height=custom_height, **arg)
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"


class ACFPlot(BasicPlotTests):
    """
    Testing different attributes of ACF plot on a vDataFrame
    """

    @pytest.fixture(autouse=True)
    def data(self, amazon_vd):
        """
        Load test data
        """
        self.data = amazon_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return ["lag", "value"]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data.acf,
            {
                "ts": "date",
                "column": "number",
                "p": 12,
                "by": ["state"],
                "unit": "month",
                "method": "spearman",
            },
        )


class VDCBarPlot(BasicPlotTests):
    """
    Testing different attributes of Bar plot on a vDataColumn
    """

    # Testing variables
    COL_NAME = "check 2"

    @pytest.fixture(autouse=True)
    def data(self, dummy_vd):
        """
        Load test data
        """
        self.data = dummy_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [self.COL_NAME, "density"]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data[self.COL_NAME].bar,
            {},
        )

    @pytest.mark.parametrize("max_cardinality, bargap", [(1, 0.1), (4, 0.4)])
    def test_properties_output_type_for_all_options(
        self,
        dummy_vd,
        plotting_library_object,
        bargap,
        max_cardinality,
    ):
        """
        Test max_cardinatlity and bar gap
        """
        # Arrange
        # Act
        result = dummy_vd[self.COL_NAME].bar(
            bargap=bargap,
            max_cardinality=max_cardinality,
        )
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"


def col_name_param():
    """
    Get column value to pass as pytest parameter
    """
    return "0"


class VDFBarPlot(BasicPlotTests):
    """
    Testing different attributes of Bar plot on a vDataFrame
    """

    # Testing variables

    COL_NAME_VDF_1 = "cats"

    @pytest.fixture(autouse=True)
    def data(self, dummy_dist_vd):
        """
        Load test data
        """
        self.data = dummy_dist_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [self.COL_NAME_VDF_1, "density"]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data.bar,
            {"columns": self.COL_NAME_VDF_1},
        )

    @pytest.mark.parametrize(
        "of_col, method", [(col_name_param(), "min"), (col_name_param(), "max")]
    )
    def test_properties_output_type_for_all_options(
        self,
        dummy_dist_vd,
        plotting_library_object,
        of_col,
        method,
    ):
        """
        Test of and method combination
        """
        # Arrange
        # Act
        result = dummy_dist_vd[self.COL_NAME_VDF_1].bar(
            of=of_col,
            method=method,
        )
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"


class VDFBarPlot2D(BasicPlotTests):
    """
    Testing different attributes of Bar plot on a vDataFrame
    """

    # Testing variables

    COL_NAME_VDF_1 = "cats"
    COL_NAME_VDF_2 = "binary"

    @pytest.fixture(autouse=True)
    def data(self, dummy_dist_vd):
        """
        Load test data
        """
        self.data = dummy_dist_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [self.COL_NAME_VDF_1, "density"]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data.bar,
            {"columns": [self.COL_NAME_VDF_1, self.COL_NAME_VDF_2]},
        )


class VDCBarhPlot(BasicPlotTests):
    """
    Testing different attributes of HHorizontal Bar plot on a vDataColumn
    """

    # Testing variables
    COL_NAME = "check 2"
    COL_NAME_2 = "check 1"

    @pytest.fixture(autouse=True)
    def data(self, dummy_vd):
        """
        Load test data
        """
        self.data = dummy_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [self.COL_NAME, "density"]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data[self.COL_NAME].barh,
            {},
        )

    @pytest.mark.parametrize(
        "max_cardinality, method", [(1, "mean"), (1, "max"), (2, "sum")]
    )
    def test_properties_output_type_for_all_options(
        self,
        dummy_vd,
        method,
        plotting_library_object,
        max_cardinality,
    ):
        """
        Test max_cardinality and method combination
        """
        # Arrange
        # Act
        result = dummy_vd[self.COL_NAME].barh(
            method=method,
            of=self.COL_NAME_2,
            max_cardinality=max_cardinality,
        )
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"


class VDFBarhPlot(VDFBarPlot):
    """
    Testing different attributes of HHorizontal Bar plot on a vDataFrame
    """

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data.barh,
            {"columns": self.COL_NAME_VDF_1},
        )


class VDFBarhPlot2D(VDFBarPlot2D):
    """
    Testing different attributes of HHorizontal Bar plot on a vDataFrame
    """

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data.barh,
            {"columns": [self.COL_NAME_VDF_1, self.COL_NAME_VDF_2]},
        )


class VDCBoxPlot(BasicPlotTests):
    """
    Testing different attributes of Box plot on a vDataColumn
    """

    COL_NAME_1 = "0"

    @pytest.fixture(autouse=True)
    def data(self, dummy_dist_vd):
        """
        Load test data
        """
        self.data = dummy_dist_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [self.COL_NAME_1, None]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data[self.COL_NAME_1].boxplot,
            {},
        )

    @pytest.mark.skip(reason="The plot does not have label on x-axis yet")
    def test_properties_xaxis_label(self):
        """
        Testing x-axis title
        """


class VDCParitionBoxPlot(VDCBoxPlot):
    """
    Testing different attributes of Box plot on a vDataColumn using "by" attribute
    """

    COL_NAME_2 = "binary"

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [self.COL_NAME_1, "density"]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data[self.COL_NAME_1].boxplot,
            {"by": self.COL_NAME_2},
        )

    @pytest.mark.skip(reason="The plot does not have label on y-axis yet")
    def test_properties_yaxis_label(self):
        """
        Testing x-axis title
        """


class VDFBoxPlot(VDCBoxPlot):
    """
    Testing different attributes of Box plot on a vDataFrame
    """

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [None, self.COL_NAME_1]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data.boxplot,
            {"columns": self.COL_NAME_1},
        )


class VDCCandlestick(BasicPlotTests):
    """
    Testing different attributes of Candlestick plot on a vDataColumn
    """

    COL_NAME_1 = "values"
    TIME_COL = "date"

    @pytest.fixture(autouse=True)
    def data(self, dummy_line_data_vd):
        """
        Load test data
        """
        self.data = dummy_line_data_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [None, None]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data[self.COL_NAME_1].candlestick,
            {"ts": self.TIME_COL},
        )

    @pytest.mark.skip(reason="The plot does not have label on x-axis yet")
    def test_properties_xaxis_label(self):
        """
        Testing x-axis title
        """

    @pytest.mark.skip(reason="The plot does not have label on y-axis yet")
    def test_properties_yaxis_label(self):
        """
        Testing x-axis title
        """

    @pytest.mark.parametrize(
        "method, start_date", [("count", 1910), ("density", 1920), ("max", 1920)]
    )
    def test_properties_output_type_for_all_options(
        self, plotting_library_object, dummy_line_data_vd, method, start_date
    ):
        """
        Test "method" and "start date" parameters
        """
        # Arrange
        # Act
        result = dummy_line_data_vd[self.COL_NAME_1].candlestick(
            ts=self.TIME_COL, method=method, start_date=start_date
        )
        # Assert - checking if correct object created


class VDFContourPlot(BasicPlotTests):
    """
    Testing different attributes of Contour plot on a vDataFrame
    """

    COL_NAME_1 = "0"
    COL_NAME_2 = "binary"

    @pytest.fixture(autouse=True)
    def data(self, dummy_dist_vd):
        """
        Load test data
        """
        self.data = dummy_dist_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [self.COL_NAME_1, self.COL_NAME_2]

    def create_plot(self):
        """
        Create the plot
        """

        def func_tmp(param_a, param_b):
            """
            Arbitrary custom function for testing
            """
            return param_b + param_a * 0

        return (
            self.data.contour,
            {"columns": [self.COL_NAME_1, self.COL_NAME_2], "func": func_tmp},
        )

    @pytest.mark.parametrize("nbins", [10, 20])
    def test_properties_output_type_for_all_options(
        self, dummy_dist_vd, plotting_library_object, nbins
    ):
        """
        Test different bin sizes
        """

        # Arrange
        def func(param_a, param_b):
            return param_b + param_a * 0

        # Act
        result = dummy_dist_vd.contour(
            [self.COL_NAME_1, self.COL_NAME_2], func, nbins=nbins
        )
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"


class VDCDensityPlot(BasicPlotTests):
    """
    Testing different attributes of Density plot on a vDataColumn
    """

    COL_NAME = "0"

    @pytest.fixture(autouse=True)
    def data(self, dummy_dist_vd):
        """
        Load test data
        """
        self.data = dummy_dist_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [self.COL_NAME, "density"]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data[self.COL_NAME].density,
            {},
        )

    @pytest.mark.parametrize(
        "kernel,nbins", [("logistic", 10), ("sigmoid", 10), ("silverman", 20)]
    )
    def test_properties_output_type_for_all_options(
        self, dummy_dist_vd, plotting_library_object, nbins, kernel
    ):
        """
        Test different bin sizes and kernel types
        """
        # Arrange
        # Act
        result = dummy_dist_vd[self.COL_NAME].density(kernel=kernel, nbins=nbins)
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"


class VDCDensityMultiPlot(BasicPlotTests):
    """
    Testing different attributes of Multiple Density plots on a vDataColumn
    """

    COL_NAME = "0"
    BY_COL = "binary"

    @pytest.fixture(autouse=True)
    def data(self, dummy_dist_vd):
        """
        Load test data
        """
        self.data = dummy_dist_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [self.COL_NAME, "density"]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data[self.COL_NAME].density,
            {"by": self.BY_COL},
        )


class VDFDensityPlot(BasicPlotTests):
    """
    Testing different attributes of Density plot on a vDataFrame
    """

    COL_NAME = "0"

    @pytest.fixture(autouse=True)
    def data(self, dummy_dist_vd):
        """
        Load test data
        """
        self.data = dummy_dist_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [self.COL_NAME, "density"]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data.density,
            {"columns": self.COL_NAME},
        )

    @pytest.mark.parametrize(
        "kernel,nbins", [("logistic", 10), ("sigmoid", 10), ("silverman", 20)]
    )
    def test_properties_output_type_for_all_options(
        self, dummy_dist_vd, plotting_library_object, nbins, kernel
    ):
        """
        Test different bin sizes and kernel types
        """
        # Arrange
        # Act
        result = dummy_dist_vd.density(
            columns=self.COL_NAME, kernel=kernel, nbins=nbins
        )
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"


class VDFPivotHeatMap(BasicPlotTests):
    """
    Testing different attributes of Heatmap plot on a vDataFrame
    """

    PIVOT_COL_1 = "survived"
    PIVOT_COL_2 = "pclass"

    @pytest.fixture(autouse=True)
    def data(self, titanic_vd):
        """
        Load test data
        """
        self.data = titanic_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [self.PIVOT_COL_1, self.PIVOT_COL_2]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data.pivot_table,
            {"columns": [self.PIVOT_COL_1, self.PIVOT_COL_2]},
        )

    def test_properties_output_type_for_corr(
        self, dummy_scatter_vd, plotting_library_object
    ):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        result = dummy_scatter_vd.corr(method="spearman")
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "wrong object crated"

    def test_properties_yaxis_labels_for_categorical_data(self, titanic_vd):
        """
        Test labels for Y-axis
        """
        # Arrange
        expected_labels = (
            '"survived"',
            '"pclass"',
            '"fare"',
            '"parch"',
            '"age"',
            '"sibsp"',
            '"body"',
        )
        # Act
        result = titanic_vd.corr(method="pearson", focus="survived")
        yaxis_labels = result.options["yAxis"].categories
        # Assert
        assert set(yaxis_labels).issubset(expected_labels), "Y-axis labels incorrect"


class VDFHeatMap(BasicPlotTests):
    """
    Testing different attributes of Heatmap plot on a vDataFrame
    """

    COL_NAME_1 = "PetalLengthCm"
    COL_NAME_2 = "SepalLengthCm"

    @pytest.fixture(autouse=True)
    def data(self, iris_vd):
        """
        Load test data
        """
        self.data = iris_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [self.COL_NAME_1, self.COL_NAME_2]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data.heatmap,
            {"columns": [self.COL_NAME_1, self.COL_NAME_2]},
        )

    @pytest.mark.parametrize("method", ["count", "density"])
    def test_properties_output_type_for_all_options(
        self, iris_vd, plotting_library_object, method
    ):
        """
        Test different method types
        """
        # Arrange
        # Act
        result = iris_vd.heatmap(
            [self.COL_NAME_1, self.COL_NAME_2],
            method=method,
        )
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"


class VDCHistogramPlot(BasicPlotTests):
    """
    Testing different attributes of Histogram plot on a vDataColumn
    """

    COL_NAME_1 = "binary"
    COL_OF = "0"

    @pytest.fixture(autouse=True)
    def data(self, dummy_dist_vd):
        """
        Load test data
        """
        self.data = dummy_dist_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [self.COL_NAME_1, "density"]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data[self.COL_NAME_1].hist,
            {},
        )

    @pytest.mark.parametrize("max_cardinality, method", [(3, "count"), (5, "density")])
    def test_properties_output_type_for_all_options(
        self, dummy_dist_vd, plotting_library_object, max_cardinality, method
    ):
        """
        Test different method types and number of max_cardinality
        """
        # Arrange
        # Act
        result = dummy_dist_vd[self.COL_NAME_1].hist(
            of=self.COL_OF, method=method, max_cardinality=max_cardinality
        )
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"


class VDFHistogramPlot(VDCHistogramPlot):
    """
    Testing different attributes of Histogram plot on a vDataFrame
    """

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data.hist,
            {"columns": self.COL_NAME_1},
        )

    @pytest.mark.skip(reason="There is a bug currently with max_cardinality")
    @pytest.mark.parametrize("max_cardinality, method", [(3, "count"), (5, "density")])
    def test_properties_output_type_for_all_options(
        self, dummy_dist_vd, plotting_library_object, max_cardinality, method
    ):
        """
        Test different method types and number of max_cardinality
        """
        # Arrange
        # Act
        result = dummy_dist_vd.hist(
            columns=[self.COL_NAME_1],
            of=self.COL_OF,
            method=method,
            max_cardinality=max_cardinality,
        )
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"


class VDFHistogramMultiPlot(BasicPlotTests):
    """
    Testing different attributes of Histogram plot on a vDataFrame
    """

    COL_NAME_1 = "0"
    COL_NAME_2 = "1"

    @pytest.fixture(autouse=True)
    def data(self, dummy_dist_vd):
        """
        Load test data
        """
        self.data = dummy_dist_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [None, "density"]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data.hist,
            {"columns": [self.COL_NAME_1, self.COL_NAME_2]},
        )


class VDCLinePlot(BasicPlotTests):
    """
    Testing different attributes of Line plot on a vDataColumn
    """

    TIME_COL = "date"
    COL_NAME_1 = "values"
    COL_NAME_2 = "category"
    CAT_OPTION = "A"

    @pytest.fixture(autouse=True)
    def data(self, dummy_line_data_vd):
        """
        Load test data
        """
        self.data = dummy_line_data_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return ["date", self.COL_NAME_1]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data[self.COL_NAME_1].plot,
            {"ts": self.TIME_COL, "by": self.COL_NAME_2},
        )

    @pytest.mark.skip(reason="The plot does not have label on y-axis yet")
    def test_properties_yaxis_label(self):
        """
        Testing y-axis title
        """

    def test_properties_output_type_for_one_trace(
        self, dummy_line_data_vd, plotting_library_object
    ):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        result = dummy_line_data_vd[
            dummy_line_data_vd[self.COL_NAME_2] == self.CAT_OPTION
        ][self.COL_NAME_1].plot(ts=self.TIME_COL)
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"

    @pytest.mark.parametrize("kind", ["spline", "area", "step"])
    @pytest.mark.parametrize("start_date", ["1930"])
    def test_properties_output_type_for_all_options(
        self, dummy_line_data_vd, plotting_library_object, start_date, kind
    ):
        """
        Testing different kinds and start date
        """
        # Arrange
        # Act
        result = dummy_line_data_vd[self.COL_NAME_1].plot(
            ts=self.TIME_COL, kind=kind, start_date=start_date
        )
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"


class VDFLinePlot(BasicPlotTests):
    """
    Testing different attributes of Line plot on a vDataFrame
    """

    TIME_COL = "date"
    COL_NAME_1 = "values"
    COL_NAME_2 = "category"
    CAT_OPTION = "A"

    @pytest.fixture(autouse=True)
    def data(self, dummy_line_data_vd):
        """
        Load test data
        """
        self.data = dummy_line_data_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return ["date", self.COL_NAME_1]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data[self.data[self.COL_NAME_2] == self.CAT_OPTION].plot,
            {"ts": self.TIME_COL, "columns": self.COL_NAME_1},
        )


class OutliersPlot(BasicPlotTests):
    """
    Testing different attributes of outliers plot on a vDataColumn
    """

    COL_NAME_1 = "0"

    @pytest.fixture(autouse=True)
    def data(self, dummy_dist_vd):
        """
        Load test data
        """
        self.data = dummy_dist_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [self.COL_NAME_1, ""]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data.outliers_plot,
            {"columns": self.COL_NAME_1},
        )


class OutliersPlot2D(BasicPlotTests):
    """
    Testing different attributes of outliers plot on a vDataFrame
    """

    COL_NAME_1 = "0"
    COL_NAME_2 = "1"

    @pytest.fixture(autouse=True)
    def data(self, dummy_dist_vd):
        """
        Load test data
        """
        self.data = dummy_dist_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [self.COL_NAME_1, self.COL_NAME_2]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data.outliers_plot,
            {"columns": [self.COL_NAME_1, self.COL_NAME_2]},
        )


class VDCPiePlot:
    """
    Testing different attributes of Pie plot on a vDataColumn
    """

    COL_NAME = "check 1"

    @pytest.fixture(scope="class")
    def plot_result(self, dummy_vd):
        """
        Create a pie plot for vDataColumn
        """
        return dummy_vd[self.COL_NAME].pie()

    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        """
        Get the plot results
        """
        self.result = plot_result

    def test_properties_output_type_for(self, plotting_library_object):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    @pytest.mark.parametrize("kind, max_cardinality", [("donut", 2), ("rose", 4)])
    def test_properties_output_type_for_all_options(
        self,
        dummy_vd,
        plotting_library_object,
        kind,
        max_cardinality,
    ):
        """
        Test different kind and max-cardinality options
        """
        # Arrange
        # Act
        result = dummy_vd[self.COL_NAME].pie(kind=kind, max_cardinality=max_cardinality)
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"


class NestedVDFPiePlot:
    """
    Testing different attributes of Pie plot on a vDataFrame
    """

    COL_NAME = "check 1"
    COL_NAME_2 = "check 2"

    @pytest.fixture(scope="class")
    def plot_result_2(self, dummy_vd):
        """
        Create a pie plot for vDataFrame
        """
        return dummy_vd.pie([self.COL_NAME, self.COL_NAME_2])

    @pytest.fixture(autouse=True)
    def result(self, plot_result_2):
        """
        Get the plot results
        """
        self.result = plot_result_2

    def test_properties_type(self, plotting_library_object):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"


class VDCRangeCurve(BasicPlotTests):
    """
    Testing different attributes of range curve plot on a vDataColumn
    """

    TIME_COL = "date"
    COL_NAME_1 = "value"

    @pytest.fixture(autouse=True)
    def data(self, dummy_date_vd):
        """
        Load test data
        """
        self.data = dummy_date_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [self.TIME_COL, self.COL_NAME_1]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data[self.COL_NAME_1].range_plot,
            {"ts": self.TIME_COL, "plot_median": True},
        )

    @pytest.mark.parametrize(
        "plot_median, date_range",
        [("True", [1920, None]), ("False", [None, 1950])],
    )
    def test_properties_output_type_for_all_options(
        self,
        dummy_date_vd,
        plotting_library_object,
        plot_median,
        date_range,
    ):
        """
        Test different values for median, start date, and end date
        """
        # Arrange
        # Act
        result = dummy_date_vd[self.COL_NAME_1].range_plot(
            ts=self.TIME_COL,
            plot_median=plot_median,
            start_date=date_range[0],
            end_date=date_range[1],
        )
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"


class VDFRangeCurve(VDCRangeCurve):
    """
    Testing different attributes of range curve plot on a vDataFrame
    """

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data.range_plot,
            {"columns": [self.COL_NAME_1], "ts": self.TIME_COL, "plot_median": True},
        )

    @pytest.mark.parametrize(
        "plot_median, date_range",
        [("True", [1920, None]), ("False", [None, 1950])],
    )
    def test_properties_output_type_for_all_options(
        self,
        dummy_date_vd,
        plotting_library_object,
        plot_median,
        date_range,
    ):
        """
        Tes different values for median, start date, and end date
        """
        # Arrange
        # Act
        result = dummy_date_vd.range_plot(
            columns=[self.COL_NAME_1],
            ts=self.TIME_COL,
            plot_median=plot_median,
            start_date=date_range[0],
            end_date=date_range[1],
        )
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"


class ScatterVDF2DPlot(BasicPlotTests):
    """
    Testing different attributes of 2D scatter plot on a vDataFrame
    """

    COL_NAME_1 = "X"
    COL_NAME_2 = "Y"
    COL_NAME_3 = "Z"
    COL_NAME_4 = "Category"
    all_categories = ["A", "B", "C"]

    @pytest.fixture(autouse=True)
    def data(self, dummy_scatter_vd):
        """
        Load test data
        """
        self.data = dummy_scatter_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [self.COL_NAME_1, self.COL_NAME_2]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data.scatter,
            {"columns": [self.COL_NAME_1, self.COL_NAME_2]},
        )

    @pytest.mark.parametrize("attributes", [["Z", 50, 2], [None, 1000, 4]])
    def test_properties_output_type_for_all_options(
        self,
        dummy_scatter_vd,
        plotting_library_object,
        attributes,
    ):
        """
        Test different sizes, number of points and max_cardinality
        """
        # Arrange
        # Act
        size, max_nb_points, max_cardinality = attributes
        result = dummy_scatter_vd.scatter(
            [self.COL_NAME_1, self.COL_NAME_2],
            size=size,
            max_nb_points=max_nb_points,
            max_cardinality=max_cardinality,
        )
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"


class ScatterVDF3DPlot(BasicPlotTests):
    """
    Testing different attributes of 3D scatter plot on a vDataFrame
    """

    COL_NAME_1 = "X"
    COL_NAME_2 = "Y"
    COL_NAME_3 = "Z"
    COL_NAME_4 = "Category"
    all_categories = ["A", "B", "C"]

    @pytest.fixture(autouse=True)
    def data(self, dummy_scatter_vd):
        """
        Load test data
        """
        self.data = dummy_scatter_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [self.COL_NAME_1, self.COL_NAME_2, self.COL_NAME_3]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.data.scatter,
            {
                "columns": [self.COL_NAME_1, self.COL_NAME_2, self.COL_NAME_3],
                "by": self.COL_NAME_4,
            },
        )


class VDCSpiderPlot:
    """
    Testing different attributes of Spider plot on a vDataColumn
    """

    # Testing variables
    COL_NAME_1 = "cats"
    BY_COL = "binary"

    @pytest.fixture(scope="class")
    def plot_result(self, dummy_dist_vd):
        """
        Create a spider plot for vDataColumn
        """
        return dummy_dist_vd[self.COL_NAME_1].spider()

    @pytest.fixture(scope="class")
    def plot_result_2(self, dummy_dist_vd):
        """
        Create a spider plot for vDataColumn using "by" parameter
        """
        return dummy_dist_vd[self.COL_NAME_1].spider(by=self.BY_COL)

    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        """
        Get the plot results
        """
        self.result = plot_result

    @pytest.fixture(autouse=True)
    def by_result(self, plot_result_2):
        """
        Get the plot results
        """
        self.by_result = plot_result_2

    def test_properties_output_type(self, plotting_library_object):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_output_type_for_multiplot(
        self,
        plotting_library_object,
    ):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(
            self.by_result, plotting_library_object
        ), "Wrong object created"


class ChampionChallengerPlot:
    """
    Testing different attributes of Champion CHallenger Plot
    """

    @pytest.fixture(scope="class")
    def plot_result(self, champion_challenger_plot):
        """
        Create a champion challenger plot using AutoML
        """
        return champion_challenger_plot.plot()

    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        """
        Get the plot results
        """
        self.result = plot_result

    def test_properties_output_type(self, plotting_library_object):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"


class ElbowCurvePlot(BasicPlotTests):
    """
    Testing different attributes of Elbow Curve plot
    """

    # Testing variables
    COL_NAME_1 = "PetalLengthCm"
    COL_NAME_2 = "PetalWidthCm"

    @pytest.fixture(autouse=True)
    def data(self, iris_vd):
        """
        Load test data
        """
        self.data = iris_vd

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return ["Number of Clusters", "Elbow Score (Between-Cluster SS / Total SS)"]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            elbow,
            {"input_relation": self.data, "X": [self.COL_NAME_1, self.COL_NAME_2]},
        )


class ImportanceBarChartPlot(BasicPlotTests):
    """
    Testing different attributes of Importance Bar Chart plot
    """

    # Testing variables
    COL_NAME_1 = "PetalLengthCm"
    COL_NAME_2 = "PetalWidthCm"
    COL_NAME_3 = "SepalWidthCm"
    COL_NAME_4 = "SepalLengthCm"
    BY_COL = "Species"

    @pytest.fixture(autouse=True)
    def model(self, iris_vd, schema_loader):
        """
        Load test model
        """
        model = RandomForestClassifier(f"{schema_loader}.importance_test")
        model.fit(
            iris_vd,
            [self.COL_NAME_1, self.COL_NAME_2, self.COL_NAME_3, self.COL_NAME_4],
            self.BY_COL,
        )
        self.model = model
        yield
        model.drop()

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return ["Features", "Importance (%)"]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.model.features_importance,
            {},
        )


class VornoiPlot(BasicPlotTests):
    """
    Testing different attributes of 2D voronoi plot
    """

    COL_NAME_1 = "PetalLengthCm"
    COL_NAME_2 = "PetalWidthCm"

    @pytest.fixture(autouse=True)
    def model(self, iris_vd, schema_loader):
        """
        Load test model
        """
        model = KMeans(name=f"{schema_loader}.test_KMeans_iris")
        model.fit(
            iris_vd,
            [self.COL_NAME_1, self.COL_NAME_2],
        )
        self.model = model
        yield
        model.drop()

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [self.COL_NAME_1, self.COL_NAME_2]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.model.plot_voronoi,
            {},
        )

    @pytest.mark.parametrize("max_nb_points,plot_crosses", [[1000, False]])
    def test_properties_output_type_for_all_options(
        self,
        iris_vd,
        plotting_library_object,
        max_nb_points,
        plot_crosses,
    ):
        """
        Test different number of points and plot_crosses options
        """
        # Arrange
        model = KMeans(name="test_KMeans_iris_2")
        model.fit(
            iris_vd,
            [self.COL_NAME_1, self.COL_NAME_2],
        )
        # Act
        result = model.plot_voronoi(
            max_nb_points,
            plot_crosses,
        )
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"
        # cleanup
        model.drop()


class LOFPlot2D(BasicPlotTests):
    """
    Testing different attributes of 2D LOF plot
    """

    # Testing variables
    COL_NAME_1 = "X"
    COL_NAME_2 = "Y"

    @pytest.fixture(autouse=True)
    def model(self, dummy_scatter_vd, schema_loader):
        """
        Load test model
        """
        model = LocalOutlierFactor(f"{schema_loader}.lof_test")
        model.fit(dummy_scatter_vd, [self.COL_NAME_1, self.COL_NAME_2])
        self.model = model
        yield
        model.drop()

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [self.COL_NAME_1, self.COL_NAME_2]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.model.plot,
            {},
        )


class LOFPlot3D(LOFPlot2D):
    """
    Testing different attributes of 3D LOF plot
    """

    COL_NAME_3 = "Z"

    @pytest.fixture(autouse=True)
    def model(self, dummy_scatter_vd, schema_loader):
        """
        Load test model
        """
        model = LocalOutlierFactor(f"{schema_loader}.lof_test")
        model.fit(dummy_scatter_vd, [self.COL_NAME_1, self.COL_NAME_2, self.COL_NAME_3])
        self.model = model
        yield
        model.drop()

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [self.COL_NAME_1, self.COL_NAME_2, self.COL_NAME_3]


class LogisticRegressionPlot2D(BasicPlotTests):
    """
    Testing different attributes of 2D Logisti Regression plot
    """

    # Testing variables
    COL_NAME_1 = "fare"
    COL_NAME_2 = "survived"

    @pytest.fixture(autouse=True)
    def model(self, titanic_vd, schema_loader):
        """
        Load test model
        """
        model = LogisticRegression(f"{schema_loader}.lof_test")
        model.fit(titanic_vd, [self.COL_NAME_1], self.COL_NAME_2)
        self.model = model
        yield
        model.drop()

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [self.COL_NAME_1, f"P({self.COL_NAME_2} = 1)"]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.model.plot,
            {},
        )

    @pytest.mark.parametrize("max_nb_points", [50])
    def test_properties_output_type_for_all_options(
        self,
        plotting_library_object,
        max_nb_points,
    ):
        """
        Test different number of maximum points
        """
        # Arrange
        # Act
        result = self.model.plot(
            max_nb_points=max_nb_points,
        )
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"


class LogisticRegressionPlot3D(LogisticRegressionPlot2D):
    """
    Testing different attributes of 3D Logisti Regression plot
    """

    COL_NAME_3 = "age"

    @pytest.fixture(autouse=True)
    def model(self, titanic_vd, schema_loader):
        """
        Load test model
        """
        model = LogisticRegression(f"{schema_loader}.lof_test")
        model.fit(titanic_vd, [self.COL_NAME_1, self.COL_NAME_3], self.COL_NAME_2)
        self.model = model
        yield
        model.drop()

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [self.COL_NAME_1, f"P({self.COL_NAME_2} = 1)", self.COL_NAME_3]


class ROCPlot(BasicPlotTests):
    """
    Testing different attributes of ROC plot
    """

    @pytest.fixture(autouse=True)
    def model(self, randon_forest_model_result):
        """
        Load test model
        """
        self.model = randon_forest_model_result

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [
            "False Positive Rate (1-Specificity)",
            "True Positive Rate (Sensitivity)",
        ]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.model.roc_curve,
            {},
        )

    def test_properties_title(self):
        """
        Test plot title
        """
        # Arrange
        test_title = "ROC Curve"
        # Act
        # Assert
        assert get_title(self.result) == test_title, "Plot Title Incorrect"


class CutoffCurvePlot(BasicPlotTests):
    """
    Testing different attributes of Curve plot
    """

    POS_LABEL = "0"

    @pytest.fixture(autouse=True)
    def model(self, randon_forest_model_result):
        """
        Load test model
        """
        self.model = randon_forest_model_result

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return ["Decision Boundary", "Values"]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.model.cutoff_curve,
            {"pos_label": self.POS_LABEL},
        )


class PRCPlot(BasicPlotTests):
    """
    Testing different attributes of PRC plot
    """

    @pytest.fixture(autouse=True)
    def data(self, dummy_probability_data):
        """
        Load test model
        """
        self.data = dummy_probability_data

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return ["Recall", "Precision"]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            prc_curve,
            {"y_true": "y_true", "y_score": "y_score", "input_relation": self.data},
        )


class LiftChartPlot(BasicPlotTests):
    """
    Testing different attributes of Lift Chart plot
    """

    @pytest.fixture(autouse=True)
    def data(self, dummy_probability_data):
        """
        Load test model
        """
        self.data = dummy_probability_data

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return ["Cumulative Data Fraction", "Values"]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            lift_chart,
            {"y_true": "y_true", "y_score": "y_score", "input_relation": self.data},
        )

    def test_properties_title(self):
        """
        Test plot title
        """
        # Arrange
        test_title = "Lift Table"
        # Act
        # Assert
        assert get_title(self.result) == test_title, "Plot Title Incorrect"


class PCACirclePlot(BasicPlotTests):
    """
    Testing different attributes of PCA circle plot
    """

    # Testing variables
    COL_NAME_1 = "X"
    COL_NAME_2 = "Y"
    COL_NAME_3 = "Z"

    @pytest.fixture(autouse=True)
    def model(self, schema_loader, dummy_scatter_vd):
        """
        Load test model
        """
        model = PCA(f"{schema_loader}.pca_circle_test")
        model.drop()
        model.fit(dummy_scatter_vd[self.COL_NAME_1, self.COL_NAME_2, self.COL_NAME_3])
        self.model = model
        yield
        model.drop()

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [
            "Dim1 (80%)",
            "Dim2 (20%)",
        ]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.model.plot_circle,
            {},
        )

    def test_properties_xaxis_label(self):
        """
        Testing x-axis label
        """
        # Arrange
        test_title = "Dim1"
        # Act
        # Assert
        assert test_title in get_xaxis_label(self.result), "X axis label incorrect"

    def test_properties_yaxis_label(self):
        """
        Testing y-axis title
        """
        # Arrange
        test_title = "Dim2"
        # Act
        # Assert
        assert test_title in get_yaxis_label(self.result), "Y axis label incorrect"


class LearningRegressionTreePlot(BasicPlotTests):
    """
    Testing different attributes of Regression Tree plot
    """

    # Testing variables
    COL_NAME_1 = "0"
    COL_NAME_2 = "1"

    @pytest.fixture(autouse=True)
    def model(self, schema_loader, dummy_dist_vd):
        """
        Load test model
        """
        model = DecisionTreeRegressor(name=f"{schema_loader}.model_titanic")
        x_col = self.COL_NAME_1
        y_col = self.COL_NAME_2
        model.fit(dummy_dist_vd, x_col, y_col)
        self.model = model
        yield
        model.drop()

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [
            self.COL_NAME_1,
            self.COL_NAME_2,
        ]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.model.plot,
            {},
        )


class LearningRegressionPlot(BasicPlotTests):
    """
    Testing different attributes of Regression plot
    """

    # Testing variables
    COL_NAME_1 = "X"
    COL_NAME_2 = "Y"

    @pytest.fixture(autouse=True)
    def model(self, schema_loader, dummy_scatter_vd):
        """
        Load test model
        """
        model = LinearRegression(f"{schema_loader}.LR_churn")
        model.fit(dummy_scatter_vd, [self.COL_NAME_1], self.COL_NAME_2)
        self.model = model
        yield
        model.drop()

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [
            self.COL_NAME_1,
            self.COL_NAME_2,
        ]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.model.plot,
            {},
        )


class StepwisePlot:
    """
    Testing different attributes of Stepwise plot
    """

    # Testing variables
    COL_NAME_1 = "age"
    COL_NAME_2 = "fare"
    COL_NAME_3 = "parch"
    COL_NAME_4 = "pclass"
    BY_COL = "survived"

    @pytest.fixture(scope="class")
    def plot_result(self, schema_loader, titanic_vd):
        """
        Create a stepwise regression plot
        """
        model = LogisticRegression(
            name=f"{schema_loader}.test_LR_titanic",
            tol=1e-4,
            max_iter=100,
            solver="newton",
        )
        stepwise_result = stepwise(
            model,
            input_relation=titanic_vd,
            X=[
                self.COL_NAME_1,
                self.COL_NAME_2,
                self.COL_NAME_3,
                self.COL_NAME_4,
            ],
            y=self.BY_COL,
            direction="backward",
        )
        return stepwise_result.step_wise_

    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        """
        Get the plot results
        """
        self.result = plot_result

    def test_properties_output_type(self, plotting_library_object):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_xaxis_label(self):
        """
        Testing x-axis label
        """
        # Arrange
        test_title = "n_features"
        # Act
        # Assert
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_label(self):
        """
        Testing y-axis title
        """
        # Arrange
        test_title = "bic"
        # Act
        # Assert
        assert get_yaxis_label(self.result) == test_title, "Y axis label incorrect"

    def test_additional_options_custom_height(self, titanic_vd):
        """
        Test custom width and height
        """
        # rrange
        custom_height = 60
        custom_width = 70
        model = LogisticRegression(
            name="test_LR_titanic", tol=1e-4, max_iter=100, solver="newton"
        )
        # Act
        stepwise_result = stepwise(
            model,
            input_relation=titanic_vd,
            X=[
                "age",
                "fare",
                "parch",
                "pclass",
            ],
            y="survived",
            direction="backward",
            height=custom_height,
            width=custom_width,
        )
        result = stepwise_result.step_wise_
        # Assert
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"


class SVMClassifier1DPlot(BasicPlotTests):
    """
    Testing different attributes of SVM classifier plot
    """

    # Testing variables
    COL_NAME_1 = "X"

    BY_COL = "Category"

    @pytest.fixture(autouse=True)
    def model(self, schema_loader, dummy_pred_data_vd):
        """
        Load test model
        """
        model = LinearSVC(name=f"{schema_loader}.SVC")
        model.fit(dummy_pred_data_vd, [self.COL_NAME_1], self.BY_COL)
        self.model = model
        yield
        model.drop()

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [
            self.COL_NAME_1,
            self.BY_COL,
        ]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.model.plot,
            {},
        )


class SVMClassifier2DPlot(SVMClassifier1DPlot):
    """
    Testing different attributes of SVM classifier plot
    """

    COL_NAME_2 = "Y"

    @pytest.fixture(autouse=True)
    def model(self, schema_loader, dummy_pred_data_vd):
        """
        Load test model
        """
        model = LinearSVC(name=f"{schema_loader}.SVC")
        model.fit(dummy_pred_data_vd, [self.COL_NAME_1, self.COL_NAME_2], self.BY_COL)
        self.model = model
        yield
        model.drop()

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [
            self.COL_NAME_1,
            self.COL_NAME_2,
        ]


class SVMClassifier3DPlot(SVMClassifier2DPlot):
    """
    Testing different attributes of SVM classifier plot
    """

    COL_NAME_3 = "Z"

    @pytest.fixture(autouse=True)
    def model(self, schema_loader, dummy_pred_data_vd):
        """
        Load test model
        """
        model = LinearSVC(name=f"{schema_loader}.SVC")
        model.fit(
            dummy_pred_data_vd,
            [self.COL_NAME_1, self.COL_NAME_2, self.COL_NAME_3],
            self.BY_COL,
        )
        self.model = model
        yield
        model.drop()

    @property
    def cols(self):
        """
        Store labels for X,Y,Z axis to check.
        """
        return [self.COL_NAME_1, self.COL_NAME_2, self.COL_NAME_3]

    def create_plot(self):
        """
        Create the plot
        """
        return (
            self.model.plot,
            {},
        )
