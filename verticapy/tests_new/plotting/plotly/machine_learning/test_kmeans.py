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


# Other Modules


# Verticapy
from verticapy.learn.cluster import KMeans

# Testing variables
COL_NAME_1 = "PetalLengthCm"
COL_NAME_2 = "PetalWidthCm"


@pytest.fixture(name="plot_result", scope="class")
def load_plot_result(iris_vd):
    """
    Create a voronoi plot
    """
    model = KMeans(name="test_KMeans_iris")
    model.fit(
        iris_vd,
        [COL_NAME_1, COL_NAME_2],
    )
    return model.plot_voronoi()


class TestPlotlyMachineLearningLiftChart:
    """
    Testing different attributes of 2D voronoi plot
    """

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
        test_title = COL_NAME_1
        # Act
        # Assert
        assert (
            self.result.layout["xaxis"]["title"]["text"] == test_title
        ), "X axis label incorrect"

    def test_properties_yaxis_label(self):
        """
        Testing y-axis title
        """
        # Arrange
        test_title = COL_NAME_2
        # Act
        # Assert
        assert (
            self.result.layout["yaxis"]["title"]["text"] == test_title
        ), "Y axis label incorrect"

    def test_properties_no_of_elements(self):
        """
        Test if all objects plotted
        """
        # Arrange
        total_items = 20
        # Act
        # Assert
        assert len(self.result.data) == pytest.approx(
            total_items, abs=2
        ), "Some elements missing"

    def test_additional_options_custom_height(self, iris_vd):
        """
        Test custom width and height
        """
        # rrange
        custom_height = 650
        custom_width = 700
        model = KMeans(name="public.KMeans_iris")
        model.fit(
            iris_vd,
            [COL_NAME_1, COL_NAME_2],
        )
        # Act
        result = model.plot_voronoi(width=custom_width, height=custom_height)
        # Assert
        assert (
            result.layout["height"] == custom_height
            and result.layout["width"] == custom_width
        ), "Custom height and width not working"

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
            [COL_NAME_1, COL_NAME_2],
        )
        # Act
        result = model.plot_voronoi(
            max_nb_points,
            plot_crosses,
        )
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"
