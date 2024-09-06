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
# Vertica
from verticapy.tests_new.plotting.base_test_files import VDCSpiderPlot


class TestPlotlyVDCSpiderPlot(VDCSpiderPlot):
    """
    Testing different attributes of Spider plot on a vDataColumn
    """

    def test_properties_method_title_at_bottom(
        self,
    ):
        """
        Test method title
        """
        # Arrange
        method_text = "(Method: Density)"
        # Act
        # Assert -
        assert (
            self.result.layout["annotations"][0]["text"] == method_text
        ), "Method title incorrect"

    def test_properties_multiple_plots_produced_for_multiplot(
        self,
    ):
        """
        Test if multiple plots produced
        """
        # Arrange
        number_of_plots = 2
        # Act
        # Assert
        assert (
            len(self.by_result.data) == number_of_plots
        ), "Two traces not produced for two classes of binary"

    def test_data_all_categories(self, dummy_dist_vd):
        """
        Test all categories
        """
        # Arrange
        no_of_category = dummy_dist_vd["cats"].nunique()
        # Act
        assert (
            self.result.data[0]["r"].shape[0] == no_of_category
        ), "The number of categories in the data differ from the plot"
