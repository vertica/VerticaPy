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
from verticapy.tests_new.plotting.base_test_files import VDCBarPlot, VDFBarPlot


class TestHighchartsVDCBarPlot(VDCBarPlot):
    """
    Testing different attributes of Bar plot on a vDataColumn
    """

    def test_data_ratios(self, dummy_vd):
        """
        Test data ratio plotted for bar chart
        """
        ### Checking if the density was plotted correctly
        nums = dummy_vd.to_pandas()[self.COL_NAME].value_counts()
        total = len(dummy_vd)
        assert set(self.result.data_temp[0].data).issubset(
            set([nums["A"] / total, nums["B"] / total, nums["C"] / total])
        )

    def test_all_categories_created(self):
        """
        Test all categories
        """
        assert set(self.result.options["xAxis"].categories).issubset(
            set(["A", "B", "C"])
        )

    def test_additional_options_bargap(self, dummy_vd):
        """
        Test bargap option
        """
        # Arrange
        # Act
        result = dummy_vd[self.COL_NAME].bar(
            bargap=0.5,
        )
        # Assert - checking if correct object created
        assert result.data_temp[0].pointPadding == 0.25, "Custom bargap not working"


class TestHighchartsVDFBarPlot(VDFBarPlot):
    """
    Testing different attributes of Bar plot on a vDataFrame
    """

    def test_data_ratios(self, dummy_dist_vd):
        """
        Test data ratio
        """
        ### Checking if the density was plotted correctly
        nums = dummy_dist_vd.to_pandas()[self.COL_NAME_VDF_1].value_counts()
        total = len(dummy_dist_vd)
        assert set(self.result.data_temp[0].data).issubset(
            set([nums["A"] / total, nums["B"] / total, nums["C"] / total])
        )

    def test_all_categories_created(self):
        """
        Test all categories
        """
        assert set(self.result.options["xAxis"].categories).issubset(
            set(["A", "B", "C"])
        )
