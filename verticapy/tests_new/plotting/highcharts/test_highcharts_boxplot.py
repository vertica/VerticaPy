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
from verticapy.tests_new.plotting.base_test_files import (
    VDCBoxPlot,
    VDCParitionBoxPlot,
    VDFBoxPlot,
)


class TestHighchartsVDCBoxPlot(VDCBoxPlot):
    """
    Testing different attributes of Box plot on a vDataColumn
    """


class TestHighchartsParitionVDCBoxPlot(VDCParitionBoxPlot):
    """
    Testing different attributes of Box plot on a vDataColumn using "by" attribute
    """


class TestHighchartsVDFBoxPlot(VDFBoxPlot):
    """
    Testing different attributes of Box plot on a vDataFrame
    """

    def test_properties_yaxis_label(self):
        """
        Testing y-axis title
        """
        # Arrange
        test_title = self.COL_NAME_1
        # Act
        # Assert - checking y axis label
        assert (
            self.result.options["xAxis"].categories[0] == test_title
        ), "X axis label incorrect"
