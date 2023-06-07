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

# Verticapy
from verticapy.tests_new.plotting.base_test_files import (
    ROCPlot,
    CutoffCurvePlot,
    PRCPlot,
    LiftChartPlot,
)


class TestMatplotlibMachineLearningROCPlot(ROCPlot):
    """
    Testing different attributes of ROC plot
    """


class TestMatplotlibMachineLearningCutoffCurvePlot(CutoffCurvePlot):
    """
    Testing different attributes of Curve plot
    """

    @pytest.mark.skip(reason="Need to fix y axis")
    def test_properties_yaxis_label(self):
        """
        Testing y-axis title
        """


class TestMatplotlibMachineLearningPRCPlot(PRCPlot):
    """
    Testing different attributes of PRC plot
    """


class TestMatplotlibMachineLearningLiftChartPlot(LiftChartPlot):
    """
    Testing different attributes of Lift Chart plot
    """

    @pytest.mark.skip(reason="Need to fix y axis")
    def test_properties_yaxis_label(self):
        """
        Testing y-axis title
        """
