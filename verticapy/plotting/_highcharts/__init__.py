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
from verticapy.plotting._highcharts.acf import ACFPACFPlot, ACFPlot
from verticapy.plotting._highcharts.bar import BarChart, BarChart2D, DrillDownBarChart
from verticapy.plotting._highcharts.hist import Histogram
from verticapy.plotting._highcharts.barh import (
    HorizontalBarChart,
    HorizontalBarChart2D,
    DrillDownHorizontalBarChart,
)
from verticapy.plotting._highcharts.boxplot import BoxPlot
from verticapy.plotting._highcharts.candlestick import CandleStick
from verticapy.plotting._highcharts.contour import ContourPlot
from verticapy.plotting._highcharts.density import DensityPlot, MultiDensityPlot
from verticapy.plotting._highcharts.heatmap import HeatMap
from verticapy.plotting._highcharts.machine_learning.champion_challenger import (
    ChampionChallengerPlot,
)
from verticapy.plotting._highcharts.machine_learning.elbow import ElbowCurve
from verticapy.plotting._highcharts.machine_learning.importance import (
    ImportanceBarChart,
)
from verticapy.plotting._highcharts.machine_learning.lof import LOFPlot
from verticapy.plotting._highcharts.machine_learning.logistic_reg import (
    LogisticRegressionPlot,
)
from verticapy.plotting._highcharts.machine_learning.model_evaluation import (
    CutoffCurve,
    LiftChart,
    PRCCurve,
    ROCCurve,
)
from verticapy.plotting._highcharts.machine_learning.pca import (
    PCACirclePlot,
    PCAScreePlot,
    PCAVarPlot,
)
from verticapy.plotting._highcharts.machine_learning.regression import RegressionPlot
from verticapy.plotting._highcharts.machine_learning.regression_tree import (
    RegressionTreePlot,
)
from verticapy.plotting._highcharts.machine_learning.stepwise import StepwisePlot
from verticapy.plotting._highcharts.machine_learning.svm import SVMClassifierPlot
from verticapy.plotting._highcharts.machine_learning.tsa import TSPlot
from verticapy.plotting._highcharts.line import LinePlot, MultiLinePlot
from verticapy.plotting._highcharts.outliers import OutliersPlot
from verticapy.plotting._highcharts.pie import NestedPieChart, PieChart
from verticapy.plotting._highcharts.range import RangeCurve
from verticapy.plotting._highcharts.scatter import ScatterPlot
from verticapy.plotting._highcharts.spider import SpiderChart
