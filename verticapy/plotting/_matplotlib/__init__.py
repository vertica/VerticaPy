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
from verticapy.plotting._matplotlib.animated.bar import AnimatedBarChart
from verticapy.plotting._matplotlib.animated.bubble import AnimatedBubblePlot
from verticapy.plotting._matplotlib.animated.line import AnimatedLinePlot
from verticapy.plotting._matplotlib.animated.pie import AnimatedPieChart

from verticapy.plotting._matplotlib.machine_learning.champion_challenger import (
    ChampionChallengerPlot,
)
from verticapy.plotting._matplotlib.machine_learning.elbow import ElbowCurve
from verticapy.plotting._matplotlib.machine_learning.importance import (
    ImportanceBarChart,
)
from verticapy.plotting._matplotlib.machine_learning.kmeans import VoronoiPlot
from verticapy.plotting._matplotlib.machine_learning.lof import LOFPlot
from verticapy.plotting._matplotlib.machine_learning.logistic_reg import (
    LogisticRegressionPlot,
)
from verticapy.plotting._matplotlib.machine_learning.model_evaluation import (
    CutoffCurve,
    LiftChart,
    PRCCurve,
    ROCCurve,
)
from verticapy.plotting._matplotlib.machine_learning.pca import (
    PCACirclePlot,
    PCAScreePlot,
    PCAVarPlot,
)
from verticapy.plotting._matplotlib.machine_learning.regression import RegressionPlot
from verticapy.plotting._matplotlib.machine_learning.regression_tree import (
    RegressionTreePlot,
)
from verticapy.plotting._matplotlib.machine_learning.stepwise import StepwisePlot
from verticapy.plotting._matplotlib.machine_learning.svm import SVMClassifierPlot
from verticapy.plotting._matplotlib.machine_learning.tsa import TSPlot

from verticapy.plotting._matplotlib.acf import ACFPlot, ACFPACFPlot
from verticapy.plotting._matplotlib.bar import BarChart, BarChart2D
from verticapy.plotting._matplotlib.barh import HorizontalBarChart, HorizontalBarChart2D
from verticapy.plotting._matplotlib.boxplot import BoxPlot
from verticapy.plotting._matplotlib.contour import ContourPlot
from verticapy.plotting._matplotlib.density import (
    DensityPlot,
    DensityPlot2D,
    MultiDensityPlot,
)
from verticapy.plotting._matplotlib.heatmap import HeatMap
from verticapy.plotting._matplotlib.hexbin import HexbinMap
from verticapy.plotting._matplotlib.hist import Histogram
from verticapy.plotting._matplotlib.line import LinePlot, MultiLinePlot
from verticapy.plotting._matplotlib.outliers import OutliersPlot
from verticapy.plotting._matplotlib.pie import PieChart, NestedPieChart
from verticapy.plotting._matplotlib.range import RangeCurve
from verticapy.plotting._matplotlib.scatter import ScatterMatrix, ScatterPlot
from verticapy.plotting._matplotlib.spider import SpiderChart
from verticapy.plotting._matplotlib.candlestick import CandleStick
