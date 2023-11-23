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
from verticapy.plotting._plotly.bar import BarChart, BarChart2D
from verticapy.plotting._plotly.pie import PieChart, NestedPieChart
from verticapy.plotting._plotly.barh import HorizontalBarChart, HorizontalBarChart2D
from verticapy.plotting._plotly.scatter import ScatterPlot
from verticapy.plotting._plotly.boxplot import BoxPlot
from verticapy.plotting._plotly.heatmap import HeatMap
from verticapy.plotting._plotly.contour import ContourPlot
from verticapy.plotting._plotly.density import DensityPlot, MultiDensityPlot
from verticapy.plotting._plotly.spider import SpiderChart
from verticapy.plotting._plotly.range import RangeCurve
from verticapy.plotting._plotly.line import LinePlot, MultiLinePlot
from verticapy.plotting._plotly.outliers import OutliersPlot
from verticapy.plotting._plotly.acf import ACFPlot
from verticapy.plotting._plotly.base import PlotlyBase
from verticapy.plotting._plotly.hist import Histogram
from verticapy.plotting._plotly.candlestick import CandleStick

from verticapy.plotting._plotly.machine_learning.regression import RegressionPlot
from verticapy.plotting._plotly.machine_learning.elbow import ElbowCurve
from verticapy.plotting._plotly.machine_learning.lof import LOFPlot
from verticapy.plotting._plotly.machine_learning.logistic_reg import (
    LogisticRegressionPlot,
)
from verticapy.plotting._plotly.machine_learning.importance import ImportanceBarChart
from verticapy.plotting._plotly.machine_learning.pca import PCACirclePlot
from verticapy.plotting._plotly.machine_learning.model_evaluation import (
    ROCCurve,
    CutoffCurve,
    PRCCurve,
    LiftChart,
)
from verticapy.plotting._plotly.machine_learning.kmeans import VoronoiPlot
from verticapy.plotting._plotly.machine_learning.svm import SVMClassifierPlot
from verticapy.plotting._plotly.machine_learning.regression_tree import (
    RegressionTreePlot,
)
from verticapy.plotting._plotly.machine_learning.champion_challenger import (
    ChampionChallengerPlot,
)
from verticapy.plotting._plotly.machine_learning.stepwise import StepwisePlot
from verticapy.plotting._plotly.machine_learning.tsa import TSPlot


import plotly.io as pio
import plotly.graph_objects as go

pio.templates["VerticaPy"] = go.layout.Template(
    layout_colorway=PlotlyBase().get_colors()
)
pio.templates.default = "VerticaPy"
