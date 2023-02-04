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
from verticapy.plotting._matplotlib.animated import (
    animated_bar,
    animated_bubble_plot,
    animated_ts_plot,
)
from verticapy.plotting._matplotlib.bar import bar, bar2D, hist, hist2D, multiple_hist
from verticapy.plotting._matplotlib.core import compute_plot_variables
from verticapy.plotting._matplotlib.boxplot import boxplot, boxplot2D
from verticapy.plotting._matplotlib.heatmap import (
    cmatrix,
    contour_plot,
    hexbin,
    pivot_table,
)
from verticapy.plotting._matplotlib.pie import nested_pie, pie
from verticapy.plotting._matplotlib.scatter import (
    bubble,
    outliers_contour_plot,
    scatter_matrix,
    scatter,
)
from verticapy.plotting._matplotlib.spider import spider
from verticapy.plotting._matplotlib.timeseries import (
    acf_plot,
    multi_ts_plot,
    range_curve,
    range_curve_vdf,
    ts_plot,
)
from verticapy.plotting._matplotlib.mlplot import (
    logit_plot,
    lof_plot,
    plot_importance,
    plot_stepwise_ml,
    plot_bubble_ml,
    plot_pca_circle,
    plot_var,
    regression_plot,
    regression_tree_plot,
    svm_classifier_plot,
    voronoi_plot,
)
