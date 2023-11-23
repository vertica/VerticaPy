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
import warnings

warning_message = (
    "Importing from 'verticapy.learn.model_selection' is deprecated, "
    "and it will no longer be possible in the next minor release. "
    "Please use 'verticapy.machine_learning.model_selection' instead "
    "to ensure compatibility with upcoming versions."
)
warnings.warn(warning_message, Warning)

from verticapy.machine_learning.metrics.plotting import (
    lift_chart,
    prc_curve,
    roc_curve,
)
from verticapy.machine_learning.model_selection.hp_tuning import (
    bayesian_search_cv,
    enet_search_cv,
    gen_params_grid,
    grid_search_cv,
    parameter_grid,
    plot_acf_pacf,
    randomized_search_cv,
    validation_curve,
)
from verticapy.machine_learning.model_selection.model_validation import (
    cross_validate,
    learning_curve,
)
from verticapy.machine_learning.model_selection.variables_selection import (
    randomized_features_search_cv,
    stepwise,
)
from verticapy.machine_learning.model_selection.kmeans import best_k, elbow
