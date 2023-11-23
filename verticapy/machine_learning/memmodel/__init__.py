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
from verticapy.machine_learning.memmodel.base import InMemoryModel
from verticapy.machine_learning.memmodel.cluster import (
    BisectingKMeans,
    KMeans,
    KPrototypes,
    NearestCentroid,
)
from verticapy.machine_learning.memmodel.decomposition import PCA, SVD
from verticapy.machine_learning.memmodel.ensemble import (
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
    XGBClassifier,
    XGBRegressor,
)
from verticapy.machine_learning.memmodel.linear_model import (
    LinearModel,
    LinearModelClassifier,
)
from verticapy.machine_learning.memmodel.naive_bayes import NaiveBayes
from verticapy.machine_learning.memmodel.preprocessing import (
    Scaler,
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
)
from verticapy.machine_learning.memmodel.tree import (
    BinaryTreeAnomaly,
    BinaryTreeClassifier,
    BinaryTreeRegressor,
    NonBinaryTree,
)
